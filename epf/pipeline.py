import re
import time
from pathlib import Path
from time import strftime

import keras_tuner as kt

import holidays
import keras
import pickle as pkl
import pandas as pd
from pandas import DataFrame
from timeit import default_timer as timer

from epf.config import FeatureConfig, ModelConfig, MODELS_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, \
    TRAIN_DATA_DIR, LOG
from epf.util import load_and_concat_data, remove_seasonal_component, detect_and_remove_outliers, split_data, \
    WindowGenerator, builder


class EpfPipeline(object):
    """
    A class to load and process raw data, build, tune and train the model and predict from new inputs.
    Configurable via ``FeatureConfig`` and ``ModelConfig``.
    """

    def __init__(
            self,
            feature_config: FeatureConfig = FeatureConfig(),
            model_config: ModelConfig = ModelConfig(),
            raw_data_dir: Path = RAW_DATA_DIR,
            interim_data_dir: Path = INTERIM_DATA_DIR,
            processed_data_dir: Path = PROCESSED_DATA_DIR,
            train_data_dir: Path = TRAIN_DATA_DIR,
            default_model_path: Path = MODELS_DIR,
    ):
        """
        Initialize the pipeline with feature and model configurations. Default values should be sufficient. Change
        directories, Config classes or ``ModelBuilder`` if desired.

        :param feature_config: Feature configuration.
        :type feature_config: FeatureConfig

        :param model_config: Model configuration.
        :type model_config: ModelConfig

        :param raw_data_dir: Raw data directory.
        :type raw_data_dir: Path

        :param interim_data_dir: Interim data directory.
        :type interim_data_dir: Path

        :param processed_data_dir: Processed data directory.
        :type processed_data_dir: Path

        :param train_data_dir: Training data directory.
        :type train_data_dir: Path

        :param default_model_path: Default path to save the model.
        :type default_model_path: Path
        """
        self._fc = feature_config
        self._mc = model_config
        self._raw_data_dir = raw_data_dir
        self._interim_data_dir = interim_data_dir
        self._processed_data_dir = processed_data_dir
        self._train_data_dir = train_data_dir
        self._default_model_path = default_model_path

        self.timings = {}

        self.best_hps = None
        self.best_model = None
        self.feature_set = None
        self.history = None
        self.predictions = None
        self.raw_data = None
        self.seasonal = None
        self.test_df = None
        self.train_df = None
        self.train_mean = None
        self.train_std = None
        self.validation_df = None
        self.window = None

        print(f"Pipeline initialized with:\n"
              f"FeatureConfig: \n{self._fc}\n"
              f"ModelConfig: \n{self._mc}\n")

    def _load_data(self):
        """ Loads raw data and saves it to interim directory using the feature configuration.
        """
        start = timer()

        if self.raw_data is not None:
            LOG.info("Raw data has already been loaded. Skipping loading.")
            return

        file_paths = self._fc.INPUT_PATHS
        to_resample = self._fc.TO_RESAMPLE
        df: list[DataFrame] = []

        # read all years for each raw feature and concat them in load_and_concat_data, then append them to the big df
        # each file_path list contains multiple paths to the raw yearly data
        for file_path in file_paths.values():
            col_name = [k for k, v in file_paths.items() if v == file_path]
            # we can assert that the result of the list comprehension for col_name only ever has one value
            # because the mapping in the config is built that way. Only that way col_name[0] can be used even though
            # it is not the best code from a style and robustness perspective.
            data = load_and_concat_data(file_path, col_name[0])

            # if the columns frequency is not hourly, resample it to hourly frequency
            if data.columns.values[0] in to_resample:
                resample_freq: int = to_resample.get(data.columns.values[0])
                data = data[::resample_freq]

            df.append(data)

        LOG.success(f"Raw data has been successfully loaded.")

        data_out = pd.concat(df, axis=1)
        raw_data_path = self._interim_data_dir / "raw_data"

        LOG.info(f"Saving raw data to {raw_data_path.as_posix()}")
        data_out.to_csv(raw_data_path, index=True)

        self.raw_data = data_out
        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'data_loading' : t_elapsed})
        LOG.info(f"Data loading took {t_elapsed}.")

    def _generate_features(self):
        """
        Generates features using the feature configuration and saves the feature DataFrame to ``processed_data_dir``.

        Note that the current implementation of ``generate_features`` is slow because the feature engineering pipeline
        is ran for each feature regardless if its used later on during training.
        """
        start = timer()
        if self.feature_set is not None:
            LOG.info("Feature Set has already been generated. Skipping feature generation.")
            return

        feature_dict = self._fc.FEATURE_DICT
        periods = self._fc.PERIODS
        window_length = self._fc.WINDOW_LENGTH
        n_sigma = self._fc.N_SIGMA
        method = self._fc.METHOD
        generate_lags = self._fc.GENERATE_LAGS
        generate_dummies = self._fc.GENERATE_DUMMIES

        data = self.raw_data

        LOG.info(f"Running feature generation.")

        # detect and remove outliers in all price timeseries
        LOG.info(f"Removing Outliers. This might take a while...")
        data = detect_and_remove_outliers(data, window_length, n_sigma, method)
        LOG.success(f"Finished outlier removal.")

        # remove seasonal component from all timeseries and store seasonal component to later add back to predictions
        LOG.info(f"Removing seasonal component.")
        data, seasonal = remove_seasonal_component(data, periods)
        LOG.success(f"Finished seasonal decomposition.")

        # seasonal components are stored for the time being to access them later when predictions are made.
        self.seasonal = seasonal

        # create calendar features
        if generate_dummies:
            LOG.info(f"Generating dummies.")

            data['month'] = pd.DatetimeIndex(data.index).month
            # 1 = Monday, 7 = Sunday, add 1 because default is 0 = Monday
            data['day_of_week'] = pd.DatetimeIndex(data.index).dayofweek + 1
            # uses holiday library see references/refs.md [4]
            de_holidays = holidays.country_holidays('DE', years=[2023, 2024])
            # set holiday to 1 if it is a holiday else 0
            data['holiday'] = data.index.to_series().apply(lambda x: 1 if x in de_holidays else 0)

        # create lagged prices according to fft analysis based off of cleaned price data de_lu_price_hat
        # 7 day, 1 day, 12 hour and 1 hour lags are used
        if generate_lags:
            LOG.info(f"Generating lagged prices.")

            data['de_lu_price_7_day_lag'] = data['de_prices_hat_rm_seasonal'].shift(168, fill_value=0)
            data['de_lu_price_1_day_lag'] = data['de_prices_hat_rm_seasonal'].shift(24, fill_value=0)
            data['de_lu_price_12_hour_lag'] = data['de_prices_hat_rm_seasonal'].shift(12, fill_value=0)
            data['de_lu_price_1_hour_lag'] = data['de_prices_hat_rm_seasonal'].shift(1, fill_value=0)

        # only select the columns that are specified in FEATURE_DICT
        feature_path = self._processed_data_dir / "features.csv"
        feature_set: pd.DataFrame = data.loc[:, [k for k, v in feature_dict.items() if v['select'] == 1]]

        LOG.info(f"Saving feature_set to {feature_path.as_posix()}.")
        feature_set.to_csv(feature_path, index=True)

        LOG.success(f"Successfully saved generated features to {feature_path.as_posix()}")
        LOG.info(f"Finished generating features.")

        self.feature_set = feature_set
        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'feature_generation': t_elapsed})
        LOG.info(f"Feature generation took {t_elapsed}.")

    def _generate_training_data(self):
        """
        Generates Training, Validation and Test data and saves them to ``processed_data_dir``.
        """
        start = timer()
        if self.train_df is not None:
            LOG.info("Training data has already been generated. Skipping training data generation.")
            return

        feature_set = self.feature_set
        train_split = self._mc.TRAIN_SPLIT
        validation_split = self._mc.VALIDATION_SPLIT
        data_path = self._train_data_dir

        # generate training splits
        train_df, validation_df, test_df = split_data(feature_set, train_split, validation_split)

        # normalize data
        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train_mean = train_mean
        self.train_std = train_std

        self.train_df = (train_df - train_mean) / train_std
        self.validation_df = (validation_df - train_mean) / train_std
        self.test_df = (test_df - train_mean) / train_std

        # save train, val and test sets to csv
        # use the highest protocol available, denoted by -1
        with open(data_path / "train_df.pkl", 'wb') as f:
            pkl.dump(self.train_df, f, -1)

        with open(data_path / "validation_df.pkl", 'wb') as f:
            pkl.dump(self.validation_df, f, -1)

        with open(data_path / "test_df.pkl", 'wb') as f:
            pkl.dump(self.test_df, f, -1)

        LOG.success(f"Successfully saved data splits to \n"
                    f"{data_path.as_posix().join('train_df.pkl')}, \n"
                    f"{data_path.as_posix().join('validation_df.pkl')} and \n"
                    f"{data_path.as_posix().join('test_df.pkl')}")
        LOG.info(f"Finished generating training data.")
        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'training_data_generation': t_elapsed})
        LOG.info(f"Training data generation took {t_elapsed}.")

    def _prep_data(self):
        """ Bundles all data preparation steps together for a single call in ``train``"""
        self._load_data()
        self._generate_features()
        self._generate_training_data()

    def _tune_hyperparameters(self,
                              window: WindowGenerator,
                              max_epochs: int,
                              model_name: str,
                              tuner_dir: Path,
                              tuned_model_path: Path,
                              tuned_hyperparams_path: Path, ):
        """
        Tunes the hyperparameters for the provided model builder. Saves the best hyperparameters and the best model to disk.
        Also saves both properties to the class instance for streamlined use in training.

        :param window: WindowGenerator object containing the training, validation and test data.
        :type window: WindowGenerator

        :param max_epochs: Maximum number of epochs for training.
        :type max_epochs: int

        :param model_name: Name of the model to be tuned.
        :type model_name: str

        :param tuner_dir: Directory to save the tuned hyperparameters.
        :type tuner_dir: Path

        :param tuned_model_path: Path to save the tuned model.
        :type tuned_model_path: Path

        :param tuned_hyperparams_path: Path to save the tuned hyperparameters.
        :type tuned_hyperparams_path: Path
        """
        start = timer()
        max_trails = self._mc.MAX_TRIALS

        tuner = (kt.BayesianOptimization
            (
            builder,
            objective='mean_absolute_error',
            max_trials=max_trails,
            directory=tuner_dir,
            overwrite=True,
            project_name=model_name,
        ))

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        tuner.search(
            window.train,
            epochs=max_epochs,
            validation_data=window.val,
            callbacks=[stop_early]
        )

        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.best_model = tuner.get_best_models(num_models=1)[0]

        with open(tuned_model_path, 'wb') as f:
            pkl.dump(self.best_model, f)

        with open(tuned_hyperparams_path, 'wb') as f:
            pkl.dump(self.best_hps, f)

        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'hp_tuning': t_elapsed})
        LOG.info(f"Hyperparameter tuning took {t_elapsed}.")

    def train(self, model_name: str, overwrite: bool, prep_data: bool = True, use_tuned_hyperparams: bool = False):
        """
        Trains a model on the training data that is currently saved on the disk. The model configuration provides
        necessary tuning and training parameters. When ``prep_data`` = ``True``, the data preparation pipeline is called
        (see ``_prep_data()``). Saves the trained model to ``MODELS_DIR/model_name``.

        :param model_name: A name for the model that identifies it when saved.
        :type model_name: str

        :param overwrite: Whether to overwrite an existing model or not.
        :type overwrite: bool

        :param prep_data: Whether to prepare the data or not. If set to False, preprocessed data is loaded from disk.
            If set to True already existing Data is overwritten.
        :type prep_data: bool

        :param use_tuned_hyperparams: Whether to use already tuned hyperparameters from disk or tune them again for this training loop.
        :type use_tuned_hyperparams: bool
        """
        start = timer()
        tuner_dir = MODELS_DIR / "tuner"
        # failsafe if tuner dir does not exist1
        if not tuner_dir.exists():
            tuner_dir.mkdir(parents=True, exist_ok=True)

        model_out_path = (self._default_model_path / model_name).with_suffix('.pkl')
        tuned_model_path = (tuner_dir / model_name).with_suffix('.pkl')
        tuned_hyperparams_path = (tuner_dir / f"{model_name}_hyperparameters").with_suffix('.pkl')
        out_steps = self._mc.OUT_STEPS
        max_epochs = self._mc.MAX_EPOCHS
        label_col = self._mc.LABEL_COL

        LOG.info(f"Data paths for training loop successfully initialized.\n"
                 f"Model output: {model_out_path.as_posix()}.\n"
                 f"Tuned model path: {tuned_model_path.as_posix()}.\n"
                 f"Tuned Hyperparameters path: {tuned_hyperparams_path.as_posix()}.\n")

        # load the model obj that stores all relevant data for training, if no model obj exists then create a new one
        if model_out_path.exists():
            with open(model_out_path, 'rb') as f:
                model_obj = pkl.load(f)
        else:
            model_obj = {
                'model_name': model_name,
                'best_model': None,
                'best_hps': None,
                'history': None,
                'train_mean': None,
                'train_std': None,
                'seasonal': None,
                'window': None,
                'train_df': None,
                'validation_df': None,
                'test_df': None
            }

        # check if training data is already prepared, if not fallback to preparing the data.
        # Otherwise, data from disk is loaded
        if (prep_data and ((model_obj['train_df'] is None)
                           or (model_obj['validation_df'] is None)
                           or (model_obj['test_df'] is None))):
            LOG.info("Preparing training data.")
            self._prep_data()
            LOG.success(f"Successfully prepared training data for {model_name}")

        elif (not prep_data and ((model_obj['train_df'] is not None)
                                or (model_obj['validation_df'] is not None)
                                or (model_obj['test_df'] is not None))):
            LOG.info("Now loading training data from disk.")
            self.train_df = model_obj['train_df']
            self.validation_df = model_obj['validation_df']
            self.test_df = model_obj['test_df']
            LOG.success(f"Successfully loaded training data for {model_name}")

        # generate windows
        window = WindowGenerator(train_df=self.train_df,
                                 validation_df=self.validation_df,
                                 test_df=self.test_df,
                                 input_width=24,
                                 label_width=out_steps,
                                 shift=out_steps,
                                 label_columns=[label_col], )

        self.window = window

        # tune hyperparams, skip hyperparameter tuning if use_tuned_hyperparams is set to True
        if use_tuned_hyperparams and (model_obj['best_hps'] is not None and model_obj['best_model'] is not None):
            LOG.info(f"Skipping hyperparameter tuning and loading tuned hyperparameters for {model_name} "
                     f"from {model_out_path.as_posix()}.")
            self.best_model = model_obj['best_model']
            self.best_hps = model_obj['best_hps']
        elif (not use_tuned_hyperparams) or (
                use_tuned_hyperparams and not (tuned_model_path.exists() and tuned_hyperparams_path.exists())):
            if not (tuned_model_path.exists() and tuned_hyperparams_path.exists()):
                LOG.warning("No tuned hyperparameters and model found at paths "
                            f"{tuned_model_path.as_posix()} and {tuned_hyperparams_path.as_posix()} for {model_name}. "
                            "Defaulting to fresh Hyperparameter tuning.")
            LOG.info(f"Now tuning hyperparameters for {model_name}. This might take a while...")
            # note that tune_hyperparameters automatically sets the properties best_model and best_hps to be used
            # further down the line
            self._tune_hyperparameters(window=window,
                                       max_epochs=max_epochs,
                                       model_name=model_name,
                                       tuner_dir=tuner_dir,
                                       tuned_model_path=tuned_model_path,
                                       tuned_hyperparams_path=tuned_hyperparams_path)
            LOG.success(f"Successfully tuned hyperparameters for {model_name}")

        # run training loop with best model
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=2,
                                                       mode='min')

        LOG.info(f"Training {model_name} on the following Features:"
                 f"{[str(feature['name']) for feature in self._fc.FEATURE_DICT.values() if feature['select'] == 1]}.")
        self.history = self.best_model.fit(window.train, epochs=max_epochs,
                                           validation_data=window.val,
                                           callbacks=[early_stopping])

        LOG.success(f"Successfully trained {model_name}. Now saving...")

        # save trained model to disk
        if not overwrite and model_out_path.exists():
            FileExistsError(
                f"Model {model_name} at Path {model_out_path.as_posix()} already exists! "
                f"If you want to overwrite it please set overwrite=True.")
        else:
            model_obj.update({
                'model_name': model_name,
                'best_model': self.best_model,
                'best_hps': self.best_hps,
                'history': self.history,
                'train_mean': self.train_mean,
                'train_std': self.train_std,
                'seasonal': self.seasonal,
                'window': self.window,
                'train_df': self.train_df,
                'validation_df': self.validation_df,
                'test_df': self.test_df,
            })
            with open(model_out_path, 'wb') as f:
                pkl.dump(model_obj, f, -1)
            LOG.success(f"Successfully saved {model_name} to {model_out_path.as_posix()}")

        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'training': t_elapsed})
        LOG.info(f"Training took {t_elapsed}.")

    def evaluate(self, model_name: str):
        """
        Evaluates the model on the test set and returns the performance metrics.

        :param model_name: Name of the model to be evaluated.
        :type model_name: str
        """
        start = timer()
        # load model to evaluate
        model_path = (self._default_model_path / model_name).with_suffix('.pkl')
        LOG.info(f"Loading trained model from {model_path}.")
        with open(model_path, 'rb') as f:
            model_obj = pkl.load(f)

        # extract the relevant objects from the model object
        model = model_obj['best_model']
        window = model_obj['window']

        # save performance to disk
        if (Path.exists(self._processed_data_dir / "val_performance.pkl") and
                Path.exists(self._processed_data_dir / "performance.pkl")):
            with open(self._processed_data_dir / "val_performance.pkl", 'rb') as f:
                val_performance = pkl.load(f)

            with open(self._processed_data_dir / "performance.pkl", 'rb') as f:
                performance = pkl.load(f)

        else:
            val_performance = {}
            performance = {}

        val_performance[model_name] = model.evaluate(window.val, return_dict=True)
        performance[model_name] = model.evaluate(window.test, verbose=0, return_dict=True)

        with open(self._processed_data_dir / "val_performance.pkl", 'wb') as f:
            pkl.dump(val_performance, f, -1)

        with open(self._processed_data_dir / "performance.pkl", 'wb') as f:
            pkl.dump(performance, f, -1)

        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'evaluation': t_elapsed})
        LOG.info(f"Evaluation took {t_elapsed}.")

    def predict(self, data: WindowGenerator, model_path: Path, predictions_dir: Path):
        """
        Make predictions using the trained model. When no data is provided it will use the test data from the model object.

        :param data: Test dataset provided by ``WindowGenerator`` class.
        :type data: WindowGenerator

        :param model_path: Path to the trained model that is used for the prediction.
        :type model_path: pathlib.Path

        :param predictions_dir: Path to save the predictions.
        :type predictions_dir: pathlib.Path
        """
        start = timer()
        # model_name extracts the literal name of the model out of the model path
        match = re.search(r'\w+?(?=\.)', str(model_path))
        model_name = match.group(0) if match is not None else "default"
        predictions_path = predictions_dir / f"predictions_from_{model_name}.pkl"

        # Load the model
        LOG.info(f"Loading model from {model_path}.")
        with open(model_path, 'rb') as f:
            model_obj = pkl.load(f)

        model = model_obj['best_model']
        LOG.success(f"Successfully loaded model from {model_path}.")

        predictions = model.predict(data)
        LOG.success(f"Successfully predicted features.")

        self.predictions = predictions

        # save predictions to disk
        with open(predictions_path, 'wb') as f:
            pkl.dump(self.predictions, f, -1)
        LOG.success(f"Successfully saved predictions to {predictions_path}")

        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'prediction': t_elapsed})
        LOG.info(f"Prediction took {t_elapsed}.")
