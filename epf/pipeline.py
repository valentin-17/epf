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

from sklearn.preprocessing import MinMaxScaler

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
        self.train_min = None
        self.train_max = None
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
        self.timings.update({'data_loading': t_elapsed})
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

    def _generate_training_data(self, path: Path | None = None):
        """
        Generates Training, Validation and Test data.
        """
        start = timer()
        if self.train_df is not None:
            LOG.info("Training data has already been generated. Skipping training data generation.")
            return

        if path is not None:
            LOG.info(f"Loading training data from {path}.")
            feature_dict = self._fc.FEATURE_DICT
            data = pd.read_csv(path, index_col=0, parse_dates=True)
            feature_set: pd.DataFrame = data.loc[:, [k for k, v in feature_dict.items() if v['select'] == 1]]
        else:
            feature_set = self.feature_set

        train_split = self._mc.TRAIN_SPLIT
        validation_split = self._mc.VALIDATION_SPLIT

        # generate training splits
        train_df, validation_df, test_df = split_data(feature_set, train_split, validation_split)

        # normalize data with min max normalization
        train_min = train_df.min()
        train_max = train_df.max()

        self.train_min = train_min
        self.train_max = train_max

        self.train_df = (train_df - train_min) / (train_max - train_min)
        self.validation_df = (validation_df - train_min) / (train_max - train_min)
        self.test_df = (test_df - train_min) / (train_max - train_min)

        LOG.info(f"Finished generating training data.")
        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'training_data_generation': t_elapsed})
        LOG.info(f"Training data generation took {t_elapsed}.")

    def prep_data(self):
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

    def train(self, model_name: str, overwrite: bool):
        """
        Trains a model on the training data that is currently saved on the disk. The model configuration provides
        necessary tuning and training parameters. When ``prep_data`` = ``True``, the data preparation pipeline is called
        (see ``_prep_data()``). Saves the trained model to ``MODELS_DIR/model_name``.

        :param model_name: A name for the model that identifies it when saved.
        :type model_name: str

        :param overwrite: Whether to overwrite an existing model or not.
        :type overwrite: bool
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
        seasonality_period = self._mc.SEASONALITY_PERIOD
        input_width_factor = self._mc.INPUT_WIDTH_FACTOR
        input_width = int(seasonality_period * input_width_factor)
        max_epochs = self._mc.MAX_EPOCHS
        label_col = self._mc.LABEL_COL

        LOG.info(f"Data paths for training loop successfully initialized.\n"
                 f"Model output: {model_out_path.as_posix()}.\n"
                 f"Tuned model path: {tuned_model_path.as_posix()}.\n"
                 f"Tuned Hyperparameters path: {tuned_hyperparams_path.as_posix()}.\n")

        model_obj = {
            'model_name': model_name,
            'best_model': None,
            'best_hps': None,
            'history': None,
            'train_min': None,
            'train_max': None,
            'seasonal': None,
            'window': None,
            'train_df': None,
            'validation_df': None,
            'test_df': None,
            'feature_set': None,
            'timings': None,
        }

        self.prep_data()

        # generate windows
        window = WindowGenerator(train_df=self.train_df,
                                 validation_df=self.validation_df,
                                 test_df=self.test_df,
                                 input_width=input_width,
                                 label_width=out_steps,
                                 shift=out_steps,
                                 label_columns=[label_col], )

        self.window = window

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

        end = timer()
        t_elapsed = strftime('%Hh:%Mm:%Ss', time.gmtime(end - start))
        self.timings.update({'training': t_elapsed})
        LOG.info(f"Training took {t_elapsed}.")

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
                'train_min': self.train_min,
                'train_max': self.train_max,
                'seasonal': self.seasonal,
                'window': self.window,
                'train_df': self.train_df,
                'validation_df': self.validation_df,
                'test_df': self.test_df,
                'feature_set': self.feature_set,
                'timings': self.timings,
            })
            with open(model_out_path, 'wb') as f:
                pkl.dump(model_obj, f, -1)
            LOG.success(f"Successfully saved {model_name} to {model_out_path.as_posix()}")

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
