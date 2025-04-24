import re
from pathlib import Path
from typing import Optional

import keras_tuner as kt
from loguru import logger

import holidays
import keras
import pickle as pkl
import pandas as pd

from epf.config import FeatureConfig, ModelConfig, MODELS_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from epf.util import load_and_concat_data, remove_seasonal_component, detect_and_remove_outliers, split_data, \
    ModelBuilder


class EpfPipeline:
    def __init__(
        self,
        feature_config: FeatureConfig = FeatureConfig(),
        model_config: ModelConfig = ModelConfig(),
        raw_data_dir = RAW_DATA_DIR,
        interim_data_dir = INTERIM_DATA_DIR,
        processed_data_dir = PROCESSED_DATA_DIR,
        default_model_path: Path = MODELS_DIR,
        model_name: Optional[str] = "lstm",
        model_builder: Optional[ModelBuilder] = DEFAULT_MODEL_BUILDER,
    ):
        """
        Initialize the pipeline with feature and model configurations.

        :param feature_config: Feature configuration.
        :param model_config: Model configuration.
        :param raw_data_dir: Raw data directory.
        :param interim_data_dir: Interim data directory.
        :param processed_data_dir: Processed data directory.
        :param default_model_path: Default path to save the model.
        :param model_name: Name of the model. If nothing is provided, the default lstm model is used.
        """

        self.fc = feature_config
        self.mc = model_config
        self.raw_data_dir = raw_data_dir
        self.interim_data_dir = interim_data_dir
        self.processed_data_dir = processed_data_dir
        self.model_path = default_model_path / model_name
        self.model_builder =

        self.model: keras.Model = keras.saving.load_model(self.model_path)

        self.raw_data = None
        self.feature_set = None
        self.predictions = None
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.num_features = None

    def _load_data(self):
        """ Loads raw data and saves it to interim directory using the feature configuration.
        """
        file_paths = self.fc.INPUT_PATHS
        col_names = self.fc.COL_NAMES
        to_resample = self.fc.TO_RESAMPLE
        df = []

        # read all years for each raw feature and concat them in load_and_concat_data, then append them to the big df
        # each file_path list contains multiple paths to the raw yearly data
        for file_path in file_paths.values():
            logger.info(f"Loading data from {file_path}")
            data = load_and_concat_data(file_path, col_names)

            # if the columns frequency is not hourly, resample it to hourly frequency
            if data.columns in to_resample:
                resample_freq: int = to_resample.get(data.columns)
                data = data[::resample_freq]

            df = df.append(data)

        logger.success(f"Raw data has been successfully loaded.")

        df = pd.concat(df, axis=1)
        raw_data_path = self.interim_data_dir / "raw_data"

        logger.info(f"Saving raw data to {raw_data_path}")
        df.to_csv(raw_data_path, index=True)

        self.raw_data = df

    def _generate_features(self):
        """
        Generates features using the feature configuration and saves the feature DataFrame to ``processed_data_dir``.

        Note that the current implementation of ``generate_features`` is slow because the feature engineering pipeline
        is ran for each feature regardless if its used later on during training.
        """
        feature_dict = self.fc.FEATURE_DICT
        periods = self.fc.PERIODS
        window_length = self.fc.WINDOW_LENGTH
        n_sigma = self.fc.N_SIGMA
        method = self.fc.METHOD
        seasonal_path = self.fc.SEASONAL_OUT_PATH / "seasonal_components.pkl"
        generate_lags = self.fc.GENERATE_LAGS
        generate_dummies = self.fc.GENERATE_DUMMIES

        data = self.raw_data

        logger.info(f"Running feature generation.")

        # detect and remove outliers in all price timeseries
        data = detect_and_remove_outliers(data, window_length, n_sigma, method)
        logger.success(f"Finished outlier removal.")

        # remove seasonal component from all timeseries and store seasonal component to later add back to predictions
        data, seasonal = remove_seasonal_component(data, periods)
        logger.success(f"Finished seasonal decomposition.")

        # seasonal components are stored for the time being to access them later when predictions are made.
        with open(seasonal_path, 'wb') as f:
            pkl.dump(seasonal, f, -1)
        logger.success(f"Successfully saved seasonal components to {seasonal_path}")

        # create calendar features
        if generate_dummies:
            logger.info(f"Generating dummies.")

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
            logger.info(f"Generating lagged prices.")

            data['de_lu_price_7_day_lag'] = data['de_lu_price_hat_rm_seasonal'].shift(7 * 24, fill_value=0)
            data['de_lu_price_1_day_lag'] = data['de_lu_price_hat_rm_seasonal'].shift(24, fill_value=0)
            data['de_lu_price_12_hour_lag'] = data['de_lu_price_hat_rm_seasonal'].shift(12, fill_value=0)
            data['de_lu_price_1_hour_lag'] = data['de_lu_price_hat_rm_seasonal'].shift(1, fill_value=0)

        # only select the columns that are specified in FEATURE_DICT
        feature_path = self.processed_data_dir / "features.csv"
        feature_set: pd.DataFrame = data.loc[:, [k for k, v in feature_dict.items() if v['select'] == 1]]
        feature_set.to_csv(feature_path, index=True)

        logger.success(f"Successfully saved generated features to {feature_path}")
        logger.info(f"Finished generating features.")

        self.feature_set = feature_set

    def _generate_training_data(self):
        """
        Generates Training, Validation and Test data and saves them to ``processed_data_dir``.
        """
        feature_set = self.feature_set
        train_split = self.mc.TRAIN_SPLIT
        validation_split = self.mc.VALIDATION_SPLIT
        data_path = self.processed_data_dir / "train_data"

        # generate training splits
        train_df, validation_df, test_df = split_data(feature_set, train_split, validation_split)

        # normalize data
        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train_df = (train_df - train_mean) / train_std
        self.validation_df = (validation_df - train_mean) / train_std
        self.test_df = (test_df - train_mean) / train_std

        # save train, val and test sets to csv
        # use the highest protocol available, denoted by -1
        with open(data_path / "train.pkl", 'wb') as f:
            pkl.dump(train_df, f, -1)

        with open(data_path / "validation_df.pkl", 'wb') as f:
            pkl.dump(validation_df, f, -1)

        with open(data_path / "test_df.pkl", 'wb') as f:
            pkl.dump(test_df, f, -1)

        self.num_features = feature_set.shape[1]

    def prep_data(self):
        """ Bundles all data preparation steps together for a single call in ``train``"""
        self._load_data()
        self._generate_features()
        self._generate_training_data()

    def _tune_hyperparameters(self):
        """ Tunes the hyperparameters for the provided model builder. Saves the hyperparameters to disk.
        """
        tuner = (kt.BayesianOptimization
        (
            self.model_builder,
            objective='mean_absolute_error',
            max_trials=10,
            directory='../models/tuner',
        ))

    def train(self, model_name: str = "epf_model", overwrite: bool = False):
        """
        Trains a model on the provided feature set using the model configuration. Saves the trained model to disk.

        :param model_name: A name for the model that identifies it when saved.
        :type model_name: str

        :param overwrite: Whether to overwrite existing model or not.
        :type overwrite: bool
        """
        model_out_path = self.model_path / model_name

        self.prep_data()
        logger.success(f"Successfully prepared training data for {model_name}")

        # build and compile model from config

        # generate windows

        # tune hyperparams

        # run training loop with best model

        self.model.save(model_out_path / ".keras", overwrite=overwrite)

    def predict(self, data: Path, model_path: Path, predictions_dir: Path):
        """
        Make predictions using the trained model.

        :param data: Path to the input data for prediction.
        :type data: pathlib.Path

        :param model_path: Path to the trained model that is used for the prediction.
        :type model_path: pathlib.Path

        :param predictions_dir: Path to save the predictions.
        :type predictions_dir: pathlib.Path
        """
        # model_name extracts the literal name of the model out of the model path
        match = re.search(r'\w+?(?=\.)', str(model_path))
        model_name = str(match) if match is not None else "default"
        predictions_path = predictions_dir / f"predictions_from_{model_name}.pkl"

        # Load the model
        self.model = keras.saving.load_model(model_path, compile=True)
        self.predictions = self.model.predict(data)
        logger.success(f"Successfully predicted features.")

        # save predictions to disk
        with open(predictions_path, 'wb') as f:
            pkl.dump(self.predictions, f, -1)
        logger.success(f"Successfully saved predictions to {predictions_path}")