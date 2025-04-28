from pathlib import Path

from loguru import logger

LOG = logger.bind(name="epf")

# Paths
def create_dir(path: Path, description: str):
    """Creates a directory if it does not exist and logs its path."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"{description} has been created. Path is: {path}")
    else:
        logger.info(f"{description} path is: {path}")

# Paths
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR: Path = PROJ_ROOT / "data"
create_dir(DATA_DIR, "DATA_DIR")

RAW_DATA_DIR: Path = DATA_DIR / "raw"
create_dir(RAW_DATA_DIR, "RAW_DATA_DIR")

INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
create_dir(INTERIM_DATA_DIR, "INTERIM_DATA_DIR")

PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
create_dir(PROCESSED_DATA_DIR, "PROCESSED_DATA_DIR")

TRAIN_DATA_DIR: Path = PROCESSED_DATA_DIR / "train_data"
create_dir(TRAIN_DATA_DIR, "TRAIN_DATA_DIR")

MODELS_DIR: Path = PROJ_ROOT / "models"
create_dir(MODELS_DIR, "MODELS_DIR")

PREDICTIONS_DIR: Path = MODELS_DIR / "predictions"
create_dir(PREDICTIONS_DIR, "PREDICTIONS_DIR")

REPORTS_DIR: Path = PROJ_ROOT / "reports"
create_dir(REPORTS_DIR, "REPORTS_DIR")

FIGURES_DIR: Path = REPORTS_DIR / "figures"
create_dir(FIGURES_DIR, "FIGURES_DIR")

class FeatureConfig:
    """
    Configuration class for feature generation.

    :ivar INPUT_PATHS: Dict containing raw data file names.
    :type INPUT_PATHS: dict[str,list[str]]

    :ivar COL_NAMES: List of column names for the features.
    :type COL_NAMES: dict[str,list[str]]

    :ivar TO_RESAMPLE: Dict of columns that need to be resampled to hourly frequency.
        Use value field to specify which frequency to use for resampling (e.g. 4 if original frequency is quarter hourly).
    :type TO_RESAMPLE: dict[str, int]

    :ivar FEATURE_DICT: Dict of available features with selector field if the feature should be considered.
    :type FEATURE_DICT: dict[str, dict[str, int | str | bool]]

    :ivar WINDOW_LENGTH: Length of the window for the Hampel filter.
    :type WINDOW_LENGTH: int

    :ivar N_SIGMA: Number of standard deviations for the Hampel filter.
    :type N_SIGMA: int

    :ivar METHOD: Method for imputation in the Hampel filter.
    :type METHOD: str
    """
    INPUT_PATHS = {
        'de_prices': ['de_prices_2023.csv', 'de_prices_2024.csv'],
        'de_load': ['de_load_2023.csv', 'de_load_2024.csv'],
        'de_solar_gen': ['de_solar_gen_2023.csv', 'de_solar_gen_2024.csv'],
        'de_wind_gen_offshore': ['de_wind_gen_offshore_2023.csv', 'de_wind_gen_offshore_2024.csv'],
        'de_wind_gen_onshore': ['de_wind_gen_onshore_2023.csv', 'de_wind_gen_onshore_2024.csv'],
        'de_gas_gen': ['de_gas_gen_2023.csv', 'de_gas_gen_2024.csv'],
        'de_lignite_gen': ['de_lignite_gen_2023.csv', 'de_lignite_gen_2024.csv'],
        'de_hard_coal_gen': ['de_hardcoal_gen_2023.csv', 'de_hardcoal_gen_2024.csv'],
        'ch_load': ['ch_load_2023.csv', 'ch_load_2024.csv'],
        'dk_load': ['dk_load_2023.csv', 'dk_load_2024.csv'],
        'fr_load': ['fr_load_2023.csv', 'fr_load_2024.csv'],
        'ch_prices': ['ch_prices_2023.csv', 'ch_prices_2024.csv'],
        'dk1_prices': ['dk_1_prices_2023.csv', 'dk_1_prices_2024.csv'],
        'dk2_prices': ['dk_2_prices_2023.csv', 'dk_2_prices_2024.csv'],
        'fr_prices': ['fr_prices_2023.csv', 'fr_prices_2024.csv'],
    }

    COL_NAMES = {
        'de_lu_prices': ['de_prices_2023', 'de_prices_2024'],
        'de_load': ['de_load_2023', 'de_load_2024'],
        'de_solar_gen': ['de_solar_gen_2023', 'de_solar_gen_2024'],
        'de_wind_gen_offshore': ['de_wind_gen_offshore_2023', 'de_wind_gen_offshore_2024'],
        'de_wind_gen_onshore': ['de_wind_gen_onshore_2023', 'de_wind_gen_onshore_2024'],
        'de_gas_gen': ['de_gas_gen_2023', 'de_gas_gen_2024'],
        'de_lignite_gen': ['de_lignite_gen_2023', 'de_lignite_gen_2024'],
        'de_hard_coal_gen': ['de_hard_coal_gen_2023', 'de_hard_coal_gen_2024'],
        'ch_load': ['ch_load_2023', 'ch_load_2024'],
        'dk_load': ['dk_load_2023', 'dk_load_2024'],
        'fr_load': ['fr_load_2023', 'fr_load_2024'],
        'ch_prices': ['ch_prices_2023', 'ch_prices_2024'],
        'dk1_prices': ['dk1_prices_2023', 'dk1_prices_2024'],
        'dk2_prices': ['dk2_prices_2023', 'dk2_prices_2024'],
        'fr_prices': ['fr_prices_2023', 'fr_prices_2024'],
    }

    TO_RESAMPLE = {
        'de_load': 4,
        'de_solar_gen': 4,
        'de_wind_gen_offshore': 4,
        'de_wind_gen_onshore': 4,
        'de_gas_gen': 4,
        'de_lignite_gen': 4,
        'de_hard_coal_gen': 4
    }

    FEATURE_DICT = {
        # prices
        'de_prices_hat_rm_seasonal': {
            'select': 1,
            'name': 'DE-LU Prices',
            'is-numerical': True
        },
        'ch_prices_hat_rm_seasonal': {
            'select': 0,
            'name': 'CH Prices',
            'is-numerical': True
        },
        'dk1_prices_hat_rm_seasonal': {
            'select': 0,
            'name': 'DK1 Prices',
            'is-numerical': True
        },
        'dk2_prices_hat_rm_seasonal': {
            'select': 0,
            'name': 'DK2 Prices',
            'is-numerical': True
        },
        'fr_prices_hat_rm_seasonal': {
            'select': 0,
            'name': 'FR Prices',
            'is-numerical': True
        },
        # price lags
        'de_lu_price_7_day_lag': {
            'select': 0,
            'name': 'DE-LU Prices 7-Day Lag',
            'is-numerical': True
        },
        'de_lu_price_1_day_lag': {
            'select': 0,
            'name': 'DE-LU Prices 24-Hour Lag',
            'is-numerical': True
        },
        'de_lu_price_12_hour_lag': {
            'select': 0,
            'name': 'DE-LU Prices 12-Hour Lag',
            'is-numerical': True
        },
        'de_lu_price_1_hour_lag': {
            'select': 0,
            'name': 'DE-LU Prices 1-Hour Lag',
            'is-numerical': True
        },
        # generation
        'de_solar_gen_rm_seasonal': {
            'select': 1,
            'name': 'DE Solar Generation',
            'is-numerical': True
        },
        'de_wind_gen_offshore_rm_seasonal': {
            'select': 1,
            'name': 'DE Wind Generation Offshore',
            'is-numerical': True
        },
        'de_wind_gen_onshore_rm_seasonal': {
            'select': 1,
            'name': 'DE Wind Generation Onshore',
            'is-numerical': True
        },
        'de_gas_gen_rm_seasonal': {
            'select': 0,
            'name': 'DE Gas Generation',
            'is-numerical': True
        },
        'de_lignite_gen_rm_seasonal': {
            'select': 0,
            'name': 'DE Lignite Generation',
            'is-numerical': True
        },
        'de_hard_coal_gen_rm_seasonal': {
            'select': 0,
            'name': 'DE Hard Coal Generation',
            'is-numerical': True
        },
        # loads
        'de_load_rm_seasonal': {
            'select': 1,
            'name': 'DE Load',
            'is-numerical': True
        },
        'ch_load_rm_seasonal': {
            'select': 0,
            'name': 'CH Load',
            'is-numerical': True
        },
        'dk_load_rm_seasonal': {
            'select': 0,
            'name': 'DK Load',
            'is-numerical': True
        },
        'fr_load_rm_seasonal': {
            'select': 0,
            'name': 'FR Load',
            'is-numerical': True
        },
        # dummies
        'month': {
            'select': 1,
            'name': 'Month',
            'is-numerical': False
        },
        'day_of_week': {
            'select': 1,
            'name': 'Day of Week',
            'is-numerical': False
        },
        'holiday': {
            'select': 1,
            'name': 'Holiday',
            'is-numerical': False
        },
    }

    # Hampel filter parameters
    WINDOW_LENGTH = 24
    N_SIGMA = 3
    METHOD = 'nearest'

    # STL Decompose Params
    SEASONAL_OUT_PATH = PROCESSED_DATA_DIR
    PERIODS: list[int] = [24, 7 * 24]

    # feature engineering
    GENERATE_LAGS: bool = False
    GENERATE_DUMMIES: bool = True


class ModelConfig:
    """
    Configuration class for model building and training.

    :ivar TRAIN_SPLIT: Upper boundary for the training split.
    :type TRAIN_SPLIT: float

    :ivar VALIDATION_SPLIT: Upper boundary for the validation split.
    :type VALIDATION_SPLIT: float

    :ivar MAX_EPOCHS: The upper boundary for epochs used during training.
    :type MAX_EPOCHS: int

    :ivar OUT_STEPS: The amount of timesteps to predict. Default is 24 i.e. 1 day.
    :type OUT_STEPS: int

    :ivar MODEL_BUILDER: The model builder to use. Choose from ``LSTM``, ``GRU`` and ``CONV``
    :type MODEL_BUILDER: str

    :ivar UNIT_MIN_VALUE: The minimum Value for units.
    :type UNIT_MIN_VALUE: int

    :ivar UNIT_MAX_VALUE: The maximum Value for units.
    :type UNIT_MAX_VALUE: int

    :ivar UNIT_MAX_VALUE_CONV: The maximum Value for units when a convolutional layer is used.
    :type UNIT_MAX_VALUE_CONV: int

    :ivar UNIT_STEP: The amount to increment unit during hyperparameter search.
    :type UNIT_STEP: int

    :ivar KERNEL_SIZE_MIN_VALUE: The minimum value for the kernel size. This only applies for convolutional layers.
    :type KERNEL_SIZE_MIN_VALUE: int

    :ivar KERNEL_SIZE_MAX_VALUE: The maximum value for the kernel size. This only applies for convolutional layers.
    :type KERNEL_SIZE_MAX_VALUE: int

    :ivar KERNEL_SIZE_STEP: The amount to increment kernel size during hyperparameter search.
    :type KERNEL_SIZE_STEP: int

    :ivar LEARNING_RATE: The learning rate for the optimizer.
    :type LEARNING_RATE: list[float]

    :ivar DROPOUT_RATE_MIN_VALUE: The minimum value for the dropout rate. This only applies if dropout is used.
    :type DROPOUT_RATE_MIN_VALUE: float

    :ivar DROPOUT_RATE_MAX_VALUE: The maximum value for the dropout rate. This only applies if dropout is used.
    :type DROPOUT_RATE_MAX_VALUE: float

    :ivar DROPOUT_RATE_STEP: The amount to increment dropout rate during hyperparameter search.
    :type DROPOUT_RATE_STEP: float

    :ivar USE_DROPOUT: Whether to use dropout when building the model.
    :type USE_DROPOUT: bool

    :ivar NUM_LAYERS_MIN: The minimum number of hidden layers in the model.
    :type NUM_LAYERS_MIN: int

    :ivar NUM_LAYERS_MAX: The maximum number of hidden layers in the model.
    :type NUM_LAYERS_MAX: int

    :ivar NUM_LAYERS_STEP: The amount to increment number of hidden layers during hyperparameter search.
    :type NUM_LAYERS_STEP: int

    :ivar LABEL_COL: Column name to use as label.
    :type LABEL_COL: str
    """

    TRAIN_SPLIT: float = 0.7
    VALIDATION_SPLIT: float = 0.9

    MAX_EPOCHS: int = 20
    OUT_STEPS: int = 24

    MODEL_BUILDER = "LSTM"
    NUM_FEATURES = sum([1 for feature in FeatureConfig.FEATURE_DICT.values() if feature['select'] == 1])

    # hp tuner params
    UNIT_MIN_VALUE: int = 32
    UNIT_MAX_VALUE: int = 512
    UNIT_MAX_VALUE_CONV: int = 2048
    UNIT_STEP: int = 32

    KERNEL_SIZE_MIN_VALUE: int = 1
    KERNEL_SIZE_MAX_VALUE: int = 10
    KERNEL_SIZE_STEP: int = 1

    LEARNING_RATE: list[float] = [1e-2, 1e-3, 1e-4]

    DROPOUT_RATE_MIN_VALUE: float = 0
    DROPOUT_RATE_MAX_VALUE: float = 0.99
    DROPOUT_RATE_STEP: float = 0.05

    USE_DROPOUT: bool = True

    NUM_LAYERS_MIN: int = 0
    NUM_LAYERS_MAX: int = 5
    NUM_LAYERS_STEP: int = 1

    LABEL_COL = 'de_prices_hat_rm_seasonal'
