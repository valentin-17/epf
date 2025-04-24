from pathlib import Path

from loguru import logger


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
PREDICTIONS_DIR = MODELS_DIR / "predictions"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


class FeatureConfig:
    """
    Configuration class for feature generation.

    :param INPUT_PATHS: Dict containing raw data file names.
    :type INPUT_PATHS: dict[str,list[str]]

    :param COL_NAMES: List of column names for the features.
    :type COL_NAMES: list[str]

    :param TO_RESAMPLE: Dict of columns that need to be resampled to hourly frequency.
        Use value field to specify which frequency to use for resampling (e.g. 4 if original frequency is quarter hourly).
    :type TO_RESAMPLE: dict[str, int]

    :param FEATURE_DICT: Dict of available features with selector field if the feature should be considered.
    :type FEATURE_DICT: dict[str, dict[str, int | str | bool]]

    :param WINDOW_LENGTH: Length of the window for the Hampel filter.
    :type WINDOW_LENGTH: int

    :param N_SIGMA: Number of standard deviations for the Hampel filter.
    :type N_SIGMA: int

    :param METHOD: Method for imputation in the Hampel filter.
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

    COL_NAMES = [
        'de_lu_prices',
        'de_load',
        'de_solar_gen',
        'de_wind_gen_offshore',
        'de_wind_gen_onshore',
        'de_gas_gen',
        'de_lignite_gen',
        'de_hard_coal_gen',
        'ch_load',
        'dk_load',
        'fr_load',
        'ch_prices',
        'dk1_prices',
        'dk2_prices',
        'fr_prices'
    ]

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
        'de_lu_price_hat_rm_seasonal': {
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
            'select': 0,
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

    :param TRAIN_SPLIT: Upper boundary for the training split.
    :type TRAIN_SPLIT: float

    :param VALIDATION_SPLIT: Upper boundary for the validation split.
    :type VALIDATION_SPLIT: float

    :param USE_DROPOUT: Whether to use dropout when building the model.
    :type USE_DROPOUT: bool
    """
    TRAIN_SPLIT: float = 0.7
    VALIDATION_SPLIT: float = 0.9

    USE_DROPOUT: bool = True