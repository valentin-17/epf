from pathlib import Path
from typing import Optional, List, final

from loguru import logger


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


class FeatureConfig:
    """
    Configuration class for feature generation.

    :param raw_data:    Dictionary containing raw data file names.
    :param col_names:   List of column names for the features.
    :param features:    List of features to be generated. For available features, see AVAILABLE_FEATURES.
                        Defaults to ['de_prices', 'de_solar_gen', 'de_wind_gen_offshore', 'de_wind_gen_onshore'].
    """

    raw_data = {
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

    col_names = [
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

    def __init__(self, input_path: Path = RAW_DATA_DIR, output_path: Path = INTERIM_DATA_DIR,
                 features: Optional[List[str]] = None,) -> None:
        """ Constructor for FeatureConfig class."""
        # use the default feature set if no feature selection is provided by the user
        if features is None:
            features = ['de_prices', 'de_solar_gen', 'de_wind_gen_offshore', 'de_wind_gen_onshore']
        self.input_path = input_path
        self.output_path = output_path
        self.features = features

    def input_paths(self) -> List[Path]:
        """
        Returns a list of file paths for the features.
        """
        return [RAW_DATA_DIR / file for file in self.raw_data.values()]