from pathlib import Path

from epf.dataset import load_data
from loguru import logger
from epf.config import FeatureConfig

from epf.config import PROCESSED_DATA_DIR

def generate_all_features(
    features: list[str],
) -> None:
    """
    Generate features from the input dataset and save to output path.

    :param features: List of features to generate.
    :type features: list[str]
    """

    fc = FeatureConfig()

    input_paths = fc.input_paths()
    output_path = PROCESSED_DATA_DIR
    column_names = fc.col_names

    logger.info(f"Generating features from {input_paths} writing to {output_path}")
    logger.info(f"Features to generate: {features}")

    # generate features
    load_data(input_paths, column_names)

    # Save the generated features
    logger.success(f"Features saved to {output_path}"
)