from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import re
import pandas as pd

from epf.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def load_data(file_paths: list[Path], column_names: list[str]) -> pd.DataFrame:
    """
    Load and concatenate data from multiple CSV files given by file_paths.

    :param file_paths: List of file paths to the CSV files.
    :param column_names: List of column names.

    :returns: pd.DataFrame: DataFrame for each feature containing raw data from the years 2023 - 2024.
    """
    df = []
    for file_path in file_paths:
        # Skip extra row for DK and FR prices because there is an extra disclaimer header row
        if re.search(r'fr_prices|dk_._prices', str(file_path)):
            df.append(pd.read_csv(file_path, skiprows=3, names=column_names))
        else:
            df.append(pd.read_csv(file_path, header=1, names=column_names))

    concat_data = pd.concat(df)
    concat_data.reset_index(drop=True, inplace=True)
    concat_data['timestamp'] = pd.to_datetime(concat_data['timestamp'])
    concat_data.set_index('timestamp', inplace=True)

    # if na values are present interpolate them based on the timestamp
    if concat_data.isna().sum().sum() > 0:
        concat_data.interpolate(method='time', inplace=True)

    return concat_data