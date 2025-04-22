import pandas as pd
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter


def detect_and_remove_outliers(feature, window_length, n_sigma, impute_method: str = None) -> pd.DataFrame:
    """
    Detect and remove outliers from features using Hampel Filter. Only imputes where outliers are present.

    :param feature: The feature where outliers should be removed.
    :param window_length: Window length for Hampel Filter.
    :param n_sigma: Number of standard deviations for outlier detection.
    :param impute_method: Method for imputing missing values.

    :returns: DataFrame containing all features with outliers removed where applicable.
    """
    hampel = HampelFilter(window_length=window_length, n_sigma=n_sigma)
    imputer = Imputer(method=impute_method) if impute_method is not None else Imputer()

    feature_hat = hampel.fit_transform(feature)
    feature_imputed = imputer.fit_transform(feature_hat)

    return feature_imputed

