import keras
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter
from statsmodels.tsa.seasonal import MSTL

from epf.config import ModelConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR


def detect_and_remove_outliers(data: pd.DataFrame, window_length: int, n_sigma: int,
                               impute_method: str = None) -> pd.DataFrame:
    """
    Detect and remove outliers from features using Hampel Filter. Only imputes data where outliers are present.

    Hampel filter implementation from ``sktime.transformations.series.outlier_detection``.
    See `Hampel Filter documentation <https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.series.outlier_detection.HampelFilter.html>`_ for more information.

    :param data: The dataframe to perform outlier removal on.
    :type data: pd.DataFrame

    :param window_length: Window length for Hampel Filter.
    :type window_length: int

    :param n_sigma: Number of standard deviations for outlier detection.
    :type n_sigma: int

    :param impute_method: Method for imputing missing values.
    :type impute_method: str

    :returns: DataFrame containing all features with outliers removed where applicable.
    :rtype: pd.DataFrame
    """
    hampel = HampelFilter(window_length=window_length, n_sigma=n_sigma)

    # use the default imputer if no impute method is specified
    imputer = Imputer(method=impute_method) if impute_method is not None else Imputer()

    for col in data.columns:
        if 'price' in col:
            feature_hat = hampel.fit_transform(data[col])
            feature_imputed = imputer.fit_transform(feature_hat)

            # append new col with suffix hat
            data[col + '_hat'] = feature_imputed

            # drop touched col
            data.drop(columns=[col], inplace=True)

    return data


def load_and_concat_data(file_paths: list, column_name: str) -> pd.DataFrame:
    """
    Load and concatenate data from multiple CSV files given by ``file_paths``.
    NaN values are interpolated with ``pandas.DataFrame.interpolate(method='time')``

    :param file_paths: List of file paths to the CSV files.
    :param column_name: The column name to use.

    :returns: DataFrame for each feature containing raw data from the years 2023 - 2024.
    :rtype: pd.DataFrame
    """
    df = []
    cols = ['timestamp', column_name]

    for file_path in file_paths:
        # Skip extra row for DK and FR prices because there is an extra disclaimer header row
        if re.search(r'fr_prices|dk_._prices', file_path):
            df.append(pd.read_csv(RAW_DATA_DIR / file_path, skiprows=3, names=cols))
        else:
            df.append(pd.read_csv(RAW_DATA_DIR / file_path, header=1, names=cols))

    concat_data = pd.concat(df)
    concat_data.reset_index(drop=True, inplace=True)
    concat_data['timestamp'] = pd.to_datetime(concat_data['timestamp'])
    concat_data.set_index('timestamp', inplace=True)

    # if na values are present interpolate them based on the timestamp
    if concat_data.isna().sum() > 0:
        concat_data.interpolate(method='time', inplace=True)

    return concat_data


def remove_seasonal_component(data: pd.DataFrame, periods: list[int] = (24, 168)) -> tuple[
    pd.DataFrame, dict[str, MSTL]]:
    """
    Removes the seasonal component from each feature by multiple STL decomposition. Seasonal decomposition is done for
    each period provided with param ``periods``.

    MSTL implementation from ``statsmodels.tsa.seasonal``.
    See `MSTL documentation <https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.MSTL.html>`_ for more information.

    :param data: The dataframe to perform outlier removal on.
    :type data: pd.DataFrame

    :param periods: List of periods to remove from each feature.
        Defaults to [24, 7 * 24].
    :type periods: list[int]

    :returns: The modified DataFrame with seasonal components removed and the seasonal components as dict of column names
        and respective seasonal components.
    :rtype: pd.DataFrame
    """
    seasonal_component = {}

    for col in data.columns:
        mstl = MSTL(data[col], periods=periods).fit()
        # remove seasonal component from timeseries by adding all seasonal components and then subtracting from original timeseries
        data[col + '_rm_seasonal'] = data[col] - sum(mstl.seasonal[f'seasonal_{p}'] for p in periods)
        seasonal_component.update({col + '_rm_seasonal': mstl})
        # drop touched cols
        data.drop(columns=[col], inplace=True)

    return data, seasonal_component


def split_data(data: pd.DataFrame, train: float = 0.7, validation: float = 0.9) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, validation and test sets.
    Train and Validation params are the upper boundary for their respective splits. Lower boundaries are calculated.

    :param data: DataFrame to be split.
    :type data: pd.DataFrame

    :param train: Fraction of data to be used for training,
        defaults to 0.7
    :type train: float

    :param validation: Fraction of data to be used for validation,
        defaults to 0.9
    :type validation: float

    :returns: Train, Validation and Test DataFrames
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    train_split = int(len(data) * train)
    validation_split = int(len(data) * validation)

    return data[0:train_split], data[train_split:validation_split], data[validation_split:]


def builder(hp):
    """
    Builds a tunable ``keras.Model``. The specific model that is returned is determined by the ``MODEL_NAME`` parameter
    in the ``ModelConfig`` class.

    :returns: A compiled tunable ``keras.Sequential`` instance.
    """
    mc = ModelConfig()
    u_min, u_max, u_step = mc.UNIT_MIN_VALUE, mc.UNIT_MAX_VALUE, mc.UNIT_STEP
    k_min, k_max, k_step = mc.KERNEL_SIZE_MIN_VALUE, mc.KERNEL_SIZE_MAX_VALUE, mc.KERNEL_SIZE_STEP
    l_rate = mc.LEARNING_RATE
    dr_min, dr_max, dr_step = mc.DROPOUT_RATE_MIN_VALUE, mc.DROPOUT_RATE_MAX_VALUE, mc.DROPOUT_RATE_STEP
    d = mc.USE_DROPOUT
    n_min, n_max, n_step = mc.NUM_LAYERS_MIN, mc.NUM_LAYERS_MAX, mc.NUM_LAYERS_STEP

    lr = hp.Choice(name='learning_rate', values=l_rate)
    drop = hp.Boolean(name='drop', default=d)

    out_steps = mc.OUT_STEPS
    model_builder = mc.MODEL_BUILDER
    num_features = mc.NUM_FEATURES

    model = keras.Sequential()

    # differentiate between the input layers whether to use lstm, gru or conv
    if model_builder == "LSTM":
        model.add(keras.layers.LSTM(hp.Int(name='units',
                                           min_value=u_min,
                                           max_value=u_max,
                                           step=u_step), return_sequences=False))

    if model_builder == "GRU":
        model.add(keras.layers.GRU(hp.Int(name='units',
                                          min_value=u_min,
                                          max_value=u_max,
                                          step=u_step), return_sequences=False))

    if model_builder == "CONV":
        ks = hp.Int(name='kernel_size', min_value=k_min, max_value=k_max, step=k_step)

        model.add(keras.layers.Lambda(lambda x: x[:, -ks:, :]))
        model.add(keras.layers.Conv1D(hp.Int(name='units',
                                             min_value=u_min,
                                             max_value=u_max,
                                             step=u_step), activation='relu', kernel_size=ks))

    # after the initial layer, model building is identical
    if drop:
        model.add(keras.layers.Dropout(rate=hp.Float(name='dropout',
                                                     min_value=dr_min,
                                                     max_value=dr_max,
                                                     step=dr_step)))

    # add hidden layers and dropout layers if dropout is used
    for i in range(0, hp.Int('num_layers', min_value=n_min, max_value=n_max, step=n_step)):
        model.add(keras.layers.Dense(units=hp.Int(name='units_' + str(i),
                                                  min_value=u_min,
                                                  max_value=u_max,
                                                  step=u_step), activation="relu"))
        if drop:
            model.add(keras.layers.Dropout(rate=hp.Float(name='dropout_' + str(i),
                                                         min_value=dr_min,
                                                         max_value=dr_max,
                                                         step=dr_step)))

    # introduce dense layer with out_steps * num_features to implement single shot forecasting, needs to be reshaped
    # to [out_steps, num_features] afterward.
    model.add(keras.layers.Dense(out_steps * num_features,
                                 kernel_initializer=keras.initializers.zeros()))
    model.add(keras.layers.Reshape([out_steps, num_features]))

    # compile the model
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model


# code taken from [3] /references/refs.md
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, validation_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.validation_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='de_prices_hat_rm_seasonal', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
