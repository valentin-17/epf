import pickle
from datetime import datetime, timedelta, timezone

import keras
import re
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from pandas import Timestamp
from sklearn.preprocessing import MinMaxScaler
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter
from statsmodels.tsa.seasonal import MSTL

from epf.config import ModelConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR


def detect_and_remove_outliers(data: pd.DataFrame, window_length: int,
                               n_sigma: int, impute_method: str = None
                               ) -> pd.DataFrame:
    """
    Detect and remove outliers from features using Hampel Filter.
    Only imputes data where outliers are present.

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
    imputer = Imputer(method=impute_method) if impute_method is not None \
        else Imputer()

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
    """Load and concatenate data from multiple CSV files given by ``file_paths``.
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
    if concat_data.isna().sum().sum() > 0:
        concat_data.interpolate(method='time', inplace=True)

    return concat_data


def remove_seasonal_component(data: pd.DataFrame,
                              periods: list[int] = (24, 168)
                              ) -> tuple[pd.DataFrame, dict[str, MSTL]]:
    """Removes the seasonal component from each feature by multiple STL decomposition. Seasonal decomposition is done for each period provided with param ``periods``.

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
        # remove seasonal component from timeseries by adding
        # all seasonal components and then subtracting from original timeseries
        data[col + '_rm_seasonal'] \
            = (data[col] - sum(mstl.seasonal[f'seasonal_{p}'] for p in periods))
        seasonal_component.update({col + '_rm_seasonal': mstl})
        # drop touched cols
        data.drop(columns=[col], inplace=True)

    return data, seasonal_component


def split_data(data: pd.DataFrame, train: float | datetime = 0.7, validation: float | datetime = 0.9) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into train, validation and test sets. Train and Validation params are the upper boundary for their respective splits. Lower boundaries are calculated.

    :param data: DataFrame to be split.
    :type data: pd.DataFrame

    :param train: Fraction of data to be used for training, or start timestamp of training data,
        defaults to 0.7
    :type train: float | datetime

    :param validation: Fraction of data to be used for validation, or start timestamp of training data,
        defaults to 0.9
    :type validation: float | datetime

    :returns: Train, Validation and Test DataFrames
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    if isinstance(train, datetime) and isinstance(validation, datetime):
        return data[:train], data[train+timedelta(hours=1):validation], data[validation+timedelta(hours=1):]
    else:
        # work out train and validation splits if they are passed as floats
        train_split = int(len(data) * train)
        validation_split = int(len(data) * validation)
        return data[:train_split], data[train_split:validation_split], data[validation_split:]


def builder(hp):
    """Builds a tunable ``keras.Model``. The specific model that is returned is determined by the ``MODEL_NAME`` parameter in the ``ModelConfig`` class.

    :returns: A compiled tunable ``keras.Sequential`` instance.
    """
    mc = ModelConfig()
    u_min, u_max, u_step = mc.UNIT_MIN_VALUE, mc.UNIT_MAX_VALUE, mc.UNIT_STEP
    l_min, l_max, l_step = mc.LR_MIN_VALUE, mc.LR_MAX_VALUE, mc.LR_STEP
    dr_min, dr_max, dr_step = mc.DROPOUT_RATE_MIN_VALUE, mc.DROPOUT_RATE_MAX_VALUE, mc.DROPOUT_RATE_STEP
    n_min, n_max, n_step = mc.NUM_LAYERS_MIN, mc.NUM_LAYERS_MAX, mc.NUM_LAYERS_STEP

    lr = hp.Float(name='learning_rate', min_value=l_min, max_value=l_max, step=l_step)
    nl = hp.Int(name = 'num_layers', min_value=n_min, max_value=n_max, step=n_step)

    hidden_layers = mc.USE_HIDDEN_LAYERS

    out_steps = mc.OUT_STEPS
    model_builder = mc.MODEL_BUILDER
    label_col = mc.LABEL_COL

    model = keras.Sequential()

    # initially return sequences is determined whether hidden layers are used or not, in a single layer approach
    # return sequences should be false because this one layer already outputs the predictions
    # otherwise set to true because the next layer needs the output of the previous layer
    # with shape [batch_size, time_steps, features] and ndims = 3
    # if hidden_layers is set to False then the model is a single layer model thus return_sequences should be false
    rs = hidden_layers

    # differentiate between the input layers whether to use lstm or gru
    if model_builder == "LSTM":
        model.add(keras.layers.LSTM(units=hp.Int(name='units', min_value=u_min, max_value=u_max, step=u_step),
                                    recurrent_dropout=hp.Float(name='rec_dropout', min_value=dr_min, max_value=dr_max,
                                                               step=dr_step),
                                    return_sequences=rs))

    if model_builder == "GRU":
        model.add(keras.layers.GRU(units=hp.Int(name='units', min_value=u_min, max_value=u_max, step=u_step),
                                   recurrent_dropout=hp.Float(name='rec_dropout', min_value=dr_min, max_value=dr_max,
                                                              step=dr_step),
                                   return_sequences=rs))

    # add hidden layers, if hidden layers is True effectively making this a rnn stacked approach with either lstm or gru layers
    if hidden_layers:
        for i in range(0, nl):
            # set the return sequences to true for all intermediary layers
            # the last layer receives return_sequences = False to output the final predictions with ndims = 2
            rs = hidden_layers if i < nl - 1 else False

            if model_builder == "LSTM":
                model.add(keras.layers.LSTM(units=hp.Int(name='units_' + str(i+1), min_value=u_min, max_value=u_max,
                                                         step=u_step),
                                            recurrent_dropout=hp.Float(name='rec_dropout_' + str(i+1), min_value=dr_min,
                                                                       max_value=dr_max, step=dr_step),
                                            return_sequences=rs))
            if model_builder == "GRU":
                model.add(keras.layers.GRU(units=hp.Int(name='units_' + str(i+1), min_value=u_min, max_value=u_max,
                                                        step=u_step),
                                           recurrent_dropout=hp.Float(name='rec_dropout_' + str(i+1), min_value=dr_min,
                                                                      max_value=dr_max, step=dr_step),
                                           return_sequences=rs))

    # introduce dense layer with out_steps * len(label_col) to implement single shot forecasting, needs to be reshaped
    # to [out_steps, len(label_col)] afterward. multiply by one since we only want to forecast one single feature, the price
    model.add(keras.layers.Dense(out_steps * len([label_col]),
                                 kernel_initializer=keras.initializers.zeros()))
    model.add(keras.layers.Reshape([out_steps, len([label_col])]))

    # compile the model
    model.compile(loss=keras.losses.MeanAbsoluteError(),
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=[
                      keras.metrics.MeanAbsoluteError(),
                      keras.metrics.RootMeanSquaredError(),
                  ])

    return model

def predict_with_timestamps(model_obj):
    """Predicts on a ((x, y), timestamps) dataset and returns a pandas DataFrame with flattened predictions and their corresponding timestamps.
    Also reseasonalizes and denormalizes the predictions data.

    :param model_obj: The model object created during training
    :type model_obj: dict

    :returns pd.DataFrame: DataFrame with columns ['timestamp'] + label_columns
    """
    all_preds = []
    all_trues = []
    all_times = []

    model = model_obj['best_model']
    dataset = model_obj['window'].test_ts
    label_columns = model_obj['window'].label_columns
    t_min = model_obj['train_min']['de_prices_hat_rm_seasonal']
    t_max = model_obj['train_max']['de_prices_hat_rm_seasonal']
    mstl = model_obj['seasonal']['de_prices_hat_rm_seasonal'].seasonal

    for (x_batch, y_batch), ts_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        all_preds.append(preds)
        all_trues.append(y_batch)
        all_times.append(ts_batch.numpy())

    all_preds  = np.concatenate(all_preds, axis=0)       # shape: [n, out_steps, features]
    all_trues  = np.concatenate(all_trues, axis=0)       # shape: [n, out_steps, features]
    all_times  = np.concatenate(all_times, axis=0)       # shape: [n, out_steps]

    # shape is [n, out_steps, 1]
    n, out_steps, features = all_preds.shape

    # reshape for dataFrame
    flat_preds = all_preds.reshape(n, out_steps)  # now [n, out_steps]
    flat_trues = all_trues.reshape(n, out_steps)  # now [n, out_steps]
    flat_times = all_times[:, 0]  # use first time per sample

    flat_times = pd.to_datetime(flat_times, unit='s')

    colname = label_columns[0] if isinstance(label_columns, list) else label_columns
    step_columns = [f"{colname}_t+{i + 1}" for i in range(out_steps)]

    pred = pd.DataFrame(flat_preds, columns=step_columns)
    pred.insert(0, "timestamp", flat_times)
    pred.set_index('timestamp', inplace=True)

    true = pd.DataFrame(flat_trues, columns=step_columns)
    true.insert(0, "timestamp", flat_times)
    true.set_index('timestamp', inplace=True)

    # denormalize
    pred = (pred * (t_max - t_min)) + t_min
    true = (true * (t_max - t_min)) + t_min

    # reseasonalize
    horizons = range(1, 25)
    col_names = [f't+{h}' for h in horizons]

    seasonal_24 = pd.DataFrame(index=mstl.index, columns=col_names)
    seasonal_168 = pd.DataFrame(index=mstl.index, columns=col_names)

    for h in horizons:
        shift_amount = 24 - h  # 168 hours = 1 week
        seasonal_24[f'de_prices_hat_rm_seasonal_t+{h}'] = mstl['seasonal_24'].shift(shift_amount)
        seasonal_168[f'de_prices_hat_rm_seasonal_t+{h}'] = mstl['seasonal_168'].shift(shift_amount)

    start = pred.index[0]
    end = pred.index[-1]

    seasonal_24.index = seasonal_24.index.tz_localize(None)
    seasonal_168.index = seasonal_24.index.tz_localize(None)
    seasonal_24 = seasonal_24.loc[start:end]
    seasonal_168 = seasonal_24.loc[start:end]

    for col in pred.columns:
        pred[col] = (pred[col] + sum([seasonal_24[col], seasonal_168[col]]))

    for col in true.columns:
        true[col] = (true[col] + sum([seasonal_24[col], seasonal_168[col]]))

    return pred, true


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

    def split_window(self, features, timestamps=None):
        """Either returns (inputs, labels) or ((inputs, labels), timestamps) depending on the value of ``timestamps``.
        ((inputs, labels), timestamps) is really only used for plotting and evaluation."""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        if timestamps is not None:
            label_times = timestamps[:, self.labels_slice]  # shape [batch, label_width]
            label_times.set_shape([None, self.label_width])
            return (inputs, labels), label_times

        return inputs, labels

    def make_dataset(self, df, return_timestamps=False):
        """Creates a dataset from the given DataFrame. If return_timestamps is True, the dataset will include timestamps."""
        data = np.array(df, dtype=np.float32)

        # when returning timestamps shuffle needs to be set to false in order to retain the order of the timestamps
        shuffle = not return_timestamps

        ds = keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=32,
        )

        if return_timestamps:
            # see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#from-timestamps-to-epoch
            timestamps = (df.index - pd.Timestamp("1970-01-01", tz=timezone.utc)) // pd.Timedelta("1s")

            ds_t = keras.utils.timeseries_dataset_from_array(
                data=timestamps,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,
                batch_size=32,
            )

            # Return ((inputs, labels), timestamps)
            ds = tf.data.Dataset.zip((ds, ds_t))

        # Default: return (inputs, labels)
        ds = ds.map(self.split_window, num_parallel_calls=tf.data.AUTOTUNE, name='split_window').cache()

        # prefetch
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE, name='prefetch')

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
    def test_ts(self):
        return self.make_dataset(self.test_df, return_timestamps=True)

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
        """Plot the `inputs`, `labels`, and `predictions` for a given model.

        :param model: The model to use for predictions.
        :type model: keras.Sequential

        :param plot_col: The column to plot.
        :type plot_col: str

        :param max_subplots: The maximum number of subplots to show.
        :type  max_subplots: int
        """
        inputs, labels = self.example

        sns.set_style("ticks")

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        pc = plot_col.replace('_hat_rm_seasonal', '').replace('de_prices', 'DE prices')

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{pc}\n(normed and deseasonalized)')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', c='#840853', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        marker='.', label='Labels', c='#840853', s=10)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='x', label='Predictions',
                            c='#3a609c', s=10)

            plt.xticks(np.arange(0, self.total_window_size, 24))

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
