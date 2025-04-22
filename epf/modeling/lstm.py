from abc import ABC

from epf.modeling import Model


class LSTM(Model, ABC):
    """
    A class to configure and build an LSTM-model for time series forecasting
    """
    def build(self):
        return

    def train(self):
        return

    def predict(self):
        return