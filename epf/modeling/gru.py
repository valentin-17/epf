from abc import ABC

from epf.modeling import Model


class GRU(Model, ABC):
    """
    A class to configure and build a GRU-model for time series forecasting
    """
    def build(self):
        return

    def train(self):
        return

    def predict(self):
        return