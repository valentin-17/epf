from abc import ABC

from epf.modeling import Model


class CNN(Model, ABC):
    """
    A class to configure and build a CNN-model for time series forecasting
    """
    def build(self):
        return

    def train(self):
        return

    def predict(self):
        return