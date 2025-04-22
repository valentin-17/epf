from abc import ABC, abstractmethod

import keras

from epf.modeling.util import WindowGenerator


class Model(ABC):
    """
    Abstract base class to represent a machine learning model. Used for implementation of specific models.
    """
    @abstractmethod
    def build(self, use_dropout: bool = False) -> keras.Model:
        """
        Build the model architecture.
        :param use_dropout: Indicates whether to use dropout layers in the model.

        :returns: A Keras model instance.
        """
        pass

    @abstractmethod
    def tune(self, model: keras.Model) -> keras.Model:
        """
        Tune the model hyperparameters and return the best model. Automatically saves the best model to disk.

        :param model: The Keras model to tune.
        :returns: The best keras model instance.
        """
        pass

    @abstractmethod
    def train(self, train: WindowGenerator.train, model: keras.Model) -> keras.Model:
        """
        Trains the provided model on the training data.

        :param train: The training data generator.
        :param model: The Keras model to train.

        :returns: A Keras model instance.
        """
        pass

    @abstractmethod
    def predict(self):
        pass
