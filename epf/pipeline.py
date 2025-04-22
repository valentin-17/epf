from typing import Optional

import keras

from epf.config import FeatureConfig, ModelConfig


class EpfPipeline:
    def __init__(
        self,
        feature_config: FeatureConfig = FeatureConfig(),
        model_config: ModelConfig = ModelConfig(),
        models: Optional[list[keras.Model]] = None
    ):
        """
        Initialize the pipeline with feature and model configurations.

        :param feature_config: Feature configuration.
        :param model_config: Model configuration.
        :param models: List of models.
        """
        self.feature_config = feature_config
        self.model_config = model_config
        self.models = models

    def load_data(self):
        """
        Load data using the feature configuration.
        """
        pass

    def generate_features(self):
        """
        Generate features using the feature configuration.
        """
        pass

    def train(self):
        """
        Train the models using the model configuration.
        """
        pass

    def predict(self):
        """
        Make predictions using the trained models.
        """
        pass

    def save_model(self):
        """
        Save the trained models.
        """
        pass

    def load_model(self):
        """
        Load the trained models.
        """
        pass