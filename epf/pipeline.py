from pathlib import Path
from typing import Optional

import keras
import numpy as np

from epf.config import FeatureConfig, ModelConfig, MODELS_DIR


class EpfPipeline:
    def __init__(
        self,
        feature_config: FeatureConfig = FeatureConfig(),
        model_config: ModelConfig = ModelConfig(),
        default_model_path: Path = MODELS_DIR,
        model_name: Optional[str] = "lstm",
    ):
        """
        Initialize the pipeline with feature and model configurations.

        :param feature_config: Feature configuration.
        :param model_config: Model configuration.
        :param default_model_path: Default path to save the model.
        :param model_name: Name of the model. If nothing is provided, the default lstm model is used.
        """

        self.fc = feature_config
        self.mc = model_config
        self.model_path = default_model_path / model_name

        self.model: keras.Model = keras.saving.load_model(self.model_path)
        self.predictions = None

    def _load_data(self):
        """
        Load raw data and save it to interim directory using the feature configuration.
        """
        pass

    def _generate_features(self):
        """
        Generate features using the feature configuration.
        """
        pass

    def train(self):
        """
        Train the models using the model configuration.
        """
        # self.model =
        keras.saving.save_model(self.model, self.model_path)

    def predict(self, data: Path, model_path: Path):
        """
        Make predictions using the trained models.

        :param data: Path to the data for prediction.
        :param model_path: Path to the trained model.
        """

        # Load the model
        self.model = keras.saving.load_model(model_path, compile=True)

        self.predictions = self.model.predict(data)

    def save_predictions(self, output_path):
        """
        Save the predictions to the specified output path.

        :param output_path: Path to save the predictions.
        """
        self.predictions.to_csv(output_path, index=False)