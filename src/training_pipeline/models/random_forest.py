import os
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.training_pipeline.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier implementation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Random Forest model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model = RandomForestClassifier(**self.config)
        self.thresholds = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """
        Train the Random Forest model.

        Args:
            features: Training features
            labels: Training labels
        """
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained model and optimal threshold.

        Args:
            features: Features to predict on

        Returns:
            Predictions
        """
        if self.thresholds is None:
            return pd.Series(self.model.predict(features))

        # Get probability predictions
        proba = self.predict_proba(features)

        # Apply thresholds for each class
        predictions = np.argmax(proba >= self.thresholds.reshape(1, -1), axis=1)
        return pd.Series(predictions)

    def save(self, path: str) -> None:
        """Save the trained model to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load a trained model from a pickle file."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the trained model.

        Args:
            features: Features to predict on

        Returns:
            Array of prediction probabilities
        """
        return self.model.predict_proba(features)

    def set_thresholds(self, thresholds: np.ndarray) -> None:
        """Set optimal thresholds for prediction."""
        self.thresholds = thresholds
