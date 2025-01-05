from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


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
        Make predictions using the trained model.

        Args:
            features: Features to predict on

        Returns:
            Predictions
        """
        return pd.Series(self.model.predict(features))
