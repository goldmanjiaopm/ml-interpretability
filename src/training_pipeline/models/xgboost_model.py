from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from src.training_pipeline.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model."""
        super().__init__(config)
        self.model = xgb.XGBClassifier(**self.config)
        self.thresholds = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train the XGBoost model."""
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model."""
        if self.thresholds is None:
            return pd.Series(self.model.predict(features))

        proba = self.predict_proba(features)
        predictions = np.argmax(proba >= self.thresholds.reshape(1, -1), axis=1)
        return pd.Series(predictions)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        return self.model.predict_proba(features)

    def set_thresholds(self, thresholds: np.ndarray) -> None:
        """Set optimal thresholds for prediction."""
        self.thresholds = thresholds

    def save(self, path: str) -> None:
        """Save the XGBoost model."""
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load the XGBoost model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
