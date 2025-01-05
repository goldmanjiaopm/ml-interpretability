from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """
        Train the model.

        Args:
            features: Training features
            labels: Training labels
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained model.

        Args:
            features: Features to predict on

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to load the model from
        """
        pass
