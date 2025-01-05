from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.metrics import classification_report

from src.training_pipeline.models.base_model import BaseModel
from src.training_pipeline.models.random_forest import RandomForestModel


def load_processed_data(
    data_path: Path = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load processed features and labels.

    Args:
        data_path: Optional path to processed data directory. If None, uses project root.
    """
    if data_path is None:
        # Get the project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data/processed"

    train_features = pd.read_csv(data_path / "train_features.csv")
    train_labels = pd.read_csv(data_path / "train_labels.csv")["Labels"]
    val_features = pd.read_csv(data_path / "val_features.csv")
    val_labels = pd.read_csv(data_path / "val_labels.csv")["Labels"]

    return train_features, train_labels, val_features, val_labels


def load_model_config(config_path: Path) -> Dict[str, Any]:
    """Load model configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    """
    Get model instance based on name and config.

    Args:
        model_name: Name of the model to use
        config: Model configuration

    Returns:
        Initialized model instance
    """
    models = {
        "random_forest": RandomForestModel,
        # Add more models here
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

    return models[model_name](config)


def evaluate_model(model: BaseModel, val_features: pd.DataFrame, val_labels: pd.Series) -> Dict[str, Any]:
    """Evaluate model performance."""
    predictions = model.predict(val_features)
    report = classification_report(val_labels, predictions, output_dict=True)
    return report


def train_and_evaluate(model_name: str, config_path: Path) -> Dict[str, Any]:
    """
    Train and evaluate a model.

    Args:
        model_name: Name of the model to train
        config_path: Path to model configuration file

    Returns:
        Evaluation metrics
    """
    # Load data
    train_features, train_labels, val_features, val_labels = load_processed_data()

    # Load config and initialize model
    configs = load_model_config(config_path)
    model = get_model(model_name, configs[model_name])

    # Train and evaluate
    model.train(train_features, train_labels)
    metrics = evaluate_model(model, val_features, val_labels)

    return metrics
