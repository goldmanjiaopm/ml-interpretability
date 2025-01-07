from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.metrics import classification_report

from src.training_pipeline.models.base_model import BaseModel
from src.training_pipeline.models.random_forest import RandomForestModel
from src.training_pipeline.tuning import tune_hyperparameters
from src.training_pipeline.models.param_spaces import get_random_forest_param_space


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


def train_and_evaluate(model_name: str, config_path: Path, tune: bool = False, n_trials: int = 100) -> Dict[str, Any]:
    """
    Train and evaluate a model.

    Args:
        model_name: Name of the model to train
        config_path: Path to model configuration file
        tune: Whether to perform hyperparameter tuning
        n_trials: Number of optimization trials if tuning

    Returns:
        Evaluation metrics
    """
    # Load data
    train_features, train_labels, val_features, val_labels = load_processed_data()

    # Get model class
    models = {
        "random_forest": RandomForestModel,
        # Add more models here
    }
    model_class = models[model_name]

    if tune:
        # Get parameter space for the model
        param_spaces = {
            "random_forest": get_random_forest_param_space,
            # Add more parameter spaces here
        }
        param_space = param_spaces[model_name]()

        # Tune hyperparameters
        best_params = tune_hyperparameters(
            model_class, train_features, train_labels, val_features, val_labels, param_space, n_trials=n_trials
        )
        model = model_class(best_params)
    else:
        # Load config and initialize model
        configs = load_model_config(config_path)
        model = model_class(configs[model_name])

    # Train and evaluate
    model.train(train_features, train_labels)
    metrics = evaluate_model(model, val_features, val_labels)

    return metrics
