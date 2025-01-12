from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, roc_curve

from src.training_pipeline.models.base_model import BaseModel
from src.training_pipeline.models.param_spaces import get_model_class, get_param_space
from src.training_pipeline.models.random_forest import RandomForestModel
from src.training_pipeline.models.xgboost_model import XGBoostModel
from src.training_pipeline.tuning import tune_hyperparameters


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
        "xgboost": XGBoostModel,
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


def find_optimal_threshold(model: BaseModel, val_features: pd.DataFrame, val_labels: pd.Series) -> float:
    """
    Find the optimal threshold that maximizes the ROC curve's distance from the diagonal.

    Args:
        model: Trained model
        val_features: Validation features
        val_labels: Validation labels

    Returns:
        Optimal threshold value
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(val_features)

    # For each class, find optimal threshold
    n_classes = y_pred_proba.shape[1]
    thresholds = []

    for i in range(n_classes):
        # Calculate ROC curve
        fpr, tpr, threshold = roc_curve((val_labels == i).astype(int), y_pred_proba[:, i])

        # Find optimal threshold (point closest to top-left corner)
        optimal_idx = np.argmax(tpr - fpr)
        thresholds.append(threshold[optimal_idx])

    return np.array(thresholds)


def train_and_evaluate(model_name: str, config_path: Path, tune: bool = False, n_trials: int = 100) -> Dict[str, Any]:
    """Train and evaluate a model."""
    # Load data
    train_features, train_labels, val_features, val_labels = load_processed_data()

    # Get model class
    model_class = get_model_class(model_name)

    if tune:
        # Get parameter space for the model
        param_space = get_param_space(model_name)()
        # Tune hyperparameters
        best_params = tune_hyperparameters(
            model_class, train_features, train_labels, val_features, val_labels, param_space, n_trials=n_trials
        )
        model = model_class(best_params)
    else:
        # Load config and initialize model
        configs = load_model_config(config_path)
        model = model_class(configs[model_name])

    # Train model
    model.train(train_features, train_labels)

    # Find and set optimal thresholds
    optimal_thresholds = find_optimal_threshold(model, val_features, val_labels)
    model.set_thresholds(optimal_thresholds)

    # Evaluate with optimal thresholds
    metrics = evaluate_model(model, val_features, val_labels)

    # Add threshold information to metrics
    metrics["optimal_thresholds"] = optimal_thresholds.tolist()

    # Save model
    models_dir = Path(f"models/{model_name}")
    models_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{model_name}_{metrics['accuracy']:.4f}_{timestamp}.pkl"
    model.save(str(model_path))

    metrics["model_path"] = str(model_path)
    return metrics
