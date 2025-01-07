from pathlib import Path
from typing import Dict, Any, Callable
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.training_pipeline.models.base_model import BaseModel


def create_objective(
    model_class: type[BaseModel],
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
    param_space: Dict[str, Callable[[optuna.Trial], Any]],
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna optimization.

    Args:
        model_class: Class of the model to tune
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        param_space: Dictionary mapping parameter names to sampling functions

    Returns:
        Objective function for optimization
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters from defined space
        config = {name: space(trial) for name, space in param_space.items()}

        # Initialize and train model
        model = model_class(config)
        model.train(train_features, train_labels)

        # Get predictions and calculate score
        predictions = model.predict(val_features)
        score = accuracy_score(val_labels, predictions)

        return score

    return objective


def tune_hyperparameters(
    model_class: type[BaseModel],
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
    param_space: Dict[str, Callable[[optuna.Trial], Any]],
    n_trials: int = 100,
    study_name: str = "model_tuning",
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna.

    Args:
        model_class: Class of the model to tune
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        param_space: Dictionary mapping parameter names to sampling functions
        n_trials: Number of optimization trials
        study_name: Name of the study

    Returns:
        Best hyperparameters found
    """
    study = optuna.create_study(direction="maximize", study_name=study_name)

    objective = create_objective(model_class, train_features, train_labels, val_features, val_labels, param_space)

    # Add progress bar
    with tqdm(total=n_trials, desc="Tuning") as pbar:

        def callback(study: optuna.Study, trial: optuna.Trial) -> None:
            pbar.update(1)
            pbar.set_postfix({"best_score": study.best_value})

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    return study.best_params
