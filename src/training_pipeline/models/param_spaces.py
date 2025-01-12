from typing import Any, Callable, Dict, Type

import optuna

from src.training_pipeline.models.base_model import BaseModel
from src.training_pipeline.models.random_forest import RandomForestModel
from src.training_pipeline.models.xgboost_model import XGBoostModel


def get_random_forest_param_space() -> Dict[str, Callable[[optuna.Trial], Any]]:
    """Define parameter space for Random Forest model."""
    return {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": lambda trial: 42,
        "n_jobs": lambda trial: -1,
        "max_features": lambda trial: trial.suggest_float("max_features", 0.1, 1.0),
    }


def get_xgboost_param_space() -> Dict[str, Callable[[optuna.Trial], Any]]:
    """Define parameter space for XGBoost model."""
    return {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 10, 100),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 10),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "min_child_weight": lambda trial: trial.suggest_int("min_child_weight", 0.2, 7),
        "subsample": lambda trial: trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": lambda trial: trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": lambda trial: 42,
        "n_jobs": lambda trial: -1,
    }


def get_model_class(model_name: str) -> Type[BaseModel]:
    """Get model class for a given model name."""
    models = {"random_forest": RandomForestModel, "xgboost": XGBoostModel}
    return models[model_name]


def get_param_space(model_name: str) -> Dict[str, Callable[[optuna.Trial], Any]]:
    """Get parameter space for a given model."""
    param_spaces = {
        "random_forest": get_random_forest_param_space,
        "xgboost": get_xgboost_param_space,
    }
    return param_spaces[model_name]
