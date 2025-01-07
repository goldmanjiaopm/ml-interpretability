from typing import Dict, Callable, Any
import optuna


def get_random_forest_param_space() -> Dict[str, Callable[[optuna.Trial], Any]]:
    """Define parameter space for Random Forest model."""
    return {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": lambda trial: 42,
        "n_jobs": lambda trial: -1,
    }
