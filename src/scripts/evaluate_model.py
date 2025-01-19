import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_pipeline.data_preprocessing import get_label_columns
from src.evaluation.model_evaluator import ModelEvaluator
from src.training_pipeline.models.random_forest import RandomForestModel
from src.training_pipeline.models.xgboost_model import XGBoostModel
from src.training_pipeline.train import load_model_config


def main():
    """Run model evaluation on validation data."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
        help="Type of model to evaluate",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file")
    args = parser.parse_args()

    # Load validation data
    val_features = pd.read_csv("data/processed/val_features.csv")
    val_labels = pd.read_csv("data/processed/val_labels.csv")["Labels"]

    # Initialize model
    model_classes = {"random_forest": RandomForestModel, "xgboost": XGBoostModel}
    model_class = model_classes[args.model_type]
    model = model_class({})  # Empty config for loading
    model.load(args.model_path)

    # Get class names from data preprocessing
    class_names = ["Barely-True", "False", "Half-True", "Mostly-True", "Not Known", "True"]

    # Initialize evaluator with correct class names
    evaluator = ModelEvaluator(model=model, class_names=class_names)

    # Run evaluation
    print(f"\nEvaluating {args.model_type} model...")
    print(evaluator.class_names)
    evaluator.evaluate(val_features, val_labels)

    # Plot visualizations
    print("\nGenerating plots...")
    evaluator.plot_confusion_matrix(val_features, val_labels)
    evaluator.plot_roc_curves(val_features, val_labels)

    # Print metrics
    evaluator.print_metrics()


if __name__ == "__main__":
    main()
