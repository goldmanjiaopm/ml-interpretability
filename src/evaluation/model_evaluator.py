from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.training_pipeline.models.base_model import BaseModel


class ModelEvaluator:
    """Class for comprehensive model evaluation."""

    def __init__(self, model: BaseModel, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            class_names: Optional list of class names for visualization
        """
        self.model = model
        self.class_names = class_names
        self.metrics: Dict = {}

    def evaluate(
        self, features: pd.DataFrame, labels: pd.Series, threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Test features
            labels: True labels
            threshold: Optional prediction threshold

        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "classification_report": classification_report(labels, predictions, output_dict=True),
        }

        # Add ROC AUC score
        try:
            self.metrics["roc_auc"] = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        except ValueError:
            self.metrics["roc_auc"] = None

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        self.metrics.update({"precision": precision, "recall": recall, "f1": f1})

        return self.metrics

    def plot_confusion_matrix(self, features: pd.DataFrame, true_labels: pd.Series, figsize: tuple = (10, 8)) -> None:
        """Plot confusion matrix."""
        predictions = self.model.predict(features)
        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    def plot_roc_curves(self, features: pd.DataFrame, true_labels: pd.Series, figsize: tuple = (10, 8)) -> None:
        """Plot ROC curves for each class."""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        probabilities = self.model.predict_proba(features)
        n_classes = probabilities.shape[1]

        # Binarize the labels
        y_bin = label_binarize(true_labels, classes=range(n_classes))

        plt.figure(figsize=figsize)
        print(self.class_names)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"{self.class_names[i] if self.class_names else i} " f"(AUC = {roc_auc:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.show()

    def print_metrics(self) -> None:
        """Print all computed metrics."""
        print("\nModel Evaluation Metrics:")
        for metric, value in self.metrics.items():
            if isinstance(value, dict):
                print(f"\n{metric}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        print(f"  {k}:")
                        for m, s in v.items():
                            print(f"    {m}: {s:.4f}")
                    else:
                        print(f"  {k}: {v:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
