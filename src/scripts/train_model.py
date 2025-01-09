import argparse
from datetime import datetime
from pathlib import Path

from src.training_pipeline.train import train_and_evaluate
from src.training_pipeline.utils import get_device_info


def main():
    """Run model training with command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument("--model", type=str, default="random_forest", help="Model type to train")
    parser.add_argument("--tune", action="store_true", help="Whether to perform hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=100, help="Number of hyperparameter tuning trials")

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "src/training_pipeline/configs/model_configs.yaml"

    # Print device information
    device_info = get_device_info()
    print("\nCompute Device Information:")
    print(f"Device: {device_info['device']}")
    print(f"Type: {device_info['type']}")
    print(f"Backend: {device_info['backend']}")

    print(f"\nStarting training for {args.model}")
    print(f"{'With' if args.tune else 'Without'} hyperparameter tuning")
    if args.tune:
        print(f"Number of trials: {args.trials}")

    # Train and evaluate
    metrics = train_and_evaluate(model_name=args.model, config_path=config_path, tune=args.tune, n_trials=args.trials)

    # Print results
    print("\nTraining Results:")
    print(f"Model saved at: {metrics.pop('model_path')}")
    print(f"Optimal thresholds: {metrics.pop('optimal_thresholds')}")
    print(f"Accuracy: {metrics.pop('accuracy')}")
    print("\nMetrics:")
    for label, scores in metrics.items():
        if isinstance(scores, dict):
            print(f"\n{label}:")
            for metric, value in scores.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"{label}: {scores:.4f}")


if __name__ == "__main__":
    main()
