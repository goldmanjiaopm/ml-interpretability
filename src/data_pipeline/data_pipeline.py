from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_pipeline.data_preprocessing import (
    create_tfidf_features,
    get_label_columns,
    load_data,
    preprocess_text,
)


def process_data(data_path: Path = Path("data/raw"), output_path: Path = Path("data/processed")) -> None:
    """
    Process raw data and save processed features and labels.

    Args:
        data_path: Path to raw data directory
        output_path: Path to save processed data
    """
    # Load data
    train_df, _ = load_data(data_path)  # Only load train.csv

    # Print column names to debug
    print("Available columns in train_df:", train_df.columns.tolist())

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Split train_df into training and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["Labels"])

    # Preprocess text
    train_df["cleaned_text"] = train_df["Text"].apply(preprocess_text)
    val_df["cleaned_text"] = val_df["Text"].apply(preprocess_text)
    # Create TF-IDF features
    tfidf, train_tfidf_features = create_tfidf_features(train_df["cleaned_text"])
    # Use the same vectorizer for validation set
    _, val_tfidf_features = create_tfidf_features(
        val_df["cleaned_text"], vectorizer=tfidf  # Pass the fitted vectorizer
    )

    # Handle comma-separated Text_Tag
    train_df["Text_Tag"] = train_df["Text_Tag"].fillna("")
    val_df["Text_Tag"] = val_df["Text_Tag"].fillna("")

    # Split tags and one-hot encode
    train_tags = train_df["Text_Tag"].str.get_dummies(",")
    val_tags = val_df["Text_Tag"].str.get_dummies(",")

    # Align validation tag features with train tag features
    val_tags = val_tags.reindex(columns=train_tags.columns, fill_value=0)

    print(len(train_tfidf_features))
    print(len(train_tags))

    # Debug information before concatenation
    print("\nShape information:")
    print(f"Train TF-IDF features shape: {train_tfidf_features.shape}")
    print(f"Train tags shape: {train_tags.shape}")
    print(f"Val TF-IDF features shape: {val_tfidf_features.shape}")
    print(f"Val tags shape: {val_tags.shape}")

    # Check if indices match
    print("\nIndex information:")
    print(f"Train TF-IDF index length: {len(train_tfidf_features.index)}")
    print(f"Train tags index length: {len(train_tags.index)}")

    # Reset indices before concatenation to ensure alignment
    train_tfidf_features = train_tfidf_features.reset_index(drop=True)
    train_tags = train_tags.reset_index(drop=True)
    val_tfidf_features = val_tfidf_features.reset_index(drop=True)
    val_tags = val_tags.reset_index(drop=True)

    # Combine TF-IDF and tag features
    train_features = pd.concat([train_tfidf_features, train_tags], axis=1)
    val_features = pd.concat([val_tfidf_features, val_tags], axis=1)

    print("\nFinal shapes:")
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")

    # Save processed features and labels
    train_features.to_csv(output_path / "train_features.csv", index=False)
    val_features.to_csv(output_path / "val_features.csv", index=False)
    train_df[get_label_columns()].to_csv(output_path / "train_labels.csv", index=False)
    val_df[get_label_columns()].to_csv(output_path / "val_labels.csv", index=False)


if __name__ == "__main__":
    process_data()
