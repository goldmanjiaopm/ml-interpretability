from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(data_path: Path = Path("data/raw")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data from raw data directory.

    Args:
        data_path: Path to raw data directory

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    return train_df, test_df


def get_label_columns() -> List[str]:
    """Get the list of label columns as defined in the data."""
    return ["Labels"]


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text data.

    Args:
        text: Raw text input

    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Basic cleaning (can be expanded based on needs)
    text = text.replace("\n", " ").replace("\r", " ")

    return text


def create_tfidf_features(texts: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """
    Create TF-IDF features from text data.

    Args:
        texts: List of preprocessed text strings
        max_features: Maximum number of features to create

    Returns:
        Tuple of (fitted TfidfVectorizer, DataFrame of TF-IDF features)
    """
    # customise stop words and ngram range (include bigrams)
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))

    features = tfidf.fit_transform(texts)
    feature_names = [f"tfidf_{f}" for f in tfidf.get_feature_names_out()]

    return tfidf, pd.DataFrame(features.toarray(), columns=feature_names)
