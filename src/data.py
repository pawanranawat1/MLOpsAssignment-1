"""Download, clean, and load the UCI Heart Disease dataset."""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.config import RAW_DATA_PATH

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset(output_path: Path = RAW_DATA_PATH) -> Path:
    """Download the Cleveland part of the UCI dataset and save it as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(UCI_URL, timeout=30) as response:
        content = response.read()
    output_path.write_bytes(content)
    return output_path


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw file and perform simple cleaning."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.data"
        )

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")

    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # UCI target is 0 = no disease and 1-4 = disease severity.
    # Convert it to the required binary classification target.
    df["target"] = (df["target"] > 0).astype(int)
    return df


if __name__ == "__main__":
    path = download_dataset()
    df = load_dataset(path)
    print(f"Dataset saved to: {path}")
    print(f"Rows: {len(df)}")
    print(df.head())
    print("Missing values:")
    print(df.isna().sum())
