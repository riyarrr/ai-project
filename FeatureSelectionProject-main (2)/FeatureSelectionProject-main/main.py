
import pandas as pd
from pathlib import Path


DATASET_DIR = Path(__file__).parent / "datasets"


def load_datasets():
    # Dataset 1: Colon Cancer
    colon = pd.read_csv(DATASET_DIR / "colon-cancer.csv", header=None)
    X_colon = colon.iloc[:, 1:].copy()
    y_colon = colon.iloc[:, 0].copy()

    # Dataset 2: Leukemia
    leukemia = pd.read_csv(DATASET_DIR / "leukemia_prepared.csv")
    X_leukemia = leukemia.drop(columns=["label"])
    y_leukemia = leukemia["label"]

    return {
        "Colon Cancer": (X_colon, y_colon),
        "Leukemia": (X_leukemia, y_leukemia)
    }


if __name__ == "__main__":
    datasets = load_datasets()

    for name, (X, y) in datasets.items():
        print(f"\nDataset: {name}")
        print("Number of records:", X.shape[0])
        print("Number of features:", X.shape[1])
        print("Number of classes:", y.nunique())
        print("Missing values:", X.isnull().sum().sum())
        print("Class distribution:")
        print(y.value_counts().sort_index())
