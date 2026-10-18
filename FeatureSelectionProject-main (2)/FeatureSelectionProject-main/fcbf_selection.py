
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mutual_info_score,
    accuracy_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from main import load_datasets


def discretize(values, bins=10):
    """Convert numeric or categorical values into discrete categories."""
    values = pd.Series(values).reset_index(drop=True)

    if values.nunique(dropna=True) <= 1:
        return values.fillna("missing").astype(str)

    if pd.api.types.is_numeric_dtype(values):
        # Rank first so discretization also works with skewed distributions.
        ranked = values.rank(method="average")
        n_bins = min(bins, int(values.nunique(dropna=True)))

        return pd.cut(
            ranked,
            bins=n_bins,
            labels=False,
            duplicates="drop"
        ).fillna(0).astype(int)

    return values.fillna("missing").astype(str)


def symmetrical_uncertainty(x, y, bins=10):
    """
    Calculate Symmetrical Uncertainty (SU).

    SU(X,Y) = 2 * MI(X,Y) / (H(X) + H(Y))
    """
    x_discrete = discretize(x, bins)
    y_discrete = discretize(y, bins)

    h_x = mutual_info_score(x_discrete, x_discrete)
    h_y = mutual_info_score(y_discrete, y_discrete)
    mi = mutual_info_score(x_discrete, y_discrete)

    denominator = h_x + h_y

    if denominator == 0:
        return 0.0

    return float(2.0 * mi / denominator)


def fcbf_selection(X, y, delta=0.0, max_features=None):
    """
    FCBF-style feature selection using SU relevance and redundancy pruning.

    Parameters
    ----------
    X : pandas.DataFrame
        Training features.
    y : array-like
        Training labels.
    delta : float, default=0.0
        Minimum SU relevance threshold.
    max_features : int or None, default=None
        Optional maximum number of selected features.

    Returns
    -------
    list
        Selected feature names.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be between 0 and 1.")

    if max_features is not None:
        if not isinstance(max_features, int) or max_features < 1:
            raise ValueError("max_features must be a positive integer or None.")

    # Calculate each feature's relevance to the target.
    relevance = {
        feature: symmetrical_uncertainty(X[feature], y)
        for feature in X.columns
    }

    # Rank features from highest to lowest relevance.
    candidates = [
        feature
        for feature in sorted(
            relevance,
            key=relevance.get,
            reverse=True
        )
        if relevance[feature] > delta
    ]

    selected = []

    # Greedy redundancy pruning.
    while candidates:
        current = candidates.pop(0)
        selected.append(current)

        if max_features is not None and len(selected) >= max_features:
            break

        remaining = []

        for candidate in candidates:
            redundancy = symmetrical_uncertainty(
                X[candidate],
                X[current]
            )

            # Remove candidate if it is at least as strongly related
            # to the selected feature as it is to the target.
            if redundancy < relevance[candidate]:
                remaining.append(candidate)

        candidates = remaining

    return selected


def evaluate_fcbf(
    X,
    y,
    dataset_name,
    delta=0.0,
    max_features=None
):
    """Select features on training data and evaluate on held-out data."""
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if X.isnull().any().any():
        raise ValueError(
            f"{dataset_name} contains missing feature values. "
            "Impute or clean them before running FCBF."
        )

    if y.isnull().any():
        raise ValueError(
            f"{dataset_name} contains missing target labels."
        )

    if y.nunique() < 2:
        raise ValueError(
            f"{dataset_name} must contain at least two target classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    start = time.perf_counter()

    # Select features using only the training split.
    selected = fcbf_selection(
        X_train,
        y_train,
        delta=delta,
        max_features=max_features
    )

    if not selected:
        raise ValueError(
            f"FCBF selected no features for {dataset_name}. "
            "Consider checking the data or lowering delta."
        )

    scaler = StandardScaler()

    X_train_selected = scaler.fit_transform(X_train[selected])
    X_test_selected = scaler.transform(X_test[selected])

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_test_selected)

    runtime = time.perf_counter() - start

    return {
        "Dataset": dataset_name,
        "Method": "FCBF",
        "Total Features": X.shape[1],
        "Selected Features": len(selected),
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "MCC": round(matthews_corrcoef(y_test, predictions), 4),
        "Runtime (s)": round(runtime, 4),
        "Selected Feature Names": ", ".join(map(str, selected))
    }


def main():
    datasets = load_datasets()
    results = []

    # Set max_features=5 to retain at most five features for comparison
    # with the existing experiments. Use None for no fixed limit.
    max_features = 5
    delta = 0.0

    for name, (X, y) in datasets.items():
        print(f"\nDataset: {name}")

        result = evaluate_fcbf(
            X,
            y,
            dataset_name=name,
            delta=delta,
            max_features=max_features
        )

        results.append(result)

        print("Selected features:", result["Selected Feature Names"])
        print("Number of selected features:", result["Selected Features"])
        print("Accuracy:", result["Accuracy"])
        print("MCC:", result["MCC"])
        print("Runtime (s):", result["Runtime (s)"])

    results_df = pd.DataFrame(results)

    print("\nFCBF Evaluation Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv("fcbf_results.csv", index=False)
    print("\nResults saved to fcbf_results.csv")


if __name__ == "__main__":
    main()
