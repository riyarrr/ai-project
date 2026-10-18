
import time
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef


def feature_reduction_percentage(total_features, selected_features):
    """Return the percentage reduction after validating feature counts."""
    try:
        total = float(total_features)
        selected = float(selected_features)
    except (TypeError, ValueError) as error:
        raise ValueError("Feature counts must be numeric integers.") from error

    if (
        not math.isfinite(total)
        or not math.isfinite(selected)
        or not total.is_integer()
        or not selected.is_integer()
        or total <= 0
        or not 1 <= selected <= total
    ):
        raise ValueError(
            "Feature counts must be integers with "
            "1 <= selected_features <= total_features."
        )

    return round((total - selected) / total * 100, 4)


def evaluate_features(
    X, y, dataset_name, method_name, selected_features
):
    """Evaluate a feature subset using a consistent KNN setup."""

    if len(selected_features) == 0:
        raise ValueError("At least one feature must be selected.")

    missing = set(selected_features) - set(X.columns)
    if missing:
        raise ValueError(f"Unknown features: {missing}")

    X_selected = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - start

    return {
        "Dataset": dataset_name,
        "Method": method_name,
        "Total Features": X.shape[1],
        "Selected Features": len(selected_features),
        "Feature Reduction (%)": feature_reduction_percentage(
            X.shape[1], len(selected_features)
        ),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        "Runtime (s)": round(elapsed, 4)
    }
