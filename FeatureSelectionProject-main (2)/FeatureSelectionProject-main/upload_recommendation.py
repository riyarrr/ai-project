"""Evaluate feature selectors on a user-provided CSV classification dataset."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cfs_classical import cfs_selection
from fcbf_selection import fcbf_selection
from fast_selection import fast_selection
from ga_selection import ga_selection
from hybrid_selection import hybrid_selection
from pso_selection import pso_selection
from evaluation import feature_reduction_percentage
from recommendation import recommend_methods


RANDOM_STATE = 42
N_NEIGHBORS = 5
INTERNAL_CV_FOLDS = 3
MAX_INPUT_FEATURES = 10_000
MAX_ENCODED_CELLS = 10_000_000
MAX_CFS_FEATURES = 8_000
METHOD_SELECTORS = {
    "CFS": cfs_selection,
    "FCBF": fcbf_selection,
    "FAST": fast_selection,
    "GA": ga_selection,
    "PSO": pso_selection,
    "Hybrid": hybrid_selection,
}


def load_uploaded_csv(csv_path, target_column):
    """Read and validate a CSV path selected by the user."""
    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Only .csv upload files are supported.")
    if not path.is_file():
        raise FileNotFoundError(f"Uploaded CSV does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        try:
            headers = next(csv.reader(source))
        except StopIteration as error:
            raise ValueError("Uploaded CSV is empty.") from error

    if not headers or any(not header.strip() for header in headers):
        raise ValueError("CSV column names must not be empty.")
    if len(headers) != len(set(headers)):
        raise ValueError("CSV column names must be unique.")

    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError("Uploaded CSV has a header but no data rows.")
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    return frame


def _validate_uploaded_frame(frame, target_column):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The uploaded dataset must contain data rows.")
    if not frame.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if frame.shape[1] < 2:
        raise ValueError("Dataset must contain a target and at least one feature.")
    if frame.shape[1] - 1 > MAX_INPUT_FEATURES:
        raise ValueError(
            f"Dataset has more than {MAX_INPUT_FEATURES} input features."
        )

    X = frame.drop(columns=[target_column]).copy()
    y = frame[target_column].copy()
    if y.isna().any():
        raise ValueError("Target values must not be missing.")
    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")
    if X.isna().all().any():
        empty_columns = X.columns[X.isna().all()].tolist()
        raise ValueError(
            f"Features containing only missing values cannot be imputed: "
            f"{empty_columns}"
        )

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        values = X[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(
                f"Numeric feature {column!r} contains an infinite value."
            )

    y = y.astype("string").astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if y.value_counts().min() < 3:
        raise ValueError(
            "Each target class needs at least 3 samples for stratified "
            "training and three-fold feature-search validation."
        )
    return X, y


def _build_preprocessor(X_train):
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in X_train.columns
        if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median", keep_empty_features=True),
            numeric_columns,
        ))
    if categorical_columns:
        categorical = X_train[categorical_columns].apply(
            lambda column: column.map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )
        )
        X_train = X_train.copy()
        X_train[categorical_columns] = categorical
        transformers.append((
            "categorical",
            _categorical_pipeline(),
            categorical_columns,
        ))

    return X_train, ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def _categorical_pipeline():
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True,
        ),
        FunctionTransformer(
            lambda values: values.astype(str),
            feature_names_out="one-to-one",
        ),
        OneHotEncoder(
            handle_unknown="ignore",
            max_categories=100,
            sparse_output=True,
        ),
    )


def _prepare_split(X, y):
    try:
        train_indices, test_indices = train_test_split(
            np.arange(len(X)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError as error:
        raise ValueError(
            "Dataset is too small or imbalanced for a stratified 80/20 "
            f"train/test split: {error}"
        ) from error

    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    class_counts = y_train.value_counts()
    if (
        len(train_indices) < 8
        or len(test_indices) < y.nunique()
        or class_counts.min() < INTERNAL_CV_FOLDS
    ):
        raise ValueError(
            "Not enough samples in the stratified split for a 5-NN "
            "classifier and three-fold feature-search validation."
        )
    smallest_inner_train = sum(
        count - math.ceil(count / INTERNAL_CV_FOLDS)
        for count in class_counts
    )
    if smallest_inner_train < N_NEIGHBORS:
        raise ValueError(
            "Each three-fold search training partition must contain at "
            "least 5 samples for the project's 5-NN classifier."
        )
    return train_indices, test_indices, y_train, y_test


def evaluate_uploaded_dataset(frame, target_column, dataset_name="Uploaded CSV"):
    """Run all supported selectors and recommend from fresh held-out results.

    Every method status is returned. Unsupported methods and execution errors
    are surfaced instead of being silently omitted from the recommendation.
    """
    X, y = _validate_uploaded_frame(frame, target_column)
    train_indices, test_indices, y_train, y_test = _prepare_split(X, y)
    X_train_raw = X.iloc[train_indices].reset_index(drop=True)
    X_test_raw = X.iloc[test_indices].reset_index(drop=True)

    X_train_for_preprocessor, preprocessor = _build_preprocessor(X_train_raw)
    if len(X_train_for_preprocessor) * X_train_for_preprocessor.shape[1] > (
        MAX_ENCODED_CELLS
    ):
        raise ValueError(
            "The training feature matrix exceeds the safety limit of "
            f"{MAX_ENCODED_CELLS:,} cells before encoding."
        )
    X_test_raw = X_test_raw.copy()
    categorical_columns = X_train_for_preprocessor.select_dtypes(
        exclude=[np.number]
    ).columns
    for column in categorical_columns:
        X_test_raw[column] = X_test_raw[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    train_array = preprocessor.fit_transform(X_train_for_preprocessor)
    test_array = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    if train_array.shape[1] == 0:
        raise ValueError("Preprocessing produced no usable features.")
    if train_array.shape[1] * len(train_indices) > MAX_ENCODED_CELLS:
        raise ValueError(
            "The one-hot encoded training matrix exceeds the safety limit "
            f"of {MAX_ENCODED_CELLS:,} cells."
        )
    if hasattr(train_array, "toarray"):
        train_array = train_array.toarray()
        test_array = test_array.toarray()

    X_train = pd.DataFrame(
        train_array, columns=feature_names, dtype=float
    )
    X_test = pd.DataFrame(test_array, columns=feature_names, dtype=float)
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(
        X_test.to_numpy()
    ).all():
        raise ValueError("Preprocessing produced non-finite feature values.")

    method_rows = []
    method_statuses = []
    for method, selector in METHOD_SELECTORS.items():
        if method == "CFS" and y.nunique() > 2:
            method_statuses.append({
                "Method": method,
                "Status": "Unsupported",
                "Details": (
                    "The current CFS implementation factorizes labels and "
                    "uses numeric correlation; it is only used here for "
                    "binary classification."
                ),
            })
            continue
        if method == "CFS" and X_train.shape[1] > MAX_CFS_FEATURES:
            method_statuses.append({
                "Method": method,
                "Status": "Unsupported",
                "Details": (
                    f"CFS computes a dense feature-correlation matrix and "
                    f"is limited here to {MAX_CFS_FEATURES:,} features."
                ),
            })
            continue

        started = time.perf_counter()
        try:
            selected = selector(X_train, y_train)
            if not selected:
                raise ValueError("Selector returned no features.")
            if len(selected) != len(set(selected)):
                raise ValueError("Selector returned duplicate features.")
            unknown = set(selected) - set(X_train.columns)
            if unknown:
                raise ValueError(
                    f"Selector returned unknown feature names: {unknown}"
                )

            selected_train = X_train[selected]
            selected_test = X_test[selected]
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(selected_train)
            scaled_test = scaler.transform(selected_test)
            classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
            classifier.fit(scaled_train, y_train)
            predictions = classifier.predict(scaled_test)
            runtime = time.perf_counter() - started
            method_rows.append({
                "Dataset": dataset_name,
                "Method": method,
                "Total Features": X_train.shape[1],
                "Selected Features": len(selected),
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "MCC": round(matthews_corrcoef(y_test, predictions), 4),
                "Feature Reduction (%)": feature_reduction_percentage(
                    X_train.shape[1], len(selected)
                ),
                "Runtime (s)": round(runtime, 4),
                "Selected Feature Names": json.dumps(
                    [str(name) for name in selected],
                    ensure_ascii=False,
                ),
            })
            method_statuses.append({
                "Method": method,
                "Status": "Completed",
                "Details": "",
            })
        except Exception as error:
            method_statuses.append({
                "Method": method,
                "Status": "Error",
                "Details": f"{type(error).__name__}: {error}",
            })

    results = pd.DataFrame(method_rows)
    statuses = pd.DataFrame(method_statuses)
    if results.empty:
        errors = "; ".join(
            f"{row['Method']}: {row['Details']}"
            for row in method_statuses
        )
        raise RuntimeError(f"No feature-selection method completed. {errors}")

    recommendations = recommend_methods(results)
    return {
        "results": results,
        "statuses": statuses,
        "recommendations": recommendations,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "target_classes": list(y.unique()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate feature selectors on an uploaded CSV dataset."
    )
    parser.add_argument("csv_file", type=Path, help="Path to the uploaded CSV")
    parser.add_argument(
        "--target",
        help="Target/class column name (prompted if omitted)",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name shown in the output (defaults to the CSV filename)",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        help="Optional path to save successful method metrics as CSV",
    )
    parser.add_argument(
        "--recommendation-output",
        type=Path,
        help="Optional path to save the recommendation as CSV",
    )
    args = parser.parse_args()

    target = args.target
    if target is None:
        frame = pd.read_csv(args.csv_file, nrows=0, encoding="utf-8-sig")
        print("CSV columns:", ", ".join(map(str, frame.columns)))
        target = input("Target/class column: ").strip()
    frame = load_uploaded_csv(args.csv_file, target)
    dataset_name = args.dataset_name or args.csv_file.stem
    outcome = evaluate_uploaded_dataset(frame, target, dataset_name)

    print(
        f"\nDataset: {dataset_name}; target: {target}; "
        f"classes: {outcome['target_classes']}; "
        f"training samples: {outcome['train_samples']}; "
        f"test samples: {outcome['test_samples']}"
    )
    print("\nMethod status (all methods reported):")
    print(outcome["statuses"].to_string(index=False))
    print("\nComputed evaluation results:")
    print(outcome["results"].to_string(index=False))
    print("\nRecommendation from this uploaded dataset's results:")
    print(outcome["recommendations"].to_string(index=False))
    for recommendation in outcome["recommendations"].to_dict("records"):
        selected_result = outcome["results"].loc[
            (outcome["results"]["Dataset"] == recommendation["Dataset"])
            & (
                outcome["results"]["Method"]
                == recommendation["Recommended Method"]
            )
        ].iloc[0]
        selected_names = json.loads(
            selected_result["Selected Feature Names"]
        )
        print(
            f"\nRecommended features for {recommendation['Dataset']} "
            f"({recommendation['Recommended Method']}): "
            f"{selected_names}"
        )

    if args.results_output:
        outcome["results"].to_csv(args.results_output, index=False)
        print(f"\nResults saved to {args.results_output}")
    if args.recommendation_output:
        outcome["recommendations"].to_csv(
            args.recommendation_output, index=False
        )
        print(f"Recommendation saved to {args.recommendation_output}")


if __name__ == "__main__":
    main()
