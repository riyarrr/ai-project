"""Dataset validation module for user-uploaded CSV classification datasets.

Reuses and aligns with constraints defined in upload_recommendation.py:
- Max input features: 10,000
- Minimum classes: 2
- Minimum samples per class: 3 (required for stratified 80/20 train/test split and 3-fold CV)
- No missing values in target column
- Safe imputation for predictor features (no 100% missing columns)
- No infinite values in numeric features
"""

import csv
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MAX_INPUT_FEATURES = 10_000
MIN_CLASSES = 2
MIN_SAMPLES_PER_CLASS = 3


def validate_csv_content(
    file_bytes: bytes,
    filename: str = "dataset.csv",
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """Safely validate and parse raw CSV bytes.

    Returns:
        (dataframe, errors_list, warnings_list)
        If errors_list is non-empty, dataframe will be None.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Empty file check
    if not file_bytes or len(file_bytes.strip()) == 0:
        errors.append("The uploaded file is empty (0 bytes). Please upload a valid CSV file.")
        return None, errors, warnings

    if not filename.lower().endswith(".csv"):
        warnings.append(
            f"The file '{filename}' does not have a standard '.csv' extension. "
            "Parsing as comma-separated values."
        )

    # 2. Text decoding check
    text: Optional[str] = None
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        errors.append(
            "Failed to decode file text encoding. Please ensure the file is an encoded text CSV "
            "(UTF-8 or ASCII preferred)."
        )
        return None, errors, warnings

    # 3. CSV header structure check
    try:
        reader = csv.reader(io.StringIO(text))
        headers = next(reader)
    except StopIteration:
        errors.append("The uploaded CSV file is empty and contains no header row.")
        return None, errors, warnings
    except Exception as exc:
        errors.append(f"Invalid CSV structure: {exc}")
        return None, errors, warnings

    if not headers or len(headers) == 0:
        errors.append("The CSV header row contains no column names.")
        return None, errors, warnings

    # Check for empty or whitespace-only header names
    empty_header_indices = [
        idx + 1 for idx, h in enumerate(headers) if not str(h).strip()
    ]
    if empty_header_indices:
        errors.append(
            f"CSV contains empty or whitespace-only column header(s) at column position: "
            f"{', '.join(map(str, empty_header_indices))}."
        )

    # Check for duplicate column names
    header_counts: Dict[str, int] = {}
    duplicates: List[str] = []
    for h in headers:
        clean_name = str(h).strip()
        header_counts[clean_name] = header_counts.get(clean_name, 0) + 1
        if header_counts[clean_name] == 2:
            duplicates.append(clean_name)

    if duplicates:
        errors.append(
            f"CSV column names must be unique. Duplicate column name(s) detected: "
            f"{', '.join(repr(d) for d in duplicates)}."
        )

    # If header validation failed, return before reading full DataFrame
    if errors:
        return None, errors, warnings

    # 4. DataFrame parsing check
    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except pd.errors.EmptyDataError:
        errors.append("The uploaded CSV has a header row but contains no data rows.")
        return None, errors, warnings
    except pd.errors.ParserError as pe:
        errors.append(f"CSV parsing error: {pe}")
        return None, errors, warnings
    except Exception as exc:
        errors.append(f"Unable to parse dataset: {exc}")
        return None, errors, warnings

    if df.empty or len(df) == 0:
        errors.append("The uploaded CSV has a header row but contains 0 data rows.")
        return None, errors, warnings

    # 5. Dimension checks
    if df.shape[1] < 2:
        errors.append(
            f"Dataset must contain at least 2 columns (1 target class column and at least "
            f"1 feature column). Found only {df.shape[1]} column."
        )

    if df.shape[1] - 1 > MAX_INPUT_FEATURES:
        errors.append(
            f"Dataset contains {df.shape[1] - 1:,} input features, exceeding the safety limit "
            f"of {MAX_INPUT_FEATURES:,} features."
        )

    # 6. Check for columns containing only missing values
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        errors.append(
            f"Feature column(s) containing 100% missing values cannot be imputed: "
            f"{', '.join(repr(str(c)) for c in all_nan_cols)}."
        )

    # 7. Check for duplicate rows (warning only, dataset remains valid)
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        pct = (dup_rows / len(df)) * 100
        warnings.append(
            f"Dataset contains {dup_rows:,} duplicate sample rows ({pct:.1f}% of total). "
            "These will be preserved during evaluation."
        )

    if errors:
        return None, errors, warnings

    return df, errors, warnings


def validate_target_selection(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[bool, List[str], List[str], pd.Series]:
    """Validate that the selected column is a valid classification target.

    Returns:
        (is_valid, errors_list, warnings_list, class_distribution_series)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if target_column not in df.columns:
        errors.append(f"Target column {target_column!r} was not found in dataset columns.")
        return False, errors, warnings, pd.Series(dtype=int)

    y = df[target_column]

    # 1. Target missing values check
    missing_count = int(y.isna().sum())
    if missing_count > 0:
        pct = (missing_count / len(y)) * 100
        errors.append(
            f"Target column {target_column!r} contains {missing_count} missing value(s) "
            f"({pct:.1f}%). Supervised classification requires all samples to have ground-truth labels."
        )

    # 2. Class count check
    y_clean = y.dropna()
    class_counts = y_clean.value_counts()
    n_classes = len(class_counts)

    if n_classes < MIN_CLASSES:
        if n_classes == 1:
            only_class = class_counts.index[0]
            errors.append(
                f"Target column {target_column!r} contains only 1 class ({only_class!r}). "
                f"Classification requires at least {MIN_CLASSES} distinct classes."
            )
        else:
            errors.append(f"Target column {target_column!r} contains no valid class values.")
        return False, errors, warnings, class_counts

    # 3. Minimum samples per class check
    min_samples = int(class_counts.min())
    if min_samples < MIN_SAMPLES_PER_CLASS:
        problem_classes = [
            f"{cls!r} ({cnt} samples)"
            for cls, cnt in class_counts[class_counts < MIN_SAMPLES_PER_CLASS].items()
        ]
        errors.append(
            f"Each target class requires at least {MIN_SAMPLES_PER_CLASS} samples for stratified "
            f"80/20 train/test splitting and 3-fold cross-validation. The following class(es) have "
            f"fewer than {MIN_SAMPLES_PER_CLASS} samples: {', '.join(problem_classes)}."
        )

    # 4. Predictor feature checks
    features_df = df.drop(columns=[target_column])
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns:
        values = features_df[col].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            errors.append(
                f"Numeric feature {col!r} contains infinite values (+inf or -inf)."
            )

    # 5. Class balance warnings
    if n_classes >= MIN_CLASSES and min_samples >= MIN_SAMPLES_PER_CLASS:
        majority = class_counts.iloc[0]
        minority = class_counts.iloc[-1]
        imbalance_ratio = majority / minority
        if imbalance_ratio > 4.0:
            warnings.append(
                f"Class imbalance detected: majority class ({class_counts.index[0]!r}: {majority}) "
                f"has {imbalance_ratio:.1f}x more samples than minority class "
                f"({class_counts.index[-1]!r}: {minority}). Matthews Correlation Coefficient (MCC) "
                "will provide a reliable metric for this imbalanced distribution."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, class_counts


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, any]:
    """Calculate comprehensive dataset inspection metrics."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in datetime_cols and c not in bool_cols
    ]

    return {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_pct, 2),
        "duplicate_rows": duplicate_rows,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "datetime_count": len(datetime_cols),
        "bool_count": len(bool_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
