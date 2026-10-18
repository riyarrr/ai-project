
import math
from pathlib import Path

import pandas as pd

from evaluation import evaluate_features, feature_reduction_percentage
from main import load_datasets


RESULT_FILES = {
    "CFS": "cfs_classical_results.csv",
    "FCBF": "fcbf_results.csv",
    "FAST": "fast_results.csv",
    "GA": "ga_results.csv",
    "PSO": "pso_results.csv",
    "Hybrid": "hybrid_results.csv"
}

RESULT_COLUMNS = (
    "Dataset",
    "Method",
    "Total Features",
    "Selected Features",
    "Accuracy",
    "MCC",
    "Feature Reduction (%)",
    "Runtime (s)",
)


def _parse_selected_features(value, dataset_name, method):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{dataset_name}/{method} has no selected feature names."
        )

    names = [name.strip() for name in value.split(",")]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(
            f"{dataset_name}/{method} has empty or duplicate feature names."
        )
    return names


def build_comparison(datasets=None, result_dir=None):
    """Load, validate, and recompute metrics for all recorded selections."""
    if datasets is None:
        datasets = load_datasets()
    if result_dir is None:
        result_dir = Path(__file__).parent
    result_dir = Path(result_dir)

    if not datasets:
        raise ValueError("No datasets were provided for comparison.")
    for dataset_name, (X, y) in datasets.items():
        if X.shape[0] != len(y) or X.shape[1] == 0:
            raise ValueError(
                f"{dataset_name} must have matching samples and at least "
                "one feature."
            )
        if X.isna().any().any() or pd.Series(y).isna().any():
            raise ValueError(f"{dataset_name} contains missing values.")
        samples = pd.concat(
            [
                X.reset_index(drop=True),
                pd.Series(y).reset_index(drop=True).rename("_label"),
            ],
            axis=1,
        )
        if samples.duplicated().any():
            raise ValueError(
                f"{dataset_name} contains duplicate sample rows."
            )

    all_results = []
    for method, filename in RESULT_FILES.items():
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required result file not found: {path}")

        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Result file is empty: {path}")

        required = {
            "Dataset",
            "Total Features",
            "Selected Features",
            "Accuracy",
            "MCC",
            "Runtime (s)",
            "Selected Feature Names",
        }
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        if "Method" in frame.columns:
            recorded_methods = set(frame["Method"].dropna().astype(str))
            if recorded_methods != {method} or frame["Method"].isna().any():
                raise ValueError(
                    f"{path.name} contains inconsistent method names: "
                    f"{sorted(recorded_methods)}"
                )
        else:
            frame["Method"] = method
        all_results.append(frame)

    comparison = pd.concat(all_results, ignore_index=True)
    if comparison[list(required)].isna().any().any():
        missing = comparison[list(required)].columns[
            comparison[list(required)].isna().any()
        ].tolist()
        raise ValueError(f"Result files contain missing values in: {missing}")

    valid_pairs = {
        (dataset_name, method)
        for dataset_name in datasets
        for method in RESULT_FILES
    }
    pair_counts = comparison.groupby(["Dataset", "Method"]).size()
    invalid_pairs = [
        pair for pair in pair_counts.index if pair not in valid_pairs
    ]
    duplicate_pairs = pair_counts[pair_counts != 1].index.tolist()
    missing_pairs = sorted(valid_pairs - set(pair_counts.index))
    if invalid_pairs or duplicate_pairs or missing_pairs:
        raise ValueError(
            "Expected exactly one result for each dataset/method pair. "
            f"Unknown={invalid_pairs}, duplicate={duplicate_pairs}, "
            f"missing={missing_pairs}"
        )

    validated_rows = []
    for _, row in comparison.iterrows():
        dataset_name = row["Dataset"]
        method = row["Method"]
        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        X, y = datasets[dataset_name]
        names = _parse_selected_features(
            row["Selected Feature Names"], dataset_name, method
        )
        columns_by_name = {}
        for column in X.columns:
            column_name = str(column)
            if column_name in columns_by_name:
                raise ValueError(
                    f"{dataset_name} has ambiguous feature name "
                    f"{column_name!r}."
                )
            columns_by_name[column_name] = column

        unknown_names = sorted(set(names) - set(columns_by_name))
        if unknown_names:
            raise ValueError(
                f"{dataset_name}/{method} selects unknown features: "
                f"{unknown_names}"
            )
        selected_features = [columns_by_name[name] for name in names]

        try:
            total_features = float(row["Total Features"])
            selected_count = float(row["Selected Features"])
            accuracy = float(row["Accuracy"])
            mcc = float(row["MCC"])
            runtime = float(row["Runtime (s)"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{dataset_name}/{method} contains a non-numeric metric."
            ) from error

        if (
            not math.isfinite(total_features)
            or total_features != X.shape[1]
        ):
            raise ValueError(
                f"{dataset_name}/{method} reports {row['Total Features']} "
                f"total features; the dataset has {X.shape[1]}."
            )
        if (
            not math.isfinite(selected_count)
            or not selected_count.is_integer()
            or int(selected_count) != len(selected_features)
            or not 1 <= selected_count <= total_features
        ):
            raise ValueError(
                f"{dataset_name}/{method} has an invalid selected-feature "
                f"count ({row['Selected Features']}); feature names list "
                f"contains {len(selected_features)}."
            )
        if (
            not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
            or not math.isfinite(mcc)
            or not -1 <= mcc <= 1
            or not math.isfinite(runtime)
            or runtime < 0
        ):
            raise ValueError(
                f"{dataset_name}/{method} contains an invalid metric value."
            )

        evaluated = evaluate_features(
            X,
            y,
            dataset_name,
            method,
            selected_features,
        )
        if not math.isclose(
            accuracy, evaluated["Accuracy"], rel_tol=0, abs_tol=0.00005
        ):
            raise ValueError(
                f"{dataset_name}/{method} stored accuracy {accuracy} does "
                f"not match re-evaluated accuracy {evaluated['Accuracy']}."
            )
        if not math.isclose(
            mcc, evaluated["MCC"], rel_tol=0, abs_tol=0.00005
        ):
            raise ValueError(
                f"{dataset_name}/{method} stored MCC {mcc} does not match "
                f"re-evaluated MCC {evaluated['MCC']}."
            )

        validated_rows.append({
            **row.to_dict(),
            "Total Features": int(total_features),
            "Selected Features": int(selected_count),
            "Accuracy": evaluated["Accuracy"],
            "MCC": evaluated["MCC"],
            "Feature Reduction (%)": feature_reduction_percentage(
                int(total_features), int(selected_count)
            ),
            "Runtime (s)": runtime,
        })

    comparison = pd.DataFrame(validated_rows)
    method_order = {method: index for index, method in enumerate(RESULT_FILES)}
    comparison["_method_order"] = comparison["Method"].map(method_order)
    comparison["_dataset_order"] = comparison["Dataset"].map(
        {name: index for index, name in enumerate(datasets)}
    )
    comparison = comparison.sort_values(
        ["_dataset_order", "_method_order"]
    ).drop(columns=["_method_order", "_dataset_order"])

    leading_columns = [
        "Dataset",
        "Method",
        "Total Features",
        "Selected Features",
        "Accuracy",
        "MCC",
        "Feature Reduction (%)",
        "Runtime (s)"
    ]
    remaining_columns = [
        column for column in comparison.columns
        if column not in leading_columns
    ]
    return comparison[leading_columns + remaining_columns].reset_index(
        drop=True
    )


def main():
    comparison = build_comparison()
    print("\nFEATURE SELECTION COMPARISON")
    print(comparison.to_string(index=False))

    output_path = Path(__file__).parent / "final_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nValidated comparison saved to {output_path}")


if __name__ == "__main__":
    main()
