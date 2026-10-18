"""Transparent, weight-free recommendations from validated results."""

import math
from pathlib import Path

import pandas as pd

from comparison import build_comparison


REQUIRED_COLUMNS = {
    "Dataset",
    "Method",
    "Total Features",
    "Selected Features",
    "Accuracy",
    "MCC",
    "Runtime (s)",
}


def _validated_results(results):
    missing_columns = REQUIRED_COLUMNS - set(results.columns)
    if missing_columns:
        raise ValueError(
            f"Results are missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if results.empty:
        raise ValueError("No experimental results were provided.")
    if results[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Results contain missing recommendation metrics.")

    rows = results.copy()
    for column in (
        "Total Features",
        "Selected Features",
        "Accuracy",
        "MCC",
        "Runtime (s)",
    ):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if rows[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Results contain non-numeric recommendation metrics.")

    if rows.duplicated(["Dataset", "Method"]).any():
        raise ValueError("Results contain duplicate dataset/method rows.")
    if (
        rows["Dataset"].astype(str).str.strip().eq("").any()
        or rows["Method"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Dataset and method names must not be empty.")

    for _, row in rows.iterrows():
        total = row["Total Features"]
        selected = row["Selected Features"]
        accuracy = row["Accuracy"]
        mcc = row["MCC"]
        runtime = row["Runtime (s)"]
        if (
            not all(math.isfinite(value) for value in (
                total, selected, accuracy, mcc, runtime
            ))
            or not float(total).is_integer()
            or not float(selected).is_integer()
            or total <= 0
            or not 1 <= selected <= total
            or not 0 <= accuracy <= 1
            or not -1 <= mcc <= 1
            or runtime < 0
        ):
            raise ValueError(
                f"Invalid result values for {row['Dataset']}/"
                f"{row['Method']}."
            )

    rows["Feature Reduction (%)"] = (
        (rows["Total Features"] - rows["Selected Features"])
        / rows["Total Features"] * 100
    ).round(4)
    return rows


def _is_dominated(candidate, other):
    no_worse = (
        other["Accuracy"] >= candidate["Accuracy"]
        and other["MCC"] >= candidate["MCC"]
    )
    strictly_better = (
        other["Accuracy"] > candidate["Accuracy"]
        or other["MCC"] > candidate["MCC"]
    )
    return no_worse and strictly_better


def recommend_methods(results):
    """Recommend one method per dataset without arbitrary metric weights.

    First retain the Accuracy/MCC Pareto frontier. Among those methods,
    maximize feature reduction (equivalent to minimizing selected features);
    use runtime as the next tie-breaker, then method name for determinism.
    """
    rows = _validated_results(results)
    recommendations = []

    for dataset_name, group in rows.groupby("Dataset", sort=False):
        records = group.to_dict("records")
        frontier = [
            candidate for candidate in records
            if not any(
                _is_dominated(candidate, other)
                for other in records
                if other is not candidate
            )
        ]
        frontier = sorted(
            frontier,
            key=lambda row: (row["Method"].casefold(), row["Method"]),
        )

        minimum_features = min(row["Selected Features"] for row in frontier)
        compact_candidates = [
            row for row in frontier
            if row["Selected Features"] == minimum_features
        ]
        best_runtime = min(row["Runtime (s)"] for row in compact_candidates)
        tied_candidates = [
            row for row in compact_candidates
            if row["Runtime (s)"] == best_runtime
        ]
        chosen = min(
            tied_candidates,
            key=lambda row: (row["Method"].casefold(), row["Method"]),
        )

        frontier_methods = ", ".join(row["Method"] for row in frontier)
        reason = (
            f"Accuracy/MCC Pareto methods: {frontier_methods}. "
            f"Among them, {chosen['Method']} retains the fewest features "
            f"({int(chosen['Selected Features'])}/"
            f"{int(chosen['Total Features'])}; "
            f"{chosen['Feature Reduction (%)']:.2f}% reduction); "
            f"runtime ({chosen['Runtime (s)']:.4f} s) breaks any remaining "
            "feature-count tie."
        )

        recommendations.append({
            "Dataset": dataset_name,
            "Recommended Method": chosen["Method"],
            "Best Predictive Method(s)": frontier_methods,
            "Accuracy": chosen["Accuracy"],
            "MCC": chosen["MCC"],
            "Total Features": int(chosen["Total Features"]),
            "Selected Features": int(chosen["Selected Features"]),
            "Feature Reduction (%)": chosen["Feature Reduction (%)"],
            "Runtime (s)": chosen["Runtime (s)"],
            "Reason": reason,
        })

    return pd.DataFrame(recommendations)


def main():
    comparison = build_comparison()
    recommendations = recommend_methods(comparison)
    output_path = Path(__file__).parent / "recommendations.csv"
    recommendations.to_csv(output_path, index=False)

    print("\nMETHOD RECOMMENDATIONS")
    print(recommendations.to_string(index=False))
    print(f"\nRecommendations saved to {output_path}")


if __name__ == "__main__":
    main()
