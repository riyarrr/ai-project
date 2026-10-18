
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

from main import load_datasets


def cfs_merit(k, feature_class_corr, feature_feature_corr):
    """Calculate the CFS merit of a feature subset."""
    if k == 0:
        return 0.0

    denominator = np.sqrt(
        k + k * (k - 1) * feature_feature_corr
    )

    if denominator == 0:
        return 0.0

    return (k * feature_class_corr) / denominator


def cfs_selection(X, y, max_features=5):
    """Select features using a greedy CFS-style search."""

    # Encode class labels numerically for correlation calculations.
    # Original labels remain unchanged for model training.
    y_numeric = pd.Series(
        pd.factorize(pd.Series(y, index=X.index))[0],
        index=X.index
    )

    # Feature-to-class correlation, calculated from training data only.
    target_corr = X.apply(
        lambda col: abs(col.corr(y_numeric))
    ).fillna(0)

    # Feature-to-feature correlation.
    corr = X.corr().abs().fillna(0)

    selected = []
    remaining = list(X.columns)
    best_merit = 0.0

    while remaining and len(selected) < max_features:
        best_feature = None
        current_best = best_merit

        for feature in remaining:
            subset = selected + [feature]
            k = len(subset)

            relevance = target_corr[subset].mean()

            if k > 1:
                pairs = [
                    corr.loc[a, b]
                    for i, a in enumerate(subset)
                    for b in subset[i + 1:]
                ]
                redundancy = float(np.mean(pairs))
            else:
                redundancy = 0.0

            merit = cfs_merit(
                k,
                relevance,
                redundancy
            )

            if merit > current_best:
                current_best = merit
                best_feature = feature

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_merit = current_best

    return selected


def main():
    datasets = load_datasets()
    results = []

    for name, (X, y) in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {name}")
        print(f"Samples: {X.shape[0]}")
        print(f"Total features: {X.shape[1]}")

        # Split before feature selection to avoid test-set leakage.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        start = time.perf_counter()

        # Select features using only the training set.
        selected = cfs_selection(
            X_train,
            y_train,
            max_features=5
        )

        if not selected:
            raise ValueError(
                f"CFS selected no features for {name}."
            )

        # Fit the scaler only on training data.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train[selected]
        )
        X_test_scaled = scaler.transform(
            X_test[selected]
        )

        # Train and evaluate the classifier.
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        # Includes feature selection, scaling, fitting and prediction.
        runtime = time.perf_counter() - start

        result = {
            "Dataset": name,
            "Method": "CFS",
            "Total Features": X.shape[1],
            "Selected Features": len(selected),
            "Accuracy": round(
                accuracy_score(y_test, predictions), 4
            ),
            "MCC": round(
                matthews_corrcoef(y_test, predictions), 4
            ),
            "Runtime (s)": round(runtime, 4),
            "Selected Feature Names": ", ".join(
                map(str, selected)
            )
        }

        results.append(result)

        print("Selected features:", selected)
        print("Accuracy:", result["Accuracy"])
        print("MCC:", result["MCC"])
        print("Runtime (s):", result["Runtime (s)"])

    results_df = pd.DataFrame(results)

    print("\nCFS Evaluation Results:")
    print(results_df.to_string(index=False))

    output_file = "cfs_classical_results.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
