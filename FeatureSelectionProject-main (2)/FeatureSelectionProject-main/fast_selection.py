
import time
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import (
    mutual_info_score,
    accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from main import load_datasets


# Computational limit: keeps the complete feature graph manageable.
MAX_CANDIDATES = 100


def discretize(values, bins=10):
    """Convert continuous feature values into discrete bins."""
    values = pd.Series(values).reset_index(drop=True)

    if values.nunique(dropna=True) <= 1:
        return values.fillna("missing").astype(str)

    if pd.api.types.is_numeric_dtype(values):
        ranked = values.rank(method="average")
        n_bins = min(bins, int(values.nunique()))

        return (
            pd.qcut(
                ranked,
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
            .fillna(0)
            .astype(int)
        )

    return values.fillna("missing").astype(str)


def symmetrical_uncertainty(x, y):
    """Calculate SU(X,Y) from mutual information and entropy."""
    x = np.asarray(x)
    y = np.asarray(y)

    h_x = mutual_info_score(x, x)
    h_y = mutual_info_score(y, y)
    mutual_info = mutual_info_score(x, y)

    denominator = h_x + h_y

    if denominator == 0:
        return 0.0

    return float(2.0 * mutual_info / denominator)


def fast_selection(X, y, max_candidates=MAX_CANDIDATES, theta=0.0):
    """
    MST-based FAST-style feature selection.

    Relevance is measured using SU(feature, class).
    A complete graph is built only over the highest-ranked candidates.
    An MST is formed using distance = 1 - SU(feature_a, feature_b).
    Edges are pruned using the FAST relevance/redundancy rule.

    Note: max_candidates is a computational approximation. It can
    exclude features that a full FAST run would retain.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("X and y must not contain missing values.")

    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer.")

    if not 0 <= theta <= 1:
        raise ValueError("theta must be between 0 and 1.")

    print("Discretizing target...", flush=True)
    y_discrete = discretize(y)

    print("Calculating feature relevance...", flush=True)
    discrete_X = {}
    relevance = {}

    for feature in X.columns:
        discrete_values = discretize(X[feature])
        discrete_X[feature] = discrete_values
        relevance[feature] = symmetrical_uncertainty(
            discrete_values,
            y_discrete,
        )

    ranked = sorted(
        relevance,
        key=lambda feature: relevance[feature],
        reverse=True,
    )

    relevant = [
        feature for feature in ranked
        if relevance[feature] > theta
    ]

    # Fallback for datasets where every computed SU is zero.
    if not relevant:
        print(
            "No features passed the relevance threshold; "
            "using the highest-ranked candidates as a fallback.",
            flush=True,
        )
        relevant = ranked

    candidates = relevant[:max_candidates]

    print(f"Features passing threshold: {len(relevant)}", flush=True)
    print(f"Candidates used for graph: {len(candidates)}", flush=True)

    if len(candidates) == 1:
        return candidates

    # Build the weighted graph. Limiting candidates bounds pairwise work.
    print("Building candidate graph...", flush=True)
    graph = nx.Graph()
    graph.add_nodes_from(candidates)

    for i, feature_a in enumerate(candidates):
        values_a = discrete_X[feature_a]

        for feature_b in candidates[i + 1:]:
            su = symmetrical_uncertainty(
                values_a,
                discrete_X[feature_b],
            )

            graph.add_edge(
                feature_a,
                feature_b,
                weight=su,
                su=su,
            )

    print("Calculating minimum spanning tree...", flush=True)
    mst = nx.minimum_spanning_tree(graph, weight="weight")
    forest = mst.copy()

    # FAST-style pruning: remove an edge when its feature-feature SU
    # is lower than the relevance of both endpoint features.
    print("Pruning MST edges...", flush=True)
    for feature_a, feature_b, data in list(mst.edges(data=True)):
        redundancy = data["su"]

        if (
            redundancy < relevance[feature_a]
            and redundancy < relevance[feature_b]
        ):
            forest.remove_edge(feature_a, feature_b)

    # Select the most class-relevant feature from each component.
    selected = []

    for component in nx.connected_components(forest):
        representative = max(
            component,
            key=lambda feature: relevance[feature],
        )
        selected.append(representative)

    selected.sort(
        key=lambda feature: relevance[feature],
        reverse=True,
    )

    print(f"Selected features: {len(selected)}", flush=True)
    return selected


def evaluate_fast(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    start = time.perf_counter()

    # Selection uses training data only.
    selected = fast_selection(X_train, y_train)

    if not selected:
        raise ValueError(
            f"FAST returned no features for {dataset_name}."
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected])
    X_test_scaled = scaler.transform(X_test[selected])

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    runtime = time.perf_counter() - start

    return {
        "Dataset": dataset_name,
        "Method": "FAST",
        "Total Features": X.shape[1],
        "Selected Features": len(selected),
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "MCC": round(matthews_corrcoef(y_test, predictions), 4),
        "Runtime (s)": round(runtime, 4),
        "Selected Feature Names": ", ".join(map(str, selected)),
    }


def main():
    datasets = load_datasets()
    results = []

    for name, (X, y) in datasets.items():
        print(f"\nDataset: {name}", flush=True)

        result = evaluate_fast(X, y, name)
        results.append(result)

        print("Selected features:", result["Selected Feature Names"])
        print("Number selected:", result["Selected Features"])
        print("Accuracy:", result["Accuracy"])
        print("MCC:", result["MCC"])
        print("Runtime (s):", result["Runtime (s)"])

    results_df = pd.DataFrame(results)

    print("\nFAST Evaluation Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv("fast_results.csv", index=False)
    print("\nResults saved to fast_results.csv")


if __name__ == "__main__":
    main()
