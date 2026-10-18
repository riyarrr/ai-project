
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

from main import load_datasets


def hybrid_selection(X_train, y_train, top_k=10, generations=8):
    """Hybrid feature selection: mutual information + genetic algorithm."""

    rng = np.random.default_rng(42)

    # Stage 1: Filter using training data only.
    mi = mutual_info_classif(
        X_train, y_train, random_state=42
    )

    top_indices = np.argsort(mi)[::-1][
        :min(top_k, X_train.shape[1])
    ]

    X_reduced = X_train.iloc[:, top_indices]
    n_features = X_reduced.shape[1]

    if n_features == 0:
        return []

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )
    splits = list(cv.split(X_reduced, y_train))

    def fitness(mask):
        if not mask.any():
            return 0.0

        cols = X_reduced.columns[mask]
        scores = []

        for tr, va in splits:
            scaler = StandardScaler()

            x_tr = scaler.fit_transform(
                X_reduced.iloc[tr][cols]
            )
            x_va = scaler.transform(
                X_reduced.iloc[va][cols]
            )

            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(x_tr, y_train.iloc[tr])

            scores.append(
                model.score(x_va, y_train.iloc[va])
            )

        return (
            np.mean(scores)
            - 0.01 * mask.sum() / n_features
        )

    population_size = 10
    population = rng.integers(
        0, 2,
        size=(population_size, n_features)
    ).astype(bool)

    for chromosome in population:
        if not chromosome.any():
            chromosome[rng.integers(n_features)] = True

    for _ in range(generations):
        scores = np.array([
            fitness(individual)
            for individual in population
        ])

        ranked_pop = population[np.argsort(scores)[::-1]]

        new_pop = [
            ranked_pop[0].copy(),
            ranked_pop[1].copy()
        ]

        while len(new_pop) < population_size:
            parent_limit = min(5, population_size)

            p1 = ranked_pop[rng.integers(parent_limit)]
            p2 = ranked_pop[rng.integers(parent_limit)]

            if n_features > 1:
                point = rng.integers(1, n_features)
                child = np.concatenate([
                    p1[:point],
                    p2[point:]
                ])
            else:
                child = p1.copy()

            mutation = rng.random(n_features) < 0.1
            child[mutation] = ~child[mutation]

            if not child.any():
                child[rng.integers(n_features)] = True

            new_pop.append(child)

        population = np.array(new_pop)

    scores = np.array([
        fitness(individual)
        for individual in population
    ])

    best = population[np.argmax(scores)]

    # Safety fallback: use the best MI-ranked reduced-space feature.
    if not best.any():
        best[0] = True

    return list(X_reduced.columns[best])


def evaluate_hybrid(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    start = time.perf_counter()

    # Feature selection is performed only on training data.
    selected = hybrid_selection(X_train, y_train)

    if not selected:
        raise ValueError(
            f"Hybrid selection returned no features for {dataset_name}."
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train[selected])
    x_test = scaler.transform(X_test[selected])

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    runtime = time.perf_counter() - start

    return {
        "Dataset": dataset_name,
        "Method": "Hybrid",
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


def main():
    datasets = load_datasets()
    results = []

    for name, (X, y) in datasets.items():
        print(f"\nDataset: {name}")

        result = evaluate_hybrid(X, y, name)
        results.append(result)

        print("Selected feature count:", result["Selected Features"])
        print("Accuracy:", result["Accuracy"])
        print("MCC:", result["MCC"])
        print("Runtime (s):", result["Runtime (s)"])

    results_df = pd.DataFrame(results)

    print("\nHybrid Evaluation Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv("hybrid_results.csv", index=False)
    print("\nResults saved to hybrid_results.csv")


if __name__ == "__main__":
    main()
