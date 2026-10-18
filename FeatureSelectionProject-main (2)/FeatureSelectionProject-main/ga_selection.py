
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

from main import load_datasets


def ga_selection(
    X_train,
    y_train,
    population_size=12,
    generations=10,
    mutation_rate=0.1,
    max_features=20,
    feature_penalty=0.10
):
    """Select a compact feature subset using a Genetic Algorithm."""

    n_features = X_train.shape[1]

    if n_features == 0:
        return []

    max_features = min(max_features, n_features)
    rng = np.random.default_rng(42)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    split_indices = list(cv.split(X_train, y_train))

    def fitness(mask):
        selected_count = int(mask.sum())

        if selected_count == 0:
            return -1.0

        # Enforce the maximum subset size.
        if selected_count > max_features:
            return -1.0 - selected_count / n_features

        selected = X_train.columns[mask]
        scores = []

        for train_idx, val_idx in split_indices:
            x_tr = X_train.iloc[train_idx][selected]
            x_val = X_train.iloc[val_idx][selected]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_val = scaler.transform(x_val)

            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(x_tr, y_tr)

            scores.append(model.score(x_val, y_val))

        # Reward validation accuracy while penalizing larger subsets.
        size_penalty = feature_penalty * (
            selected_count / max_features
        )

        return float(np.mean(scores) - size_penalty)

    def repair(chromosome):
        """Keep chromosomes nonempty and within the feature limit."""
        chromosome = chromosome.copy()
        active = np.flatnonzero(chromosome)

        if len(active) == 0:
            chromosome[rng.integers(n_features)] = True
            active = np.flatnonzero(chromosome)

        if len(active) > max_features:
            remove = rng.choice(
                active,
                size=len(active) - max_features,
                replace=False
            )
            chromosome[remove] = False

        return chromosome

    # Initialize compact random chromosomes.
    population = []

    for _ in range(population_size):
        count = rng.integers(1, max_features + 1)
        indices = rng.choice(
            n_features,
            size=count,
            replace=False
        )
        chromosome = np.zeros(n_features, dtype=bool)
        chromosome[indices] = True
        population.append(chromosome)

    population = np.array(population)

    for _ in range(generations):
        scores = np.array([
            fitness(individual)
            for individual in population
        ])

        ranked = population[np.argsort(scores)[::-1]]

        elite_count = min(2, population_size)
        new_population = [
            ranked[i].copy()
            for i in range(elite_count)
        ]

        while len(new_population) < population_size:
            parent_limit = min(5, population_size)

            parent1 = ranked[rng.integers(parent_limit)]
            parent2 = ranked[rng.integers(parent_limit)]

            if n_features > 1:
                point = rng.integers(1, n_features)
                child = np.concatenate([
                    parent1[:point],
                    parent2[point:]
                ])
            else:
                child = parent1.copy()

            mutation = rng.random(n_features) < mutation_rate
            child[mutation] = ~child[mutation]

            new_population.append(repair(child))

        population = np.array(new_population)

    scores = np.array([
        fitness(individual)
        for individual in population
    ])

    best = population[np.argmax(scores)]

    return list(X_train.columns[best])


def evaluate_ga(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    start = time.perf_counter()

    selected = ga_selection(X_train, y_train)

    if not selected:
        raise ValueError(
            f"GA selected no features for {dataset_name}."
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
        "Method": "GA",
        "Total Features": X.shape[1],
        "Selected Features": len(selected),
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "MCC": round(
            matthews_corrcoef(y_test, predictions), 4
        ),
        "Runtime (s)": round(runtime, 4),
        "Selected Feature Names": ", ".join(map(str, selected))
    }


def main():
    datasets = load_datasets()
    results = []

    for name, (X, y) in datasets.items():
        print(f"\nDataset: {name}")

        result = evaluate_ga(X, y, name)
        results.append(result)

        print("Selected feature count:", result["Selected Features"])
        print("Accuracy:", result["Accuracy"])
        print("MCC:", result["MCC"])
        print("Runtime (s):", result["Runtime (s)"])

    results_df = pd.DataFrame(results)

    print("\nGenetic Algorithm Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv("ga_results.csv", index=False)
    print("\nResults saved to ga_results.csv")


if __name__ == "__main__":
    main()
