
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

from main import load_datasets


def pso_selection(
    X_train,
    y_train,
    particles=10,
    iterations=8,
    max_features=20,
    feature_penalty=0.10
):
    """Binary PSO feature selection using cross-validated KNN fitness."""

    rng = np.random.default_rng(42)
    n_features = X_train.shape[1]

    if n_features == 0:
        return []

    max_features = min(max_features, n_features)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )
    splits = list(cv.split(X_train, y_train))

    def repair(mask):
        """Ensure the particle selects between 1 and max_features."""
        mask = mask.copy()
        active = np.flatnonzero(mask)

        if len(active) == 0:
            mask[rng.integers(n_features)] = True
            active = np.flatnonzero(mask)

        if len(active) > max_features:
            remove = rng.choice(
                active,
                size=len(active) - max_features,
                replace=False
            )
            mask[remove] = False

        return mask

    def fitness(mask):
        selected_count = int(mask.sum())

        if selected_count == 0 or selected_count > max_features:
            return -1.0

        selected = X_train.columns[mask]
        scores = []

        for train_idx, val_idx in splits:
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

        size_penalty = feature_penalty * (
            selected_count / max_features
        )

        return float(np.mean(scores) - size_penalty)

    # Initialize compact particles.
    positions = np.zeros((particles, n_features), dtype=float)
    velocities = rng.uniform(-0.5, 0.5, (particles, n_features))

    for i in range(particles):
        count = rng.integers(1, max_features + 1)
        chosen = rng.choice(n_features, size=count, replace=False)
        positions[i, chosen] = 1.0

    personal_masks = np.array([
        repair(position > 0.5) for position in positions
    ])

    personal_scores = np.array([
        fitness(mask) for mask in personal_masks
    ])

    best_idx = int(np.argmax(personal_scores))
    global_best_mask = personal_masks[best_idx].copy()
    global_score = personal_scores[best_idx]

    # Binary PSO update loop.
    for _ in range(iterations):
        for i in range(particles):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)

            velocities[i] = (
                0.5 * velocities[i]
                + 1.5 * r1 * (
                    personal_masks[i].astype(float) - positions[i]
                )
                + 1.5 * r2 * (
                    global_best_mask.astype(float) - positions[i]
                )
            )

            positions[i] = np.clip(
                positions[i] + velocities[i], 0, 1
            )

            mask = repair(positions[i] > 0.5)
            positions[i] = mask.astype(float)

            score = fitness(mask)

            if score > personal_scores[i]:
                personal_masks[i] = mask.copy()
                personal_scores[i] = score

            if score > global_score:
                global_best_mask = mask.copy()
                global_score = score

    return list(X_train.columns[global_best_mask])


def evaluate_pso(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    start = time.perf_counter()

    selected = pso_selection(X_train, y_train)

    if not selected:
        raise ValueError(
            f"PSO selected no features for {dataset_name}."
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
        "Method": "PSO",
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

        result = evaluate_pso(X, y, name)
        results.append(result)

        print("Selected feature count:", result["Selected Features"])
        print("Accuracy:", result["Accuracy"])
        print("MCC:", result["MCC"])
        print("Runtime (s):", result["Runtime (s)"])

    results_df = pd.DataFrame(results)

    print("\nPSO Evaluation Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv("pso_results.csv", index=False)
    print("\nResults saved to pso_results.csv")


if __name__ == "__main__":
    main()
