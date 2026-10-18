
import pandas as pd

from main import load_datasets
from evaluation import evaluate_features


datasets = load_datasets()
results = []

for dataset_name, (X, y) in datasets.items():
    # Test the evaluation function using the first five features.
    selected_features = list(X.columns[:5])

    results.append(
        evaluate_features(
            X,
            y,
            dataset_name,
            "Manual Test",
            selected_features
        )
    )

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
