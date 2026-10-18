import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from upload_recommendation import (
    evaluate_uploaded_dataset,
    load_uploaded_csv,
    main as upload_cli_main,
)


def binary_upload_data():
    rng = np.random.default_rng(42)
    rows = 48
    target = np.array(["case" if index % 2 else "control" for index in range(rows)])
    signal = np.array([1 if label == "case" else -1 for label in target])
    frame = pd.DataFrame({
        "signal": signal + rng.normal(0, 0.2, rows),
        "numeric_missing": rng.normal(size=rows),
        "group": ["A" if label == "case" else "B" for label in target],
        "site": [f"site-{index % 4}" for index in range(rows)],
        "target": target,
    })
    frame.loc[::9, "numeric_missing"] = np.nan
    frame.loc[::11, "group"] = None
    return frame


class UploadedDatasetTests(unittest.TestCase):
    def test_cli_prints_selected_names_for_recommended_method(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "cli.csv"
            csv_path.write_text(
                "signal,target\n1,case\n-1,control\n",
                encoding="utf-8",
            )
            outcome = {
                "train_samples": 1,
                "test_samples": 1,
                "target_classes": ["case", "control"],
                "statuses": pd.DataFrame([{
                    "Method": "FCBF",
                    "Status": "Completed",
                    "Details": "",
                }]),
                "results": pd.DataFrame([{
                    "Dataset": "cli",
                    "Method": "FCBF",
                    "Total Features": 1,
                    "Selected Features": 1,
                    "Accuracy": 1.0,
                    "MCC": 1.0,
                    "Feature Reduction (%)": 0.0,
                    "Runtime (s)": 0.01,
                    "Selected Feature Names": json.dumps(["numeric__signal"]),
                }]),
                "recommendations": pd.DataFrame([{
                    "Dataset": "cli",
                    "Recommended Method": "FCBF",
                    "Best Predictive Method(s)": "FCBF",
                    "Accuracy": 1.0,
                    "MCC": 1.0,
                    "Total Features": 1,
                    "Selected Features": 1,
                    "Feature Reduction (%)": 0.0,
                    "Runtime (s)": 0.01,
                    "Reason": "Best trade-off",
                }]),
            }
            output = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "upload_recommendation.py",
                        str(csv_path),
                        "--target",
                        "target",
                    ],
                ),
                patch(
                    "upload_recommendation.evaluate_uploaded_dataset",
                    return_value=outcome,
                ),
                contextlib.redirect_stdout(output),
            ):
                upload_cli_main()

        self.assertIn(
            "Recommended features for cli (FCBF): ['numeric__signal']",
            output.getvalue(),
        )

    def test_csv_loader_rejects_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "duplicate.csv"
            csv_path.write_text("value,value,target\n1,2,A\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unique"):
                load_uploaded_csv(csv_path, "target")

    def test_unseen_binary_csv_runs_methods_and_recommends_from_fresh_results(self):
        frame = binary_upload_data()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "new_clinical_data.csv"
            frame.to_csv(csv_path, index=False)
            uploaded = load_uploaded_csv(csv_path, "target")
            outcome = evaluate_uploaded_dataset(
                uploaded,
                "target",
                dataset_name="new_clinical_data",
            )

        self.assertEqual(
            set(outcome["statuses"]["Method"]),
            {"CFS", "FCBF", "FAST", "GA", "PSO", "Hybrid"},
        )
        self.assertTrue(
            outcome["statuses"]["Status"].isin(["Completed", "Error"]).all()
        )
        self.assertFalse(outcome["results"].empty)
        self.assertEqual(
            set(outcome["results"]["Dataset"]),
            {"new_clinical_data"},
        )
        self.assertEqual(
            outcome["recommendations"].iloc[0]["Dataset"],
            "new_clinical_data",
        )
        self.assertIn(
            outcome["recommendations"].iloc[0]["Recommended Method"],
            set(outcome["results"]["Method"]),
        )
        self.assertEqual(len(outcome["results"]), 6)
        self.assertTrue(
            outcome["results"]["Accuracy"].between(0, 1).all()
        )
        self.assertTrue(
            outcome["results"]["MCC"].between(-1, 1).all()
        )

    def test_multiclass_upload_reports_cfs_limitation_and_evaluates_others(self):
        rng = np.random.default_rng(7)
        rows = 60
        labels = np.array([f"class-{index % 3}" for index in range(rows)])
        class_signal = np.array([
            int(label[-1]) for label in labels
        ])
        frame = pd.DataFrame({
            "signal": class_signal + rng.normal(0, 0.15, rows),
            "numeric_noise": rng.normal(size=rows),
            "category": [f"group-{value}" for value in class_signal],
            "target": labels,
        })

        outcome = evaluate_uploaded_dataset(
            frame,
            "target",
            dataset_name="unseen_multiclass",
        )

        cfs = outcome["statuses"].set_index("Method").loc["CFS"]
        self.assertEqual(cfs["Status"], "Unsupported")
        self.assertIn("binary classification", cfs["Details"])
        self.assertEqual(
            set(outcome["results"]["Method"]),
            {"FCBF", "FAST", "GA", "PSO", "Hybrid"},
        )
        self.assertEqual(
            outcome["recommendations"].iloc[0]["Dataset"],
            "unseen_multiclass",
        )

    def test_thousand_feature_upload_respects_high_dimensional_workflow(self):
        rng = np.random.default_rng(21)
        rows = 42
        labels = np.array(["positive" if index % 2 else "negative" for index in range(rows)])
        features = rng.normal(size=(rows, 1_000))
        features[:, 0] += np.where(labels == "positive", 2.0, -2.0)
        frame = pd.DataFrame(
            features,
            columns=[f"feature_{index}" for index in range(features.shape[1])],
        )
        frame["target"] = labels

        outcome = evaluate_uploaded_dataset(
            frame,
            "target",
            dataset_name="unseen_high_dimensional",
        )

        self.assertEqual(len(outcome["results"]), 6)
        self.assertTrue(
            outcome["statuses"]["Status"].eq("Completed").all()
        )
        self.assertEqual(
            outcome["results"]["Total Features"].unique().tolist(),
            [1_000],
        )

    def test_rejects_too_small_or_single_class_data(self):
        tiny = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "target": ["A", "A", "B", "B"],
        })
        with self.assertRaisesRegex(ValueError, "at least 3 samples"):
            evaluate_uploaded_dataset(tiny, "target")

        one_class = pd.DataFrame({
            "feature": range(12),
            "target": ["A"] * 12,
        })
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            evaluate_uploaded_dataset(one_class, "target")


if __name__ == "__main__":
    unittest.main()
