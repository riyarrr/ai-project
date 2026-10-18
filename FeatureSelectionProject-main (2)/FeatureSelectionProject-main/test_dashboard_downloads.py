"""Tests for dashboard download/export helpers: result filtering edge cases and
CSV/JSON export header, data-type, and row-count validation (Requirement 11).

These complement ``test_dashboard_exports.py`` and never modify existing modules.
"""

import io
import json
import unittest

import numpy as np
import pandas as pd

from dashboard_export_utils import (
    filter_comparison_results,
    format_all_features_csv,
    format_comparison_export,
    format_feature_list_csv,
    format_feature_payload_json,
    format_recommendation_export,
)


def _sample_results_df():
    return pd.DataFrame([
        {
            "Dataset": "Colon Cancer",
            "Method": "CFS",
            "Accuracy": 0.6154,
            "MCC": 0.30,
            "Selected Features": 12,
            "Total Features": 2000,
            "Feature Reduction (%)": 99.40,
            "Runtime (s)": 0.0177,
            "Execution Status": "Completed",
            "Selected Feature Names": json.dumps(["493", "1002", "1234"]),
        },
        {
            "Dataset": "Colon Cancer",
            "Method": "Hybrid",
            "Accuracy": 0.9231,
            "MCC": 0.8433,
            "Selected Features": 4,
            "Total Features": 2000,
            "Feature Reduction (%)": 99.80,
            "Runtime (s)": 5.0921,
            "Execution Status": "Completed",
            "Selected Feature Names": json.dumps(["493", "1002", "1234", "2010"]),
        },
        {
            "Dataset": "Colon Cancer",
            "Method": "FCBF",
            "Accuracy": np.nan,
            "MCC": np.nan,
            "Selected Features": np.nan,
            "Total Features": 2000,
            "Feature Reduction (%)": np.nan,
            "Runtime (s)": np.nan,
            "Execution Status": "Unsupported",
            "Selected Feature Names": "",
        },
    ])


class ResultFilteringEdgeCaseTests(unittest.TestCase):
    def test_filter_by_min_mcc_excludes_low_and_nan(self):
        df = _sample_results_df()
        filtered = filter_comparison_results(df, min_mcc=0.50)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["Method"], "Hybrid")

    def test_combined_method_accuracy_and_status_filters(self):
        df = _sample_results_df()
        filtered = filter_comparison_results(
            df,
            selected_methods=["CFS", "Hybrid", "FCBF"],
            min_accuracy=0.60,
            status_filter="Completed",
        )
        self.assertEqual(set(filtered["Method"]), {"CFS", "Hybrid"})

    def test_status_all_sentinel_keeps_everything(self):
        df = _sample_results_df()
        filtered = filter_comparison_results(df, status_filter="All")
        self.assertEqual(len(filtered), 3)

    def test_empty_and_none_inputs_return_empty_frame(self):
        self.assertTrue(filter_comparison_results(pd.DataFrame()).empty)
        self.assertTrue(filter_comparison_results(None).empty)

    def test_filter_never_mutates_source(self):
        df = _sample_results_df()
        original_len = len(df)
        filter_comparison_results(df, min_accuracy=0.90)
        self.assertEqual(len(df), original_len)


class DownloadExportValidationTests(unittest.TestCase):
    """Requirement 11: downloads contain the right headers, data types, and rows."""

    def test_comparison_csv_headers_types_and_rows(self):
        df = _sample_results_df()
        export_df = format_comparison_export(df)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        reparsed = pd.read_csv(io.BytesIO(csv_bytes))
        expected_headers = {
            "Dataset", "Method", "Accuracy", "MCC", "Selected Features",
            "Total Features", "Feature Reduction (%)", "Runtime (s)", "Execution Status",
        }
        self.assertTrue(expected_headers.issubset(set(reparsed.columns)))
        self.assertEqual(len(reparsed), len(df))

        # Numeric metric columns must survive the CSV round-trip as numbers.
        for col in ["Accuracy", "MCC", "Selected Features", "Total Features"]:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(reparsed[col]),
                f"{col} should round-trip as numeric",
            )
        # Accuracy rounding to 4 decimals preserved.
        hybrid_row = reparsed[reparsed["Method"] == "Hybrid"].iloc[0]
        self.assertAlmostEqual(float(hybrid_row["Accuracy"]), 0.9231, places=4)

    def test_empty_comparison_export_preserves_column_contract(self):
        export_df = format_comparison_export(pd.DataFrame())
        for col in [
            "Dataset", "Method", "Accuracy", "MCC", "Selected Features",
            "Total Features", "Feature Reduction (%)", "Runtime (s)", "Execution Status",
        ]:
            self.assertIn(col, export_df.columns)
        self.assertTrue(export_df.empty)

    def test_recommendation_csv_includes_provenance_columns(self):
        rec_data = {
            "Dataset": "Colon Cancer",
            "Recommended Method": "Hybrid",
            "Accuracy": 0.9231,
            "MCC": 0.8433,
            "Selected Features": 4,
            "Total Features": 2000,
            "Feature Reduction (%)": 99.80,
            "Runtime (s)": 5.0921,
        }
        export_rec = format_recommendation_export(
            rec_data, run_id="EVAL-20261018-120000", run_timestamp="2026-10-18 12:00:00 UTC"
        )
        csv_bytes = export_rec.to_csv(index=False).encode("utf-8")
        reparsed = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(reparsed), 1)
        for col in ["Recommended Method", "Run Identifier", "Evaluation Timestamp", "Scientific Disclaimer"]:
            self.assertIn(col, reparsed.columns)
        self.assertEqual(reparsed.iloc[0]["Run Identifier"], "EVAL-20261018-120000")

    def test_recommendation_export_handles_none_and_empty(self):
        self.assertTrue(format_recommendation_export(None).empty)
        self.assertTrue(format_recommendation_export(pd.DataFrame()).empty)

    def test_feature_list_csv_sequential_index_and_headers(self):
        feats = ["493", "1002", "2010"]
        feat_df = format_feature_list_csv(feats, method_name="CFS", dataset_name="Colon Cancer")
        csv_bytes = feat_df.to_csv(index=False).encode("utf-8")
        reparsed = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(reparsed), 3)
        self.assertEqual(reparsed["Feature Index"].tolist(), [1, 2, 3])
        for col in ["Dataset", "Method", "Feature Index", "Feature Name", "Original Column"]:
            self.assertIn(col, reparsed.columns)

    def test_consolidated_all_features_row_count_matches_total_selections(self):
        df = _sample_results_df()
        all_df = format_all_features_csv(df, dataset_name="Colon Cancer")
        # CFS contributes 3 features, Hybrid 4, FCBF 0 -> 7 total rows.
        self.assertEqual(len(all_df), 7)
        self.assertEqual(set(all_df["Method"]), {"CFS", "Hybrid"})
        self.assertTrue((all_df["Feature Index"] >= 1).all())

    def test_feature_payload_json_is_serializable(self):
        df = _sample_results_df()
        hybrid_row = df[df["Method"] == "Hybrid"].iloc[0]
        payload = format_feature_payload_json(
            hybrid_row, dataset_name="Colon Cancer", run_id="RUN-7", run_timestamp="TS-7"
        )
        # Must be JSON-serializable (downloads a valid file).
        dumped = json.dumps(payload)
        reloaded = json.loads(dumped)
        self.assertEqual(reloaded["run_id"], "RUN-7")
        self.assertEqual(reloaded["feature_count"], 4)
        self.assertEqual(reloaded["method"], "Hybrid")


if __name__ == "__main__":
    unittest.main()
