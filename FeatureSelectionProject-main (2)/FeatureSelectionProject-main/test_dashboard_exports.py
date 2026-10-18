"""Unit tests for dashboard export formatting, sanitization, filtering, and provenance decoding."""

import datetime
import json
import unittest
import numpy as np
import pandas as pd

from dashboard_export_utils import (
    decode_feature_info,
    filter_comparison_results,
    format_all_features_csv,
    format_comparison_export,
    format_feature_list_csv,
    format_feature_payload_json,
    format_recommendation_export,
    generate_run_identifier,
    parse_feature_names,
    sanitize_error_message,
)


class DashboardExportUtilsTests(unittest.TestCase):
    def test_sanitize_error_message(self):
        # Path sanitization
        win_path_err = "FileNotFoundError: Cannot find file C:\\Users\\secret_admin\\project\\data.csv on drive"
        sanitized_win = sanitize_error_message(win_path_err)
        self.assertNotIn("Users", sanitized_win)
        self.assertNotIn("secret_admin", sanitized_win)
        self.assertIn("<system_path>", sanitized_win)

        unix_path_err = "IOError: /home/testuser/workspace/dataset.csv not readable"
        sanitized_unix = sanitize_error_message(unix_path_err)
        self.assertNotIn("/home/testuser", sanitized_unix)
        self.assertIn("<system_path>", sanitized_unix)

        # Credentials sanitization
        cred_err = "AuthError: Invalid Bearer token=abc123xyz890_secret provided"
        sanitized_cred = sanitize_error_message(cred_err)
        self.assertNotIn("abc123xyz890_secret", sanitized_cred)
        self.assertIn("<redacted>", sanitized_cred)

        # Clean string passes through
        clean_err = "ValueError: Target must have at least 2 classes"
        self.assertEqual(sanitize_error_message(clean_err), clean_err)

    def test_generate_run_identifier(self):
        fixed_dt = datetime.datetime(2026, 10, 18, 12, 30, 45, tzinfo=datetime.timezone.utc)
        run_id, ts_str = generate_run_identifier(prefix="TEST_RUN", timestamp_dt=fixed_dt)
        self.assertEqual(run_id, "TEST_RUN-20261018-123045")
        self.assertEqual(ts_str, "2026-10-18 12:30:45 UTC")

    def test_parse_feature_names(self):
        # JSON string format
        json_feats = '["gene_alpha", "gene_beta", "gene_gamma"]'
        self.assertEqual(parse_feature_names(json_feats), ["gene_alpha", "gene_beta", "gene_gamma"])

        # Comma-separated format
        csv_feats = "gene_1, gene_2 ,gene_3"
        self.assertEqual(parse_feature_names(csv_feats), ["gene_1", "gene_2", "gene_3"])

        # List format
        list_feats = ["probe_A", "probe_B"]
        self.assertEqual(parse_feature_names(list_feats), ["probe_A", "probe_B"])

        # Empty / None formats
        self.assertEqual(parse_feature_names(None), [])
        self.assertEqual(parse_feature_names(""), [])
        self.assertEqual(parse_feature_names(np.nan), [])

    def test_decode_feature_info(self):
        # Encoded numeric
        num_dec = decode_feature_info("numeric__tumor_size", dataset_name="Custom")
        self.assertTrue(num_dec["Is Encoded"])
        self.assertEqual(num_dec["Original Column"], "tumor_size")
        self.assertEqual(num_dec["Encoding Status"], "Pipeline Encoded")

        # Encoded categorical
        cat_dec = decode_feature_info("categorical__stage_T2", dataset_name="Custom")
        self.assertTrue(cat_dec["Is Encoded"])
        self.assertEqual(cat_dec["Original Column"], "stage")
        self.assertIn("One-Hot Encoded", cat_dec["Encoding / Origin"])

        # Benchmark raw features
        colon_dec = decode_feature_info("493", dataset_name="Colon Cancer")
        self.assertFalse(colon_dec["Is Encoded"])
        self.assertEqual(colon_dec["Encoding Status"], "Original Raw")
        self.assertIn("Probe #493", colon_dec["Original Column"])

        leuk_dec = decode_feature_info("M27891_at", dataset_name="Leukemia")
        self.assertFalse(leuk_dec["Is Encoded"])
        self.assertEqual(leuk_dec["Original Column"], "M27891_at")

    def test_filter_comparison_results(self):
        df = pd.DataFrame([
            {"Method": "CFS", "Accuracy": 0.85, "MCC": 0.70, "Execution Status": "Completed"},
            {"Method": "GA", "Accuracy": 0.80, "MCC": 0.65, "Execution Status": "Completed"},
            {"Method": "PSO", "Accuracy": 0.60, "MCC": 0.30, "Execution Status": "Completed"},
            {"Method": "FCBF", "Accuracy": np.nan, "MCC": np.nan, "Execution Status": "Unsupported"},
        ])

        # Filter by method
        m_filtered = filter_comparison_results(df, selected_methods=["CFS", "GA"])
        self.assertEqual(len(m_filtered), 2)
        self.assertEqual(set(m_filtered["Method"]), {"CFS", "GA"})

        # Filter by min accuracy
        acc_filtered = filter_comparison_results(df, min_accuracy=0.80)
        self.assertEqual(len(acc_filtered), 2)
        self.assertTrue(all(acc_filtered["Accuracy"] >= 0.80))

        # Filter by status
        status_filtered = filter_comparison_results(df, status_filter="Completed")
        self.assertEqual(len(status_filtered), 3)
        self.assertNotIn("FCBF", status_filtered["Method"].values)

    def test_format_comparison_export(self):
        df = pd.DataFrame([
            {
                "Dataset": "Colon Cancer",
                "Method": "Hybrid",
                "Accuracy": 0.9230769,
                "MCC": 0.843274,
                "Selected Features": 4,
                "Total Features": 2000,
                "Feature Reduction (%)": 99.80000,
                "Runtime (s)": 5.09214,
                "Execution Status": "Completed",
            }
        ])
        export_df = format_comparison_export(df)
        self.assertEqual(len(export_df), 1)
        expected_cols = [
            "Dataset", "Method", "Accuracy", "MCC", "Selected Features",
            "Total Features", "Feature Reduction (%)", "Runtime (s)", "Execution Status"
        ]
        for col in expected_cols:
            self.assertIn(col, export_df.columns)

        # Check rounded values
        row = export_df.iloc[0]
        self.assertEqual(row["Accuracy"], 0.9231)
        self.assertEqual(row["MCC"], 0.8433)
        self.assertEqual(row["Runtime (s)"], 5.0921)
        self.assertEqual(row["Selected Features"], 4)

    def test_format_recommendation_export(self):
        rec_data = {
            "Dataset": "Leukemia",
            "Recommended Method": "Hybrid",
            "Best Predictive Method(s)": "CFS, Hybrid",
            "Accuracy": 1.0,
            "MCC": 1.0,
            "Selected Features": 3,
            "Total Features": 7128,
            "Feature Reduction (%)": 99.9579,
            "Runtime (s)": 6.223,
            "Dominating Count": 0,
            "Reason": "Pareto optimal subset with fewest features",
        }
        export_rec = format_recommendation_export(
            rec_data,
            run_id="BENCHMARK-RUN-LEUK",
            run_timestamp="2026-10-18 10:00:00 UTC",
        )
        self.assertEqual(len(export_rec), 1)
        row = export_rec.iloc[0]
        self.assertEqual(row["Recommended Method"], "Hybrid")
        self.assertEqual(row["Run Identifier"], "BENCHMARK-RUN-LEUK")
        self.assertEqual(row["Evaluation Timestamp"], "2026-10-18 10:00:00 UTC")
        self.assertIn("Non-causal", row["Scientific Disclaimer"])

    def test_format_feature_list_csv(self):
        feats = ["numeric__age", "numeric__bmi", "categorical__smoker_yes"]
        df_feats = format_feature_list_csv(feats, method_name="GA", dataset_name="Health Study")
        self.assertEqual(len(df_feats), 3)
        self.assertEqual(df_feats["Feature Index"].tolist(), [1, 2, 3])
        self.assertEqual(df_feats["Original Column"].tolist(), ["age", "bmi", "smoker"])
        self.assertTrue(all(df_feats["Method"] == "GA"))

    def test_format_all_features_csv(self):
        results_df = pd.DataFrame([
            {
                "Method": "CFS",
                "Accuracy": 0.90,
                "MCC": 0.80,
                "Runtime (s)": 0.12,
                "Selected Feature Names": json.dumps(["feat_1", "feat_2"]),
            },
            {
                "Method": "FCBF",
                "Accuracy": 0.85,
                "MCC": 0.70,
                "Runtime (s)": 0.08,
                "Selected Feature Names": json.dumps(["feat_1"]),
            },
        ])
        all_df = format_all_features_csv(results_df, dataset_name="Experiment A")
        self.assertEqual(len(all_df), 3)
        self.assertEqual(set(all_df["Method"]), {"CFS", "FCBF"})
        self.assertEqual(set(all_df["Feature Name"]), {"feat_1", "feat_2"})

    def test_format_feature_payload_json(self):
        row = {
            "Method": "Hybrid",
            "Accuracy": 0.95,
            "MCC": 0.89,
            "Selected Features": 2,
            "Total Features": 100,
            "Feature Reduction (%)": 98.0,
            "Runtime (s)": 4.12,
            "Selected Feature Names": json.dumps(["probe_493", "probe_1002"]),
        }
        payload = format_feature_payload_json(row, dataset_name="Colon Cancer", run_id="RUN-100")
        self.assertEqual(payload["dataset"], "Colon Cancer")
        self.assertEqual(payload["method"], "Hybrid")
        self.assertEqual(payload["run_id"], "RUN-100")
        self.assertEqual(payload["feature_count"], 2)
        self.assertEqual(payload["features"], ["probe_493", "probe_1002"])
        self.assertEqual(payload["metrics"]["accuracy"], 0.95)
        self.assertIn("scientific_disclaimer", payload)


if __name__ == "__main__":
    unittest.main()
