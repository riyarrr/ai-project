"""Unit tests for Feature Explorer functions and consensus calculations."""

import unittest
import json
import numpy as np
import pandas as pd

from app import parse_feature_names, decode_feature_info


class FeatureExplorerTests(unittest.TestCase):
    def test_parse_feature_names_formats(self):
        # 1. Comma-separated string format (benchmark results)
        raw_bench = "493, 627, 377, 1679, 792"
        parsed_bench = parse_feature_names(raw_bench)
        self.assertEqual(parsed_bench, ["493", "627", "377", "1679", "792"])

        # 2. JSON-encoded list format (upload results)
        raw_upload = '["numeric__age", "numeric__cholesterol", "categorical__stage_II"]'
        parsed_upload = parse_feature_names(raw_upload)
        self.assertEqual(
            parsed_upload,
            ["numeric__age", "numeric__cholesterol", "categorical__stage_II"]
        )

        # 3. Single item without commas or brackets
        self.assertEqual(parse_feature_names("1400"), ["1400"])
        self.assertEqual(parse_feature_names("gene_1834"), ["gene_1834"])

        # 4. Empty or null inputs
        self.assertEqual(parse_feature_names(""), [])
        self.assertEqual(parse_feature_names(None), [])
        self.assertEqual(parse_feature_names(np.nan), [])
        self.assertEqual(parse_feature_names("   "), [])

    def test_decode_feature_info_numeric_encoded(self):
        info = decode_feature_info("numeric__blood_pressure", "Uploaded Dataset")
        self.assertEqual(info["Feature Name"], "numeric__blood_pressure")
        self.assertEqual(info["Original Column"], "blood_pressure")
        self.assertIn("Numeric", info["Encoding / Origin"])
        self.assertTrue(info["Is Encoded"])
        self.assertEqual(info["Encoding Status"], "Pipeline Encoded")

    def test_decode_feature_info_categorical_encoded(self):
        info = decode_feature_info("categorical__diagnosis_Malignant", "Uploaded Dataset")
        self.assertEqual(info["Feature Name"], "categorical__diagnosis_Malignant")
        self.assertEqual(info["Original Column"], "diagnosis")
        self.assertIn("One-Hot Encoded", info["Encoding / Origin"])
        self.assertIn("Malignant", info["Encoding / Origin"])
        self.assertTrue(info["Is Encoded"])
        self.assertEqual(info["Encoding Status"], "Pipeline Encoded")

    def test_decode_feature_info_benchmark_colon_cancer(self):
        info = decode_feature_info("493", "Colon Cancer")
        self.assertEqual(info["Feature Name"], "493")
        self.assertEqual(info["Original Column"], "Probe #493")
        self.assertIn("Microarray Probe ID", info["Encoding / Origin"])
        self.assertFalse(info["Is Encoded"])
        self.assertEqual(info["Encoding Status"], "Original Raw")

    def test_decode_feature_info_benchmark_leukemia(self):
        info = decode_feature_info("gene_1834", "Leukemia")
        self.assertEqual(info["Feature Name"], "gene_1834")
        self.assertEqual(info["Original Column"], "gene_1834")
        self.assertIn("Affymetrix Gene ID", info["Encoding / Origin"])
        self.assertFalse(info["Is Encoded"])
        self.assertEqual(info["Encoding Status"], "Original Raw")

    def test_decode_feature_info_raw_fallback(self):
        info = decode_feature_info("custom_sensor_1", "Unknown Dataset")
        self.assertEqual(info["Feature Name"], "custom_sensor_1")
        self.assertEqual(info["Original Column"], "custom_sensor_1")
        self.assertEqual(info["Encoding / Origin"], "Original Raw Feature")
        self.assertFalse(info["Is Encoded"])

    def test_consensus_and_jaccard_similarity(self):
        # Simulate sets of features
        method_sets = {
            "CFS": {"feat_A", "feat_B", "feat_C"},
            "FCBF": {"feat_A", "feat_D"},
            "FAST": {"feat_A", "feat_D"},
            "Hybrid": {"feat_A", "feat_B"},
        }

        # Jaccard index between FCBF and FAST (identical sets)
        jaccard_fcbf_fast = len(method_sets["FCBF"].intersection(method_sets["FAST"])) / len(
            method_sets["FCBF"].union(method_sets["FAST"])
        )
        self.assertEqual(jaccard_fcbf_fast, 1.0)

        # Jaccard index between CFS and Hybrid
        # intersection = {feat_A, feat_B} (2), union = {feat_A, feat_B, feat_C} (3)
        jaccard_cfs_hybrid = len(method_sets["CFS"].intersection(method_sets["Hybrid"])) / len(
            method_sets["CFS"].union(method_sets["Hybrid"])
        )
        self.assertAlmostEqual(jaccard_cfs_hybrid, 2 / 3, places=4)

        # Consensus frequency count
        all_feats = list(set.union(*method_sets.values()))
        freq = {f: sum(1 for m in method_sets.values() if f in m) for f in all_feats}
        self.assertEqual(freq["feat_A"], 4)  # Selected by all 4 methods
        self.assertEqual(freq["feat_B"], 2)  # Selected by CFS & Hybrid
        self.assertEqual(freq["feat_C"], 1)  # Unique to CFS
        self.assertEqual(freq["feat_D"], 2)  # Selected by FCBF & FAST


if __name__ == "__main__":
    unittest.main()
