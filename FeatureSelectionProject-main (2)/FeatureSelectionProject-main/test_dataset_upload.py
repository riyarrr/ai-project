"""Unit and integration tests for Dataset Upload validation."""

import unittest
import numpy as np
import pandas as pd

from dataset_validator import (
    validate_csv_content,
    validate_target_selection,
    get_dataset_statistics,
)


class DatasetUploadValidationTests(unittest.TestCase):
    def test_valid_csv_passes_all_checks(self):
        csv_bytes = (
            b"gene_1,gene_2,gene_3,diagnosis\n"
            b"1.2,0.5,3.1,tumor\n"
            b"2.1,1.1,2.8,normal\n"
            b"1.9,0.8,3.4,tumor\n"
            b"0.9,0.2,1.5,normal\n"
            b"1.5,0.7,2.9,tumor\n"
            b"0.8,0.3,1.4,normal\n"
        )
        df, errors, warnings = validate_csv_content(csv_bytes, "valid_biomarkers.csv")
        self.assertIsNotNone(df)
        self.assertEqual(len(errors), 0)
        self.assertEqual(df.shape, (6, 4))

        stats = get_dataset_statistics(df)
        self.assertEqual(stats["num_rows"], 6)
        self.assertEqual(stats["num_cols"], 4)
        self.assertEqual(stats["missing_cells"], 0)
        self.assertEqual(stats["duplicate_rows"], 0)
        self.assertEqual(stats["numeric_count"], 3)
        self.assertEqual(stats["categorical_count"], 1)

        is_valid, target_errors, target_warnings, class_counts = validate_target_selection(
            df, "diagnosis"
        )
        self.assertTrue(is_valid)
        self.assertEqual(len(target_errors), 0)
        self.assertEqual(len(class_counts), 2)
        self.assertEqual(class_counts["tumor"], 3)
        self.assertEqual(class_counts["normal"], 3)

    def test_invalid_case_1_empty_file(self):
        # Test 0-byte file
        df, errors, warnings = validate_csv_content(b"", "empty.csv")
        self.assertIsNone(df)
        self.assertTrue(any("empty (0 bytes)" in err for err in errors))

        # Test whitespace-only file
        df, errors, warnings = validate_csv_content(b"   \n  \n", "whitespace.csv")
        self.assertIsNone(df)
        self.assertTrue(any("empty (0 bytes)" in err for err in errors))

    def test_invalid_case_2_duplicate_column_names(self):
        csv_bytes = (
            b"gene_a,gene_b,gene_a,diagnosis\n"
            b"1.0,2.0,3.0,case\n"
            b"4.0,5.0,6.0,control\n"
            b"7.0,8.0,9.0,case\n"
        )
        df, errors, warnings = validate_csv_content(csv_bytes, "duplicate_cols.csv")
        self.assertIsNone(df)
        self.assertTrue(any("Duplicate column name(s) detected" in err for err in errors))
        self.assertTrue(any("'gene_a'" in err for err in errors))

    def test_invalid_case_3_empty_column_header(self):
        csv_bytes = (
            b"gene_a,,gene_c,diagnosis\n"
            b"1.0,2.0,3.0,case\n"
            b"4.0,5.0,6.0,control\n"
        )
        df, errors, warnings = validate_csv_content(csv_bytes, "empty_col_header.csv")
        self.assertIsNone(df)
        self.assertTrue(any("empty or whitespace-only column header" in err for err in errors))

    def test_invalid_case_4_missing_target_values(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": ["A", "B", None, "A", "B", "A"],
        })
        is_valid, errors, warnings, class_counts = validate_target_selection(df, "target")
        self.assertFalse(is_valid)
        self.assertTrue(any("missing value(s)" in err for err in errors))

    def test_invalid_case_5_target_with_single_class(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": ["A", "A", "A", "A", "A"],
        })
        is_valid, errors, warnings, class_counts = validate_target_selection(df, "target")
        self.assertFalse(is_valid)
        self.assertTrue(any("contains only 1 class" in err for err in errors))

    def test_invalid_case_6_minority_class_fewer_than_three_samples(self):
        df = pd.DataFrame({
            "feature1": range(10),
            "target": ["A"] * 8 + ["B"] * 2,  # B has only 2 samples (< 3)
        })
        is_valid, errors, warnings, class_counts = validate_target_selection(df, "target")
        self.assertFalse(is_valid)
        self.assertTrue(any("at least 3 samples" in err for err in errors))
        self.assertTrue(any("'B'" in err for err in errors))

    def test_invalid_case_7_infinite_feature_values(self):
        df = pd.DataFrame({
            "feature1": [1.0, np.inf, 3.0, 4.0, 5.0, 6.0],
            "target": ["A", "A", "A", "B", "B", "B"],
        })
        is_valid, errors, warnings, class_counts = validate_target_selection(df, "target")
        self.assertFalse(is_valid)
        self.assertTrue(any("infinite values" in err for err in errors))

    def test_disk_sample_files(self):
        from pathlib import Path
        test_dir = Path(__file__).parent / "test_datasets"

        # 1. Valid sample
        valid_bytes = (test_dir / "valid_sample.csv").read_bytes()
        df, errors, warnings = validate_csv_content(valid_bytes, "valid_sample.csv")
        self.assertIsNotNone(df)
        self.assertEqual(len(errors), 0)
        is_valid, t_errors, _, _ = validate_target_selection(df, "diagnosis")
        self.assertTrue(is_valid)
        self.assertEqual(len(t_errors), 0)

        # 2. Invalid empty
        empty_bytes = (test_dir / "invalid_empty.csv").read_bytes()
        df_e, errors_e, _ = validate_csv_content(empty_bytes, "invalid_empty.csv")
        self.assertIsNone(df_e)
        self.assertTrue(len(errors_e) > 0)

        # 3. Invalid duplicate headers
        dup_bytes = (test_dir / "invalid_duplicate_headers.csv").read_bytes()
        df_d, errors_d, _ = validate_csv_content(dup_bytes, "invalid_duplicate_headers.csv")
        self.assertIsNone(df_d)
        self.assertTrue(any("Duplicate" in err for err in errors_d))

        # 4. Invalid single class
        single_bytes = (test_dir / "invalid_single_class.csv").read_bytes()
        df_s, errors_s, _ = validate_csv_content(single_bytes, "invalid_single_class.csv")
        self.assertIsNotNone(df_s)
        is_valid_s, t_errors_s, _, _ = validate_target_selection(df_s, "diagnosis")
        self.assertFalse(is_valid_s)
        self.assertTrue(any("only 1 class" in err for err in t_errors_s))

        # 5. Invalid missing target
        missing_bytes = (test_dir / "invalid_missing_target.csv").read_bytes()
        df_m, errors_m, _ = validate_csv_content(missing_bytes, "invalid_missing_target.csv")
        self.assertIsNotNone(df_m)
        is_valid_m, t_errors_m, _, _ = validate_target_selection(df_m, "diagnosis")
        self.assertFalse(is_valid_m)
        self.assertTrue(any("missing value(s)" in err for err in t_errors_m))


if __name__ == "__main__":
    unittest.main()
