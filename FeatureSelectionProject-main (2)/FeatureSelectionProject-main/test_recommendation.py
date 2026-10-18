import unittest

import pandas as pd

from evaluation import feature_reduction_percentage
from recommendation import recommend_methods


def result(method, accuracy, mcc, selected, runtime, total=100):
    return {
        "Dataset": "Example",
        "Method": method,
        "Accuracy": accuracy,
        "MCC": mcc,
        "Total Features": total,
        "Selected Features": selected,
        "Runtime (s)": runtime,
    }


class FeatureReductionTests(unittest.TestCase):
    def test_calculates_reduction_percentage(self):
        self.assertEqual(feature_reduction_percentage(2000, 4), 99.8)

    def test_rejects_invalid_feature_counts(self):
        for total, selected in ((0, 1), (10, 0), (10, 11), (10.5, 2)):
            with self.subTest(total=total, selected=selected):
                with self.assertRaises(ValueError):
                    feature_reduction_percentage(total, selected)


class RecommendationTests(unittest.TestCase):
    def test_uses_predictive_pareto_then_compactness_and_runtime(self):
        results = pd.DataFrame([
            result("Dominated", 0.7, 0.4, 1, 0.1),
            result("Predictive but larger", 0.9, 0.8, 8, 2.0),
            result("Compact slower", 0.9, 0.8, 2, 1.0),
            result("Compact faster", 0.9, 0.8, 2, 0.5),
        ])

        recommendation = recommend_methods(results).iloc[0]

        self.assertEqual(recommendation["Recommended Method"], "Compact faster")
        self.assertEqual(
            recommendation["Best Predictive Method(s)"],
            "Compact faster, Compact slower, Predictive but larger",
        )
        self.assertEqual(recommendation["Feature Reduction (%)"], 98.0)
        self.assertIn("Accuracy/MCC Pareto methods", recommendation["Reason"])

    def test_keeps_predictive_tradeoff_methods_on_frontier(self):
        results = pd.DataFrame([
            result("Higher accuracy", 0.9, 0.6, 5, 1.0),
            result("Higher MCC", 0.8, 0.8, 3, 0.5),
        ])

        recommendation = recommend_methods(results).iloc[0]

        self.assertEqual(recommendation["Recommended Method"], "Higher MCC")
        self.assertEqual(
            recommendation["Best Predictive Method(s)"],
            "Higher accuracy, Higher MCC",
        )

    def test_rejects_missing_metrics(self):
        results = pd.DataFrame([
            result("Incomplete", 0.9, None, 2, 1.0),
        ])

        with self.assertRaisesRegex(ValueError, "missing"):
            recommend_methods(results)

    def test_rejects_invalid_metric_values(self):
        results = pd.DataFrame([
            result("Invalid", 1.1, 0.8, 2, 1.0),
        ])

        with self.assertRaisesRegex(ValueError, "Invalid result values"):
            recommend_methods(results)

    def test_ties_are_resolved_deterministically_by_method_name(self):
        results = pd.DataFrame([
            result("Zulu", 0.9, 0.8, 2, 0.5),
            result("Alpha", 0.9, 0.8, 2, 0.5),
        ])

        recommendation = recommend_methods(results).iloc[0]

        self.assertEqual(recommendation["Recommended Method"], "Alpha")


if __name__ == "__main__":
    unittest.main()
