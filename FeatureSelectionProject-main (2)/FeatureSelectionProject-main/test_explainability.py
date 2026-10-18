"""Unit tests for Recommendation & Explainability page logic."""

import unittest
import pandas as pd
import numpy as np

from app import compute_pareto_dominance, load_benchmark_recommendations, load_benchmark_comparison
from recommendation import recommend_methods


class ExplainabilityTests(unittest.TestCase):
    def test_benchmark_colon_cancer_recommendation(self):
        comp = load_benchmark_comparison()
        colon_comp = comp[comp["Dataset"] == "Colon Cancer"]
        recs = recommend_methods(colon_comp)
        self.assertFalse(recs.empty)
        row = recs.iloc[0]
        self.assertEqual(row["Recommended Method"], "Hybrid")
        self.assertEqual(row["Best Predictive Method(s)"], "Hybrid")
        self.assertAlmostEqual(row["Accuracy"], 0.9231, places=4)
        self.assertAlmostEqual(row["MCC"], 0.8433, places=4)
        self.assertEqual(row["Selected Features"], 4)

        # Dominance proofs verification
        dom_df = compute_pareto_dominance(colon_comp)
        hybrid_dom = dom_df[dom_df["Method"] == "Hybrid"].iloc[0]
        self.assertTrue(hybrid_dom["Is Pareto"])
        self.assertEqual(hybrid_dom["Dominating Count"], 0)

        # CFS, GA, FCBF, FAST, PSO should all be dominated
        dominated = dom_df[~dom_df["Is Pareto"]]
        self.assertEqual(len(dominated), 5)
        for _, d_row in dominated.iterrows():
            self.assertIn("Hybrid", d_row["Dominated By"])
            self.assertTrue(len(d_row["Dominance Proofs"]) > 0)

    def test_benchmark_leukemia_recommendation(self):
        comp = load_benchmark_comparison()
        leuk_comp = comp[comp["Dataset"] == "Leukemia"]
        recs = recommend_methods(leuk_comp)
        self.assertFalse(recs.empty)
        row = recs.iloc[0]
        self.assertEqual(row["Recommended Method"], "Hybrid")
        self.assertEqual(row["Best Predictive Method(s)"], "CFS, Hybrid")
        self.assertAlmostEqual(row["Accuracy"], 1.0, places=4)
        self.assertAlmostEqual(row["MCC"], 1.0, places=4)
        self.assertEqual(row["Selected Features"], 3)

        # Both CFS and Hybrid should be on Pareto frontier
        dom_df = compute_pareto_dominance(leuk_comp)
        pareto_methods = sorted(dom_df[dom_df["Is Pareto"]]["Method"].tolist())
        self.assertEqual(pareto_methods, ["CFS", "Hybrid"])

        # CFS selected 5 features, Hybrid selected 3 features -> Hybrid wins Occam's razor
        cfs_row = dom_df[dom_df["Method"] == "CFS"].iloc[0]
        hybrid_row = dom_df[dom_df["Method"] == "Hybrid"].iloc[0]
        self.assertEqual(cfs_row["Selected Features"], 5)
        self.assertEqual(hybrid_row["Selected Features"], 3)
        self.assertLess(hybrid_row["Selected Features"], cfs_row["Selected Features"])

    def test_pareto_dominance_proof_string_generation(self):
        sample_results = pd.DataFrame([
            {
                "Method": "Alpha",
                "Accuracy": 0.90,
                "MCC": 0.80,
                "Selected Features": 5,
                "Total Features": 100,
                "Feature Reduction (%)": 95.0,
                "Runtime (s)": 1.2,
            },
            {
                "Method": "Beta",
                "Accuracy": 0.70,
                "MCC": 0.50,
                "Selected Features": 2,
                "Total Features": 100,
                "Feature Reduction (%)": 98.0,
                "Runtime (s)": 0.4,
            },
        ])
        dom_df = compute_pareto_dominance(sample_results)
        beta_row = dom_df[dom_df["Method"] == "Beta"].iloc[0]
        self.assertFalse(beta_row["Is Pareto"])
        self.assertEqual(beta_row["Dominated By"], "Alpha")
        self.assertTrue(len(beta_row["Dominance Proofs"]) == 1)
        self.assertIn("Alpha (Acc: 90.00%, MCC: 0.8000) dominates Beta", beta_row["Dominance Proofs"][0])

    def test_compactness_tie_breaking_by_runtime(self):
        # Two methods tied on Pareto efficiency and feature count, broken by runtime
        sample_results = pd.DataFrame([
            {
                "Dataset": "Test",
                "Method": "Method_Slow",
                "Accuracy": 0.95,
                "MCC": 0.90,
                "Selected Features": 4,
                "Total Features": 50,
                "Feature Reduction (%)": 92.0,
                "Runtime (s)": 5.4,
            },
            {
                "Dataset": "Test",
                "Method": "Method_Fast",
                "Accuracy": 0.95,
                "MCC": 0.90,
                "Selected Features": 4,
                "Total Features": 50,
                "Feature Reduction (%)": 92.0,
                "Runtime (s)": 1.1,
            },
        ])
        recs = recommend_methods(sample_results)
        self.assertEqual(recs.iloc[0]["Recommended Method"], "Method_Fast")
        self.assertIn("Method_Fast", recs.iloc[0]["Reason"])

    def test_uploaded_csv_recommendation_and_explainability(self):
        from upload_recommendation import evaluate_uploaded_dataset
        df = pd.read_csv("test_datasets/valid_sample.csv")
        eval_outcome = evaluate_uploaded_dataset(df, "diagnosis", "Valid Sample")
        
        results = eval_outcome["results"]
        recs = eval_outcome["recommendations"]
        self.assertFalse(results.empty)
        self.assertFalse(recs.empty)
        
        # Verify recommended method is present and has proper fields
        rec_row = recs.iloc[0]
        self.assertIn("Recommended Method", rec_row)
        self.assertIn(rec_row["Recommended Method"], ["FCBF", "FAST", "GA", "PSO", "Hybrid"])
        self.assertEqual(rec_row["Accuracy"], 1.0)
        self.assertEqual(rec_row["MCC"], 1.0)
        self.assertEqual(rec_row["Selected Features"], 1)
        
        # Verify Pareto dominance computation on uploaded results
        dom_df = compute_pareto_dominance(results)
        self.assertEqual(len(dom_df), 6)
        # All methods achieve 1.0/1.0, so none are dominated on predictive performance
        self.assertTrue(all(dom_df["Is Pareto"]))
        
        # Verify CFS selected 2 features while others selected 1
        cfs_features = dom_df[dom_df["Method"] == "CFS"]["Selected Features"].iloc[0]
        self.assertEqual(cfs_features, 2)
        rec_features = dom_df[dom_df["Method"] == rec_row["Recommended Method"]]["Selected Features"].iloc[0]
        self.assertEqual(rec_features, 1)

    def test_empty_dataframe_safe_handling(self):
        empty_dom = compute_pareto_dominance(pd.DataFrame())
        self.assertTrue(empty_dom.empty)
        none_dom = compute_pareto_dominance(None)
        self.assertTrue(none_dom.empty)


if __name__ == "__main__":
    unittest.main()

