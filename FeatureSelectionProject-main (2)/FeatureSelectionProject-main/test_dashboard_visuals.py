"""Unit tests for dashboard visual presentation components (Bank Marketing, ML Metrics, OmniXAI)."""

import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard_visuals import (
    render_speedometer_gauge,
    render_mini_metric_bar,
    render_correlation_heatmap,
    render_violin_distribution,
    render_feature_consensus_diverging_bar,
    render_executive_banner_html,
    render_throughput_area_chart,
    render_omnixai_predict_bar,
    render_omnixai_lime_bar,
)


class TestDashboardVisuals(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "Method": ["CFS", "FCBF", "FAST", "GA", "PSO", "Hybrid"],
            "Accuracy": [0.85, 0.77, 0.69, 0.62, 0.69, 0.85],
            "MCC": [0.77, 0.65, 0.45, 0.35, 0.45, 0.77],
            "Feature Reduction (%)": [99.75, 99.90, 99.85, 99.80, 99.80, 99.75],
            "Selected Features": [5, 2, 3, 4, 4, 5],
            "Runtime (s)": [0.014, 0.008, 0.025, 2.15, 1.45, 0.095],
        })

    def test_render_speedometer_gauge(self):
        fig = render_speedometer_gauge(
            value=0.85,
            title="Model score (Holdout CV)",
            min_val=0.0,
            max_val=1.0,
            is_percentage=True,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.data[0].type, "indicator")
        self.assertEqual(fig.data[0].value, 85.0)

    def test_render_mini_metric_bar(self):
        fig = render_mini_metric_bar(
            self.sample_df,
            metric_col="Accuracy",
            title="test__accuracy",
            is_percentage=True,
            path_subtitle="Path: metrics.csv",
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(len(fig.data[0].x), 6)

    def test_render_correlation_heatmap(self):
        X = pd.DataFrame(np.random.randn(20, 5), columns=[f"Feature_{i}" for i in range(5)])
        fig = render_correlation_heatmap(X, selected_features=["Feature_0", "Feature_1", "Feature_2"])
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_render_correlation_heatmap_insufficient_features(self):
        X = pd.DataFrame(np.random.randn(20, 5), columns=[f"Feature_{i}" for i in range(5)])
        fig = render_correlation_heatmap(X, selected_features=["Feature_0"])
        self.assertIsNone(fig)

    def test_render_violin_distribution(self):
        X = pd.DataFrame({"Gene_A": np.random.randn(30), "Gene_B": np.random.randn(30)})
        y = pd.Series([0, 1] * 15)
        fig = render_violin_distribution(X, y, feature_name="Gene_A")
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_render_feature_consensus_diverging_bar(self):
        freq = pd.Series({"Feature_A": 6, "Feature_B": 4, "Feature_C": 2, "Feature_D": 1})
        fig = render_feature_consensus_diverging_bar(freq, top_n=4)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_render_executive_banner_html(self):
        html = render_executive_banner_html(
            dataset_name="UCI Colon Cancer Dataset",
            num_samples=62,
            num_features=2000,
            best_acc=0.85,
            best_mcc=0.77,
            reduction_pct=99.75,
            num_selected=5,
            source_type="CSV (Benchmark)",
        )
        self.assertIn("UCI Colon Cancer Dataset - Score", html)
        self.assertIn("TRAINED ON", html)
        self.assertIn("Approved", html)
        self.assertIn("Pareto-Optimal", html)

    def test_render_throughput_area_chart(self):
        fig = render_throughput_area_chart()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)

    def test_render_omnixai_predict_bar(self):
        fig = render_omnixai_predict_bar()
        self.assertIsInstance(fig, go.Figure)

    def test_render_omnixai_lime_bar(self):
        fig = render_omnixai_lime_bar()
        self.assertIsInstance(fig, go.Figure)

    def test_resolve_feature_column(self):
        from dashboard_visuals import resolve_feature_column
        df_int = pd.DataFrame(np.random.randn(10, 3), columns=[1, 2, 3])
        self.assertEqual(resolve_feature_column(df_int, "1"), 1)
        self.assertEqual(resolve_feature_column(df_int, 2), 2)
        self.assertIsNone(resolve_feature_column(df_int, "99"))

    def test_render_correlation_heatmap_integer_columns(self):
        X = pd.DataFrame(np.random.randn(20, 5), columns=[1, 2, 3, 4, 5])
        fig = render_correlation_heatmap(X, selected_features=["1", "2", "3"])
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()

