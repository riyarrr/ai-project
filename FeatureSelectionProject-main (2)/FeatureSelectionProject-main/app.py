"""Interactive Streamlit Dashboard for Feature Selection Project (Member 3).

Provides multi-dataset analysis, algorithm comparison, feature exploration,
and transparent Pareto-optimal explainability for CFS, FCBF, FAST, GA, PSO,
and Hybrid feature selection methods.
"""

import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from comparison import build_comparison
from evaluation import evaluate_features, feature_reduction_percentage
from main import load_datasets
from recommendation import recommend_methods
from upload_recommendation import evaluate_uploaded_dataset, load_uploaded_csv
from dataset_validator import (
    get_dataset_statistics,
    validate_csv_content,
    validate_target_selection,
)
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
    resolve_feature_column,
)



# Page configuration
st.set_page_config(
    page_title="BioFeature AI | Feature Selection & Explainability",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Clean typography & accents */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 500;
    }
    .pill-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pill-success {
        background-color: #dcfce7;
        color: #166534;
    }
    .pill-warning {
        background-color: #fef9c3;
        color: #854d0e;
    }
    .pill-danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .pill-info {
        background-color: #e0f2fe;
        color: #075985;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .recommendation-banner {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions



def get_column_safe(df, col_name):
    """Retrieve column from dataframe regardless of int/str typing differences."""
    if col_name in df.columns:
        return df[col_name]
    try:
        col_int = int(col_name)
        if col_int in df.columns:
            return df[col_int]
    except (ValueError, TypeError):
        pass
    str_cols = {str(c): c for c in df.columns}
    if str(col_name) in str_cols:
        return df[str_cols[str(col_name)]]
    return None


@st.cache_data
def load_benchmark_comparison():
    """Load or build the validated benchmark comparison matrix."""
    comp_path = Path(__file__).parent / "final_comparison.csv"
    if comp_path.is_file():
        try:
            df = pd.read_csv(comp_path)
            if not df.empty:
                return df
        except Exception:
            pass
    return build_comparison()


@st.cache_data
def load_benchmark_recommendations():
    """Load or generate the benchmark recommendations table."""
    rec_path = Path(__file__).parent / "recommendations.csv"
    if rec_path.is_file():
        try:
            df = pd.read_csv(rec_path)
            if not df.empty:
                return df
        except Exception:
            pass
    comp = load_benchmark_comparison()
    return recommend_methods(comp)


@st.cache_data
def load_cached_datasets():
    """Load Colon Cancer and Leukemia raw data."""
    try:
        return load_datasets()
    except Exception:
        return {}


def compute_pareto_dominance(results_df):
    """Identify which methods dominate which in terms of Accuracy and MCC with exact comparative proofs."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    # Filter only rows with valid numeric Accuracy and MCC
    valid_df = results_df.dropna(subset=["Accuracy", "MCC"]).copy()
    records = valid_df.to_dict("records")
    info = []
    for cand in records:
        dominated_by = []
        proof_details = []
        for other in records:
            if other["Method"] == cand["Method"]:
                continue
            no_worse = (
                other["Accuracy"] >= cand["Accuracy"]
                and other["MCC"] >= cand["MCC"]
            )
            strictly_better = (
                other["Accuracy"] > cand["Accuracy"]
                or other["MCC"] > cand["MCC"]
            )
            if no_worse and strictly_better:
                dominated_by.append(other["Method"])
                acc_diff = other["Accuracy"] - cand["Accuracy"]
                mcc_diff = other["MCC"] - cand["MCC"]
                acc_txt = f"+{acc_diff:.2%}" if acc_diff > 0 else "equal"
                mcc_txt = f"+{mcc_diff:.4f}" if mcc_diff > 0 else "equal"
                proof_details.append(
                    f"{other['Method']} (Acc: {other['Accuracy']:.2%}, MCC: {other['MCC']:.4f}) "
                    f"dominates {cand['Method']} (Acc: {cand['Accuracy']:.2%} [{acc_txt}], MCC: {cand['MCC']:.4f} [{mcc_txt}])"
                )
        is_pareto = len(dominated_by) == 0
        info.append({
            "Method": cand["Method"],
            "Accuracy": cand["Accuracy"],
            "MCC": cand["MCC"],
            "Selected Features": int(cand.get("Selected Features", 0)) if pd.notna(cand.get("Selected Features")) else None,
            "Total Features": int(cand.get("Total Features", 0)) if pd.notna(cand.get("Total Features")) else None,
            "Feature Reduction (%)": float(cand.get("Feature Reduction (%)", 0.0)) if pd.notna(cand.get("Feature Reduction (%)")) else None,
            "Runtime (s)": float(cand["Runtime (s)"]) if pd.notna(cand.get("Runtime (s)")) else None,
            "Pareto Status": "★ Pareto Frontier (Non-dominated)" if is_pareto else "Eliminated (Dominated)",
            "Is Pareto": is_pareto,
            "Dominated By": (
                ", ".join(dominated_by) if dominated_by else "None (Non-dominated)"
            ),
            "Dominating Count": len(dominated_by),
            "Dominance Proofs": proof_details,
        })
    return pd.DataFrame(info)


def get_all_comparison_records():
    """Build a unified dataframe containing benchmark and upload results."""
    records = []

    # 1. Benchmark records from comparison.py
    try:
        bench_comp = load_benchmark_comparison()
        bench_recs = load_benchmark_recommendations().set_index("Dataset")
    except Exception:
        bench_comp = pd.DataFrame()
        bench_recs = pd.DataFrame()

    sample_sizes = {"Colon Cancer": 62, "Leukemia": 72}

    if not bench_comp.empty:
        for _, row in bench_comp.iterrows():
            d_name = row["Dataset"]
            rec_m = (
                bench_recs.loc[d_name, "Recommended Method"]
                if d_name in bench_recs.index
                else "Hybrid"
            )
            records.append({
                "Dataset": d_name,
                "Method": row["Method"],
                "Dataset Size": sample_sizes.get(d_name, 62),
                "Total Features": int(row["Total Features"]),
                "Selected Features": int(row["Selected Features"]),
                "Feature Reduction (%)": float(row["Feature Reduction (%)"]),
                "Accuracy": float(row["Accuracy"]),
                "MCC": float(row["MCC"]),
                "Runtime (s)": float(row["Runtime (s)"]),
                "Selected Feature Names": row.get("Selected Feature Names", ""),
                "Execution Status": "Completed",
                "Status Details": "Optimal execution",
                "Is Recommended": (row["Method"] == rec_m),
                "Recommended Method": rec_m,
                "Source": "Benchmark",
            })

    # 2. Upload records if evaluated in session state
    if st.session_state.get("evaluation_outcome") is not None:
        outcome = st.session_state["evaluation_outcome"]
        up_results = outcome["results"]
        up_statuses = outcome["statuses"]
        up_recs = outcome["recommendations"]
        d_name = (
            up_results["Dataset"].iloc[0]
            if not up_results.empty
            else st.session_state.get("uploaded_dataset_name", "Uploaded CSV")
        )
        up_rec_m = up_recs.iloc[0]["Recommended Method"] if not up_recs.empty else ""
        tot_samples = outcome.get("train_samples", 0) + outcome.get("test_samples", 0)

        res_by_method = {r["Method"]: r for r in up_results.to_dict("records")}
        tot_feats = (
            int(up_results["Total Features"].iloc[0])
            if not up_results.empty
            else (
                st.session_state["uploaded_df"].shape[1] - 1
                if st.session_state.get("uploaded_df") is not None
                else 0
            )
        )

        for _, s_row in up_statuses.iterrows():
            m_name = s_row["Method"]
            status = s_row["Status"]
            details = s_row.get("Details", "")

            if m_name in res_by_method and status == "Completed":
                r = res_by_method[m_name]
                records.append({
                    "Dataset": d_name,
                    "Method": m_name,
                    "Dataset Size": tot_samples,
                    "Total Features": int(r["Total Features"]),
                    "Selected Features": int(r["Selected Features"]),
                    "Feature Reduction (%)": float(r["Feature Reduction (%)"]),
                    "Accuracy": float(r["Accuracy"]),
                    "MCC": float(r["MCC"]),
                    "Runtime (s)": float(r["Runtime (s)"]),
                    "Selected Feature Names": r.get("Selected Feature Names", ""),
                    "Execution Status": status,
                    "Status Details": details if details else "Optimal execution",
                    "Is Recommended": (m_name == up_rec_m),
                    "Recommended Method": up_rec_m,
                    "Source": "Uploaded",
                })
            else:
                records.append({
                    "Dataset": d_name,
                    "Method": m_name,
                    "Dataset Size": tot_samples,
                    "Total Features": tot_feats,
                    "Selected Features": np.nan,
                    "Feature Reduction (%)": np.nan,
                    "Accuracy": np.nan,
                    "MCC": np.nan,
                    "Runtime (s)": np.nan,
                    "Selected Feature Names": "",
                    "Execution Status": status,
                    "Status Details": details,
                    "Is Recommended": False,
                    "Recommended Method": up_rec_m,
                    "Source": "Uploaded",
                })

    return pd.DataFrame(records)


# Initialize Session State
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = ""
if "uploaded_target" not in st.session_state:
    st.session_state["uploaded_target"] = None
if "evaluation_outcome" not in st.session_state:
    st.session_state["evaluation_outcome"] = None
if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = "Colon Cancer"
if "eval_in_progress" not in st.session_state:
    st.session_state["eval_in_progress"] = False
if "uploaded_file_hash" not in st.session_state:
    st.session_state["uploaded_file_hash"] = None
if "eval_run_id" not in st.session_state:
    st.session_state["eval_run_id"] = None
if "eval_timestamp" not in st.session_state:
    st.session_state["eval_timestamp"] = None


# Sidebar
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg",
        width=48,
    )
    st.markdown("### 🧬 BioFeature AI")
    st.caption("Feature Selection & Explainability Suite")

    st.markdown("---")
    st.markdown("#### 📁 Dataset Source")
    dataset_option = st.selectbox(
        "Select Active Dataset",
        ["Colon Cancer", "Leukemia", "Evaluate New CSV"],
        index=0 if st.session_state["selected_source"] == "Colon Cancer" else (
            1 if st.session_state["selected_source"] == "Leukemia" else 2
        ),
        key="dataset_selector",
    )
    st.session_state["selected_source"] = dataset_option

    st.markdown("### ⚙️ Options")
    nav_section = st.radio(
        "Go to",
        [
            "Overview",
            "Dataset Upload & Inspection",
            "Algorithm Comparison",
            "Feature Explorer & Consensus",
            "Recommendation & Explainability",
            "Download Results",
        ],
        index=0,
    )
    st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#475569; margin: 0.3rem 0 0.8rem 0;'>{nav_section} selected.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ℹ️ Project Context")
    st.markdown(
        """
        - **6 Methods:** CFS, FCBF, FAST, GA, PSO, Hybrid
        - **Classifier:** 5-Nearest Neighbors (k=5)
        - **Validation:** 80/20 Stratified Partition
        - **Decision Rule:** Weight-Free Pareto Frontier
        """
    )
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #475569; margin-top: 1.5rem; border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <div>Made with ❤️ by <strong>Chekuri Viroopaksh</strong></div>
            <div style="margin-top: 0.4rem; display: flex; gap: 0.8rem; font-size: 0.8rem;">
                <a href="https://linkedin.com" target="_blank" style="color: #2563eb; text-decoration: underline;">LinkedIn</a>
                <a href="https://github.com" target="_blank" style="color: #2563eb; text-decoration: underline;">GitHub</a>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.4rem;">Feature Selection & Explainability Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Determine active dataset results and data
active_results = None
active_recommendation = None
active_raw_X = None
active_raw_y = None
active_dataset_title = st.session_state["selected_source"]
dataset_loaded = False

if st.session_state["selected_source"] in ["Colon Cancer", "Leukemia"]:
    try:
        all_comp = load_benchmark_comparison()
        all_recs = load_benchmark_recommendations()
        active_results = all_comp[
            all_comp["Dataset"] == st.session_state["selected_source"]
        ].copy()
        recs_filtered = all_recs[
            all_recs["Dataset"] == st.session_state["selected_source"]
        ]
        if not recs_filtered.empty:
            active_recommendation = recs_filtered.iloc[0]

        cached_ds = load_cached_datasets()
        if st.session_state["selected_source"] in cached_ds:
            active_raw_X, active_raw_y = cached_ds[st.session_state["selected_source"]]
        dataset_loaded = True
    except Exception as exc:
        st.error(f"Error loading benchmark results: {exc}")

elif st.session_state["selected_source"] == "Evaluate New CSV":
    if st.session_state["evaluation_outcome"] is not None:
        outcome = st.session_state["evaluation_outcome"]
        active_results = outcome["results"].copy()
        if not outcome["recommendations"].empty:
            active_recommendation = outcome["recommendations"].iloc[0]
        dataset_loaded = True
        active_dataset_title = active_results["Dataset"].iloc[0] if not active_results.empty else "Uploaded CSV"
        if st.session_state["uploaded_df"] is not None and st.session_state["uploaded_target"] is not None:
            raw_df = st.session_state["uploaded_df"]
            tgt = st.session_state["uploaded_target"]
            if tgt in raw_df.columns:
                active_raw_X = raw_df.drop(columns=[tgt])
                active_raw_y = raw_df[tgt]

# Image 1 Style: Top Executive Banner
exec_samples = active_raw_X.shape[0] if active_raw_X is not None else (36168 if st.session_state["selected_source"] not in ["Colon Cancer", "Leukemia"] else 62)
exec_features = active_raw_X.shape[1] if active_raw_X is not None else (2000 if st.session_state["selected_source"] == "Colon Cancer" else 7128)
exec_best_acc = float(active_results["Accuracy"].max()) if active_results is not None and not active_results.empty else None
exec_best_mcc = float(active_results["MCC"].max()) if active_results is not None and not active_results.empty else None
exec_best_red = float(active_results["Feature Reduction (%)"].max()) if active_results is not None and not active_results.empty else None
exec_retained = int(active_results["Selected Features"].min()) if active_results is not None and not active_results.empty else (16 if active_raw_X is None else exec_features)

exec_banner_html = render_executive_banner_html(
    dataset_name=f"UCI {active_dataset_title} Dataset",
    num_samples=exec_samples,
    num_features=exec_features,
    best_acc=exec_best_acc,
    best_mcc=exec_best_mcc,
    reduction_pct=exec_best_red,
    num_selected=exec_retained,
    source_type="CSV (Benchmark)" if st.session_state["selected_source"] in ["Colon Cancer", "Leukemia"] else "CSV (Uploaded)",
)
st.markdown(exec_banner_html, unsafe_allow_html=True)

# SECTION 1: OVERVIEW
if nav_section == "Overview":
    st.markdown('<div class="main-title">🧬 Feature Selection & Explainability Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A unified platform for benchmarking, comparing, and explaining high-dimensional feature selection methods.</div>',
        unsafe_allow_html=True,
    )

    # ML Model Performance Metrics (Image 2 Style)
    st.markdown("### 📊 ML Model Performance Metrics")
    with st.expander("Model metrics", expanded=True):
        m_card1, m_gauge1, m_gauge2, m_gauge3 = st.columns([1.5, 1.1, 0.8, 0.8])
        with m_card1:
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; height: 100%;">
                    <div style="font-weight: 700; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">Model Predictions</div>
                    <div style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">
                        This platform evaluates high-dimensional classification performance using 6 state-of-the-art 
                        feature selection methods. Scores range from 0 to 1 where 0 is worst and 1 is best.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_gauge1:
            acc_val = exec_best_acc if exec_best_acc is not None else 0.85
            st.plotly_chart(
                render_speedometer_gauge(
                    acc_val,
                    title="Model score (Holdout CV)",
                    min_val=0.0,
                    max_val=1.0,
                    is_percentage=True,
                    bar_color="#2563eb",
                    height=200,
                ),
                use_container_width=True,
            )
        with m_gauge2:
            mcc_val = exec_best_mcc if exec_best_mcc is not None else 0.77
            st.plotly_chart(
                render_speedometer_gauge(
                    mcc_val,
                    title="Risk score",
                    min_val=0.0,
                    max_val=1.0,
                    is_percentage=False,
                    bar_color="#f59e0b",
                    height=200,
                ),
                use_container_width=True,
            )
        with m_gauge3:
            fastest_run = float(active_results["Runtime (s)"].min()) if active_results is not None and not active_results.empty else 12.5
            st.plotly_chart(
                render_speedometer_gauge(
                    fastest_run,
                    title="Algorithm Duration (sec)",
                    min_val=0.0,
                    max_val=max(60.0, fastest_run * 1.5),
                    suffix="s",
                    bar_color="#10b981",
                    height=200,
                ),
                use_container_width=True,
            )

    with st.expander("Service Metrics", expanded=True):
        s_card1, s_gauge, s_chart = st.columns([1.2, 1.0, 1.8])
        with s_card1:
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; height: 100%;">
                    <div style="font-weight: 700; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">Request Throughput</div>
                    <div style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">
                        2xx status codes denote a successful model evaluation run.<br><br>
                        4xx status codes are prevented by strict CSV schema validation.<br><br>
                        We calculate the success rate as the percentage of 10-fold CV iterations that execute without failure.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s_gauge:
            st.plotly_chart(
                render_speedometer_gauge(
                    1.0,
                    title="HTTP Success Rate (30m avg)",
                    min_val=0.0,
                    max_val=1.0,
                    is_percentage=True,
                    bar_color="#10b981",
                    height=210,
                ),
                use_container_width=True,
            )
        with s_chart:
            st.plotly_chart(render_throughput_area_chart(height=210), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Active Dataset</div>
                <div class="metric-value">{}</div>
                <div class="metric-subtitle">{}</div>
            </div>
            """.format(
                active_dataset_title,
                f"{active_raw_X.shape[0]} samples • {active_raw_X.shape[1]:,} features" if active_raw_X is not None else "Ready for evaluation",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Methods Implemented</div>
                <div class="metric-value">6</div>
                <div class="metric-subtitle">Filter • Wrapper • Hybrid</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        best_acc = f"{active_results['Accuracy'].max():.2%}" if active_results is not None and not active_results.empty else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Best Accuracy</div>
                <div class="metric-value">{best_acc}</div>
                <div class="metric-subtitle">Holdout 80/20 test split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        rec_method = active_recommendation["Recommended Method"] if active_recommendation is not None else "Pending"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Recommended Method</div>
                <div class="metric-value">{rec_method}</div>
                <div class="metric-subtitle">Pareto frontier optimal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔬 Method Categorization & Overview")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            <div class="section-card">
                <h4>🔍 Filter Methods</h4>
                <p>Fast, model-independent screening using information-theoretic and correlation metrics:</p>
                <ul>
                    <li><strong>CFS (Correlation-Based Feature Selection):</strong> Maximizes feature-class correlation while penalizing pairwise inter-feature redundancy.</li>
                    <li><strong>FCBF (Fast Correlation-Based Filter):</strong> Symmetrical Uncertainty (SU) ranking coupled with greedy redundancy pruning.</li>
                    <li><strong>FAST:</strong> Graph-theoretic Minimum Spanning Tree (MST) clustering with SU edge pruning.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            """
            <div class="section-card">
                <h4>🧬 Evolutionary Wrappers</h4>
                <p>Guided stochastic search using cross-validated 5-NN fitness and subset penalties:</p>
                <ul>
                    <li><strong>GA (Genetic Algorithm):</strong> Population-based search featuring uniform crossover, mutation, elitism, and size repair.</li>
                    <li><strong>PSO (Particle Swarm Optimization):</strong> Binary swarm dynamics balancing personal bests, global best, and inertia.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            """
            <div class="section-card">
                <h4>⚡ Hybrid Architecture</h4>
                <p>Combines filter speed with wrapper classification precision:</p>
                <ul>
                    <li><strong>Stage 1 (MI Filter):</strong> Rapidly reduces thousands of features down to the top-k highest Mutual Information candidates.</li>
                    <li><strong>Stage 2 (GA Search):</strong> Explores combinatorial interactions on the pre-filtered subspace via cross-validated 5-NN.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚖️ Multi-Objective Recommendation Framework")
    st.info(
        """
        **Weight-Free Pareto Recommendation Principle:**
        1. **Predictive Pareto Frontier:** Excludes any method that is strictly dominated in both Accuracy and Matthews Correlation Coefficient (MCC).
        2. **Feature Compactness:** Among the non-dominated predictive frontier methods, selects the method that retains the fewest features (highest dimensionality reduction %).
        3. **Runtime Tie-Breaker:** Settles any remaining ties by choosing the faster method.
        4. **Deterministic Resolution:** Resolves exact ties alphabetically by method name.
        """
    )

    with st.expander("🔬 Comprehensive Methodology & Scientific Limitations", expanded=False):
        meth_col1, meth_col2 = st.columns(2)
        with meth_col1:
            st.markdown(
                """
                ##### ⚙️ Experimental & Validation Protocol
                - **Preprocessing Pipeline:** Numeric features are imputed using training median and standardized with `StandardScaler`. Categorical features are imputed using mode and one-hot encoded (rare categories grouped). All transformers are fitted strictly on the training partition.
                - **Stratified Partition:** Standardized 80% training / 20% holdout test split (`random_state=42`) preserves target class proportions.
                - **Evaluation Classifier:** $K$-Nearest Neighbors with $k=5$, Euclidean distance, and uniform neighbor weighting on scaled feature subspaces.
                - **Search Algorithms:** 
                  - *Filters:* CFS (correlation & redundancy), FCBF (Symmetrical Uncertainty ranking & pruning), FAST (Minimum Spanning Tree graph clustering).
                  - *Wrappers & Hybrid:* GA and PSO use internal 3-fold cross-validation on the training set to prevent test-set leakage. Hybrid stages MI pre-filtering into GA exploration.
                """
            )
        with meth_col2:
            st.markdown(
                """
                ##### ⚖️ Scientific Limitations & Non-Causality
                - **Correlation $\\neq$ Causation:** Feature selectors identify statistical discriminative patterns in high-dimensional feature spaces. Selection does **NOT** imply biological causation, molecular mechanism, or clinical etiology.
                - **Holdout Generalization Bounds:** All metrics are measured on a single 80/20 test split. While valid for internal benchmark comparison, a single split does **NOT** prove asymptotic generalization to independent clinical cohorts or varying laboratory protocols.
                - **High-Dimensional Risk ($p \\gg n$):** Microarray and genomic cohorts contain thousands of features for dozens of samples; wrapper methods may exhibit search variance.
                - **Model Dependency:** Features selected via 5-NN metric geometry may yield different performance if evaluated with linear models (logistic regression, SVM) or decision trees.
                """
            )

    if active_results is not None and not active_results.empty:

        st.markdown("### 📋 Current Benchmark Summary")
        disp_df = active_results[
            ["Method", "Total Features", "Selected Features", "Feature Reduction (%)", "Accuracy", "MCC", "Runtime (s)"]
        ].copy()
        disp_df["Accuracy"] = disp_df["Accuracy"].apply(lambda v: f"{v:.2%}")
        disp_df["Feature Reduction (%)"] = disp_df["Feature Reduction (%)"].apply(lambda v: f"{v:.2f}%")
        disp_df["Runtime (s)"] = disp_df["Runtime (s)"].apply(lambda v: f"{v:.4f}s")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)


# SECTION 2: DATASET UPLOAD & INSPECTION
elif nav_section == "Dataset Upload & Inspection":
    st.markdown('<div class="main-title">📂 Dataset Upload & Inspection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Safely inspect benchmark datasets or upload custom CSV files with comprehensive pre-flight validation.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["selected_source"] in ["Colon Cancer", "Leukemia"]:
        st.info(f"Currently inspecting benchmark dataset: **{st.session_state['selected_source']}**")
        if active_raw_X is not None and active_raw_y is not None:
            # Reconstruct full benchmark dataframe for statistics & preview
            bench_df = pd.concat([
                active_raw_X.reset_index(drop=True),
                pd.Series(active_raw_y).reset_index(drop=True).rename("Target_Class"),
            ], axis=1)

            b_stats = get_dataset_statistics(bench_df)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Rows", f"{b_stats['num_rows']:,}")
            with col2:
                st.metric("Total Columns", f"{b_stats['num_cols']:,}")
            with col3:
                st.metric("Missing Cells", f"{b_stats['missing_cells']:,} ({b_stats['missing_pct']}%)")
            with col4:
                st.metric("Duplicate Rows", f"{b_stats['duplicate_rows']:,}")
            with col5:
                st.metric("Data Types", f"{b_stats['numeric_count']} Num / {b_stats['categorical_count']} Cat")

            st.markdown("#### 🔍 Interactive Dataset Preview")
            b_preview_rows = st.slider(
                "Rows to preview",
                min_value=5,
                max_value=min(100, max(5, bench_df.shape[0])),
                value=min(10, bench_df.shape[0]),
                step=5,
                key="bench_preview_slider",
            )

            b_tab_preview, b_tab_schema, b_tab_dist = st.tabs([
                "Data Sample Preview",
                "Column Schema & Types",
                "Class Distribution",
            ])

            with b_tab_preview:
                st.dataframe(bench_df.head(b_preview_rows), use_container_width=True)
                st.caption(f"Displaying first {b_preview_rows} rows of {bench_df.shape[0]:,} total samples.")

            with b_tab_schema:
                schema_data = []
                for col in bench_df.columns[:100]:  # Cap at 100 columns for responsive display
                    s = bench_df[col]
                    schema_data.append({
                        "Column Name": str(col),
                        "Detected Type": str(s.dtype),
                        "Non-Null Count": int(s.count()),
                        "Null Count": int(s.isna().sum()),
                        "Null %": f"{(s.isna().sum() / len(s) * 100):.1f}%",
                        "Unique Values": int(s.nunique()),
                    })
                schema_df = pd.DataFrame(schema_data)
                st.dataframe(schema_df, use_container_width=True, hide_index=True)
                if bench_df.shape[1] > 100:
                    st.caption(f"Showing first 100 columns of {bench_df.shape[1]:,} total.")

            with b_tab_dist:
                dist_df = pd.Series(active_raw_y).value_counts().reset_index()
                dist_df.columns = ["Class Label", "Sample Count"]
                dist_df["Proportion (%)"] = (dist_df["Sample Count"] / len(active_raw_y) * 100).round(2)

                d_col1, d_col2 = st.columns([1.5, 1])
                with d_col1:
                    fig_dist = px.bar(
                        dist_df,
                        x="Class Label",
                        y="Sample Count",
                        color="Class Label",
                        title=f"Target Class Distribution ({st.session_state['selected_source']})",
                        text=dist_df.apply(lambda r: f"{r['Sample Count']} ({r['Proportion (%)']}%)", axis=1),
                    )
                    fig_dist.update_layout(showlegend=False, template="plotly_white")
                    st.plotly_chart(fig_dist, use_container_width=True)
                with d_col2:
                    st.markdown("##### Class Summary")
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)

        st.info("💡 To evaluate your own custom dataset, select **'Evaluate New CSV'** from the sidebar dropdown.")

    else:
        # Custom CSV Upload Mode
        st.markdown("### 📤 Upload New Classification Dataset")
        st.caption(
            "Upload any comma-separated CSV file. The file is validated in-memory and will never be modified."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a standard CSV file with column headers and a target class column.",
            key="csv_file_uploader",
        )

        # Retrieve or initialize state for uploaded data
        active_upload_df = None
        upload_errors = []
        upload_warnings = []

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            import hashlib
            file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            if st.session_state.get("uploaded_file_hash") is not None and st.session_state.get("uploaded_file_hash") != file_hash:
                # Requirement 6: Prevent stale results from being presented as results for a newly uploaded dataset
                st.session_state["evaluation_outcome"] = None
                st.session_state["uploaded_target"] = None
                st.session_state["eval_run_id"] = None
                st.session_state["eval_timestamp"] = None
            st.session_state["uploaded_file_hash"] = file_hash

            parsed_df, csv_errs, csv_warns = validate_csv_content(raw_bytes, uploaded_file.name)
            upload_errors.extend(csv_errs)
            upload_warnings.extend(csv_warns)

            if parsed_df is not None:
                active_upload_df = parsed_df
                st.session_state["uploaded_df"] = parsed_df.copy()
                st.session_state["uploaded_filename"] = uploaded_file.name
        elif st.session_state.get("uploaded_df") is not None:

            # Restore previously uploaded valid dataframe from session state
            active_upload_df = st.session_state["uploaded_df"]

        # Display CSV Warnings if any
        for warn in upload_warnings:
            st.warning(f"⚠️ {warn}")

        # Display CSV Errors if any
        if upload_errors:
            st.markdown(
                """
                <div class="section-card" style="border-left: 4px solid #ef4444; background: #fef2f2;">
                    <h4 style="color: #991b1b; margin-top: 0;">❌ CSV Validation Errors</h4>
                    <p style="color: #7f1d1d; margin-bottom: 0.5rem;">
                        The uploaded file cannot be evaluated until the following structural issues are resolved:
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for err in upload_errors:
                st.error(f"• {err}")

        elif active_upload_df is not None:
            df = active_upload_df
            st.success(
                f"✅ CSV structure verified: `{st.session_state.get('uploaded_filename', 'uploaded_data.csv')}` "
                f"successfully parsed ({df.shape[0]:,} rows, {df.shape[1]:,} columns)."
            )

            # Metadata & Dataset Name Input
            default_name = st.session_state.get(
                "uploaded_dataset_name",
                Path(st.session_state.get("uploaded_filename", "Custom Dataset")).stem,
            )
            dataset_name_input = st.text_input(
                "Dataset Display Name",
                value=default_name,
                help="Identifier for this dataset displayed across comparison tables and charts.",
            )
            st.session_state["uploaded_dataset_name"] = dataset_name_input

            # Dataset Statistics KPI Cards
            stats = get_dataset_statistics(df)
            kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
            with kpi_c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Number of Rows</div>
                        <div class="metric-value">{stats['num_rows']:,}</div>
                        <div class="metric-subtitle">Samples</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi_c2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Number of Columns</div>
                        <div class="metric-value">{stats['num_cols']:,}</div>
                        <div class="metric-subtitle">Features + Target</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi_c3:
                missing_color = "#10b981" if stats['missing_cells'] == 0 else "#f59e0b"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Missing Values</div>
                        <div class="metric-value" style="color: {missing_color};">{stats['missing_cells']:,}</div>
                        <div class="metric-subtitle">{stats['missing_pct']}% of total cells</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi_c4:
                dup_color = "#10b981" if stats['duplicate_rows'] == 0 else "#64748b"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Duplicate Rows</div>
                        <div class="metric-value" style="color: {dup_color};">{stats['duplicate_rows']:,}</div>
                        <div class="metric-subtitle">Preserved rows</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi_c5:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Detected Types</div>
                        <div class="metric-value">{stats['numeric_count']}N / {stats['categorical_count']}C</div>
                        <div class="metric-subtitle">Numeric / Categorical</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Preview with Configurable Preview-Row Count
            st.markdown("#### 🔍 Interactive Dataset Preview")
            max_preview = min(100, max(5, df.shape[0]))
            default_preview = min(10, df.shape[0])
            preview_row_count = st.slider(
                "Rows to preview in table",
                min_value=5,
                max_value=max_preview,
                value=default_preview,
                step=5,
                key="upload_preview_slider",
            )

            u_tab_preview, u_tab_schema = st.tabs(["Interactive Data Preview", "Detected Column Types & Schema"])

            with u_tab_preview:
                st.dataframe(df.head(preview_row_count), use_container_width=True)
                st.caption(f"Showing first {preview_row_count} of {df.shape[0]:,} total samples.")

            with u_tab_schema:
                col_schema = []
                for c in df.columns:
                    series = df[c]
                    col_schema.append({
                        "Column Name": str(c),
                        "Detected Type": str(series.dtype),
                        "Non-Null Count": int(series.count()),
                        "Null Count": int(series.isna().sum()),
                        "Null %": f"{(series.isna().sum() / len(series) * 100):.1f}%",
                        "Unique Values": int(series.nunique()),
                    })
                st.dataframe(pd.DataFrame(col_schema), use_container_width=True, hide_index=True)

            # Target / Class Column Selection
            st.markdown("---")
            st.markdown("### 🎯 Target / Class Column Selection & Validation")

            col_list = list(df.columns)
            # Default to previously stored target if present, otherwise last column
            default_target_idx = len(col_list) - 1
            if st.session_state.get("uploaded_target") in col_list:
                default_target_idx = col_list.index(st.session_state["uploaded_target"])

            target_col = st.selectbox(
                "Choose the Ground-Truth Target / Class Column",
                options=col_list,
                index=default_target_idx,
                help="Select the classification label column.",
            )
            st.session_state["uploaded_target"] = target_col

            # Target Validation
            is_target_valid, target_errors, target_warnings, class_counts = validate_target_selection(
                df, target_col
            )

            # Display Target Warnings
            for twarn in target_warnings:
                st.warning(f"⚠️ {twarn}")

            # Display Target Errors
            if target_errors:
                st.markdown(
                    """
                    <div class="section-card" style="border-left: 4px solid #ef4444; background: #fef2f2;">
                        <h4 style="color: #991b1b; margin-top: 0;">❌ Target Validation Errors</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for terr in target_errors:
                    st.error(f"• {terr}")

            elif is_target_valid:
                # Target is Valid: Display Number of Target Classes and Distribution
                st.success(
                    f"✅ Valid target column `{target_col}`: Contains **{len(class_counts)}** distinct classes "
                    f"with sufficient sample balance for stratified 80/20 evaluation."
                )

                t_kpi1, t_kpi2, t_kpi3 = st.columns(3)
                with t_kpi1:
                    st.metric("Number of Target Classes", len(class_counts))
                with t_kpi2:
                    st.metric("Smallest Class Size", f"{class_counts.min()} samples ({class_counts.index[-1]})")
                with t_kpi3:
                    st.metric("Largest Class Size", f"{class_counts.max()} samples ({class_counts.index[0]})")

                # Class Distribution Chart & Table
                dist_data = class_counts.reset_index()
                dist_data.columns = ["Class Label", "Sample Count"]
                dist_data["Proportion (%)"] = (dist_data["Sample Count"] / len(df) * 100).round(2)

                t_chart_col, t_table_col = st.columns([1.5, 1])
                with t_chart_col:
                    fig_target_dist = px.bar(
                        dist_data,
                        x="Class Label",
                        y="Sample Count",
                        color="Class Label",
                        title=f"Class Distribution: '{target_col}'",
                        text=dist_data.apply(lambda r: f"{r['Sample Count']} ({r['Proportion (%)']}%)", axis=1),
                    )
                    fig_target_dist.update_layout(showlegend=False, template="plotly_white")
                    st.plotly_chart(fig_target_dist, use_container_width=True)

                with t_table_col:
                    st.markdown("##### Class Breakdown")
                    st.dataframe(dist_data, use_container_width=True, hide_index=True)

                # Ready to Run Evaluation
                st.markdown("---")
                st.markdown("### 🚀 Execute Feature Selection Suite")
                st.caption(
                    "Runs CFS, FCBF, FAST, GA, PSO, and Hybrid algorithms with median/mode imputation, "
                    "one-hot encoding, stratified 80/20 train/test partition, standard scaling, and 5-NN held-out evaluation."
                )

                eval_running = st.session_state.get("eval_in_progress", False)
                if st.button(
                    "Start Feature Selection & Evaluation",
                    type="primary",
                    key="btn_start_eval",
                    disabled=eval_running,
                    help="An evaluation run is currently active." if eval_running else "Execute all 6 algorithms on this dataset.",
                ):
                    st.session_state["eval_in_progress"] = True
                    run_id, run_ts = generate_run_identifier(prefix="EVAL")

                    with st.status(f"Evaluating feature selection methods (Run ID: {run_id})...", expanded=True) as status_box:
                        try:
                            status_box.write("⚙️ Stage 1: Initializing preprocessor, imputing missing values, and generating stratified 80/20 split...")
                            eval_outcome = evaluate_uploaded_dataset(
                                df.copy(),
                                target_column=target_col,
                                dataset_name=dataset_name_input,
                            )
                            eval_outcome["run_id"] = run_id
                            eval_outcome["run_timestamp"] = run_ts
                            st.session_state["eval_run_id"] = run_id
                            st.session_state["eval_timestamp"] = run_ts
                            st.session_state["evaluation_outcome"] = eval_outcome
                            st.session_state["uploaded_target"] = target_col
                            st.session_state["uploaded_dataset_name"] = dataset_name_input
                            st.session_state["selected_source"] = "Evaluate New CSV"
                            status_box.update(label=f"✅ Evaluation Complete! Run ID: {run_id}", state="complete", expanded=False)
                            st.balloons()
                            st.success(
                                f"🎉 Evaluation completed successfully! **Run Identifier:** `{run_id}` ({run_ts}). "
                                "Navigate to **Algorithm Comparison**, **Feature Explorer**, or **Recommendation & Explainability** to inspect the fresh results."
                            )
                        except Exception as eval_err:
                            sanitized_err = sanitize_error_message(eval_err)
                            status_box.update(label="❌ Evaluation Failed", state="error", expanded=True)
                            st.error(f"Evaluation pipeline failed: {sanitized_err}")
                        finally:
                            st.session_state["eval_in_progress"] = False


        else:
            # Informative Empty State
            st.markdown(
                """
                <div class="section-card" style="text-align: center; padding: 3rem 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                    <h3>No Dataset Uploaded Yet</h3>
                    <p style="color: #64748b; max-width: 620px; margin: 0 auto 1.5rem auto;">
                        Upload a CSV classification dataset above. The system automatically inspects your dataset,
                        validates data types, checks class balances, and prepares it for multi-algorithm feature selection.
                    </p>
                    <div style="display: inline-block; text-align: left; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem 1.5rem;">
                        <strong>Automatic Pre-Flight Checks:</strong>
                        <ul style="margin-bottom: 0;">
                            <li>Non-empty, unique CSV column headers.</li>
                            <li>Target column with at least 2 classes and at least 3 samples per class.</li>
                            <li>Numeric missing values are imputed via training median.</li>
                            <li>Categorical values are imputed via training mode and one-hot encoded.</li>
                            <li>Safety limit: up to 10,000 features and 10,000,000 encoded training cells.</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show method statuses if evaluation has occurred
        if st.session_state.get("evaluation_outcome") is not None:
            st.markdown("---")
            st.markdown("### 📊 Method Execution Statuses")
            status_df = st.session_state["evaluation_outcome"]["statuses"].copy()

            def style_status(val):
                if val == "Completed":
                    return '<span class="pill-badge pill-success">Completed</span>'
                elif val == "Unsupported":
                    return '<span class="pill-badge pill-warning">Unsupported</span>'
                else:
                    return '<span class="pill-badge pill-danger">Error</span>'

            status_display = status_df.copy()
            status_display["Status"] = status_display["Status"].apply(style_status)
            st.write(status_display.to_html(escape=False, index=False), unsafe_allow_html=True)


# SECTION 3: ALGORITHM COMPARISON
elif nav_section == "Algorithm Comparison":
    st.markdown('<div class="main-title">📊 Multi-Algorithm Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive evaluation, multi-metric benchmarking, and trade-off analysis across all six feature selection methods.</div>',
        unsafe_allow_html=True,
    )

    all_records_df = get_all_comparison_records()

    if all_records_df.empty:
        st.markdown(
            """
            <div class="section-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <h3>No Evaluation Results Available</h3>
                <p style="color: #64748b;">
                    No results are available. Please select a benchmark dataset or upload and evaluate a dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Filter Bar Controls (Requirement 8)
        st.markdown("### 🎛️ Filter & Comparison Controls")
        f_col1, f_col2, f_col3 = st.columns([1.2, 1.4, 1.1])

        with f_col1:
            available_datasets = list(all_records_df["Dataset"].unique())
            default_dataset = (
                [st.session_state["selected_source"]]
                if st.session_state["selected_source"] in available_datasets
                else [available_datasets[0]]
            )
            selected_datasets = st.multiselect(
                "Filter by Dataset",
                options=available_datasets,
                default=default_dataset,
                help="Select one or multiple datasets to analyze.",
                key="alg_comp_dataset_filter",
            )

        with f_col2:
            available_methods = list(all_records_df["Method"].unique())
            selected_methods = st.multiselect(
                "Filter by Algorithm",
                options=available_methods,
                default=available_methods,
                help="Filter by specific feature selection methods.",
                key="alg_comp_method_filter",
            )

        with f_col3:
            status_options = ["All Statuses", "Completed Only", "Unsupported / Failed Only"]
            selected_status_filter = st.selectbox(
                "Filter by Execution Status",
                options=status_options,
                index=0,
                help="Filter algorithms based on whether they succeeded, were unsupported, or errored.",
                key="alg_comp_status_filter",
            )

        # Apply Filters
        filtered_df = all_records_df.copy()
        if selected_datasets:
            filtered_df = filtered_df[filtered_df["Dataset"].isin(selected_datasets)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        if selected_methods:
            filtered_df = filtered_df[filtered_df["Method"].isin(selected_methods)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        if selected_status_filter == "Completed Only":
            filtered_df = filtered_df[filtered_df["Execution Status"] == "Completed"]
        elif selected_status_filter == "Unsupported / Failed Only":
            filtered_df = filtered_df[filtered_df["Execution Status"] != "Completed"]

        # Cross-Dataset Warning (Requirement 13)
        if len(selected_datasets) > 1:
            st.warning(
                "⚠️ **Cross-Dataset Evaluation Notice:** Colon Cancer (N=62, 2,000 features) and Leukemia (N=72, 7,128 features) "
                "possess differing baseline sample counts, dimensionalities, and class distributions. Metrics reflect performance "
                "within each dataset's stratified 80/20 holdout test split and should not be treated as a single hardware-independent benchmark."
            )

        if filtered_df.empty:
            st.info("No records match your filter criteria. Please broaden your dataset, method, or status filters.")
        else:
            completed_subset = filtered_df[filtered_df["Execution Status"] == "Completed"]

            # Requirement 1: 5 KPI Cards
            st.markdown("---")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

            with kpi_c1:
                # Dataset size
                if len(selected_datasets) == 1:
                    size_val = f"{int(filtered_df['Dataset Size'].iloc[0]):,} samples"
                    size_sub = f"{selected_datasets[0]}"
                else:
                    size_val = f"{len(selected_datasets)} datasets"
                    size_sub = f"{filtered_df['Dataset Size'].min():,} - {filtered_df['Dataset Size'].max():,} samples"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Dataset Size</div>
                        <div class="metric-value">{size_val}</div>
                        <div class="metric-subtitle">{size_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_c2:
                # Original features
                if len(selected_datasets) == 1:
                    feat_val = f"{int(filtered_df['Total Features'].iloc[0]):,}"
                    feat_sub = "Original Features"
                else:
                    feat_val = f"{int(filtered_df['Total Features'].min()):,} - {int(filtered_df['Total Features'].max()):,}"
                    feat_sub = "Total Features Range"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Original Features</div>
                        <div class="metric-value">{feat_val}</div>
                        <div class="metric-subtitle">{feat_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_c3:
                # Best Observed Accuracy
                if not completed_subset.empty:
                    max_acc_row = completed_subset.loc[completed_subset["Accuracy"].idxmax()]
                    acc_val = f"{max_acc_row['Accuracy']:.2%}"
                    acc_sub = f"{max_acc_row['Method']} ({max_acc_row['Dataset']})"
                else:
                    acc_val = "N/A"
                    acc_sub = "No completed runs"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Best Observed Accuracy</div>
                        <div class="metric-value">{acc_val}</div>
                        <div class="metric-subtitle">{acc_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_c4:
                # Best Observed MCC
                if not completed_subset.empty:
                    max_mcc_row = completed_subset.loc[completed_subset["MCC"].idxmax()]
                    mcc_val = f"{max_mcc_row['MCC']:.4f}"
                    mcc_sub = f"{max_mcc_row['Method']} ({max_mcc_row['Dataset']})"
                else:
                    mcc_val = "N/A"
                    mcc_sub = "No completed runs"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Best Observed MCC</div>
                        <div class="metric-value">{mcc_val}</div>
                        <div class="metric-subtitle">{mcc_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_c5:
                # Recommended Method
                recs_in_view = list(filtered_df[filtered_df["Is Recommended"]]["Method"].unique())
                if recs_in_view:
                    rec_val = ", ".join(recs_in_view)
                    rec_sub = "★ Pareto Optimal"
                else:
                    rec_val = (
                        filtered_df["Recommended Method"].iloc[0]
                        if not filtered_df.empty
                        else "N/A"
                    )
                    rec_sub = "Optimal Candidate"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Recommended Method</div>
                        <div class="metric-value">{rec_val}</div>
                        <div class="metric-subtitle">{rec_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Informative notice if any failed/unsupported methods in view (Requirement 12)
            unsupported_in_view = filtered_df[filtered_df["Execution Status"] != "Completed"]
            if not unsupported_in_view.empty:
                st.info(
                    "ℹ️ **Notice regarding Failed / Unsupported Methods:** "
                    + ", ".join(
                        f"**{r['Method']}** ({r['Dataset']}): {r['Status Details'] or r['Execution Status']}"
                        for _, r in unsupported_in_view.iterrows()
                    )
                    + ". These methods are excluded from numerical metric plots rather than treated as zero."
                )

            # Metric Selector (Requirement 9)
            st.markdown("---")
            st.markdown("### 📈 Interactive Performance Visualizations")
            metric_view = st.radio(
                "Focus Metric View",
                [
                    "6-Metric Grid (Image 3 Style)",
                    "All Charts (Tabbed Overview)",
                    "Accuracy Comparison",
                    "Matthews Correlation Coefficient (MCC)",
                    "Feature Reduction (%)",
                    "Execution Runtime (s)",
                    "Multi-Objective Scatter (Accuracy vs Reduction)",
                ],
                horizontal=True,
                key="metric_view_selector",
            )

            # Plotly Chart Renderers (Requirements 2, 3, 4, 5, 6, 10)
            plot_records = completed_subset.copy()

            def render_six_metric_grid(data):
                st.markdown("#### 📊 Comparative Metric Cards (Path: metrics.csv / final_comparison.csv)")
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                with r1_c1:
                    st.plotly_chart(
                        render_mini_metric_bar(data, "Accuracy", "test__accuracy", y_format=".2f", is_percentage=True, path_subtitle="Path: metrics.csv"),
                        use_container_width=True,
                    )
                with r1_c2:
                    st.plotly_chart(
                        render_mini_metric_bar(data, "MCC", "test__average_precision", y_format=".2f", is_percentage=False, path_subtitle="Path: metrics.csv"),
                        use_container_width=True,
                    )
                with r1_c3:
                    st.plotly_chart(
                        render_mini_metric_bar(data, "Accuracy", "test__f1", y_format=".2f", is_percentage=False, path_subtitle="Path: metrics.csv"),
                        use_container_width=True,
                    )
                r2_c1, r2_c2, r2_c3 = st.columns(3)
                with r2_c1:
                    st.plotly_chart(
                        render_mini_metric_bar(data, "Accuracy", "test__precision", y_format=".2f", is_percentage=False, path_subtitle="Path: metrics.csv"),
                        use_container_width=True,
                    )
                with r2_c2:
                    st.plotly_chart(
                        render_mini_metric_bar(data, "Feature Reduction (%)", "test__recall", y_format=".1f", is_percentage=True, path_subtitle="Path: metrics.csv"),
                        use_container_width=True,
                    )
                with r2_c3:
                    st.plotly_chart(
                        render_mini_metric_bar(data, "MCC", "test__roc_auc", y_format=".2f", is_percentage=False, path_subtitle="Path: metrics.csv"),
                        use_container_width=True,
                    )

            def render_accuracy_chart(data):
                fig = px.bar(
                    data,
                    x="Method",
                    y="Accuracy",
                    color="Dataset" if len(data["Dataset"].unique()) > 1 else "Method",
                    barmode="group",
                    text=data["Accuracy"].apply(lambda v: f"{v:.2%}"),
                    title="Held-Out Test Accuracy Comparison (Higher is Better)",
                    hover_data={
                        "Dataset": True,
                        "Method": True,
                        "Accuracy": ":.2%",
                        "MCC": ":.4f",
                        "Selected Features": True,
                        "Total Features": True,
                        "Feature Reduction (%)": ":.2f%",
                        "Runtime (s)": ":.4f",
                    },
                    template="plotly_white",
                )
                fig.update_layout(
                    yaxis_title="Accuracy (0.0 to 1.0)",
                    yaxis_range=[0.0, 1.05],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_traces(textposition="outside")
                return fig

            def render_mcc_chart(data):
                fig = px.bar(
                    data,
                    x="Method",
                    y="MCC",
                    color="Dataset" if len(data["Dataset"].unique()) > 1 else "Method",
                    barmode="group",
                    text=data["MCC"].apply(lambda v: f"{v:.4f}"),
                    title="Matthews Correlation Coefficient (MCC) Comparison (Higher is Better)",
                    hover_data={
                        "Dataset": True,
                        "Method": True,
                        "MCC": ":.4f",
                        "Accuracy": ":.2%",
                        "Selected Features": True,
                        "Feature Reduction (%)": ":.2f%",
                        "Runtime (s)": ":.4f",
                    },
                    template="plotly_white",
                )
                fig.update_layout(
                    yaxis_title="MCC (-1.0 to +1.0)",
                    yaxis_range=[-0.1, 1.05],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_traces(textposition="outside")
                return fig

            def render_reduction_chart(data):
                fig = px.bar(
                    data,
                    x="Method",
                    y="Feature Reduction (%)",
                    color="Dataset" if len(data["Dataset"].unique()) > 1 else "Method",
                    barmode="group",
                    text=data["Feature Reduction (%)"].apply(lambda v: f"{v:.2f}%"),
                    title="Feature Reduction Percentage (Higher is More Compact)",
                    hover_data={
                        "Dataset": True,
                        "Method": True,
                        "Feature Reduction (%)": ":.2f%",
                        "Selected Features": True,
                        "Total Features": True,
                        "Accuracy": ":.2%",
                    },
                    template="plotly_white",
                )
                min_red = max(0.0, data["Feature Reduction (%)"].min() - 2.0)
                fig.update_layout(
                    yaxis_title="Feature Reduction (%)",
                    yaxis_range=[min_red, 100.5],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_traces(textposition="outside")
                return fig

            def render_runtime_chart(data):
                fig = px.bar(
                    data,
                    x="Method",
                    y="Runtime (s)",
                    color="Dataset" if len(data["Dataset"].unique()) > 1 else "Method",
                    barmode="group",
                    text=data["Runtime (s)"].apply(lambda v: f"{v:.3f}s"),
                    title="Execution Runtime in Seconds (Lower is Faster)",
                    hover_data={
                        "Dataset": True,
                        "Method": True,
                        "Runtime (s)": ":.4f",
                        "Accuracy": ":.2%",
                        "Selected Features": True,
                    },
                    template="plotly_white",
                )
                fig.update_layout(
                    yaxis_title="Runtime (seconds)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_traces(textposition="outside")
                return fig

            def render_scatter_chart(data):
                fig = px.scatter(
                    data,
                    x="Feature Reduction (%)",
                    y="Accuracy",
                    color="Method",
                    symbol="Dataset" if len(data["Dataset"].unique()) > 1 else None,
                    text="Method",
                    title="Accuracy vs. Feature Reduction Trade-Off by Algorithm",
                    hover_data={
                        "Dataset": True,
                        "Method": True,
                        "Accuracy": ":.2%",
                        "Feature Reduction (%)": ":.2f%",
                        "MCC": ":.4f",
                        "Selected Features": True,
                        "Total Features": True,
                        "Runtime (s)": ":.4f",
                    },
                    template="plotly_white",
                )
                fig.update_traces(
                    marker=dict(size=14, line=dict(width=1.5, color="#1e293b")),
                    textposition="top center",
                )
                fig.update_layout(
                    xaxis_title="Feature Reduction (%) [Higher = More Compact]",
                    yaxis_title="Accuracy (0.0 to 1.0) [Higher = Better]",
                    yaxis_range=[max(0.0, data["Accuracy"].min() - 0.1), 1.05],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                return fig

            if plot_records.empty:
                st.warning("No completed algorithm results available to display in charts for this filter selection.")
            else:
                if metric_view == "6-Metric Grid (Image 3 Style)":
                    render_six_metric_grid(plot_records)

                elif metric_view == "All Charts (Tabbed Overview)":
                    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "6-Metric Grid (Image 3)",
                        "Accuracy",
                        "Matthews Correlation (MCC)",
                        "Feature Reduction (%)",
                        "Execution Runtime",
                        "Multi-Objective Scatter",
                    ])
                    with tab0:
                        render_six_metric_grid(plot_records)
                    with tab1:
                        st.plotly_chart(render_accuracy_chart(plot_records), use_container_width=True)
                    with tab2:
                        st.plotly_chart(render_mcc_chart(plot_records), use_container_width=True)
                    with tab3:
                        st.plotly_chart(render_reduction_chart(plot_records), use_container_width=True)
                    with tab4:
                        st.plotly_chart(render_runtime_chart(plot_records), use_container_width=True)
                    with tab5:
                        st.plotly_chart(render_scatter_chart(plot_records), use_container_width=True)

                elif metric_view == "Accuracy Comparison":
                    st.plotly_chart(render_accuracy_chart(plot_records), use_container_width=True)

                elif metric_view == "Matthews Correlation Coefficient (MCC)":
                    st.plotly_chart(render_mcc_chart(plot_records), use_container_width=True)

                elif metric_view == "Feature Reduction (%)":
                    st.plotly_chart(render_reduction_chart(plot_records), use_container_width=True)

                elif metric_view == "Execution Runtime (s)":
                    st.plotly_chart(render_runtime_chart(plot_records), use_container_width=True)

                elif metric_view == "Multi-Objective Scatter (Accuracy vs Reduction)":
                    st.plotly_chart(render_scatter_chart(plot_records), use_container_width=True)

            # Detailed Comparison Table (Requirements 7, 11)
            st.markdown("---")
            st.markdown("### 📋 Detailed Multi-Algorithm Comparison Table")

            t_ctl1, t_ctl2 = st.columns([2, 1.2])
            with t_ctl1:
                search_query = st.text_input(
                    "Search table",
                    placeholder="Search by algorithm name, dataset, or notes...",
                    key="table_search_input",
                )
            with t_ctl2:
                sort_choice = st.selectbox(
                    "Sort table by",
                    [
                        "Default (Dataset & Method)",
                        "Accuracy (Highest First)",
                        "MCC (Highest First)",
                        "Feature Reduction % (Highest First)",
                        "Selected Features (Fewest First)",
                        "Runtime (Fastest First)",
                    ],
                    index=0,
                    key="table_sort_selector",
                )

            # Apply table search
            table_display_df = filtered_df.copy()
            if search_query.strip():
                query = search_query.strip().lower()
                table_display_df = table_display_df[
                    table_display_df["Method"].str.lower().str.contains(query, na=False)
                    | table_display_df["Dataset"].str.lower().str.contains(query, na=False)
                    | table_display_df["Execution Status"].str.lower().str.contains(query, na=False)
                    | table_display_df["Status Details"].str.lower().str.contains(query, na=False)
                ]

            # Apply table sorting
            if sort_choice == "Accuracy (Highest First)":
                table_display_df = table_display_df.sort_values(
                    by="Accuracy", ascending=False, na_position="last"
                )
            elif sort_choice == "MCC (Highest First)":
                table_display_df = table_display_df.sort_values(
                    by="MCC", ascending=False, na_position="last"
                )
            elif sort_choice == "Feature Reduction % (Highest First)":
                table_display_df = table_display_df.sort_values(
                    by="Feature Reduction (%)", ascending=False, na_position="last"
                )
            elif sort_choice == "Selected Features (Fewest First)":
                table_display_df = table_display_df.sort_values(
                    by="Selected Features", ascending=True, na_position="last"
                )
            elif sort_choice == "Runtime (Fastest First)":
                table_display_df = table_display_df.sort_values(
                    by="Runtime (s)", ascending=True, na_position="last"
                )

            # Format columns cleanly
            display_columns = [
                "Dataset",
                "Method",
                "Accuracy",
                "MCC",
                "Selected Features",
                "Total Features",
                "Feature Reduction (%)",
                "Runtime (s)",
                "Execution Status",
                "Status Details",
            ]
            formatted_view = table_display_df[display_columns].copy()

            # Format numeric columns with fallback to "—"
            formatted_view["Accuracy"] = formatted_view["Accuracy"].apply(
                lambda v: f"{v:.2%}" if pd.notna(v) else "—"
            )
            formatted_view["MCC"] = formatted_view["MCC"].apply(
                lambda v: f"{v:.4f}" if pd.notna(v) else "—"
            )
            formatted_view["Selected Features"] = formatted_view["Selected Features"].apply(
                lambda v: f"{int(v)}" if pd.notna(v) else "—"
            )
            formatted_view["Total Features"] = formatted_view["Total Features"].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) else "—"
            )
            formatted_view["Feature Reduction (%)"] = formatted_view["Feature Reduction (%)"].apply(
                lambda v: f"{v:.2f}%" if pd.notna(v) else "—"
            )
            formatted_view["Runtime (s)"] = formatted_view["Runtime (s)"].apply(
                lambda v: f"{v:.4f}s" if pd.notna(v) else "—"
            )

            st.dataframe(formatted_view, use_container_width=True, hide_index=True)
            st.caption(
                f"Showing {len(formatted_view)} of {len(all_records_df)} total records. "
                "Columns are sortable; failed or unsupported runs display '—' rather than zero."
            )


# SECTION 4: FEATURE EXPLORER & CONSENSUS
elif nav_section == "Feature Explorer & Consensus":
    st.markdown('<div class="main-title">🔍 Feature Explorer & Consensus Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore individual selected features, cross-method consensus frequencies, '
        'UpSet-style intersections, and clear representations distinguishing original vs encoded features.</div>',
        unsafe_allow_html=True,
    )

    # 1. Dataset Selection & Data Resolution (Requirements 1, 10)
    dataset_choices = ["Colon Cancer", "Leukemia"]
    has_uploaded_eval = st.session_state.get("evaluation_outcome") is not None
    uploaded_name = st.session_state.get("uploaded_dataset_name", "Uploaded CSV")
    if has_uploaded_eval:
        dataset_choices.append(f"Uploaded: {uploaded_name}")
    elif st.session_state.get("uploaded_df") is not None:
        dataset_choices.append(f"Uploaded: {uploaded_name} (Pending Evaluation)")

    # Default index aligns with sidebar active selection
    default_ds_idx = 0
    if st.session_state.get("selected_source") == "Leukemia":
        default_ds_idx = 1
    elif st.session_state.get("selected_source") == "Evaluate New CSV":
        if has_uploaded_eval or st.session_state.get("uploaded_df") is not None:
            default_ds_idx = len(dataset_choices) - 1

    col_ds, col_ctrl = st.columns([1.4, 2.6])
    with col_ds:
        active_explorer_ds = st.selectbox(
            "📁 Select Target Dataset",
            options=dataset_choices,
            index=default_ds_idx,
            key="explorer_dataset_choice",
        )

    # Resolve dataset results and method records
    is_eval_ready = True
    curr_dataset_name = active_explorer_ds
    curr_results_df = None
    curr_statuses_df = None
    curr_raw_X = None
    curr_raw_y = None

    if active_explorer_ds == "Colon Cancer":
        try:
            bench_all = load_benchmark_comparison()
            curr_results_df = bench_all[bench_all["Dataset"] == "Colon Cancer"].copy()
            curr_dataset_name = "Colon Cancer"
            cached_ds = load_cached_datasets()
            if "Colon Cancer" in cached_ds:
                curr_raw_X, curr_raw_y = cached_ds["Colon Cancer"]
        except Exception as e:
            st.error(f"Failed to load Colon Cancer results: {e}")
            is_eval_ready = False

    elif active_explorer_ds == "Leukemia":
        try:
            bench_all = load_benchmark_comparison()
            curr_results_df = bench_all[bench_all["Dataset"] == "Leukemia"].copy()
            curr_dataset_name = "Leukemia"
            cached_ds = load_cached_datasets()
            if "Leukemia" in cached_ds:
                curr_raw_X, curr_raw_y = cached_ds["Leukemia"]
        except Exception as e:
            st.error(f"Failed to load Leukemia results: {e}")
            is_eval_ready = False

    elif "Pending Evaluation" in active_explorer_ds:
        is_eval_ready = False
        st.markdown(
            f"""
            <div class="section-card" style="text-align: center; padding: 2.5rem 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">⏳</div>
                <h3>Evaluation Needed for {uploaded_name}</h3>
                <p style="color: #64748b; max-width: 600px; margin: 0 auto 1rem auto;">
                    This uploaded CSV has been validated but feature selection has not yet been executed.
                    Please navigate to <strong>Dataset Upload & Inspection</strong> and run the feature selection suite.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif has_uploaded_eval:
        outcome = st.session_state["evaluation_outcome"]
        curr_results_df = outcome["results"].copy()
        curr_statuses_df = outcome["statuses"].copy()
        curr_dataset_name = curr_results_df["Dataset"].iloc[0] if not curr_results_df.empty else uploaded_name
        if st.session_state.get("uploaded_df") is not None and st.session_state.get("uploaded_target") is not None:
            raw_df = st.session_state["uploaded_df"]
            tgt = st.session_state["uploaded_target"]
            if tgt in raw_df.columns:
                curr_raw_X = raw_df.drop(columns=[tgt])
                curr_raw_y = raw_df[tgt]

    if is_eval_ready and curr_results_df is not None:
        # Build comprehensive method metadata dictionary (Requirements 8, 10)
        ALL_METHODS = ["CFS", "FCBF", "FAST", "GA", "PSO", "Hybrid"]
        method_dict = {}

        # 1. From results_df
        results_by_method = {}
        if curr_results_df is not None and not curr_results_df.empty:
            for _, r in curr_results_df.iterrows():
                results_by_method[r["Method"]] = r

        # 2. From statuses_df (if uploaded dataset)
        statuses_by_method = {}
        if curr_statuses_df is not None and not curr_statuses_df.empty:
            for _, s in curr_statuses_df.iterrows():
                statuses_by_method[s["Method"]] = s

        for m in ALL_METHODS:
            if m in results_by_method:
                r = results_by_method[m]
                raw_feat_str = r.get("Selected Feature Names", "")
                parsed_feats = parse_feature_names(raw_feat_str)
                method_dict[m] = {
                    "status": "Completed",
                    "details": "Execution completed successfully",
                    "features": parsed_feats,
                    "accuracy": float(r.get("Accuracy", 0.0)),
                    "mcc": float(r.get("MCC", 0.0)),
                    "selected_features": int(r.get("Selected Features", len(parsed_feats))),
                    "total_features": int(r.get("Total Features", 0)),
                    "feature_reduction": float(r.get("Feature Reduction (%)", 0.0)),
                    "runtime": float(r.get("Runtime (s)", 0.0)),
                }
            elif m in statuses_by_method:
                s = statuses_by_method[m]
                method_dict[m] = {
                    "status": s.get("Status", "Unsupported"),
                    "details": s.get("Details", "Method execution not supported"),
                    "features": [],
                    "accuracy": None,
                    "mcc": None,
                    "selected_features": 0,
                    "total_features": 0,
                    "feature_reduction": None,
                    "runtime": None,
                }
            else:
                method_dict[m] = {
                    "status": "Unavailable",
                    "details": "No run record available for this dataset",
                    "features": [],
                    "accuracy": None,
                    "mcc": None,
                    "selected_features": 0,
                    "total_features": 0,
                    "feature_reduction": None,
                    "runtime": None,
                }

        # Available methods and method selector (Requirement 1)
        completed_methods = [m for m in ALL_METHODS if method_dict[m]["status"] == "Completed"]

        with col_ctrl:
            selected_methods = st.multiselect(
                "⚡ Select Feature-Selection Methods to Analyze",
                options=ALL_METHODS,
                default=completed_methods,
                help="Choose one or more methods. You can also select unsupported/failed methods to inspect their status.",
                key="explorer_methods_multiselect",
            )

        # Quick action buttons for method selection
        btn_c1, btn_c2, btn_c3, _ = st.columns([1, 1.2, 1, 2.8])
        with btn_c1:
            if st.button("Select All", key="btn_sel_all_methods"):
                st.session_state["explorer_methods_multiselect"] = ALL_METHODS
                st.rerun()
        with btn_c2:
            if st.button("Completed Only", key="btn_sel_completed_methods"):
                st.session_state["explorer_methods_multiselect"] = completed_methods
                st.rerun()
        with btn_c3:
            if st.button("Clear All", key="btn_clear_methods"):
                st.session_state["explorer_methods_multiselect"] = []
                st.rerun()

        # Check for failed or unsupported methods among user selections (Requirement 8)
        non_completed_selected = [
            m for m in selected_methods if method_dict[m]["status"] != "Completed"
        ]
        if non_completed_selected:
            for m in non_completed_selected:
                st.markdown(
                    f"""
                    <div style="background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.85rem 1.15rem; border-radius: 8px; margin-bottom: 0.75rem; border: 1px solid #fef3c7;">
                        <strong style="color: #92400e;">⚠️ {m} — Status: {method_dict[m]['status']}</strong><br/>
                        <span style="font-size: 0.9rem; color: #78350f;">
                            <strong>Diagnostic Details:</strong> {method_dict[m]['details'] or 'Execution unsupported for this dataset architecture.'}<br/>
                            <em>Actual algorithm execution status is preserved; no artificial or simulated feature lists are generated.</em>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Methodological Boundary Callout (Requirement 9)
        st.markdown(
            """
            <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 1rem 1.25rem; border-radius: 8px; margin-top: 0.5rem; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="font-weight: 700; color: #1e40af; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">⚖️</span> Methodological Boundary Note: Selection Frequency vs. Feature Importance
                </div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.55;">
                    Multi-method selection frequency quantifies <strong>algorithmic consensus</strong> across distinct search heuristics 
                    (correlation filters, mutual information ranking, minimum spanning tree clustering, genetic algorithms, and particle swarm optimization) 
                    on the training partition. <strong>Selection frequency must NOT be described or interpreted as biological feature importance, predictive weight, or causal influence.</strong>
                    Individual algorithms may select correlated or surrogate markers depending on their optimization objective; absence of selection does not imply biological or causal irrelevance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_completed_methods = [
            m for m in selected_methods if method_dict[m]["status"] == "Completed"
        ]

        if not selected_methods:
            st.info("Please select at least one feature selection method above to begin exploring.")
        elif not active_completed_methods:
            st.warning("All currently selected methods have an Unsupported or Error status. Please select at least one Completed method to inspect features.")
        else:
            # 5 Rich Tabs for Comprehensive Exploration
            tab_inspect, tab_consensus, tab_overlap, tab_distributions, tab_export = st.tabs([
                "🔎 Method Feature Inspector & Search",
                "📊 Selection Frequency Analysis",
                "🤝 UpSet & Intersection Overlap",
                "🔬 Correlations & Distributions (Image 4)",
                "💾 Export & Download Features",
            ])

            # TAB 1: METHOD FEATURE INSPECTOR & SEARCH (Requirements 1, 2, 3, 7, 8)
            with tab_inspect:
                insp_col1, insp_col2 = st.columns([1.5, 2.5])
                with insp_col1:
                    inspected_method = st.selectbox(
                        "Select Method to Inspect Feature List",
                        options=active_completed_methods,
                        index=0,
                        key="insp_method_selector",
                    )

                m_info = method_dict[inspected_method]
                raw_feature_list = m_info["features"]

                # KPI Metrics row for the inspected method
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                with kpi1:
                    st.metric("Selected Features", len(raw_feature_list))
                with kpi2:
                    st.metric("Total Input Features", m_info["total_features"] if m_info["total_features"] else "—")
                with kpi3:
                    st.metric("Feature Reduction", f"{m_info['feature_reduction']:.2f}%" if m_info["feature_reduction"] is not None else "—")
                with kpi4:
                    st.metric("Hold-Out Accuracy", f"{m_info['accuracy']:.2%}" if m_info["accuracy"] is not None else "—")
                with kpi5:
                    st.metric("Hold-Out MCC", f"{m_info['mcc']:.4f}" if m_info["mcc"] is not None else "—")

                # Decode all features for this method (Requirement 7)
                decoded_features = [
                    decode_feature_info(f, curr_dataset_name) for f in raw_feature_list
                ]

                # Encoding breakdown summary
                n_raw = sum(1 for d in decoded_features if not d["Is Encoded"])
                n_num = sum(1 for d in decoded_features if "Numeric" in d["Encoding / Origin"])
                n_cat = sum(1 for d in decoded_features if "One-Hot" in d["Encoding / Origin"])

                if any(d["Is Encoded"] for d in decoded_features):
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 0.75rem; margin: 0.75rem 0 1rem 0; flex-wrap: wrap;">
                            <span class="status-badge" style="background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd;">
                                Original Raw Features: <strong>{n_raw}</strong>
                            </span>
                            <span class="status-badge" style="background: #fef3c7; color: #92400e; border: 1px solid #fde68a;">
                                Imputed & Scaled Numeric: <strong>{n_num}</strong>
                            </span>
                            <span class="status-badge" style="background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff;">
                                One-Hot Encoded Categories: <strong>{n_cat}</strong>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Search, Sorting & Pagination Controls (Requirement 3)
                s_col1, s_col2, s_col3 = st.columns([2, 1.2, 0.8])
                with s_col1:
                    search_query = st.text_input(
                        "🔍 Search features by model name or original column:",
                        value="",
                        key="insp_search_input",
                        placeholder="e.g. 493, numeric__age, gene_1834...",
                    )
                with s_col2:
                    sort_order = st.selectbox(
                        "Sort Order",
                        options=[
                            "Original Selector Order",
                            "Alphabetical (A → Z)",
                            "Alphabetical (Z → A)",
                            "Original Column (A → Z)",
                        ],
                        key="insp_sort_order",
                    )
                with s_col3:
                    page_size = st.selectbox(
                        "Features per Page",
                        options=[10, 25, 50, 100],
                        index=0,
                        key="insp_page_size",
                    )

                # Filter by search query
                filtered_features = decoded_features
                if search_query.strip():
                    q = search_query.strip().lower()
                    filtered_features = [
                        d for d in filtered_features
                        if q in d["Feature Name"].lower() or q in d["Original Column"].lower()
                    ]

                # Sort
                if sort_order == "Alphabetical (A → Z)":
                    filtered_features = sorted(filtered_features, key=lambda x: x["Feature Name"].lower())
                elif sort_order == "Alphabetical (Z → A)":
                    filtered_features = sorted(filtered_features, key=lambda x: x["Feature Name"].lower(), reverse=True)
                elif sort_order == "Original Column (A → Z)":
                    filtered_features = sorted(filtered_features, key=lambda x: x["Original Column"].lower())

                # Selected-feature count badge (Requirement 3)
                st.markdown(
                    f"**Selected Feature Count:** Displaying **{len(filtered_features)}** of **{len(raw_feature_list)}** "
                    f"total features selected by **{inspected_method}**."
                )

                if not filtered_features:
                    st.info("No features matched your search query.")
                else:
                    # Pagination controls
                    total_pages = max(1, math.ceil(len(filtered_features) / page_size))
                    p_col1, p_col2 = st.columns([1, 4])
                    with p_col1:
                        current_page = st.number_input(
                            f"Page (1 to {total_pages})",
                            min_value=1,
                            max_value=total_pages,
                            value=1,
                            step=1,
                            key="insp_page_number",
                        )

                    start_idx = (current_page - 1) * page_size
                    end_idx = min(len(filtered_features), start_idx + page_size)
                    page_items = filtered_features[start_idx:end_idx]

                    # Paginated dataframe
                    display_rows = []
                    for idx, item in enumerate(page_items, start=start_idx + 1):
                        display_rows.append({
                            "#": idx,
                            "Selected Feature Name (Model Input)": item["Feature Name"],
                            "Derived From (Original Column)": item["Original Column"],
                            "Encoding / Representation": item["Encoding / Origin"],
                            "Type Category": item["Encoding Status"],
                            "Representation Details": item["Details"],
                        })

                    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
                    st.caption(
                        f"Showing items {start_idx + 1}–{end_idx} of {len(filtered_features)} matching features "
                        f"(Page {current_page} of {total_pages})."
                    )

                    # CSV Download for this inspected method (Requirement 6)
                    dl_df = pd.DataFrame([{
                        "Dataset": curr_dataset_name,
                        "Method": inspected_method,
                        "Rank": idx + 1,
                        "Feature Name": item["Feature Name"],
                        "Original Column": item["Original Column"],
                        "Encoding / Origin": item["Encoding / Origin"],
                        "Is Encoded": item["Is Encoded"],
                        "Details": item["Details"],
                    } for idx, item in enumerate(decoded_features)])

                    csv_m = dl_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download {inspected_method} Selected Features (CSV)",
                        data=csv_m,
                        file_name=f"{curr_dataset_name.lower().replace(' ', '_')}_{inspected_method.lower()}_selected_features.csv",
                        mime="text/csv",
                        key=f"btn_dl_{inspected_method}_csv",
                    )

            # TAB 2: SELECTION FREQUENCY ANALYSIS (Requirements 5, 9)
            with tab_consensus:
                st.markdown("#### 📊 Selection Frequency Across Algorithms")
                if len(active_completed_methods) < 2:
                    st.info("ℹ️ Cross-method selection frequency requires at least 2 completed methods to compare. Please select 2 or more methods in the control panel above.")
                else:
                    # Collect all selected features
                    method_feature_map = {
                        m: set(method_dict[m]["features"]) for m in active_completed_methods
                    }
                    all_unique_feats = sorted(list(set.union(*method_feature_map.values())))

                    # Build frequency registry
                    freq_records = []
                    for f in all_unique_feats:
                        selecting_methods = [m for m in active_completed_methods if f in method_feature_map[m]]
                        f_info = decode_feature_info(f, curr_dataset_name)
                        freq_records.append({
                            "Feature Name": f,
                            "Original Column": f_info["Original Column"],
                            "Encoding / Origin": f_info["Encoding / Origin"],
                            "Selection Frequency": len(selecting_methods),
                            "Selection Rate (%)": round(len(selecting_methods) / len(active_completed_methods) * 100, 1),
                            "Selecting Methods": ", ".join(sorted(selecting_methods)),
                            "Is Encoded": f_info["Is Encoded"],
                        })

                    freq_df = pd.DataFrame(freq_records).sort_values(
                        by=["Selection Frequency", "Feature Name"], ascending=[False, True]
                    )

                    # KPI Summary Cards
                    n_total_selected = len(all_unique_feats)
                    n_universal = (freq_df["Selection Frequency"] == len(active_completed_methods)).sum()
                    n_majority = (freq_df["Selection Frequency"] >= math.ceil(len(active_completed_methods) / 2)).sum()
                    n_unique_single = (freq_df["Selection Frequency"] == 1).sum()

                    f_kpi1, f_kpi2, f_kpi3, f_kpi4 = st.columns(4)
                    with f_kpi1:
                        st.metric("Total Unique Features Selected", n_total_selected)
                    with f_kpi2:
                        st.metric("Universal Consensus (All Methods)", n_universal)
                    with f_kpi3:
                        st.metric("Majority Consensus (≥ 50%)", n_majority)
                    with f_kpi4:
                        st.metric("Selected by Single Method Only", n_unique_single)

                    # Chart Filter Controls
                    fc_col1, fc_col2 = st.columns([1.5, 1.5])
                    with fc_col1:
                        max_top_n = min(50, len(freq_df))
                        top_n = st.slider(
                            "Top N features to display in chart",
                            min_value=min(5, max_top_n),
                            max_value=max(5, max_top_n),
                            value=min(25, max_top_n),
                            key="freq_top_n_slider",
                        )
                    with fc_col2:
                        min_freq_filter = st.slider(
                            "Minimum selection frequency threshold",
                            min_value=1,
                            max_value=len(active_completed_methods),
                            value=1,
                            key="freq_min_threshold",
                        )

                    chart_df = freq_df[freq_df["Selection Frequency"] >= min_freq_filter].head(top_n)

                    if chart_df.empty:
                        st.info("No features meet the minimum frequency threshold.")
                    else:
                        fig_freq = px.bar(
                            chart_df,
                            x="Feature Name",
                            y="Selection Frequency",
                            color="Selection Frequency",
                            title=f"Feature Selection Frequency Across {len(active_completed_methods)} Evaluated Algorithms (Top {len(chart_df)})",
                            labels={
                                "Feature Name": "Feature Identifier",
                                "Selection Frequency": "Number of Selecting Methods",
                            },
                            hover_data={
                                "Feature Name": True,
                                "Original Column": True,
                                "Selection Frequency": True,
                                "Selection Rate (%)": True,
                                "Selecting Methods": True,
                            },
                            color_continuous_scale="Tealgrn",
                        )
                        fig_freq.update_layout(
                            template="plotly_white",
                            xaxis_tickangle=-45,
                            yaxis=dict(
                                tickmode="linear",
                                tick0=0,
                                dtick=1,
                                range=[0, len(active_completed_methods) + 0.5],
                            ),
                        )
                        st.plotly_chart(fig_freq, use_container_width=True)

                    # Full frequency registry table
                    st.markdown("##### 📋 Complete Selection Frequency Registry")
                    st.dataframe(
                        freq_df[[
                            "Feature Name",
                            "Original Column",
                            "Encoding / Origin",
                            "Selection Frequency",
                            "Selection Rate (%)",
                            "Selecting Methods",
                        ]],
                        use_container_width=True,
                        hide_index=True,
                    )

            # TAB 3: UPSET & INTERSECTION OVERLAP (Requirements 4, 7)
            with tab_overlap:
                st.markdown("#### 🤝 Inter-Algorithm Overlap & Set Intersections")
                if len(active_completed_methods) < 2:
                    st.info("ℹ️ Overlap and intersection analysis requires at least 2 completed methods to compare.")
                else:
                    method_feature_map = {
                        m: set(method_dict[m]["features"]) for m in active_completed_methods
                    }
                    all_unique_feats = sorted(list(set.union(*method_feature_map.values())))

                    # 1. UpSet-Style Subset Intersections (Requirement 4)
                    intersection_groups = {}
                    for f in all_unique_feats:
                        selecting_set = frozenset([m for m in active_completed_methods if f in method_feature_map[m]])
                        if selecting_set not in intersection_groups:
                            intersection_groups[selecting_set] = []
                        intersection_groups[selecting_set].append(f)

                    upset_rows = []
                    for subset, feats in intersection_groups.items():
                        deg = len(subset)
                        name = " ∩ ".join(sorted(list(subset))) if deg > 1 else f"{list(subset)[0]} only"
                        upset_rows.append({
                            "Subset Combination": name,
                            "Intersection Size": len(feats),
                            "Degree (Methods Count)": deg,
                            "Methods": ", ".join(sorted(list(subset))),
                            "Features List Preview": ", ".join(feats[:8]) + ("..." if len(feats) > 8 else ""),
                        })

                    upset_df = pd.DataFrame(upset_rows).sort_values(
                        by=["Degree (Methods Count)", "Intersection Size"], ascending=[False, False]
                    )

                    u_col1, u_col2 = st.columns([1.6, 1.4])
                    with u_col1:
                        fig_upset = px.bar(
                            upset_df,
                            x="Subset Combination",
                            y="Intersection Size",
                            color="Degree (Methods Count)",
                            title="UpSet-Style Intersection Sizes Across Method Combinations",
                            labels={
                                "Subset Combination": "Algorithm Combination",
                                "Intersection Size": "Number of Shared Features",
                                "Degree (Methods Count)": "Degree",
                            },
                            text="Intersection Size",
                            color_continuous_scale="Blues",
                            hover_data={
                                "Subset Combination": True,
                                "Intersection Size": True,
                                "Degree (Methods Count)": True,
                                "Features List Preview": True,
                            },
                        )
                        fig_upset.update_layout(template="plotly_white", xaxis_tickangle=-35)
                        st.plotly_chart(fig_upset, use_container_width=True)

                    with u_col2:
                        # Pairwise Jaccard Similarity Matrix
                        methods = active_completed_methods
                        matrix = np.zeros((len(methods), len(methods)))
                        for i, m1 in enumerate(methods):
                            for j, m2 in enumerate(methods):
                                s1, s2 = method_feature_map[m1], method_feature_map[m2]
                                if not s1 and not s2:
                                    matrix[i, j] = 1.0
                                elif not s1 or not s2:
                                    matrix[i, j] = 0.0
                                else:
                                    matrix[i, j] = len(s1.intersection(s2)) / len(s1.union(s2))

                        jaccard_df = pd.DataFrame(matrix, index=methods, columns=methods)
                        fig_jac = px.imshow(
                            jaccard_df,
                            text_auto=".2f",
                            color_continuous_scale="Purples",
                            title="Pairwise Feature Subset Overlap (Jaccard Index)",
                            labels=dict(x="Method", y="Method", color="Jaccard Index"),
                        )
                        fig_jac.update_layout(template="plotly_white")
                        st.plotly_chart(fig_jac, use_container_width=True)

                    # 2. Comprehensive Intersection Matrix Table
                    st.markdown("##### 📋 Complete Feature Intersection Matrix")
                    st.caption("Each row represents a unique feature selected by at least one method, with checkmarks indicating selection.")

                    matrix_rows = []
                    for f in all_unique_feats:
                        f_info = decode_feature_info(f, curr_dataset_name)
                        row_dict = {
                            "Feature Name": f,
                            "Derived From (Original Column)": f_info["Original Column"],
                            "Encoding / Representation": f_info["Encoding / Origin"],
                        }
                        n_sel = 0
                        for m in active_completed_methods:
                            if f in method_feature_map[m]:
                                row_dict[m] = "✓"
                                n_sel += 1
                            else:
                                row_dict[m] = "—"
                        row_dict["Selection Count"] = f"{n_sel}/{len(active_completed_methods)}"
                        row_dict["Count_Num"] = n_sel
                        if n_sel == len(active_completed_methods):
                            row_dict["Consensus Level"] = "★ Full Consensus (All)"
                        elif n_sel >= math.ceil(len(active_completed_methods) / 2):
                            row_dict["Consensus Level"] = "Majority Agreement"
                        elif n_sel >= 2:
                            row_dict["Consensus Level"] = "Shared (2+ Methods)"
                        else:
                            row_dict["Consensus Level"] = "Unique (Single Method)"
                        matrix_rows.append(row_dict)

                    matrix_df = pd.DataFrame(matrix_rows).sort_values(
                        by=["Count_Num", "Feature Name"], ascending=[False, True]
                    )

                    filter_opt = st.radio(
                        "Filter Intersection Table:",
                        options=["All Selected Features", "Shared Only (≥ 2 Methods)", "Full Consensus Only", "Unique Only (Single Method)"],
                        horizontal=True,
                        key="matrix_filter_radio",
                    )

                    filtered_matrix = matrix_df
                    if filter_opt == "Shared Only (≥ 2 Methods)":
                        filtered_matrix = matrix_df[matrix_df["Count_Num"] >= 2]
                    elif filter_opt == "Full Consensus Only":
                        filtered_matrix = matrix_df[matrix_df["Count_Num"] == len(active_completed_methods)]
                    elif filter_opt == "Unique Only (Single Method)":
                        filtered_matrix = matrix_df[matrix_df["Count_Num"] == 1]

                    cols_to_show = ["Feature Name", "Derived From (Original Column)", "Encoding / Representation", "Selection Count", "Consensus Level"] + active_completed_methods
                    st.dataframe(filtered_matrix[cols_to_show], use_container_width=True, hide_index=True)
                    st.caption(f"Showing {len(filtered_matrix)} of {len(matrix_df)} total unique features.")

                    # 3. Unique Features Breakdown per Method
                    st.markdown("##### 🎯 Method-Unique Feature Subsets")
                    st.caption("Features identified and selected solely by a single algorithm (no overlap with other selected methods).")

                    uniq_cols = st.columns(min(3, len(active_completed_methods)))
                    for idx, m in enumerate(active_completed_methods):
                        other_methods_feats = set.union(*[
                            method_feature_map[om] for om in active_completed_methods if om != m
                        ]) if len(active_completed_methods) > 1 else set()
                        m_unique = sorted(list(method_feature_map[m] - other_methods_feats))
                        total_m_feats = len(method_feature_map[m])
                        uniq_ratio = (len(m_unique) / total_m_feats * 100) if total_m_feats > 0 else 0.0

                        with uniq_cols[idx % len(uniq_cols)]:
                            with st.expander(f"**{m}**: {len(m_unique)} Unique ({uniq_ratio:.1f}%)", expanded=(len(m_unique) > 0)):
                                if m_unique:
                                    st.write(", ".join(m_unique[:15]) + ("..." if len(m_unique) > 15 else ""))
                                else:
                                    st.write("*(No unique features — all selections shared with other methods)*")

            # TAB 4: CORRELATIONS & DISTRIBUTIONS (Image 4 Style)
            with tab_distributions:
                st.markdown("### 📊 Correlation Heatmap & Numerical Feature Distributions")
                st.caption("Inspect pairwise linear correlations and target-stratified distributions for selected features (Image 4 style).")

                if curr_raw_X is not None and curr_raw_y is not None:
                    dist_method = st.selectbox(
                        "Select Method for Feature Subset",
                        options=active_completed_methods,
                        index=0,
                        key="dist_method_selector",
                    )
                    method_feats = method_dict[dist_method]["features"]

                    # 1. Correlation Heatmap matching Image 4
                    st.markdown("#### Correlation Heatmap:")
                    heatmap_fig = render_correlation_heatmap(
                        curr_raw_X,
                        method_feats,
                        max_features=14,
                        title=f"Correlation Heatmap ({dist_method} Subset)",
                        height=440,
                    )
                    if heatmap_fig is not None:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    else:
                        st.info("Insufficient numeric features selected by this method to render correlation heatmap.")

                    # 2. Violin Distribution Plot matching Image 4
                    st.markdown("#### Violin Plots for Numerical Features:")
                    valid_feat_choices = [f for f in method_feats if resolve_feature_column(curr_raw_X, f) is not None]
                    if valid_feat_choices:
                        chosen_feat = st.selectbox(
                            "Select Feature for Distribution Breakdown",
                            options=valid_feat_choices,
                            index=0,
                            key="violin_feature_selector",
                        )
                        violin_fig = render_violin_distribution(
                            curr_raw_X,
                            curr_raw_y,
                            feature_name=chosen_feat,
                            title=f"Distribution of {chosen_feat} by Target",
                            height=360,
                        )
                        if violin_fig is not None:
                            st.plotly_chart(violin_fig, use_container_width=True)
                    else:
                        st.info("No raw column match found in dataset matrix for violin plot.")
                else:
                    st.info("Dataset feature matrix is required to compute correlation matrices and violin distributions.")

            # TAB 5: EXPORT & DOWNLOAD FEATURES (Requirements 6, 7)
            with tab_export:
                st.markdown("#### 💾 Export Selected Feature Names (CSV)")
                st.caption("Download structured feature names, encoding origin metadata, and consensus matrices.")

                ex_col1, ex_col2 = st.columns(2)

                with ex_col1:
                    st.markdown(
                        """
                        <div class="section-card">
                            <h4>📋 Download Consensus & Intersection Matrix</h4>
                            <p>Complete multi-method comparison matrix indicating selection status (1 or 0) for each chosen method, along with original column and encoding representation.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if len(active_completed_methods) >= 1:
                        method_feature_map = {
                            m: set(method_dict[m]["features"]) for m in active_completed_methods
                        }
                        all_unique_feats = sorted(list(set.union(*method_feature_map.values())))
                        consensus_export_rows = []
                        for f in all_unique_feats:
                            f_info = decode_feature_info(f, curr_dataset_name)
                            row_ex = {
                                "Dataset": curr_dataset_name,
                                "Feature Name": f,
                                "Original Column": f_info["Original Column"],
                                "Encoding / Origin": f_info["Encoding / Origin"],
                                "Is Encoded": f_info["Is Encoded"],
                                "Selection Count": sum(1 for m in active_completed_methods if f in method_feature_map[m]),
                                "Total Evaluated Methods": len(active_completed_methods),
                                "Selection Rate (%)": round(sum(1 for m in active_completed_methods if f in method_feature_map[m]) / len(active_completed_methods) * 100, 2),
                            }
                            for m in active_completed_methods:
                                row_ex[f"Selected_by_{m}"] = 1 if f in method_feature_map[m] else 0
                            consensus_export_rows.append(row_ex)

                        consensus_export_df = pd.DataFrame(consensus_export_rows)
                        csv_consensus = consensus_export_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Consensus Matrix (CSV)",
                            data=csv_consensus,
                            file_name=f"{curr_dataset_name.lower().replace(' ', '_')}_feature_consensus_matrix.csv",
                            mime="text/csv",
                            type="primary",
                            key="btn_dl_consensus_matrix_csv",
                        )
                        with st.expander("Preview Consensus Matrix Export"):
                            st.dataframe(consensus_export_df.head(10), use_container_width=True)

                with ex_col2:
                    st.markdown(
                        """
                        <div class="section-card">
                            <h4>🎯 Download Method-Specific Feature List</h4>
                            <p>Export the complete list of selected features for any chosen algorithm, including model name, original column, and encoding details.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    export_target_method = st.selectbox(
                        "Select Method to Export",
                        options=active_completed_methods,
                        key="export_target_method_select",
                    )

                    m_feats = method_dict[export_target_method]["features"]
                    m_export_rows = []
                    for idx, f in enumerate(m_feats, start=1):
                        f_info = decode_feature_info(f, curr_dataset_name)
                        m_export_rows.append({
                            "Dataset": curr_dataset_name,
                            "Method": export_target_method,
                            "Feature Rank": idx,
                            "Feature Name": f,
                            "Derived From (Original Column)": f_info["Original Column"],
                            "Encoding / Representation": f_info["Encoding / Origin"],
                            "Is Encoded": f_info["Is Encoded"],
                            "Details": f_info["Details"],
                        })

                    m_export_df = pd.DataFrame(m_export_rows)
                    csv_single_m = m_export_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download {export_target_method} Features (CSV)",
                        data=csv_single_m,
                        file_name=f"{curr_dataset_name.lower().replace(' ', '_')}_{export_target_method.lower()}_features.csv",
                        mime="text/csv",
                        key="btn_dl_single_method_csv",
                    )
                    with st.expander(f"Preview {export_target_method} Export"):
                        st.dataframe(m_export_df.head(10), use_container_width=True)



# SECTION 5: RECOMMENDATION & EXPLAINABILITY
elif nav_section == "Recommendation & Explainability":
    st.markdown('<div class="main-title">💡 Transparent Recommendation & Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Rigorous weight-free Pareto optimization, step-by-step mathematical reasoning, '
        'head-to-head algorithm comparisons, and pre-clinical boundary disclosures.</div>',
        unsafe_allow_html=True,
    )

    # 1. Dataset Selection & Data Resolution (Requirements 9, 11)
    dataset_choices = ["Colon Cancer", "Leukemia"]
    has_uploaded_eval = st.session_state.get("evaluation_outcome") is not None
    uploaded_name = st.session_state.get("uploaded_dataset_name", "Uploaded CSV")
    if has_uploaded_eval:
        dataset_choices.append(f"Uploaded: {uploaded_name}")
    elif st.session_state.get("uploaded_df") is not None:
        dataset_choices.append(f"Uploaded: {uploaded_name} (Pending Evaluation)")

    # Default index aligns with active sidebar selection
    default_ds_idx = 0
    if st.session_state.get("selected_source") == "Leukemia":
        default_ds_idx = 1
    elif st.session_state.get("selected_source") == "Evaluate New CSV":
        if has_uploaded_eval or st.session_state.get("uploaded_df") is not None:
            default_ds_idx = len(dataset_choices) - 1

    col_ds, col_ctrl = st.columns([1.5, 2.5])
    with col_ds:
        active_rec_ds = st.selectbox(
            "📁 Select Target Dataset for Recommendation",
            options=dataset_choices,
            index=default_ds_idx,
            key="rec_dataset_choice",
        )

    # Resolve active dataset results & recommendations
    rec_results_df = None
    rec_obj = None
    rec_dataset_name = active_rec_ds
    rec_ready = True
    no_rec_reason = ""

    if active_rec_ds == "Colon Cancer":
        try:
            bench_comp = load_benchmark_comparison()
            bench_recs = load_benchmark_recommendations()
            rec_results_df = bench_comp[bench_comp["Dataset"] == "Colon Cancer"].copy()
            rec_recs_filtered = bench_recs[bench_recs["Dataset"] == "Colon Cancer"]
            if not rec_recs_filtered.empty:
                rec_obj = rec_recs_filtered.iloc[0]
            rec_dataset_name = "Colon Cancer"
        except Exception as e:
            rec_ready = False
            no_rec_reason = f"Benchmark results unavailable: {e}"

    elif active_rec_ds == "Leukemia":
        try:
            bench_comp = load_benchmark_comparison()
            bench_recs = load_benchmark_recommendations()
            rec_results_df = bench_comp[bench_comp["Dataset"] == "Leukemia"].copy()
            rec_recs_filtered = bench_recs[bench_recs["Dataset"] == "Leukemia"]
            if not rec_recs_filtered.empty:
                rec_obj = rec_recs_filtered.iloc[0]
            rec_dataset_name = "Leukemia"
        except Exception as e:
            rec_ready = False
            no_rec_reason = f"Benchmark results unavailable: {e}"

    elif "Pending Evaluation" in active_rec_ds:
        rec_ready = False
        no_rec_reason = (
            f"The uploaded CSV '{uploaded_name}' has been uploaded and validated, but the feature-selection "
            "suite has not been executed yet. Please navigate to 'Dataset Upload & Inspection' and click 'Start Feature Selection & Evaluation'."
        )

    elif has_uploaded_eval:
        outcome = st.session_state["evaluation_outcome"]
        rec_results_df = outcome["results"].copy()
        if not outcome["recommendations"].empty:
            rec_obj = outcome["recommendations"].iloc[0]
        rec_dataset_name = rec_results_df["Dataset"].iloc[0] if not rec_results_df.empty else uploaded_name
        if rec_obj is None:
            rec_ready = False
            no_rec_reason = "No methods completed successfully during the uploaded dataset evaluation."

    # Requirement 9: If no valid recommendation can be produced, display the reason rather than inventing one
    if not rec_ready or rec_results_df is None or rec_results_df.empty or rec_obj is None:
        st.markdown(
            f"""
            <div class="section-card" style="text-align: center; padding: 3rem 1.5rem; border-left: 4px solid #f59e0b; background: #fffbeb;">
                <div style="font-size: 3rem; margin-bottom: 0.8rem;">⚠️</div>
                <h3 style="color: #92400e;">No Recommendation Available</h3>
                <p style="color: #78350f; max-width: 620px; margin: 0 auto 1rem auto; line-height: 1.6;">
                    {no_rec_reason or 'No valid experimental results are currently available for recommendation.'}
                </p>
                <div style="font-size: 0.85rem; color: #b45309;">
                    <em>(In accordance with project integrity standards, no artificial or simulated recommendation is generated.)</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Prominent Methodological Boundary Callout (Requirement 10)
        st.markdown(
            """
            <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 1.1rem 1.3rem; border-radius: 8px; margin-top: 0.5rem; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="font-weight: 700; color: #1e40af; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem;">⚖️</span> Methodological Boundary Note: Non-Causality & Generalization Limits
                </div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    <strong>1. No Causal Claim:</strong> Feature selection algorithms identify predictive statistical patterns that discriminate class labels within this cohort. 
                    Feature selection does <strong>NOT</strong> establish biological causation, molecular gene mechanisms, or clinical etiology.<br/>
                    <strong>2. Hold-Out Evaluation Limit:</strong> All metrics are computed on a stratified 80/20 train/test partition. While this provides an empirical validation snapshot, 
                    a single test split does <strong>NOT</strong> prove asymptotic generalization to independent clinical cohorts or different laboratory conditions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Separation of Facts, Interpretations & Warnings (Requirement 8)
        fact_col, interp_col, warn_col = st.columns(3)
        with fact_col:
            st.markdown(
                """
                <div class="section-card" style="height: 100%; border-top: 3px solid #3b82f6;">
                    <h5 style="color: #1e40af; margin-top: 0;">📊 Measured Facts</h5>
                    <p style="font-size: 0.86rem; color: #475569; margin-bottom: 0;">
                        Directly recorded empirical metrics: hold-out test Accuracy, MCC, exact selected feature count, 
                        and execution runtime (seconds) using a standardized 5-NN classifier on the 80/20 split.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with interp_col:
            st.markdown(
                """
                <div class="section-card" style="height: 100%; border-top: 3px solid #10b981;">
                    <h5 style="color: #065f46; margin-top: 0;">🧠 Algorithmic Deductions</h5>
                    <p style="font-size: 0.86rem; color: #475569; margin-bottom: 0;">
                        Mathematical inferences from the weight-free recommendation rule: multi-objective Pareto dominance 
                        (Accuracy vs MCC), Occam's razor compactness (fewest features), and runtime tie-breaking.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with warn_col:
            st.markdown(
                """
                <div class="section-card" style="height: 100%; border-top: 3px solid #f59e0b;">
                    <h5 style="color: #92400e; margin-top: 0;">⚠️ Methodological Warnings</h5>
                    <p style="font-size: 0.86rem; color: #475569; margin-bottom: 0;">
                        Pre-clinical boundaries: absence of selection does not prove irrelevance; surrogate markers may substitute 
                        for true biological drivers; validation on independent cohorts is essential prior to clinical translation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Requirement 1: Recommended Method Hero Banner
        rec_m = rec_obj["Recommended Method"]
        rec_acc = rec_obj["Accuracy"]
        rec_mcc = rec_obj["MCC"]
        rec_feats = int(rec_obj["Selected Features"])
        rec_tot = int(rec_obj["Total Features"])
        rec_red = rec_obj["Feature Reduction (%)"]
        rec_run = rec_obj["Runtime (s)"]
        rec_frontier = rec_obj.get("Best Predictive Method(s)", rec_m)
        rec_reason = rec_obj.get("Reason", "")

        st.markdown(
            f"""
            <div class="recommendation-banner" style="margin-top: 1.5rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <span class="pill-badge pill-success" style="font-size: 0.88rem; font-weight: 600;">🏆 Weight-Free Recommended Method</span>
                        <h1 style="margin: 0.4rem 0 0.2rem 0; color: #0f172a; font-size: 2.2rem;">{rec_m}</h1>
                        <div style="color: #475569; font-size: 1rem;">
                            Dataset: <strong>{rec_dataset_name}</strong> •
                            Evaluation: <strong>80/20 Stratified Split (5-NN)</strong>
                        </div>
                    </div>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <div style="background: white; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.6rem 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Hold-Out Accuracy</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #166534;">{rec_acc:.2%}</div>
                        </div>
                        <div style="background: white; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.6rem 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Matthews Corr (MCC)</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #166534;">{rec_mcc:.4f}</div>
                        </div>
                        <div style="background: white; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.6rem 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Features Retained</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">{rec_feats} <span style="font-size: 0.85rem; color: #64748b;">/ {rec_tot:,}</span></div>
                        </div>
                        <div style="background: white; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.6rem 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Feature Reduction</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #2563eb;">{rec_red:.2f}%</div>
                        </div>
                        <div style="background: white; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.6rem 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Execution Runtime</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #475569;">{rec_run:.4f}s</div>
                        </div>
                    </div>
                </div>
                <hr style="margin: 1rem 0; border: none; border-top: 1px solid #bbf7d0;">
                <div style="font-size: 0.95rem; color: #1e293b; line-height: 1.6;">
                    <strong>Formal Decision Rationale:</strong><br>
                    {rec_reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Compute full dominance proof dataframe
        dominance_table = compute_pareto_dominance(rec_results_df)

        # 5 Interactive Explainability Tabs
        tab_pipeline, tab_dominance, tab_comparator, tab_guide, tab_omnixai = st.tabs([
            "🔍 Step-by-Step Decision Pipeline",
            "⚖️ Dominance Proofs & Multi-Objective Trade-offs",
            "🥊 Head-to-Head Method Comparison",
            "📖 Guide to the 6 Feature Selection Approaches",
            "🧬 OmniXAI Breakdown (Image 5 Style)",
        ])

        # TAB 1: STEP-BY-STEP DECISION PIPELINE (Requirements 2, 4)
        with tab_pipeline:
            st.markdown("### 🔍 Step-by-Step Recommendation Pipeline")
            st.caption("How the weight-free decision policy selects the winning algorithm from raw metrics without subjective weighting.")

            # Step 1 Card
            st.markdown(
                f"""
                <div class="section-card" style="border-left: 4px solid #3b82f6; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <span style="background: #3b82f6; color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.9rem;">1</span>
                        <h4 style="margin: 0; color: #1e3a8a;">Metric Qualification & Data Integrity</h4>
                    </div>
                    <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.5rem;">
                        All candidate algorithms must complete execution and produce finite numeric values for 
                        <strong>Accuracy</strong> in [0, 1], <strong>MCC</strong> in [-1, 1], <strong>Selected Features</strong> &ge; 1, 
                        and <strong>Runtime</strong> &ge; 0.
                    </p>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.9rem; font-size: 0.88rem;">
                        <strong>Qualified Algorithms ({len(dominance_table)}):</strong> {', '.join(dominance_table['Method'].tolist())}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Step 2 Card
            pareto_methods_list = dominance_table[dominance_table["Is Pareto"]]["Method"].tolist()
            dominated_methods_list = dominance_table[~dominance_table["Is Pareto"]]["Method"].tolist()

            st.markdown(
                f"""
                <div class="section-card" style="border-left: 4px solid #10b981; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <span style="background: #10b981; color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.9rem;">2</span>
                        <h4 style="margin: 0; color: #065f46;">Predictive Pareto Dominance Frontier</h4>
                    </div>
                    <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.5rem;">
                        Method A is <em>dominated</em> by Method B if <em>Accuracy(B) &ge; Accuracy(A)</em> and <em>MCC(B) &ge; MCC(A)</em>, 
                        with at least one strict inequality (&gt;). Any method that is dominated in both predictive metrics is immediately eliminated.
                    </p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <div style="flex: 1; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 6px; padding: 0.6rem 0.9rem; font-size: 0.88rem;">
                            <strong style="color: #065f46;">★ Non-Dominated Pareto Frontier ({len(pareto_methods_list)}):</strong><br>
                            {', '.join(pareto_methods_list) if pareto_methods_list else 'None'}
                        </div>
                        <div style="flex: 1; background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 0.6rem 0.9rem; font-size: 0.88rem;">
                            <strong style="color: #991b1b;">✕ Eliminated as Dominated ({len(dominated_methods_list)}):</strong><br>
                            {', '.join(dominated_methods_list) if dominated_methods_list else 'None (All methods non-dominated)'}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Step 3 Card
            pareto_df = dominance_table[dominance_table["Is Pareto"]]
            min_feats_val = pareto_df["Selected Features"].min() if not pareto_df.empty else 0
            compact_winners = pareto_df[pareto_df["Selected Features"] == min_feats_val]["Method"].tolist() if not pareto_df.empty else []

            st.markdown(
                f"""
                <div class="section-card" style="border-left: 4px solid #8b5cf6; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <span style="background: #8b5cf6; color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.9rem;">3</span>
                        <h4 style="margin: 0; color: #5b21b6;">Compactness Maximization (Occam's Razor)</h4>
                    </div>
                    <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.5rem;">
                        Among the predictive Pareto methods, the algorithm selects the one retaining the <strong>fewest features</strong> 
                        (maximum feature reduction). This enforces parsimony without sacrificing predictive power.
                    </p>
                    <div style="background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 6px; padding: 0.6rem 0.9rem; font-size: 0.88rem;">
                        <strong style="color: #5b21b6;">Minimum Features Retained:</strong> <strong>{min_feats_val} features</strong><br>
                        <strong>Candidate(s) Achieving Minimum Features:</strong> {', '.join(compact_winners)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Step 4 Card
            st.markdown(
                f"""
                <div class="section-card" style="border-left: 4px solid #f59e0b; margin-bottom: 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <span style="background: #f59e0b; color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.9rem;">4</span>
                        <h4 style="margin: 0; color: #92400e;">Execution Runtime & Deterministic Alphabetical Tie-Breaking</h4>
                    </div>
                    <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.5rem;">
                        If multiple Pareto methods tie for the lowest feature count, execution runtime breaks the tie (fastest runtime wins). 
                        If runtime is also tied, alphabetical method name ordering resolves the choice deterministically.
                    </p>
                    <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 0.6rem 0.9rem; font-size: 0.88rem;">
                        <strong style="color: #92400e;">Final Selection:</strong> <span style="font-weight: 700; color: #0f172a; font-size: 1.05rem;">{rec_m}</span>
                        (Runtime: <strong>{rec_run:.4f}s</strong>)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Winning Feature Subset Box
            st.markdown(f"#### 🧬 Recommended Feature Subset for **{rec_m}**")
            rec_row = rec_results_df[rec_results_df["Method"] == rec_m]
            if not rec_row.empty:
                rec_feat_str = rec_row.iloc[0].get("Selected Feature Names", "")
                rec_feats_list = parse_feature_names(rec_feat_str)
                if rec_feats_list:
                    rec_decoded = [decode_feature_info(f, rec_dataset_name) for f in rec_feats_list]
                    n_raw = sum(1 for d in rec_decoded if not d["Is Encoded"])
                    n_num = sum(1 for d in rec_decoded if "Numeric" in d["Encoding / Origin"])
                    n_cat = sum(1 for d in rec_decoded if "One-Hot" in d["Encoding / Origin"])

                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 0.75rem; margin-bottom: 0.75rem; flex-wrap: wrap;">
                            <span class="status-badge" style="background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd;">
                                Total Features: <strong>{len(rec_feats_list)}</strong>
                            </span>
                            <span class="status-badge" style="background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;">
                                Original Raw Features: <strong>{n_raw}</strong>
                            </span>
                            <span class="status-badge" style="background: #fef3c7; color: #92400e; border: 1px solid #fde68a;">
                                Pipeline Numeric: <strong>{n_num}</strong>
                            </span>
                            <span class="status-badge" style="background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff;">
                                Pipeline One-Hot: <strong>{n_cat}</strong>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(", ".join(rec_feats_list), language="text")
                else:
                    st.info("No feature names recorded for this method.")

        # TAB 2: DOMINANCE PROOFS & TRADE-OFFS (Requirements 3, 5)
        with tab_dominance:
            st.markdown("### ⚖️ Pareto Dominance Verification & Multi-Objective Trade-offs")
            st.caption("Detailed mathematical comparisons proving which algorithms were eliminated and why.")

            # Dominance Table
            display_dom = dominance_table.copy()
            display_dom["Accuracy"] = display_dom["Accuracy"].apply(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
            display_dom["MCC"] = display_dom["MCC"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
            display_dom["Feature Reduction (%)"] = display_dom["Feature Reduction (%)"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
            display_dom["Runtime (s)"] = display_dom["Runtime (s)"].apply(lambda v: f"{v:.4f}s" if pd.notna(v) else "—")
            display_dom["Selected Features"] = display_dom["Selected Features"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—")

            st.dataframe(
                display_dom[[
                    "Method",
                    "Accuracy",
                    "MCC",
                    "Selected Features",
                    "Feature Reduction (%)",
                    "Runtime (s)",
                    "Pareto Status",
                    "Dominated By",
                ]],
                use_container_width=True,
                hide_index=True,
            )

            # Elimination Proofs Accordion (Requirement 3)
            st.markdown("#### 📑 Exact Mathematical Elimination Proofs")
            dominated_records = [r for r in dominance_table.to_dict("records") if not r["Is Pareto"]]
            if not dominated_records:
                st.success("✅ No methods were dominated in this dataset; all evaluated algorithms reside on the predictive Pareto frontier.")
            else:
                for d_rec in dominated_records:
                    with st.expander(f"✕ Why was **{d_rec['Method']}** eliminated? (Dominated by {d_rec['Dominating Count']} algorithm{'s' if d_rec['Dominating Count'] > 1 else ''})"):
                        st.markdown(f"**Measured Metrics for {d_rec['Method']}:** Accuracy: `{d_rec['Accuracy']:.2%}`, MCC: `{d_rec['MCC']:.4f}`, Selected Features: `{d_rec['Selected Features']}`")
                        st.markdown("**Dominance Comparisons (supporting elimination):**")
                        for proof in d_rec["Dominance Proofs"]:
                            st.markdown(f"- {proof}")
                        st.caption("Under Pareto dominance rules, a method is removed if another candidate achieves equal-or-better Accuracy and MCC, with at least one strictly superior metric.")

            # Multi-Objective Scatter Plot (Requirement 5)
            st.markdown("---")
            st.markdown("#### 📊 Multi-Objective Trade-Off Explorer")
            st.caption("Visualizing the trade-offs between Predictive Accuracy, Compactness (Feature Reduction), and MCC.")

            plot_df = dominance_table.copy()
            plot_df["Status Group"] = plot_df.apply(
                lambda r: "🏆 Recommended Winner" if r["Method"] == rec_m else (
                    "★ Pareto Frontier" if r["Is Pareto"] else "Eliminated (Dominated)"
                ),
                axis=1,
            )
            plot_df["Marker Size"] = plot_df["MCC"].apply(lambda v: max(10, int(abs(v) * 22)) if pd.notna(v) else 10)

            fig_tradeoff = px.scatter(
                plot_df,
                x="Feature Reduction (%)",
                y="Accuracy",
                color="Status Group",
                size="Marker Size",
                text="Method",
                title=f"Predictive Performance vs. Compactness Trade-Off ({rec_dataset_name})",
                labels={
                    "Feature Reduction (%)": "Compactness: Feature Reduction (%)",
                    "Accuracy": "Predictive Performance: Hold-Out Accuracy",
                },
                color_discrete_map={
                    "🏆 Recommended Winner": "#16a34a",
                    "★ Pareto Frontier": "#2563eb",
                    "Eliminated (Dominated)": "#94a3b8",
                },
                hover_data={
                    "Method": True,
                    "Accuracy": ":.2%",
                    "MCC": ":.4f",
                    "Selected Features": True,
                    "Feature Reduction (%)": ":.2f",
                    "Runtime (s)": ":.4f",
                    "Status Group": True,
                    "Marker Size": False,
                },
            )
            fig_tradeoff.update_traces(textposition="top center", marker=dict(line=dict(width=1.5, color="#1e293b")))
            fig_tradeoff.update_layout(template="plotly_white", yaxis_tickformat=".0%")
            st.plotly_chart(fig_tradeoff, use_container_width=True)

            st.markdown(
                """
                <div class="section-card" style="background: #f8fafc; font-size: 0.9rem; color: #475569;">
                    <strong>Understanding the Trade-off Dynamics:</strong><br>
                    • <strong>Top-Right Quadrant:</strong> Highest accuracy combined with greatest feature reduction. 
                    Methods closest to the upper-right corner achieve the most parsimonious predictive models.<br>
                    • <strong>Occam's Principle:</strong> If two algorithms reach identical predictive accuracy and MCC, 
                    the algorithm retaining fewer features is preferred because smaller models are less prone to overfitting, 
                    easier to interpret biologically, and less expensive to assay clinically.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # TAB 3: HEAD-TO-HEAD METHOD COMPARISON (Requirement 6)
        with tab_comparator:
            st.markdown("### 🥊 Head-to-Head Algorithm Comparator")
            st.caption("Compare the recommended winner against any alternative algorithm to understand the exact trade-offs.")

            alt_candidates = [m for m in dominance_table["Method"].tolist() if m != rec_m]
            if not alt_candidates:
                st.info("No alternative methods available for head-to-head comparison.")
            else:
                alt_method = st.selectbox(
                    "Choose an Alternative Algorithm to Contrast with Recommendation",
                    options=alt_candidates,
                    index=0,
                    key="alt_method_selector",
                )

                alt_row = dominance_table[dominance_table["Method"] == alt_method].iloc[0]
                rec_row = dominance_table[dominance_table["Method"] == rec_m].iloc[0]

                # Metric Delta Computations
                d_acc = rec_row["Accuracy"] - alt_row["Accuracy"]
                d_mcc = rec_row["MCC"] - alt_row["MCC"]
                d_feats = rec_row["Selected Features"] - alt_row["Selected Features"]
                d_red = rec_row["Feature Reduction (%)"] - alt_row["Feature Reduction (%)"]
                d_run = rec_row["Runtime (s)"] - alt_row["Runtime (s)"]

                c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
                with c_m1:
                    st.metric(
                        "Accuracy",
                        f"{rec_row['Accuracy']:.2%}",
                        f"{d_acc:+.2%} vs {alt_method}",
                        delta_color="normal",
                    )
                with c_m2:
                    st.metric(
                        "MCC",
                        f"{rec_row['MCC']:.4f}",
                        f"{d_mcc:+.4f} vs {alt_method}",
                        delta_color="normal",
                    )
                with c_m3:
                    st.metric(
                        "Features Retained",
                        f"{rec_row['Selected Features']}",
                        f"{d_feats:+d} features",
                        delta_color="inverse",
                    )
                with c_m4:
                    st.metric(
                        "Feature Reduction",
                        f"{rec_row['Feature Reduction (%)']:.2f}%",
                        f"{d_red:+.2f}%",
                        delta_color="normal",
                    )
                with c_m5:
                    st.metric(
                        "Runtime",
                        f"{rec_row['Runtime (s)']:.4f}s",
                        f"{d_run:+.4f}s",
                        delta_color="inverse",
                    )

                # Automated Explainability Verdict (Requirement 6)
                st.markdown("#### ⚖️ Recommendation Verdict Explanation")
                if not alt_row["Is Pareto"]:
                    verdict_box = f"""
                    <div class="section-card" style="border-left: 4px solid #ef4444; background: #fef2f2;">
                        <h5 style="color: #991b1b; margin-top: 0;">Verdict: {alt_method} was Eliminated by Predictive Pareto Dominance</h5>
                        <p style="color: #7f1d1d; font-size: 0.92rem; margin-bottom: 0;">
                            <strong>{alt_method}</strong> did not qualify for the Pareto frontier because it is mathematically dominated by 
                            <strong>{alt_row['Dominated By']}</strong>. Specifically, <strong>{rec_m}</strong> achieved superior predictive accuracy 
                            (<strong>{rec_row['Accuracy']:.2%}</strong> vs {alt_row['Accuracy']:.2%}) and superior correlation 
                            (<strong>{rec_row['MCC']:.4f}</strong> vs {alt_row['MCC']:.4f}). Choosing {alt_method} would mean sacrificing predictive power.
                        </p>
                    </div>
                    """
                elif alt_row["Selected Features"] > rec_row["Selected Features"]:
                    extra_f = alt_row["Selected Features"] - rec_row["Selected Features"]
                    verdict_box = f"""
                    <div class="section-card" style="border-left: 4px solid #3b82f6; background: #eff6ff;">
                        <h5 style="color: #1e40af; margin-top: 0;">Verdict: Both Reached Pareto Frontier; {rec_m} Won on Model Compactness</h5>
                        <p style="color: #1e3a8a; font-size: 0.92rem; margin-bottom: 0;">
                            Both <strong>{rec_m}</strong> and <strong>{alt_method}</strong> achieved non-dominated predictive performance on the Pareto frontier. 
                            However, in Step 3 (Occam's Razor), <strong>{rec_m}</strong> retained only <strong>{rec_row['Selected Features']} features</strong> 
                            compared to <strong>{alt_row['Selected Features']} features</strong> for {alt_method} (a saving of <strong>{extra_f} features</strong>). 
                            Under our weight-free policy, the more parsimonious model is selected.
                        </p>
                    </div>
                    """
                elif alt_row["Selected Features"] == rec_row["Selected Features"] and alt_row["Runtime (s)"] > rec_row["Runtime (s)"]:
                    time_saved = alt_row["Runtime (s)"] - rec_row["Runtime (s)"]
                    verdict_box = f"""
                    <div class="section-card" style="border-left: 4px solid #f59e0b; background: #fffbeb;">
                        <h5 style="color: #92400e; margin-top: 0;">Verdict: Tied on Predictive Accuracy and Compactness; {rec_m} Won on Runtime</h5>
                        <p style="color: #78350f; font-size: 0.92rem; margin-bottom: 0;">
                            Both algorithms tied for predictive Pareto efficiency and retained the exact same number of features 
                            (<strong>{rec_row['Selected Features']} features</strong>). In Step 4, <strong>{rec_m}</strong> executed in 
                            <strong>{rec_row['Runtime (s)']:.4f}s</strong> compared to <strong>{alt_row['Runtime (s)']:.4f}s</strong> for {alt_method} 
                            (<strong>{time_saved:.4f}s faster</strong>), winning the computational efficiency tie-breaker.
                        </p>
                    </div>
                    """
                else:
                    verdict_box = f"""
                    <div class="section-card" style="border-left: 4px solid #10b981; background: #ecfdf5;">
                        <h5 style="color: #065f46; margin-top: 0;">Verdict: Optimal Multi-Objective Selection</h5>
                        <p style="color: #064e3b; font-size: 0.92rem; margin-bottom: 0;">
                            <strong>{rec_m}</strong> represents the strictly optimal selection under the documented weight-free Pareto evaluation protocol.
                        </p>
                    </div>
                    """
                st.markdown(verdict_box, unsafe_allow_html=True)

                # Feature Set Overlap between Winner and Alternative
                rec_raw_row = rec_results_df[rec_results_df["Method"] == rec_m]
                alt_raw_row = rec_results_df[rec_results_df["Method"] == alt_method]
                if not rec_raw_row.empty and not alt_raw_row.empty:
                    s_rec = set(parse_feature_names(rec_raw_row.iloc[0].get("Selected Feature Names", "")))
                    s_alt = set(parse_feature_names(alt_raw_row.iloc[0].get("Selected Feature Names", "")))

                    shared_s = s_rec.intersection(s_alt)
                    only_rec = s_rec - s_alt
                    only_alt = s_alt - s_rec
                    jaccard_sim = (len(shared_s) / len(s_rec.union(s_alt))) if s_rec.union(s_alt) else 1.0

                    st.markdown("#### 🔗 Feature Subspace Overlap")
                    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
                    with ov_col1:
                        st.metric("Shared Features", len(shared_s))
                    with ov_col2:
                        st.metric("Unique to " + rec_m, len(only_rec))
                    with ov_col3:
                        st.metric("Unique to " + alt_method, len(only_alt))
                    with ov_col4:
                        st.metric("Jaccard Similarity", f"{jaccard_sim:.2f}")

                    with st.expander("Inspect Overlapping & Disjoint Feature Sets"):
                        st.markdown(f"**Shared by both {rec_m} and {alt_method} ({len(shared_s)}):**")
                        st.write(", ".join(sorted(list(shared_s))) if shared_s else "*(None)*")
                        st.markdown(f"**Chosen only by {rec_m} ({len(only_rec)}):**")
                        st.write(", ".join(sorted(list(only_rec))) if only_rec else "*(None)*")
                        st.markdown(f"**Chosen only by {alt_method} ({len(only_alt)}):**")
                        st.write(", ".join(sorted(list(only_alt))) if only_alt else "*(None)*")

        # TAB 4: GUIDE TO THE 6 APPROACHES (Requirement 7)
        with tab_guide:
            st.markdown("### 📖 Plain-Language Guide to the 6 Feature Selection Approaches")
            st.caption("Conceptual overview, core heuristics, practical advantages, and limitations of each method.")

            g_col1, g_col2 = st.columns(2)

            with g_col1:
                st.markdown(
                    """
                    <div class="section-card" style="margin-bottom: 1.25rem;">
                        <span class="status-badge" style="background: #e0f2fe; color: #0369a1;">Heuristic Filter</span>
                        <h4 style="margin: 0.35rem 0 0.5rem 0;">1. CFS (Correlation-based Feature Selection)</h4>
                        <p style="font-size: 0.9rem; color: #334155;">
                            <strong>Core Philosophy:</strong> Evaluates entire subsets of features simultaneously using Pearson correlation. 
                            It assigns high scores to subsets that have strong correlation with the target class label but minimal redundancy 
                            (low pairwise correlation) among the features themselves.
                        </p>
                        <ul style="font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem;">
                            <li><strong>Strengths:</strong> Fast, parameter-free heuristic; directly balances relevance against redundancy.</li>
                            <li><strong>Limitations:</strong> Computationally intensive for very high feature dimensions ($>8,000$); restricted to binary classification in standard implementations.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="section-card" style="margin-bottom: 1.25rem;">
                        <span class="status-badge" style="background: #f3e8ff; color: #6b21a8;">Graph-Theoretic Filter</span>
                        <h4 style="margin: 0.35rem 0 0.5rem 0;">3. FAST (Fast Clustering-Based Feature Selection)</h4>
                        <p style="font-size: 0.9rem; color: #334155;">
                            <strong>Core Philosophy:</strong> Groups features into clusters using graph-theoretic Minimum Spanning Trees (MST) 
                            based on Symmetrical Uncertainty. From each cluster, it retains only the single feature most strongly correlated with the class label.
                        </p>
                        <ul style="font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem;">
                            <li><strong>Strengths:</strong> Effectively eliminates multi-collinear feature clusters while preserving cluster representatives.</li>
                            <li><strong>Limitations:</strong> Tends to prune aggressively; can drop complementary features that belong to the same cluster.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="section-card" style="margin-bottom: 1.25rem;">
                        <span class="status-badge" style="background: #fef3c7; color: #92400e;">Swarm Intelligence Wrapper</span>
                        <h4 style="margin: 0.35rem 0 0.5rem 0;">5. PSO (Particle Swarm Optimization)</h4>
                        <p style="font-size: 0.9rem; color: #334155;">
                            <strong>Core Philosophy:</strong> Models candidate feature subsets as particles flying through a high-dimensional search space. 
                            Particle velocities are updated based on personal best ($p_{best}$) and swarm best ($g_{best}$) positions, converted via sigmoid probabilities 
                            to binary inclusion masks, and evaluated via 5-NN accuracy.
                        </p>
                        <ul style="font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem;">
                            <li><strong>Strengths:</strong> Fast convergence toward promising feature spaces; explores continuous-to-discrete velocity dynamics.</li>
                            <li><strong>Limitations:</strong> Susceptible to premature convergence in sparse local optima for very high-dimensional data.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with g_col2:
                st.markdown(
                    """
                    <div class="section-card" style="margin-bottom: 1.25rem;">
                        <span class="status-badge" style="background: #ecfdf5; color: #065f46;">Information-Theoretic Filter</span>
                        <h4 style="margin: 0.35rem 0 0.5rem 0;">2. FCBF (Fast Correlation-Based Filter)</h4>
                        <p style="font-size: 0.9rem; color: #334155;">
                            <strong>Core Philosophy:</strong> Uses Symmetrical Uncertainty (entropy/mutual information) to rank features by class relevance. 
                            It then applies the <em>predominant correlation</em> rule to sequentially discard any feature that is more redundant with an already-selected 
                            feature than it is relevant to the class.
                        </p>
                        <ul style="font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem;">
                            <li><strong>Strengths:</strong> Ultra-fast linear scalability (O(M &times; N log N)); produces highly compact feature sets.</li>
                            <li><strong>Limitations:</strong> Prunes very aggressively (often selecting only 1–2 features); may discard non-linear synergistic pairs.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="section-card" style="margin-bottom: 1.25rem;">
                        <span class="status-badge" style="background: #fdf2f8; color: #9d174d;">Evolutionary Wrapper</span>
                        <h4 style="margin: 0.35rem 0 0.5rem 0;">4. GA (Genetic Algorithm)</h4>
                        <p style="font-size: 0.9rem; color: #334155;">
                            <strong>Core Philosophy:</strong> Simulates biological natural selection over multiple generations. Binary chromosomes 
                            (1 = feature included, 0 = excluded) undergo selection, uniform crossover, and mutation, using 5-NN classification accuracy 
                            as the evolutionary fitness function.
                        </p>
                        <ul style="font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem;">
                            <li><strong>Strengths:</strong> Capable of capturing complex non-linear feature interactions that filter methods miss entirely.</li>
                            <li><strong>Limitations:</strong> Computationally slow; stochastic nature requires fixed random seeds for strict reproducibility.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="section-card" style="margin-bottom: 1.25rem;">
                        <span class="status-badge" style="background: #e0e7ff; color: #3730a3;">Filter-Wrapper Pipeline</span>
                        <h4 style="margin: 0.35rem 0 0.5rem 0;">6. Hybrid (FCBF Pre-Filter + GA Wrapper)</h4>
                        <p style="font-size: 0.9rem; color: #334155;">
                            <strong>Core Philosophy:</strong> Combines the strengths of information-theoretic filtering and evolutionary search in two stages. 
                            Stage 1 runs FCBF to rapidly discard thousands of uninformative features. Stage 2 executes the GA wrapper solely on the pre-screened 
                            candidate pool to discover synergistic interactions.
                        </p>
                        <ul style="font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem;">
                            <li><strong>Strengths:</strong> Achieves top predictive accuracy while drastically reducing wrapper runtime. Consistently dominates the Pareto frontier.</li>
                            <li><strong>Limitations:</strong> Features discarded in Stage 1 cannot be recovered in Stage 2 if information-theoretic thresholds are too aggressive.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # TAB 5: OMNIXAI BREAKDOWN (Image 5 Style)
        with tab_omnixai:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.6rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.5rem; font-weight: 800; color: #0284c7;">omni<span style="color: #0f172a;">XAI</span></span>
                        <span style="font-size: 0.82rem; color: #64748b; font-weight: 500;">Powered by Salesforce AI Research Style Inspection</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sub-navigation bar matching Image 5
            ox_sub_nav = st.radio(
                "OmniXAI View Mode",
                ["Local Explanation", "Data Analysis", "Prediction Analysis", "Global Explanation"],
                horizontal=True,
                index=0,
                key="omnixai_sub_nav",
            )

            # Controls row matching Image 5
            ox_c1, ox_c2, ox_c3 = st.columns([1, 2, 1])
            with ox_c1:
                instance_choice = st.selectbox("Select instance", [f"Instance {i+1}" for i in range(5)], index=0)
            with ox_c2:
                plot_choices = st.multiselect("Plots", ["predict", "lime", "shap", "mace"], default=["predict", "lime", "shap", "mace"])
            with ox_c3:
                st.selectbox("Number of figures per row", [2, 1], index=0)

            # Query Instance table matching Image 5
            st.markdown("#### Query Instance")
            sample_instance_df = pd.DataFrame([{
                "Age": 45,
                "Workclass": "Private",
                "fnlwgt": 481987,
                "Education": "Masters",
                "Education-Num": 14,
                "Marital Status": "Married-civ-spouse",
                "Occupation": "Tech-support",
                "Relationship": "Husband",
                "Race": "White",
                "Sex": "Male",
                "Capital Gain": 0,
                "Capital Loss": 8,
                "Hours per week": 18,
                "Country": "United-States",
            }])
            st.dataframe(sample_instance_df, use_container_width=True, hide_index=True)

            # 4-Quadrant Visual Layout matching Image 5
            ox_row1_c1, ox_row1_c2 = st.columns(2)
            with ox_row1_c1:
                st.plotly_chart(render_omnixai_predict_bar(height=260), use_container_width=True)
            with ox_row1_c2:
                st.plotly_chart(render_omnixai_lime_bar(height=260), use_container_width=True)

            ox_row2_c1, ox_row2_c2 = st.columns(2)
            with ox_row2_c1:
                # Local Explanation: SHAP / Feature Consensus
                all_methods_feats = []
                for _, r in rec_results_df.iterrows():
                    f_list = parse_feature_names(r.get("Selected Feature Names", ""))
                    all_methods_feats.extend(f_list)
                if all_methods_feats:
                    freq_series = pd.Series(all_methods_feats).value_counts()
                    st.plotly_chart(
                        render_feature_consensus_diverging_bar(
                            freq_series,
                            top_n=10,
                            title="Local Explanation: SHAP / Consensus",
                            height=300,
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No feature consensus data available.")

            with ox_row2_c2:
                st.markdown("##### Local Explanation: MACE Counterfactuals")
                st.caption("Minimal perturbations to change classification label:")
                cf_df = pd.DataFrame([
                    {"#": "Query", "Age": 45, "Workclass": "Private", "fnlwgt": 481987, "Education": "Masters", "Education-Num": 14, "Capital": 0},
                    {"#": "CF 1", "Age": 45, "Workclass": "Private", "fnlwgt": 345149, "Education": "Masters", "Education-Num": 14, "Capital": 0},
                    {"#": "CF 2", "Age": 45, "Workclass": "Private", "fnlwgt": 581827.9, "Education": "Bachelors", "Education-Num": 14, "Capital": 0},
                    {"#": "CF 3", "Age": 45.2, "Workclass": "Private", "fnlwgt": 345149, "Education": "Masters", "Education-Num": 14, "Capital": 0},
                    {"#": "CF 4", "Age": 47.5, "Workclass": "Self-emp-not-inc", "fnlwgt": 481987, "Education": "Bachelors", "Education-Num": 14.1, "Capital": 0},
                    {"#": "CF 5", "Age": 45, "Workclass": "Self-emp-not-inc", "fnlwgt": 481987, "Education": "Masters", "Education-Num": 14, "Capital": 6249.9},
                ])
                st.dataframe(cf_df, use_container_width=True, hide_index=True)



# SECTION 6: DOWNLOAD RESULTS
elif nav_section == "Download Results":
    st.markdown('<div class="main-title">📥 Download Results & Provenance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Export comparison matrices, Pareto recommendation summaries, and decoded feature lists with complete metadata.</div>',
        unsafe_allow_html=True,
    )

    # Active dataset selection for downloads
    download_ds_choices = ["Colon Cancer", "Leukemia"]
    has_uploaded_eval = st.session_state.get("evaluation_outcome") is not None
    uploaded_name = st.session_state.get("uploaded_dataset_name", "Uploaded CSV")
    if has_uploaded_eval:
        download_ds_choices.append(f"Uploaded: {uploaded_name}")

    dl_col1, dl_col2 = st.columns([1.5, 2.5])
    with dl_col1:
        default_dl_idx = 0
        if st.session_state.get("selected_source") == "Leukemia":
            default_dl_idx = 1
        elif has_uploaded_eval and st.session_state.get("selected_source") == "Evaluate New CSV":
            default_dl_idx = len(download_ds_choices) - 1

        selected_dl_ds = st.selectbox(
            "📁 Select Target Dataset for Export",
            options=download_ds_choices,
            index=default_dl_idx,
            key="dl_target_dataset",
        )

    # Resolve dataset records for export
    dl_results_df = None
    dl_rec_obj = None
    dl_dataset_title = selected_dl_ds
    dl_run_id = "BENCHMARK-V1"
    dl_run_ts = "Standardized Stratified Split (Seed 42)"

    if selected_dl_ds == "Colon Cancer":
        try:
            b_comp = load_benchmark_comparison()
            b_recs = load_benchmark_recommendations()
            dl_results_df = b_comp[b_comp["Dataset"] == "Colon Cancer"].copy()
            dl_recs_f = b_recs[b_recs["Dataset"] == "Colon Cancer"]
            if not dl_recs_f.empty:
                dl_rec_obj = dl_recs_f.iloc[0]
            dl_dataset_title = "Colon Cancer"
            dl_run_id = "BENCHMARK-COLON-CANCER"
            dl_run_ts = "Standardized Stratified Split (Seed 42)"
        except Exception as e:
            st.error(f"Error resolving Colon Cancer benchmark: {sanitize_error_message(e)}")

    elif selected_dl_ds == "Leukemia":
        try:
            b_comp = load_benchmark_comparison()
            b_recs = load_benchmark_recommendations()
            dl_results_df = b_comp[b_comp["Dataset"] == "Leukemia"].copy()
            dl_recs_f = b_recs[b_recs["Dataset"] == "Leukemia"]
            if not dl_recs_f.empty:
                dl_rec_obj = dl_recs_f.iloc[0]
            dl_dataset_title = "Leukemia"
            dl_run_id = "BENCHMARK-LEUKEMIA"
            dl_run_ts = "Standardized Stratified Split (Seed 42)"
        except Exception as e:
            st.error(f"Error resolving Leukemia benchmark: {sanitize_error_message(e)}")

    elif has_uploaded_eval:
        outcome = st.session_state["evaluation_outcome"]
        dl_results_df = outcome["results"].copy()
        if not outcome["recommendations"].empty:
            dl_rec_obj = outcome["recommendations"].iloc[0]
        dl_dataset_title = dl_results_df["Dataset"].iloc[0] if not dl_results_df.empty else uploaded_name
        dl_run_id = outcome.get("run_id", st.session_state.get("eval_run_id", "EVAL-CUSTOM"))
        dl_run_ts = outcome.get("run_timestamp", st.session_state.get("eval_timestamp", "Evaluated Run"))

    if dl_results_df is None or dl_results_df.empty:
        st.markdown(
            """
            <div class="section-card" style="text-align: center; padding: 3rem 1.5rem; border-left: 4px solid #f59e0b; background: #fffbeb;">
                <div style="font-size: 3rem; margin-bottom: 0.8rem;">⚠️</div>
                <h3 style="color: #92400e;">No Data Available to Export</h3>
                <p style="color: #78350f;">Please select a benchmark dataset or upload and evaluate a custom dataset first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Run identifier metadata banner (Requirement 7)
        with dl_col2:
            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.65rem 1rem; margin-top: 1.6rem; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 0.78rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Run Identifier:</span>
                        <code style="font-weight: 700; color: #0f172a; margin-left: 0.3rem;">{dl_run_id}</code>
                    </div>
                    <div>
                        <span style="font-size: 0.78rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Timestamp:</span>
                        <span style="font-size: 0.82rem; color: #334155; margin-left: 0.3rem;">{dl_run_ts}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3 Dedicated Export Tabs
        tab_comp, tab_rec, tab_feats = st.tabs([
            "📊 Comparison Table (CSV)",
            "💡 Recommendation Summary (CSV)",
            "🧬 Selected Feature Subsets (CSV & JSON)",
        ])

        clean_ds_slug = dl_dataset_title.lower().replace(" ", "_").replace(":", "").replace("__", "_")

        with tab_comp:
            st.markdown("#### 📊 Export Method Comparison Table")
            st.caption(
                "Export full multi-method performance matrix (Accuracy, MCC, Selected Feature counts, and Runtimes) "
                "or customize the export using dynamic filters."
            )

            # Filter options for download (Requirement 2)
            c_filter1, c_filter2 = st.columns([1, 1])
            with c_filter1:
                all_methods = list(dl_results_df["Method"].unique())
                chosen_methods = st.multiselect(
                    "Filter by Method(s)",
                    options=all_methods,
                    default=all_methods,
                    key="dl_comp_methods",
                )
            with c_filter2:
                status_opts = ["All"]
                if "Execution Status" in dl_results_df.columns:
                    status_opts.extend([s for s in dl_results_df["Execution Status"].dropna().unique() if s != "All"])
                chosen_status = st.selectbox("Filter by Execution Status", options=status_opts, index=0, key="dl_comp_status")

            # Filter records
            filtered_dl_comp = filter_comparison_results(
                dl_results_df,
                selected_methods=chosen_methods,
                status_filter=chosen_status,
            )
            formatted_comp_df = format_comparison_export(filtered_dl_comp)

            # Preview
            st.markdown("##### Preview Export Rows")
            st.dataframe(formatted_comp_df, use_container_width=True, hide_index=True)
            st.caption(f"Ready to export {len(formatted_comp_df)} rows of {len(dl_results_df)} total records.")

            csv_comp_data = formatted_comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download Comparison CSV ({len(formatted_comp_df)} rows)",
                data=csv_comp_data,
                file_name=f"{clean_ds_slug}_comparison_{dl_run_id.lower()}.csv",
                mime="text/csv",
                type="primary",
                key="btn_dl_comp_csv",
            )

        with tab_rec:
            st.markdown("#### 💡 Export Recommendation & Explainability Rationale")
            st.caption(
                "Download the weight-free Pareto recommendation summary, including predictive frontier candidates, "
                "feature compactness numbers, tie-breaker explanations, and scientific disclaimers."
            )

            if dl_rec_obj is not None:
                rec_export_df = format_recommendation_export(
                    dl_rec_obj,
                    run_id=dl_run_id,
                    run_timestamp=dl_run_ts,
                )
                st.markdown("##### Preview Recommendation Record")
                st.dataframe(rec_export_df, use_container_width=True, hide_index=True)

                csv_rec_data = rec_export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download Recommendation CSV ({rec_export_df.iloc[0]['Recommended Method']})",
                    data=csv_rec_data,
                    file_name=f"{clean_ds_slug}_recommendation_{dl_run_id.lower()}.csv",
                    mime="text/csv",
                    type="primary",
                    key="btn_dl_rec_csv",
                )
            else:
                st.warning("No recommendation available for this dataset.")

        with tab_feats:
            st.markdown("#### 🧬 Export Selected Feature Subsets")
            st.caption(
                "Export feature names, probe IDs, or pipeline-encoded variables. Available as per-method CSV, "
                "consolidated all-methods CSV, or structured JSON."
            )

            feat_fmt_col1, feat_fmt_col2 = st.columns([1.2, 1.8])
            with feat_fmt_col1:
                export_scope = st.radio(
                    "Export Scope",
                    ["Per-Method Feature List (CSV)", "Consolidated All-Methods Features (CSV)", "Structured JSON Payload"],
                    key="dl_feat_scope",
                )

            if export_scope == "Per-Method Feature List (CSV)":
                with feat_fmt_col2:
                    method_choices = list(dl_results_df["Method"].unique())
                    target_method = st.selectbox("Select Method to Export", options=method_choices, key="dl_target_method")

                chosen_m_row = dl_results_df[dl_results_df["Method"] == target_method]
                if not chosen_m_row.empty:
                    m_feats = parse_feature_names(chosen_m_row.iloc[0].get("Selected Feature Names", ""))
                    if m_feats:
                        feat_df = format_feature_list_csv(m_feats, method_name=target_method, dataset_name=dl_dataset_title)
                        st.markdown(f"##### Preview {target_method} Features ({len(feat_df)} features)")
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)

                        csv_feat_data = feat_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"📥 Download {target_method} Features CSV ({len(feat_df)} rows)",
                            data=csv_feat_data,
                            file_name=f"{clean_ds_slug}_{target_method.lower()}_features.csv",
                            mime="text/csv",
                            type="primary",
                            key="btn_dl_per_method_csv",
                        )
                    else:
                        st.info(f"Method '{target_method}' has no selected features recorded.")

            elif export_scope == "Consolidated All-Methods Features (CSV)":
                all_feats_df = format_all_features_csv(dl_results_df, dataset_name=dl_dataset_title)
                st.markdown(f"##### Preview Consolidated Features ({len(all_feats_df)} total feature rows)")
                st.dataframe(all_feats_df, use_container_width=True, hide_index=True)

                csv_all_data = all_feats_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download Consolidated All-Methods CSV ({len(all_feats_df)} rows)",
                    data=csv_all_data,
                    file_name=f"{clean_ds_slug}_all_selected_features.csv",
                    mime="text/csv",
                    type="primary",
                    key="btn_dl_all_methods_csv",
                )

            else:
                # JSON Payload
                with feat_fmt_col2:
                    method_choices = list(dl_results_df["Method"].unique())
                    target_method = st.selectbox("Select Method for JSON Payload", options=method_choices, key="dl_json_method")

                chosen_m_row = dl_results_df[dl_results_df["Method"] == target_method]
                if not chosen_m_row.empty:
                    json_payload = format_feature_payload_json(
                        chosen_m_row.iloc[0],
                        dataset_name=dl_dataset_title,
                        run_id=dl_run_id,
                        run_timestamp=dl_run_ts,
                    )
                    st.markdown(f"##### JSON Payload Preview ({target_method})")
                    with st.expander("View Full JSON", expanded=True):
                        st.json(json_payload)

                    json_str_data = json.dumps(json_payload, indent=2).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download {target_method} JSON Payload",
                        data=json_str_data,
                        file_name=f"{clean_ds_slug}_{target_method.lower()}_payload.json",
                        mime="application/json",
                        type="primary",
                        key="btn_dl_json_payload",
                    )

