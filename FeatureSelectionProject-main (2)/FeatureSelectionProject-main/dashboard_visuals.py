"""Visual presentation components matching enterprise ML dashboards (Bank Marketing, ML Metrics, OmniXAI).

Provides semi-circular gauge meters, 6-metric comparison grids, correlation heatmaps,
violin distribution plots, and executive KPI metadata banners.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Standard cohesive algorithm palette
ALGO_PALETTE = {
    "CFS": "#06b6d4",     # Cyan
    "FCBF": "#10b981",    # Emerald Green
    "FAST": "#6366f1",    # Indigo
    "GA": "#8b5cf6",      # Purple
    "PSO": "#f59e0b",     # Amber
    "Hybrid": "#3b82f6",  # Royal Blue
}


def render_speedometer_gauge(
    value: float,
    title: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
    is_percentage: bool = False,
    suffix: str = "",
    bar_color: str = "#2563eb",
    height: int = 220,
) -> go.Figure:
    """Create a semi-circular speedometer gauge meter matching Image 2.
    
    Includes colored threshold zones (red for low, yellow for medium, green for high).
    """
    display_val = value * 100 if is_percentage and max_val <= 1.0 else value
    display_max = max_val * 100 if is_percentage and max_val <= 1.0 else max_val
    display_min = min_val * 100 if is_percentage and max_val <= 1.0 else min_val
    num_suffix = "%" if is_percentage else suffix

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{title}</b>", "font": {"size": 13, "color": "#334155"}},
            number={
                "suffix": num_suffix,
                "font": {"size": 26, "color": "#0f172a", "family": "Inter, sans-serif"},
                "valueformat": ".2f" if not is_percentage and display_val < 10 else (".1f" if is_percentage else ".2f"),
            },
            gauge={
                "axis": {
                    "range": [display_min, display_max],
                    "tickwidth": 1,
                    "tickcolor": "#94a3b8",
                    "tickfont": {"size": 10, "color": "#64748b"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [display_min, display_min + 0.65 * (display_max - display_min)], "color": "#fee2e2"},  # Soft Red
                    {"range": [display_min + 0.65 * (display_max - display_min), display_min + 0.85 * (display_max - display_min)], "color": "#fef9c3"},  # Soft Yellow
                    {"range": [display_min + 0.85 * (display_max - display_min), display_max], "color": "#dcfce7"},  # Soft Green
                ],
            },
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 20, "r": 20, "t": 35, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def render_mini_metric_bar(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_format: str = ".2f",
    is_percentage: bool = False,
    path_subtitle: str = "Path: final_comparison.csv",
    height: int = 240,
) -> go.Figure:
    """Create a clean flat comparative bar chart card matching Image 3."""
    clean_df = df.dropna(subset=[metric_col]).copy()
    if clean_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Data Available", showarrow=False)
        fig.update_layout(height=height)
        return fig

    # Map colors from ALGO_PALETTE
    colors = [ALGO_PALETTE.get(m, "#3b82f6") for m in clean_df["Method"]]
    y_vals = clean_df[metric_col] * (100 if is_percentage and clean_df[metric_col].max() <= 1.0 else 1)

    fig = go.Figure(
        data=[
            go.Bar(
                x=clean_df["Method"],
                y=y_vals,
                marker_color=colors,
                text=[f"{v:{y_format}}{'%' if is_percentage else ''}" for v in y_vals],
                textposition="auto",
                textfont={"size": 10, "color": "#0f172a"},
            )
        ]
    )
    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><span style='font-size:10px; color:#64748b; font-weight:normal;'>{path_subtitle}</span>",
            "font": {"size": 13, "color": "#1e293b"},
            "x": 0.02,
            "y": 0.94,
        },
        height=height,
        margin={"l": 30, "r": 20, "t": 45, "b": 30},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        yaxis={"gridcolor": "#f1f5f9", "zeroline": True, "zerolinecolor": "#cbd5e1"},
        xaxis={"tickfont": {"size": 11, "color": "#334155"}},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def resolve_feature_column(df: pd.DataFrame, col_identifier: Any) -> Optional[Any]:
    """Resolve column name whether it is an int, str, or quoted identifier in the dataframe."""
    if df is None or df.empty or col_identifier is None:
        return None
    if col_identifier in df.columns:
        return col_identifier
    # Try as integer
    try:
        int_c = int(str(col_identifier).strip("'\""))
        if int_c in df.columns:
            return int_c
    except (ValueError, TypeError):
        pass
    # Try as string
    str_c = str(col_identifier).strip("'\"")
    if str_c in df.columns:
        return str_c
    return None


def render_correlation_heatmap(
    X: pd.DataFrame,
    selected_features: List[str],
    max_features: int = 15,
    title: str = "Correlation Heatmap: Selected Features",
    height: int = 400,
) -> Optional[go.Figure]:
    """Create a 2D correlation matrix heatmap matching Image 4."""
    if X is None or X.empty or not selected_features:
        return None

    # Filter to available features (resolving int/str naming differences)
    valid_cols = []
    for c in selected_features:
        resolved = resolve_feature_column(X, c)
        if resolved is not None and resolved not in valid_cols:
            valid_cols.append(resolved)
        if len(valid_cols) >= max_features:
            break

    if len(valid_cols) < 2:
        return None

    subset_df = X[valid_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = subset_df.corr().round(2)

    # Simplified display labels if probe or encoded names are long
    display_labels = [str(c) if len(str(c)) <= 12 else f"{str(c)[:10]}.." for c in valid_cols]

    fig = px.imshow(
        corr_matrix.values,
        x=display_labels,
        y=display_labels,
        color_continuous_scale="Blues",
        zmin=-1.0,
        zmax=1.0,
        aspect="auto",
        labels={"color": "Correlation"},
    )
    fig.update_traces(
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 9},
    )
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 14, "color": "#1e293b"}},
        height=height,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def render_violin_distribution(
    X: pd.DataFrame,
    y: pd.Series,
    feature_name: str,
    title: Optional[str] = None,
    height: int = 340,
) -> Optional[go.Figure]:
    """Create a violin distribution plot broken down by target class matching Image 4."""
    if X is None or X.empty or y is None:
        return None

    resolved_feat = resolve_feature_column(X, feature_name)
    if resolved_feat is None:
        return None

    plot_df = pd.DataFrame({
        "Feature Value": pd.to_numeric(X[resolved_feat], errors="coerce"),
        "Target Class": y.astype(str),
    }).dropna()

    if plot_df.empty:
        return None

    plot_title = title or f"<b>Distribution of {feature_name} by Target</b>"
    fig = px.violin(
        plot_df,
        x="Target Class",
        y="Feature Value",
        color="Target Class",
        box=True,
        points="all",
        color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"],
    )
    fig.update_layout(
        title={"text": plot_title, "font": {"size": 13, "color": "#1e293b"}},
        height=height,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        yaxis={"gridcolor": "#f1f5f9"},
        xaxis={"title": "Target Class", "tickfont": {"size": 11, "color": "#334155"}},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def render_feature_consensus_diverging_bar(
    feature_freq_series: pd.Series,
    top_n: int = 12,
    title: str = "Feature Selection Consensus Across 6 Algorithms",
    height: int = 360,
) -> Optional[go.Figure]:
    """Create a horizontal diverging / consensus bar chart matching Image 5 (OmniXAI)."""
    if feature_freq_series is None or feature_freq_series.empty:
        return None

    top_series = feature_freq_series.head(top_n).sort_values(ascending=True)
    df = top_series.reset_index()
    df.columns = ["Feature", "Selection Count"]
    df["Consensus Level"] = df["Selection Count"].apply(
        lambda c: "High Consensus (>= 3 Methods)" if c >= 3 else "Moderate Consensus (1-2 Methods)"
    )

    color_map = {
        "High Consensus (>= 3 Methods)": "#059669",      # Emerald Green
        "Moderate Consensus (1-2 Methods)": "#f43f5e",   # Rose Red
    }

    fig = px.bar(
        df,
        x="Selection Count",
        y="Feature",
        orientation="h",
        color="Consensus Level",
        color_discrete_map=color_map,
        text="Selection Count",
    )
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 13, "color": "#1e293b"}},
        height=height,
        margin={"l": 80, "r": 20, "t": 40, "b": 30},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"gridcolor": "#f1f5f9", "dtick": 1, "range": [0, 6.5], "title": "Number of Algorithms Selecting Feature"},
        yaxis={"title": ""},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
        font={"family": "Inter, sans-serif"},
    )
    fig.update_traces(textposition="outside")
    return fig


def render_executive_banner_html(
    dataset_name: str,
    num_samples: int,
    num_features: int,
    best_acc: Optional[float],
    best_mcc: Optional[float],
    reduction_pct: Optional[float],
    num_selected: int,
    source_type: str = "CSV",
) -> str:
    """Generate HTML for the Bank Marketing style top executive banner matching Image 1."""
    acc_str = f"{best_acc:.2%}" if best_acc is not None else "N/A"
    mcc_str = f"{best_mcc:.2f}" if best_mcc is not None else "N/A"
    red_str = f"{reduction_pct:.1f}%" if reduction_pct is not None else "N/A"

    return f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.6rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.8rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{dataset_name} - Score</span>
                <span style="background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Approved</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">
                <span style="background: #f1f5f9; padding: 0.3rem 0.7rem; border-radius: 6px; margin-right: 0.5rem;">⚙️ 10-Fold Stratified CV</span>
                <span style="background: #dcfce7; color: #166534; padding: 0.3rem 0.7rem; border-radius: 6px; font-weight: 600;">★ Pareto-Optimal</span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.2rem; font-size: 0.82rem;">
            <div>
                <div style="color: #64748b; text-transform: uppercase; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px;">ENGAGEMENT</div>
                <div style="font-weight: 600; color: #1e293b; margin-top: 0.15rem;">Biomarker Selection Active</div>
            </div>
            <div>
                <div style="color: #64748b; text-transform: uppercase; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px;">DATE CRITERIA</div>
                <div style="font-weight: 600; color: #1e293b; margin-top: 0.15rem;">Holdout 80/20 Test Split</div>
            </div>
            <div>
                <div style="color: #64748b; text-transform: uppercase; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px;">SOURCE</div>
                <div style="font-weight: 600; color: #1e293b; margin-top: 0.15rem;">{source_type}</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #f1f5f9; padding-top: 1rem;">
            <div style="display: flex; gap: 2.2rem; align-items: baseline;">
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">{num_samples:,}</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">TRAINED ON</div>
                </div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">{num_features:,}</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">ORIGINAL POOL</div>
                </div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #16a34a;">{acc_str}</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">ACCURACY</div>
                </div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #16a34a;">{mcc_str}</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">MCC SCORE</div>
                </div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #2563eb;">{red_str}</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">REDUCTION</div>
                </div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">{num_selected}</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">FIELDS RETAINED</div>
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <span style="border: 1px solid #cbd5e1; color: #334155; padding: 0.4rem 0.9rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600; background: #ffffff;">📝 Inspect Metadata</span>
                <span style="background: #4f46e5; color: #ffffff; padding: 0.4rem 1rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">⚡ Active Run</span>
            </div>
        </div>
    </div>
    """


def render_throughput_area_chart(height: int = 240) -> go.Figure:
    """Create a service requests/throughput area chart matching Image 2."""
    time_points = ["09:39:30", "09:40:00", "09:40:30", "09:41:00", "09:41:30", "09:42:00", "09:42:30"]
    s2xx = [140, 150, 130, 160, 110, 260, 210]
    s4xx = [220, 200, 180, 210, 220, 250, 230]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=s2xx,
            mode="lines+markers",
            name='status="2xx" (Success)',
            fill="tozeroy",
            line={"color": "#6366f1", "width": 2},
            fillcolor="rgba(99, 102, 241, 0.4)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=[a + b for a, b in zip(s2xx, s4xx)],
            mode="lines+markers",
            name='status="2xx + 4xx" (Total)',
            fill="tonexty",
            line={"color": "#22c55e", "width": 2},
            fillcolor="rgba(34, 197, 94, 0.3)",
        )
    )
    fig.update_layout(
        title={"text": "<b>Requests / Execution Throughput</b>", "font": {"size": 13, "color": "#1e293b"}},
        height=height,
        margin={"l": 30, "r": 20, "t": 35, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis={"gridcolor": "#f1f5f9"},
        xaxis={"tickfont": {"size": 10, "color": "#64748b"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 10}},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def render_omnixai_predict_bar(
    class_probs: Optional[Dict[str, float]] = None,
    height: int = 240,
) -> go.Figure:
    """Create local prediction explanation bar chart matching Image 5 (PREDICT)."""
    probs = class_probs or {"Class 0 (Benign)": 0.18, "Class 1 (Malignant)": 0.82}
    labels = list(probs.keys())
    values = list(probs.values())

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#6366f1",
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title={"text": "<b>Local Explanation: PREDICT</b>", "font": {"size": 13, "color": "#1e293b"}},
        height=height,
        margin={"l": 120, "r": 20, "t": 35, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"range": [0, 1.0], "tickformat": ".0%", "gridcolor": "#f1f5f9", "title": "Predicted Probability"},
        yaxis={"title": ""},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def render_omnixai_lime_bar(
    feature_weights: Optional[Dict[str, float]] = None,
    height: int = 260,
) -> go.Figure:
    """Create LIME local feature importance attribution bar chart matching Image 5 (LIME)."""
    weights = feature_weights or {
        "Biomarker_01": 0.38,
        "Biomarker_02": 0.14,
        "Biomarker_03": 0.12,
        "Biomarker_04": -0.08,
        "Biomarker_05": -0.06,
        "Biomarker_06": -0.03,
        "Biomarker_07": -0.02,
        "Biomarker_08": -0.01,
    }
    feats = list(weights.keys())[::-1]
    vals = [weights[f] for f in feats]
    colors = ["#059669" if v >= 0 else "#e11d48" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title={"text": "<b>Local Explanation: LIME Attribution</b>", "font": {"size": 13, "color": "#1e293b"}},
        height=height,
        margin={"l": 100, "r": 30, "t": 35, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"gridcolor": "#f1f5f9", "title": "Importance Score (Contribution)"},
        yaxis={"title": ""},
        font={"family": "Inter, sans-serif"},
    )
    return fig

