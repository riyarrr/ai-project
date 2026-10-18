"""Export formatting, result filtering, data sanitization, and helper utilities

for the Feature Selection & Explainability Dashboard (Member 3).
"""

import datetime
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def sanitize_error_message(error: Any) -> str:
    """Sanitize error messages to eliminate local system paths, user folders, and credentials.
    
    Ensures requirement 10: Keep credentials, local paths, and private data out of the interface and logs.
    """
    if error is None:
        return ""
    text = str(error)
    # Remove Windows full file paths (e.g., C:\Users\...\)
    text = re.sub(r"[a-zA-Z]:\\[^ \t\n\r\"\'`<>]+", "<system_path>", text)
    # Remove Unix full file paths (/home/..., /tmp/..., etc.)
    text = re.sub(r"/(?:home|Users|root|tmp|var|etc)/[^ \t\n\r\"\'`<>]+", "<system_path>", text)
    # Remove user account names if lingering in patterns like Users\username or Users/username
    text = re.sub(r"(Users[/\\])[^/\\]+", r"\1<user>", text)
    # Remove potential tokens / credentials
    text = re.sub(r"(token|bearer|key|password|secret)[=:\s]+[A-Za-z0-9_\-\.]{8,}", r"\1=<redacted>", text, flags=re.IGNORECASE)
    return text.strip()


def generate_run_identifier(prefix: str = "EVAL", timestamp_dt: Optional[datetime.datetime] = None) -> Tuple[str, str]:
    """Generate a unique run identifier and UTC timestamp for distinguishing evaluation runs.
    
    Returns:
        (run_id, formatted_timestamp_string)
    """
    if timestamp_dt is None:
        timestamp_dt = datetime.datetime.now(datetime.timezone.utc)
    date_str = timestamp_dt.strftime("%Y%m%d")
    time_str = timestamp_dt.strftime("%H%M%S")
    run_id = f"{prefix}-{date_str}-{time_str}"
    formatted_ts = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return run_id, formatted_ts


def parse_feature_names(val: Any) -> List[str]:
    """Parse selected feature names safely from comma-separated or JSON string formats."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return [str(x).strip() for x in val if pd.notna(x) and str(x).strip()]
    try:
        if pd.isna(val):
            return []
    except (ValueError, TypeError):
        pass
    val_str = str(val).strip()
    if not val_str:
        return []
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in val_str.split(",") if x.strip()]



def decode_feature_info(feature_name: Any, dataset_name: str = "") -> Dict[str, Any]:
    """Decodes internal feature identifiers into human-readable representations, distinguishing

    original raw feature names from pipeline-encoded variables (e.g., numeric imputation/scaling, one-hot encoding).
    """
    feat_str = str(feature_name).strip()

    # 1. Pipeline encoded features from upload_recommendation
    if feat_str.startswith("numeric__"):
        orig_col = feat_str[len("numeric__"):]
        return {
            "Feature Name": feat_str,
            "Original Column": orig_col,
            "Encoding / Origin": "Numeric (Imputed & Standardized)",
            "Is Encoded": True,
            "Encoding Status": "Pipeline Encoded",
            "Details": f"Original numeric column '{orig_col}' (median imputed & standard scaled)",
        }
    elif feat_str.startswith("categorical__"):
        rest = feat_str[len("categorical__"):]
        parts = rest.split("_", 1)
        orig_col = parts[0] if len(parts) > 1 else rest
        val = parts[1] if len(parts) > 1 else "Encoded"
        return {
            "Feature Name": feat_str,
            "Original Column": orig_col,
            "Encoding / Origin": f"One-Hot Encoded (Category: {val})",
            "Is Encoded": True,
            "Encoding Status": "Pipeline Encoded",
            "Details": f"Binary one-hot category '{val}' derived from categorical column '{orig_col}'",
        }

    # 2. Benchmark datasets
    if dataset_name == "Colon Cancer":
        return {
            "Feature Name": feat_str,
            "Original Column": f"Probe #{feat_str}",
            "Encoding / Origin": "Microarray Probe ID (Original Raw)",
            "Is Encoded": False,
            "Encoding Status": "Original Raw",
            "Details": f"Alon et al. Colon Cancer oligonucleotide probe #{feat_str}",
        }
    elif dataset_name == "Leukemia":
        return {
            "Feature Name": feat_str,
            "Original Column": feat_str,
            "Encoding / Origin": "Affymetrix Gene ID (Original Raw)",
            "Is Encoded": False,
            "Encoding Status": "Original Raw",
            "Details": f"Golub et al. Leukemia Affymetrix high-density gene probe '{feat_str}'",
        }

    # 3. Default fallback
    return {
        "Feature Name": feat_str,
        "Original Column": feat_str,
        "Encoding / Origin": "Original Raw Feature",
        "Is Encoded": False,
        "Encoding Status": "Original Raw",
        "Details": "Original raw dataset column without internal pipeline transformation",
    }


def filter_comparison_results(
    df: pd.DataFrame,
    selected_methods: Optional[List[str]] = None,
    min_accuracy: Optional[float] = None,
    min_mcc: Optional[float] = None,
    status_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Filter comparison records by method, performance metrics, and execution status.
    
    Ensures requirement 2: Downloads use the currently selected dataset, methods, and filters where appropriate.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    filtered = df.copy()

    # Filter methods
    if selected_methods:
        filtered = filtered[filtered["Method"].isin(selected_methods)]

    # Filter by min accuracy
    if min_accuracy is not None:
        filtered = filtered[filtered["Accuracy"].fillna(-1.0) >= min_accuracy]

    # Filter by min MCC
    if min_mcc is not None:
        filtered = filtered[filtered["MCC"].fillna(-2.0) >= min_mcc]

    # Filter by status
    if status_filter and "Execution Status" in filtered.columns and status_filter != "All":
        filtered = filtered[filtered["Execution Status"] == status_filter]

    return filtered


def format_comparison_export(df: pd.DataFrame) -> pd.DataFrame:
    """Format comparison dataframe with consistent headers, data types, and rounding for export.
    
    Ensures requirement 11: Check that downloads contain the right headers, data types, and rows.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Dataset", "Method", "Accuracy", "MCC", "Selected Features",
            "Total Features", "Feature Reduction (%)", "Runtime (s)", "Execution Status"
        ])
    
    cols = [
        "Dataset", "Method", "Accuracy", "MCC", "Selected Features",
        "Total Features", "Feature Reduction (%)", "Runtime (s)"
    ]
    if "Execution Status" in df.columns:
        cols.append("Execution Status")
    
    export_df = pd.DataFrame()
    for col in cols:
        if col in df.columns:
            export_df[col] = df[col]
        else:
            export_df[col] = np.nan
            
    # Round numerical metrics cleanly
    if "Accuracy" in export_df.columns:
        export_df["Accuracy"] = pd.to_numeric(export_df["Accuracy"], errors="coerce").round(4)
    if "MCC" in export_df.columns:
        export_df["MCC"] = pd.to_numeric(export_df["MCC"], errors="coerce").round(4)
    if "Feature Reduction (%)" in export_df.columns:
        export_df["Feature Reduction (%)"] = pd.to_numeric(export_df["Feature Reduction (%)"], errors="coerce").round(4)
    if "Runtime (s)" in export_df.columns:
        export_df["Runtime (s)"] = pd.to_numeric(export_df["Runtime (s)"], errors="coerce").round(4)
    if "Selected Features" in export_df.columns:
        export_df["Selected Features"] = pd.to_numeric(export_df["Selected Features"], errors="coerce")
    if "Total Features" in export_df.columns:
        export_df["Total Features"] = pd.to_numeric(export_df["Total Features"], errors="coerce")
        
    return export_df


def format_recommendation_export(
    rec_obj_or_df: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    run_id: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """Format recommendation summary table with explicit methodology headers and disclaimers.
    
    Ensures requirement 1, 7, 10, 11.
    """
    if rec_obj_or_df is None:
        return pd.DataFrame()
        
    if isinstance(rec_obj_or_df, pd.Series):
        raw_dict = rec_obj_or_df.to_dict()
    elif isinstance(rec_obj_or_df, pd.DataFrame):
        if rec_obj_or_df.empty:
            return pd.DataFrame()
        raw_dict = rec_obj_or_df.iloc[0].to_dict()
    elif isinstance(rec_obj_or_df, dict):
        raw_dict = rec_obj_or_df.copy()
    else:
        return pd.DataFrame()

    row = {
        "Dataset": raw_dict.get("Dataset", "Active Dataset"),
        "Recommended Method": raw_dict.get("Recommended Method", ""),
        "Best Predictive Method(s)": raw_dict.get("Best Predictive Method(s)", raw_dict.get("Recommended Method", "")),
        "Accuracy": round(float(raw_dict.get("Accuracy", 0.0)), 4) if pd.notna(raw_dict.get("Accuracy")) else np.nan,
        "MCC": round(float(raw_dict.get("MCC", 0.0)), 4) if pd.notna(raw_dict.get("MCC")) else np.nan,
        "Selected Features": int(raw_dict.get("Selected Features", 0)) if pd.notna(raw_dict.get("Selected Features")) else np.nan,
        "Total Features": int(raw_dict.get("Total Features", 0)) if pd.notna(raw_dict.get("Total Features")) else np.nan,
        "Feature Reduction (%)": round(float(raw_dict.get("Feature Reduction (%)", 0.0)), 4) if pd.notna(raw_dict.get("Feature Reduction (%)")) else np.nan,
        "Runtime (s)": round(float(raw_dict.get("Runtime (s)", 0.0)), 4) if pd.notna(raw_dict.get("Runtime (s)")) else np.nan,
        "Dominating Count": int(raw_dict.get("Dominating Count", 0)) if pd.notna(raw_dict.get("Dominating Count")) else 0,
        "Reason": raw_dict.get("Reason", "Pareto optimal subset selection"),
        "Run Identifier": run_id or raw_dict.get("Run Identifier", "BENCHMARK-RUN"),
        "Evaluation Timestamp": run_timestamp or raw_dict.get("Evaluation Timestamp", "Standardized Partition"),
        "Scientific Disclaimer": "Non-causal statistical correlation; holdout 80/20 test split does not prove generalization to independent clinical cohorts.",
    }
    return pd.DataFrame([row])


def format_feature_list_csv(
    feature_names: List[str],
    method_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Format single-method selected features as a clean CSV with feature indexing and provenance decoding.
    
    Ensures requirement 1, 2, 11.
    """
    rows = []
    for idx, feat in enumerate(feature_names, start=1):
        dec = decode_feature_info(feat, dataset_name=dataset_name)
        rows.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "Feature Index": idx,
            "Feature Name": feat,
            "Original Column": dec["Original Column"],
            "Encoding / Origin": dec["Encoding / Origin"],
            "Encoding Status": dec["Encoding Status"],
            "Details": dec["Details"],
        })
    if not rows:
        return pd.DataFrame(columns=[
            "Dataset", "Method", "Feature Index", "Feature Name",
            "Original Column", "Encoding / Origin", "Encoding Status", "Details"
        ])
    return pd.DataFrame(rows)


def format_all_features_csv(
    results_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Format a consolidated tidy long-format table containing all selected features across all methods.
    
    Ensures requirement 1, 2, 11.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame(columns=[
            "Dataset", "Method", "Feature Index", "Feature Name",
            "Original Column", "Encoding / Origin", "Accuracy", "MCC", "Runtime (s)"
        ])
        
    all_rows = []
    for _, r in results_df.iterrows():
        m_name = r["Method"]
        feats = parse_feature_names(r.get("Selected Feature Names", ""))
        acc = round(float(r["Accuracy"]), 4) if pd.notna(r.get("Accuracy")) else np.nan
        mcc = round(float(r["MCC"]), 4) if pd.notna(r.get("MCC")) else np.nan
        rt = round(float(r["Runtime (s)"]), 4) if pd.notna(r.get("Runtime (s)")) else np.nan
        
        for idx, f in enumerate(feats, start=1):
            dec = decode_feature_info(f, dataset_name=dataset_name)
            all_rows.append({
                "Dataset": dataset_name,
                "Method": m_name,
                "Feature Index": idx,
                "Feature Name": f,
                "Original Column": dec["Original Column"],
                "Encoding / Origin": dec["Encoding / Origin"],
                "Method Accuracy": acc,
                "Method MCC": mcc,
                "Method Runtime (s)": rt,
            })
            
    if not all_rows:
        return pd.DataFrame(columns=[
            "Dataset", "Method", "Feature Index", "Feature Name",
            "Original Column", "Encoding / Origin", "Method Accuracy", "Method MCC", "Method Runtime (s)"
        ])
    return pd.DataFrame(all_rows)


def format_feature_payload_json(
    chosen_row: Union[pd.Series, Dict[str, Any]],
    dataset_name: str,
    run_id: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Format complete structured JSON export payload containing metadata, metrics, and feature lists.
    
    Ensures requirement 1, 7, 10, 11.
    """
    if isinstance(chosen_row, pd.Series):
        r_dict = chosen_row.to_dict()
    elif isinstance(chosen_row, dict):
        r_dict = chosen_row.copy()
    else:
        r_dict = {}

    m_name = r_dict.get("Method", "Unknown")
    feats = parse_feature_names(r_dict.get("Selected Feature Names", ""))
    decoded_feats = [decode_feature_info(f, dataset_name=dataset_name) for f in feats]
    
    return {
        "dataset": dataset_name,
        "method": m_name,
        "run_id": run_id or r_dict.get("Run Identifier", "BENCHMARK-RUN"),
        "evaluation_timestamp": run_timestamp or r_dict.get("Evaluation Timestamp", "Standardized Partition"),
        "feature_count": len(feats),
        "metrics": {
            "accuracy": float(r_dict.get("Accuracy", 0.0)) if pd.notna(r_dict.get("Accuracy")) else None,
            "mcc": float(r_dict.get("MCC", 0.0)) if pd.notna(r_dict.get("MCC")) else None,
            "feature_reduction_pct": float(r_dict.get("Feature Reduction (%)", 0.0)) if pd.notna(r_dict.get("Feature Reduction (%)")) else None,
            "runtime_sec": float(r_dict.get("Runtime (s)", 0.0)) if pd.notna(r_dict.get("Runtime (s)")) else None,
            "total_features": int(r_dict.get("Total Features", 0)) if pd.notna(r_dict.get("Total Features")) else None,
        },
        "features": feats,
        "decoded_features": decoded_feats,
        "scientific_disclaimer": "Non-causal statistical correlation; single holdout test split does not prove clinical generalization.",
    }
