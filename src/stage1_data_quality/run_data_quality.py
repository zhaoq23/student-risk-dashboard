"""
Stage 1: Data Quality & Cleaning Pipeline

Features:
1. Read weekly raw data: week1.csv, week2.csv, week3.csv
2. Comprehensive missing value handling (indicator + imputation)
3. Domain-driven outlier detection (signal vs noise)
4. Systematic sanity checks (logical validation)
5. Visualization generation (base64 embedding)
6. Output cleaned data to data/processed/cleaned_data.csv
7. Export diagnostic payload to outputs/stage1_quality/pipeline_data.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path for internal module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / "stage1_data_quality"))

# Import specialized data quality modules
from data_quality.schema import infer_schema, coerce_to_schema, dtype_summary
from data_quality.audit import run_integrity_audit
from data_quality.missingness import (
    propose_missing_strategy, 
    apply_missing_strategy
)
from data_quality.outliers import outlier_report, apply_outlier_handling
from data_quality.rules import group_columns_by_type

# Import profiling and visualization suites
from profiling.profiling import group_profile_with_charts
# Ensure we use the renamed target analysis module
from visualization.target_analysis import analyze_target_slices


# ============================================================
# Pipeline Configuration
# ============================================================

class Config:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Relative input paths
        self.raw_data_dir = project_root / "data" / "raw"
        self.week_files = [
            self.raw_data_dir / "week1.csv",
            self.raw_data_dir / "week2.csv", 
            self.raw_data_dir / "week3.csv"
        ]
        
        # Absolute output paths
        self.processed_dir = project_root / "data" / "processed"
        self.output_dir = project_root / "outputs" / "stage1_quality"
        
        # Ensure directory infrastructure exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output artifacts
        self.cleaned_data_path = self.processed_dir / "cleaned_data.csv"
        self.pipeline_json_path = self.output_dir / "pipeline_data.json"
        
        # Domain validation parameters
        self.min_age = 16
        self.max_age = 80
        self.zscore_threshold = 4.0
        self.iqr_multiplier = 3.0


# ============================================================
# Data Ingestion Functions
# ============================================================

def read_weekly_data(week_files: list) -> tuple:
    """
    Ingests weekly CSV files and reconciles headers.
    
    Args:
        week_files: List of paths to weekly data artifacts.
        
    Returns:
        tuple: (list of weekly DataFrames, combined DataFrame)
    """
    weeks = []
    
    print(f"\n{'='*70}")
    print("Ingesting weekly raw data files...")
    print(f"{'='*70}")
    
    for i, filepath in enumerate(week_files, 1):
        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file missing: Week {i} at {filepath}\n"
                f"   Please verify the data/raw/ directory."
            )
        
        # Standard CSV read with whitespace cleaning
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        weeks.append(df)
        print(f"  ✓ Week {i} Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\n{'='*70}")
    
    return weeks, pd.concat(weeks, ignore_index=True)


# ============================================================
# Diagnostic Helper Functions
# ============================================================

def basic_check(df: pd.DataFrame) -> pd.DataFrame:
    """Executes high-level structural integrity checks."""
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    return pd.DataFrame([{
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "missing_cell_pct": float(df.isna().mean().mean() * 100),
        "duplicate_rows": duplicate_rows,
        "has_missing": "Yes" if missing_cells > 0 else "No",
        "has_duplicates": "Yes" if duplicate_rows > 0 else "No",
    }])


def outcome_dist(df: pd.DataFrame, target_col: str = "Target") -> pd.DataFrame:
    """
    Analyzes the distribution of the Education Outcome variable.
    Internal Key: 'Target' | UI Label: 'Education Outcome'
    """
    cnt = df[target_col].value_counts(dropna=False)
    pct = df[target_col].value_counts(normalize=True, dropna=False)
    return pd.DataFrame({
        "Target": cnt.index.astype(str),  # ← 保持 Target
        "count": cnt.values,
        "pct": pct.values * 100
    })


def compare_outcome_share(
    this_df: pd.DataFrame, 
    ref_df: pd.DataFrame, 
    ref_name: str = "ref", 
    target_col: str = "Target"
) -> pd.DataFrame:
    """Calculates distribution drift between current and reference data batches."""
    this = this_df[target_col].value_counts(normalize=True)
    ref = ref_df[target_col].value_counts(normalize=True)
    
    out = pd.concat([this, ref], axis=1).fillna(0)
    out.columns = ["this_pct", f"{ref_name}_pct"]
    out["delta_pct"] = (out["this_pct"] - out[f"{ref_name}_pct"]) * 100
    
    out = out.reset_index().rename(columns={"index": "Target"})  # ← 保持 Target
    out["Target"] = out["Target"].astype(str)  # ← 保持 Target
    out["this_pct"] = out["this_pct"] * 100
    out[f"{ref_name}_pct"] = out[f"{ref_name}_pct"] * 100
    return out


def df_to_records(df: pd.DataFrame) -> list:
    """Serializes DataFrame to JSON-compatible record format."""
    if df is None or df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def make_panel(
    name: str, 
    df_week: pd.DataFrame, 
    min_age: int,
    max_age: int,
    df_last=None, 
    df_cum=None, 
    cum_name="cum"
):
    """Generates a comprehensive diagnostic panel for a specific data segment."""
    panel = {"name": name}
    
    # Structural Metrics
    panel["basic"] = df_to_records(basic_check(df_week))
    
    # Metadata and Dtype Audit
    dtype_counts, cols_by_type = dtype_summary(df_week)
    panel["dtype_counts"] = df_to_records(dtype_counts)
    panel["dtype_cols"] = {k: list(v) for k, v in cols_by_type.items()}
    
    # Logic and Integrity Audit
    panel["audit"] = df_to_records(
        run_integrity_audit(df_week, min_age=min_age, max_age=max_age)
    )
    
    # Educational Outcome Distribution (Mapped from 'Target')
    panel["education_outcome"] = df_to_records(outcome_dist(df_week))
    
    # Data Drift Telemetry
    drift = {}
    if df_last is not None:
        drift["vs_last"] = df_to_records(
            compare_outcome_share(df_week, df_last, ref_name="last")
        )
    if df_cum is not None:
        drift["vs_cum"] = df_to_records(
            compare_outcome_share(df_week, df_cum, ref_name=cum_name)
        )
    panel["drift"] = drift
    
    return panel


# ============================================================
# Main Pipeline Entry Point
# ============================================================

def main():
    """Executes the end-to-end Data Quality and Cleaning Pipeline."""
    
    # Initialize Pipeline Context
    cfg = Config(project_root=PROJECT_ROOT)
    
    print("\n" + "="*70)
    print("STAGE 1: Data Quality & Cleaning Pipeline")
    print("="*70)
    print("Includes: Schema Alignment + Missingness + Outliers + Logic Audits")
    print("="*70)
    
    # ------------------------------------------------------------
    # 1. Raw Data Ingestion
    # ------------------------------------------------------------
    weeks_raw, all_df_raw = read_weekly_data(cfg.week_files)
    
    print(f"\n Aggregate Statistics:")
    print(f"  ✓ Total records ingested: {all_df_raw.shape[0]}")
    print(f"  ✓ Total features identified: {all_df_raw.shape[1]}")
    
    # ------------------------------------------------------------
    # 2. Structural Schema Reconciliation
    # ------------------------------------------------------------
    print(f"\n[1/8] Executing schema alignment across batches...")
    schema = infer_schema(weeks_raw[0])
    weeks = [coerce_to_schema(w, schema) for w in weeks_raw]
    all_df = pd.concat(weeks, ignore_index=True)
    print(f"  ✓ Schema synchronized for {len(weeks)} weekly batches")
    
    # ------------------------------------------------------------
    # 3. Domain Logic Cleaning (Macro Context)
    # ------------------------------------------------------------
    print(f"\n[2/8] Validating macroeconomic indicator constraints...")
    macro_cols = ["GDP", "Inflation rate", "Unemployment rate"]
    for col in macro_cols:
        if col in all_df.columns:
            # Enforce non-negativity for economic indices
            invalid_count = (pd.to_numeric(all_df[col], errors="coerce") < 0).sum()
            if invalid_count > 0:
                all_df[col] = pd.to_numeric(all_df[col], errors="coerce").clip(lower=0)
                print(f"    - {col}: Fixed {invalid_count} negative entries (clipped to 0)")
    
    # ------------------------------------------------------------
    # 4. Missingness Diagnostics & Imputation
    # ------------------------------------------------------------
    print(f"\n[3/8] Processing missing values using signal-preserving logic...")
    
    missing_plan = propose_missing_strategy(all_df)
    n_missing_cols = len(missing_plan)
    print(f"  ✓ Detected {n_missing_cols} columns requiring imputation")
    
    if n_missing_cols > 0:
        print(f"    - Strategic distribution: {missing_plan['strategy'].value_counts().to_dict()}")
    
    # Apply imputation and generate indicator flags
    all_df_imputed = apply_missing_strategy(all_df, missing_plan)
    n_indicators = len([c for c in all_df_imputed.columns if c.endswith("__is_missing")])
    print(f"  ✓ Generated {n_indicators} binary missingness indicators")
    
    # ------------------------------------------------------------
    # 5. Outlier Detection & Strategy Execution
    # ------------------------------------------------------------
    print(f"\n[4/8] Executing domain-driven outlier management...")
    
    # Categorize features for detection
    type_groups = group_columns_by_type(all_df_imputed, exclude_cols=["Target"])
    continuous_cols = type_groups.get("continuous", []) + type_groups.get("count", [])
    
    outlier_rep = outlier_report(
        all_df_imputed, 
        continuous_cols,
        z_thresh=cfg.zscore_threshold,
        iqr_mult=cfg.iqr_multiplier
    )
    n_outlier_cols = len(outlier_rep)
    print(f"  ✓ Extreme values identified in {n_outlier_cols} features")
    
    if n_outlier_cols > 0:
        print(f"    - Handling policy distribution: {outlier_rep['handling'].value_counts().to_dict()}")
    
    # Apply outlier policies (Winsorization vs. Flagging)
    all_df_cleaned = apply_outlier_handling(all_df_imputed, outlier_rep)
    n_outlier_indicators = len([c for c in all_df_cleaned.columns if c.endswith("__is_outlier")])
    print(f"  ✓ Generated {n_outlier_indicators} binary outlier indicators")
    
    # ------------------------------------------------------------
    # 6. Persistent Asset Storage
    # ------------------------------------------------------------
    print(f"\n[5/8] Exporting cleaned data assets...")
    
    all_df_cleaned.to_csv(cfg.cleaned_data_path, index=False)
    
    print(f"  ✓ Artifact Saved: {cfg.cleaned_data_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Original Feature Count: {all_df.shape[1]}")
    print(f"  ✓ Processed Feature Count: {all_df_cleaned.shape[1]} (including signal flags)")
    
    # ------------------------------------------------------------
    # 7. Exploratory Data Profiling (Charts)
    # ------------------------------------------------------------
    print(f"\n[6/8] Visualizing feature distributions...")
    profiles_all = group_profile_with_charts(all_df_cleaned, exclude_target=True)
    print(f"  ✓ Profiling completed for {len(profiles_all)} feature clusters")
    
    # ------------------------------------------------------------
    # 8. Educational Outcome Stratification
    # ------------------------------------------------------------
    print(f"\n[7/8] Analyzing feature-Education Outcome relationship slices...")
    
    # Define features for bivariate outcome analysis
    slice_cols = [
        "Gender",
        "Scholarship holder",
        "Tuition fees up to date",
        "Debtor",
        "International",
        "Age at enrollment",
        "Admission grade",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
    ]
    
    # CRITICAL: target_col must remain 'Target' for internal lookup
    target_slices = analyze_target_slices(all_df_cleaned, slice_cols, target_col="Target")
    print(f"  ✓ Stratified analysis complete for {len(target_slices)} variables")
    
    # ------------------------------------------------------------
    # 9. Longitudinal Diagnostic Panels
    # ------------------------------------------------------------
    print(f"\n[8/8] Constructing weekly longitudinal panels...")
    
    week1, week2, week3 = weeks[0], weeks[1], weeks[2]
    cum1 = week1.copy()
    cum12 = pd.concat([week1, week2], ignore_index=True)
    
    panels = {
        "Overall": make_panel(
            "Overall", all_df, cfg.min_age, cfg.max_age
        ),
        "Week 1": make_panel(
            "Week 1", week1, cfg.min_age, cfg.max_age
        ),
        "Week 2": make_panel(
            "Week 2", week2, cfg.min_age, cfg.max_age,
            df_last=week1, df_cum=cum1, cum_name="week1"
        ),
        "Week 3": make_panel(
            "Week 3", week3, cfg.min_age, cfg.max_age,
            df_last=week2, df_cum=cum12, cum_name="cum(w1+w2)"
        ),
    }
    
    # ------------------------------------------------------------
    # 10. API/JSON Payload Construction
    # ------------------------------------------------------------
    print(f"  ✓ Serializing distribution profiles for report generator...")
    profiles_json = {}
    for group_name, block in profiles_all.items():
        profiles_json[group_name] = {}
        for key, value in block.items():
            if key.endswith("_table"):
                profiles_json[group_name][key] = df_to_records(value)
            elif key.endswith("_chart"):
                profiles_json[group_name][key] = value  # base64 encoded string
    
    print(f"  ✓ Serializing Education Outcome analysis datasets...")
    target_slices_json = {}
    for var_name, item in target_slices.items():
        target_slices_json[var_name] = {
            "type": item["type"],
            "chart": item.get("chart"),
            "table": df_to_records(item.get("table", pd.DataFrame()))
        }
    
    # ------------------------------------------------------------
    # 11. Final Diagnostic Payload Persistence
    # ------------------------------------------------------------
    print(f"  ✓ Assembling holistic diagnostic payload...")
    
    # Final data integrity audit on combined set
    audit_all = run_integrity_audit(all_df, min_age=cfg.min_age, max_age=cfg.max_age)
    
    payload = {
        "meta": {
            "total_rows": int(all_df.shape[0]),
            "total_cols": int(all_df.shape[1]),
            "n_weeks": len(weeks),
            "cleaned_data_path": str(cfg.cleaned_data_path.relative_to(PROJECT_ROOT)),
            "n_missing_indicators": n_indicators,
            "n_outlier_indicators": n_outlier_indicators,
        },
        "panels": panels,
        "quality_reports": {
            "missing_plan": df_to_records(missing_plan),
            "outlier_report": df_to_records(outlier_rep),
            "integrity_audit": df_to_records(audit_all),
        },
        "profiles": profiles_json,
        "target_slices": target_slices_json,
        "slice_cols": slice_cols,
    }
    
    with open(cfg.pipeline_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Analysis payload saved: {cfg.pipeline_json_path.relative_to(PROJECT_ROOT)}")
    
    # ------------------------------------------------------------
    # 12. Pipeline Summary Output
    # ------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STAGE 1 COMPLETE: DATA QUALITY PIPELINE TERMINATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\n Execution Summary:")
    print(f"\n  1. Cleaned Dataset:")
    print(f"     Path: {cfg.cleaned_data_path.relative_to(PROJECT_ROOT)}")
    print(f"     Dimensions: {all_df_cleaned.shape[0]} rows × {all_df_cleaned.shape[1]} features")
    print(f"     Signals: {n_indicators} missing indicators, {n_outlier_indicators} outlier flags")
    print(f"\n  2. Diagnostic Metadata:")
    print(f"     Path: {cfg.pipeline_json_path.relative_to(PROJECT_ROOT)}")
    print(f"     Modules Executed: 12 (Ingestion to Serialization)")
    print(f"     Diagnostic Blocks: {len(panels)} Panels, {len(profiles_all)} Profiles, {len(target_slices)} Slices")
    print(f"     Quality Exceptions: {len(audit_all)} Critical Audit Hits")
    print(f"\n Next Step Recommendation:")
    print(f"   python stage2_feature_engineering/run_feature_strategy.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n Configuration Error: {e}")
        print("\n Suggested Remediation:")
        print("  1. Verify working directory is project root")
        print("  2. Validate contents of data/raw/ are present")
        sys.exit(1)
    except Exception as e:
        print(f"\n Pipeline Terminal Failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)