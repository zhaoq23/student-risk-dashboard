"""
missingness.py — Missing Data Handling

Core Logic:
1. Missing values are not necessarily "bad data"; they can serve as strong signals 
   (e.g., a missing grade might indicate an absent exam).
2. Create indicator variables to preserve missingness information for the model.
3. Select imputation strategies based on predefined business rules.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_quality.rules import get_missing_rule, classify_rule_bucket


def propose_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a handling strategy for each column containing missing values.
    
    Returns: DataFrame with columns:
        - column: Column name
        - rule_bucket: Variable category (e.g., critical_academic, finance, parent)
        - missing_count: Number of missing values
        - missing_pct: Percentage of missing values
        - strategy: Imputation strategy
        - rationale: Business justification
    """
    rows = []
    n = len(df)

    for c in df.columns:
        miss = int(df[c].isna().sum())
        if miss == 0:
            continue

        rule = get_missing_rule(c)
        bucket = classify_rule_bucket(c)

        rows.append({
            "column": c,
            "rule_bucket": bucket,
            "missing_count": miss,
            "missing_pct": miss / n * 100,
            "strategy": rule.strategy,
            "rationale": rule.rationale
        })

    out = pd.DataFrame(
        rows,
        columns=["column", "rule_bucket", "missing_count", "missing_pct", "strategy", "rationale"]
    )
    
    if len(out) == 0:
        return out
    
    return out.sort_values(["missing_pct", "missing_count"], ascending=False)


def apply_missing_strategy(df: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """
    Applies missing data handling strategies based on the provided plan.
    
    Key Features:
    1. Creates a binary {col}__is_missing indicator for every column with missing values.
    2. Performs data imputation based on the designated strategy.
    3. Preserves original signals for use in downstream modeling.
    """
    if plan is None or len(plan) == 0:
        return df.copy()

    out = df.copy()

    for _, r in plan.iterrows():
        c = r["column"]
        strategy = r["strategy"]

        # 1. Create a missingness indicator (Crucial for capturing signal)
        ind_col = f"{c}__is_missing"
        out[ind_col] = out[c].isna().astype(int)

        # 2. Execute imputation based on strategy
        if strategy == "indicator+median":
            # Continuous variables: Median imputation
            med = pd.to_numeric(out[c], errors="coerce").median()
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(med)

        elif strategy == "indicator+mode":
            # Binary variables: Mode imputation
            s = pd.to_numeric(out[c], errors="coerce")
            if len(s.dropna()) > 0:
                mode = s.mode().iloc[0]
            else:
                mode = 0
            out[c] = s.fillna(mode)

        elif strategy == "indicator+assume_no":
            # Financial variables: Assume 'No' (0)
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64")

        elif strategy == "indicator+unknown":
            # Categorical variables: Fill with 'Unknown' label
            out[c] = out[c].astype("object").fillna("Unknown")

        elif strategy == "indicator+zero":
            # Count variables: Fill with 0 (e.g., total credit units)
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

        else:
            # Others: Retain the binary indicator only, without value imputation
            pass

    return out


def missing_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a missing data summary table for dashboard visualization.
    """
    rows = []
    n = len(df)
    
    for c in df.columns:
        miss = int(df[c].isna().sum())
        if miss > 0:
            rows.append({
                "column": c,
                "missing_count": miss,
                "missing_pct": miss / n * 100,
                "non_missing_count": n - miss
            })
    
    if not rows:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct", "non_missing_count"])
    
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)