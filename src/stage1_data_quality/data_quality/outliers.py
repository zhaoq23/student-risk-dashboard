"""
outliers.py — Outlier Handling

Core Principles:
1. Outliers are not necessarily noise; they may represent meaningful signals.
2. Mature students or low admission grades constitute high-risk signals and must be preserved.
3. Extremes in macroeconomic indicators are treated as noise and are subject to Winsorization.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_quality.rules import get_outlier_policy, classify_variable_type


def detect_outliers_iqr(series: pd.Series, iqr_mult: float = 3.0) -> pd.Series:
    """
    Identifies outliers using the Interquartile Range (IQR) method.
    
    Returns: Boolean Series where True indicates an outlier.
    """
    x = pd.to_numeric(series, errors="coerce")
    x_clean = x.dropna()
    
    if len(x_clean) < 20:
        # Sample size is insufficient for detection
        return pd.Series(False, index=series.index)
    
    q1 = x_clean.quantile(0.25)
    q3 = x_clean.quantile(0.75)
    iqr = q3 - q1
    
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=series.index)
    
    lo = q1 - iqr_mult * iqr
    hi = q3 + iqr_mult * iqr
    
    return (x < lo) | (x > hi)


def detect_outliers_zscore(series: pd.Series, z_thresh: float = 4.0) -> pd.Series:
    """
    Identifies outliers using the Z-score method.
    """
    x = pd.to_numeric(series, errors="coerce")
    x_clean = x.dropna()
    
    if len(x_clean) < 20:
        return pd.Series(False, index=series.index)
    
    mu = x_clean.mean()
    sd = x_clean.std()
    
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=series.index)
    
    z = (x - mu) / sd
    return z.abs() > z_thresh


def outlier_report(df: pd.DataFrame, continuous_cols: list, 
                   z_thresh: float = 4.0, iqr_mult: float = 3.0) -> pd.DataFrame:
    """
    Generates an outlier detection report.
    
    Heuristic filtering:
    1. Skip low-cardinality columns (likely categorical codes).
    2. Skip 'grade' columns (bounded variables are unsuitable for IQR).
    3. Use non-missing values as the denominator for calculations.
    """
    rows = []

    for c in continuous_cols:
        if c not in df.columns:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        s_non = s.dropna()
        denom = int(s_non.shape[0])
        
        if denom == 0:
            continue

        # 1. Skip low-cardinality variables as they likely represent categorical codes
        nunique = int(s_non.nunique())
        if nunique <= 25:
            continue

        # 2. Bounded variables such as grades are unsuitable for IQR-based detection
        cl = str(c).lower()
        if "grade" in cl:
            continue

        # 3. Detect outliers
        mask = detect_outliers_iqr(s, iqr_mult=iqr_mult)
        n = int(mask.sum())
        
        if n > 0:
            # Retrieve the specific handling policy for the column
            policy = get_outlier_policy(c)
            
            rows.append({
                "column": c,
                "outlier_count": n,
                "outlier_pct": n / denom * 100,
                "handling": policy.handling,
                "rationale": policy.rationale
            })

    out = pd.DataFrame(
        rows, 
        columns=["column", "outlier_count", "outlier_pct", "handling", "rationale"]
    )
    
    if out.empty:
        return out
    
    return out.sort_values("outlier_pct", ascending=False)


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorization: Caps extreme values at the specified percentiles.
    
    Primarily used for handling noise-prone outliers such as macroeconomic indicators.
    """
    x = pd.to_numeric(series, errors="coerce")
    lo = x.quantile(lower)
    hi = x.quantile(upper)
    return x.clip(lower=lo, upper=hi)


def apply_outlier_handling(df: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """
    Executes outlier handling strategies based on the generated outlier report.
    
    Strategies:
    - flag_only: Creates a binary {col}__is_outlier indicator.
    - winsorize: Creates a binary indicator and applies Winsorization to numerical values.
    """
    if report is None or len(report) == 0:
        return df.copy()
    
    out = df.copy()
    
    for _, row in report.iterrows():
        col = row["column"]
        handling = row["handling"]
        
        if col not in out.columns:
            continue
        
        # Identify outliers
        mask = detect_outliers_iqr(out[col])
        
        # Generate binary indicator
        ind_col = f"{col}__is_outlier"
        out[ind_col] = mask.astype(int)
        
        # Apply strategy-specific handling
        if handling == "winsorize":
            out[col] = winsorize_series(out[col])
    
    return out