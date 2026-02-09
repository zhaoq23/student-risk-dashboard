"""
longitudinal.py — Longitudinal Feature Engineering

Generates temporal features based on 12 variables from Sem 1 and Sem 2.
"""

import pandas as pd
import numpy as np


def generate_longitudinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate longitudinal features
    """
    out = df.copy()
    
    # ============================================================
    # (A) Velocity and Direction: "Slope" of the learning trajectory
    # ============================================================
    
    grade_sem1 = pd.to_numeric(out.get("Curricular units 1st sem (grade)", 0), errors="coerce")
    grade_sem2 = pd.to_numeric(out.get("Curricular units 2nd sem (grade)", 0), errors="coerce")
    
    out["Delta_Grade"] = grade_sem2 - grade_sem1
    out["Grade_Improvement"] = (out["Delta_Grade"] > 0).astype(int)
    out["Grade_Decline"] = (out["Delta_Grade"] < 0).astype(int)
    
    # ============================================================
    # (B) Completion Rate Stability
    # ============================================================
    
    approved_sem1 = pd.to_numeric(out.get("Curricular units 1st sem (approved)", 0), errors="coerce")
    enrolled_sem1 = pd.to_numeric(out.get("Curricular units 1st sem (enrolled)", 1), errors="coerce")
    
    approved_sem2 = pd.to_numeric(out.get("Curricular units 2nd sem (approved)", 0), errors="coerce")
    enrolled_sem2 = pd.to_numeric(out.get("Curricular units 2nd sem (enrolled)", 1), errors="coerce")
    
    out["CR_Sem1"] = np.where(enrolled_sem1 > 0, approved_sem1 / enrolled_sem1, np.nan)
    out["CR_Sem2"] = np.where(enrolled_sem2 > 0, approved_sem2 / enrolled_sem2, np.nan)
    out["Delta_CR"] = out["CR_Sem2"] - out["CR_Sem1"]
    
    out["Completion_Collapse"] = ((out["CR_Sem1"] >= 0.9) & (out["CR_Sem2"] <= 0.6)).astype(int)
    
    # ============================================================
    # (C) Threshold Triggers
    # ============================================================
    
    pass_threshold = 9.5
    
    out["Fail_Crossing"] = ((grade_sem1 >= pass_threshold) & (grade_sem2 < pass_threshold)).astype(int)
    out["Borderline_Collapse"] = ((grade_sem1 >= 9) & (grade_sem1 <= 11) & (grade_sem2 < 9)).astype(int)
    
    # ============================================================
    # (D) Supplementary Features
    # ============================================================
    
    eval_sem1 = pd.to_numeric(out.get("Curricular units 1st sem (evaluations)", 0), errors="coerce")
    eval_sem2 = pd.to_numeric(out.get("Curricular units 2nd sem (evaluations)", 0), errors="coerce")
    
    out["EvalPressure_Sem1"] = np.where(enrolled_sem1 > 0, eval_sem1 / enrolled_sem1, np.nan)
    out["EvalPressure_Sem2"] = np.where(enrolled_sem2 > 0, eval_sem2 / enrolled_sem2, np.nan)
    out["Delta_EvalPressure"] = out["EvalPressure_Sem2"] - out["EvalPressure_Sem1"]
    
    without_eval_sem1 = pd.to_numeric(out.get("Curricular units 1st sem (without evaluations)", 0), errors="coerce")
    without_eval_sem2 = pd.to_numeric(out.get("Curricular units 2nd sem (without evaluations)", 0), errors="coerce")
    
    out["GhostRate_Sem1"] = np.where(enrolled_sem1 > 0, without_eval_sem1 / enrolled_sem1, np.nan)
    out["GhostRate_Sem2"] = np.where(enrolled_sem2 > 0, without_eval_sem2 / enrolled_sem2, np.nan)
    out["Delta_GhostRate"] = out["GhostRate_Sem2"] - out["GhostRate_Sem1"]
    out["Ghost_Worsening"] = ((out["GhostRate_Sem1"] < 0.2) & (out["GhostRate_Sem2"] > 0.5)).astype(int)
    
    return out


def get_longitudinal_feature_list() -> dict:
    return {
        "Grade Trajectory": ["Delta_Grade", "Grade_Improvement", "Grade_Decline"],
        "Completion Rate": ["CR_Sem1", "CR_Sem2", "Delta_CR", "Completion_Collapse"],
        "Threshold Crossing": ["Fail_Crossing", "Borderline_Collapse"],
        "Evaluation Pressure": ["EvalPressure_Sem1", "EvalPressure_Sem2", "Delta_EvalPressure"],
        "Ghost Enrollment": ["GhostRate_Sem1", "GhostRate_Sem2", "Delta_GhostRate", "Ghost_Worsening"]
    }


def longitudinal_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    features = get_longitudinal_feature_list()
    rows = []
    for category, feat_list in features.items():
        for feat in feat_list:
            if feat in df.columns:
                rows.append({
                    "Category": category,
                    "Feature": feat,
                    "Type": "Binary" if df[feat].nunique() <= 2 else "Continuous",
                    "Non-null": int(df[feat].notna().sum()),
                    "Mean": f"{df[feat].mean():.3f}" if pd.api.types.is_numeric_dtype(df[feat]) else "N/A"
                })
    return pd.DataFrame(rows)