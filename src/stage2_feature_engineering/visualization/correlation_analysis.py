"""
correlation_analysis.py — Threee kinds of heatmap
"""

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_correlation_heatmap(df: pd.DataFrame, features: list, title: str, figsize=(12, 10)) -> str:
    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        return None
    
    df_numeric = df[features].select_dtypes(include=[np.number])
    if df_numeric.shape[1] < 2:
        return None
    
    corr = df_numeric.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    
    return fig_to_b64(fig)


def generate_three_heatmaps(df: pd.DataFrame) -> dict:
    # Type 1: Static
    static_features = [
        "Age at enrollment", "Admission grade", "Previous qualification (grade)",
        "Gender", "International", "Scholarship holder", "Debtor",
        "Tuition fees up to date", "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)", "Unemployment rate", "Inflation rate", "GDP"
    ]
    
    heatmap1 = plot_correlation_heatmap(df, static_features,
                                        "Type 1: Static Variables Correlation", figsize=(14, 12))
    
    # Type 2: Static vs Longitudinal
    static_subset = [
        "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
        "Age at enrollment", "Admission grade"
    ]
    
    longitudinal_features = [
        "Delta_Grade", "Grade_Improvement", "CR_Sem1", "CR_Sem2", "Delta_CR",
        "Completion_Collapse", "Fail_Crossing", "EvalPressure_Sem2",
        "Delta_EvalPressure", "GhostRate_Sem2", "Delta_GhostRate"
    ]
    
    heatmap2 = plot_correlation_heatmap(df, static_subset + longitudinal_features,
                                        "Type 2: Static vs Longitudinal Features", figsize=(14, 12))
    
    # Type 3: Longitudinal vs NLP
    nlp_features = ["Academic_Stress", "Home_Support_Risk", "Subject_Difficulty_Score"]
    key_longitudinal = [
        "Delta_Grade", "Delta_CR", "Completion_Collapse", "Fail_Crossing",
        "Delta_EvalPressure", "Delta_GhostRate", "Ghost_Worsening"
    ]
    
    heatmap3 = plot_correlation_heatmap(df, key_longitudinal + nlp_features,
                                        "Type 3: Longitudinal vs NLP Features", figsize=(10, 9))
    
    return {
        "static": heatmap1,
        "static_vs_longitudinal": heatmap2,
        "longitudinal_vs_nlp": heatmap3
    }


def correlation_summary_table(df: pd.DataFrame, target_col: str = "Target") -> pd.DataFrame:
    target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    if target_col in df.columns:
        df_temp = df.copy()
        df_temp["Target_numeric"] = df_temp[target_col].map(target_map)
        
        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols 
                       if c != "Target_numeric" and not c.endswith("__is_missing") and not c.endswith("__is_outlier")]
        
        correlations = []
        for col in numeric_cols:
            corr = df_temp[[col, "Target_numeric"]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append({
                    "Feature": col,
                    "Correlation_with_Target": corr,
                    "Abs_Correlation": abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations)
        if not corr_df.empty:
            corr_df = corr_df.sort_values("Abs_Correlation", ascending=False).head(20)
            return corr_df[["Feature", "Correlation_with_Target"]]
    
    return pd.DataFrame(columns=["Feature", "Correlation_with_Target"])
