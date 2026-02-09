"""
target_analysis.py — Education Outcome Stratified Analysis (Visualization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from io import BytesIO
import base64
from typing import List, Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep your existing logic for variable classification
from data_quality.rules import classify_variable_type, VALUE_MAPPINGS

# Matplotlib Global Configuration
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#E5E7EB",
    "axes.labelcolor": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "font.size": 10,
})

# Mapping for Target values to professional labels
TARGET_LABEL_MAP = {
    "0": "Dropout",
    "1": "Enrolled",
    "2": "Graduate",
    "Dropout": "Dropout",
    "Enrolled": "Enrolled",
    "Graduate": "Graduate"
}

COLORS = {
    "Graduate": "#22C55E",
    "Dropout": "#EF4444",
    "Enrolled": "#3B82F6"
}


def fig_to_b64(fig) -> str:
    """Converts a matplotlib figure to a base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_stacked_by_target(df: pd.DataFrame, col: str, target_col: str = "Target") -> str:
    """
    For Binary/Categorical Variables: 
    Displays a 100% stacked bar chart of Education Outcomes grouped by feature values.
    """
    if col not in df.columns:
        return None
    
    # Building the crosstab
    sub = df[[col, target_col]].copy()
    
    # LABEL MAPPING LOGIC: Map Target codes to Words
    sub[target_col] = sub[target_col].astype(str).map(TARGET_LABEL_MAP).fillna("Unknown")
    
    # Map feature values (e.g., 1 -> Male)
    mapping = VALUE_MAPPINGS.get(col, {})
    sub[col] = sub[col].apply(lambda x: mapping.get(x, x)).astype(str)
    
    # Calculate percentages
    tab = pd.crosstab(sub[col], sub[target_col], normalize="index") * 100
    
    # Ensure column order
    for k in ["Graduate", "Enrolled", "Dropout"]:
        if k not in tab.columns:
            tab[k] = 0.0
    tab = tab[["Graduate", "Enrolled", "Dropout"]]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab.index))
    
    for tgt in tab.columns:
        ax.bar(
            x, tab[tgt].values, bottom=bottom,
            label=tgt, color=COLORS.get(tgt, "#9CA3AF"),
            edgecolor="white", linewidth=1.5
        )
        bottom += tab[tgt].values
    
    ax.set_xticks(x)
    # Truncate labels for cleaner UI
    ax.set_xticklabels([str(lbl)[:20] for lbl in tab.index], rotation=0 if len(tab) <= 5 else 15)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    # Updated title name per your request
    ax.set_title(f"Education Outcome distribution by {col}", loc="left", fontsize=12)
    ax.legend(frameon=False, ncols=3, bbox_to_anchor=(0, -0.15), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_box_by_target(df: pd.DataFrame, col: str, target_col: str = "Target") -> str:
    """
    For Continuous Variables: 
    Boxplots categorized by Education Outcome.
    """
    if col not in df.columns:
        return None
    
    sub = df[[col, target_col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    
    # LABEL MAPPING LOGIC: Map Target codes to Words
    sub[target_col] = sub[target_col].astype(str).map(TARGET_LABEL_MAP).fillna("Unknown")
    sub = sub.dropna()
    
    if len(sub) == 0:
        return None
    
    order = ["Graduate", "Enrolled", "Dropout"]
    data = [sub.loc[sub[target_col] == t, col].values for t in order]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6)
    
    # Boxplot aesthetics
    fill_colors = [COLORS["Graduate"], COLORS["Enrolled"], COLORS["Dropout"]]
    for patch, fc in zip(bp["boxes"], fill_colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.3)
        patch.set_edgecolor(fc)
        patch.set_linewidth(1.5)
    
    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color="#374151", linewidth=1.2)
    
    # Updated title name per your request
    ax.set_title(f"{col} by Education Outcome", loc="left", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    
    fig.tight_layout()
    return fig_to_b64(fig)


def analyze_target_slices(df: pd.DataFrame, slice_cols: List[str], target_col: str = "Target") -> Dict:
    """
    Generates Education Outcome comparative analysis for each slice variable.
    
    Returns:
    {
        "variable_name": {
            "type": "binary" | "continuous" | "categorical",
            "chart": base64_str,
            "table": DataFrame (Optional)
        }
    }
    """
    
    results = {}
    
    for col in slice_cols:
        if col not in df.columns:
            continue
        
        var_type = classify_variable_type(col, df[col])
        
        item = {"type": var_type}
        
        if var_type in ["binary", "categorical_code", "categorical_text"]:
            # Categorical variables: Stacked bar chart
            item["chart"] = plot_stacked_by_target(df, col, target_col)
            
            # Optional: Generate crosstab with mapped names for the table
            sub_tmp = df[[col, target_col]].copy()
            sub_tmp[target_col] = sub_tmp[target_col].astype(str).map(TARGET_LABEL_MAP).fillna("Unknown")
            # Map X-axis feature labels in the table too
            mapping = VALUE_MAPPINGS.get(col, {})
            sub_tmp[col] = sub_tmp[col].apply(lambda x: mapping.get(x, x))
            
            tab = pd.crosstab(sub_tmp[col], sub_tmp[target_col], normalize="index") * 100
            tab = tab.reset_index()
            item["table"] = tab
        
        elif var_type in ["continuous", "count"]:
            # Continuous variables: Boxplot
            item["chart"] = plot_box_by_target(df, col, target_col)
            
            # Generate statistics table with mapped names
            sub_tmp = df[[col, target_col]].copy()
            sub_tmp[target_col] = sub_tmp[target_col].astype(str).map(TARGET_LABEL_MAP).fillna("Unknown")
            grouped = sub_tmp.groupby(target_col)[col].describe()
            item["table"] = grouped.reset_index()
        
        results[col] = item
    
    return results