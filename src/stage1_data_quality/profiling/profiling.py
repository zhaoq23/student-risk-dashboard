"""
profiling.py — Exploratory Data Analysis and Visualization

This module generates statistical summaries and visual distributions 
for different feature groups using Matplotlib and Pandas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from io import BytesIO
import base64
from typing import Dict, List, Tuple
import sys
from pathlib import Path
from data_quality.rules import VALUE_MAPPINGS

# Set up project pathing
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_quality.rules import FEATURE_GROUPS, classify_variable_type, group_columns_by_type


# Matplotlib Global Configuration for High-Quality Export
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


def fig_to_b64(fig) -> str:
    """Converts a Matplotlib figure into a base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_binary_bars(df: pd.DataFrame, cols: List[str], title: str = "") -> str:
    """
    Binary Variables: Generates percentage-based bar charts.
    Returns a base64 encoded PNG string.
    """
    if not cols:
        return None
    
    # Filter valid columns and limit to top 8 to maintain readability
    cols = [c for c in cols if c in df.columns][:8]
    if not cols:
        return None
    
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n))
    if n == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce")
        pct1 = (s == 1).mean() * 100
        pct0 = (s == 0).mean() * 100
        
        # Pull business labels from VALUE_MAPPINGS
        mapping = VALUE_MAPPINGS.get(col, {0: "0", 1: "1"})
        label_0 = str(mapping.get(0, "0"))
        label_1 = str(mapping.get(1, "1"))
        
        ax.bar([label_0, label_1], [pct0, pct1], color=["#93C5FD", "#34D399"])
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_title(col, loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    if title:
        fig.suptitle(title, y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    return fig_to_b64(fig)


def plot_continuous_hist(df: pd.DataFrame, cols: List[str], title: str = "") -> str:
    """
    Continuous Variables: Generates histograms in a 3-column grid layout.
    """
    if not cols:
        return None
    
    cols = [c for c in cols if c in df.columns][:12]
    if not cols:
        return None
    
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    
    for i, ax in enumerate(axes):
        if i < n:
            col = cols[i]
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(x) > 0:
                ax.hist(x, bins=30, color="#60A5FA", alpha=0.85, edgecolor="white")
            ax.set_title(col, loc="left", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.axis("off")
    
    if title:
        fig.suptitle(title, y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    return fig_to_b64(fig)


def plot_macro_line(df: pd.DataFrame, cols: List[str], title: str = "") -> str:
    """
    Macro Indicators: Generates smoothed line charts for sorted observations 
    to visualize distribution trends.
    """
    if not cols:
        return None
    
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.5 * n))
    if n == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols):
        x = pd.to_numeric(df[col], errors="coerce").dropna().sort_values()
        if len(x) > 0:
            # Plot line and area fill for visibility
            ax.plot(range(len(x)), x.values, color="#3B82F6", linewidth=2, alpha=0.8)
            ax.fill_between(range(len(x)), x.values, alpha=0.2, color="#3B82F6")
        
        ax.set_title(col, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Observation (sorted)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
    
    if title:
        fig.suptitle(title, y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    return fig_to_b64(fig)



def plot_categorical_bars(df: pd.DataFrame, cols: List[str], title: str = "", topn: int = 10) -> str:
    """
    Categorical Variables: Generates horizontal frequency bar charts for Top-N values.
    """
    if not cols:
        return None
    
    cols = [c for c in cols if c in df.columns][:6]
    if not cols:
        return None
    
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n))
    if n == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts(dropna=False).head(topn)
        
        # Truncate labels that are too long
        mapping = VALUE_MAPPINGS.get(col, {})
        labels = [str(mapping.get(x, x))[:25] for x in vc.index]
        
        ax.barh(range(len(vc)), vc.values, color="#A78BFA", alpha=0.9)
        ax.set_yticks(range(len(vc)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(col, loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()
    
    if title:
        fig.suptitle(title, y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    return fig_to_b64(fig)


def group_profile_with_charts(df: pd.DataFrame, exclude_target: bool = True) -> Dict[str, Dict]:
    """
    Orchestrates the profiling process for each feature group, 
    generating statistical tables and corresponding distribution charts.
    """
    
    results = {}
    exclude = ["Target"] if exclude_target else []
    
    for group_name, group_cols in FEATURE_GROUPS.items():
        if group_name == "Outcome":
            continue
        
        # Filter valid columns present in the dataset
        cols = [c for c in group_cols if c in df.columns and c not in exclude]
        if not cols:
            continue
        
        # Segment columns by their semantic type
        type_groups = group_columns_by_type(df[cols], exclude_cols=[])
        
        block = {}
        
        # 1. Binary Variables Processing
        binary_cols = type_groups.get("binary", [])
        if binary_cols:
            # Summary Table
            shares = df[binary_cols].apply(lambda x: pd.to_numeric(x, errors="coerce").mean() * 100)
            binary_table = pd.DataFrame({
                "column": shares.index,
                "share_1_pct": shares.values
            }).sort_values("share_1_pct", ascending=False)
            block["binary_table"] = binary_table
            
            # Visualization
            block["binary_chart"] = plot_binary_bars(
                df, binary_cols, 
                title=f"{group_name} — Binary variables"
            )
        
        # 2. Continuous & Count Variables Processing
        cont_cols = type_groups.get("continuous", []) + type_groups.get("count", [])
        if cont_cols:
            # Summary Table (descriptive statistics)
            cont_table = df[cont_cols].describe(percentiles=[.25, .5, .75]).T.reset_index()
            cont_table.rename(columns={"index": "column"}, inplace=True)
            block["continuous_table"] = cont_table
            
            # Visualization: specialized handling for Macro indicators
            if group_name == "Macro context":
                block["continuous_chart"] = plot_macro_line(
                    df, cont_cols,
                    title=f"{group_name} — Macro indicators (line chart)"
                )
            else:
                block["continuous_chart"] = plot_continuous_hist(
                    df, cont_cols,
                    title=f"{group_name} — Continuous/Count variables"
                )
        
        # 3. Categorical Variables Processing (Codes + Text)
        cat_cols = type_groups.get("categorical_code", []) + type_groups.get("categorical_text", [])
        if cat_cols:
            # Summary Table: Top frequencies
            cat_summaries = []
            for c in cat_cols[:10]:
                vc = df[c].value_counts(dropna=False).head(10)
                tmp = pd.DataFrame({
                    "column": c,
                    "value": vc.index.astype(str),
                    "count": vc.values,
                    "pct": vc.values / max(len(df), 1) * 100
                })
                cat_summaries.append(tmp)
            
            if cat_summaries:
                block["categorical_table"] = pd.concat(cat_summaries, ignore_index=True)
                
                # Visualization
                block["categorical_chart"] = plot_categorical_bars(
                    df, cat_cols,
                    title=f"{group_name} — Categorical variables"
                )
        
        if block:
            results[group_name] = block
    
    return results