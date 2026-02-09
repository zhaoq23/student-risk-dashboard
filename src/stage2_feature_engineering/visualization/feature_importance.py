"""
feature_importance.py — Feature Importance Exploration
"""

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Matplotlib configuration
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prepare_modeling_data(df: pd.DataFrame, target_col: str = "Target"):
    """Prepare data for modeling, encoding Target as binary (Dropout vs Others)"""
    target_map = {"Dropout": 1, "Enrolled": 0, "Graduate": 0}
    df_model = df.copy()
    df_model["Target_binary"] = df_model[target_col].map(target_map)
    df_model = df_model[df_model["Target_binary"].notna()]
    
    # Exclude non-numeric and target columns
    exclude_cols = [target_col, "Target_binary", "Subject_Specific", "Academic_Stress_Level"]
    feature_cols = [c for c in df_model.columns if c not in exclude_cols]
    
    X = df_model[feature_cols].select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    y = df_model["Target_binary"]
    
    return X, y, X.columns.tolist()


def l1_logistic_importance(df: pd.DataFrame, target_col: str = "Target", top_n: int = 20):
    """Calculate feature importance using L1-penalized Logistic Regression"""
    X, y, feature_names = prepare_modeling_data(df, target_col)
    
    if len(X) == 0 or len(feature_names) == 0:
        return pd.DataFrame(), None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Using L1 (Lasso) to encourage sparsity and feature selection
    model = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    coefficients = model.coef_[0]
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Abs_Coefficient": np.abs(coefficients)
    })
    
    importance_df = importance_df.sort_values("Abs_Coefficient", ascending=False).head(top_n)
    
    # Plotting L1 Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#EF4444' if c < 0 else '#22C55E' for c in importance_df["Coefficient"]]
    
    ax.barh(range(len(importance_df)), importance_df["Coefficient"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["Feature"], fontsize=9)
    ax.set_xlabel("Coefficient (L1 Logistic)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Features — L1 Logistic Regression", fontsize=13, fontweight="bold", pad=15)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF4444', label='Negative (↓ Dropout Risk)'),
        Patch(facecolor='#22C55E', label='Positive (↑ Dropout Risk)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False)
    
    fig.tight_layout()
    plot_b64 = fig_to_b64(fig)
    
    return importance_df[["Feature", "Coefficient"]], plot_b64


def random_forest_importance(df: pd.DataFrame, target_col: str = "Target", top_n: int = 20):
    """Calculate feature importance using Random Forest Gini Importance"""
    X, y, feature_names = prepare_modeling_data(df, target_col)
    
    if len(X) == 0 or len(feature_names) == 0:
        return pd.DataFrame(), None
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=20,
        random_state=42, n_jobs=-1
    )
    
    model.fit(X, y)
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    
    importance_df = importance_df.sort_values("Importance", ascending=False).head(top_n)
    
    # Plotting RF Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df["Importance"],
            color='#3B82F6', alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["Feature"], fontsize=9)
    ax.set_xlabel("Importance (Gini)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Features — Random Forest", fontsize=13, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    
    fig.tight_layout()
    plot_b64 = fig_to_b64(fig)
    
    return importance_df, plot_b64


def combined_importance_summary(df: pd.DataFrame, target_col: str = "Target", top_n: int = 20):
    """Combine L1 and RF rankings into a single visualization"""
    l1_df, _ = l1_logistic_importance(df, target_col, top_n=50)
    rf_df, _ = random_forest_importance(df, target_col, top_n=50)
    
    if l1_df.empty or rf_df.empty:
        return pd.DataFrame(), None
    
    # Rank mapping
    l1_df = l1_df.copy()
    rf_df = rf_df.copy()
    l1_df['L1_Rank'] = range(1, len(l1_df) + 1)
    rf_df['RF_Rank'] = range(1, len(rf_df) + 1)
    
    # Merge results
    merged = pd.merge(
        l1_df[['Feature', 'Coefficient', 'L1_Rank']], 
        rf_df[['Feature', 'Importance', 'RF_Rank']], 
        on='Feature', 
        how='outer'
    )
    
    # Fill missing rankings for features not in top 50
    merged['Coefficient'] = merged['Coefficient'].fillna(0)
    merged['Importance'] = merged['Importance'].fillna(0)
    merged['L1_Rank'] = merged['L1_Rank'].fillna(51)
    merged['RF_Rank'] = merged['RF_Rank'].fillna(51)
    
    # Calculate Combined Score
    merged['Abs_Coef'] = merged['Coefficient'].abs()
    merged['Combined_Score'] = (merged['Abs_Coef'] * 10 + merged['Importance']) / 2
    
    merged = merged.sort_values('Combined_Score', ascending=False).head(top_n)
    
    # Plotting Combined Ranking
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
    
    features = merged['Feature'].tolist()
    scores = merged['Combined_Score'].tolist()
    l1_ranks = merged['L1_Rank'].astype(int).tolist()
    rf_ranks = merged['RF_Rank'].astype(int).tolist()
    
    y_pos = np.arange(len(features))
    
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.85, edgecolor='white', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Combined Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features: Combined Ranking (L1 + RF)', fontsize=14, fontweight='bold', pad=20)
    
    # Annotate with individual ranks (L1# and RF#)
    for i, (score, l1_r, rf_r) in enumerate(zip(scores, l1_ranks, rf_ranks)):
        rank_text = f'L1#{l1_r}  RF#{rf_r}' if l1_r <= 50 and rf_r <= 50 else \
                    f'L1#{l1_r}' if l1_r <= 50 else \
                    f'RF#{rf_r}' if rf_r <= 50 else ''
        
        ax.text(score + max(scores) * 0.02, i, rank_text, 
                va='center', fontsize=8, color='#374151', fontweight='500')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    
    fig.tight_layout()
    plot_b64 = fig_to_b64(fig)
    
    result_df = merged[['Feature', 'Coefficient', 'Importance', 'Combined_Score', 'L1_Rank', 'RF_Rank']]
    
    return result_df, plot_b64