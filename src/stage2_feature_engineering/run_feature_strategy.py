"""
Stage 2: Feature Engineering Pipeline

Functions:
1. Read data/processed/cleaned_data.csv (output from Stage 1)
2. Generate longitudinal features
3. Generate NLP features (simulated student feedback)
4. Perform correlation analysis and create heatmaps
5. Conduct feature importance analysis
6. Output modeling_data.csv to data/processed/
7. Output feature_strategy.json to outputs/stage2_features/

Usage:
    Run from the project root:
    python src/stage2_feature_engineering/run_feature_strategy.py
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "stage2_feature_engineering"))

# Import feature engineering modules
from feature_engineering.longitudinal import (
    generate_longitudinal_features,
    longitudinal_summary_table
)
from feature_engineering.nlp_features import (
    generate_nlp_features,
    nlp_summary_table,
    generate_nlp_distribution_plot
)

# Import visualization modules
from visualization.correlation_analysis import (
    generate_three_heatmaps,
    correlation_summary_table
)
from visualization.feature_importance import (
    l1_logistic_importance,
    random_forest_importance,
    combined_importance_summary
)


# Configuration
class Config:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Input path (from Stage 1)
        self.input_path = project_root / "data" / "processed" / "cleaned_data.csv"
        
        # Output paths
        self.processed_dir = project_root / "data" / "processed"
        self.output_dir = project_root / "outputs" / "stage2_features"
        
        # Create required directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file paths
        self.modeling_data_path = self.processed_dir / "modeling_data.csv"
        self.strategy_json_path = self.output_dir / "feature_strategy.json"
        
        # Feature engineering parameters
        self.nlp_coverage_rate = 0.7
        self.nlp_seed = 42
        self.importance_top_n = 20


# ============================================================
# Main Pipeline
# ============================================================

def main():
    """Main execution function"""
    
    # Initialize configuration
    cfg = Config(project_root=PROJECT_ROOT)
    
    print("\n" + "="*70)
    print("STAGE 2: Feature Engineering Pipeline")
    print("="*70)
    
    # 1. Check if input file exists
    if not cfg.input_path.exists():
        print(f"\nError: Input file not found!")
        print(f"Expected: {cfg.input_path.relative_to(PROJECT_ROOT)}")
        print(f"\nPlease run Stage 1 first:")
        print(f"python src/stage1_data_quality/run_data_quality.py")
        sys.exit(1)

    # 2. Load data
    print(f"\n[1/7] Loading cleaned data...")
    print(f"From: {cfg.input_path.relative_to(PROJECT_ROOT)}")
    
    df = pd.read_csv(cfg.input_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 3. Generate longitudinal features
    print(f"\n[2/7] Generating longitudinal features...")
    df_long = generate_longitudinal_features(df)
    
    n_new_features = df_long.shape[1] - df.shape[1]
    print(f"Created {n_new_features} longitudinal features")
    
    # 4. Generate NLP features
    print(f"\n[3/7] Generating NLP features (simulated)...")
    df_nlp = generate_nlp_features(
        df_long, 
        coverage_rate=cfg.nlp_coverage_rate, 
        seed=cfg.nlp_seed
    )
    
    n_nlp_features = df_nlp.shape[1] - df_long.shape[1]
    print(f"Created {n_nlp_features} NLP features")
    print(f"Coverage: ~{int(cfg.nlp_coverage_rate*100)}% of students have survey feedback")
    
    # Retrieve NLP sample texts
    nlp_samples = df_nlp.attrs.get('nlp_samples', [])
    print(f"Generated {len(nlp_samples)} sample texts for reporting")
    
    # 5. Generate NLP distribution plot
    print(f"\n[4/7] Generating NLP distribution plot...")
    nlp_distribution_plot = generate_nlp_distribution_plot(df_nlp)
    
    if nlp_distribution_plot:
        print(f"Generated Academic Stress Level distribution plot (base64)")
    else:
        print(f"No NLP distribution data available")
    
    # 6. Correlation analysis + heatmaps
    print(f"\n[5/7] Analyzing correlations and generating heatmaps...")
    heatmaps = generate_three_heatmaps(df_nlp)
    print(f"Generated 3 correlation heatmaps (base64)")
    
    corr_summary = correlation_summary_table(df_nlp, target_col="Target")
    print(f"Top features correlated with Target: {len(corr_summary)}")
    
    # 7. Feature importance + plots
    print(f"\n[6/7] Computing feature importance and generating plots...")
    
    # L1 Logistic Regression
    l1_importance, l1_plot_base64 = l1_logistic_importance(
        df_nlp, 
        target_col="Target", 
        top_n=cfg.importance_top_n
    )
    print(f"L1 Logistic identified {len(l1_importance)} top features")
    
    # Random Forest
    rf_importance, rf_plot_base64 = random_forest_importance(
        df_nlp, 
        target_col="Target", 
        top_n=cfg.importance_top_n
    )
    print(f"Random Forest identified {len(rf_importance)} top features")
    
    # Combined ranking
    combined, combined_plot_base64 = combined_importance_summary(
        df_nlp, 
        target_col="Target", 
        top_n=cfg.importance_top_n
    )
    print(f"Combined ranking includes {len(combined)} features")
    
    # 8. Save outputs
    print(f"\n[7/7] Saving outputs...")
    
    # Save modeling_data.csv
    df_nlp.to_csv(cfg.modeling_data_path, index=False)
    print(f"Saved: {cfg.modeling_data_path.relative_to(PROJECT_ROOT)}")
    
    # Collect summary tables
    long_summary = longitudinal_summary_table(df_nlp)
    nlp_summary = nlp_summary_table(df_nlp)
    
    # Build JSON structure
    strategy_data = {
        "metadata": {
            "n_rows": int(df_nlp.shape[0]),
            "n_original_cols": int(df.shape[1]),
            "n_final_cols": int(df_nlp.shape[1]),
            "n_added_features": int(df_nlp.shape[1] - df.shape[1]),
            "n_longitudinal_features": n_new_features,
            "n_nlp_features": n_nlp_features,
            "n_nlp_samples": len(nlp_samples),
            "nlp_coverage_rate": cfg.nlp_coverage_rate,
        },
        "tables": {
            "longitudinal_summary": long_summary.to_dict(orient='records'),
            "nlp_summary": nlp_summary.to_dict(orient='records'),
            "correlation_summary": corr_summary.to_dict(orient='records'),
            "l1_importance": l1_importance.to_dict(orient='records'),
            "rf_importance": rf_importance.to_dict(orient='records'),
            "combined_importance": combined.to_dict(orient='records'),
            "nlp_samples": nlp_samples
        },
        "visualizations": {
            "heatmaps": {
                "type1_static": heatmaps['static'],
                "type2_static_vs_longitudinal": heatmaps['static_vs_longitudinal'],
                "type3_longitudinal_vs_nlp": heatmaps['longitudinal_vs_nlp']
            },
            "feature_importance": {
                "l1_logistic": l1_plot_base64,
                "random_forest": rf_plot_base64,
                "combined": combined_plot_base64
            },
            "nlp_distribution": nlp_distribution_plot
        }
    }
    
    # Save feature_strategy.json
    with open(cfg.strategy_json_path, 'w', encoding='utf-8') as f:
        json.dump(strategy_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved: {cfg.strategy_json_path.relative_to(PROJECT_ROOT)}")
    
    # 9. Final summary
    print(f"\n{'='*70}")
    print("STAGE 2 COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"\n1. Modeling Data:")
    print(f"{cfg.modeling_data_path.relative_to(PROJECT_ROOT)}")
    print(f"{df_nlp.shape[0]} rows × {df_nlp.shape[1]} columns")
    
    print(f"\n2. Feature Strategy:")
    print(f"{cfg.strategy_json_path.relative_to(PROJECT_ROOT)}")
    
    print(f"\nNext step:")
    print(f"python src/stage3_modeling_action/run_modeling_action.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Ensure Stage 1 has been executed")
        print(f"2. Check that data/processed/cleaned_data.csv exists")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
