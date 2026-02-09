# 🎓 Student Success Pipeline

**An end-to-end machine learning pipeline for predicting college student dropout risk and generating actionable intervention strategies.**

---

## The Problem

While high school dropout rates are declining, **32.9% of U.S. college students still fail to graduate**. In New York alone, over 1.9 million individuals have some college credit but no credential — a population that grew by 2.3% in the past year.

The core priority for education leaders: **identify the 1-in-4 students at risk of dropping out before they lose momentum.**

## Our Solution

This pipeline uses machine learning to identify at-risk students at an early stage. By providing actionable insights across three dimensions, we enable educators to implement timely support strategies:

| Level | Purpose |
|-------|---------|
| **Student-Level** | Individual risk profiles for personalized intervention |
| **Cohort-Level** | Trend analysis for specific student demographics |
| **System-Level** | Strategic dashboards for school and district leadership |

## Pipeline Architecture

The system is organized into four sequential stages, orchestrated by a single master controller:

```
run_master_pipeline.py          ← One-click execution
        │
        ├── Stage 1: Data Quality Assessment
        │     └── Profiling, validation, missing data analysis
        │
        ├── Stage 2: Feature Engineering
        │     └── Longitudinal features, interaction terms, selection
        │
        ├── Stage 3: Modeling & Action Planning
        │     ├── Multi-model training (Random Forest, XGBoost, LASSO)
        │     ├── Fairness auditing across demographic groups
        │     └── Three-tiered intervention strategy generation
        │
        └── Stage 4: Integrated Reporting
              └── Interactive HTML dashboard with executive summary
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your data

Put the raw dataset in `data/raw/data.csv`. This project uses the [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset.

### 3. Run the full pipeline

```bash
python run_master_pipeline.py
```

That's it. The pipeline will execute all four stages sequentially and produce a final interactive report at `outputs/reports/final_report.html`.

## Project Structure

```
student-success-pipeline/
│
├── run_master_pipeline.py            # 🔥 Master entry point
├── config.yaml                       # Unified configuration
├── requirements.txt                  # Python dependencies
│
├── src/
│   ├── stage1_data_quality/          # Data profiling & validation
│   ├── stage2_feature_engineering/   # Feature construction & selection
│   ├── stage3_modeling_action/       # ML models, fairness audit, interventions
│   └── stage4_reporting/             # HTML report generation
│
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Auto-generated cleaned data
│
└── outputs/
    ├── stage1_quality/               # Data quality metrics
    ├── stage2_features/              # Feature strategy artifacts
    ├── stage3_modeling/              # Trained models & action plans
    └── reports/                      # Final interactive HTML report
```

## Key Features

- **Automated Data Governance** — Schema validation, missing value profiling, outlier detection, and data quality scoring
- **Advanced Feature Engineering** — Longitudinal academic trajectory features, demographic interaction terms, and statistical feature selection
- **Multi-Model Comparison** — Trains and benchmarks Random Forest, XGBoost, and LASSO with hyperparameter tuning
- **Fairness Auditing** — Evaluates model performance across demographic subgroups to flag potential bias
- **Tiered Intervention Plans** — Generates student-level, cohort-level, and system-level action recommendations
- **Interactive Dashboard** — Single HTML report combining executive summary, data governance findings, and modeling results

## Configuration

All pipeline parameters are centralized in `config.yaml`:

```yaml
# Example configuration
data:
  raw_path: data/raw/data.csv
  target_column: Target

modeling:
  test_size: 0.2
  random_state: 42
  models:
    - random_forest
    - xgboost
    - lasso
```

## Data Source

This project utilizes the [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset from the UCI Machine Learning Repository. Originally created to help reduce academic attrition in higher education, this data allows us to demonstrate how machine learning can effectively flag students at risk during their academic journey.
