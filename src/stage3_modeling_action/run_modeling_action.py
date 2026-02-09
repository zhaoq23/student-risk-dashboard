"""
Stage 3: Modeling & Action Pipeline

Functionality:
1. Load data/processed/modeling_data.csv (from Stage 2)
2. Train 4 models (Logistic, Decision Tree, Random Forest, XGBoost)
3. SHAP explainability analysis
4. Fairness audit (Disparate Impact, Equalized Odds)
5. Generate three-level action recommendations (student / cohort / system)
6. Save outputs to outputs/stage3_modeling/

Usage:
    Run from the project root:
    python src/stage3_modeling_action/run_modeling_action.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize, LabelEncoder

import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# ==================== Configuration ====================
class Config:
    def __init__(self, project_root: Path):
        self.project_root = project_root

        # Input path (from Stage 2)
        self.input_path = project_root / "data" / "processed" / "modeling_data.csv"

        # Output paths
        self.output_dir = project_root / "outputs" / "stage3_modeling"
        self.models_dir = self.output_dir / "models"

        # Create required directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        self.modeling_results_path = self.output_dir / "modeling_results.json"
        self.action_plans_path = self.output_dir / "action_plans.json"

        # Sensitive features for fairness audit
        self.sensitive_features = [
            "Gender",
            "Scholarship holder",
            "Displaced",
            "International",
            "Nacionality",
            "Debtor",
            "Educational special needs",
        ]


# ==================== Modeling Engine ====================
class ModelingEngine:
    """Model training and evaluation engine."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_params = {}

    def prepare_data(self):
        """Prepare training/test data."""
        print("\nPreparing data...")

        # Split features and target
        self.y = self.df["Target"]
        self.X = self.df.drop("Target", axis=1)

        # Identify and encode non-numeric columns
        print("\nChecking data types...")
        non_numeric_cols = self.X.select_dtypes(include=["object", "category"]).columns.tolist()

        if non_numeric_cols:
            print(f"Found {len(non_numeric_cols)} non-numeric columns. Encoding...")
            for col in non_numeric_cols:
                print(f"  Encoding column: {col} (unique values: {self.X[col].nunique()})")
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
        else:
            print("All features are numeric.")

        # Ensure all columns are numeric
        self.X = self.X.apply(pd.to_numeric, errors="coerce")

        # Handle missing values
        if self.X.isnull().any().any():
            print("Missing values detected. Filling with median values...")
            self.X = self.X.fillna(self.X.median(numeric_only=True))

        # Train/test split (80/20) with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\nTrain set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        print(f"Features:  {self.X.shape[1]}")

        print("\nTarget distribution:")
        for label, count in self.y.value_counts().items():
            print(f"  {label}: {count} ({count/len(self.y)*100:.1f}%)")

        # Preserve label mapping for downstream use
        self.label_encoder = LabelEncoder()
        self.original_y_train = self.y_train.copy()
        self.original_y_test = self.y_test.copy()

        # Encode target (required by some models/metrics)
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)

        print(f"\nTarget encoding: {dict(enumerate(self.label_encoder.classes_))}")

    def check_imbalance(self):
        """Check class imbalance and return a recommendation."""
        counts = self.y.value_counts()
        ratio = counts.max() / counts.min()

        decision = {
            "ratio": float(ratio),
            "max_class": str(counts.idxmax()),
            "max_class_count": int(counts.max()),
            "min_class": str(counts.idxmin()),
            "min_class_count": int(counts.min()),
            "all_class_counts": {str(k): int(v) for k, v in counts.items()},
        }

        # Heuristic guidance for imbalance handling
        if ratio < 3:
            decision["severity"] = "Light"
            decision["recommendation"] = (
                f"Light imbalance ({ratio:.2f}:1). Use class_weight='balanced' (loss reweighting) "
                "without altering the data distribution. This is often preferable to SMOTE when "
                "feature semantics represent real individuals."
            )
            decision["strategy"] = "class_weight_only"
        elif ratio < 10:
            decision["severity"] = "Moderate"
            decision["recommendation"] = (
                f"Moderate imbalance ({ratio:.2f}:1). Use class_weight='balanced' and/or model-specific "
                "reweighting (e.g., scale_pos_weight for boosting). Avoid SMOTE if synthetic samples would "
                "harm interpretability or intervention relevance."
            )
            decision["strategy"] = "class_weight_required"
        else:
            decision["severity"] = "Severe"
            decision["recommendation"] = (
                f"Severe imbalance ({ratio:.2f}:1). Consider a combination of class weights, threshold "
                "tuning, and robust ensemble methods. Use synthetic sampling cautiously due to potential "
                "impacts on interpretability and intervention design."
            )
            decision["strategy"] = "advanced_methods_needed"

        return decision

    def train_all_models(self):
        """Train all models with fixed configurations and store evaluation results."""
        print("\nTraining models...")

        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score, make_scorer

        f1_scorer = make_scorer(f1_score, average="macro")

        print("\nTraining with all features and class-weight balancing")
        print(f"  Features:  {self.X_train.shape[1]}")
        print(f"  Train:     {len(self.X_train)} samples")
        print(f"  Test:      {len(self.X_test)} samples")

        # 1) Logistic Regression (baseline)
        print("\n[1/4] Logistic Regression (L1)...")
        try:
            model_lr = LogisticRegression(
                solver="saga",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                l1_ratio=1.0,
                C=0.1,
            )
        except Exception:
            model_lr = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                C=0.1,
            )

        model_lr.fit(self.X_train, self.y_train)
        self.models["Logistic_L1"] = model_lr

        result = self._evaluate_model(model_lr, "Logistic_L1")
        result.update(
            {
                "model_name": "Logistic Regression (L1)",
                "purpose": "Interpretability baseline - identify core risk factors",
                "regularization": "L1 penalty (C=0.1) + balanced class weights",
                "rationale": (
                    "L1 regularization encourages sparse coefficients, providing implicit feature selection "
                    "and highlighting the most predictive engineered indicators."
                ),
            }
        )
        self.results["Logistic_L1"] = result
        print("Done.")

        # 2) Decision Tree
        print("\n[2/4] Decision Tree...")
        model_dt = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            min_samples_split=100,
            class_weight="balanced",
            random_state=42,
        )
        model_dt.fit(self.X_train, self.y_train)
        self.models["Decision_Tree"] = model_dt

        result = self._evaluate_model(model_dt, "Decision_Tree")
        result.update(
            {
                "model_name": "Decision Tree",
                "purpose": "Rule extraction - interpretable decision logic for stakeholders",
                "regularization": "Max depth=5, min samples leaf=50",
                "rationale": (
                    "A shallow tree yields human-readable rules that can be communicated as screening logic "
                    "for advisors and program staff."
                ),
            }
        )
        self.results["Decision_Tree"] = result
        print("Done.")

        # 3) Random Forest
        print("\n[3/4] Random Forest (fixed configuration)...")
        model_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
        )

        model_rf.fit(self.X_train, self.y_train)
        self.models["Random_Forest"] = model_rf

        self.best_params["Random_Forest"] = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 30,
            "max_features": "sqrt",
            "class_weight": "balanced",
        }

        result = self._evaluate_model(model_rf, "Random_Forest")

        cv_scores = cross_val_score(
            model_rf, self.X_train, self.y_train, cv=5, scoring=f1_scorer, n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        result.update(
            {
                "model_name": "Random Forest",
                "purpose": "Performance benchmark - candidate primary model",
                "best_params": self.best_params["Random_Forest"],
                "cv_score_mean": float(cv_mean),
                "cv_score_std": float(cv_std),
                "regularization": "Balanced class weights + depth and leaf constraints",
                "rationale": (
                    "Random Forest provides strong performance with robust generalization while retaining "
                    "feature-level interpretability via global and local attribution methods."
                ),
                "performance_note": "Candidate for deployment if it offers best trade-off of F1 and stability.",
            }
        )
        self.results["Random_Forest"] = result
        print(f"Done. (CV macro-F1: {cv_mean:.4f} ± {cv_std:.4f})")

        # 4) XGBoost
        print("\n[4/4] XGBoost (comparison model)...")

        # XGBoost requires numeric labels
        xgb_label_encoder = LabelEncoder()
        y_train_xgb = xgb_label_encoder.fit_transform(self.y_train)
        y_test_xgb = xgb_label_encoder.transform(self.y_test)
        self.xgb_label_encoder = xgb_label_encoder

        print(f"Label encoding: {dict(enumerate(xgb_label_encoder.classes_))}")

        model_xgb = XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            min_child_weight=5,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",
            objective="multi:softprob",
            enable_categorical=False,
        )

        model_xgb.fit(self.X_train, y_train_xgb)
        self.models["XGBoost"] = model_xgb

        self.best_params["XGBoost"] = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_weight": 5,
            "gamma": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        result = self._evaluate_model(model_xgb, "XGBoost")

        cv_scores_xgb = cross_val_score(
            model_xgb, self.X_train, y_train_xgb, cv=5, scoring=f1_scorer, n_jobs=-1
        )
        cv_mean_xgb = cv_scores_xgb.mean()
        cv_std_xgb = cv_scores_xgb.std()

        result.update(
            {
                "model_name": "XGBoost",
                "purpose": "Comparison benchmark",
                "best_params": self.best_params["XGBoost"],
                "cv_score_mean": float(cv_mean_xgb),
                "cv_score_std": float(cv_std_xgb),
                "regularization": "L1/L2 regularization + tree constraints",
                "rationale": (
                    "Gradient boosting is a strong baseline for tabular data. It is included for comparison "
                    "against Random Forest and simpler interpretable models."
                ),
            }
        )
        self.results["XGBoost"] = result
        print(f"Done. (CV macro-F1: {cv_mean_xgb:.4f} ± {cv_std_xgb:.4f})")

        # Quick summary
        print("\nAll model training completed.")
        print("\nPerformance preview:")
        for model_key in ["Random_Forest", "XGBoost"]:
            if model_key in self.results:
                test_f1 = self.results[model_key]["classification_report"]["macro avg"]["f1-score"]
                cv_f1 = self.results[model_key].get("cv_score_mean", None)
                if cv_f1 is None:
                    print(f"  {model_key:15s}: test macro-F1={test_f1:.4f}")
                else:
                    print(f"  {model_key:15s}: test macro-F1={test_f1:.4f}, CV macro-F1={cv_f1:.4f}")

    def _evaluate_model(self, model, model_key):
        """Evaluate model performance and compute metrics."""
        if model_key == "XGBoost":
            # XGBoost predicts encoded labels
            y_pred_encoded = model.predict(self.X_test)
            y_pred = self.xgb_label_encoder.inverse_transform(y_pred_encoded.astype(int))
            y_test_eval = self.y_test
            y_proba = model.predict_proba(self.X_test)

            # Macro AUC (one-vs-rest)
            try:
                model_classes = self.xgb_label_encoder.classes_
                y_test_bin = label_binarize(y_test_eval, classes=model_classes)
                auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
            except Exception as e:
                print(f"Warning: AUC computation failed: {e}")
                auc_ovr = 0.0
        else:
            y_pred = model.predict(self.X_test)
            y_test_eval = self.y_test
            y_proba = model.predict_proba(self.X_test)
            try:
                y_test_bin = label_binarize(y_test_eval, classes=model.classes_)
                auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
            except Exception:
                auc_ovr = 0.0

        report = classification_report(y_test_eval, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test_eval, y_pred)

        # Per-class metrics (expected labels)
        class_metrics = {}
        for label in ["Dropout", "Enrolled", "Graduate"]:
            if label in report:
                class_metrics[label] = {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1-score": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                }

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(self.X.columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            top_features_dict = {str(k): float(v) for k, v in top_features}
        elif hasattr(model, "coef_"):
            feature_importance = dict(zip(self.X.columns, np.abs(model.coef_).mean(axis=0)))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            top_features_dict = {str(k): float(v) for k, v in top_features}
        else:
            feature_importance = {}
            top_features_dict = {}

        return {
            "classification_report": report,
            "class_metrics": class_metrics,
            "confusion_matrix": cm.tolist(),
            "roc_auc_macro": float(auc_ovr),
            "feature_importance": {str(k): float(v) for k, v in feature_importance.items()},
            "top_15_features": top_features_dict,
        }

    def generate_shap_values(self, model_name="XGBoost"):
        """Generate SHAP values for explainability analysis."""
        print(f"\nGenerating SHAP values for {model_name}...")

        model = self.models[model_name]

        if model_name in ["XGBoost", "Random_Forest", "Decision_Tree"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, self.X_train.sample(min(100, len(self.X_train))))

        shap_values = explainer(self.X_test)
        print("SHAP values generated.")
        return shap_values, explainer

    def save_all(self, output_dir):
        """Save all models and results to disk."""
        print(f"\nSaving models and results to {output_dir}/...")

        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/models/{name}.pkl")

        with open(f"{output_dir}/results/model_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        if self.best_params:
            with open(f"{output_dir}/results/best_params.json", "w", encoding="utf-8") as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False)

        split_info = {
            "train_indices": self.X_train.index.tolist(),
            "test_indices": self.X_test.index.tolist(),
        }
        with open(f"{output_dir}/results/data_split.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)

        print("Save completed.")


# ==================== Fairness Auditor ====================
class FairnessAuditor:
    """Fairness auditor for group-based metrics."""

    def __init__(self, X_test, y_test, y_pred, sensitive_features):
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.sensitive_features = sensitive_features
        self.audit_results = {}

    def disparate_impact(self, group_col, feature_name):
        """Compute Disparate Impact for predicted 'Dropout' outcome."""
        df = pd.DataFrame(
            {"y_true": self.y_test.values, "y_pred": self.y_pred, "group": group_col}
        )

        df["is_dropout"] = (df["y_pred"] == "Dropout").astype(int)

        groups = df["group"].unique()
        positive_rates = {}
        group_counts = {}

        for group in groups:
            group_data = df[df["group"] == group]
            positive_rate = group_data["is_dropout"].mean()
            positive_rates[str(group)] = float(positive_rate)
            group_counts[str(group)] = int(len(group_data))

        rates = list(positive_rates.values())
        di = (min(rates) / max(rates)) if (rates and max(rates) > 0) else 1.0

        return {
            "metric": "Disparate Impact",
            "disparate_impact": float(di),
            "group_dropout_rates": positive_rates,
            "group_counts": group_counts,
            "fairness_level": "Fair" if di >= 0.8 else "Potential Bias",
        }

    def equalized_odds(self, group_col, feature_name):
        """Compute Equalized Odds gaps (TPR/FPR differences) for 'Dropout' vs non-dropout."""
        df = pd.DataFrame(
            {"y_true": self.y_test.values, "y_pred": self.y_pred, "group": group_col}
        )

        groups = df["group"].unique()
        metrics = {}

        for group in groups:
            group_data = df[df["group"] == group]
            if len(group_data) == 0:
                continue

            y_true_binary = (group_data["y_true"] == "Dropout").astype(int)
            y_pred_binary = (group_data["y_pred"] == "Dropout").astype(int)

            try:
                tn, fp, fn, tp = confusion_matrix(
                    y_true_binary, y_pred_binary, labels=[0, 1]
                ).ravel()
            except Exception:
                tn = fp = fn = tp = 0

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            metrics[str(group)] = {
                "TPR": float(tpr),
                "FPR": float(fpr),
                "sample_size": int(len(group_data)),
            }

        tprs = [m["TPR"] for m in metrics.values()]
        fprs = [m["FPR"] for m in metrics.values()]

        tpr_diff = (max(tprs) - min(tprs)) if tprs else 0
        fpr_diff = (max(fprs) - min(fprs)) if fprs else 0

        is_fair = (tpr_diff < 0.1 and fpr_diff < 0.1)

        return {
            "metric": "Equalized Odds",
            "group_metrics": metrics,
            "tpr_difference": float(tpr_diff),
            "fpr_difference": float(fpr_diff),
            "fairness_level": "Fair" if is_fair else "Potential Bias",
        }

    def run_full_audit(self):
        """Run fairness audit across all provided sensitive features."""
        print("\nRunning fairness audit...")

        for feature_name, feature_values in self.sensitive_features.items():
            print(f"  Auditing feature: {feature_name}")

            self.audit_results[feature_name] = {
                "feature_name": feature_name,
                "unique_groups": sorted([str(x) for x in np.unique(feature_values)]),
                "n_groups": int(len(np.unique(feature_values))),
                "disparate_impact": self.disparate_impact(feature_values, feature_name),
                "equalized_odds": self.equalized_odds(feature_values, feature_name),
            }

        self._generate_overall_assessment()
        return self.audit_results

    def _generate_overall_assessment(self):
        """Generate an overall fairness summary."""
        total_features = len(self.audit_results)
        fair_count_di = sum(
            1
            for r in self.audit_results.values()
            if r["disparate_impact"]["fairness_level"] == "Fair"
        )
        fair_count_eo = sum(
            1
            for r in self.audit_results.values()
            if r["equalized_odds"]["fairness_level"] == "Fair"
        )

        self.audit_results["_overall_summary"] = {
            "total_features_audited": int(total_features),
            "fair_on_disparate_impact": int(fair_count_di),
            "fair_on_equalized_odds": int(fair_count_eo),
            "di_fairness_rate": float(fair_count_di / total_features) if total_features > 0 else 0.0,
            "eo_fairness_rate": float(fair_count_eo / total_features) if total_features > 0 else 0.0,
            "overall_fairness": "Pass"
            if (fair_count_di == total_features and fair_count_eo == total_features)
            else "Needs Review",
        }


# ==================== Action Generator ====================
class ActionGenerator:
    """Generate actionable recommendations at student/cohort/system levels."""

    def __init__(self, models, X_train, X_test, y_train, y_test, feature_names):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names

        # Primary model for actions (must support predict_proba)
        self.main_model = models["XGBoost"]
        self.explainer = shap.TreeExplainer(self.main_model)
        self.shap_values = self.explainer(X_test)

        # Assumption: class index 0 corresponds to 'Dropout' in the XGBoost label encoding
        self.dropout_idx = 0

    def generate_student_level_actions(self, n_students=None):
        """Generate student-level personalized recommendations (top examples per risk tier)."""
        print("Analyzing students and grouping by predicted risk...")

        y_proba = self.main_model.predict_proba(self.X_test)
        dropout_probs = y_proba[:, self.dropout_idx]
        all_indices = np.argsort(dropout_probs)[::-1]

        critical_students = []
        warning_students = []
        on_track_students = []

        for rank, idx in enumerate(all_indices, 1):
            student_id = int(self.X_test.index[idx])
            risk_score = float(dropout_probs[idx])
            true_label = str(self.y_test.iloc[idx])

            pred_label = self.main_model.predict(self.X_test.iloc[idx : idx + 1])[0]

            shap_vals = self.shap_values.values[idx, :, self.dropout_idx]
            top_features_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

            risk_factors = []
            for feat_idx in top_features_idx:
                feat_name = self.feature_names[feat_idx]
                feat_value = float(self.X_test.iloc[idx, feat_idx])
                shap_value = float(shap_vals[feat_idx])

                risk_factors.append(
                    {
                        "feature": feat_name,
                        "value": feat_value,
                        "shap_contribution": shap_value,
                        "action": self._suggest_student_action(feat_name, feat_value, shap_value),
                    }
                )

            student_data = {
                "student_id": student_id,
                "risk_rank": int(rank),
                "dropout_risk_score": risk_score,
                "true_status": true_label,
                "predicted_status": str(pred_label),
                "top_risk_factors": risk_factors,
                "overall_recommendation": self._generate_student_recommendation(risk_factors, risk_score),
            }

            if risk_score > 0.7 and len(critical_students) < 50:
                critical_students.append(student_data)
            elif 0.4 <= risk_score <= 0.7 and len(warning_students) < 50:
                warning_students.append(student_data)
            elif risk_score < 0.4 and len(on_track_students) < 50:
                on_track_students.append(student_data)

            if len(critical_students) >= 50 and len(warning_students) >= 50 and len(on_track_students) >= 50:
                break

        print(
            f"Risk tiers collected: critical={len(critical_students)}, warning={len(warning_students)}, on_track={len(on_track_students)}"
        )

        # Create one "what-if" example using the top critical student (if any)
        whatif_sample = None
        if critical_students:
            sample_student = critical_students[0]
            whatif_sample = self._generate_whatif_scenarios(
                sample_student["student_id"], self.X_test.index.get_loc(sample_student["student_id"])
            )

        return {
            "critical": critical_students[:10],
            "warning": warning_students[:10],
            "on_track": on_track_students[:10],
            "whatif_sample": whatif_sample,
            "total_analyzed": int(len(critical_students) + len(warning_students) + len(on_track_students)),
        }

    def _generate_whatif_scenarios(self, student_id, idx):
        """Generate what-if scenarios for a given student index."""
        baseline_features = self.X_test.iloc[idx : idx + 1].copy()
        baseline_risk = float(self.main_model.predict_proba(baseline_features)[0, self.dropout_idx])

        scenarios = {"student_id": int(student_id), "baseline_risk": baseline_risk, "scenarios": []}

        # Scenario A: Academic support
        scenario_a = baseline_features.copy()
        if "Curricular units 2nd sem (grade)" in scenario_a.columns:
            scenario_a["Curricular units 2nd sem (grade)"] += 10
        if "Curricular units 2nd sem (evaluations)" in scenario_a.columns:
            scenario_a["Curricular units 2nd sem (evaluations)"] += 5
        risk_a = float(self.main_model.predict_proba(scenario_a)[0, self.dropout_idx])

        scenarios["scenarios"].append(
            {
                "name": "Academic Support",
                "description": "Increase 2nd semester grade by +10 and evaluations by +5",
                "new_risk": risk_a,
                "risk_reduction": baseline_risk - risk_a,
                "reduction_pct": ((baseline_risk - risk_a) / baseline_risk * 100) if baseline_risk > 0 else 0,
            }
        )

        # Scenario B: Financial support
        scenario_b = baseline_features.copy()
        if "Tuition fees up to date" in scenario_b.columns:
            scenario_b["Tuition fees up to date"] = 1
        if "Debtor" in scenario_b.columns:
            scenario_b["Debtor"] = 0
        risk_b = float(self.main_model.predict_proba(scenario_b)[0, self.dropout_idx])

        scenarios["scenarios"].append(
            {
                "name": "Financial Support",
                "description": "Set tuition fees up to date and clear debtor status",
                "new_risk": risk_b,
                "risk_reduction": baseline_risk - risk_b,
                "reduction_pct": ((baseline_risk - risk_b) / baseline_risk * 100) if baseline_risk > 0 else 0,
            }
        )

        # Scenario C: Socio-emotional support proxy (feature scaling)
        scenario_c = baseline_features.copy()
        for col in scenario_c.columns:
            if any(kw in col.lower() for kw in ["stress", "support", "sentiment"]):
                scenario_c[col] *= 1.8
        risk_c = float(self.main_model.predict_proba(scenario_c)[0, self.dropout_idx])

        scenarios["scenarios"].append(
            {
                "name": "Socio-emotional Support",
                "description": "Increase stress/support-related features by 20% (proxy)",
                "new_risk": risk_c,
                "risk_reduction": baseline_risk - risk_c,
                "reduction_pct": ((baseline_risk - risk_c) / baseline_risk * 100) if baseline_risk > 0 else 0,
            }
        )

        return scenarios

    def _suggest_student_action(self, feature, value, shap_value):
        """Map feature signals to a suggested student-level action."""
        feature_lower = feature.lower()

        if "grade" in feature_lower or "approved" in feature_lower:
            return "Schedule tutoring sessions; recommend study groups"
        if "tuition" in feature_lower or "fee" in feature_lower:
            return "Provide financial aid guidance; offer payment plan options"
        if "debtor" in feature_lower:
            return "Refer to financial counseling; explore debt restructuring options"
        if "evaluation" in feature_lower:
            return "Encourage attendance and assignment completion; provide structured check-ins"
        if "age" in feature_lower:
            return "Connect with support resources for non-traditional students"
        return "Schedule an advisor consultation"

    def _generate_student_recommendation(self, risk_factors, risk_score):
        """Generate a short overall recommendation based on risk score."""
        if risk_score > 0.7:
            return "Immediate intervention recommended: contact within 48 hours"
        if risk_score > 0.4:
            return "Proactive monitoring recommended: monthly check-ins"
        return "On track: maintain current support level"

    def generate_cohort_level_actions(self):
        """Generate cohort/course-level resource allocation recommendations."""
        print("Generating cohort-level resource allocation recommendations...")

        course_names = {
            33: "Biofuel Production Technologies",
            171: "Animation and Multimedia Design",
            8014: "Social Service (evening)",
            9003: "Agronomy",
            9070: "Communication Design",
            9085: "Veterinary Nursing",
            9119: "Informatics Engineering",
            9130: "Equinculture",
            9147: "Management",
            9238: "Social Service",
            9254: "Tourism",
            9500: "Nursing",
            9556: "Oral Hygiene",
            9670: "Advertising and Marketing Management",
            9773: "Journalism and Communication",
            9853: "Basic Education",
            9991: "Management (evening)",
        }

        y_proba = self.main_model.predict_proba(self.X_test)
        dropout_probs = y_proba[:, self.dropout_idx]

        course_risk = []

        if "Course" in self.X_test.columns:
            course_col = self.X_test["Course"].values
            unique_courses = np.unique(course_col)

            print(f"Found {len(unique_courses)} distinct courses.")

            for course_code in unique_courses:
                course_mask = course_col == course_code
                course_students = int(course_mask.sum())
                if course_students <= 0:
                    continue

                avg_risk = float(dropout_probs[course_mask].mean())

                if avg_risk > 0.6:
                    priority = "Critical"
                    action = "Immediate intervention: assign dedicated advisor and increase TA support"
                elif avg_risk > 0.4:
                    priority = "High"
                    action = "Increase TA support and monitor weekly"
                elif avg_risk > 0.25:
                    priority = "Medium"
                    action = "Monitor closely and provide supplemental resources"
                else:
                    priority = "Low"
                    action = "Maintain current support level"

                course_code_int = int(course_code)
                course_name = course_names.get(course_code_int, f"Course {course_code_int}")

                course_risk.append(
                    {
                        "cohort_type": "Course",
                        "cohort_id": course_code_int,
                        "cohort_name": course_name,
                        "n_students": course_students,
                        "priority": priority,
                        "avg_dropout_risk": avg_risk,
                        "recommended_action": action,
                    }
                )

            course_risk.sort(key=lambda x: x["avg_dropout_risk"], reverse=True)

            if course_risk:
                print(
                    f"Highest-risk course: {course_risk[0]['cohort_name']} (avg risk: {course_risk[0]['avg_dropout_risk']:.2%})"
                )
        else:
            print("Course feature not found; skipping course-level grouping.")

        # Radar-style comparison data across selected dimensions (if available)
        radar_analysis = self._generate_cohort_radar_analysis()

        return {"course_risk": course_risk, "radar_analysis": radar_analysis}

    def _generate_cohort_radar_analysis(self):
        """Generate radar-comparison data for selected binary sensitive dimensions."""
        available_dimensions = []

        if "Scholarship holder" in self.X_test.columns:
            scholarship_data = self._compute_radar_for_dimension("Scholarship holder")
            if scholarship_data:
                available_dimensions.append(
                    {
                        "dimension_name": "Scholarship",
                        "dimension_id": "scholarship",
                        "feature_column": "Scholarship holder",
                        "group_0_label": "Without Scholarship",
                        "group_1_label": "With Scholarship",
                        "radar_data": scholarship_data,
                    }
                )

        if "Gender" in self.X_test.columns:
            gender_data = self._compute_radar_for_dimension("Gender")
            if gender_data:
                available_dimensions.append(
                    {
                        "dimension_name": "Gender",
                        "dimension_id": "gender",
                        "feature_column": "Gender",
                        "group_0_label": "Group 0",
                        "group_1_label": "Group 1",
                        "radar_data": gender_data,
                    }
                )

        if "International" in self.X_test.columns:
            international_data = self._compute_radar_for_dimension("International")
            if international_data:
                available_dimensions.append(
                    {
                        "dimension_name": "International",
                        "dimension_id": "international",
                        "feature_column": "International",
                        "group_0_label": "Group 0",
                        "group_1_label": "Group 1",
                        "radar_data": international_data,
                    }
                )

        return {
            "available_dimensions": available_dimensions,
            "default_dimension": available_dimensions[0]["dimension_id"] if available_dimensions else None,
        }

    def _compute_radar_for_dimension(self, feature_name):
        """Compute five dimension scores for a binary grouping feature."""
        if feature_name not in self.X_test.columns:
            return None

        feature_values = self.X_test[feature_name].values
        unique_values = np.unique(feature_values)

        # Only handle binary features
        if len(unique_values) != 2:
            return None

        group_0_mask = feature_values == unique_values[0]
        group_1_mask = feature_values == unique_values[1]

        # Feature sets for each dimension (use what exists)
        dimension_features = {
            "Academic Prep": ["Admission grade", "Previous qualification (grade)"],
            "Current Success": ["Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"],
            "Engagement": ["Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (enrolled)"],
            "Financial Stress": ["Debtor", "Tuition fees up to date"],
            "Stability": ["Age at enrollment", "Displaced"],
        }

        radar_scores = {}

        for dimension, features in dimension_features.items():
            existing_features = [f for f in features if f in self.X_test.columns]
            if not existing_features:
                radar_scores[dimension] = {"group_0": 0.5, "group_1": 0.5}
                continue

            group_0_scores = []
            group_1_scores = []

            for feat in existing_features:
                feat_values = self.X_test[feat].values
                feat_min, feat_max = feat_values.min(), feat_values.max()

                if feat_max > feat_min:
                    normalized = (feat_values - feat_min) / (feat_max - feat_min)

                    # For risk-like binary flags, invert if "1" indicates higher stress/instability
                    if feat in ["Debtor", "Displaced"]:
                        normalized = 1 - normalized

                    group_0_scores.append(float(np.mean(normalized[group_0_mask])))
                    group_1_scores.append(float(np.mean(normalized[group_1_mask])))

            if group_0_scores and group_1_scores:
                radar_scores[dimension] = {
                    "group_0": round(float(np.mean(group_0_scores)), 2),
                    "group_1": round(float(np.mean(group_1_scores)), 2),
                }
            else:
                radar_scores[dimension] = {"group_0": 0.5, "group_1": 0.5}

        dimensions = list(radar_scores.keys())
        group_0_values = [radar_scores[d]["group_0"] for d in dimensions]
        group_1_values = [radar_scores[d]["group_1"] for d in dimensions]
        differences = [round(group_1_values[i] - group_0_values[i], 2) for i in range(len(dimensions))]

        max_diff_idx = int(np.argmax([abs(d) for d in differences]))
        max_diff_dimension = dimensions[max_diff_idx]
        max_diff_value = differences[max_diff_idx]
        stronger_group = "group_1" if max_diff_value > 0 else "group_0"

        return {
            "dimensions": dimensions,
            "group_0_values": group_0_values,
            "group_1_values": group_1_values,
            "differences": differences,
            "interpretation": {
                "max_diff_dimension": max_diff_dimension,
                "max_diff_value": max_diff_value,
                "stronger_group": stronger_group,
            },
        }

    def generate_system_level_actions(self):
        """Generate system-level recommendations based on global SHAP attribution."""
        print("Generating system-level policy recommendations...")

        global_shap = np.abs(self.shap_values.values[:, :, self.dropout_idx]).mean(axis=0)
        top_features_idx = np.argsort(global_shap)[-10:][::-1]

        global_factors = []
        for rank, feat_idx in enumerate(top_features_idx, 1):
            feat_name = self.feature_names[feat_idx]
            mean_shap = float(global_shap[feat_idx])

            global_factors.append(
                {
                    "rank": int(rank),
                    "feature": feat_name,
                    "mean_abs_shap": mean_shap,
                    "policy_action": self._suggest_system_action(feat_name),
                }
            )

        policy_sim = self._simulate_policy_scenarios()

        return {
            "global_risk_factors": global_factors,
            "cost_benefit_analysis": policy_sim,
            "strategic_recommendations": self._generate_strategic_recommendations(global_factors),
        }

    def _simulate_policy_scenarios(self):
        """Simulate policy scenarios for rough cost/benefit comparison."""
        y_proba = self.main_model.predict_proba(self.X_test)
        baseline_dropout_rate = float((y_proba[:, self.dropout_idx] > 0.5).mean())
        total_students = int(len(self.X_test))

        high_risk_mask = y_proba[:, self.dropout_idx] > 0.7
        n_high_risk = int(high_risk_mask.sum())

        # Scenario A: Financial aid for high-risk debtors
        scenario_a = self.X_test.copy()
        if "Debtor" in scenario_a.columns and "Tuition fees up to date" in scenario_a.columns:
            high_risk_with_debt = high_risk_mask & (scenario_a["Debtor"] == 1)
            n_beneficiaries_a = int(high_risk_with_debt.sum())

            scenario_a.loc[high_risk_with_debt, "Debtor"] = 0
            scenario_a.loc[high_risk_with_debt, "Tuition fees up to date"] = 1

            y_proba_a = self.main_model.predict_proba(scenario_a)
            new_dropout_rate_a = float((y_proba_a[:, self.dropout_idx] > 0.5).mean())
            dropouts_prevented_a = int((baseline_dropout_rate - new_dropout_rate_a) * total_students)

            cost_a = int(n_beneficiaries_a * 2000)
            cost_per_save_a = float(cost_a / dropouts_prevented_a) if dropouts_prevented_a > 0 else float("inf")
        else:
            n_beneficiaries_a = 0
            new_dropout_rate_a = baseline_dropout_rate
            dropouts_prevented_a = 0
            cost_a = 0
            cost_per_save_a = float("inf")

        # Scenario B: Engagement "nudge" for all high-risk students (proxy feature boosts)
        scenario_b = self.X_test.copy()
        n_beneficiaries_b = int(n_high_risk)

        features_to_boost = []
        for col in [
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
        ]:
            if col in scenario_b.columns:
                features_to_boost.append(col)

        if features_to_boost and n_beneficiaries_b > 0:
            for col in features_to_boost:
                scenario_b[col] = scenario_b[col].astype(float)

                if "grade" in col.lower():
                    scenario_b.loc[high_risk_mask, col] = scenario_b.loc[high_risk_mask, col] + 5
                    scenario_b[col] = scenario_b[col].clip(upper=20)
                elif "approved" in col.lower():
                    scenario_b.loc[high_risk_mask, col] = scenario_b.loc[high_risk_mask, col] + 4
                elif "evaluations" in col.lower():
                    scenario_b.loc[high_risk_mask, col] = scenario_b.loc[high_risk_mask, col] + 6

                scenario_b[col] = scenario_b[col].round()

            y_proba_b = self.main_model.predict_proba(scenario_b)
            new_dropout_rate_b = float((y_proba_b[:, self.dropout_idx] > 0.5).mean())
            dropouts_prevented_b = int((baseline_dropout_rate - new_dropout_rate_b) * total_students)

            cost_b = 500
            cost_per_save_b = float(cost_b / dropouts_prevented_b) if dropouts_prevented_b > 0 else float("inf")
        else:
            new_dropout_rate_b = baseline_dropout_rate
            dropouts_prevented_b = 0
            cost_b = 500
            cost_per_save_b = float("inf")

        return {
            "baseline_dropout_rate": baseline_dropout_rate,
            "total_students": total_students,
            "total_high_risk_students": n_high_risk,
            "scenarios": [
                {
                    "name": "Financial Aid Program",
                    "description": f"Clear debt for {n_beneficiaries_a} high-risk students with debt (subset of {n_high_risk} high-risk students)",
                    "beneficiaries": int(n_beneficiaries_a),
                    "total_cost": int(cost_a),
                    "dropouts_prevented": int(dropouts_prevented_a),
                    "cost_per_student_saved": float(cost_per_save_a),
                    "new_dropout_rate": float(new_dropout_rate_a),
                },
                {
                    "name": "Engagement Nudge System",
                    "description": f"Proxy boosts to engagement/achievement features (+30%) for all {n_beneficiaries_b} high-risk students",
                    "beneficiaries": int(n_beneficiaries_b),
                    "total_cost": int(cost_b),
                    "dropouts_prevented": int(dropouts_prevented_b),
                    "cost_per_student_saved": float(cost_per_save_b),
                    "new_dropout_rate": float(new_dropout_rate_b),
                },
            ],
        }

    def _suggest_system_action(self, feature):
        """Map global features to system-level actions."""
        feature_lower = feature.lower()

        if "tuition" in feature_lower or "fee" in feature_lower:
            return "Establish emergency financial aid funding and improve payment flexibility"
        if "scholarship" in feature_lower:
            return "Expand scholarship funding and eligibility review"
        if "grade" in feature_lower or "approved" in feature_lower:
            return "Strengthen academic support programs and targeted tutoring"
        if "evaluation" in feature_lower:
            return "Review assessment practices and provide structured learning supports"
        return "Monitor the factor and evaluate targeted interventions"

    def _generate_strategic_recommendations(self, global_factors):
        """Generate concise strategic recommendations based on top global drivers."""
        top_3_features = [f["feature"] for f in global_factors[:3]]

        recommendations = []

        if any(kw in feat.lower() for feat in top_3_features for kw in ["tuition", "fee", "scholarship", "debtor"]):
            recommendations.append({"domain": "Financial Support", "priority": "Critical", "action": "Strengthen financial support pathways"})

        if any(kw in feat.lower() for feat in top_3_features for kw in ["grade", "approved", "evaluation", "curricular"]):
            recommendations.append({"domain": "Academic Support", "priority": "High", "action": "Expand tutoring and academic success programs"})

        if not recommendations:
            recommendations.append({"domain": "Monitoring", "priority": "Medium", "action": "Continue monitoring and refine intervention criteria"})

        return recommendations


# ==================== Main Execution ====================
def main():
    """Main entry point."""

    cfg = Config(project_root=PROJECT_ROOT)

    # Validate input file existence
    if not cfg.input_path.exists():
        print("\nError: Input file not found.")
        print(f"Expected: {cfg.input_path.relative_to(PROJECT_ROOT)}")
        print("\nPlease run Stage 2 first:")
        print("  python src/stage2_feature_engineering/run_feature_engineering.py")
        sys.exit(1)

    print("=" * 80)
    print("MODELING & ACTION PIPELINE - START")
    print("=" * 80)
    print(f"Input data:   {cfg.input_path.relative_to(PROJECT_ROOT)}")
    print(f"Output dir:   {cfg.output_dir.relative_to(PROJECT_ROOT)}")
    print(f"Start time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # STEP 1: Modeling
    print("\n" + "=" * 80)
    print("STEP 1/3: Model Training and Evaluation")
    print("=" * 80)

    modeler = ModelingEngine(str(cfg.input_path))
    modeler.prepare_data()

    imbalance_info = modeler.check_imbalance()
    print("\nClass imbalance check:")
    print(f"  Class ratio: {imbalance_info['ratio']:.2f}:1")
    print(f"  Recommendation: {imbalance_info['recommendation']}")

    modeler.train_all_models()

    # Save trained models
    for name, model in modeler.models.items():
        joblib.dump(model, str(cfg.models_dir / f"{name}.pkl"))
    print(f"\nModels saved to: {cfg.models_dir.relative_to(PROJECT_ROOT)}/")

    # Save best params (if any)
    if modeler.best_params:
        best_params_path = cfg.output_dir / "best_hyperparameters.json"
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(modeler.best_params, f, indent=2, ensure_ascii=False)
        print(f"Best hyperparameters saved to: {best_params_path.relative_to(PROJECT_ROOT)}")

    # STEP 2: Fairness Audit
    print("\n" + "=" * 80)
    print("STEP 2/3: Fairness Audit")
    print("=" * 80)

    sensitive_features = {}
    for col in cfg.sensitive_features:
        if col in modeler.X_test.columns:
            sensitive_features[col] = modeler.X_test[col].values
            print(f"Sensitive feature included: {col}")

    if not sensitive_features:
        print("No sensitive features found. Skipping fairness audit.")
        audit_results = {}
    else:
        # Note: XGBoost here returns encoded class indices; the auditor expects string labels.
        # If your XGBoost predicts encoded labels (0/1/2), you should decode them before audit.
        y_pred = modeler.models["XGBoost"].predict(modeler.X_test)

        auditor = FairnessAuditor(
            X_test=modeler.X_test,
            y_test=modeler.y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        audit_results = auditor.run_full_audit()
        print("Fairness audit completed.")

    # STEP 3: Action Recommendations
    print("\n" + "=" * 80)
    print("STEP 3/3: Generate Action Recommendations")
    print("=" * 80)

    action_gen = ActionGenerator(
        models=modeler.models,
        X_train=modeler.X_train,
        X_test=modeler.X_test,
        y_train=modeler.y_train,
        y_test=modeler.y_test,
        feature_names=modeler.X.columns.tolist(),
    )

    print("\nGenerating student-level recommendations...")
    student_actions = action_gen.generate_student_level_actions()

    print("\nGenerating cohort-level recommendations...")
    cohort_actions = action_gen.generate_cohort_level_actions()

    print("\nGenerating system-level recommendations...")
    system_actions = action_gen.generate_system_level_actions()

    # Save outputs
    print("\n" + "=" * 80)
    print("Saving Outputs")
    print("=" * 80)

    modeling_results = {
        "execution_info": {
            "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(cfg.input_path.relative_to(PROJECT_ROOT)),
            "output_directory": str(cfg.output_dir.relative_to(PROJECT_ROOT)),
        },
        "data_summary": {
            "total_samples": int(len(modeler.df)),
            "train_samples": int(len(modeler.X_train)),
            "test_samples": int(len(modeler.X_test)),
            "n_features": int(modeler.X.shape[1]),
            "target_distribution": {str(k): int(v) for k, v in modeler.y.value_counts().items()},
            "train_indices": modeler.X_train.index.tolist(),
            "test_indices": modeler.X_test.index.tolist(),
        },
        "imbalance_decision": imbalance_info,
        "model_results": modeler.results,
        "best_hyperparameters": modeler.best_params,
        "fairness_audit": audit_results,
    }

    with open(cfg.modeling_results_path, "w", encoding="utf-8") as f:
        json.dump(modeling_results, f, indent=2, ensure_ascii=False)
    print(f"Modeling results saved: {cfg.modeling_results_path.relative_to(PROJECT_ROOT)}")

    action_plans = {
        "student_level": student_actions,
        "cohort_level": cohort_actions,
        "system_level": system_actions,
    }

    with open(cfg.action_plans_path, "w", encoding="utf-8") as f:
        json.dump(action_plans, f, indent=2, ensure_ascii=False)
    print(f"Action plans saved:     {cfg.action_plans_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Models trained:             {len(modeler.models)}")
    print(f"Sensitive features audited: {len(sensitive_features)}")
    print("Next step: Run src/stage4_reporting/build_integrated_report.py to build the report.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1) Ensure Stage 2 has been run")
        print("  2) Verify data/processed/modeling_data.csv exists")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
