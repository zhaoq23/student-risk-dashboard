"""
get_features_modeling.py
"""

from html import escape
import pandas as pd


def df_to_html(df):
    """Render a DataFrame as an HTML table."""
    if df is None or df.empty:
        return '<p class="note">No data available.</p>'
    return df.to_html(index=False, border=0, classes="data-table")


# ============================================================================
# FEATURE STRATEGY
# ============================================================================

def generate_feature_strategy_engineering(pipeline_data):
    """Generate the Feature Engineering page (Longitudinal + NLP with dropdown)."""

    long_summary = pipeline_data.get("longitudinal_summary", [])
    nlp_summary = pipeline_data.get("nlp_summary", [])
    nlp_samples = pipeline_data.get("nlp_samples", [])
    nlp_dist_plot = pipeline_data.get("nlp_distribution_plot", "")

    # Feature description map
    feature_descriptions = {
        "Delta_Grade": "Change in average grade from Semester 1 to Semester 2",
        "Grade_Improvement": "Binary indicator of grade improvement",
        "Grade_Decline": "Binary indicator of grade decline",
        "CR_Sem1": "Credit Ratio Semester 1 (approved/enrolled)",
        "CR_Sem2": "Credit Ratio Semester 2 (approved/enrolled)",
        "Delta_CR": "Change in Credit Ratio from Semester 1 to Semester 2",
        "Completion_Collapse": "Binary indicator of completion rate collapse",
        "Fail_Crossing": "Binary indicator of crossing the failure threshold",
        "Borderline_Collapse": "Binary indicator of borderline performance collapse",
        "EvalPressure_Sem1": "Evaluation pressure Semester 1 (evaluations/enrolled)",
        "EvalPressure_Sem2": "Evaluation pressure Semester 2 (evaluations/enrolled)",
        "Delta_EvalPressure": "Change in evaluation pressure between semesters",
        "GhostRate_Sem1": "Ghost enrollment rate Semester 1",
        "GhostRate_Sem2": "Ghost enrollment rate Semester 2",
        "Delta_GhostRate": "Change in ghost enrollment rate",
        "GhostFlag": "Binary indicator of significant ghost enrollment",
    }

    # Build longitudinal table rows
    long_rows = []
    for item in long_summary:
        feat_name = item.get("Feature", "")
        description = feature_descriptions.get(feat_name, "N/A")
        feat_type = item.get("Type", "N/A")
        mean_val = item.get("Mean", "N/A")

        long_rows.append(
            f"""
            <tr>
                <td><strong>{escape(str(feat_name))}</strong></td>
                <td>{escape(str(description))}</td>
                <td>{escape(str(feat_type))}</td>
                <td>{escape(str(mean_val))}</td>
            </tr>
        """
        )

    longitudinal_html = f"""
    <div class="sub-panel active" id="engineering_longitudinal">
        <h3>Longitudinal Features</h3>
        <p class="description">Time-series features capturing student trajectory over semesters</p>

        <table class="data-table" style="font-size: 12px;">
            <tr>
                <th>Feature Name</th>
                <th>Description</th>
                <th>Type</th>
                <th>Mean</th>
            </tr>
            {''.join(long_rows) if long_rows else '<tr><td colspan="4">No longitudinal features available</td></tr>'}
        </table>

        <div class="alert alert-info">
            <strong>Key Insight:</strong> Longitudinal features (especially <strong>Delta_CR</strong> and <strong>CR_Sem2</strong>)
            often capture whether performance is improving or declining over time.
            Delta-based features can add signal beyond point-in-time snapshots by reflecting trajectory and momentum.
        </div>
    </div>
    """

    # NLP feature description map
    nlp_feature_descriptions = {
        "Academic_Stress": "Stress level indicator derived from sentiment analysis",
        "Academic_Stress_Level": "Categorical stress level (Low/Medium/High)",
        "Home_Support_Risk": "Family support risk score (0=stable, 1=at-risk)",
        "Subject_Specific": "Subject-specific difficulties identified",
        "Subject_Risk_Flag": "Binary indicator of subject-specific difficulty",
        "Subject_Difficulty_Score": "Numerical difficulty score for subject",
    }

    nlp_source_map = {
        "Academic_Stress": "Sentiment analysis",
        "Academic_Stress_Level": "Sentiment analysis",
        "Home_Support_Risk": "Topic modeling",
        "Subject_Specific": "Keyword extraction",
        "Subject_Risk_Flag": "Keyword extraction",
        "Subject_Difficulty_Score": "Keyword extraction",
    }

    # Build NLP table rows
    nlp_rows = []
    for item in nlp_summary:
        feat_name = item.get("Feature", "")
        description = nlp_feature_descriptions.get(feat_name, "N/A")
        source = nlp_source_map.get(feat_name, "N/A")
        coverage = item.get("Coverage", "N/A")
        mean_val = item.get("Mean", "N/A")

        nlp_rows.append(
            f"""
            <tr>
                <td><strong>{escape(str(feat_name))}</strong></td>
                <td>{escape(str(description))}</td>
                <td>{escape(str(source))}</td>
                <td>{escape(str(coverage))}</td>
                <td>{escape(str(mean_val))}</td>
            </tr>
        """
        )

    # Build sample text boxes
    sample_boxes = []
    for sample in nlp_samples[:3]: 
        student_id = sample.get("student_id", "N/A")
        text = sample.get("text", "No text available")
        stress = sample.get("academic_stress", "N/A")
        home_risk = sample.get("home_support_risk", "N/A")
        subject = sample.get("subject_specific", "None")

        # Convert stress into a coarse level
        try:
            stress_val = float(stress)
            if stress_val > 0.67:
                stress_label = "High"
            elif stress_val > 0.33:
                stress_label = "Medium"
            else:
                stress_label = "Low"
        except Exception:
            stress_label = "N/A"

        # Convert home risk into a coarse level
        try:
            risk_val = float(home_risk)
            if risk_val > 0.5:
                risk_label = "High"
            elif risk_val > 0.25:
                risk_label = "Medium"
            else:
                risk_label = "Low"
        except Exception:
            risk_label = "N/A"

        sample_boxes.append(
            f"""
        <div class="sample-box">
            <strong>Student {escape(str(student_id))}:</strong>
            <p class="sample-text">"{escape(str(text))}"</p>
            <span class="label">Stress: {escape(str(stress_label))}</span>
            <span class="label">Home Risk: {escape(str(risk_label))}</span>
            <span class="label">Subject: {escape(str(subject))}</span>
        </div>
        """
        )

    # NLP distribution plot
    nlp_plot_html = ""
    if nlp_dist_plot:
        nlp_plot_html = f"""
        <div class="divider"></div>
        <h4>Academic Stress Level Distribution</h4>
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{nlp_dist_plot}"
                 alt="NLP Distribution"
                 style="max-width: 800px; width: 450px; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        </div>
        """

    nlp_html = f"""
    <div class="sub-panel" id="engineering_nlp">
        <h3>NLP Features (Simulated)</h3>
        <p class="description">Text-derived psychological and behavioral signals</p>

        <table class="data-table">
            <tr>
                <th>Feature Name</th>
                <th>Description</th>
                <th>Source</th>
                <th>Coverage</th>
                <th>Mean</th>
            </tr>
            {''.join(nlp_rows) if nlp_rows else '<tr><td colspan="5">No NLP features available</td></tr>'}
        </table>

        {nlp_plot_html}

        <div class="divider"></div>
        <h4>Sample Simulated Texts</h4>

        {''.join(sample_boxes) if sample_boxes else '<p class="note">No sample texts available</p>'}

        <div class="alert alert-warning">
            <strong>Note:</strong> These features are simulated for demonstration.
            A production implementation would use actual student text data.
        </div>
    </div>
    """

    return f"""
    <div class="card">
        <h2>Feature Engineering</h2>
        <p class="description">Temporal trajectories and text-derived features</p>

        <div class="select-wrapper">
            <label class="select-label">Select subsection:</label>
            <select id="engineering_select" onchange="showEngSub(this.value)">
                <option value="longitudinal">Longitudinal Features</option>
                <option value="nlp">NLP Features</option>
            </select>
        </div>

        {longitudinal_html}
        {nlp_html}
    </div>
    """


def generate_feature_strategy_correlation(pipeline_data):
    """Generate the Correlation Analysis page (3 heatmap types with dropdown)."""

    # Get correlation heatmaps
    heatmaps = pipeline_data.get("correlation_heatmaps", {})

    # Support multiple key formats
    static_img = heatmaps.get("static", "") or heatmaps.get("type1_static", "")
    static_long_img = heatmaps.get("static_vs_longitudinal", "") or heatmaps.get("type2_static_vs_longitudinal", "")
    long_nlp_img = heatmaps.get("longitudinal_vs_nlp", "") or heatmaps.get("type3_longitudinal_vs_nlp", "")

    return f"""
    <div class="card">
        <h2>Correlation Analysis</h2>
        <p class="description">Examining relationships between different feature types</p>

        <div class="select-wrapper">
            <label class="select-label">Select type:</label>
            <select id="corr_select" onchange="showCorrSub(this.value)">
                <option value="static">Type 1: Static Variables</option>
                <option value="static_long">Type 2: Static vs Longitudinal</option>
                <option value="long_nlp">Type 3: Longitudinal vs NLP</option>
            </select>
        </div>

        <div class="sub-panel active" id="corr_static">
            <h3>Type 1: Static Variables</h3>
            <p class="description">Correlations among demographic and enrollment features</p>
            {f'<img src="data:image/png;base64,{static_img}" style="max-width:900px;"/>' if static_img else '''
            <div class="alert alert-warning">
                <strong>Note:</strong> Correlation heatmap not available in the current pipeline output.
                To display this section, export a base64-encoded PNG in pipeline_data.json under
                <code>correlation_heatmaps["static"]</code>.
            </div>
            '''}
            <div class="alert alert-info">
                <strong>Interpretation:</strong> Static features often show modest correlations.
                Typical patterns include relationships among age, admission grade, and prior qualifications.
            </div>
        </div>

        <div class="sub-panel" id="corr_static_long">
            <h3>Type 2: Static vs Longitudinal</h3>
            <p class="description">How baseline characteristics relate to performance trajectories</p>
            {f'<img src="data:image/png;base64,{static_long_img}" style="max-width:900px;"/>' if static_long_img else '''
            <div class="alert alert-warning">
                <strong>Note:</strong> Correlation heatmap not available.
                Add <code>correlation_heatmaps["static_vs_longitudinal"]</code> to pipeline_data.json to display.
            </div>
            '''}
            <div class="alert alert-info">
                <strong>Interpretation:</strong> Baseline demographic features often have weak-to-moderate
                correlations with longitudinal change features, suggesting that semester-to-semester dynamics
                are not fully explained by starting characteristics alone.
            </div>
        </div>

        <div class="sub-panel" id="corr_long_nlp">
            <h3>Type 3: Longitudinal vs NLP</h3>
            <p class="description">Links between performance metrics and psychological indicators</p>
            {f'<img src="data:image/png;base64,{long_nlp_img}" style="max-width:900px;"/>' if long_nlp_img else '''
            <div class="alert alert-warning">
                <strong>Note:</strong> Correlation heatmap not available.
                Add <code>correlation_heatmaps["longitudinal_vs_nlp"]</code> to pipeline_data.json to display.
            </div>
            '''}
            <div class="alert alert-info">
                <strong>Interpretation:</strong> Stress-related features may be negatively related to credit ratios,
                while home-support risk can align with indicators of academic decline, depending on how the simulated
                signals were generated.
            </div>
        </div>
    </div>
    """


def generate_feature_strategy_importance(pipeline_data):
    """Generate the Feature Importance page (L1 + RF + Combined with dropdown)."""

    # Get feature importance plots
    importance_plots = pipeline_data.get("feature_importance_plots", {})
    l1_plot = importance_plots.get("l1_logistic", "")
    rf_plot = importance_plots.get("random_forest", "")
    combined_plot = importance_plots.get("combined", "")

    # Get importance tables
    l1_importance_data = pipeline_data.get("l1_importance", [])
    rf_importance_data = pipeline_data.get("rf_importance", [])
    combined_data = pipeline_data.get("combined_importance", [])

    # Convert to DataFrame (top 10)
    l1_importance = pd.DataFrame(l1_importance_data[:10]) if l1_importance_data else pd.DataFrame()
    rf_importance = pd.DataFrame(rf_importance_data[:10]) if rf_importance_data else pd.DataFrame()
    combined = pd.DataFrame(combined_data[:10]) if combined_data else pd.DataFrame()

    # Add rank columns
    if not l1_importance.empty:
        l1_importance.insert(0, "Rank", range(1, len(l1_importance) + 1))
    if not rf_importance.empty:
        rf_importance.insert(0, "Rank", range(1, len(rf_importance) + 1))

    return f"""
    <div class="card">
        <h2>Feature Importance</h2>
        <p class="description">Identifying predictive features using multiple methods</p>

        <div class="select-wrapper">
            <label class="select-label">Select method:</label>
            <select id="importance_select" onchange="showImpSub(this.value)">
                <option value="l1">L1 Logistic Regression</option>
                <option value="rf">Random Forest</option>
                <option value="combined">Combined Ranking</option>
            </select>
        </div>

        <div class="sub-panel active" id="importance_l1">
            <h3>L1 Logistic Regression</h3>
            <p class="description">Sparse feature selection through L1 regularization</p>
            {f'<img src="data:image/png;base64,{l1_plot}" style="max-width:900px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"/>' if l1_plot else '''
            <div class="alert alert-warning">
                <strong>Note:</strong> Feature importance plot not available.
                To display the L1 feature importance bar chart, add <code>feature_importance_plots["l1_logistic"]</code>
                (base64-encoded PNG) to pipeline_data.json.
            </div>
            '''}
            {df_to_html(l1_importance) if not l1_importance.empty else '<p class="note">No L1 importance data available.</p>'}
            <div class="alert alert-info">
                <strong>Interpretation:</strong> L1 regularization performs feature selection by shrinking
                less informative coefficients toward zero. Features with non-zero coefficients carry the strongest signal.
            </div>
        </div>

        <div class="sub-panel" id="importance_rf">
            <h3>Random Forest</h3>
            <p class="description">Feature importance based on impurity reduction</p>
            {f'<img src="data:image/png;base64,{rf_plot}" style="max-width:900px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"/>' if rf_plot else '''
            <div class="alert alert-warning">
                <strong>Note:</strong> Feature importance plot not available.
                To display the Random Forest feature importance bar chart, add <code>feature_importance_plots["random_forest"]</code>
                (base64-encoded PNG) to pipeline_data.json.
            </div>
            '''}
            {df_to_html(rf_importance) if not rf_importance.empty else '<p class="note">No Random Forest importance data available.</p>'}
            <div class="alert alert-info">
                <strong>Interpretation:</strong> Random Forest importance reflects how much each feature reduces
                prediction uncertainty across splits. This method can capture non-linear relationships.
            </div>
        </div>

        <div class="sub-panel" id="importance_combined">
            <h3>Combined Ranking</h3>
            <p class="description">Consensus ranking across L1 and Random Forest methods</p>
            {f'<img src="data:image/png;base64,{combined_plot}" style="max-width:900px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"/>' if combined_plot else '''
            <div class="alert alert-warning">
                <strong>Note:</strong> Combined ranking plot not available.
                To display the combined ranking chart, add <code>feature_importance_plots["combined"]</code>
                (base64-encoded PNG) to pipeline_data.json.
            </div>
            '''}
            {df_to_html(combined) if not combined.empty else '<p class="note">No combined importance data available.</p>'}
            <div class="alert alert-success">
                <strong>Summary:</strong> Features that rank highly across both methods are typically the most stable
                candidates for downstream decision support and intervention design.
            </div>
        </div>
    </div>
    """


# ============================================================================
# MODEL RIGOR & EQUITY
# ============================================================================

def generate_model_overview(modeling_data):
    """Generate the Model Overview page (includes class imbalance and best model selection)."""

    data_summary = modeling_data.get("data_summary", {})

    overview_html = f"""
    <div class="card">
        <h2>Model Overview</h2>
        <p class="description">Dataset summary and modeling approach</p>

        <h3>Dataset Statistics</h3>
        <div class="metric-grid">
            <div class="metric">
                <div class="label">Total Samples</div>
                <div class="value">{data_summary.get('total_samples', 0)}</div>
            </div>
            <div class="metric">
                <div class="label">Training Set</div>
                <div class="value">{data_summary.get('train_samples', 0)}</div>
            </div>
            <div class="metric">
                <div class="label">Test Set</div>
                <div class="value">{data_summary.get('test_samples', 0)}</div>
            </div>
            <div class="metric">
                <div class="label">Features</div>
                <div class="value">{data_summary.get('n_features', 0)}</div>
            </div>
        </div>

        <h3>Target Distribution</h3>
        <table style="max-width: 600px;">
            <tr>
                <th>Outcome</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
    """

    target_dist = data_summary.get("target_distribution", {})
    total = data_summary.get("total_samples", 1)
    for label, count in target_dist.items():
        pct = (count / total * 100) if total else 0
        overview_html += f"""
            <tr>
                <td><strong>{escape(str(label))}</strong></td>
                <td>{count}</td>
                <td>{pct:.1f}%</td>
            </tr>
        """

    overview_html += """
        </table>

        <div class="alert alert-info">
            <strong>Modeling Strategy:</strong> Multi-class classification with balanced class weights to address class imbalance.
        </div>
    </div>
    """

    # Class imbalance analysis
    if target_dist:
        counts = list(target_dist.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = (max_count / min_count) if min_count > 0 else 0

        max_class = max(target_dist, key=target_dist.get)
        min_class = min(target_dist, key=target_dist.get)

        if imbalance_ratio < 1.5:
            imbalance_level = "Minimal Imbalance"
            imbalance_color = "#27ae60"
        elif imbalance_ratio < 3:
            imbalance_level = "Light Imbalance"
            imbalance_color = "#f39c12"
        elif imbalance_ratio < 5:
            imbalance_level = "Moderate Imbalance"
            imbalance_color = "#e67e22"
        else:
            imbalance_level = "Severe Imbalance"
            imbalance_color = "#e74c3c"

        overview_html += f"""
        <div class="card" style="margin-top: 30px;">
            <h2 style="color: {imbalance_color};">Class Imbalance Analysis</h2>

            <div class="alert alert-info" style="background-color: {imbalance_color}15; border-left-color: {imbalance_color};">
                <strong>{imbalance_level}</strong> - Class ratio: {imbalance_ratio:.2f}:1
                (Largest class: {escape(str(max_class))} = {max_count:,}, Smallest class: {escape(str(min_class))} = {min_count:,})
            </div>

            <h3>Handling Strategy</h3>
            <table style="max-width: 100%; margin: 20px 0;">
                <tr>
                    <th style="width: 30%;">Strategy</th>
                    <th>Rationale</th>
                </tr>
                <tr>
                    <td><strong>class_weight_only</strong></td>
                    <td>
                        With a class ratio of {imbalance_ratio:.2f}:1, a conservative <code>class_weight="balanced"</code>
                        approach is a reasonable baseline. This adjusts the loss contribution by class without modifying
                        the observed data distribution.
                    </td>
                </tr>
            </table>
        </div>
        """

    # Best model selection
    model_results = modeling_data.get("model_results", {})

    best_model_name = None
    best_roc_auc = -1
    best_model_data = {}

    for model_name, model_data in model_results.items():
        roc_auc = model_data.get("roc_auc_macro", None)
        if roc_auc is None:
            continue
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = model_data.get("model_name", model_name)
            best_model_data = model_data

    if best_model_name is not None and best_roc_auc >= 0:
        overview_html += f"""
        <div class="card" style="margin-top: 30px;">
            <h2>Best Model Selection</h2>

            <div class="metric-grid" style="grid-template-columns: repeat(2, 1fr); margin: 20px 0;">
                <div class="metric">
                    <div class="label">Model</div>
                    <div class="value" style="font-size: 1.5em;">{escape(str(best_model_name))}</div>
                </div>
                <div class="metric">
                    <div class="label">ROC-AUC (Macro)</div>
                    <div class="value" style="font-size: 1.8em; color: #27ae60;">{best_roc_auc:.3f}</div>
                </div>
            </div>

            <table style="max-width: 100%; margin: 20px 0;">
                <tr>
                    <th style="width: 30%;">Attribute</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td><strong>Purpose</strong></td>
                    <td>{escape(str(best_model_data.get('purpose', 'N/A')))}</td>
                </tr>
                <tr>
                    <td><strong>Regularization Strategy</strong></td>
                    <td>{escape(str(best_model_data.get('regularization', 'N/A')))}</td>
                </tr>
            </table>

            <div class="alert alert-success">
                <strong>Selection Criteria:</strong> Best model selected based on macro-averaged ROC-AUC and F1-score,
                balancing performance across the three outcome classes.
            </div>
        </div>
        """

    return overview_html


def generate_model_comparison(modeling_data):
    """Generate the Model Comparison page."""

    model_results = modeling_data.get("model_results", {})

    comparison_rows = []
    for model_name, model_data in model_results.items():
        if "classification_report" in model_data:
            report = model_data["classification_report"]
            comparison_rows.append(
                f"""
            <tr>
                <td><strong>{escape(str(model_data.get('model_name', model_name)))}</strong></td>
                <td>{escape(str(model_data.get('purpose', 'N/A')))}</td>
                <td>{report['macro avg']['precision']:.3f}</td>
                <td>{report['macro avg']['recall']:.3f}</td>
                <td>{report['macro avg']['f1-score']:.3f}</td>
                <td>{model_data.get('roc_auc_macro', 0):.3f}</td>
            </tr>
            """
            )

    detailed_sections = []
    for model_name, model_data in list(model_results.items())[:3]:
        if "confusion_matrix" not in model_data:
            continue

        cm = model_data["confusion_matrix"]

        cm_html = f"""
        <table style="max-width: 500px; margin: 20px 0;">
            <thead>
                <tr>
                    <th></th>
                    <th>Pred: Graduate</th>
                    <th>Pred: Dropout</th>
                    <th>Pred: Enrolled</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>True: Graduate</strong></td>
                    <td>{cm[0][0] if len(cm) > 0 else 0}</td>
                    <td>{cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0}</td>
                    <td>{cm[0][2] if len(cm) > 0 and len(cm[0]) > 2 else 0}</td>
                </tr>
                <tr>
                    <td><strong>True: Dropout</strong></td>
                    <td>{cm[1][0] if len(cm) > 1 else 0}</td>
                    <td>{cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0}</td>
                    <td>{cm[1][2] if len(cm) > 1 and len(cm[1]) > 2 else 0}</td>
                </tr>
                <tr>
                    <td><strong>True: Enrolled</strong></td>
                    <td>{cm[2][0] if len(cm) > 2 else 0}</td>
                    <td>{cm[2][1] if len(cm) > 2 and len(cm[2]) > 1 else 0}</td>
                    <td>{cm[2][2] if len(cm) > 2 and len(cm[2]) > 2 else 0}</td>
                </tr>
            </tbody>
        </table>
        """

        feature_importance = model_data.get("feature_importance", {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        feature_rows = "".join(
            [
                f"""
        <tr>
            <td>{rank}</td>
            <td>{escape(str(feat))}</td>
            <td>{importance:.4f}</td>
        </tr>
        """
                for rank, (feat, importance) in enumerate(sorted_features, 1)
            ]
        )

        detailed_sections.append(
            f"""
        <div class="card">
            <h3>{escape(str(model_data.get('model_name', model_name)))}</h3>

            <div class="alert alert-info">
                <strong>Purpose:</strong> {escape(str(model_data.get('purpose', 'N/A')))}<br>
                <strong>Regularization:</strong> {escape(str(model_data.get('regularization', 'N/A')))}
            </div>

            <h4>Confusion Matrix</h4>
            {cm_html}

            <h4>Top 10 Feature Importance</h4>
            <table style="max-width: 600px;">
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
                {feature_rows if feature_rows else '<tr><td colspan="3">No feature importance available</td></tr>'}
            </table>
        </div>
        """
        )

    comparison_html = f"""
    <div class="card">
        <h2>Model Performance Comparison</h2>

        <table>
            <tr>
                <th>Model</th>
                <th>Purpose</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
            </tr>
            {''.join(comparison_rows) if comparison_rows else '<tr><td colspan="6">No model results available</td></tr>'}
        </table>

        <div class="alert alert-info">
            <strong>Model Selection:</strong> Best model selected based on macro-averaged F1-score and ROC-AUC.
        </div>
    </div>

    <h2 style="color: #2c3e50; margin: 30px 0 20px 0;">Detailed Model Information</h2>
    {''.join(detailed_sections)}
    """

    return comparison_html


def generate_model_fairness(modeling_data):
    """Generate the Fairness Audit page."""

    fairness_audit = modeling_data.get("fairness_audit", {})

    if not fairness_audit or len(fairness_audit) <= 1:
        return """
        <div class="card">
            <h2>Fairness Audit</h2>
            <div class="alert alert-warning">
                No fairness audit data available.
            </div>
        </div>
        """

    overall = fairness_audit.get("_overall_summary", {})

    feature_details = []
    for feature_name, audit_data in fairness_audit.items():
        if feature_name == "_overall_summary":
            continue

        di_result = audit_data.get("disparate_impact", {})
        eo_result = audit_data.get("equalized_odds", {})

        di_level = di_result.get("fairness_level", "N/A")
        eo_level = eo_result.get("fairness_level", "N/A")
        di_badge = "success" if di_level == "Fair" else "warning"
        eo_badge = "success" if eo_level == "Fair" else "warning"

        fpr_val = eo_result.get("fpr_difference", 0)
        fpr_label = "Fair" if fpr_val < 0.1 else "Review"
        fpr_badge = "success" if fpr_val < 0.1 else "warning"

        feature_details.append(
            f"""
        <div class="card">
            <h3>Feature: {escape(str(feature_name))}</h3>

            <table style="max-width: 700px;">
                <tr>
                    <th>Metric</th>
                    <th>Result</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>Disparate Impact</td>
                    <td>{di_result.get('disparate_impact', 0):.3f}</td>
                    <td><span class="badge badge-{di_badge}">{escape(str(di_level))}</span></td>
                </tr>
                <tr>
                    <td>TPR Difference</td>
                    <td>{eo_result.get('tpr_difference', 0):.3f}</td>
                    <td><span class="badge badge-{eo_badge}">{escape(str(eo_level))}</span></td>
                </tr>
                <tr>
                    <td>FPR Difference</td>
                    <td>{fpr_val:.3f}</td>
                    <td><span class="badge badge-{fpr_badge}">{fpr_label}</span></td>
                </tr>
            </table>
        </div>
        """
        )

    overall_status = overall.get("overall_fairness", "N/A")
    overall_text = "Pass" if overall_status == "Pass" else "Review"

    fairness_html = f"""
    <div class="card">
        <h2>Fairness Audit Overview</h2>

        <div class="metric-grid">
            <div class="metric">
                <div class="label">Features Audited</div>
                <div class="value">{overall.get('total_features_audited', 0)}</div>
            </div>
            <div class="metric">
                <div class="label">DI Pass Rate</div>
                <div class="value">{overall.get('di_fairness_rate', 0) * 100:.0f}%</div>
            </div>
            <div class="metric">
                <div class="label">EO Pass Rate</div>
                <div class="value">{overall.get('eo_fairness_rate', 0) * 100:.0f}%</div>
            </div>
            <div class="metric">
                <div class="label">Overall Status</div>
                <div class="value" style="font-size: 1.5em;">{overall_text}</div>
            </div>
        </div>

        <div class="alert alert-info">
            <strong>Fairness Criteria:</strong> Disparate Impact &ge; 0.8 and Equalized Odds differences &lt; 0.1.
        </div>
    </div>

    {''.join(feature_details)}
    """

    return fairness_html
