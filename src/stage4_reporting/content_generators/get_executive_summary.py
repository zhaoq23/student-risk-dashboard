"""
get_executive_summary.py
Adapted for OLD radar_analysis data format
"""

from html import escape
import pandas as pd
import json

# Try to import NLP module (optional)
try:
    import sys
    from pathlib import Path
    # Add content_generators to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from nlp_simulations import (
        generate_nlp_analysis_card,
        generate_sentiment_timeline_card
    )
    NLP_AVAILABLE = True
    print("✓ NLP simulations module loaded successfully")
except ImportError as e:
    print(f"Warning: NLP simulations module not available: {e}")
    NLP_AVAILABLE = False


# Utility functions
def records_to_df(records):
    """Convert records to DataFrame"""
    if not records:
        return pd.DataFrame()
    if isinstance(records, dict) and 'data' in records:
        records = records['data']
    return pd.DataFrame(records)


def df_to_html_table(df, max_rows=None):
    """DataFrame to HTML table"""
    if df is None or df.empty:
        return '<p class="note">No data available.</p>'
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, border=0, escape=False, classes="data-table")


def render_dtype_blocks(dtype_counts_records, dtype_cols_dict):
    """Render dtype summary"""
    if isinstance(dtype_counts_records, dict):
        dtype_counts_records = dtype_counts_records.get('data', [])
    
    dtype_counts = records_to_df(dtype_counts_records)
    
    parts = []
    parts.append("<h4>Column type summary</h4>")
    if dtype_counts.empty:
        parts.append('<p class="note">No dtype summary available.</p>')
    else:
        parts.append(df_to_html_table(dtype_counts))
    
    parts.append('<p class="note">Details (click to expand):</p>')
    if isinstance(dtype_cols_dict, dict) and dtype_cols_dict:
        for dt, cols in dtype_cols_dict.items():
            cols = cols or []
            safe_cols = ", ".join([escape(str(c)) for c in cols[:50]])
            tail = "" if len(cols) <= 50 else f"<br/><span class='note'>(+{len(cols)-50} more)</span>"
            parts.append(f"""
            <details style="margin:8px 0;">
              <summary><strong>{escape(str(dt))}</strong> — {len(cols)} columns</summary>
              <div class="note" style="margin-top:8px; padding-left:16px;">{safe_cols}{tail}</div>
            </details>
            """)
    
    return "".join(parts)


def generate_student_level_actions(actions):
    """Generate Student-Level Actions section"""
    student_level = actions.get('student_level', {})
    
    critical_students = student_level.get('critical', [])
    warning_students = student_level.get('warning', [])
    on_track_students = student_level.get('on_track', [])
    
    # Build table rows
    critical_rows = []
    for s in critical_students[:10]:
        top_factor = s.get('top_risk_factors', [{}])[0] if s.get('top_risk_factors') else {}
        critical_rows.append(f"""
        <tr>
            <td><strong>#{s.get('student_id', 'N/A')}</strong></td>
            <td><span class="badge badge-critical">Critical</span></td>
            <td class="risk-high">{s.get('dropout_risk_score', 0):.2%}</td>
            <td>{escape(str(top_factor.get('feature', 'N/A')))}</td>
            <td>{escape(str(s.get('overall_recommendation', 'N/A')))}</td>
        </tr>
        """)
    
    warning_rows = []
    for s in warning_students[:10]:
        top_factor = s.get('top_risk_factors', [{}])[0] if s.get('top_risk_factors') else {}
        warning_rows.append(f"""
        <tr>
            <td><strong>#{s.get('student_id', 'N/A')}</strong></td>
            <td><span class="badge badge-warning">Warning</span></td>
            <td class="risk-medium">{s.get('dropout_risk_score', 0):.2%}</td>
            <td>{escape(str(top_factor.get('feature', 'N/A')))}</td>
            <td>{escape(str(s.get('overall_recommendation', 'N/A')))}</td>
        </tr>
        """)
    
    on_track_rows = []
    for s in on_track_students[:10]:
        top_factor = s.get('top_risk_factors', [{}])[0] if s.get('top_risk_factors') else {}
        on_track_rows.append(f"""
        <tr>
            <td><strong>#{s.get('student_id', 'N/A')}</strong></td>
            <td><span class="badge badge-success">On Track</span></td>
            <td class="risk-low">{s.get('dropout_risk_score', 0):.2%}</td>
            <td>{escape(str(top_factor.get('feature', 'N/A')))}</td>
            <td>{escape(str(s.get('overall_recommendation', 'N/A')))}</td>
        </tr>
        """)
    
    # What-If Analysis
    whatif_html = ""
    whatif_sample = student_level.get('whatif_sample')
    if whatif_sample:
        scenarios = whatif_sample.get('scenarios', [])
        best_scenario = max(scenarios, key=lambda x: x.get('risk_reduction', 0)) if scenarios else {}
        
        whatif_html = f"""
        <div class="card">
            <h2>What-If Analysis - Intervention Simulation</h2>
            <p><strong>Sample Student:</strong> #{whatif_sample.get('student_id', 'N/A')} - Baseline Dropout Risk: {whatif_sample.get('baseline_risk', 0):.2%}</p>
            
            <table class="whatif-table">
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>Description</th>
                        <th>New Risk</th>
                        <th>Risk Reduction</th>
                        <th>Reduction %</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="baseline">
                        <td><strong>Baseline</strong></td>
                        <td>No intervention</td>
                        <td class="risk-high">{whatif_sample.get('baseline_risk', 0):.2%}</td>
                        <td>—</td>
                        <td>—</td>
                    </tr>
                    {''.join([f'''
                    <tr class="scenario">
                        <td><strong>{escape(str(sc.get('name', 'N/A')))}</strong></td>
                        <td>{escape(str(sc.get('description', 'N/A')))}</td>
                        <td class="{'risk-high' if sc.get('new_risk', 0) > 0.7 else 'risk-medium' if sc.get('new_risk', 0) > 0.4 else 'risk-low'}">{sc.get('new_risk', 0):.2%}</td>
                        <td>{sc.get('risk_reduction', 0):.2%}</td>
                        <td>{sc.get('reduction_pct', 0):.1f}%</td>
                    </tr>
                    ''' for sc in scenarios])}
                </tbody>
            </table>
            
            <div class="alert alert-success">
                <strong>Key Finding:</strong> The most effective intervention is 
                <strong>{escape(str(best_scenario.get('name', 'N/A')))}</strong>, 
                reducing dropout risk by {best_scenario.get('risk_reduction', 0):.2%}.
            </div>
        </div>
        """
    
    return f"""
    <div class="card">
        <h2>Student-Level Actions - Personalized Interventions</h2>
        
        <div class="select-wrapper">
            <label style="font-weight: bold; margin-right: 10px;">Select Priority Group:</label>
            <select onchange="showPriorityGroup(this.value)">
                <option value="critical">🔴 Critical (High Risk > 0.7)</option>
                <option value="warning">🟡 Warning (Medium Risk 0.4 - 0.7)</option>
                <option value="on-track">🟢 On Track (< 0.4)</option>
            </select>
        </div>
        
        <div class="sub-panel active" id="priority-critical">
            <h3>🔴 Critical Priority Students (Top 10)</h3>
            <table>
                <tr>
                    <th>Student ID</th>
                    <th>Priority</th>
                    <th>Risk Score</th>
                    <th>Top Risk Factor</th>
                    <th>Recommended Action</th>
                </tr>
                {''.join(critical_rows) if critical_rows else '<tr><td colspan="5">No critical priority students</td></tr>'}
            </table>
        </div>
        
        <div class="sub-panel" id="priority-warning">
            <h3>🟡 Warning Priority Students (Top 10)</h3>
            <table>
                <tr>
                    <th>Student ID</th>
                    <th>Priority</th>
                    <th>Risk Score</th>
                    <th>Top Risk Factor</th>
                    <th>Recommended Action</th>
                </tr>
                {''.join(warning_rows) if warning_rows else '<tr><td colspan="5">No warning priority students</td></tr>'}
            </table>
        </div>
        
        <div class="sub-panel" id="priority-on-track">
            <h3>🟢 On Track Students (Top 10)</h3>
            <table>
                <tr>
                    <th>Student ID</th>
                    <th>Priority</th>
                    <th>Risk Score</th>
                    <th>Top Risk Factor</th>
                    <th>Recommended Action</th>
                </tr>
                {''.join(on_track_rows) if on_track_rows else '<tr><td colspan="5">No on-track students</td></tr>'}
            </table>
        </div>
    </div>
    
    {whatif_html}
    """


def generate_cohort_comparison(actions):
    """Generate Cohort Comparison section - ADAPTED FOR OLD radar_analysis FORMAT"""
    cohort_level = actions.get('cohort_level', {})
    
    # Part 1: Course Risk Analysis Table
    course_risk = cohort_level.get('course_risk', [])
    course_table = f"""
    <div class="card">
        <h2>Cohort-Level Actions - Resource Allocation</h2>
        
        <h3>Course Risk Analysis</h3>
        <p style="margin-bottom: 15px;">
            Courses ranked by <strong>average dropout risk</strong> of enrolled students.
        </p>
        <table>
            <tr>
                <th>Rank</th>
                <th>Course Code</th>
                <th>Course Name</th>
                <th>Students</th>
                <th>Avg Risk</th>
                <th>Priority</th>
                <th>Recommended Action</th>
            </tr>
            {''.join([f'''<tr>
                <td><strong>#{i+1}</strong></td>
                <td>{c.get('cohort_id', 'N/A')}</td>
                <td><strong>{escape(str(c.get('cohort_name', 'Unknown')))}</strong></td>
                <td>{c.get('n_students', 'N/A')}</td>
                <td class="{'risk-high' if c.get('avg_dropout_risk', 0) > 0.6 else 'risk-medium' if c.get('avg_dropout_risk', 0) > 0.4 else 'risk-low'}">{c.get('avg_dropout_risk', 0):.2%}</td>
                <td><span class="badge badge-{'critical' if c.get('priority') == 'Critical' else 'warning' if c.get('priority') in ['High', 'Medium'] else 'success'}">{c.get('priority', 'N/A')}</span></td>
                <td>{escape(str(c.get('recommended_action', 'N/A')))}</td>
            </tr>''' for i, c in enumerate(course_risk[:10])]) if course_risk else '<tr><td colspan="7">No course data available</td></tr>'}
        </table>
        
        <div class="alert alert-info">
            <strong>Methodology:</strong> Average dropout risk calculated as mean predicted probability across all students.
        </div>
    </div>
    """
    
    # Part 2: Radar Charts - USING OLD FORMAT
    radar_analysis = cohort_level.get('radar_analysis', {})
    radar_html = generate_radar_charts_OLD_FORMAT(radar_analysis)
    
    # Part 3: NLP Entity-Driven Feedback (if available)
    nlp_card = ""
    if NLP_AVAILABLE:
        try:
            nlp_card = generate_nlp_analysis_card()
        except Exception as e:
            print(f"Warning: Failed to generate NLP card: {e}")
            nlp_card = ""
    
    return course_table + radar_html + nlp_card


def generate_radar_charts_OLD_FORMAT(radar_analysis):
    """Generate radar charts using OLD radar_analysis format"""
    
    if not radar_analysis or not radar_analysis.get('available_dimensions'):
        return """
        <div class="card">
            <h2>Cohort Comparison - Radar Chart Analysis</h2>
            <div class="alert alert-warning">No cohort comparison data available.</div>
        </div>
        """
    
    available_dims = radar_analysis['available_dimensions']
    default_dim = radar_analysis.get('default_dimension', available_dims[0]['dimension_id'] if available_dims else None)
    
    select_options = ''.join([
        f'<option value="{dim["dimension_id"]}"{"selected" if dim["dimension_id"] == default_dim else ""}>{escape(str(dim.get("dimension_name", "Dimension " + str(dim["dimension_id"]))))}</option>'
        for dim in available_dims
    ])
    
    dimension_panels = []
    chart_scripts = []
    
    for dim in available_dims:
        dim_id = dim['dimension_id']
        radar_data = dim.get('radar_data', {})
        
        # Extract data from OLD format
        dimensions = radar_data.get('dimensions', [])
        group_0_vals = radar_data.get('group_0_values', [])
        group_1_vals = radar_data.get('group_1_values', [])
        differences = radar_data.get('differences', [])
        interp = radar_data.get('interpretation', {})
        
        group_0_label = dim.get('group_0_label', 'Group 0')
        group_1_label = dim.get('group_1_label', 'Group 1')
        
        # Build comparison table
        comparison_rows = ''.join([
            f"""
            <tr>
                <td><strong>{escape(str(dimensions[i]))}</strong></td>
                <td>{group_1_vals[i]:.2f}</td>
                <td>{group_0_vals[i]:.2f}</td>
                <td style="color: {'green' if differences[i] > 0 else 'red' if differences[i] < 0 else 'gray'}; font-weight: bold;">
                    {'+' if differences[i] > 0 else ''}{differences[i]:.2f}
                </td>
            </tr>
            """
            for i in range(len(dimensions))
        ])
        
        canvas_id = f"radarCanvas_{dim_id}"
        stronger_label = group_1_label if interp.get('stronger_group') == 'group_1' else group_0_label
        
        dimension_panels.append(f"""
        <div class="radar-panel {'active' if dim_id == default_dim else ''}" id="radar-{dim_id}">
            <h3>{escape(str(group_1_label))} vs {escape(str(group_0_label))} (5 Dimensions)</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 20px 0; align-items: center;">
                <div style="position: relative; height: 400px;">
                    <canvas id="{canvas_id}"></canvas>
                </div>
                <div>
                    <table>
                        <tr>
                            <th>Dimension</th>
                            <th>{escape(str(group_1_label))}</th>
                            <th>{escape(str(group_0_label))}</th>
                            <th>Difference</th>
                        </tr>
                        {comparison_rows}
                    </table>
                </div>
            </div>
            
            <div class="alert alert-info">
                <strong>Interpretation:</strong> Students <strong>{escape(str(stronger_label))}</strong> show stronger {escape(str(interp.get('max_diff_dimension', 'N/A')))} (Δ={interp.get('max_diff_value', 0):.2f})
            </div>
        </div>
        """)
        
        # Generate Chart.js script
        chart_scripts.append(f"""
            var ctx_{dim_id} = document.getElementById('{canvas_id}');
            if (ctx_{dim_id}) {{
                new Chart(ctx_{dim_id}, {{
                    type: 'radar',
                    data: {{
                        labels: {json.dumps(dimensions)},
                        datasets: [
                            {{
                                label: '{escape(str(group_1_label))}',
                                data: {group_1_vals},
                                borderColor: 'rgba(102, 126, 234, 1)',
                                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                                borderWidth: 2,
                                pointRadius: 4
                            }},
                            {{
                                label: '{escape(str(group_0_label))}',
                                data: {group_0_vals},
                                borderColor: 'rgba(118, 75, 162, 1)',
                                backgroundColor: 'rgba(118, 75, 162, 0.2)',
                                borderWidth: 2,
                                pointRadius: 4
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        scales: {{
                            r: {{
                                beginAtZero: true,
                                max: 1,
                                min: 0,
                                ticks: {{ stepSize: 0.2 }}
                            }}
                        }},
                        plugins: {{ legend: {{ position: 'bottom' }} }}
                    }}
                }});
            }}
        """)
    
    all_chart_scripts = ''.join(chart_scripts)
    
    return f"""
    <div class="card">
        <h2>Cohort Comparison - Radar Chart Analysis</h2>
        
        <div class="select-wrapper">
            <label style="font-weight: bold; margin-right: 10px;">Select Comparison Dimension:</label>
            <select onchange="showRadarDimension(this.value)">
                {select_options}
            </select>
        </div>
        
        {''.join(dimension_panels)}
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            if (typeof Chart !== 'undefined') {{
                {all_chart_scripts}
            }}
        }});
    </script>
    """


def generate_system_level_actions(actions):
    """Generate System-Level Actions section"""
    system_level = actions.get('system_level', {})
    
    # Policy-level features
    global_factors = system_level.get('policy_level_features', []) or system_level.get('global_risk_factors', []) or []
    
    max_shap = max([f.get('mean_abs_shap', 0) for f in global_factors[:10]], default=1) if global_factors else 1
    
    global_rows = ''.join([f"""
    <tr>
        <td>{f.get('rank', i+1)}</td>
        <td>{escape(str(f.get('feature', 'N/A')))}</td>
        <td style="position: relative;">
            <div style="display: flex; align-items: center;">
                <div style="position: absolute; left: 0; top: 50%; transform: translateY(-50%); 
                            width: {(f.get('mean_abs_shap', 0) / max_shap * 100):.1f}%; 
                            height: 70%; 
                            background: linear-gradient(90deg, rgba(102, 126, 234, 0.2), rgba(102, 126, 234, 0.4)); 
                            border-radius: 3px; 
                            z-index: 0;">
                </div>
                <span style="position: relative; z-index: 1; padding-left: 8px; font-weight: 600;">
                    {f.get('mean_abs_shap', 0):.4f}
                </span>
            </div>
        </td>
        <td>{escape(str(f.get('policy_action', 'No action specified')))}</td>
    </tr>
    """ for i, f in enumerate(global_factors[:10])])
    
    policy_html = f"""
    <div class="card">
        <h2>System-Level Actions - Policy Interventions</h2>
        
        <h3>Top 10 Global Risk Factors</h3>
        <p style="margin-bottom: 15px;">
            Features with the <strong>highest mean absolute SHAP values</strong> across all students.
        </p>
        <table>
            <tr>
                <th>Rank</th>
                <th>Feature</th>
                <th>Mean |SHAP|</th>
                <th>Policy Action</th>
            </tr>
            {global_rows if global_rows else '<tr><td colspan="4">No policy features available</td></tr>'}
        </table>
        
        <div class="alert alert-success">
            <strong>Interpretation:</strong> These features represent institution-wide patterns that require systemic interventions.
        </div>
    </div>
    """
    
    # NLP Sentiment Timeline (if available)
    sentiment_card = ""
    if NLP_AVAILABLE:
        try:
            sentiment_card = generate_sentiment_timeline_card()
        except Exception as e:
            print(f"Warning: Failed to generate sentiment timeline: {e}")
            sentiment_card = ""
    
    # Cost-Benefit Analysis
    cba = system_level.get('cost_benefit_analysis', {})
    cba_html = ""
    
    if cba:
        scenarios = cba.get('scenarios', [])
        scenario_cards = ''.join([f"""
        <div class="cost-card">
            <h4>{escape(str(sc.get('name', 'N/A')))}</h4>
            <div class="cost-metric"><strong>Description:</strong> {escape(str(sc.get('description', 'N/A')))}</div>
            <div class="cost-metric"><strong>Beneficiaries:</strong> {sc.get('beneficiaries', 0)} students</div>
            <div class="cost-metric"><strong>Total Cost:</strong> ${sc.get('total_cost', 0):,}</div>
            <div class="cost-metric"><strong>Dropouts Prevented:</strong> {sc.get('dropouts_prevented', 0)}</div>
            <div class="cost-metric"><strong>Cost per Student Saved:</strong> ${sc.get('cost_per_student_saved', 0):,.0f}</div>
            <div class="cost-metric"><strong>New Dropout Rate:</strong> {sc.get('new_dropout_rate', 0):.2%}</div>
        </div>
        """ for sc in scenarios])
        
        cba_html = f"""
        <div class="card">
            <h2>Policy Simulation - Cost-Benefit Analysis</h2>
            
            <h3>Baseline Metrics</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="label">Total Students</div>
                    <div class="value">{cba.get('total_students', 0)}</div>
                </div>
                <div class="metric">
                    <div class="label">High-Risk Students</div>
                    <div class="value">{cba.get('total_high_risk_students', 0)}</div>
                </div>
                <div class="metric">
                    <div class="label">Baseline Dropout Rate</div>
                    <div class="value">{cba.get('baseline_dropout_rate', 0):.1%}</div>
                </div>
            </div>
            
            <h3>Intervention Scenarios</h3>
            <div class="cost-comparison">
                {scenario_cards}
            </div>
            
            <div class="alert alert-success">
                <strong>Recommendation:</strong> Prioritize interventions with lowest cost per student saved.
            </div>
        </div>
        """
    
    return policy_html + sentiment_card + cba_html
