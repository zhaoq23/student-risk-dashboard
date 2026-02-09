"""
get_data_governance.py
"""

from html import escape
import pandas as pd


def records_to_df(records):
    """Convert records to a DataFrame."""
    if not records:
        return pd.DataFrame()
    if isinstance(records, dict) and "data" in records:
        records = records["data"]
    return pd.DataFrame(records)


def df_to_html_table(df, max_rows=None):
    """Render a DataFrame as an HTML table."""
    if df is None or df.empty:
        return '<p class="note">No data available.</p>'
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, border=0, escape=False, classes="data-table")


def render_dtype_blocks(dtype_counts_records, dtype_cols_dict):
    """Render dtype summary blocks."""
    if isinstance(dtype_counts_records, dict):
        dtype_counts_records = dtype_counts_records.get("data", [])

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
            parts.append(
                f"""
            <details style="margin:8px 0;">
              <summary><strong>{escape(str(dt))}</strong> — {len(cols)} columns</summary>
              <div class="note" style="margin-top:8px; padding-left:16px;">{safe_cols}{tail}</div>
            </details>
            """
            )

    return "".join(parts)


def generate_data_governance_intake(pipeline_data):
    """Generate the Data Governance - Intake & Quality page"""

    panels = pipeline_data.get("panels", {})
    quality_reports = pipeline_data.get("quality_reports", {})

    # Collect available intake batches
    intake_keys = [k for k in ["Overall", "Week 1", "Week 2", "Week 3"] if k in panels]

    if not intake_keys:
        return """
        <div class="card">
            <h2>Intake & Quality Checks</h2>
            <p class="note">No intake data available.</p>
        </div>
        """

    # Build select options
    select_options = "".join([f'<option value="{escape(k)}">{escape(k)}</option>' for k in intake_keys])

    # Build panels for each batch
    batch_panels = []
    for k in intake_keys:
        p = panels[k]

        panel_html = [f'<div class="sub-panel" id="intake_{escape(k)}">']

        # Basic checks
        panel_html.append("<h3>Basic checks</h3>")
        basic_data = p.get("basic", {})
        basic_df = records_to_df(basic_data)
        panel_html.append(df_to_html_table(basic_df))

        # Dtype
        panel_html.append('<div class="divider"></div>')
        dtype_counts = p.get("dtype_counts", {})
        dtype_cols = p.get("dtype_cols", {})
        panel_html.append(render_dtype_blocks(dtype_counts, dtype_cols))

        # Target distribution
        panel_html.append('<div class="divider"></div>')
        panel_html.append("<h3>Outcome distribution</h3>")
        target_data = p.get("target", {})
        target_df = records_to_df(target_data)
        panel_html.append(df_to_html_table(target_df))

        # Drift
        drift = p.get("drift", {})
        if drift and (drift.get("vs_last") or drift.get("vs_cum")):
            panel_html.append('<div class="divider"></div>')
            panel_html.append("<h3>Drift check</h3>")
            if drift.get("vs_last"):
                panel_html.append("<h4>This vs last</h4>")
                panel_html.append(df_to_html_table(records_to_df(drift["vs_last"])))
            if drift.get("vs_cum"):
                panel_html.append("<h4>This vs cumulative</h4>")
                panel_html.append(df_to_html_table(records_to_df(drift["vs_cum"])))

        # Data quality checks
        panel_html.append('<div class="divider"></div>')
        panel_html.append("<h3>Data Quality Checks</h3>")

        # 1. Missing data
        panel_html.append("<h4>1. Missing Data Check</h4>")
        if k == "Overall" and quality_reports:
            missing_df = records_to_df(quality_reports.get("missing_plan", []))
            if not missing_df.empty:
                top5 = missing_df.head(5)
                panel_html.append(df_to_html_table(top5))
                if len(missing_df) > 5:
                    panel_html.append(f'<p class="note">Top 5 of {len(missing_df)} columns with missing values.</p>')
            else:
                panel_html.append('<p class="note">No missing data issues found.</p>')
        else:
            panel_html.append('<p class="note">No missing data issues found (or see Overall tab).</p>')

        # 2. Outliers
        panel_html.append("<h4>2. Outlier Check</h4>")
        if k == "Overall" and quality_reports:
            outlier_df = records_to_df(quality_reports.get("outlier_report", []))
            if not outlier_df.empty:
                panel_html.append(df_to_html_table(outlier_df))
            else:
                panel_html.append('<p class="note">No outlier issues found.</p>')
        else:
            panel_html.append('<p class="note">No outlier issues found (or see Overall tab).</p>')

        # 3. Sanity check (integrity audit)
        panel_html.append("<h4>3. Sanity Check (Integrity Audit)</h4>")
        audit_data = p.get("audit", {})
        audit_df = records_to_df(audit_data)

        if not audit_df.empty:
            fails = audit_df[audit_df["severity"] == "FAIL"] if "severity" in audit_df.columns else pd.DataFrame()
            warns = audit_df[audit_df["severity"] == "WARN"] if "severity" in audit_df.columns else pd.DataFrame()
            infos = audit_df[audit_df["severity"] == "INFO"] if "severity" in audit_df.columns else pd.DataFrame()

            if len(fails) > 0:
                panel_html.append('<div style="color: #EF4444; font-weight: 600;">Critical Issues (FAIL)</div>')
                panel_html.append(df_to_html_table(fails))

            if len(warns) > 0:
                panel_html.append('<div style="color: #F59E0B; font-weight: 600;">Warnings (WARN)</div>')
                panel_html.append(df_to_html_table(warns))

            if len(infos) > 0:
                panel_html.append('<div style="color: #3B82F6; font-weight: 600;">Info (Review)</div>')
                panel_html.append(df_to_html_table(infos))
        else:
            panel_html.append('<p class="note">No sanity issues found.</p>')

        panel_html.append("</div>")
        batch_panels.append("".join(panel_html))

    return f"""
    <div class="card">
        <h2>Intake & Quality Checks</h2>
        <p class="description">Weekly intake → schema alignment → integrity audit → target balance & drift</p>

        <div class="select-wrapper">
            <label class="select-label">Select batch:</label>
            <select id="intake_select" onchange="showIntakeBatch(this.value)">
                {select_options}
            </select>
        </div>

        {''.join(batch_panels)}
    </div>
    """


def generate_data_governance_profiling(pipeline_data):
    """Generate the Data Governance - Profiling page"""

    profiles = pipeline_data.get("profiles", {})

    if not profiles:
        return """
        <div class="card">
            <h2>Feature Profiling</h2>
            <p class="note">No profiling data available.</p>
        </div>
        """

    profile_keys = list(profiles.keys())
    select_options = "".join([f'<option value="{escape(k)}">{escape(k)}</option>' for k in profile_keys])

    # Build a panel for each feature group
    profile_panels = []
    for k in profile_keys:
        block = profiles[k]

        panel_html = [f'<div class="sub-panel" id="profile_{escape(k)}">']
        panel_html.append(f"<h3>{escape(k)}</h3>")

        # Binary variables
        if "binary_chart" in block and block["binary_chart"]:
            panel_html.append("<h4>Binary variables</h4>")
            panel_html.append(f'<img src="data:image/png;base64,{block["binary_chart"]}" style="max-width:800px;"/>')

        # Continuous variables
        if "continuous_chart" in block and block["continuous_chart"]:
            panel_html.append("<h4>Continuous/Count variables</h4>")
            panel_html.append(
                f'<img src="data:image/png;base64,{block["continuous_chart"]}" style="max-width:800px;"/>'
            )

        # Categorical variables
        if "categorical_chart" in block and block["categorical_chart"]:
            panel_html.append("<h4>Categorical variables</h4>")
            panel_html.append(
                f'<img src="data:image/png;base64,{block["categorical_chart"]}" style="max-width:800px;"/>'
            )

        panel_html.append("</div>")
        profile_panels.append("".join(panel_html))

    return f"""
    <div class="card">
        <h2>Feature Profiling</h2>
        <p class="description">Variable distributions by feature group (binary, continuous, categorical)</p>

        <div style="background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 14px; color: #92400E;">
                 <strong>Data Dictionary:</strong> For coded variables, see
                <a href="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
                   target="_blank" style="color: #2563EB; text-decoration: underline;">
                    UCI Dataset Documentation ↗
                </a>
            </p>
        </div>

        <div class="select-wrapper">
            <label class="select-label">Select feature group:</label>
            <select id="profile_select" onchange="showProfileGroup(this.value)">
                {select_options}
            </select>
        </div>

        {''.join(profile_panels)}
    </div>
    """


def generate_data_governance_target(pipeline_data):
    """Generate the Data Governance - Target Analysis page."""

    target_slices = pipeline_data.get("target_slices", {})
    slice_cols = pipeline_data.get("slice_cols", [])

    if not slice_cols or not target_slices:
        return """
        <div class="card">
            <h2>Target Analysis</h2>
            <p class="note">No target analysis data available.</p>
        </div>
        """

    select_options = "".join([f'<option value="{escape(col)}">{escape(col)}</option>' for col in slice_cols])

    # Build a panel for each slice variable
    target_panels = []
    for col in slice_cols:
        if col not in target_slices:
            continue

        item = target_slices[col]

        panel_html = [f'<div class="sub-panel" id="target_{escape(col)}">']
        panel_html.append(f"<h3>{escape(col)}</h3>")

        # Display chart
        if "chart" in item and item["chart"]:
            panel_html.append(f'<img src="data:image/png;base64,{item["chart"]}" style="max-width:800px;"/>')
        else:
            panel_html.append('<p class="note">No chart available for this variable.</p>')

        panel_html.append("</div>")
        target_panels.append("".join(panel_html))

    return f"""
    <div class="card">
        <h2>Target Analysis</h2>
        <p class="description">How different variables relate to outcomes (Graduate / Enrolled / Dropout)</p>

        <div class="select-wrapper">
            <label class="select-label">Select variable:</label>
            <select id="target_select" onchange="showTargetVariable(this.value)">
                {select_options}
            </select>
        </div>

        {''.join(target_panels)}
    </div>
    """
