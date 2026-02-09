"""
build_integrated_report.py

Orchestrates the generation of the complete integrated HTML report
by importing and calling functions from specialized content generator modules.
"""

import json
import pandas as pd
from html import escape
from datetime import datetime
import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "stage4_reporting"))

# ============================================================================
# Import content generators from modular files
# ============================================================================

# Part 1: Executive Summary (Decision Support)
try:
    from content_generators.get_executive_summary import (
        generate_student_level_actions,
        generate_cohort_comparison,
        generate_system_level_actions
    )
    EXECUTIVE_SUMMARY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: content_generators.get_executive_summary not found: {e}")
    EXECUTIVE_SUMMARY_AVAILABLE = False

# Part 2: Data Governance
try:
    from content_generators.get_data_governance import (
        generate_data_governance_intake,
        generate_data_governance_profiling,
        generate_data_governance_target
    )
    DATA_GOVERNANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: content_generators.get_data_governance not found: {e}")
    print("   Data Governance sections will use fallback content.")
    DATA_GOVERNANCE_AVAILABLE = False

# Part 3: Feature Strategy & Model Rigor
try:
    from content_generators.get_features_modeling import (
        generate_feature_strategy_engineering,
        generate_feature_strategy_correlation,
        generate_feature_strategy_importance,
        generate_model_overview,
        generate_model_comparison,
        generate_model_fairness
    )
    FEATURES_MODELING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: content_generators.get_features_modeling not found: {e}")
    print("   Feature Strategy and Model Rigor sections will use fallback content.")
    FEATURES_MODELING_AVAILABLE = False

# Home Page
try:
    from content_generators.home_page import generate_home_page
    HOME_AVAILABLE = True
except ImportError as e:
    print(f"Warning: content_generators.home_page not found: {e}")
    HOME_AVAILABLE = False
    
    def generate_home_page():
        return '<div class="card"><h2>Home</h2><p>Welcome to EduTech Analytics</p></div>'


# ============================================================================
# Fallback functions (if modules not available)
# ============================================================================

def generate_student_level_actions_fallback(actions):
    """Fallback if executive summary not available"""
    return """
    <div class="card">
        <h2>Student-Level Actions</h2>
        <div class="alert alert-warning">
            Executive summary module not available. Please check content_generators/get_executive_summary.py
        </div>
    </div>
    """


def generate_cohort_comparison_fallback(actions):
    """Fallback if executive summary not available"""
    return """
    <div class="card">
        <h2>Cohort Comparison</h2>
        <div class="alert alert-warning">
            Executive summary module not available. Please check content_generators/get_executive_summary.py
        </div>
    </div>
    """


def generate_system_level_actions_fallback(actions):
    """Fallback if executive summary not available"""
    return """
    <div class="card">
        <h2>System-Level Actions</h2>
        <div class="alert alert-warning">
            Executive summary module not available. Please check content_generators/get_executive_summary.py
        </div>
    </div>
    """


def generate_data_governance_intake_fallback(pipeline_data):
    """Fallback if data governance module not available"""
    return """
    <div class="card">
        <h2>Intake & Quality Checks</h2>
        <div class="alert alert-warning">
            Data governance module not available. Please check content_generators/get_data_governance.py
        </div>
    </div>
    """


def generate_data_governance_profiling_fallback(pipeline_data):
    """Fallback if data governance module not available"""
    return """
    <div class="card">
        <h2>Feature Profiling</h2>
        <div class="alert alert-warning">
            Data governance module not available. Please check content_generators/get_data_governance.py
        </div>
    </div>
    """


def generate_data_governance_target_fallback(pipeline_data):
    """Fallback if data governance module not available"""
    return """
    <div class="card">
        <h2>Target Analysis</h2>
        <div class="alert alert-warning">
            Data governance module not available. Please check content_generators/get_data_governance.py
        </div>
    </div>
    """


def generate_feature_strategy_fallback(section_name):
    """Fallback if features/modeling module not available"""
    return f"""
    <div class="card">
        <h2>{section_name}</h2>
        <div class="alert alert-warning">
            Feature strategy module not available. Please check content_generators/get_features_modeling.py
        </div>
    </div>
    """


# ============================================================================
# Data Loading
# ============================================================================

def load_all_data(project_root: Path):
    """Load all data files from previous pipeline stages"""
    print("Loading all data files...")
    
    stage1_dir = project_root / "outputs" / "stage1_quality"
    stage2_dir = project_root / "outputs" / "stage2_features"
    stage3_dir = project_root / "outputs" / "stage3_modeling"

    pipeline_path = stage1_dir / "pipeline_data.json"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {pipeline_path.relative_to(project_root)}\n"
            f"   Please run Stage 1 first: python src/stage1_data_quality/run_data_quality.py"
        )
    with open(pipeline_path, 'r', encoding='utf-8') as f:
        pipeline_data = json.load(f)
    print(f"   Loaded {pipeline_path.relative_to(project_root)}")

    feature_path = stage2_dir / "feature_strategy.json"
    if not feature_path.exists():
        print(f"Stage 2 output not found: {feature_path.relative_to(project_root)}")
        feature_strategy = {}
    else:
        with open(feature_path, 'r', encoding='utf-8') as f:
            feature_strategy = json.load(f)
        print(f"   Loaded {feature_path.relative_to(project_root)}")
    
    modeling_path = stage3_dir / "modeling_results.json"
    if not modeling_path.exists():
        raise FileNotFoundError(
            f"Stage 3 modeling results not found: {modeling_path.relative_to(project_root)}\n"
            f"   Please run Stage 3 first: python src/stage3_modeling_action/run_modeling_action.py"
        )
    with open(modeling_path, 'r', encoding='utf-8') as f:
        modeling_results = json.load(f)
    print(f"   Loaded {modeling_path.relative_to(project_root)}")
  
    action_path = stage3_dir / "action_plans.json"
    if not action_path.exists():
        raise FileNotFoundError(
            f"Stage 3 action plans not found: {action_path.relative_to(project_root)}"
        )
    with open(action_path, 'r', encoding='utf-8') as f:
        action_plans = json.load(f)
    print(f"   Loaded {action_path.relative_to(project_root)}")
    
    print("   All data loaded successfully")
    
    return {
        'actions': action_plans,
        'modeling': modeling_results,
        'pipeline': pipeline_data,
        'feature_strategy': feature_strategy
    }


# ============================================================================
# CSS Styles
# ============================================================================

def get_css_styles():
    """Return CSS styles for the integrated report"""
    return """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #F8FAFC; 
    color: #1E293B; 
    min-height: 100vh;
}

.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    width: 280px;
    height: 100vh;
    background-color: #0A1172; 
    padding: 24px 0;
    overflow-y: auto;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    z-index: 1000;
}

.sidebar h1 {
    font-size: 1.2rem;
    font-weight: 700;
    margin: 0 20px 30px;
    color: #FFFFFF;
}

.nav-section {
    margin: 24px 0;
}

.nav-title {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    padding: 0 20px;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

.nav-item {
    display: block;
    padding: 12px 20px;
    color: #CBD5E1;
    text-decoration: none;
    font-size: 14px;
    transition: all 0.2s;
    border-left: 4px solid transparent;
}

.nav-item:hover, .nav-item.active {
    background-color: rgba(255, 255, 255, 0.1);
    color: #FFFFFF;
    border-left-color: #4D77FF;
}

.nav-item.active {
    background: rgba(77, 119, 255, 0.2);
    color: #FFF;
    border-left-color: #4D77FF;
    font-weight: 500;
}

.nav-sub {
    display: none;
    padding-left: 20px;
}

.nav-sub.active {
    display: block;
}

.nav-sub-item {
    display: block;
    padding: 8px 20px;
    font-size: 13px;
    color: #BFDBFE;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.15s;
}

.nav-sub-item:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #FFF;
}

.nav-sub-item.active {
    color: #FFF;
    font-weight: 500;
}

.main-content {
    margin-left: 280px;
    padding: 50px;
    background-color: #FFFFFF;
    min-height: 100vh;
}

.content-section {
    display: none;
    animation: fadeIn 0.3s;
}

.content-section.active {
    display: block;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.sub-section {
    display: none;
}

.sub-section.active {
    display: block;
}

.card {
    background: #FFFFFF;
    padding: 30px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    margin-bottom: 40px;
    border-top: 6px solid #0A1172; 
}

.card h2 {
    color: #0A1172;
    font-size: 1.6rem;
    margin-bottom: 25px;
    font-weight: 800;
}

.card h3 {
    color: #0A1172;
    margin: 20px 0 15px 0;
    font-size: 1.3em;
    font-weight: 700;
}

.card h4 {
    color: #334155;
    margin: 15px 0 10px 0;
    font-size: 1.1em;
    font-weight: 600;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.metric {
    padding: 20px;
    border-radius: 4px;
    color: #FFFFFF;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    justify-content: center;
}

.metric:nth-child(1) { background-color: #4D77FF; }
.metric:nth-child(2) { background-color: #A855F7; }
.metric:nth-child(3) { background-color: #EF1233; } 
.metric:nth-child(4) { background-color: #0A1172; } 

.metric .label {
    font-size: 0.9em;
    color: rgba(255, 255, 255, 0.9);
    opacity: 0.9;
    font-weight: 500;
    margin-bottom: 8px;
    background: transparent !important;
    padding: 0 !important;
    display: block;
}

.metric .value {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 5px;
    display: block;
}

.metric .note {
    font-size: 0.85em;
    opacity: 0.95;
}

table, .data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td, .data-table th, .data-table td {
    padding: 12px;
    text-align: center;
    background-color: #F9FAFB;
    color: #4B5563;
    font-weight: 600;
    border-bottom: 2px solid #E5E7EB;
}

th, .data-table th {
    background: #4D77FF;
    color: white;
    font-weight: bold;
}

tr:hover {
    background-color: #F9FAFB;
}

.alert {
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}

.alert-success {
    background: #d4edda;
    border-left: 5px solid #28a745;
    color: #155724;
}

.alert-warning {
    background: #fff3cd;
    border-left: 5px solid #ffc107;
    color: #856404;
}

.alert-info {
    background-color: #E0E7FF; 
    border-left: 5px solid #4D77FF;
    padding: 15px;
    border-radius: 8px;
    margin: 20px 0;
    color: #1E1B4B; 
    font-weight: 500;
}

.badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: bold;
}

.badge-critical {
    background: #dc3545;
    color: white;
}

.badge-warning {
    background-color: #EF1233 !important;
    color: white !important;
}

.badge-success {
    background: #28a745;
    color: white;
}

.select-wrapper {
    margin: 20px 0;
}

.select-wrapper select, .select-label + select {
    padding: 10px 15px;
    font-size: 1em;
    border: 2px solid #667eea;
    border-radius: 8px;
    background: white;
    cursor: pointer;
    min-width: 300px;
}

.sub-panel {
    display: none;
}

.sub-panel.active {
    display: block;
}

.radar-panel {
    display: none;
}

.radar-panel.active {
    display: block;
}

.cost-comparison {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.cost-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}

.cost-card h4 {
    font-size: 1.3em;
    margin-bottom: 15px;
    border-bottom: 2px solid white;
    padding-bottom: 10px;
    color: white;
}

.cost-metric {
    margin: 10px 0;
    font-size: 0.95em;
}

.whatif-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

.whatif-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    text-align: center;
}

.whatif-table td {
    padding: 15px;
    text-align: center;
    border: 1px solid #ddd;
}

.whatif-table .baseline {
    background: #f8f9fa;
    font-weight: bold;
}

.whatif-table .scenario {
    background: #e7f3ff;
}

.risk-high {
    color: #dc3545;
    font-weight: bold;
}

.risk-medium {
    color: #ffc107;
    font-weight: bold;
}

.risk-low {
    color: #28a745;
    font-weight: bold;
}

.description {
    font-size: 14px;
    color: #666;
    margin-bottom: 20px;
}

.note {
    font-size: 13px;
    color: #666;
}

.sample-box {
    background: #F9FAFB;
    border-left: 4px solid #3B82F6;
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 4px;
}

.sample-text {
    font-size: 13px;
    color: #374151;
    font-style: italic;
    margin-bottom: 8px;
}

.label {
    display: inline-block;
    padding: 3px 8px;
    background: #DBEAFE;
    border-radius: 4px;
    margin-right: 8px;
    margin-top: 4px;
    font-size: 12px;
}

.divider {
    height: 1px;
    background: #E5E7EB;
    margin: 24px 0;
}

.select-label {
    display: block;
    font-size: 14px;
    font-weight: 500;
    color: #374151;
    margin-bottom: 8px;
}
"""


# ============================================================================
# JavaScript
# ============================================================================

def get_javascript():
    """Return JavaScript for interactive navigation and switching"""
    return """
function showSection(sectionId) {
    document.querySelectorAll('.content-section').forEach(s => {
        s.classList.remove('active');
    });
    
    const section = document.getElementById('section-' + sectionId);
    if (section) {
        section.classList.add('active');
    }
    
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const navItem = document.querySelector(`[onclick*="showSection('${sectionId}')"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
    
    document.querySelectorAll('.nav-sub').forEach(sub => {
        sub.classList.remove('active');
    });
    
    const subNav = document.getElementById('nav-sub-' + sectionId);
    if (subNav) {
        subNav.classList.add('active');
    }
}

function showSubSection(parentId, subId) {
    document.querySelectorAll(`#section-${parentId} .sub-section`).forEach(s => {
        s.classList.remove('active');
    });
    
    const subSection = document.getElementById(`subsection-${parentId}-${subId}`);
    if (subSection) {
        subSection.classList.add('active');
    }
    
    document.querySelectorAll(`#nav-sub-${parentId} .nav-sub-item`).forEach(item => {
        item.classList.remove('active');
    });
    
    const navItem = document.querySelector(`[onclick*="showSubSection('${parentId}', '${subId}')"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
}

function showPriorityGroup(groupId) {
    document.querySelectorAll('[id^="priority-"]').forEach(p => {
        p.classList.remove('active');
    });
    document.getElementById('priority-' + groupId).classList.add('active');
}

function showRadarDimension(dimId) {
    document.querySelectorAll('.radar-panel').forEach(p => {
        p.classList.remove('active');
    });
    document.getElementById('radar-' + dimId).classList.add('active');
}

function showIntakeBatch(batchId) {
    document.querySelectorAll('[id^="intake_"]').forEach(p => {
        p.classList.remove('active');
    });
    document.getElementById('intake_' + batchId).classList.add('active');
}

function showProfileGroup(groupId) {
    document.querySelectorAll('[id^="profile_"]').forEach(p => {
        p.classList.remove('active');
    });
    document.getElementById('profile_' + groupId).classList.add('active');
}

function showTargetVariable(varId) {
    document.querySelectorAll('[id^="target_"]').forEach(p => {
        p.classList.remove('active');
    });
    document.getElementById('target_' + varId).classList.add('active');
}

function showEngSub(val) {
    document.querySelectorAll('[id^="engineering_"]').forEach(p => p.classList.remove('active'));
    document.getElementById('engineering_' + val).classList.add('active');
}

function showCorrSub(val) {
    document.querySelectorAll('[id^="corr_"]').forEach(p => p.classList.remove('active'));
    let id = val === 'static' ? 'corr_static' : val === 'static_long' ? 'corr_static_long' : 'corr_long_nlp';
    document.getElementById(id).classList.add('active');
}

function showImpSub(val) {
    document.querySelectorAll('[id^="importance_"]').forEach(p => p.classList.remove('active'));
    document.getElementById('importance_' + val).classList.add('active');
}

document.addEventListener('DOMContentLoaded', function() {
    const intakeSelect = document.getElementById('intake_select');
    if (intakeSelect && intakeSelect.options.length > 0) {
        showIntakeBatch(intakeSelect.value);
    }
    
    const profileSelect = document.getElementById('profile_select');
    if (profileSelect && profileSelect.options.length > 0) {
        showProfileGroup(profileSelect.value);
    }
    
    const targetSelect = document.getElementById('target_select');
    if (targetSelect && targetSelect.options.length > 0) {
        showTargetVariable(targetSelect.value);
    }
    
    const engSelect = document.getElementById('engineering_select');
    if (engSelect && engSelect.options.length > 0) {
        showEngSub(engSelect.value);
    }
    
    const corrSelect = document.getElementById('corr_select');
    if (corrSelect && corrSelect.options.length > 0) {
        showCorrSub(corrSelect.value);
    }
    
    const impSelect = document.getElementById('importance_select');
    if (impSelect && impSelect.options.length > 0) {
        showImpSub(impSelect.value);
    }
});
"""


# ============================================================================
# Main HTML Generation
# ============================================================================

def generate_integrated_html(data, output_path):
    """Generate complete integrated HTML report by calling imported content generators"""
    print("\nGenerating complete integrated HTML report...")
    
    home_html = generate_home_page()
    actions = data['actions']
    modeling = data['modeling']
    pipeline = data['pipeline']
    feature_strategy = data.get('feature_strategy', {})
    
    # Merge feature strategy data into pipeline for backward compatibility
    if feature_strategy:
        tables = feature_strategy.get('tables', {})
        pipeline['longitudinal_summary'] = tables.get('longitudinal_summary', [])
        pipeline['nlp_summary'] = tables.get('nlp_summary', [])
        pipeline['nlp_samples'] = tables.get('nlp_samples', [])
        pipeline['l1_importance'] = tables.get('l1_importance', [])
        pipeline['rf_importance'] = tables.get('rf_importance', [])
        pipeline['combined_importance'] = tables.get('combined_importance', [])
        
        visualizations = feature_strategy.get('visualizations', {})
        pipeline['correlation_heatmaps'] = visualizations.get('heatmaps', {})
        pipeline['feature_importance_plots'] = visualizations.get('feature_importance', {})
        pipeline['nlp_distribution_plot'] = visualizations.get('nlp_distribution', '')
    
    # Part 1: Decision Support (Executive Summary)
    print("   → Generating Decision Support sections...")
    if EXECUTIVE_SUMMARY_AVAILABLE:
        student_actions = generate_student_level_actions(actions)
        cohort_actions = generate_cohort_comparison(actions)
        system_actions = generate_system_level_actions(actions)
    else:
        print("      Warning: Using fallback functions for executive summary")
        student_actions = generate_student_level_actions_fallback(actions)
        cohort_actions = generate_cohort_comparison_fallback(actions)
        system_actions = generate_system_level_actions_fallback(actions)
    
    # Part 2: Data Governance
    print("   → Generating Data Governance sections...")
    if DATA_GOVERNANCE_AVAILABLE:
        data_intake = generate_data_governance_intake(pipeline)
        data_profiling = generate_data_governance_profiling(pipeline)
        data_target = generate_data_governance_target(pipeline)
    else:
        print("      Warning: Using fallback functions for data governance")
        data_intake = generate_data_governance_intake_fallback(pipeline)
        data_profiling = generate_data_governance_profiling_fallback(pipeline)
        data_target = generate_data_governance_target_fallback(pipeline)
    
    # Part 3: Feature Strategy
    print("   → Generating Feature Strategy sections...")
    if FEATURES_MODELING_AVAILABLE:
        feature_engineering = generate_feature_strategy_engineering(pipeline)
        feature_correlation = generate_feature_strategy_correlation(pipeline)
        feature_importance = generate_feature_strategy_importance(pipeline)
    else:
        print("      Warning: Using fallback functions for feature strategy")
        feature_engineering = generate_feature_strategy_fallback("Feature Engineering")
        feature_correlation = generate_feature_strategy_fallback("Correlation Analysis")
        feature_importance = generate_feature_strategy_fallback("Feature Importance")
    
    # Part 4: Model Rigor & Equity
    print("   → Generating Model Rigor sections...")
    if FEATURES_MODELING_AVAILABLE:
        model_overview = generate_model_overview(modeling)
        model_comparison = generate_model_comparison(modeling)
        model_fairness = generate_model_fairness(modeling)
    else:
        print("      Warning: Using fallback functions for model rigor")
        model_overview = generate_feature_strategy_fallback("Model Overview")
        model_comparison = generate_feature_strategy_fallback("Model Comparison")
        model_fairness = generate_feature_strategy_fallback("Fairness Audit")
    
    # Assemble HTML
    print("   → Assembling complete HTML...")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EduTech Analytics - Complete Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {get_css_styles()}
    </style>
</head>
<body>

<div class="sidebar">
    <h1>EduTech Analytics</h1>
    
    <div class="nav-section">
        <a class="nav-item active" onclick="showSection('home')">Home</a>
    </div>
    
    <div class="nav-section">
        <div class="nav-title">DECISION SUPPORT</div>
        <a class="nav-item" onclick="showSection('student')">Student-Level</a>
        <a class="nav-item" onclick="showSection('cohort')">Cohort-Level</a>
        <a class="nav-item" onclick="showSection('system')">System-Level</a>
    </div>
    
    <div class="nav-section">
        <div class="nav-title">DATA SCIENCE & ENGINEERING</div>
        
        <a class="nav-item" onclick="showSection('data')">Data Governance</a>
        <div class="nav-sub" id="nav-sub-data">
            <a class="nav-sub-item active" onclick="showSubSection('data', 'intake')">Intake & Quality</a>
            <a class="nav-sub-item" onclick="showSubSection('data', 'profiling')">Profiling</a>
            <a class="nav-sub-item" onclick="showSubSection('data', 'target')">Target Analysis</a>
        </div>
        
        <a class="nav-item" onclick="showSection('feature')">Feature Strategy</a>
        <div class="nav-sub" id="nav-sub-feature">
            <a class="nav-sub-item active" onclick="showSubSection('feature', 'engineering')">Feature Engineering</a>
            <a class="nav-sub-item" onclick="showSubSection('feature', 'correlation')">Correlation Analysis</a>
            <a class="nav-sub-item" onclick="showSubSection('feature', 'importance')">Feature Importance</a>
        </div>
        
        <a class="nav-item" onclick="showSection('model')">Model Rigor & Equity</a>
        <div class="nav-sub" id="nav-sub-model">
            <a class="nav-sub-item active" onclick="showSubSection('model', 'overview')">Overview</a>
            <a class="nav-sub-item" onclick="showSubSection('model', 'comparison')">Model Comparison</a>
            <a class="nav-sub-item" onclick="showSubSection('model', 'fairness')">Fairness Audit</a>
        </div>
    </div>
</div>

<div class="main-content">
    <div class="content-section active" id="section-home">
        {home_html}
    </div>
 
    <div class="content-section" id="section-student">
        {student_actions}
    </div>
    
    <div class="content-section" id="section-cohort">
        {cohort_actions}
    </div>
    
    <div class="content-section" id="section-system">
        {system_actions}
    </div>
    
    <div class="content-section" id="section-data">
        <div class="sub-section active" id="subsection-data-intake">
            {data_intake}
        </div>
        <div class="sub-section" id="subsection-data-profiling">
            {data_profiling}
        </div>
        <div class="sub-section" id="subsection-data-target">
            {data_target}
        </div>
    </div>
    
    <div class="content-section" id="section-feature">
        <div class="sub-section active" id="subsection-feature-engineering">
            {feature_engineering}
        </div>
        <div class="sub-section" id="subsection-feature-correlation">
            {feature_correlation}
        </div>
        <div class="sub-section" id="subsection-feature-importance">
            {feature_importance}
        </div>
    </div>
    
    <div class="content-section" id="section-model">
        <div class="sub-section active" id="subsection-model-overview">
            {model_overview}
        </div>
        <div class="sub-section" id="subsection-model-comparison">
            {model_comparison}
        </div>
        <div class="sub-section" id="subsection-model-fairness">
            {model_fairness}
        </div>
    </div>
</div>

<script>
    {get_javascript()}
</script>

</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"   Complete report saved: {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point"""
    
    output_dir = PROJECT_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "integrated_report.html"
    
    print("=" * 70)
    print("EduTech Analytics - Complete Integrated Report Generator")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)
    
    try:
        data = load_all_data(PROJECT_ROOT)
        
        generate_integrated_html(data, str(output_path))
        
        print("\n" + "=" * 70)
        print("Complete report generated successfully!")
        print(f"Saved to: {output_path.relative_to(PROJECT_ROOT)}")
        print("=" * 70)
        print(f"\nOpen in browser:")
        print(f"   file://{output_path.absolute()}")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nMake sure you've run all previous stages:")
        print(f"   1. python src/stage1_data_quality/run_data_quality.py")
        print(f"   2. python src/stage2_feature_engineering/run_feature_strategy.py")
        print(f"   3. python src/stage3_modeling_action/run_modeling_action.py")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()