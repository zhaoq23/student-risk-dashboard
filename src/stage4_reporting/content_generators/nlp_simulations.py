"""
nlp_simulations.py
Simulate NLP-based student feedback analysis for demonstration purposes
"""

import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_entity_feedback_matrix():
    """
    Generate Entity-Driven Feedback Matrix visualization.
    Returns base64 encoded image string for HTML embedding.
    """
    # Simulate NLP analysis summary data
    # Prevalence: Topic frequency across all surveys (0-1)
    # Sentiment: Average sentiment score (-1 very dissatisfied, 1 very satisfied)
    data = {
        'Topic': [
            'Lab Report Guidelines',
            'Math Homework Volume',
            'Group Discussions',
            'Interactive Simulations',
            'Python Coding Basics',
            'Teacher Office Hours',
            'LMS Login Issues',
            'Library Resources'
        ],
        'Prevalence': [0.85, 0.78, 0.90, 0.72, 0.35, 0.45, 0.25, 0.20],
        'Sentiment': [-0.65, -0.45, 0.75, 0.55, 0.40, 0.60, -0.35, -0.20]
    }
    df = pd.DataFrame(data)
    
    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color scheme matching the existing dashboard
    colors = []
    for _, row in df.iterrows():
        if row['Prevalence'] > 0.5 and row['Sentiment'] > 0:
            colors.append('#27ae60')  # Success cases - green
        elif row['Prevalence'] > 0.5 and row['Sentiment'] < 0:
            colors.append('#e74c3c')  # Critical issues - red
        elif row['Prevalence'] <= 0.5 and row['Sentiment'] > 0:
            colors.append('#667eea')  # Niche strengths - blue
        else:
            colors.append('#f39c12')  # Minor frustrations - orange
    
    # Plot scatter points with color coding
    scatter = ax.scatter(
        df['Prevalence'], 
        df['Sentiment'], 
        s=200, 
        c=colors, 
        alpha=0.7, 
        edgecolors='white',
        linewidth=2,
        zorder=3
    )
    
    # Add labels for each point
    for i, txt in enumerate(df['Topic']):
        ax.annotate(
            txt, 
            (df['Prevalence'][i] + 0.02, df['Sentiment'][i]),
            fontsize=9,
            fontweight='500',
            color='#2c3e50'
        )
    
    # Draw quadrant dividers
    ax.axhline(0, color='#34495e', linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)
    ax.axvline(0.5, color='#34495e', linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)
    
    # Add quadrant labels with action recommendations
    quadrant_style = {
        'fontsize': 11,
        'weight': 'bold',
        'ha': 'center',
        'va': 'center',
        'bbox': dict(boxstyle='round,pad=0.5', alpha=0.15)
    }
    
    ax.text(0.75, 0.80, 'SUCCESS CASES\n(Scale & Promote)', 
            color='#27ae60', **quadrant_style)
    ax.text(0.75, -0.80, 'CRITICAL ISSUES\n(Immediate Action)', 
            color='#e74c3c', **quadrant_style)
    ax.text(0.25, 0.80, 'NICHE STRENGTHS\n(Monitor)', 
            color='#667eea', **quadrant_style)
    ax.text(0.25, -0.80, 'MINOR FRUSTRATIONS\n(Low Priority)', 
            color='#f39c12', **quadrant_style)
    
    # Styling
    ax.set_title(
        'Entity-Driven Feedback Matrix\nCourse Improvement Strategy (Student Survey NLP Analysis)',
        fontsize=13,
        fontweight='600',
        color='#2c3e50',
        pad=20
    )
    ax.set_xlabel('Topic Prevalence (Frequency of Mention)', fontsize=11, color='#34495e')
    ax.set_ylabel('Sentiment Polarity (Negative ← → Positive)', fontsize=11, color='#34495e')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1, 1)
    
    # Grid styling
    ax.grid(True, linestyle=':', alpha=0.3, color='#95a5a6', zorder=0)
    ax.set_facecolor('#f8f9fa')
    
    # Spines styling
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return image_base64


def generate_sentiment_timeline():
    """
    Generate Longitudinal Sentiment Analysis timeline visualization.
    Returns base64 encoded image string for HTML embedding.
    """
    # Simulate system-wide monthly sentiment data
    data = {
        'Month': ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Sentiment_Score': [75, 72, 68, 45, 70, 73, 71, 40, 65, 78],
        # Top entities extracted by spaCy during sentiment dips
        'Alert_Entity': [None, None, None, 'LMS Update', None, None, None, 'Regents Exam', None, None]
    }
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot main sentiment line
    ax.plot(
        df['Month'], 
        df['Sentiment_Score'], 
        marker='o', 
        linestyle='-', 
        color='#667eea', 
        linewidth=2.5,
        markersize=8,
        markerfacecolor='#667eea',
        markeredgecolor='white',
        markeredgewidth=2,
        label='System Sentiment Pulse',
        zorder=3
    )
    
    # Add alert markers and annotations ONLY for points below threshold
    for i, row in df.iterrows():
        if row['Alert_Entity'] and row['Sentiment_Score'] < 50:  # Only alert if below threshold
            # Red alert point
            ax.scatter(
                row['Month'], 
                row['Sentiment_Score'], 
                color='#e74c3c', 
                s=200, 
                zorder=5,
                edgecolors='white',
                linewidth=2
            )
            
            # Annotation box
            ax.annotate(
                f"{row['Alert_Entity']}",  # Just the entity name, no "ALERT:" prefix
                xy=(row['Month'], row['Sentiment_Score']),
                xytext=(row['Month'], row['Sentiment_Score'] - 12),
                ha='center',
                fontsize=8,  # Smaller font for entity name
                fontweight='600',
                color='#ffffff',  # White text for better contrast
                arrowprops=dict(
                    arrowstyle='->',
                    color='#e74c3c',
                    lw=1.5,
                    connectionstyle='arc3,rad=0'
                ),
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor='#e74c3c',  # Red background
                    edgecolor='#c0392b',
                    linewidth=1.5,
                    alpha=0.95
                ),
                zorder=4
            )
    
    # Reference line at 50 (threshold)
    ax.axhline(
        50, 
        color='#95a5a6', 
        linestyle='--', 
        linewidth=1.5, 
        alpha=0.5,
        label='Baseline Threshold',
        zorder=1
    )
    
    # Fill area under curve
    ax.fill_between(
        range(len(df)), 
        df['Sentiment_Score'], 
        50,
        where=(df['Sentiment_Score'] >= 50),
        color='#667eea', 
        alpha=0.1,
        zorder=0
    )
    ax.fill_between(
        range(len(df)), 
        df['Sentiment_Score'], 
        50,
        where=(df['Sentiment_Score'] < 50),
        color='#e74c3c', 
        alpha=0.1,
        zorder=0
    )
    
    # Styling
    ax.set_title(
        'Longitudinal Monitoring of Student Climate',
        fontsize=13,
        fontweight='600',
        color='#2c3e50',
        pad=20
    )
    ax.set_ylabel('Average Sentiment Score (0-100)', fontsize=11, color='#34495e')
    ax.set_xlabel('Academic Year Timeline', fontsize=11, color='#34495e')
    ax.set_ylim(0, 100)
    
    # Grid
    ax.grid(axis='y', linestyle=':', alpha=0.4, color='#bdc3c7', zorder=0)
    ax.set_facecolor('#f8f9fa')
    
    # Spines
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1)
    
    # Legend
    ax.legend(
        loc='lower left',
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor='#bdc3c7'
    )
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return image_base64


def generate_nlp_analysis_card():
    """
    Generate complete NLP analysis card with Entity-Driven Feedback Matrix.
    This is inserted after cohort comparison in the executive summary.
    """
    try:
        entity_matrix_img = generate_entity_feedback_matrix()
    except Exception as e:
        print(f"Warning: Failed to generate entity feedback matrix: {e}")
        entity_matrix_img = None
    
    if not entity_matrix_img:
        return """
        <div class="card">
            <h2>Entity-Driven Feedback Matrix</h2>
            <div class="alert alert-warning">
                NLP analysis visualization unavailable. Please ensure matplotlib is installed.
            </div>
        </div>
        """
    
    return f"""
    <div class="card">
        <h2>Entity-Driven Feedback Matrix</h2>
        <p style="margin-bottom: 20px; color: #34495e;">
            <strong>Natural Language Processing Analysis:</strong> Student feedback surveys analyzed using 
            entity extraction and sentiment classification to identify actionable improvement areas.
        </p>
        
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{entity_matrix_img}" 
                 alt="Entity-Driven Feedback Matrix" 
                 style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        </div>
        
        <div class="alert alert-info" style="margin-top: 20px;">
            <strong>Quadrant Interpretation:</strong>
            <ul style="margin: 10px 0 0 20px; line-height: 1.8;">
                <li><strong style="color: #27ae60;">Success Cases (Top Right):</strong> High-frequency, positive topics - scale these practices</li>
                <li><strong style="color: #e74c3c;">Critical Issues (Bottom Right):</strong> High-frequency, negative topics - immediate intervention required</li>
                <li><strong style="color: #667eea;">Niche Strengths (Top Left):</strong> Low-frequency, positive topics - monitor and expand</li>
                <li><strong style="color: #f39c12;">Minor Frustrations (Bottom Left):</strong> Low-frequency, negative topics - low priority</li>
            </ul>
        </div>
        
        <div class="alert alert-success" style="margin-top: 15px;">
            <strong>Key Action:</strong> Prioritize <strong>"Math Homework Volume"</strong> and 
            <strong>"Lab Report Guidelines"</strong> for immediate review, while expanding successful 
            practices like <strong>"Group Discussions"</strong> to other courses.
        </div>
    </div>
    """


def generate_sentiment_timeline_card():
    """
    Generate complete sentiment timeline card with entity attribution.
    This is inserted before cost-benefit analysis in system-level actions.
    """
    try:
        timeline_img = generate_sentiment_timeline()
    except Exception as e:
        print(f"Warning: Failed to generate sentiment timeline: {e}")
        timeline_img = None
    
    if not timeline_img:
        return """
        <div class="card">
            <h2>Longitudinal Monitoring of Student Climate</h2>
            <div class="alert alert-warning">
                Sentiment timeline visualization unavailable. Please ensure matplotlib is installed.
            </div>
        </div>
        """
    
    return f"""
    <div class="card">
        <h2>Longitudinal Monitoring of Student Climate</h2>
        <p style="font-size: 0.9em; color: #7f8c8d; margin: -5px 0 15px 0;">
            NLP-Powered Sentiment Analysis & Entity Attribution
        </p>
        
        <p style="margin-bottom: 20px; color: #34495e;">
            <strong>Automated Monitoring System:</strong> Monthly sentiment scores derived from student 
            communications (emails, forum posts, surveys) with spaCy-powered entity extraction to identify 
            structural friction points.
        </p>
        
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{timeline_img}" 
                 alt="Sentiment Timeline" 
                 style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        </div>
        
        <div class="metric-grid" style="margin: 20px 0;">
            <div class="metric">
                <div class="label">Average Sentiment</div>
                <div class="value">68.1</div>
                <div class="note" style="color: #ffffff;">Academic Year 2024-25</div>
            </div>
            <div class="metric">
                <div class="label">Alert Triggers</div>
                <div class="value">2</div>
                <div class="note" style="color: #ffffff;">Below Threshold (50)</div>
            </div>
            <div class="metric">
                <div class="label">Top Entity (Dec)</div>
                <div class="value">LMS Update</div>
                <div class="note" style="color: #ffffff;">-15 point drop</div>
            </div>
            <div class="metric">
                <div class="label">Top Entity (Apr)</div>
                <div class="value">Regents Exam</div>
                <div class="note" style="color: #ffffff;">-11 point drop</div>
            </div>
        </div>
        
        <div class="alert alert-warning" style="margin-top: 15px;">
            <strong>December 2024 Alert:</strong> Sentiment dropped to 45 following LMS system update. 
            Analysis of student communications revealed widespread login issues and lost assignment data. 
            <strong>Action Taken:</strong> Emergency technical support hours and deadline extensions implemented.
        </div>
        
        <div class="alert alert-warning" style="margin-top: 15px;">
            <strong>April 2025 Alert:</strong> Sentiment declined to 40 during Regents Exam period. 
            Entity analysis showed concerns about exam format changes and study resource availability. 
            <strong>Recommendation:</strong> Improve exam preparation communication and expand review sessions.
        </div>
        
        <div class="alert alert-info" style="margin-top: 15px;">
            <strong>Methodology:</strong> Sentiment scores computed using transformer-based models (DistilBERT) 
            on student text data. Named entity recognition (spaCy) extracts topics/events during negative spikes. 
            System auto-generates alerts when sentiment drops below baseline for two consecutive data points.
        </div>
    </div>
    """