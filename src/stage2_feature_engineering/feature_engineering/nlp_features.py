"""
nlp_features.py — Simulated NLP/LLM Output Features
"""

import pandas as pd
import numpy as np


SAMPLE_TEXTS = {
    "high_stress": [
        "I'm overwhelmed with assignments. Can't keep up.",
        "Too much pressure, feeling anxious about grades.",
        "Struggling to balance coursework with part-time job.",
        "Constant deadlines are exhausting. Not sure if I can continue.",
    ],
    "medium_stress": [
        "The workload is manageable but sometimes challenging.",
        "Some courses are harder than expected, but I'm coping.",
        "Balancing studies and personal life is tricky but doable.",
    ],
    "low_stress": [
        "I'm enjoying my courses and managing time well.",
        "The program is interesting and I feel prepared.",
        "I have good support from family and doing well.",
    ],
    "home_risk": [
        "Family issues are affecting my focus on studies.",
        "Financial problems at home, worried about continuing.",
        "Taking care of sick parent while studying.",
        "Family doesn't understand why I'm in school.",
    ],
    "math_difficulty": [
        "Mathematics courses are extremely challenging for me.",
        "Struggling specifically with calculus and statistics.",
    ],
    "language_difficulty": [
        "English writing assignments are very difficult.",
        "Language barrier is making comprehension hard.",
    ],
}


def generate_nlp_features(df: pd.DataFrame, coverage_rate: float = 0.9, seed: int = 42) -> pd.DataFrame:
    """Generate simulated NLP features"""
    
    np.random.seed(seed)
    n = len(df)
    out = df.copy()
    
    # Determine which students have feedback
    has_feedback = np.random.rand(n) < coverage_rate
    
    # ============================================================
    # (A) Academic_Stress: [0, 1]
    # ============================================================
    
    # Simulated based on academic performance
    if "Delta_Grade" in out.columns and "CR_Sem2" in out.columns:
        grade_stress = np.clip(-out["Delta_Grade"].fillna(0) / 10, 0, 1)
        cr_stress = np.clip(1 - out["CR_Sem2"].fillna(0.5), 0, 1)
        base_stress = (grade_stress + cr_stress) / 2
    else:
        base_stress = np.random.beta(2, 5, n)
    
    noise = np.random.normal(0, 0.1, n)
    academic_stress = np.clip(base_stress + noise, 0, 1)
    out["Academic_Stress"] = np.where(has_feedback, academic_stress, np.nan)
    
    out["Academic_Stress_Level"] = pd.cut(
        out["Academic_Stress"],
        bins=[0, 0.33, 0.67, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    
    # ============================================================
    # (B) Home_Support_Risk: [0, 1]
    # ============================================================
    
    age = pd.to_numeric(out.get("Age at enrollment", 20), errors="coerce")
    international = pd.to_numeric(out.get("International", 0), errors="coerce")
    debtor = pd.to_numeric(out.get("Debtor", 0), errors="coerce")
    
    age_risk = np.clip((age - 20) / 30, 0, 1)
    intl_risk = international * 0.3
    debt_risk = debtor * 0.4
    
    base_home_risk = np.clip(age_risk + intl_risk + debt_risk, 0, 1)
    noise = np.random.normal(0, 0.15, n)
    home_risk = np.clip(base_home_risk + noise, 0, 1)
    
    out["Home_Support_Risk"] = np.where(has_feedback, home_risk, np.nan)
    
    # ============================================================
    # (C) Subject_Specific
    # ============================================================
    
    subject_types = ["None", "Math", "Language", "Other"]
    subject_probs = [0.6, 0.2, 0.15, 0.05]
    
    subject_specific = np.random.choice(subject_types, size=n, p=subject_probs)
    
    # Students with poor grades are more likely to report difficulties
    if "Curricular units 2nd sem (grade)" in out.columns:
        low_grade = pd.to_numeric(out["Curricular units 2nd sem (grade)"], errors="coerce") < 10
        subject_specific = np.where(
            low_grade & (np.random.rand(n) < 0.5),
            np.random.choice(["Math", "Language"], n),
            subject_specific
        )
    
    out["Subject_Specific"] = np.where(has_feedback, subject_specific, "No_Feedback")
    
    out["Subject_Risk_Flag"] = (
        (out["Subject_Specific"] != "None") & 
        (out["Subject_Specific"] != "No_Feedback")
    ).astype(int)
    
    difficulty_map = {
        "None": 0.0,
        "Math": 1.0,
        "Language": 0.5,
        "Other": 0.3,
        "No_Feedback": np.nan
    }
    out["Subject_Difficulty_Score"] = out["Subject_Specific"].map(difficulty_map)
    
    # ============================================================
    # Generate Sample Texts (For HTML Display)
    # ============================================================
    
    def generate_sample_text(stress, home_risk, subject):
        parts = []
        
        if stress > 0.67:
            parts.append(np.random.choice(SAMPLE_TEXTS["high_stress"]))
        elif stress > 0.33:
            parts.append(np.random.choice(SAMPLE_TEXTS["medium_stress"]))
        else:
            parts.append(np.random.choice(SAMPLE_TEXTS["low_stress"]))
        
        if home_risk > 0.5:
            parts.append(np.random.choice(SAMPLE_TEXTS["home_risk"]))
        
        if subject == "Math":
            parts.append(np.random.choice(SAMPLE_TEXTS["math_difficulty"]))
        elif subject == "Language":
            parts.append(np.random.choice(SAMPLE_TEXTS["language_difficulty"]))
        
        return " ".join(parts)
    
    # Save samples (Top 10) - Use dictionary instead of attributes
    sample_texts = []
    for i in range(min(10, n)):
        if has_feedback[i]:
            text = generate_sample_text(academic_stress[i], home_risk[i], subject_specific[i])
            sample_texts.append({
                "student_id": i,
                "text": text,
                "academic_stress": f"{academic_stress[i]:.3f}",
                "home_support_risk": f"{home_risk[i]:.3f}",
                "subject_specific": subject_specific[i]
            })
    
    # Store as list, do not use DataFrame attributes (to avoid warnings)
    out.attrs['nlp_samples'] = sample_texts
    
    return out


def get_nlp_feature_list() -> dict:
    return {
        "Academic Stress": ["Academic_Stress", "Academic_Stress_Level"],
        "Home Support": ["Home_Support_Risk"],
        "Subject Difficulty": ["Subject_Specific", "Subject_Risk_Flag", "Subject_Difficulty_Score"]
    }


def nlp_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = get_nlp_feature_list()
    
    for category, feat_list in features.items():
        for feat in feat_list:
            if feat in df.columns:
                non_null = int(df[feat].notna().sum())
                coverage = non_null / len(df) * 100
                
                if pd.api.types.is_numeric_dtype(df[feat]):
                    mean_val = f"{df[feat].mean():.3f}"
                    std_val = f"{df[feat].std():.3f}"
                else:
                    mean_val = "N/A"
                    std_val = "N/A"
                
                rows.append({
                    "Category": category,
                    "Feature": feat,
                    "Coverage": f"{coverage:.1f}%",
                    "Mean": mean_val,
                    "Std": std_val
                })
    
    return pd.DataFrame(rows)


def generate_nlp_distribution_plot(df: pd.DataFrame) -> str:
    """
    Generate Academic Stress Level distribution plot (Pie Chart)
    Returns base64 encoded image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    # Count distribution of Academic_Stress_Level
    if 'Academic_Stress_Level' not in df.columns:
        return None
    
    stress_counts = df['Academic_Stress_Level'].value_counts().sort_index()
    
    # If no data
    if stress_counts.empty:
        return None
    
    # Create Pie Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = ['#22C55E', '#FBBF24', '#EF4444']  # Green(Low), Yellow(Medium), Red(High)
    explode = (0.05, 0.05, 0.1)  # Emphasize High stress
    
    wedges, texts, autotexts = ax.pie(
        stress_counts.values,
        labels=stress_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 9, 'fontweight': 'bold'}
    )
    
    # Beautify percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)
        autotext.set_fontweight('bold')
    
    ax.set_title('Academic Stress Level Distribution', 
                 fontsize=10, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return plot_base64