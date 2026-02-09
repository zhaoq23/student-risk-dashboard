"""
rules.py — Business Logic and Rule Engine

This module defines the organizational structure of features, variable types, 
missing data strategies, and data integrity audits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np

# ============================================================
# 1) Feature Grouping
# ============================================================

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Outcome": ["Target"],

    "Demographics": [
        "Age at enrollment",
        "Gender",
        "Marital status",
        "Nacionality",
        "International",
        "Educational special needs",
        "Displaced",
    ],

    "Family background": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ],

    "Financial / administrative": [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ],

    "Admissions & pathway": [
        "Admission grade",
        "Previous qualification (grade)",
        "Application mode",
        "Application order",
        "Previous qualification",
        "Course",
        "Daytime/evening attendance",
    ],

    "Academic signals (Sem 1)": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],

    "Academic signals (Sem 2)": [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ],

    "Macro context": [
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ],
}


# ============================================================
# 2) Variable Categorization Sets
# ============================================================

# Coded categorical variables (Must not be treated as continuous)
CODED_CATEGORICAL_VARS: Set[str] = {
    "Course",
    "Application mode",
    "Previous qualification",
    "Application order",
    "Marital status",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Daytime/evening attendance",
}

# Binary variables (0/1)
BINARY_VARS: Set[str] = {
    "Gender",
    "International",
    "Educational special needs",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
}

# Truly continuous variables
TRUE_CONTINUOUS_VARS: Set[str] = {
    "Age at enrollment",
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}

# Count-based variables (Discrete but treated as continuous)
COUNT_LIKE_VARS: Set[str] = {
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (without evaluations)",
}


# ============================================================
# 3) Variable Type Classification Functions
# ============================================================

def classify_variable_type(col_name: str, series: pd.Series) -> str:
    """
    Classifies variables based on business semantics and data characteristics.
    
    Returns:
    - "binary": 0/1 binary variable
    - "categorical_code": Numerically coded categorical variable
    - "categorical_text": Textual categorical variable
    - "continuous": Truly continuous variable
    - "count": Count-based variable
    """
    
    # 1. Check business rules (Highest priority)
    if col_name in BINARY_VARS:
        return "binary"
    
    if col_name in CODED_CATEGORICAL_VARS:
        return "categorical_code"
    
    if col_name in TRUE_CONTINUOUS_VARS:
        return "continuous"
    
    if col_name in COUNT_LIKE_VARS:
        return "count"
    
    # 2. Data-driven inference for unknown variables
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical_text"
    
    s_numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(s_numeric) == 0:
        return "categorical_text"
    
    unique_vals = set(s_numeric.unique())
    n_unique = len(unique_vals)
    
    # Binary check
    if n_unique <= 2 and unique_vals.issubset({0, 1}):
        return "binary"
    
    # Low cardinality integers are likely coded categories
    if (pd.api.types.is_integer_dtype(series) or str(series.dtype).startswith("Int")) and n_unique <= 20:
        return "categorical_code"
    
    return "continuous"


def group_columns_by_type(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Groups DataFrame columns by their identified variable types."""
    if exclude_cols is None:
        exclude_cols = []
    
    groups = {
        "binary": [],
        "categorical_code": [],
        "categorical_text": [],
        "continuous": [],
        "count": []
    }
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        var_type = classify_variable_type(col, df[col])
        groups[var_type].append(col)
    
    return groups


# ============================================================
# 4) Missing Data Strategies
# ============================================================

@dataclass(frozen=True)
class MissingRule:
    strategy: str
    rationale: str


DEFAULT_MISSING_RULES: Dict[str, MissingRule] = {
    "binary": MissingRule(
        strategy="indicator+mode",
        rationale="Binary variable: use mode and indicator"
    ),
    "categorical_code": MissingRule(
        strategy="indicator+mode",
        rationale="Coded category: use mode and indicator"
    ),
    "categorical_text": MissingRule(
        strategy="indicator+unknown",
        rationale="Text category: fill with 'Unknown' and indicator"
    ),
    "continuous": MissingRule(
        strategy="indicator+median",
        rationale="Continuous variable: use median and indicator"
    ),
    "count": MissingRule(
        strategy="indicator+zero",
        rationale="Count variable: assume zero and indicator"
    ),
    "critical_academic": MissingRule(
        strategy="indicator+median",
        rationale="Academic signal: maintain indicator, use median imputation"
    ),
    "finance": MissingRule(
        strategy="indicator+assume_no",
        rationale="Financial variable: assume 'No' (0) and indicator"
    ),
    "parent": MissingRule(
        strategy="indicator+unknown",
        rationale="Parent background: fill with 'Unknown' and indicator"
    ),
    "macro": MissingRule(
        strategy="indicator+median",
        rationale="Macroeconomic context: use median and indicator"
    ),
}

# Category sets for rule buckets
CRITICAL_ACADEMIC_VARS = set(
    FEATURE_GROUPS["Academic signals (Sem 1)"] + FEATURE_GROUPS["Academic signals (Sem 2)"]
)
FINANCE_VARS = set(FEATURE_GROUPS["Financial / administrative"])
PARENT_VARS = set(FEATURE_GROUPS["Family background"])
DEMO_VARS = set(FEATURE_GROUPS["Demographics"])
MACRO_VARS = set(FEATURE_GROUPS["Macro context"])


def classify_rule_bucket(col: str) -> str:
    """Classifies a column into a specific rule bucket for missing data strategies."""
    if col in CRITICAL_ACADEMIC_VARS:
        return "critical_academic"
    if col in FINANCE_VARS:
        return "finance"
    if col in PARENT_VARS:
        return "parent"
    if col in DEMO_VARS:
        return "demographic"
    if col in CODED_CATEGORICAL_VARS:
        return "coded_categorical"
    if col in MACRO_VARS:
        return "macro"
    if col == "Target":
        return "outcome"
    return "continuous_general"


def get_missing_rule(col: str) -> MissingRule:
    """Retrieves the missing data handling rule based on the column bucket."""
    bucket = classify_rule_bucket(col)
    
    if bucket in DEFAULT_MISSING_RULES:
        return DEFAULT_MISSING_RULES[bucket]
    
    return DEFAULT_MISSING_RULES["continuous"]


# ============================================================
# 5) Integrity Checks
# ============================================================

@dataclass(frozen=True)
class AuditCheck:
    name: str
    severity: str
    columns: Tuple[str, ...]
    description: str


AUDIT_CHECKS: List[AuditCheck] = [
    AuditCheck(
        name="age_range",
        severity="WARN",
        columns=("Age at enrollment",),
        description="Flag unusually low or high enrollment ages"
    ),
    AuditCheck(
        name="approved_le_enrolled_sem1",
        severity="FAIL",
        columns=("Curricular units 1st sem (approved)", "Curricular units 1st sem (enrolled)"),
        description="Approved units must not exceed enrolled units in Semester 1"
    ),
    AuditCheck(
        name="approved_le_enrolled_sem2",
        severity="FAIL",
        columns=("Curricular units 2nd sem (approved)", "Curricular units 2nd sem (enrolled)"),
        description="Approved units must not exceed enrolled units in Semester 2"
    ),
    AuditCheck(
        name="tuition_vs_scholarship",
        severity="WARN",
        columns=("Tuition fees up to date", "Scholarship holder"),
        description="Flag cases where tuition is not up to date for scholarship holders"
    ),
]


# ============================================================
# 6) Outlier Handling Strategies
# ============================================================

@dataclass(frozen=True)
class OutlierPolicy:
    handling: str  # "flag_only" | "winsorize"
    rationale: str


OUTLIER_POLICIES: Dict[str, OutlierPolicy] = {
    "Age at enrollment": OutlierPolicy(
        handling="flag_only",
        rationale="High age represents non-traditional students (Risk Signal)"
    ),
    "Previous qualification (grade)": OutlierPolicy(
        handling="flag_only",
        rationale="Low prior grades indicate preparedness gaps (Risk Signal)"
    ),
    "Admission grade": OutlierPolicy(
        handling="flag_only",
        rationale="Extremes in admission grades serve as stratification signals"
    ),
    "GDP": OutlierPolicy(
        handling="winsorize",
        rationale="Macroeconomic extremes are external noise"
    ),
    "Inflation rate": OutlierPolicy(
        handling="winsorize",
        rationale="Macroeconomic extremes are external noise"
    ),
    "Unemployment rate": OutlierPolicy(
        handling="winsorize",
        rationale="Macroeconomic extremes are external noise"
    ),
}


def get_outlier_policy(col: str) -> OutlierPolicy:
    """Retrieves the outlier handling policy for a specific column."""
    return OUTLIER_POLICIES.get(
        col,
        OutlierPolicy(
            handling="flag_only",
            rationale="Default: Flag for monitoring without modification"
        )
    )

# ============================================================
# Value Mappings for Professional Visualization
# ============================================================

# Shared mapping for parental education levels
EDUCATION_MAPPING = {
    1: "Secondary Education", 2: "Bachelor's Degree", 3: "Degree", 4: "Master's", 
    5: "Doctorate", 6: "Freq. Higher Ed", 9: "12th Year (Incomplete)", 
    10: "11th Year (Incomplete)", 11: "7th Year (Old)", 12: "Other (11th Year)", 
    14: "10th Year", 18: "General Commerce", 19: "Basic Ed 3rd Cycle", 
    22: "Tech-Professional", 26: "7th Year", 27: "2nd Cycle Gen High School", 
    29: "9th Year (Incomplete)", 30: "8th Year", 34: "Unknown", 35: "Illiterate", 
    36: "Can Read/No Schooling", 37: "Basic Ed 1st Cycle", 38: "Basic Ed 2nd Cycle", 
    39: "Tech Specialization", 40: "Degree (1st Cycle)", 41: "Specialized Higher Studies", 
    42: "Prof. Higher Tech", 43: "Master (2nd Cycle)", 44: "Doctorate (3rd cycle)"
}

# Shared mapping for parental occupations
OCCUPATION_MAPPING = {
    0: "Student", 1: "Directors/Execs", 2: "Scientists/Intellectuals", 
    3: "Intermediate Techs", 4: "Admin Staff", 5: "Service/Security/Sales", 
    6: "Agri/Fishery Workers", 7: "Industry/Construction", 8: "Machine Operators", 
    9: "Unskilled Workers", 10: "Armed Forces", 90: "Other Situation", 99: "Blank", 
    122: "Health Professionals", 123: "Teachers", 125: "ICT Specialists", 
    131: "Science/Eng Techs", 132: "Health Techs", 134: "Legal/Social/Cultural Techs", 
    141: "Office/Secretaries", 143: "Data/Finance Operators", 144: "Other Admin Support", 
    151: "Personal Service", 152: "Sellers", 153: "Personal Care", 
    171: "Skilled Construction", 173: "Precision/Arts Workers", 175: "Food/Wood/Clothing", 
    191: "Cleaning Workers", 192: "Unskilled Agri", 193: "Unskilled Industry", 194: "Meal Prep Assistants"
}

VALUE_MAPPINGS: Dict[str, Dict[int, str]] = {
    "Marital status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"},
    
    "Application mode": {
        1: "1st Phase (General)", 2: "Ord. 612/93", 5: "1st Phase (Azores)", 7: "Holders Other Higher", 
        10: "Ord. 854-B/99", 15: "International Student", 16: "1st Phase (Madeira)", 17: "2nd Phase (General)", 
        18: "3rd Phase (General)", 26: "Ord. 533-A/99 (Diff Plan)", 27: "Ord. 533-A/99 (Other Inst)", 
        39: "Over 23 Years Old", 42: "Transfer", 43: "Change of Course", 44: "Tech Diploma", 
        51: "Change Inst/Course", 53: "Short Cycle Diploma", 57: "Change Inst/Course (Intl)"
    },

    "Course": {
        33: "Biofuel Production", 171: "Animation/Multimedia", 8014: "Social Service (Eve)", 9003: "Agronomy", 
        9070: "Comm. Design", 9085: "Vet Nursing", 9119: "Informatics Eng.", 9130: "Equinculture", 
        9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 
        9556: "Oral Hygiene", 9670: "Ad/Marketing Mgmt", 9773: "Journalism", 9853: "Basic Ed", 
        9991: "Management (Eve)"
    },

    "Daytime/evening attendance": {1: "Daytime", 0: "Evening"},
    
    "Previous qualification": EDUCATION_MAPPING,
    "Nacionality": {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English", 
        17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 
        26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian", 100: "Moldova", 
        101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban", 109: "Colombian"
    },

    "Mother's qualification": EDUCATION_MAPPING,
    "Father's qualification": EDUCATION_MAPPING,
    "Mother's occupation": OCCUPATION_MAPPING,
    "Father's occupation": OCCUPATION_MAPPING,

    "Displaced": {1: "Yes", 0: "No"},
    "Educational special needs": {1: "Yes", 0: "No"},
    "Debtor": {1: "Yes", 0: "No"},
    "Tuition fees up to date": {1: "Yes", 0: "No"},
    "Gender": {1: "Male", 0: "Female"},
    "Scholarship holder": {1: "Yes", 0: "No"},
    "International": {1: "Yes", 0: "No"}
}