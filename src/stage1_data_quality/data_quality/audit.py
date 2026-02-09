"""Data Integrity Audit Module"""
import pandas as pd
from .rules import AUDIT_CHECKS

def run_integrity_audit(df: pd.DataFrame, min_age: int = 16, max_age: int = 80) -> pd.DataFrame:
    issues = []

    def add(check_name, severity, mask, details):
        n = int(mask.sum())
        if n > 0:
            issues.append({
                "check_name": check_name,
                "severity": severity,
                "affected_rows": n,
                "details": details
            })

    for chk in AUDIT_CHECKS:
        cols = chk.columns
        if not all(c in df.columns for c in cols):
            continue

        if chk.name == "age_range":
            age = pd.to_numeric(df["Age at enrollment"], errors="coerce")
            add("age_too_low", "WARN", age < min_age, f"Age < {min_age}")
            add("age_too_high", "WARN", age > max_age, f"Age > {max_age}")

        elif chk.name == "approved_le_enrolled_sem1":
            A = pd.to_numeric(df["Curricular units 1st sem (approved)"], errors="coerce")
            E = pd.to_numeric(df["Curricular units 1st sem (enrolled)"], errors="coerce")
            add(chk.name, chk.severity, (A > E) & A.notna() & E.notna(), chk.description)

        elif chk.name == "approved_le_enrolled_sem2":
            A = pd.to_numeric(df["Curricular units 2nd sem (approved)"], errors="coerce")
            E = pd.to_numeric(df["Curricular units 2nd sem (enrolled)"], errors="coerce")
            add(chk.name, chk.severity, (A > E) & A.notna() & E.notna(), chk.description)
        
        elif chk.name == "tuition_vs_scholarship":
            tuition = pd.to_numeric(df["Tuition fees up to date"], errors="coerce")
            schol = pd.to_numeric(df["Scholarship holder"], errors="coerce")
            mask = (tuition == 0) & (schol == 1) & tuition.notna() & schol.notna()
            add("tuition_vs_scholarship_review", "INFO", mask, 
                "Scholarship=1 but tuition=0. Review: partial scholarship or payment timing?")
    
    # Macro index check
    if "GDP" in df.columns:
        gdp = pd.to_numeric(df["GDP"], errors="coerce")
        add("gdp_negative", "WARN", gdp < 0, "GDP < 0 (data quality issue)")
    
    if "Inflation rate" in df.columns:
        inf = pd.to_numeric(df["Inflation rate"], errors="coerce")
        add("inflation_negative", "WARN", inf < 0, "Inflation rate < 0 (deflation or data issue)")
    
    if "Unemployment rate" in df.columns:
        unemp = pd.to_numeric(df["Unemployment rate"], errors="coerce")
        add("unemployment_negative", "WARN", unemp < 0, "Unemployment rate < 0 (data quality issue)")
        add("unemployment_extreme", "WARN", unemp > 50, "Unemployment rate > 50% (check data)")

    return pd.DataFrame(issues) if issues else pd.DataFrame(columns=["check_name", "severity", "affected_rows", "details"])
