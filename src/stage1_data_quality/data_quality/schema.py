"""Schema Management Module"""
import pandas as pd
import numpy as np

def infer_schema(df: pd.DataFrame) -> dict:
    return {c: str(df[c].dtype) for c in df.columns}

def coerce_to_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()
    
    for c in schema:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in schema]
    if extra:
        df = df.drop(columns=extra)
    
    for c, dt in schema.items():
        if c not in df.columns:
            continue
        if c == "Target":
            continue
        if dt.startswith("int") or dt.startswith("Int"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif dt.startswith("float"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype(str)
    return df

def dtype_summary(df: pd.DataFrame):
    dtypes = pd.Series({c: str(df[c].dtype) for c in df.columns})
    counts = dtypes.value_counts().reset_index()
    counts.columns = ["dtype", "n_cols"]
    cols_by_type = {dt: dtypes[dtypes == dt].index.tolist() for dt in counts["dtype"]}
    return counts, cols_by_type
