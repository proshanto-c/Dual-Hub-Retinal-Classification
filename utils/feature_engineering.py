import re
from collections import Counter
from typing import List, Optional, Sequence
import pandas as pd

from gnn.patientgraphmodel import DemographicSpec

DEFAULT_DEMOGRAPHIC_CANDIDATES = [
    "patient_age", "age", "patient_sex", "sex", 
    "diabetes_time_y", "insuline",
    "exam_eye",
]

NUMERIC_HINTS = ("age", "time", "years", "duration", "count", "score", "num")

def infer_demographic_columns(
    frame: pd.DataFrame, 
    requested: Optional[List[str]], 
    label_columns: Sequence[str]
) -> List[str]:
    if requested:
        return requested
    excluded = {"image_id", "image_path", "image_name", "patient_id", "patient", "split", *label_columns}
    candidates_pool = list(DEFAULT_DEMOGRAPHIC_CANDIDATES) + [
        c for c in frame.columns if c.startswith("comorbidity_") and c not in DEFAULT_DEMOGRAPHIC_CANDIDATES
    ]
    candidates = [c for c in candidates_pool if c in frame.columns and c not in excluded]
    if candidates:
        return candidates
    raise ValueError("No demographic columns found.")

def fit_demographic_specs(frame: pd.DataFrame, demographic_columns: Sequence[str]) -> List[DemographicSpec]:
    specs: List[DemographicSpec] = []
    for col in demographic_columns:
        series = frame[col]
        numeric_mask = pd.to_numeric(series, errors="coerce").notna()
        is_numeric = numeric_mask.mean() >= 0.8 or any(h in col.lower() for h in NUMERIC_HINTS)
        
        if is_numeric:
            numeric = pd.to_numeric(series, errors="coerce")
            mean = float(numeric.mean()) if numeric.notna().any() else 0.0
            std = float(numeric.std(ddof=0)) if numeric.notna().any() else 1.0
            specs.append(DemographicSpec(name=col, kind="numeric", mean=mean, std=max(std, 1e-6)))
        else:
            vocab = {"__UNK__": 0}
            values = [str(v) for v in series.dropna().astype(str).unique().tolist()]
            for i, v in enumerate(sorted(values), start=1):
                vocab[v] = i
            specs.append(DemographicSpec(name=col, kind="categorical", vocab=vocab))
    return specs

def preprocess_comorbidities(df: pd.DataFrame, valid_comorb: Optional[List[str]] = None, top_k: int = 20):
    if "comorbidities" not in df.columns:
        return df, []
        
    df = df.copy()
    
    if valid_comorb is None:
        all_conditions = []
        raw = df["comorbidities"].dropna().astype(str).tolist()
        for c in raw:
            parts = re.split(r',|\sand\s', c.lower())
            for p in parts:
                p = p.strip()
                if p and p != '0':
                    all_conditions.append(p)
        counts = Counter(all_conditions)
        valid_comorb = [item[0] for item in counts.most_common(top_k)]
        
    for condition in valid_comorb:
        col_name = f"comorbidity_{condition.replace(' ', '_')}"
        def has_condition(val):
            if pd.isna(val): return 0.0
            parts = [p.strip() for p in re.split(r',|\sand\s', str(val).lower())]
            return 1.0 if condition in parts else 0.0
        df[col_name] = df["comorbidities"].apply(has_condition)
        
    return df, valid_comorb
