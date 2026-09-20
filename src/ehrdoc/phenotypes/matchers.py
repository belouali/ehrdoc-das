from __future__ import annotations
import re
import pandas as pd
from .definitions import TextRule

def series_contains_any(s: pd.Series, terms, case=False) -> pd.Series:
    if not terms:
        return pd.Series([False]*len(s), index=s.index)
    pattern = "|".join(re.escape(t) for t in terms)
    return s.astype(str).str.contains(pattern, case=case, na=False)

def apply_text_rule(df: pd.DataFrame, col: str, rule: TextRule, case: bool=False) -> pd.Series:
    mask = series_contains_any(df[col], rule.include, case=case)
    if rule.exclude:
        mask &= ~series_contains_any(df[col], rule.exclude, case=case)
    return mask

def apply_icd_regex(df: pd.DataFrame, col: str, icd_regex: str) -> pd.Series:
    return df[col].astype(str).str.contains(icd_regex, case=False, na=False)
