from __future__ import annotations
import pandas as pd
from .definitions import PhenotypeRule
from .matchers import apply_text_rule, apply_icd_regex

def build_condition_flags(
    patient_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    past_history_df: pd.DataFrame,
    medication_df: pd.DataFrame,
    rule: PhenotypeRule,
    patient_id_col: str = "patientunitstayid",
    diag_text_col: str = "diagnosisstring",
    diag_icd_col: str = "icd9code",
    hx_text_col: str = "pasthistoryvalue",
    med_text_col: str = "drugname",
) -> pd.DataFrame:
    """Return patient-level boolean evidence flags for a condition.

    Output columns: patientunitstayid, exists_in_diagnosis, exists_in_pastHistory, exists_in_medication.
    """
    diag_mask = apply_text_rule(diagnosis_df, diag_text_col, rule.diagnosis, case=False)
    if rule.diagnosis.icd_regex and diag_icd_col in diagnosis_df.columns:
        diag_mask |= apply_icd_regex(diagnosis_df, diag_icd_col, rule.diagnosis.icd_regex)

    hx_mask = apply_text_rule(past_history_df, hx_text_col, rule.past_history, case=False)
    med_mask = apply_text_rule(medication_df, med_text_col, rule.medication, case=False)

    diag_ids = set(diagnosis_df.loc[diag_mask, patient_id_col].dropna().unique())
    hx_ids = set(past_history_df.loc[hx_mask, patient_id_col].dropna().unique())
    med_ids = set(medication_df.loc[med_mask, patient_id_col].dropna().unique())

    out = patient_df[[patient_id_col]].drop_duplicates().copy()
    out["exists_in_diagnosis"] = out[patient_id_col].isin(diag_ids)
    out["exists_in_pastHistory"] = out[patient_id_col].isin(hx_ids)
    out["exists_in_medication"] = out[patient_id_col].isin(med_ids)
    return out
