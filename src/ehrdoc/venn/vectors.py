from __future__ import annotations
import numpy as np
import pandas as pd
from .lobes import encode_lobe, LOBE_LABELS

def hospital_venn_counts(
    flags_df: pd.DataFrame,
    hospital_col: str = "hospitalid",
    patient_col: str = "patientunitstayid",
    dx_col: str = "exists_in_diagnosis",
    hx_col: str = "exists_in_pastHistory",
    med_col: str = "exists_in_medication",
    min_patients_per_hospital: int | None = None,
) -> pd.DataFrame:
    """Compute 7-lobe counts per hospital for a condition."""
    df = flags_df[[hospital_col, patient_col, dx_col, hx_col, med_col]].drop_duplicates()
    df["lobe"] = [encode_lobe(d,h,m) for d,h,m in zip(df[dx_col], df[hx_col], df[med_col])]
    df = df[df["lobe"]>=0]

    counts = df.groupby([hospital_col, "lobe"])[patient_col].nunique().unstack(fill_value=0)
    # ensure all 7 columns exist
    for i in range(7):
        if i not in counts.columns:
            counts[i] = 0
    counts = counts[sorted(counts.columns)]
    counts.columns = LOBE_LABELS

    if min_patients_per_hospital is not None:
        totals = counts.sum(axis=1)
        counts = counts.loc[totals >= min_patients_per_hospital]
    return counts.reset_index()

def hospital_venn_vectors(counts_df: pd.DataFrame, hospital_col: str="hospitalid") -> pd.DataFrame:
    """Normalize 7-lobe counts to probabilities."""
    lobes = [c for c in counts_df.columns if c != hospital_col]
    mat = counts_df[lobes].to_numpy(dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    probs = np.divide(mat, row_sums, where=row_sums>0)
    out = counts_df[[hospital_col]].copy()
    for j,lab in enumerate(lobes):
        out[lab] = probs[:,j]
    return out


SOURCE_LOBES = {
    "Dx": [l for l in LOBE_LABELS if "Dx" in l],
    "Hx": [l for l in LOBE_LABELS if "Hx" in l],
    "Med": [l for l in LOBE_LABELS if "Med" in l],
}


def source_shares(counts_df: pd.DataFrame, hospital_col: str = "hospitalid") -> pd.DataFrame:
    """Share of a hospital's condition patients with evidence in each source."""
    total = counts_df[LOBE_LABELS].sum(axis=1)
    out = counts_df[[hospital_col]].copy()
    for src, cols in SOURCE_LOBES.items():
        out[f"share_{src}"] = counts_df[cols].sum(axis=1) / total.where(total > 0, np.nan)
    return out


def zero_source_hospitals(counts_df: pd.DataFrame, hospital_col: str = "hospitalid") -> dict[str, list[int]]:
    """Hospitals where an entire evidence source is empty (zero patients).

    In eICU a block of hospitals contributes no medication table at all. Such
    a hospital sits at ~90 degrees from every hospital that does, for every
    condition, so it inflates cross-condition concordance and dominates the
    'dissimilar' end of any DAS ranking. Callers use this to run sensitivity
    analyses that exclude these data-availability strata.
    """
    shares = source_shares(counts_df, hospital_col)
    return {src: sorted(int(h) for h in shares.loc[shares[f"share_{src}"] == 0, hospital_col])
            for src in SOURCE_LOBES}
