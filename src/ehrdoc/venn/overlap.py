from __future__ import annotations
import pandas as pd

def overlap_summary(df: pd.DataFrame, patient_col: str, dx_col: str, hx_col: str, med_col: str) -> dict:
    """Return raw counts for 7 lobes + total evidence patients."""
    d = df[df[dx_col]].set_index(patient_col).index.unique()
    h = df[df[hx_col]].set_index(patient_col).index.unique()
    m = df[df[med_col]].set_index(patient_col).index.unique()

    D=set(d); H=set(h); M=set(m)
    all_any = len(D|H|M)
    # counts by lobe
    out = {
        "Dx∩Hx∩Med": len(D & H & M),
        "Dx∩Hx": len((D & H) - M),
        "Dx∩Med": len((D & M) - H),
        "Dx only": len(D - H - M),
        "Hx∩Med": len((H & M) - D),
        "Hx only": len(H - D - M),
        "Med only": len(M - D - H),
        "Total_any": all_any,
    }
    return out
