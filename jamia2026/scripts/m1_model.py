"""The eICU paper's mortality model (M1, ehrdoc_variation_repo/scripts/09) for the external test.

  actual hospital mortality ~ APACHE IV score + age + gender + ethnicity
                              + unit stay type + unit admission source

The paper scales the two numeric inputs, one-hot encodes the four categorical
ones and fits an unpenalized logistic regression (C = 1e6). Here the one-hot
columns are built with fixed categories (the eICU values), so eICU and MIMIC-IV
share one design matrix; the stage 4 pipeline then scales every column, which
does not change an unpenalized fit.
"""

from __future__ import annotations

import pandas as pd

C_M1 = 1e6
NUMERIC = ["apachescore", "agenum"]
CATS = {
    "gender": ["Male", "Female", "Unknown", "Other"],
    "ethnicity": ["Caucasian", "African American", "Other/Unknown", "Hispanic", "Asian", "Native American"],
    "unitstaytype": ["admit", "transfer", "stepdown/other", "readmit"],
    "unitadmitsource": ["Emergency Department", "Operating Room", "Floor", "Direct Admit", "Recovery Room",
                        "Other Hospital", "Acute Care/Floor", "Step-Down Unit (SDU)", "ICU to SDU", "Other ICU",
                        "PACU", "ICU", "Chest Pain Center", "Observation", "Other"],
}


def design_columns() -> list[str]:
    return NUMERIC + [f"{c}={v}" for c, vals in CATS.items() for v in vals]


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric inputs plus one-hot columns for the fixed categories."""
    out = df[NUMERIC].astype(float).copy()
    for c, vals in CATS.items():
        unknown = set(df[c].dropna().unique()) - set(vals)
        assert not unknown, f"{c}: values outside the eICU categories: {sorted(unknown)}"
        for v in vals:
            out[f"{c}={v}"] = (df[c] == v).astype(float)
    return out
