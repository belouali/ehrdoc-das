#!/usr/bin/env python
"""Documentation-dependent model M2 = the paper's model M1 + a Charlson comorbidity index.

The Charlson index (original 1987 weights, without the age part, which M1 already has)
is built from each dataset's own documentation:
  eICU      ICD codes in the diagnosis table (icd9code field, ICD-9 and ICD-10) with the
            mimic-code Quan rules, OR the matching pastHistory picklist entry
  MIMIC-IV  mimic-code derived.charlson for the index admission (billing ICD codes)
The Quan rules are read from mimic-code's charlson.sql (path below) and run unchanged
on the eICU codes.

Outputs (row-level, external drive): eicu_m2_features.parquet, mimic_m2_features.parquet.
Aggregate: results/charlson_checks.csv, results/m2_internal_eicu.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      python scripts/11_charlson_features.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import re
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_model import C_M1, design_columns  # noqa: E402
from p0_model import make_pipeline, metrics  # noqa: E402

# charlson.sql from the MIT-LCP mimic-code repository (concepts/comorbidity/charlson.sql); set MIMIC_CHARLSON_SQL to its path
QUAN_SQL = Path(os.environ.get("MIMIC_CHARLSON_SQL", "charlson.sql"))
COMPONENTS = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease", "cerebrovascular_disease",
              "dementia", "chronic_pulmonary_disease", "rheumatic_disease", "peptic_ulcer_disease", "mild_liver_disease",
              "diabetes_without_cc", "diabetes_with_cc", "paraplegia", "renal_disease", "malignant_cancer",
              "severe_liver_disease", "metastatic_solid_tumor", "aids"]
# eICU pastHistory picklist path -> Charlson component (case-insensitive regex on pasthistorypath)
PASTHISTORY = {
    "myocardial_infarct": r"/myocardial infarction/",
    "congestive_heart_failure": r"/congestive heart failure/",
    "peripheral_vascular_disease": r"/peripheral vascular disease/",
    "cerebrovascular_disease": r"/strokes/|/tia\(s\)/",
    "dementia": r"/dementia/",
    "chronic_pulmonary_disease": r"/copd/|/asthma/|/restrictive disease/",
    "rheumatic_disease": r"/rheumatic/",
    "peptic_ulcer_disease": r"/peptic ulcer disease/",
    "mild_liver_disease": r"/cirrhosis/(clinical diagnosis|biopsy proven)",
    "severe_liver_disease": r"/cirrhosis/(varices|ascites|encephalopathy|ugi bleeding|coma|jaundice)",
    "diabetes_without_cc": r"diabetes/",
    "paraplegia": r"plegia",
    "renal_disease": r"/renal failure/|/renal insufficiency/|/s/p renal transplant/",
    "malignant_cancer": r"/cancer/cancer-primary site/|/hematologic malignancy/",
    "metastatic_solid_tumor": r"/cancer/metastases/",
    "aids": r"/aids/",
}
ICD9_SHAPE = re.compile(r"^(\d{3}|V\d{2}|E\d{3})(\.|$)")


def charlson_score(df: pd.DataFrame) -> pd.Series:
    """Original Charlson weights as in mimic-code, without the age score."""
    c = df[COMPONENTS].fillna(0).astype(int)
    return (c.myocardial_infarct + c.congestive_heart_failure + c.peripheral_vascular_disease
            + c.cerebrovascular_disease + c.dementia + c.chronic_pulmonary_disease + c.rheumatic_disease
            + c.peptic_ulcer_disease + np.maximum(c.mild_liver_disease, 3 * c.severe_liver_disease)
            + np.maximum(2 * c.diabetes_with_cc, c.diabetes_without_cc)
            + np.maximum(2 * c.malignant_cancer, 6 * c.metastatic_solid_tumor)
            + 2 * c.paraplegia + 2 * c.renal_disease + 6 * c.aids)


def quan_case_block() -> str:
    """The component CASE expressions of mimic-code's charlson.sql (between the com CTE's SELECT and FROM)."""
    sql = QUAN_SQL.read_text()
    start = sql.index("SELECT\n        ad.hadm_id") + len("SELECT\n        ad.hadm_id")
    end = sql.index("FROM `physionet-data.mimic_core.admissions` ad\n    LEFT JOIN diag")
    return sql[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    con = duckdb.connect()

    # ---- eICU: codes with the Quan rules
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    con.register("ecoh", e[["patientunitstayid"]])
    raw = con.execute(f"""SELECT patientunitstayid, icd9code FROM '{E}/diagnosis.parquet'
                          WHERE icd9code IS NOT NULL AND patientunitstayid IN (SELECT patientunitstayid FROM ecoh)""").df()
    rows = []
    for sid, field in zip(raw.patientunitstayid, raw.icd9code):
        for code in str(field).split(","):
            code = code.strip().upper()
            if code:
                v9 = bool(ICD9_SHAPE.match(code))
                c = code.replace(".", "")
                rows.append((sid, c if v9 else None, None if v9 else c))
    diag = pd.DataFrame(rows, columns=["hadm_id", "icd9_code", "icd10_code"]).drop_duplicates()
    con.register("diag", diag)
    coded = con.execute(f"SELECT hadm_id {quan_case_block()} FROM diag GROUP BY hadm_id").df()
    coded = coded.rename(columns={"hadm_id": "patientunitstayid"}).set_index("patientunitstayid")

    # ---- eICU: pastHistory picklist
    ph = con.execute(f"""SELECT DISTINCT patientunitstayid, lower(pasthistorypath) p FROM '{E}/pastHistory.parquet'
                         WHERE patientunitstayid IN (SELECT patientunitstayid FROM ecoh)""").df()
    paths = pd.Series(ph.p.unique())
    hist = pd.DataFrame(index=e.patientunitstayid.unique())
    for comp, rx in PASTHISTORY.items():
        hit_paths = set(paths[paths.str.contains(rx, regex=True)])
        hist[comp] = hist.index.isin(set(ph.loc[ph.p.isin(hit_paths), "patientunitstayid"]))
    ec = pd.DataFrame(index=e.patientunitstayid)
    for comp in COMPONENTS:
        a = coded[comp].reindex(ec.index).fillna(0).astype(bool) if comp in coded else False
        b = hist[comp].reindex(ec.index).fillna(False).astype(bool) if comp in hist else False
        ec[comp] = (a | b).astype(int)
        ec[f"{comp}__codes"] = coded[comp].reindex(ec.index).fillna(0).astype(int) if comp in coded else 0
        ec[f"{comp}__history"] = hist[comp].reindex(ec.index).fillna(False).astype(int) if comp in hist else 0
    e["charlson"] = charlson_score(ec.reset_index(drop=True)).to_numpy()

    # ---- MIMIC-IV: mimic-code derived charlson, index admission
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet")
    ch = con.execute(f"SELECT * FROM '{M}/derived/charlson.parquet'").df()
    mc = m[["hadm_id"]].merge(ch, on="hadm_id", how="left")
    m["charlson"] = charlson_score(mc).to_numpy()

    checks = []
    for name, df_c, sc in [("eICU", ec, e.charlson), ("MIMIC-IV", mc, m.charlson)]:
        checks.append(dict(check=f"{name}: stays", value=len(sc)))
        checks.append(dict(check=f"{name}: Charlson mean (no age)", value=float(sc.mean())))
        checks.append(dict(check=f"{name}: Charlson median", value=float(sc.median())))
        checks.append(dict(check=f"{name}: share Charlson 0", value=float((sc == 0).mean())))
        for comp in COMPONENTS:
            checks.append(dict(check=f"{name}: share {comp}", value=float(df_c[comp].fillna(0).astype(bool).mean())))
    for comp in COMPONENTS:
        checks.append(dict(check=f"eICU codes only: share {comp}", value=float(ec[f"{comp}__codes"].mean())))
        checks.append(dict(check=f"eICU pastHistory only: share {comp}", value=float(ec[f"{comp}__history"].mean())))
    checks = pd.DataFrame(checks)
    checks.round(4).to_csv(res / "charlson_checks.csv", index=False)

    cols1 = design_columns()
    cols2 = cols1 + ["charlson"]
    e.to_parquet(rowlevel / "eicu_m2_features.parquet", index=False)
    m.to_parquet(rowlevel / "mimic_m2_features.parquet", index=False)

    # eICU internal: M1 against M2, same hospital-grouped folds
    y = e.death.to_numpy(int)
    folds = list(GroupKFold(n_splits=5).split(e, y, e.hospitalid))
    perf = []
    for name, cols in [("M1", cols1), ("M2 = M1 + Charlson", cols2)]:
        X = e[cols].to_numpy(float)
        p = np.zeros(len(e))
        for tr, te in folds:
            p[te] = make_pipeline(C_M1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        perf.append(dict(model=name, **metrics(y, p)))
    perf = pd.DataFrame(perf)
    perf.round(4).to_csv(res / "m2_internal_eicu.csv", index=False)
    pd.set_option("display.width", 200)
    print(perf.round(4).to_string(index=False))
    print(checks.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
