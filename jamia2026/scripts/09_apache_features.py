#!/usr/bin/env python
"""Inputs of the paper's model (M1) in eICU and MIMIC-IV, on the stage 1 cohort.

eICU: the official APACHE IV score (apachePatientResult, version IVa) and the
patient table's age, gender, ethnicity, unit stay type and admission source.
Stays without a valid score or with a missing input are dropped, as in the paper.

MIMIC-IV: an APACHE IV score approximated as
    acute physiology  mimic-code derived.apsiii (APACHE III physiology weights,
                      which APACHE IV reuses)
  + age points        APACHE III/IV table
  + chronic health    APACHE III points from index-admission ICD codes
                      (not for elective surgical admissions)
The age and chronic-health rules are checked first on eICU, where
apachescore - acutephysiologyscore must equal them.

Covariate mappings for MIMIC-IV:
  ethnicity        race grouped into the eICU categories
  unit stay type   admit; transfer if an earlier ICU stay of the same admission
                   ended right before (care unit before is an ICU); readmit if an
                   earlier ICU stay of the same admission exists otherwise
  admission source care unit just before the ICU stay (transfers table), with
                   PACU, and the discharge lounge for same-day surgery, as
                   Operating Room; no earlier unit: Other Hospital when admitted
                   by transfer from a hospital, else Direct Admit

Outputs (row-level, external drive): eicu_m1_features.parquet, mimic_m1_features.parquet.
Aggregate: results/apache_checks.csv, results/m1_internal_eicu.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      python scripts/09_apache_features.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_model import C_M1, CATS, design, design_columns  # noqa: E402
from p0_model import make_pipeline, metrics  # noqa: E402

CHRONIC_POINTS = {"aids": 23, "hepaticfailure": 16, "lymphoma": 13, "metastaticcancer": 11,
                  "leukemia": 10, "immunosuppression": 10, "cirrhosis": 4}
# ICD codes without dots; ICD-10 then ICD-9 prefixes
CHRONIC_ICD = {
    "aids": (r"^(B20|B21|B22|B23|B24)", r"^(042|043|044)"),
    "hepaticfailure": (r"^(K72|K767)", r"^(570|5722|5724)"),
    "lymphoma": (r"^(C81|C82|C83|C84|C85|C86|C88)", r"^(200|201|202)"),
    "metastaticcancer": (r"^(C77|C78|C79|C800)", r"^(196|197|198|1990)"),
    "leukemia": (r"^(C90|C91|C92|C93|C94|C95)", r"^(203|204|205|206|207|208)"),
    "immunosuppression": (r"^(D80|D81|D82|D83|D84|Z7952|Z9221|Z94|T86)", r"^(279|V5865|V8741|V42|9968)"),
    "cirrhosis": (r"^(K703|K743|K744|K745|K746)", r"^(5712|5715|5716)"),
}
ICU_UNITS = {"Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)",
             "Medical/Surgical Intensive Care Unit (MICU/SICU)", "Cardiac Vascular Intensive Care Unit (CVICU)",
             "Coronary Care Unit (CCU)", "Trauma SICU (TSICU)", "Neuro Surgical Intensive Care Unit (Neuro SICU)",
             "Neuro Intermediate", "Neuro Stepdown"}


def age_points(age: pd.Series) -> pd.Series:
    bins = [-np.inf, 44, 59, 64, 69, 74, 84, np.inf]
    return pd.cut(age, bins=bins, labels=[0, 5, 11, 13, 16, 17, 24]).astype(float)


def chronic_points(flags: pd.DataFrame, rule: str) -> pd.Series:
    pts = pd.DataFrame({k: flags[k].astype(float) * v for k, v in CHRONIC_POINTS.items()})
    return pts.sum(axis=1) if rule == "sum" else pts.max(axis=1)


def race_group(r: str) -> str:
    r = (r or "").upper()
    if r.startswith("WHITE") or r == "PORTUGUESE":
        return "Caucasian"
    if r.startswith("BLACK"):
        return "African American"
    if r.startswith("HISPANIC") or r == "SOUTH AMERICAN":
        return "Hispanic"
    if r.startswith("ASIAN"):
        return "Asian"
    if r.startswith("AMERICAN INDIAN"):
        return "Native American"
    return "Other/Unknown"


def admit_source(unit, adm_type: str, adm_loc: str) -> str:
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return "Other Hospital" if adm_loc == "TRANSFER FROM HOSPITAL" else "Direct Admit"
    if unit.startswith("Emergency Department"):
        return "Emergency Department"
    if unit == "PACU":
        return "Operating Room"
    if unit == "Discharge Lounge":
        return "Operating Room" if adm_type in ("SURGICAL SAME DAY ADMISSION", "ELECTIVE") else "Floor"
    if unit in ICU_UNITS:
        return "Other ICU"
    if "Intermediate" in unit or "Stepdown" in unit:
        return "Step-Down Unit (SDU)"
    return "Floor"


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
    checks = []

    # ---- eICU
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet", columns=["patientunitstayid", "hospitalid"])
    con.register("coh", flags)
    e = con.execute(f"""
        SELECT c.patientunitstayid, c.hospitalid, r.apachescore::DOUBLE apachescore,
               r.acutephysiologyscore::DOUBLE aps,
               CASE WHEN p.age = '> 89' THEN 90 ELSE TRY_CAST(p.age AS INTEGER) END agenum,
               p.gender, p.ethnicity, p.unitstaytype, p.unitadmitsource,
               CASE p.hospitaldischargestatus WHEN 'Expired' THEN 1 WHEN 'Alive' THEN 0 END death,
               v.aids, v.hepaticfailure, v.lymphoma, v.metastaticcancer, v.leukemia, v.immunosuppression,
               v.cirrhosis, v.electivesurgery
        FROM coh c JOIN '{E}/patient.parquet' p USING (patientunitstayid)
        JOIN '{E}/apachePatientResult.parquet' r USING (patientunitstayid)
        LEFT JOIN '{E}/apachePredVar.parquet' v USING (patientunitstayid)
        WHERE r.apacheversion = 'IVa' AND r.apachescore > 0 AND r.acutephysiologyscore > 0""").df()
    n0 = len(e)
    e = e.dropna(subset=["apachescore", "agenum", "death"] + list(CATS)).reset_index(drop=True)
    print(f"eICU: {n0:,} stays with an APACHE IVa score, {len(e):,} with every model input")

    # check the age and chronic-health rules against the official score
    resid = e.apachescore - e.aps
    fl = e[list(CHRONIC_POINTS)].fillna(0)
    nonelective = (e.electivesurgery.fillna(0) != 1)
    for rule in ["sum", "max"]:
        pred = age_points(e.agenum) + chronic_points(fl, rule) * nonelective
        checks.append(dict(check=f"eICU: score - APS equals age + chronic points ({rule} rule)",
                           value=float(np.mean(np.isclose(pred, resid)))))
    checks.append(dict(check="eICU: score - APS equals age points alone", value=float(np.mean(np.isclose(age_points(e.agenum), resid)))))
    best = max(["sum", "max"], key=lambda r: [c["value"] for c in checks if r in c["check"]][0])
    print(pd.DataFrame(checks).round(4).to_string(index=False))

    # ---- MIMIC-IV
    coh = pd.read_parquet(rowlevel / "mimic_flags.parquet", columns=["subject_id", "hadm_id", "stay_id"])
    con.register("mcoh", coh)
    m = con.execute(f"""
        WITH i AS (SELECT c.*, x.intime FROM mcoh c JOIN '{M}/icu/icustays.parquet' x USING (stay_id)),
        prev AS (SELECT i.stay_id, tr.careunit,
                        row_number() OVER (PARTITION BY i.stay_id ORDER BY tr.intime DESC) rn
                 FROM i JOIN '{M}/hosp/transfers.parquet' tr
                   ON tr.hadm_id = i.hadm_id AND tr.intime < i.intime - INTERVAL 1 MINUTE AND tr.eventtype <> 'discharge')
        SELECT i.stay_id, i.hadm_id, a.apsiii::DOUBLE aps, ad.race, ad.admission_type, ad.admission_location,
               p.gender, p.anchor_age + (year(i.intime) - p.anchor_year) agenum, pv.careunit prev_unit,
               EXISTS (SELECT 1 FROM '{M}/icu/icustays.parquet' b WHERE b.hadm_id = i.hadm_id AND b.intime < i.intime) earlier_icu
        FROM i JOIN '{M}/derived/apsiii.parquet' a USING (stay_id)
        JOIN '{M}/hosp/admissions.parquet' ad ON ad.hadm_id = i.hadm_id
        JOIN '{M}/hosp/patients.parquet' p ON p.subject_id = i.subject_id
        LEFT JOIN (SELECT * FROM prev WHERE rn = 1) pv USING (stay_id)""").df()
    dx = con.execute(f"""SELECT d.hadm_id, d.icd_code, d.icd_version FROM '{M}/hosp/diagnoses_icd.parquet' d
                         WHERE d.hadm_id IN (SELECT hadm_id FROM mcoh)""").df()
    dx["code"] = dx.icd_code.str.replace(".", "", regex=False).str.upper().str.strip()
    for k, (r10, r9) in CHRONIC_ICD.items():
        hit = np.where(dx.icd_version == 10, dx.code.str.match(r10), dx.code.str.match(r9))
        m[k] = m.hadm_id.isin(set(dx.loc[hit, "hadm_id"]))
    elective = m.admission_type.isin(["ELECTIVE", "SURGICAL SAME DAY ADMISSION"])
    m["apachescore"] = m.aps + age_points(m.agenum) + chronic_points(m[list(CHRONIC_POINTS)], best) * (~elective)
    m["gender"] = m.gender.map({"M": "Male", "F": "Female"})
    m["ethnicity"] = m.race.map(race_group)
    m["unitadmitsource"] = [admit_source(u, t, l) for u, t, l in zip(m.prev_unit, m.admission_type, m.admission_location)]
    m["unitstaytype"] = np.where(~m.earlier_icu, "admit", np.where(m.prev_unit.isin(ICU_UNITS), "transfer", "readmit"))
    print(f"MIMIC-IV: {len(m):,} of {len(coh):,} cohort stays with an APS III; chronic-health rule '{best}'")

    for name, d in [("eICU", e), ("MIMIC-IV", m)]:
        checks.append(dict(check=f"{name}: stays", value=len(d)))
        for col in ["aps", "apachescore", "agenum"]:
            checks.append(dict(check=f"{name}: median {col}", value=float(d[col].median())))
            checks.append(dict(check=f"{name}: IQR {col}",
                               value=float(d[col].quantile(0.75) - d[col].quantile(0.25))))
        for k in CHRONIC_POINTS:
            checks.append(dict(check=f"{name}: share {k}", value=float(d[k].fillna(0).astype(bool).mean())))
        for c in ["unitadmitsource", "unitstaytype", "ethnicity"]:
            for v, s in d[c].value_counts(normalize=True).items():
                checks.append(dict(check=f"{name}: {c} = {v}", value=float(s)))
    pd.DataFrame(checks).round(4).to_csv(res / "apache_checks.csv", index=False)

    keep_e = ["patientunitstayid", "hospitalid", "death"]
    pd.concat([e[keep_e], design(e)], axis=1).to_parquet(rowlevel / "eicu_m1_features.parquet", index=False)
    pd.concat([m[["stay_id", "hadm_id"]], design(m)], axis=1).to_parquet(rowlevel / "mimic_m1_features.parquet", index=False)

    # eICU internal performance of M1 on this cohort, hospital-grouped 5-fold CV
    cols = design_columns()
    X, y = design(e)[cols].to_numpy(float), e.death.to_numpy(int)
    p = np.zeros(len(e))
    for tr, te in GroupKFold(n_splits=5).split(X, y, e.hospitalid):
        p[te] = make_pipeline(C_M1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    perf = pd.DataFrame([dict(model="M1 (paper's model), eICU hospital-grouped CV", **metrics(y, p))])
    perf.round(4).to_csv(res / "m1_internal_eicu.csv", index=False)
    print(perf.round(4).to_string(index=False))
    print(pd.DataFrame(checks).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
