#!/usr/bin/env python
"""Harmonized evidence flags for eICU 2.0 and MIMIC-IV 2.2 (external DAS, stage 2).

Cohort in both datasets: adults, alive and still in the ICU at 24 h, first
qualifying ICU stay per patient.

Sources per condition (columns <condition>_<source>):
  D     diagnosis. eICU: published text rule on diagnosisstring.
                   MIMIC-IV: ICD codes in diagnoses_icd for the index admission.
  Dicd  eICU only: the same ICD rules applied to the icd9code column (sensitivity).
  H     eICU only: published text rule on pastHistory (original construct).
  A     home medication. eICU: admissionDrug. MIMIC-IV: ED medrecon for the index admission.
  M     in-stay medication. eICU: medication. MIMIC-IV: prescriptions overlapping the ICU stay.
  Aspec, Mspec  A and M with the condition-specific medication classes
                (phenotypes_specific_meds.yaml, the paper's sensitivity rules).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml \
      python scripts/01_build_flags.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]


def contains_any(values: pd.Series, terms) -> pd.Series:
    """Same semantics as ehrdoc.phenotypes.matchers.series_contains_any."""
    if not terms:
        return pd.Series(False, index=values.index)
    pattern = "|".join(re.escape(t) for t in terms)
    return values.astype(str).str.contains(pattern, case=False, na=False)


def text_mask(values: pd.Series, rule: dict) -> pd.Series:
    mask = contains_any(values, rule.get("include", []))
    if rule.get("exclude"):
        mask &= ~contains_any(values, rule["exclude"])
    return mask


def icd_match(code: str, version: int, rule: dict) -> bool:
    code = code.replace(".", "").strip().upper()
    if version == 10:
        if rule.get("icd10_exclude") and re.search(rule["icd10_exclude"], code):
            return False
        return bool(re.search(rule["icd10"], code))
    return bool(re.search(rule["icd9"], code))


ICD9_SHAPE = re.compile(r"^(\d{3}|V\d{2}|E\d{3})(\.|$)")


def eicu_icd_match(field: str, rule: dict) -> bool:
    """eICU icd9code holds 'ICD-9, ICD-10' pairs; decide the version from the dotted shape."""
    for code in str(field).split(","):
        code = code.strip().upper()
        if not code:
            continue
        version = 9 if ICD9_SHAPE.match(code) else 10
        if icd_match(code, version, rule):
            return True
    return False


def flag_ids(con, sql: str, id_col: str, value_col: str, matcher) -> dict[str, set]:
    """Run sql -> (id, value); match distinct values once per condition; return ids per condition."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE src AS {sql}")
    vals = con.execute(f"SELECT DISTINCT {value_col} AS v FROM src WHERE {value_col} IS NOT NULL").df()
    out = {}
    for cond in CONDITIONS:
        hit = vals.loc[matcher(cond, vals), "v"]
        con.register("hit", pd.DataFrame({"v": hit}))
        ids = con.execute(f"SELECT DISTINCT {id_col} FROM src JOIN hit ON src.{value_col} = hit.v").df()[id_col]
        out[cond] = set(ids)
        con.unregister("hit")
    n_ids = con.execute(f"SELECT count(DISTINCT {id_col}) FROM src").fetchone()[0]
    return out, n_ids


def add_specific(sources: dict, sp: dict) -> None:
    """Reuse the A and M queries with the condition-specific medication lists."""
    for src in ["A", "M"]:
        sources[f"{src}spec"] = (sources[src][0],
                                 lambda c, df: contains_any(df["v"], sp[c]["medication"]["include"]))


def build_eicu(con, E: str, ph: dict, sp: dict, icd: dict, min_age: int, landmark_min: int) -> pd.DataFrame:
    con.execute(f"""
        CREATE OR REPLACE TABLE ecoh AS
        WITH p AS (
          SELECT patientunitstayid, patienthealthsystemstayid, uniquepid, hospitalid,
                 CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_n,
                 unitdischargeoffset, hospitaldischargeyear, unitvisitnumber
          FROM '{E}/patient.parquet'),
        q AS (SELECT * FROM p WHERE age_n >= {min_age} AND unitdischargeoffset > {landmark_min})
        SELECT patientunitstayid, uniquepid, hospitalid FROM (
          SELECT *, row_number() OVER (PARTITION BY uniquepid
                   ORDER BY hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber) rn FROM q)
        WHERE rn = 1""")
    coh = con.execute("SELECT * FROM ecoh").df()
    print(f"eICU cohort: {len(coh):,} stays, {coh.hospitalid.nunique()} hospitals")

    sources = {
        "D": (f"SELECT d.patientunitstayid id, d.diagnosisstring v FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid)",
              lambda c, df: text_mask(df["v"], ph[c]["diagnosis"])),
        "Dicd": (f"SELECT d.patientunitstayid id, d.icd9code v FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid)",
                 lambda c, df: df["v"].map(lambda x: eicu_icd_match(x, icd[c]))),
        "H": (f"SELECT h.patientunitstayid id, h.pasthistoryvalue v FROM '{E}/pastHistory.parquet' h JOIN ecoh USING (patientunitstayid)",
              lambda c, df: text_mask(df["v"], ph[c]["past_history"])),
        "A": (f"SELECT a.patientunitstayid id, a.drugname v FROM '{E}/admissionDrug.parquet' a JOIN ecoh USING (patientunitstayid)",
              lambda c, df: contains_any(df["v"], ph[c]["medication"]["include"])),
        "M": (f"SELECT m.patientunitstayid id, m.drugname v FROM '{E}/medication.parquet' m JOIN ecoh USING (patientunitstayid)",
              lambda c, df: contains_any(df["v"], ph[c]["medication"]["include"])),
    }
    add_specific(sources, sp)
    for src, (sql, matcher) in sources.items():
        ids, n_any = flag_ids(con, sql, "id", "v", matcher)
        con.execute("CREATE OR REPLACE TEMP TABLE anyrow AS SELECT DISTINCT id FROM src")
        coh[f"any_{src}"] = coh.patientunitstayid.isin(set(con.execute("SELECT id FROM anyrow").df()["id"]))
        for cond in CONDITIONS:
            coh[f"{cond}_{src}"] = coh.patientunitstayid.isin(ids[cond])
        print(f"  eICU {src}: {n_any:,} cohort stays with any row; " +
              ", ".join(f"{c} {coh[f'{c}_{src}'].mean():.1%}" for c in CONDITIONS))
    return coh


def build_mimic(con, M: str, ph: dict, sp: dict, icd: dict, min_age: int, landmark_h: int) -> pd.DataFrame:
    con.execute(f"""
        CREATE OR REPLACE TABLE mcoh AS
        WITH s AS (
          SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, a.deathtime,
                 p.anchor_year_group, p.anchor_age + (year(i.intime) - p.anchor_year) AS age
          FROM '{M}/icu/icustays.parquet' i
          JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)
          JOIN '{M}/hosp/patients.parquet' p ON p.subject_id = i.subject_id),
        q AS (SELECT * FROM s WHERE age >= {min_age}
                AND outtime > intime + INTERVAL {landmark_h} HOUR
                AND (deathtime IS NULL OR deathtime > intime + INTERVAL {landmark_h} HOUR))
        SELECT subject_id, hadm_id, stay_id, intime, outtime, anchor_year_group FROM (
          SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY intime) rn FROM q)
        WHERE rn = 1""")
    coh = con.execute("SELECT subject_id, hadm_id, stay_id, anchor_year_group FROM mcoh").df()
    print(f"MIMIC-IV cohort: {len(coh):,} stays")

    sources = {
        "D": (f"""SELECT d.stay_id id, d.icd_code || '|' || d.icd_version v FROM (
                    SELECT c.stay_id, x.icd_code, x.icd_version FROM '{M}/hosp/diagnoses_icd.parquet' x
                    JOIN mcoh c USING (hadm_id)) d""",
              lambda c, df: df["v"].map(lambda s: icd_match(s.split("|")[0], int(s.split("|")[1]), icd[c]))),
        "A": (f"""SELECT c.stay_id id, m.name v FROM '{M}/ed/medrecon.parquet' m
                  JOIN '{M}/ed/edstays.parquet' e ON e.stay_id = m.stay_id
                  JOIN mcoh c ON c.hadm_id = e.hadm_id""",
              lambda c, df: contains_any(df["v"], ph[c]["medication"]["include"])),
        "M": (f"""SELECT c.stay_id id, r.drug v FROM '{M}/hosp/prescriptions.parquet' r
                  JOIN mcoh c ON c.hadm_id = r.hadm_id
                  WHERE r.starttime < c.outtime AND coalesce(r.stoptime, r.starttime) >= c.intime""",
              lambda c, df: contains_any(df["v"], ph[c]["medication"]["include"])),
    }
    add_specific(sources, sp)
    for src, (sql, matcher) in sources.items():
        ids, n_any = flag_ids(con, sql, "id", "v", matcher)
        coh[f"any_{src}"] = coh.stay_id.isin(set(con.execute("SELECT DISTINCT id FROM src").df()["id"]))
        for cond in CONDITIONS:
            coh[f"{cond}_{src}"] = coh.stay_id.isin(ids[cond])
        print(f"  MIMIC {src}: {n_any:,} cohort stays with any row; " +
              ", ".join(f"{c} {coh[f'{c}_{src}'].mean():.1%}" for c in CONDITIONS))
    return coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    ph = yaml.safe_load(open(base / cfg["paths"]["phenotypes"]))
    sp = yaml.safe_load(open(base / cfg["paths"]["phenotypes_specific"]))
    out = Path(cfg["paths"]["rowlevel"])
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    e = build_eicu(con, cfg["paths"]["eicu"], ph, sp, cfg["icd"], cfg["cohort"]["min_age"],
                   60 * cfg["cohort"]["landmark_hours"])
    e.to_parquet(out / "eicu_flags.parquet", index=False)
    m = build_mimic(con, cfg["paths"]["mimic"], ph, sp, cfg["icd"], cfg["cohort"]["min_age"],
                    cfg["cohort"]["landmark_hours"])
    m.to_parquet(out / "mimic_flags.parquet", index=False)
    print(f"Wrote {out / 'eicu_flags.parquet'} and {out / 'mimic_flags.parquet'}")


if __name__ == "__main__":
    main()
