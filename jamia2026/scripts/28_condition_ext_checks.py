#!/usr/bin/env python
"""A_construct_condition_ext part 2, steps 0-1: eICU flags for the added conditions and the
outcome-free checks (coverage and B1 split-half noise). No outcome is read.

Settings: configs/screen_settings.yaml (A_construct_condition_ext) and configs/phenotypes_ext.yaml,
both approved 2026-11-07 before computing.

Cohort as scripts/01_build_flags.py (adults alive in the ICU at 24 h, first stay per patient).
Sources: D diagnosisstring, H pasthistorypath, A admissionDrug drugname, M medication drugname
(case-insensitive substring, exclude terms remove a match).
Angles: DHM lobes (D, H, M) for cancer, COPD, CKD, cirrhosis; D and H for stroke; for obesity the
BMI-anchored D-U profile among stays with BMI >= 30 (lobe D coded, lobe U not coded).
Checks per condition:
  coverage  hospitals with >= 100 evidence-positive records: >= 60 of the 82 targets and >= 100 hospitals
  B1        share of hospital pairs beyond the 95th percentile of the split-half within-hospital angle >= 0.50

Outputs: rowlevel/eicu_flags_ext.parquet (row level, stays on the external drive);
results/condition_ext_checks.csv (aggregate).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml \
      python scripts/28_condition_ext_checks.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "01_build_flags.py")
s02 = _load("s02", "02_das_external.py")

CONDITIONS = ["cancer", "obesity", "copd", "ckd", "cirrhosis", "stroke"]
MIN_RECORDS, MIN_TARGETS, MIN_HOSPITALS, B1_BAR = 100, 60, 100, 0.50
SEED = 20261120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    ph = yaml.safe_load(open(base / "configs" / "phenotypes_ext.yaml"))
    assert ph["status"] == "locked"

    con = duckdb.connect()
    lm = 60 * cfg["cohort"]["landmark_hours"]
    con.execute(f"""
        CREATE OR REPLACE TABLE ecoh AS
        WITH p AS (
          SELECT patientunitstayid, patienthealthsystemstayid, uniquepid, hospitalid,
                 CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_n,
                 unitdischargeoffset, hospitaldischargeyear, unitvisitnumber,
                 admissionheight h, admissionweight w
          FROM '{E}/patient.parquet'),
        q AS (SELECT * FROM p WHERE age_n >= {cfg['cohort']['min_age']} AND unitdischargeoffset > {lm})
        SELECT patientunitstayid, uniquepid, hospitalid, h, w FROM (
          SELECT *, row_number() OVER (PARTITION BY uniquepid
                   ORDER BY hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber) rn FROM q)
        WHERE rn = 1""")
    coh = con.execute("SELECT * FROM ecoh").df()
    ref = pd.read_parquet(rowlevel / "eicu_flags.parquet", columns=["patientunitstayid"])
    assert set(coh.patientunitstayid) == set(ref.patientunitstayid), "cohort differs from eicu_flags"
    print(f"eICU cohort: {len(coh):,} stays, {coh.hospitalid.nunique()} hospitals")

    rules = {"D": ("diagnosis.parquet", "diagnosisstring", "diagnosis"),
             "H": ("pastHistory.parquet", "pasthistorypath", "past_history"),
             "A": ("admissionDrug.parquet", "drugname", "medication"),
             "M": ("medication.parquet", "drugname", "medication")}
    for src, (table, col, key) in rules.items():
        con.execute(f"""CREATE OR REPLACE TEMP TABLE src AS SELECT t.patientunitstayid id, t.{col} v
                        FROM '{E}/{table}' t JOIN ecoh USING (patientunitstayid) WHERE t.{col} IS NOT NULL""")
        vals = con.execute("SELECT DISTINCT v FROM src").df()
        for cond in CONDITIONS:
            rule = ph[cond].get(key)
            if not rule:
                coh[f"{cond}_{src}"] = False
                continue
            hit = vals.loc[s01.text_mask(vals["v"], rule), "v"]
            con.register("hit", pd.DataFrame({"v": hit}))
            ids = set(con.execute("SELECT DISTINCT id FROM src JOIN hit USING (v)").df()["id"])
            con.unregister("hit")
            coh[f"{cond}_{src}"] = coh.patientunitstayid.isin(ids)
        print(f"  {src}: " + ", ".join(f"{c} {coh[f'{c}_{src}'].mean():.1%}" for c in CONDITIONS), flush=True)
    ok = coh.h.between(120, 220) & coh.w.between(30, 300)
    coh["bmi"] = (coh.w / (coh.h / 100) ** 2).where(ok)
    coh["obesity_U"] = (coh.bmi >= 30) & ~coh.obesity_D
    coh.drop(columns=["h", "w"]).to_parquet(rowlevel / "eicu_flags_ext.parquet", index=False)

    split = pd.read_csv(res / "target_split.csv")
    targets = set(split.hospitalid)
    rng = np.random.default_rng(SEED)
    rows = []
    for cond in CONDITIONS:
        if cond == "obesity":
            e = coh[coh.bmi >= 30].reset_index(drop=True)
            codes = np.where(e.obesity_D, 0, 1)  # lobe 0 = D coded, lobe 1 = U not coded
            n_lobes, construct = 2, "D-U (BMI >= 30)"
        else:
            srcs = ("D", "H") if cond == "stroke" else ("D", "H", "M")
            e = coh
            codes = s02.lobe_codes(e, cond, srcs)
            n_lobes, construct = 2 ** len(srcs) - 1, "".join(srcs)
        cnt = s02.hospital_counts(e, codes, n_lobes)
        cnt = cnt[cnt.sum(axis=1) >= MIN_RECORDS]
        hs = cnt.index.to_numpy()
        n_t = len(set(hs) & targets)
        row = dict(condition=cond, construct=construct, stays_with_evidence=int((codes >= 0).sum()),
                   share_D=coh[f"{cond}_D"].mean(), share_H=coh[f"{cond}_H"].mean(),
                   share_A=coh[f"{cond}_A"].mean(), share_M=coh[f"{cond}_M"].mean(),
                   hospitals=len(hs), targets=n_t, coverage_pass=(n_t >= MIN_TARGETS) and (len(hs) >= MIN_HOSPITALS))
        if len(hs) >= 10:
            pooled = cnt.sum().to_numpy(float)
            row |= s02.noise_floor(e, codes, hs, n_lobes, pooled / pooled.sum(), rng)
        row["B1_pass"] = row.get("share_pairs_beyond_within95", 0) >= B1_BAR
        row["enters_step2"] = row["coverage_pass"] and row["B1_pass"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(res / "condition_ext_checks.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
