#!/usr/bin/env python
"""Whole-stay and first-24-hour flags for COPD and hypothyroidism in eICU (exploratory screen, 2026-11-11).

Rules: configs/phenotypes_extra.yaml. Matching, cohort and time windows as 01_build_flags.py and 46_flags_24h.py.
Writes <rowlevel>/eicu_flags_extra.parquet (whole stay) and eicu_flags_extra_24h.parquet. Prints prevalence only.
"""
import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s01", HERE / "01_build_flags.py")
s01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s01)
cfg = yaml.safe_load(open(HERE.parent / "configs" / "external.yaml"))
ph = yaml.safe_load(open(HERE.parent / "configs" / "phenotypes_extra.yaml"))
s01.CONDITIONS = list(ph)
E, RL = cfg["paths"]["eicu"], Path(cfg["paths"]["rowlevel"])
LM = 60 * cfg["cohort"]["landmark_hours"]

con = duckdb.connect()
coh = pd.read_parquet(RL / "eicu_flags.parquet", columns=["patientunitstayid", "hospitalid"])
con.register("cohdf", coh)
con.execute("CREATE TABLE ecoh AS SELECT patientunitstayid FROM cohdf")
TIME = {"D": "d.diagnosisoffset", "H": "h.pasthistoryenteredoffset", "M": "m.drugorderoffset"}
prev = {}
for window, fname in (("whole", "eicu_flags_extra.parquet"), ("24h", "eicu_flags_extra_24h.parquet")):
    w = lambda s: f" WHERE {TIME[s]} <= {LM}" if window == "24h" else ""
    sources = {
        "D": (f"SELECT d.patientunitstayid id, d.diagnosisstring v FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid){w('D')}",
              lambda c, df: s01.text_mask(df["v"], ph[c]["diagnosis"])),
        "H": (f"SELECT h.patientunitstayid id, h.pasthistoryvalue v FROM '{E}/pastHistory.parquet' h JOIN ecoh USING (patientunitstayid){w('H')}",
              lambda c, df: s01.text_mask(df["v"], ph[c]["past_history"])),
        "M": (f"SELECT m.patientunitstayid id, m.drugname v FROM '{E}/medication.parquet' m JOIN ecoh USING (patientunitstayid){w('M')}",
              lambda c, df: s01.contains_any(df["v"], ph[c]["medication"]["include"])),
    }
    out = coh.copy()
    for src, (sql, matcher) in sources.items():
        ids, _ = s01.flag_ids(con, sql, "id", "v", matcher)
        for c in ph:
            out[f"{c}_{src}"] = out.patientunitstayid.isin(ids[c])
    out.to_parquet(RL / fname, index=False)
    prev[window] = out.drop(columns=["patientunitstayid", "hospitalid"]).mean()
print(pd.DataFrame(prev).round(3).to_string())
