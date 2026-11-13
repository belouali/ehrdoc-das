#!/usr/bin/env python
"""First-24-hour condition flags for eICU (site-selection rerun --spec v3, 2026-11-10).

Same cohort and matching rules as 01_build_flags.py (diagnosis D, past history H, in-stay medication M), restricted
to records available by the 24-h landmark: diagnosis by diagnosisoffset, past history by pasthistoryenteredoffset,
medication by drugorderoffset, each <= 1440 minutes from unit admission (records before admission included).
Used only as prediction-model inputs. The documentation vectors (DAS) still use the whole-stay flags.
MIMIC-IV has no times for its diagnosis codes or discharge-summary past history, so no MIMIC-IV counterpart.

Writes <rowlevel>/eicu_flags_24h.parquet (patientunitstayid + 12 flags) and prints prevalence (aggregates only).
Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml python scripts/46_flags_24h.py
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
ph = yaml.safe_load(open(HERE.parent / cfg["paths"]["phenotypes"]))
E, RL = cfg["paths"]["eicu"], Path(cfg["paths"]["rowlevel"])
LM = 60 * cfg["cohort"]["landmark_hours"]

con = duckdb.connect()
coh = pd.read_parquet(RL / "eicu_flags.parquet", columns=["patientunitstayid", "hospitalid"])
con.register("cohdf", coh)
con.execute("CREATE TABLE ecoh AS SELECT patientunitstayid FROM cohdf")
sources = {
    "D": (f"SELECT d.patientunitstayid id, d.diagnosisstring v FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid) WHERE d.diagnosisoffset <= {LM}",
          lambda c, df: s01.text_mask(df["v"], ph[c]["diagnosis"])),
    "H": (f"SELECT h.patientunitstayid id, h.pasthistoryvalue v FROM '{E}/pastHistory.parquet' h JOIN ecoh USING (patientunitstayid) WHERE h.pasthistoryenteredoffset <= {LM}",
          lambda c, df: s01.text_mask(df["v"], ph[c]["past_history"])),
    "M": (f"SELECT m.patientunitstayid id, m.drugname v FROM '{E}/medication.parquet' m JOIN ecoh USING (patientunitstayid) WHERE m.drugorderoffset <= {LM}",
          lambda c, df: s01.contains_any(df["v"], ph[c]["medication"]["include"])),
}
out = coh[["patientunitstayid"]].copy()
for src, (sql, matcher) in sources.items():
    ids, _ = s01.flag_ids(con, sql, "id", "v", matcher)
    for c in s01.CONDITIONS:
        out[f"{c}_{src}"] = out.patientunitstayid.isin(ids[c])
out.to_parquet(RL / "eicu_flags_24h.parquet", index=False)
whole = pd.read_parquet(RL / "eicu_flags.parquet")
cols = [c for c in out.columns if c != "patientunitstayid"]
print(pd.DataFrame({"first_24h": out[cols].mean(), "whole_stay": whole[cols].mean()}).round(3).to_string())
