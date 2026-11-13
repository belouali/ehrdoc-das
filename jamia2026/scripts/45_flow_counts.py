#!/usr/bin/env python
"""Cohort flow counts for the eICU cohort and the mean-DAS range between hospitals with and without medication records.

Aggregates only. Writes results/cohort_flow_eicu.csv (step, stays, hospitals) and results/nomed_das_range.csv.
Steps mirror 01_build_flags.py (age, 24-h landmark, first qualifying stay per patient), 09_apache_features.py
(APACHE IVa score, every model input) and 38_iecv.py (hospitals with >= 100 stays, DAS vector for >= 1 condition).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml python scripts/45_flow_counts.py
"""
import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s02", HERE / "02_das_external.py")
s02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s02)
cfg = yaml.safe_load(open(HERE.parent / "configs" / "external.yaml"))
E, RL, RES = cfg["paths"]["eicu"], Path(cfg["paths"]["rowlevel"]), HERE.parent / cfg["paths"]["results"]
LM = 60 * cfg["cohort"]["landmark_hours"]
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]

con = duckdb.connect()
con.execute(f"""CREATE TABLE p AS SELECT patientunitstayid, uniquepid, hospitalid, unitdischargeoffset,
                CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END age_n FROM '{E}/patient.parquet'""")
q = lambda w: con.execute(f"SELECT count(*), count(DISTINCT hospitalid) FROM p WHERE {w}").fetchone()
rows = [("All ICU stays in eICU", *q("TRUE")),
        ("Age 18 or older", *q("age_n >= 18")),
        ("Alive and in the ICU at 24 h", *q(f"age_n >= 18 AND unitdischargeoffset > {LM}"))]
flags = pd.read_parquet(RL / "eicu_flags.parquet")
rows.append(("First qualifying stay per patient", len(flags), flags.hospitalid.nunique()))
apache = con.execute(f"""SELECT count(*), count(DISTINCT f.hospitalid) FROM flags f
                         JOIN '{E}/apachePatientResult.parquet' r USING (patientunitstayid)
                         WHERE r.apacheversion = 'IVa' AND r.apachescore > 0 AND r.acutephysiologyscore > 0""").fetchone()
rows.append(("APACHE IVa score", *apache))
m1 = pd.read_parquet(RL / "eicu_m1_features.parquet", columns=["patientunitstayid", "hospitalid"])
rows.append(("Every model input", len(m1), m1.hospitalid.nunique()))
size = m1.hospitalid.value_counts()
big = size[size >= 100].index
rows.append(("Hospitals with at least 100 stays", int(size[big].sum()), len(big)))

# documentation vectors as in 38_iecv.py
V = {}
for c in CONDITIONS:
    cnt = s02.hospital_counts(flags, s02.lobe_codes(flags, c, ("D", "H", "M")), 7)
    cnt = cnt[cnt.sum(axis=1) >= cfg["das"]["min_records_per_hospital"]]
    V[c] = cnt.div(cnt.sum(axis=1), axis=0)
das_h = sorted(set(big) & set().union(*[set(V[c].index) for c in CONDITIONS]))
rows.append(("With a DAS vector for at least one condition", int(size[das_h].sum()), len(das_h)))
flow = pd.DataFrame(rows, columns=["step", "stays", "hospitals"])
flow.to_csv(RES / "cohort_flow_eicu.csv", index=False)
print(flow.to_string(index=False))

# mean DAS between no-medication and medication hospitals (over the conditions both have)
m1f = m1.merge(flags[["patientunitstayid", "any_M"]], on="patientunitstayid")
nomed = set(m1f[m1f.hospitalid.isin(das_h)].groupby("hospitalid").any_M.max().loc[lambda s: s == 0].index)
ang = []
for a in nomed:
    for b in set(das_h) - nomed:
        v = [s02.angles(V[c].loc[[b]].to_numpy(float), V[c].loc[a].to_numpy(float))[0]
             for c in CONDITIONS if a in V[c].index and b in V[c].index]
        if v:
            ang.append(np.mean(v))
ang = np.array(ang)
out = pd.DataFrame([dict(nomed_hospitals=len(nomed), pairs=len(ang), min=ang.min(), q05=np.percentile(ang, 5),
                         median=np.median(ang), max=ang.max())]).round(1)
out.to_csv(RES / "nomed_das_range.csv", index=False)
print(out.to_string(index=False))
