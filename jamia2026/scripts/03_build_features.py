#!/usr/bin/env python
"""P0 first-day predictors for eICU and MIMIC-IV (stage 3).

Same cohort as 01_build_flags.py. For each vital: hourly medians over
[0, 24) h, then min, max and median of those hours. For each lab: min and max
over [-6, 24) h. GCS: minimum. Age and sex. Values outside plausible ranges
are dropped. Missing values are left missing (imputation is fitted later on
eICU training data only).

The eICU file also carries the outcome (hospitaldischargestatus). The MIMIC-IV
file carries no outcome: MIMIC-IV mortality is read only by the stage 4 script.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml \
      python scripts/03_build_features.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def plaus_case(plaus: dict) -> str:
    """SQL CASE that keeps a value only inside its variable's plausible range."""
    whens = " ".join(f"WHEN var = '{v}' AND value BETWEEN {lo} AND {hi} THEN value" for v, (lo, hi) in plaus.items())
    return f"CASE {whens} ELSE NULL END"


def summarize(con, long_sql: str, fcfg: dict, id_col: str) -> pd.DataFrame:
    """long_sql -> (id, var, hr, value); hourly medians for vitals, then per-stay summaries."""
    vitals = ", ".join(f"'{v}'" for v in fcfg["vitals"])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE lng AS
        SELECT id, var, hr, {plaus_case(fcfg['plausible'])} AS value FROM ({long_sql})""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hourly AS
        SELECT id, var, floor(hr) AS h, median(value) AS value FROM lng
        WHERE value IS NOT NULL AND var IN ({vitals}) GROUP BY ALL""")
    v = con.execute("""SELECT id, var, min(value) mn, max(value) mx, median(value) md, count(*) n_hours
                       FROM hourly GROUP BY ALL""").df()
    l = con.execute(f"""SELECT id, var, min(value) mn, max(value) mx FROM lng
                        WHERE value IS NOT NULL AND var NOT IN ({vitals}) GROUP BY ALL""").df()
    wide = []
    for var in fcfg["vitals"]:
        s = v[v["var"] == var].set_index("id")
        cols = {"mn": f"{var}_min"} if var == "gcs" else {"mn": f"{var}_min", "mx": f"{var}_max", "md": f"{var}_median"}
        wide.append(s[list(cols)].rename(columns=cols))
        wide.append(s[["n_hours"]].rename(columns={"n_hours": f"{var}_hours"}))
    for var in fcfg["labs"]:
        s = l[l["var"] == var].set_index("id")
        wide.append(s[["mn", "mx"]].rename(columns={"mn": f"{var}_min", "mx": f"{var}_max"}))
    out = pd.concat(wide, axis=1)
    out.index.name = id_col
    return out.reset_index()


def eicu(con, E: str, fcfg: dict, rowlevel: Path) -> pd.DataFrame:
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet", columns=["patientunitstayid", "hospitalid"])
    con.register("coh_df", flags)
    con.execute("CREATE OR REPLACE TEMP TABLE coh AS SELECT * FROM coh_df")
    v0, v1 = fcfg["vital_window"]
    l0, l1 = fcfg["lab_window"]
    win = lambda col, a, b: f"{col} >= {60 * a} AND {col} < {60 * b}"
    labs = " ".join(f"WHEN labname = '{n}' THEN '{k}'" for k, n in fcfg["eicu_labs"].items())
    labnames = ", ".join(f"'{n}'" for n in fcfg["eicu_labs"].values())
    vp = f"'{E}/vitalPeriodic.parquet'"
    long_sql = f"""
      SELECT patientunitstayid id, 'hr' var, observationoffset / 60.0 hr, heartrate::DOUBLE value FROM {vp}
        WHERE {win('observationoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid, 'rr', observationoffset / 60.0, respiration::DOUBLE FROM {vp}
        WHERE {win('observationoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid, 'spo2', observationoffset / 60.0, sao2::DOUBLE FROM {vp}
        WHERE {win('observationoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid, 'temp', observationoffset / 60.0, temperature::DOUBLE FROM {vp}
        WHERE {win('observationoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid, 'map', observationoffset / 60.0, systemicmean::DOUBLE FROM {vp}
        WHERE {win('observationoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid, 'map', observationoffset / 60.0, noninvasivemean FROM '{E}/vitalAperiodic.parquet'
        WHERE {win('observationoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid,
             CASE WHEN nursingchartcelltypevalname = 'GCS Total' THEN 'gcs' ELSE 'temp' END,
             nursingchartoffset / 60.0, TRY_CAST(nursingchartvalue AS DOUBLE)
        FROM '{E}/nurseCharting.parquet'
        WHERE nursingchartcelltypevalname IN ('GCS Total', 'Temperature (C)')
          AND {win('nursingchartoffset', v0, v1)} AND patientunitstayid IN (SELECT patientunitstayid FROM coh)
      UNION ALL SELECT patientunitstayid, CASE {labs} END, labresultoffset / 60.0, labresult::DOUBLE
        FROM '{E}/lab.parquet'
        WHERE labname IN ({labnames}) AND {win('labresultoffset', l0, l1)}
          AND patientunitstayid IN (SELECT patientunitstayid FROM coh)"""
    feats = summarize(con, long_sql, fcfg, "patientunitstayid")
    demo = con.execute(f"""
        SELECT p.patientunitstayid, CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END age,
               CASE gender WHEN 'Male' THEN 1 WHEN 'Female' THEN 0 END male,
               CASE hospitaldischargestatus WHEN 'Expired' THEN 1 WHEN 'Alive' THEN 0 END death
        FROM '{E}/patient.parquet' p JOIN coh USING (patientunitstayid)""").df()
    return flags.merge(demo, on="patientunitstayid").merge(feats, on="patientunitstayid", how="left")


def mimic(con, M: str, fcfg: dict, rowlevel: Path) -> pd.DataFrame:
    coh = pd.read_parquet(rowlevel / "mimic_flags.parquet", columns=["subject_id", "hadm_id", "stay_id"])
    con.register("coh_df", coh)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE coh AS
        SELECT c.*, i.intime FROM coh_df c JOIN '{M}/icu/icustays.parquet' i USING (stay_id)""")
    v0, v1 = fcfg["vital_window"]
    l0, l1 = fcfg["lab_window"]
    mv, ml = fcfg["mimic_vitals"], fcfg["mimic_labs"]
    ids = lambda xs: ", ".join(str(x) for x in xs)
    vit = {k: v for k, v in mv.items() if k in ("hr", "map", "rr", "spo2")}
    vcase = " ".join(f"WHEN itemid IN ({ids(v)}) THEN '{k}'" for k, v in vit.items())
    lcase = " ".join(f"WHEN itemid IN ({ids(v)}) THEN '{k}'" for k, v in ml.items())
    all_v = [x for v in vit.values() for x in v] + mv["temp_c"] + mv["temp_f"]
    all_l = [x for v in ml.values() for x in v]
    hrs = "date_diff('second', c.intime, x.charttime) / 3600.0"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ce AS
        SELECT x.stay_id id, x.itemid, {hrs} hr, x.charttime, x.valuenum FROM '{M}/icu/chartevents.parquet' x
        JOIN coh c USING (stay_id)
        WHERE x.itemid IN ({ids(all_v + mv['gcs_parts'])}) AND x.valuenum IS NOT NULL
          AND {hrs} >= {v0} AND {hrs} < {v1}""")
    long_sql = f"""
      SELECT id, CASE {vcase} WHEN itemid IN ({ids(mv['temp_c'] + mv['temp_f'])}) THEN 'temp' END var, hr,
             CASE WHEN itemid IN ({ids(mv['temp_f'])}) THEN (valuenum - 32) * 5 / 9 ELSE valuenum END value
        FROM ce WHERE itemid IN ({ids(all_v)})
      UNION ALL
      SELECT id, 'gcs', min(hr), sum(valuenum) FROM ce WHERE itemid IN ({ids(mv['gcs_parts'])})
        GROUP BY id, charttime HAVING count(DISTINCT itemid) = 3
      UNION ALL
      SELECT c.stay_id, CASE {lcase} END, {hrs}, x.valuenum FROM '{M}/hosp/labevents.parquet' x
        JOIN coh c ON c.hadm_id = x.hadm_id
        WHERE x.itemid IN ({ids(all_l)}) AND x.valuenum IS NOT NULL AND {hrs} >= {l0} AND {hrs} < {l1}"""
    feats = summarize(con, long_sql, fcfg, "stay_id")
    demo = con.execute(f"""
        SELECT c.stay_id, p.anchor_age + (year(c.intime) - p.anchor_year) age,
               CASE p.gender WHEN 'M' THEN 1 WHEN 'F' THEN 0 END male
        FROM coh c JOIN '{M}/hosp/patients.parquet' p USING (subject_id)""").df()
    return coh.merge(demo, on="stay_id").merge(feats, on="stay_id", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rowlevel = Path(cfg["paths"]["rowlevel"])
    con = duckdb.connect()
    con.execute("SET memory_limit = '12GB'")
    e = eicu(con, cfg["paths"]["eicu"], cfg["features"], rowlevel)
    e.to_parquet(rowlevel / "eicu_features.parquet", index=False)
    print(f"eICU features: {e.shape}")
    m = mimic(con, cfg["paths"]["mimic"], cfg["features"], rowlevel)
    m.to_parquet(rowlevel / "mimic_features.parquet", index=False)
    print(f"MIMIC-IV features: {m.shape}")


if __name__ == "__main__":
    main()
