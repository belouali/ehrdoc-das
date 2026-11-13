#!/usr/bin/env python
"""SOFA score over the last 24 h of each eICU ICU stay (site-selection --spec v9, 2026-11-11, author's request).

Window: unitdischargeoffset - 1440 to unitdischargeoffset (minutes from unit admission). Worst value per component.
  resp   PaO2 / FiO2: each lab paO2 over the most recent lab FiO2 in the window (21% if none). <400 1, <300 2,
         <200 3, <100 4. Ventilation is not required for 3 and 4 (not reliably recorded).
  coag   platelets x 1000: <150 1, <100 2, <50 3, <20 4.
  liver  total bilirubin (mg/dL): >=1.2 1, >=2 2, >=6 3, >=12 4.
  cv     norepinephrine, epinephrine, phenylephrine or vasopressin infusion 3; dopamine or dobutamine 2; else lowest
         mean arterial pressure (invasive or non-invasive) < 70 1. Doses are not used (rates are recorded unevenly).
  cns    lowest GCS Total (nurse charting): 13-14 1, 10-12 2, 6-9 3, <6 4.
  renal  highest creatinine (mg/dL): >=1.2 1, >=2 2, >=3.5 3, >=5 4. Urine output not used.
A component with no record in the window scores 0. Writes <rowlevel>/eicu_sofa_lastday.parquet and prints
aggregates only.
Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml python scripts/50_sofa_lastday.py
"""
from pathlib import Path

import duckdb
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load(open(HERE.parent / "configs" / "external.yaml"))
E, RL = cfg["paths"]["eicu"], Path(cfg["paths"]["rowlevel"])
con = duckdb.connect()
coh = pd.read_parquet(RL / "eicu_m1_features.parquet", columns=["patientunitstayid"])
con.register("cohdf", coh)
con.execute(f"""CREATE TABLE w AS SELECT c.patientunitstayid id, p.unitdischargeoffset - 1440 lo, p.unitdischargeoffset hi
                FROM cohdf c JOIN '{E}/patient.parquet' p USING (patientunitstayid)""")
J = "JOIN w ON t.patientunitstayid = w.id AND {o} BETWEEN w.lo AND w.hi"


def q(sql):
    return con.execute(sql).df()


lab = q(f"""SELECT w.id, lower(t.labname) n, t.labresult v, t.labresultoffset o FROM '{E}/lab.parquet' t
            {J.format(o='t.labresultoffset')}
            WHERE lower(t.labname) IN ('pao2', 'fio2', 'platelets x 1000', 'total bilirubin', 'creatinine')
              AND t.labresult IS NOT NULL""")
agg = lambda name, f: lab[lab.n == name].groupby("id").v.agg(f)
out = coh.rename(columns={"patientunitstayid": "id"}).set_index("id")
out["plt"], out["bili"], out["creat"] = agg("platelets x 1000", "min"), agg("total bilirubin", "max"), agg("creatinine", "max")
# PaO2/FiO2 with the most recent FiO2
pa = lab[lab.n == "pao2"].sort_values("o")
fi = lab[lab.n == "fio2"].sort_values("o")[["id", "o", "v"]].rename(columns={"v": "fio2"})
pf = pd.merge_asof(pa, fi, on="o", by="id", direction="backward")
pf["fio2"] = pf.fio2.where(pf.fio2.between(21, 100), 21) / 100
out["pf"] = (pf.v / pf.fio2).groupby(pf.id).min()
mapv = q(f"""SELECT w.id, min(m) mapmin FROM (
               SELECT patientunitstayid, observationoffset o, systemicmean m FROM '{E}/vitalPeriodic.parquet'
               WHERE systemicmean BETWEEN 20 AND 200
               UNION ALL
               SELECT patientunitstayid, observationoffset o, noninvasivemean m FROM '{E}/vitalAperiodic.parquet'
               WHERE noninvasivemean BETWEEN 20 AND 200) t {J.format(o='t.o')} GROUP BY 1""").set_index("id")
out["mapmin"] = mapv.mapmin
vas = q(f"""SELECT w.id,
              max(CASE WHEN regexp_matches(lower(drugname), 'norepinephrine|levophed|epinephrine|phenylephrine|neosynephrine|vasopressin') THEN 1 ELSE 0 END) hi_v,
              max(CASE WHEN regexp_matches(lower(drugname), 'dopamine|dobutamine') THEN 1 ELSE 0 END) lo_v
            FROM '{E}/infusionDrug.parquet' t {J.format(o='t.infusionoffset')} GROUP BY 1""").set_index("id")
out = out.join(vas)
gcs = q(f"""SELECT w.id, min(TRY_CAST(nursingchartvalue AS DOUBLE)) gcs FROM '{E}/nurseCharting.parquet' t
            {J.format(o='t.nursingchartoffset')}
            WHERE nursingchartcelltypevalname = 'GCS Total' GROUP BY 1""").set_index("id")
out["gcs"] = gcs.gcs.where(gcs.gcs.between(3, 15))


def cut(x, edges, rising=True):
    """Points 0-4 from thresholds; NaN scores 0."""
    s = pd.Series(0, index=x.index)
    for i, t in enumerate(edges, 1):
        s[(x >= t) if rising else (x < t)] = i
    return s.where(x.notna(), 0)


out["sofa_resp"] = cut(out.pf, [400, 300, 200, 100], rising=False)
out["sofa_coag"] = cut(out.plt, [150, 100, 50, 20], rising=False)
out["sofa_liver"] = cut(out.bili, [1.2, 2, 6, 12])
out["sofa_renal"] = cut(out.creat, [1.2, 2, 3.5, 5])
out["sofa_cns"] = cut(out.gcs, [15, 13, 10, 6], rising=False)
cv = (out.mapmin < 70).astype(int)
cv[out.lo_v.fillna(0) == 1] = 2
cv[out.hi_v.fillna(0) == 1] = 3
out["sofa_cv"] = cv
C = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv", "sofa_cns", "sofa_renal"]
out["sofa"] = out[C].sum(axis=1)
res = out[["sofa"] + C].reset_index().rename(columns={"id": "patientunitstayid"})
res.to_parquet(RL / "eicu_sofa_lastday.parquet", index=False)
raw = {"pf": "resp", "plt": "coag", "bili": "liver", "mapmin": "cv (MAP)", "gcs": "cns", "creat": "renal"}
print("stays", len(res))
print("share with a record:", {v: round(out[k].notna().mean(), 3) for k, v in raw.items()})
print(res[["sofa"] + C].describe().round(2).to_string())
