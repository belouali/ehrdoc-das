#!/usr/bin/env python
"""Documentation patterns at 24 h against the whole stay, and candidate further chronic conditions (aggregates only).

1. Per condition: share of evidenced stays in each D/H/M lobe, hospitals with a vector (>= min records), median
   pairwise DAS between hospitals, and the correlation of pairwise DAS (whole stay vs first 24 h).
2. Past-history categories by share of cohort stays (eICU pastHistory paths, depth 4), to list common chronic
   conditions beyond the four studied.
Writes results/das_24h_compare.csv and results/pasthistory_categories.csv.
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
MINR = cfg["das"]["min_records_per_hospital"]
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
LAB = s02.lobe_labels(("D", "H", "M"))

whole = pd.read_parquet(RL / "eicu_flags.parquet")
f24 = pd.read_parquet(RL / "eicu_flags_24h.parquet").merge(whole[["patientunitstayid", "hospitalid"]], on="patientunitstayid")
f24 = f24.set_index("patientunitstayid").loc[whole.patientunitstayid].reset_index()
rows = []
for c in CONDITIONS:
    V, share = {}, {}
    for name, df in (("whole", whole), ("24h", f24)):
        codes = s02.lobe_codes(df, c, ("D", "H", "M"))
        share[name] = np.bincount(codes[codes >= 0], minlength=7) / (codes >= 0).sum()
        cnt = s02.hospital_counts(df, codes, 7)
        cnt = cnt[cnt.sum(axis=1) >= MINR]
        V[name] = cnt.div(cnt.sum(axis=1), axis=0)
    common = sorted(set(V["whole"].index) & set(V["24h"].index))
    iu = np.triu_indices(len(common), 1)
    pw = {k: s02.pairwise_angles(V[k].loc[common].to_numpy(float))[iu] for k in V}
    for name in ("whole", "24h"):
        rows.append(dict(condition=c, window=name, hospitals=len(V[name]),
                         median_das=np.median(s02.pairwise_angles(V[name].to_numpy(float))[np.triu_indices(len(V[name]), 1)]),
                         p90_das=np.percentile(s02.pairwise_angles(V[name].to_numpy(float))[np.triu_indices(len(V[name]), 1)], 90),
                         r_pairwise=np.corrcoef(pw["whole"], pw["24h"])[0, 1],
                         **{l: share[name][i] for i, l in enumerate(LAB)}))
out = pd.DataFrame(rows)
out.to_csv(RES / "das_24h_compare.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))

n = len(whole)
con = duckdb.connect(); con.register("coh", whole[["patientunitstayid"]])
ph = con.execute(f"""SELECT split_part(pasthistorypath, '/', 5) AS cat, split_part(pasthistorypath, '/', 6) AS item,
                            count(DISTINCT patientunitstayid) AS stays
                     FROM '{E}/pastHistory.parquet' JOIN coh USING (patientunitstayid)
                     WHERE pasthistorypath LIKE 'notes/Progress Notes/Past History/Organ Systems/%'
                     GROUP BY 1, 2 ORDER BY stays DESC""").df()
ph["share"] = ph.stays / n
ph.to_csv(RES / "pasthistory_categories.csv", index=False, float_format="%.4f")
print(f"\ncohort stays {n:,}\n", ph.head(30).round(3).to_string(index=False))
