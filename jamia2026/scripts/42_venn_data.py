#!/usr/bin/env python
"""Figure 1 data: diabetes lobe counts and pairwise DAS for three example hospitals of the 141-hospital cohort.

Aggregate output only: results/paper_venn_counts.csv (one row per hospital: characteristics, stays with evidence,
seven lobe counts in the s02 lobe order) and results/paper_venn_das.csv (pairwise DAS, degrees).
Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib \
      --with statsmodels --with scipy python scripts/42_venn_data.py [--hospitals 73 71 188] [--condition diabetes]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s38 = _load("s38", "38_iecv.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospitals", type=int, nargs=3, default=[73, 71, 188])
    ap.add_argument("--condition", default="diabetes")
    args = ap.parse_args()
    cfg, _, res, rowlevel = s38.setup("configs/external.yaml")
    split = pd.read_csv(res / "iecv_split.csv")
    hs = args.hospitals
    assert set(hs) <= set(split.hospitalid), "an example hospital is not in the 141-hospital cohort"
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    srcs = ("D", "H", "M")
    codes = s02.lobe_codes(flags, args.condition, srcs)
    cnt = s02.hospital_counts(flags, codes, 7)
    assert all(cnt.loc[h].sum() >= cfg["das"]["min_records_per_hospital"] for h in hs), "an example hospital has no DAS"
    E = cfg["paths"]["eicu"]
    hosp = duckdb.execute(f"SELECT hospitalid, region, numbedscategory, teachingstatus FROM '{E}/hospital.parquet'").df() \
        .set_index("hospitalid")
    labels = s02.lobe_labels(srcs)
    rows = []
    for h in hs:
        r = dict(hospitalid=h, region=hosp.region.get(h), beds=hosp.numbedscategory.get(h),
                 teaching=bool(hosp.teachingstatus.get(h)), no_medication=bool(split.set_index("hospitalid").no_medication[h]),
                 stays_with_evidence=int(cnt.loc[h].sum()))
        r.update({f"lobe_{i}_{labels[i]}": int(cnt.loc[h, i]) for i in range(7)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(res / "paper_venn_counts.csv", index=False)
    v = cnt.loc[hs].to_numpy(float)
    A = s02.pairwise_angles(v / v.sum(1, keepdims=True))
    pairs = [dict(a=hs[i], b=hs[j], das=A[i, j]) for i in range(3) for j in range(i + 1, 3)]
    pd.DataFrame(pairs).to_csv(res / "paper_venn_das.csv", index=False, float_format="%.2f")
    # where the pairs sit among all hospital pairs with this condition's DAS
    keep = [h for h in split.hospitalid if h in cnt.index and cnt.loc[h].sum() >= cfg["das"]["min_records_per_hospital"]]
    allv = cnt.loc[keep].to_numpy(float)
    allA = s02.pairwise_angles(allv / allv.sum(1, keepdims=True))[np.triu_indices(len(keep), 1)]
    for p in pairs:
        p["percentile_among_pairs"] = 100 * (allA < p["das"]).mean()
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(pairs).round(2).to_string(index=False), f"| {len(keep)} hospitals, {len(allA)} pairs")


if __name__ == "__main__":
    main()
