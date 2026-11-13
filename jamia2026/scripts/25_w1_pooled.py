#!/usr/bin/env python
"""W1 against pooling: nearest 10 eICU hospitals by the CHF DHM angle vs every pool hospital.

Settings: configs/ceiling_devconfirm_settings.yaml (W1_pooled).
  training    as scripts/24_w1_mimic.py (eICU task cohort, M1 inputs + Charlson + 12 flags,
              pool = hospitals with >= 100 stays and a CHF DHM angle, ref_2011_2019)
  nearest10   as script 24: subsampled to the W1_mimic training size, 10 repeats averaged
  pooled_all  every pool hospital, all stays
  evaluation  MIMIC-IV 75% evaluation subset, outcome icustays los > 7 days
  statistic   AUROC(nearest10) - AUROC(pooled_all), 2,000 paired patient bootstrap
Descriptive: pooled_size_matched (pool subsampled to the W1_mimic size, 10 repeats),
nearest-k with all stays for k = 10, 20, 40, 80, calibration of every arm.

--check builds the data and training sets and prints their sizes. No model is fitted
and no MIMIC-IV outcome is read.

Output (aggregate): results/w1_pooled.csv

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with statsmodels python scripts/25_w1_pooled.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import design_columns  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s24 = _load("s24", "24_w1_mimic.py")

FLAGS, MIN_POOL, N_REPEAT, N_BOOT = s24.FLAGS, s24.MIN_POOL, s24.N_REPEAT, s24.N_BOOT
KS_FULL = [10, 20, 40, 80]
SEED = 20261113


def boot_one(y, a, b, idx):
    return roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--condition", default="chf")
    ap.add_argument("--task", default="los_gt7d", choices=list(s24.TASKS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    rng = np.random.default_rng(SEED)
    tag = s24.tag_for(args.task, args.condition)
    cols = design_columns() + ["charlson"] + FLAGS

    # ---- eICU training data (as script 24)
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    X, y = e[cols].to_numpy(float), s24.eicu_outcome(args.task, e, E)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
    size_h = pd.Series({h: len(r) for h, r in rows_of.items()})

    # ---- pool, sets and the W1_mimic training size
    das = s24.mimic_angles(res, args.condition)
    d_ref = das[das.reference == "ref_2011_2019"].set_index("hospitalid").das
    pool = np.array(sorted(set(d_ref.index) & s24.pool_ok(y, hid, size_h)))
    order = d_ref.loc[pool].sort_values(kind="stable").index.to_numpy()
    w1 = pd.read_csv(res / f"w1_mimic{tag}.csv")
    w1n = w1[w1.arm == "nearest"].iloc[0]
    n_target = int(w1n.n_train)
    assert ";".join(map(str, order[:10])) == str(w1n.hospitals), "nearest 10 differ from W1_mimic"
    allrows = np.concatenate([rows_of[h] for h in pool])
    jobs = []
    near = np.concatenate([rows_of[h] for h in order[:10]])
    for _ in range(N_REPEAT):
        jobs.append(("nearest10", near if len(near) <= n_target else rng.choice(near, n_target, replace=False)))
    jobs.append(("pooled_all", allrows))
    for _ in range(N_REPEAT):
        jobs.append(("pooled_size_matched", rng.choice(allrows, n_target, replace=False)))
    for k in KS_FULL:
        jobs.append((f"nearest{k}_full", np.concatenate([rows_of[h] for h in order[:k]])))
    print(f"pool {len(pool)} hospitals, {len(allrows):,} stays; W1_mimic training size {n_target:,}", flush=True)
    for a, n in pd.Series({a: len(r) for a, r in jobs}).groupby(level=0).first().items():
        print(f"  {a:22s} {n:>8,} training stays", flush=True)
    if args.check:
        return

    # ---- MIMIC-IV evaluation data (as script 24)
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    m = m[~m.reference].reset_index(drop=True)
    m = m.merge(pd.read_parquet(rowlevel / "mimic_m2_features.parquet")[["stay_id", "charlson"]], on="stay_id", how="left")
    dm = [f for f in FLAGS if not f.endswith("_H")]
    hf = [f for f in FLAGS if f.endswith("_H")]
    m = m.merge(pd.read_parquet(rowlevel / "mimic_flags.parquet")[["stay_id"] + dm], on="stay_id", how="left")
    m = m.merge(pd.read_parquet(rowlevel / "mimic_history_flags.parquet")[["stay_id"] + hf], on="stay_id", how="left")
    m[FLAGS] = m[FLAGS].fillna(0).astype(float)
    Xm, ym = m[cols].to_numpy(float), s24.mimic_outcome(args.task, m, M)

    preds = Parallel(n_jobs=args.jobs, batch_size=2)(delayed(s24.fit_pred)(X, y, rows, Xm) for _, rows in jobs)
    P, N = {}, {}
    for (a, rows), p in zip(jobs, preds):
        P.setdefault(a, []).append(p)
        N[a] = len(rows)
    P = {a: np.mean(v, axis=0) for a, v in P.items()}

    brng = np.random.default_rng(SEED + 1)
    idxs = [brng.integers(0, len(ym), len(ym)) for _ in range(N_BOOT)]
    bd = np.array(Parallel(n_jobs=args.jobs)(delayed(boot_one)(ym, P["nearest10"], P["pooled_all"], ix) for ix in idxs))

    a_pool = roc_auc_score(ym, P["pooled_all"])
    out = []
    for a in P:
        auc_a = roc_auc_score(ym, P[a])
        citl, slope = s24.calib(ym, P[a])
        row = dict(arm=a, auroc=auc_a, minus_pooled_all=auc_a - a_pool, citl=citl, cal_slope=slope, n_train=N[a])
        if a == "nearest10":
            row.update(lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5),
                       p_one_sided=max((bd <= 0).mean(), 1 / N_BOOT),
                       success=bool(auc_a - a_pool > 0 and np.percentile(bd, 2.5) > 0))
        out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(res / f"w1_pooled{tag}.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
