#!/usr/bin/env python
"""eICU-internal replication of the MIMIC-IV 15-lobe result, with the paper's model (M1).

The MIMIC-IV test found nearest-10 by the 15-lobe DHAM angle beating random-10 by
+0.024 AUROC for hypertension, but not for CHF. This asks whether the same holds when
each large eICU hospital in turn is the target.

For every target hospital (>= 300 stays, >= 20 deaths) and condition:
  pool      eICU hospitals other than the target with >= min_records evidence-positive
            stays under both DHM and DHAM (one pool, so the constructs differ only in ranking)
  nearest   10 pool hospitals with the smallest angle to the target, DHAM and DHM
  random    30 draws of 10 pool hospitals, shared by both constructs
  model     M1 (C = 1e6) fitted on all M1-eligible stays of the selected hospitals,
            scored on all the target's stays (AUROC)
  size      primary: every training set subsampled to the smallest of the two nearest sets
            and the 5th percentile of the random sets (nearest: 3 repeats averaged);
            also equal hospital count
Summary per condition: mean over targets of AUROC(nearest) - mean AUROC(random), 95% CI
from 2,000 bootstrap resamples of targets, share of targets with a gain, and DHAM - DHM.
Settings fixed before the first run (2026-11-06). eICU only, no MIMIC-IV data.

Outputs (aggregate): results/eicu_transport_dham_targets.csv (one row per target,
condition, version), results/eicu_transport_dham_summary.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib python scripts/12_eicu_transport_dham.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import C_M1, design_columns  # noqa: E402
from p0_model import make_pipeline  # noqa: E402

spec = importlib.util.spec_from_file_location("s02", HERE / "02_das_external.py")
s02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s02)

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
CONSTRUCTS = {"DHAM": ("D", "H", "A", "M"), "DHM": ("D", "H", "M")}
MIN_STAYS, MIN_DEATHS = 300, 20
K, N_RANDOM, N_REPEAT, N_BOOT = 10, 30, 3, 2000
SEED = 20261106


def target_jobs(near, rand, rows_of, n_target, rng_seed):
    """Training rows for every fit of one target and condition."""
    rng = np.random.default_rng(rng_seed)
    sets = {f"near_{c}": h for c, h in near.items()} | {f"rand_{i}": r for i, r in enumerate(rand)}
    out = []
    for name, hs in sets.items():
        rows = np.concatenate([rows_of[h] for h in hs])
        out.append((name, "equal_count", None, rows))
        reps = N_REPEAT if name.startswith("near") else 1
        for r in range(reps):
            sub = rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)
            out.append((name, "size_controlled", r, sub))
    return out


def fit_auc(X, y, rows, test):
    p = make_pipeline(C_M1).fit(X[rows], y[rows]).predict_proba(X[test])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    rng = np.random.default_rng(SEED)

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    X, y = e[design_columns()].to_numpy(float), e.death.to_numpy(int)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h) for h in np.unique(hid)}
    g = e.groupby("hospitalid").death.agg(["size", "sum"])
    targets = np.array(sorted(g[(g["size"] >= MIN_STAYS) & (g["sum"] >= MIN_DEATHS)].index))

    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    jobs, meta = [], []
    for cond in CONDITIONS:
        V = {}
        for cname, srcs in CONSTRUCTS.items():
            cnt = s02.hospital_counts(flags, s02.lobe_codes(flags, cond, srcs), 2 ** len(srcs) - 1)
            cnt = cnt[cnt.sum(axis=1) >= cfg["das"]["min_records_per_hospital"]]
            V[cname] = cnt.div(cnt.sum(axis=1), axis=0)
        eligible = sorted(set(V["DHAM"].index) & set(V["DHM"].index) & set(rows_of))
        for t in targets:
            if t not in eligible:
                continue
            pool = np.array([h for h in eligible if h != t])
            near = {}
            for cname in CONSTRUCTS:
                a = s02.angles(V[cname].loc[pool].to_numpy(float), V[cname].loc[t].to_numpy(float))
                near[cname] = pool[np.argsort(a, kind="stable")[:K]]
            rand = [rng.choice(pool, K, replace=False) for _ in range(N_RANDOM)]
            size = lambda hs: sum(len(rows_of[h]) for h in hs)
            n_target = int(min(size(near["DHAM"]), size(near["DHM"]), np.percentile([size(r) for r in rand], 5)))
            for name, version, rep, rows in target_jobs(near, rand, rows_of, n_target, int(rng.integers(1 << 31))):
                jobs.append(rows)
                meta.append(dict(condition=cond, target=t, set=name, version=version, rep=rep, n_train=len(rows),
                                 overlap=len(set(near["DHAM"]) & set(near["DHM"]))))
    print(f"{len(targets)} candidate targets, {len(jobs):,} fits on {args.jobs} workers", flush=True)
    preds = Parallel(n_jobs=args.jobs, batch_size=16)(
        delayed(fit_auc)(X, y, rows, rows_of[m["target"]]) for rows, m in zip(jobs, meta))
    for m, p in zip(meta, preds):
        m["auroc"] = roc_auc_score(y[rows_of[m["target"]]], p)
    d = pd.DataFrame(meta)

    # nearest repeats: average predictions would be better, average AUROC is close enough at 3 repeats
    d["kind"] = np.where(d.set.str.startswith("near"), d.set, "random")
    per = d.groupby(["condition", "target", "version", "kind"]).agg(auroc=("auroc", "mean"), n_train=("n_train", "mean"),
                                                                     overlap=("overlap", "first")).unstack("kind")
    rows = []
    for (cond, t, ver), r in per.iterrows():
        rows.append(dict(condition=cond, target=t, version=ver,
                         auroc_random=r[("auroc", "random")], auroc_near_DHAM=r[("auroc", "near_DHAM")],
                         auroc_near_DHM=r[("auroc", "near_DHM")],
                         diff_DHAM=r[("auroc", "near_DHAM")] - r[("auroc", "random")],
                         diff_DHM=r[("auroc", "near_DHM")] - r[("auroc", "random")],
                         n_train_near_DHAM=r[("n_train", "near_DHAM")], n_train_random=r[("n_train", "random")],
                         nearest_overlap=r[("overlap", "random")]))
    tg = pd.DataFrame(rows)
    tg.round(5).to_csv(res / "eicu_transport_dham_targets.csv", index=False)

    brng = np.random.default_rng(SEED + 1)
    summ = []
    for (cond, ver), s in tg.groupby(["condition", "version"]):
        row = dict(condition=cond, version=ver, n_targets=len(s), mean_nearest_overlap=s.nearest_overlap.mean())
        for col in ["diff_DHAM", "diff_DHM"]:
            v = s[col].to_numpy()
            b = np.array([v[brng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
            row |= {col: v.mean(), f"{col}_lo": np.percentile(b, 2.5), f"{col}_hi": np.percentile(b, 97.5),
                    f"{col}_share_pos": (v > 0).mean(), f"{col}_median": np.median(v)}
        v = (s.diff_DHAM - s.diff_DHM).to_numpy()
        b = np.array([v[brng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
        row |= dict(DHAM_minus_DHM=v.mean(), DHAM_minus_DHM_lo=np.percentile(b, 2.5), DHAM_minus_DHM_hi=np.percentile(b, 97.5))
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.round(5).to_csv(res / "eicu_transport_dham_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
