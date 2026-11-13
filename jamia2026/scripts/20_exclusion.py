#!/usr/bin/env python
"""Exclusion rule (E1): train on every pool hospital except the farthest.

Settings: configs/ceiling_devconfirm_settings.yaml (E1_exclusion), approved 2026-11-06
before computing. Development targets only (results/target_split.csv).

For each development target and exclusion share q (10%, 25%, 50% of the pool):
  rules       drop the farthest q of the pool by
                das_<condition>  DHM angle (the paper's construct), 4 conditions
                das_joined       mean DHM angle over the conditions both hospitals have
                casemix          script-07 case-mix distance
                das_casemix      mean of the das_joined and casemix rank percentiles
              hospitals without a distance are never dropped
  random      50 draws dropping the same number at random (primary comparator)
  oracle      drop the worst by single-source AUROC on S (ceiling, not a candidate)
  pooled_all  every pool hospital, full size (secondary)
The target's stays are split 50/50 by death into S and E; everything is scored on E.
Size control: rules, oracle and random draws subsampled to the smaller of the smallest
rule set and the 5th percentile of random sizes; rules and oracle 3 repeats averaged.

Outputs (aggregate): results/exclusion_targets.csv, results/exclusion_summary.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib python scripts/20_exclusion.py --config configs/external.yaml
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
from m1_model import C_M1, design_columns  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s07 = _load("s07", "07_casemix_nearest.py")
s19 = _load("s19", "19_ceiling.py")

CONDITIONS = s19.CONDITIONS
SHARES = [0.10, 0.25, 0.50]
N_RANDOM, N_REPEAT, N_BOOT = 50, 3, 2000
MIN_POOL, SINGLE_C = s19.MIN_POOL, s19.SINGLE_C
SEED = 20261108


def drop_farthest(dist: pd.Series, n_drop: int) -> set:
    """The n_drop hospitals with the largest distance; hospitals without one are kept."""
    d = dist.dropna().sort_values(ascending=False, kind="stable")
    return set(d.index[:n_drop])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="timing test on the first n targets")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    rng = np.random.default_rng(SEED)

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    X, y = e[design_columns()].to_numpy(float), e.death.to_numpy(int)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
    g = e.groupby("hospitalid").death.agg(["size", "sum"])
    pool_all = np.array(sorted(g[g["size"] >= MIN_POOL].index))
    split = pd.read_csv(res / "target_split.csv")
    dev = split[split.half == "development"].hospitalid.to_numpy()
    if args.limit:
        dev = dev[: args.limit]

    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    V = {}
    for cond in CONDITIONS:
        cnt = s02.hospital_counts(flags, s02.lobe_codes(flags, cond, ("D", "H", "M")), 7)
        cnt = cnt[cnt.sum(axis=1) >= cfg["das"]["min_records_per_hospital"]]
        V[cond] = cnt.div(cnt.sum(axis=1), axis=0)
    feat = pd.read_parquet(rowlevel / "eicu_features.parquet")
    feat = feat[feat.death.notna()]
    eu = duckdb.execute(f"SELECT patientunitstayid, unittype FROM '{E}/patient.parquet'").df()
    feat = s07.add_units(feat.reset_index(drop=True), feat[["patientunitstayid"]].merge(eu, how="left").unittype,
                         s07.EICU_UNITS)
    hmeans = feat.groupby("hospitalid")[s07.CASEMIX].mean()
    Z = (hmeans - hmeans.mean()) / hmeans.std()

    print(f"{len(dev)} development targets, pool {len(pool_all)}; single-source fits", flush=True)
    single = Parallel(n_jobs=args.jobs)(delayed(s19.fit_scores)(X, y, rows_of[h], X, SINGLE_C) for h in pool_all)
    single = dict(zip(pool_all, single))

    size = lambda hs: sum(len(rows_of[h]) for h in hs)
    trows = []
    for ti, t in enumerate(dev):
        r_t = rows_of[t]
        d_t = y[r_t]
        S = np.concatenate([rng.permutation(r_t[d_t == v])[: (d_t == v).sum() // 2] for v in (0, 1)])
        Ev = np.setdiff1d(r_t, S)
        pool = pool_all[pool_all != t]
        # distances from the target to every pool hospital
        dist = {}
        for cond in CONDITIONS:
            if t in V[cond].index:
                cand = [h for h in pool if h in V[cond].index]
                dist[f"das_{cond}"] = pd.Series(
                    s02.angles(V[cond].loc[cand].to_numpy(float), V[cond].loc[t].to_numpy(float)), index=cand
                ).reindex(pool)
        if dist:
            dist["das_joined"] = pd.concat([dist[f"das_{c}"] for c in CONDITIONS if f"das_{c}" in dist],
                                           axis=1).mean(axis=1, skipna=True)
        if t in Z.index:
            dist["casemix"] = np.sqrt(((Z.reindex(pool) - Z.loc[t]) ** 2).mean(axis=1, skipna=True))
        if "das_joined" in dist and "casemix" in dist:
            dist["das_casemix"] = pd.concat([dist["das_joined"].rank(pct=True), dist["casemix"].rank(pct=True)],
                                            axis=1).mean(axis=1, skipna=True)
        oracle_loss = -pd.Series({h: s19.auc(y[S], single[h][S]) for h in pool})
        test = np.concatenate([S, Ev])
        ns = len(S)
        for q in SHARES:
            n_drop = int(round(q * len(pool)))
            sets = {a: np.array([h for h in pool if h not in drop_farthest(d, n_drop)]) for a, d in dist.items()}
            sets["oracle"] = np.array([h for h in pool if h not in drop_farthest(oracle_loss, n_drop)])
            rand = [np.setdiff1d(pool, rng.choice(pool, n_drop, replace=False)) for _ in range(N_RANDOM)]
            n_target = int(min(min(size(h) for h in sets.values()), np.percentile([size(r) for r in rand], 5)))
            jobs = []
            for a, hs in sets.items():
                rows = np.concatenate([rows_of[h] for h in hs])
                for r in range(N_REPEAT):
                    jobs.append((a, rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
            for i, hs in enumerate(rand):
                rows = np.concatenate([rows_of[h] for h in hs])
                jobs.append((f"rand_{i}", rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
            jobs.append(("pooled_all", np.concatenate([rows_of[h] for h in pool])))
            preds = Parallel(n_jobs=args.jobs, batch_size=4)(
                delayed(s19.fit_scores)(X, y, rows, X[test], C_M1) for _, rows in jobs)
            r = pd.DataFrame([dict(arm=a, auc_E=s19.auc(y[Ev], p[ns:])) for (a, _), p in zip(jobs, preds)])
            arm = r.groupby("arm").auc_E.mean()
            rnd = arm[arm.index.str.startswith("rand_")]
            row = dict(target=t, share=q, n_drop=n_drop, n_train=n_target, random_mean_E=rnd.mean(),
                       random_sd_E=rnd.std(), pooled_all_E=arm["pooled_all"])
            for a in sets:
                row[f"{a}_E"] = arm[a]
                row[f"{a}_n_dropped"] = len(pool) - len(sets[a])
            trows.append(row)
        print(f"  target {ti + 1}/{len(dev)} done", flush=True)
    tg = pd.DataFrame(trows)
    tag = "" if not args.limit else "_test"
    tg.round(5).to_csv(res / f"exclusion_targets{tag}.csv", index=False)

    brng = np.random.default_rng(SEED + 1)
    arms = ["oracle", "das_joined", "casemix", "das_casemix"] + [f"das_{c}" for c in CONDITIONS]
    srows = []
    for q, s in tg.groupby("share"):
        orc = s.oracle_E - s.random_mean_E
        for a in arms + ["pooled_all"]:
            ok = s[f"{a}_E"].notna()
            dif = (s[f"{a}_E"] - s.random_mean_E)[ok]
            est, lo, hi = s19.boot_mean(dif, brng)
            vs_pool = (s[f"{a}_E"] - s.pooled_all_E)[ok].mean() if a != "pooled_all" else 0.0
            srows.append(dict(share=q, arm=a, targets=int(ok.sum()), gain_vs_random=est, lo=lo, hi=hi,
                              share_targets_positive=(dif > 0).mean(), minus_pooled_all=vs_pool,
                              capture_of_oracle=est / orc[ok].mean() if a != "oracle" else 1.0))
    sm = pd.DataFrame(srows)
    sm.round(5).to_csv(res / f"exclusion_summary{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
