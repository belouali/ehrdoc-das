#!/usr/bin/env python
"""Ceiling on hospital selection (C1) and the development/confirmation split (D1).

Settings: configs/ceiling_devconfirm_settings.yaml, approved 2026-11-06 before computing.

D1  The 82 eICU targets (>= 300 stays, >= 20 deaths) are split 41/41, stratified by
    region and size tertile. Written once to results/target_split.csv. Only the
    development half is read here.
C1  For each development target, its stays are split 50/50 by death into a selection
    half S and an evaluation half E. Pool: every other eICU hospital with >= 100 M1
    stays. Arms, all scored on E:
      random           200 draws of k hospitals
      rank_best/worst  M1 fitted on each pool hospital alone (C = 1.0), ranked by AUROC
                       on S, top or bottom k pooled
      set_best/worst   best or worst random draw by AUROC on S
      das_<condition>  nearest k by the DHM angle (the paper's construct)
      casemix          nearest k by the script-07 case-mix distance
    Size control as script 12 (non-random arms 3 repeats averaged). k = 10 primary,
    5 and 20 secondary.
    MIMIC-IV secondary: one target, select on the 25% development subset, score on
    the 75% evaluation subset (not blind).

Outputs (aggregate): results/target_split.csv, ceiling_targets.csv, ceiling_summary.csv,
ceiling_mimic.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib python scripts/19_ceiling.py --config configs/external.yaml
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
from p0_model import make_pipeline  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s07 = _load("s07", "07_casemix_nearest.py")

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
MIN_STAYS, MIN_DEATHS, MIN_POOL = 300, 20, 100
SINGLE_C = 1.0
KS = [10, 5, 20]
N_RANDOM, N_REPEAT, N_BOOT = 200, 3, 2000
SEED = 20261107


def fit_scores(X, y, rows, Xt, C):
    return make_pipeline(C).fit(X[rows], y[rows]).predict_proba(Xt)[:, 1].astype(np.float32)


def auc(y, p):
    return roc_auc_score(y, p)


def split_targets(targets, hosp, size, rng):
    t = pd.DataFrame({"hospitalid": targets})
    t["region"] = t.hospitalid.map(hosp.region).fillna("unknown")
    t["stays"] = t.hospitalid.map(size)
    t["size_tertile"] = pd.qcut(t.stays, 3, labels=[1, 2, 3]).astype(int)
    t["half"] = ""
    # within each stratum, shuffle and alternate; the leftover of odd strata alternates across strata
    flip = int(rng.integers(2))
    for _, g in t.groupby(["region", "size_tertile"]):
        idx = rng.permutation(g.index.to_numpy())
        for i, j in enumerate(idx):
            t.loc[j, "half"] = "development" if (i + flip) % 2 == 0 else "confirmation"
        if len(idx) % 2:
            flip = 1 - flip
    return t


def size_jobs(sets, rand, rows_of, rng):
    """Size-controlled training rows: (arm, repeat, rows)."""
    size = lambda hs: sum(len(rows_of[h]) for h in hs)
    # the target follows the settings: oracle-best, nearest and random sets; worst sets are only subsampled to it
    n_target = int(min(min(size(h) for a, h in sets.items() if "worst" not in a),
                       np.percentile([size(r) for r in rand], 5)))
    out = []
    for name, hs in sets.items():
        rows = np.concatenate([rows_of[h] for h in hs])
        for r in range(N_REPEAT):
            out.append((name, r, rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
    for i, hs in enumerate(rand):
        rows = np.concatenate([rows_of[h] for h in hs])
        out.append((f"rand_{i}", 0, rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
    return out, n_target


def boot_mean(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    b = rng.integers(0, len(x), (n, len(x)))
    m = x[b].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


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
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    rng = np.random.default_rng(SEED)

    # ---- eICU M1 data
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    cols = design_columns()
    X, y = e[cols].to_numpy(float), e.death.to_numpy(int)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
    g = e.groupby("hospitalid").death.agg(["size", "sum"])
    targets = np.array(sorted(g[(g["size"] >= MIN_STAYS) & (g["sum"] >= MIN_DEATHS)].index))
    pool_all = np.array(sorted(g[g["size"] >= MIN_POOL].index))

    # ---- D1 split
    hosp = duckdb.execute(f"SELECT hospitalid, region FROM '{E}/hospital.parquet'").df().set_index("hospitalid")
    split_file = res / "target_split.csv"
    split = split_targets(targets, hosp, g["size"], rng)  # always drawn, so later draws do not depend on the file
    if split_file.exists():
        assert pd.read_csv(split_file).half.tolist() == split.half.tolist(), "target split changed"
    else:
        split.to_csv(split_file, index=False)
    dev = split[split.half == "development"].hospitalid.to_numpy()
    if args.limit:
        dev = dev[: args.limit]
    print(f"{len(targets)} targets, {len(dev)} development, pool {len(pool_all)} hospitals", flush=True)

    # ---- hospital profiles: DHM vectors and case mix
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

    # ---- MIMIC-IV M1 data (secondary)
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    out = duckdb.execute(f"SELECT hadm_id, hospital_expire_flag death FROM '{M}/hosp/admissions.parquet'").df()
    m = m.merge(out, on="hadm_id", how="left")
    assert m.death.notna().all()
    Xm, ym, mdev = m[cols].to_numpy(float), m.death.to_numpy(int), m.reference.to_numpy(bool)

    # ---- single-source fits, once per pool hospital, scored on every eICU row and MIMIC-IV
    print(f"single-source fits: {len(pool_all)}", flush=True)
    single = Parallel(n_jobs=args.jobs)(delayed(fit_scores)(X, y, rows_of[h], np.vstack([X, Xm]), SINGLE_C)
                                        for h in pool_all)
    single = dict(zip(pool_all, single))
    n_e = len(y)

    # ---- C1 per development target
    trows = []
    for ti, t in enumerate(dev):
        r_t = rows_of[t]
        d_t = y[r_t]
        S = np.concatenate([rng.permutation(r_t[d_t == v])[: (d_t == v).sum() // 2] for v in (0, 1)])
        Ev = np.setdiff1d(r_t, S)
        pool = pool_all[pool_all != t]
        s_auc = pd.Series({h: auc(y[S], single[h][S]) for h in pool}).sort_values(ascending=False, kind="stable")
        cm = np.sqrt(((Z.reindex(pool) - Z.loc[t]) ** 2).mean(axis=1, skipna=True)).dropna().sort_values(kind="stable") \
            if t in Z.index else None
        for k in KS:
            sets = {"rank_best": s_auc.index[:k].to_numpy(), "rank_worst": s_auc.index[-k:].to_numpy()}
            for cond in CONDITIONS:
                if t in V[cond].index:
                    cand = np.array([h for h in pool if h in V[cond].index])
                    a = s02.angles(V[cond].loc[cand].to_numpy(float), V[cond].loc[t].to_numpy(float))
                    sets[f"das_{cond}"] = cand[np.argsort(a, kind="stable")[:k]]
            if cm is not None:
                sets["casemix"] = cm.index[:k].to_numpy()
            rand = [rng.choice(pool, k, replace=False) for _ in range(N_RANDOM)]
            jobs, n_target = size_jobs(sets, rand, rows_of, rng)
            test = np.concatenate([S, Ev])
            preds = Parallel(n_jobs=args.jobs, batch_size=8)(
                delayed(fit_scores)(X, y, rows, X[test], C_M1) for _, _, rows in jobs)
            ns = len(S)
            res_k = pd.DataFrame([dict(arm=a, rep=r, auc_S=auc(y[S], p[:ns]), auc_E=auc(y[Ev], p[ns:]))
                                  for (a, r, _), p in zip(jobs, preds)])
            arm = res_k.groupby("arm")[["auc_S", "auc_E"]].mean()
            rnd = arm[arm.index.str.startswith("rand_")]
            row = dict(target=t, k=k, n_train=n_target, random_mean_E=rnd.auc_E.mean(), random_sd_E=rnd.auc_E.std(),
                       random_p5_E=rnd.auc_E.quantile(.05), random_p95_E=rnd.auc_E.quantile(.95),
                       set_best_E=rnd.auc_E.loc[rnd.auc_S.idxmax()], set_worst_E=rnd.auc_E.loc[rnd.auc_S.idxmin()])
            for a in sets:
                row[f"{a}_E"] = arm.loc[a, "auc_E"]
            trows.append(row)
        print(f"  target {ti + 1}/{len(dev)} done", flush=True)
    tg = pd.DataFrame(trows)
    tag = "" if not args.limit else "_test"
    tg.round(5).to_csv(res / f"ceiling_targets{tag}.csv", index=False)

    srows = []
    brng = np.random.default_rng(SEED + 1)
    arms = ["rank_best", "set_best", "rank_worst", "set_worst", "casemix"] + [f"das_{c}" for c in CONDITIONS]
    for k, s in tg.groupby("k"):
        ceil = s.rank_best_E - s.random_mean_E
        for a in arms:
            ok = s[f"{a}_E"].notna()
            dif = (s[f"{a}_E"] - s.random_mean_E)[ok]
            est, lo, hi = boot_mean(dif, brng)
            srows.append(dict(k=k, arm=a, targets=int(ok.sum()), gain_vs_random=est, lo=lo, hi=hi,
                              share_targets_positive=(dif > 0).mean(),
                              capture_of_rank_ceiling=est / ceil[ok].mean() if a not in ("rank_best",) else 1.0))
        srows.append(dict(k=k, arm="random_sd_median", targets=len(s), gain_vs_random=s.random_sd_E.median()))
        srows.append(dict(k=k, arm="random_p95_minus_p5_median", targets=len(s),
                          gain_vs_random=(s.random_p95_E - s.random_p5_E).median()))
    sm = pd.DataFrame(srows)
    sm.round(5).to_csv(res / f"ceiling_summary{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))
    if args.limit:
        return

    # ---- MIMIC-IV secondary: select on the development subset, score on the evaluation subset
    mrows = []
    Sd, Ed = np.flatnonzero(mdev), np.flatnonzero(~mdev)
    m_auc = pd.Series({h: auc(ym[Sd], single[h][n_e:][Sd]) for h in pool_all}).sort_values(ascending=False, kind="stable")
    for k in KS:
        sets = {"rank_best": m_auc.index[:k].to_numpy(), "rank_worst": m_auc.index[-k:].to_numpy()}
        rand = [rng.choice(pool_all, k, replace=False) for _ in range(N_RANDOM)]
        jobs, n_target = size_jobs(sets, rand, rows_of, rng)
        preds = Parallel(n_jobs=args.jobs, batch_size=8)(
            delayed(fit_scores)(X, y, rows, Xm, C_M1) for _, _, rows in jobs)
        r = pd.DataFrame([dict(arm=a, auc_S=auc(ym[Sd], p[Sd]), auc_E=auc(ym[Ed], p[Ed]))
                          for (a, _, _), p in zip(jobs, preds)]).groupby("arm").mean()
        rnd = r[r.index.str.startswith("rand_")]
        base_ = rnd.auc_E.mean()
        for a, v in [("rank_best", r.loc["rank_best", "auc_E"]), ("rank_worst", r.loc["rank_worst", "auc_E"]),
                     ("set_best", rnd.auc_E.loc[rnd.auc_S.idxmax()]), ("set_worst", rnd.auc_E.loc[rnd.auc_S.idxmin()])]:
            mrows.append(dict(k=k, arm=a, n_train=n_target, auc_E=v, random_mean_E=base_, gain_vs_random=v - base_,
                              percentile_among_random=100 * (rnd.auc_E < v).mean()))
        mrows.append(dict(k=k, arm="random_sd", n_train=n_target, auc_E=rnd.auc_E.std(), random_mean_E=base_))
    mm = pd.DataFrame(mrows)
    mm.round(5).to_csv(res / "ceiling_mimic.csv", index=False)
    print(mm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
