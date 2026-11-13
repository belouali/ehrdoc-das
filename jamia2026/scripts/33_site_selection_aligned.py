#!/usr/bin/env python
"""S_site_selection_aligned: the paper's site-selection analysis redone with the design of the new work
(configs/screen_settings.yaml, S_site_selection_aligned, locked 2026-11-07 before computing).

Targets: the 82 eICU targets of results/target_split.csv (41 development, 41 held-out), each left out
of its own training. Tasks: in-hospital mortality, ICU stay > 7 d, hospital stay > 14 d; a target needs
>= 20 events. Model: M1 + Charlson + the four conditions' D, H, M flags for every task. Pool: every other
hospital with >= 100 stays and both outcome classes. Stays of each target split 50/50 by outcome; arms are
scored on the evaluation half E.
Arms: nearest and farthest k by the DHM angle (four conditions and joined), nearest k by case mix,
random k (200 draws at k = 10, 100 at k = 5 and 20); size control as script 19.
Runs: both halves at k = 10, 5, 20 on all hospitals; k = 10 without hospitals that have no medication
records (targets and pool).
Estimate: gain in AUROC over the random mean, 95% target bootstrap; calibration-in-the-large and slope.

Outputs (aggregate): results/site_selection_targets.csv (resumable), results/site_selection_summary.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with statsmodels python scripts/33_site_selection_aligned.py --config configs/external.yaml
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
import statsmodels.api as sm
import yaml
from joblib import Parallel, delayed

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
FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
MIN_POOL, MIN_EVENTS = 100, 20
N_RAND = {10: 200, 5: 100, 20: 100}
RUNS = [("all", 10), ("all", 5), ("all", 20), ("with_medication", 10)]


def calib(y, p):
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:  # noqa: BLE001  perfect separation on a small evaluation half
        citl = slope = np.nan
    return citl, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    st = yaml.safe_load(open(base / "configs" / "screen_settings.yaml"))["S_site_selection_aligned"]
    assert st["status"] == "locked"

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    pt = duckdb.execute(f"""SELECT patientunitstayid, unitdischargeoffset / 1440.0 AS los_d,
                                   (hospitaldischargeoffset - hospitaladmitoffset) / 1440.0 AS hosp_los_d
                            FROM '{E}/patient.parquet'""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    tasks = {"mortality": e.death.astype(int), "A3_los_gt7d": (e.los_d > 7).astype(int),
             "A6_hospital_los_gt14d": (e.hosp_los_d > 14).astype(int)}
    X = e[design_columns() + ["charlson"] + FLAGS].to_numpy(float)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}

    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    no_med = set(flags.groupby("hospitalid").any_M.max().loc[lambda s: s == 0].index)
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
    split = pd.read_csv(res / "target_split.csv").set_index("hospitalid").half

    out_path = res / ("site_selection_targets.csv" if not args.limit else "site_selection_targets_test.csv")
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["task", "sample", "k"])
    done_keys = set(zip(done.task, done["sample"], done.k))
    for task, yy in tasks.items():
        y = np.asarray(yy, int)
        g = pd.DataFrame({"h": hid, "y": y}).groupby("h").y.agg(["size", "sum"])
        pool_ok = g[(g["size"] >= MIN_POOL) & (g["sum"] > 0) & (g["sum"] < g["size"])].index
        for sample, K in RUNS:
            if (task, sample, K) in done_keys:
                continue
            rng = np.random.default_rng([st["seed"], len(task), K, len(sample)])
            pool_all = np.array(sorted(h for h in pool_ok if sample == "all" or h not in no_med))
            tg = [t for t in split.index if t in g.index and g.loc[t, "sum"] >= MIN_EVENTS
                  and (sample == "all" or t not in no_med)]
            if args.limit:
                tg = tg[: args.limit]
            print(f"{task} {sample} k={K}: {len(tg)} targets, pool {len(pool_all)}", flush=True)
            trows = []
            for t in tg:
                r_t = rows_of[t]
                d_t = y[r_t]
                S = np.concatenate([rng.permutation(r_t[d_t == v])[: (d_t == v).sum() // 2] for v in (0, 1)])
                Ev = np.setdiff1d(r_t, S)
                pool = pool_all[pool_all != t]
                sets, ang = {}, {}
                for cond in CONDITIONS:
                    if t in V[cond].index:
                        cand = np.array([h for h in pool if h in V[cond].index])
                        a = pd.Series(s02.angles(V[cond].loc[cand].to_numpy(float),
                                                 V[cond].loc[t].to_numpy(float)), index=cand)
                        ang[cond] = a
                        sets[f"near_{cond}"] = a.sort_values(kind="stable").index[:K].to_numpy()
                        sets[f"far_{cond}"] = a.sort_values(ascending=False, kind="stable").index[:K].to_numpy()
                if ang:
                    j = pd.concat(ang.values(), axis=1).mean(axis=1, skipna=True)
                    sets["near_joined"] = j.sort_values(kind="stable").index[:K].to_numpy()
                    sets["far_joined"] = j.sort_values(ascending=False, kind="stable").index[:K].to_numpy()
                if t in Z.index:
                    cm = np.sqrt(((Z.reindex(pool) - Z.loc[t]) ** 2).mean(axis=1, skipna=True)).dropna()
                    sets["casemix"] = cm.sort_values(kind="stable").index[:K].to_numpy()
                rand = [rng.choice(pool, K, replace=False) for _ in range(N_RAND[K])]
                jobs, n_target = s19.size_jobs(sets, rand, rows_of, rng)
                preds = Parallel(n_jobs=args.jobs, batch_size=8)(
                    delayed(s19.fit_scores)(X, y, rows, X[Ev], C_M1) for _, _, rows in jobs)
                yE = y[Ev]
                per = pd.DataFrame([dict(arm=a, auc=s19.auc(yE, p), **dict(zip(("citl", "slope"), calib(yE, p))))
                                    for (a, _, _), p in zip(jobs, preds)])
                per["grp"] = np.where(per.arm.str.startswith("rand_"), "random", per.arm)
                m = per.groupby("grp")[["auc", "citl", "slope"]].mean()
                row = dict(task=task, sample=sample, k=K, target=t, half=split[t], n_train=n_target,
                           random_sd=per.loc[per.grp == "random", "auc"].std())
                for a in m.index:
                    for c in ("auc", "citl", "slope"):
                        row[f"{a}_{c}"] = m.loc[a, c]
                trows.append(row)
            new = pd.DataFrame(trows)
            done = pd.concat([done, new], ignore_index=True)
            done.round(5).to_csv(out_path, index=False)
            print(f"  done {task} {sample} k={K}", flush=True)

    tg = done
    brng = np.random.default_rng(st["seed"] + 1)
    arms = [f"{p}_{c}" for p in ("near", "far") for c in CONDITIONS + ["joined"]] + ["casemix"]
    srows = []
    for (task, sample, K, half), s in tg.groupby(["task", "sample", "k", "half"]):
        for a in arms:
            if f"{a}_auc" not in s:
                continue
            ok = s[f"{a}_auc"].notna()
            dif = (s[f"{a}_auc"] - s.random_auc)[ok]
            est, lo, hi = s19.boot_mean(dif, brng)
            srows.append(dict(task=task, sample=sample, k=K, half=half, arm=a, targets=int(ok.sum()),
                              gain_vs_random=est, lo=lo, hi=hi, share_targets_positive=(dif > 0).mean(),
                              passes=bool(lo > 0), citl=s.loc[ok, f"{a}_citl"].median(),
                              slope=s.loc[ok, f"{a}_slope"].median(),
                              random_citl=s.loc[ok, "random_citl"].median(),
                              random_slope=s.loc[ok, "random_slope"].median(),
                              random_auc_median=s.loc[ok, "random_auc"].median()))
    sm_ = pd.DataFrame(srows)
    sm_.round(5).to_csv(res / ("site_selection_summary.csv" if not args.limit else "site_selection_summary_test.csv"),
                        index=False)
    pd.set_option("display.width", 220)
    show = sm_[(sm_["sample"] == "all") & (sm_.k == 10)]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
