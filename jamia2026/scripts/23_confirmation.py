#!/usr/bin/env python
"""Single confirmation run for the two D1 winners on the unread confirmation half.

Settings: configs/ceiling_devconfirm_settings.yaml (D1_winners), approved 2026-11-06
before computing.
  W1  A3 ICU stay > 7 days,  nearest 10 by the CHF DHM angle
  W2  A4 ICU readmission,    nearest 10 by the diabetes DHM angle
Targets: the 41 confirmation targets (A4: >= 20 events); every stay of the target is
scored. Comparator: 200 random sets of 10. Size control as script 22. Statistic: mean
over targets of AUROC(nearest) - mean AUROC(random), 2,000 target-bootstrap CI,
one-sided p = share of bootstrap means <= 0, Holm across W1 and W2.
Descriptive: case-mix nearest-10.

--check runs the code on two development targets and writes nothing (no confirmation
outcome is read).

Outputs (aggregate): results/confirmation_targets.csv, results/confirmation_summary.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib python scripts/23_confirmation.py --config configs/external.yaml
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
WINNERS = {"W1": ("A3_los_gt7d", "chf"), "W2": ("A4_icu_readmission", "diabetes"),
           "W1d": ("A3_los_gt7d", "diabetes"),  # W1d: settings W1_diabetes, run alone with --only W1d
           # W3, W3d: screen_settings.yaml A_task_screen_ext winners, run with --only W3 W3d --no-holm
           "W3": ("A6_hospital_los_gt14d", "chf"), "W3d": ("A6_hospital_los_gt14d", "diabetes")}
K, N_RANDOM, MIN_POOL, MIN_EVENTS, N_BOOT = 10, 200, 100, 20, 2000
SEED = 20261111


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="*", default=["W1", "W2"])
    ap.add_argument("--no-holm", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    rng = np.random.default_rng(SEED)

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    pt = duckdb.execute(f"""
        SELECT a.patientunitstayid, a.unitdischargeoffset / 1440.0 AS los_d, a.unitdischargestatus,
               (a.hospitaldischargeoffset - a.hospitaladmitoffset) / 1440.0 AS hosp_los_d,
               EXISTS (SELECT 1 FROM '{E}/patient.parquet' b
                       WHERE b.patienthealthsystemstayid = a.patienthealthsystemstayid
                         AND b.unitvisitnumber > a.unitvisitnumber) AS readmit
        FROM '{E}/patient.parquet' a""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    m1 = design_columns()
    alive = e[e.unitdischargestatus == "Alive"].reset_index(drop=True)
    tasks = {"A3_los_gt7d": (e, (e.los_d > 7).astype(int), m1),
             "A4_icu_readmission": (alive, alive.readmit.astype(int), m1 + ["los_d"]),
             "A6_hospital_los_gt14d": (e, (e.hosp_los_d > 14).astype(int), m1)}

    split = pd.read_csv(res / "target_split.csv")
    half = "development" if args.check else "confirmation"
    tgt_all = split[split.half == half].hospitalid.to_numpy()
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

    trows = []
    tag = "" if args.only == ["W1", "W2"] else "_" + "_".join(args.only)
    for w, (task, cond) in ((w, WINNERS[w]) for w in args.only):
        d, yy, basecols = tasks[task]
        X = d[basecols + ["charlson"] + FLAGS].to_numpy(float)
        y = np.asarray(yy, int)
        hid = d.hospitalid.to_numpy()
        rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
        g = pd.DataFrame({"h": hid, "y": y}).groupby("h").y.agg(["size", "sum"])
        pool_all = np.array(sorted(g[(g["size"] >= MIN_POOL) & (g["sum"] > 0) & (g["sum"] < g["size"])].index))
        tg = [t for t in tgt_all if t in g.index and g.loc[t, "sum"] >= MIN_EVENTS and t in V[cond].index]
        if args.check:
            tg = tg[:2]
        print(f"{w} {task} by {cond}: {len(tg)} targets", flush=True)
        for t in tg:
            pool = pool_all[pool_all != t]
            cand = np.array([h for h in pool if h in V[cond].index])
            a = s02.angles(V[cond].loc[cand].to_numpy(float), V[cond].loc[t].to_numpy(float))
            sets = {"nearest": cand[np.argsort(a, kind="stable")[:K]]}
            if t in Z.index:
                cm = np.sqrt(((Z.reindex(pool) - Z.loc[t]) ** 2).mean(axis=1, skipna=True)).dropna()
                sets["casemix"] = cm.sort_values(kind="stable").index[:K].to_numpy()
            rand = [rng.choice(pool, K, replace=False) for _ in range(N_RANDOM)]
            # casemix is descriptive, so the size target follows nearest and random only
            jobs, n_target = s19.size_jobs({"nearest": sets["nearest"]}, rand, rows_of, rng)
            if "casemix" in sets:
                rows = np.concatenate([rows_of[h] for h in sets["casemix"]])
                for r in range(s19.N_REPEAT):
                    jobs.append(("casemix", r, rows if len(rows) <= n_target
                                 else rng.choice(rows, n_target, replace=False)))
            test = rows_of[t]
            preds = Parallel(n_jobs=args.jobs, batch_size=8)(
                delayed(s19.fit_scores)(X, y, rows, X[test], C_M1) for _, _, rows in jobs)
            r = pd.DataFrame([dict(arm=a_, auc=s19.auc(y[test], p)) for (a_, _, _), p in zip(jobs, preds)]) \
                .groupby("arm").auc.mean()
            rnd = r[r.index.str.startswith("rand_")]
            trows.append(dict(winner=w, task=task, condition=cond, target=t, n_train=n_target,
                              stays=len(test), events=int(y[test].sum()), nearest=r["nearest"],
                              casemix=r.get("casemix", np.nan), random_mean=rnd.mean(), random_sd=rnd.std(),
                              nearest_pct_among_random=100 * (rnd < r["nearest"]).mean()))
    tg = pd.DataFrame(trows)
    if args.check:
        print(tg.round(4).to_string(index=False))
        return
    tg.round(5).to_csv(res / f"confirmation_targets{tag}.csv", index=False)

    brng = np.random.default_rng(SEED + 1)
    srows = []
    for w, s in tg.groupby("winner"):
        dif = (s.nearest - s.random_mean).to_numpy()
        b = dif[brng.integers(0, len(dif), (N_BOOT, len(dif)))].mean(axis=1)
        cmd = (s.casemix - s.random_mean).dropna().to_numpy()
        srows.append(dict(winner=w, task=s.task.iloc[0], condition=s.condition.iloc[0], targets=len(s),
                          gain_vs_random=dif.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                          p_one_sided=max((b <= 0).mean(), 1 / N_BOOT), share_targets_positive=(dif > 0).mean(),
                          casemix_gain_descriptive=cmd.mean() if len(cmd) else np.nan,
                          median_random_auroc=s.random_mean.median()))
    sm = pd.DataFrame(srows).sort_values("p_one_sided").reset_index(drop=True)
    m = len(sm)
    adj = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(sm.p_one_sided)])
    sm["p_holm"] = adj
    sm["confirmed"] = (sm.p_one_sided < 0.05) & (sm.lo > 0) if args.no_holm else sm.p_holm < 0.05
    sm.round(5).to_csv(res / f"confirmation_summary{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
