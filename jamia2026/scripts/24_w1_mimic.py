#!/usr/bin/env python
"""W1 in MIMIC-IV: ICU stay > 7 days, nearest 10 eICU hospitals by the CHF DHM angle.

Settings: configs/ceiling_devconfirm_settings.yaml (W1_mimic), approved 2026-11-06
before computing.
  training    eICU M1 stays, M1 inputs + Charlson + 12 flags (scripts/21_task_gate.py)
  pool        eICU hospitals with >= 100 stays and a CHF DHM angle (ref_2011_2019)
  nearest     10 smallest angles; 500 random draws of 10 as comparator
  size        nearest subsampled to min(nearest, 5th percentile of random), 10 repeats
              with predictions averaged; each random draw subsampled once
  evaluation  MIMIC-IV 75% evaluation subset, outcome icustays los > 7 days
  statistic   AUROC(nearest) - mean AUROC(random), 2,000 patient bootstrap (random mean
              from the first 100 draws), percentile among the 500 draws
Descriptive: case-mix nearest-10, ref_all_years nearest-10, calibration of nearest.

Output (aggregate): results/w1_mimic.csv

--task death (settings W_mimic_aligned): in-hospital death (eICU death, MIMIC-IV hospital_expire_flag); outputs tagged _death.
W_mimic_aligned also adds the farthest-10 arm (with a bootstrap CI) and sizes every set to one n.
--task hosp_los_gt14d (settings W3_mimic): eICU outcome hospital stay > 14 days
(hospitaldischargeoffset - hospitaladmitoffset), MIMIC-IV outcome admissions dischtime -
admittime > 14 days; pool hospitals with no events dropped; outputs tagged _a6.
--check builds the sets and prints sizes; no MIMIC-IV outcome is read.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with statsmodels python scripts/24_w1_mimic.py --config configs/external.yaml
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import C_M1, design_columns  # noqa: E402
from p0_model import make_pipeline  # noqa: E402

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
K, N_RANDOM, N_REPEAT, N_BOOT, N_BOOT_RAND, MIN_POOL = 10, 500, 10, 2000, 100, 100
SEED = 20261112


TASKS = {"los_gt7d": "", "hosp_los_gt14d": "_a6", "death": "_death"}


def tag_for(task, condition):
    return TASKS[task] + ("" if condition == "chf" else f"_{condition}")


def mimic_angles(res, condition):
    """DHM angles of eICU hospitals to MIMIC-IV; "combined" is the mean over the four conditions' angles
    available for a hospital, as the joined angle of script 33."""
    das = pd.read_csv(res / "das_to_mimic.csv")
    das = das[das.construct == "DHM"]
    if condition == "combined":
        return das[das.condition.isin(CONDITIONS)].groupby(["reference", "hospitalid"], as_index=False).das.mean()
    if condition == "max":  # S_max_angle (configs/screen_settings.yaml)
        return das[das.condition.isin(CONDITIONS)].groupby(["reference", "hospitalid"], as_index=False).das.max()
    return das[das.condition == condition]


def eicu_outcome(task, e, E):
    """Outcome for the eICU training stays, in the row order of e."""
    if task == "death":
        assert e.death.notna().all()
        return e.death.to_numpy(int)
    if task == "los_gt7d":
        q = f"SELECT patientunitstayid, unitdischargeoffset / 1440.0 AS v FROM '{E}/patient.parquet'"
        lim = 7
    else:
        q = (f"SELECT patientunitstayid, (hospitaldischargeoffset - hospitaladmitoffset) / 1440.0 AS v "
             f"FROM '{E}/patient.parquet'")
        lim = 14
    v = e[["patientunitstayid"]].merge(duckdb.execute(q).df(), on="patientunitstayid", how="left").v
    assert v.notna().all()
    return (v > lim).to_numpy(int)


def mimic_outcome(task, m, M):
    """Outcome for the MIMIC-IV evaluation stays, in the row order of m."""
    if task == "death":
        q = (f"SELECT i.stay_id, a.hospital_expire_flag AS v "
             f"FROM '{M}/icu/icustays.parquet' i JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)")
        v = m[["stay_id"]].merge(duckdb.execute(q).df(), on="stay_id", how="left").v
        assert v.notna().all()
        return v.to_numpy(int)
    if task in ("los_gt7d", "los_gt5d"):
        q, lim = f"SELECT stay_id, los AS v FROM '{M}/icu/icustays.parquet'", int(task[6])
    else:
        q = (f"SELECT i.stay_id, date_diff('second', a.admittime, a.dischtime) / 86400.0 AS v "
             f"FROM '{M}/icu/icustays.parquet' i JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)")
        lim = 14
    v = m[["stay_id"]].merge(duckdb.execute(q).df(), on="stay_id", how="left").v
    assert v.notna().all()
    return (v > lim).to_numpy(int)


def pool_ok(y, hid, size_h):
    """Hospitals with >= MIN_POOL stays and at least one event and one non-event."""
    g = pd.DataFrame({"h": hid, "y": y}).groupby("h").y.agg(["size", "sum"])
    return set(g[(g["size"] >= MIN_POOL) & (g["sum"] > 0) & (g["sum"] < g["size"])].index)


def fit_pred(X, y, rows, Xt):
    return make_pipeline(C_M1).fit(X[rows], y[rows]).predict_proba(Xt)[:, 1].astype(np.float32)


def calib(y, p):
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    return citl, slope


def boot_one(y, p_near, P_rand, idx):
    yb = y[idx]
    return roc_auc_score(yb, p_near[idx]) - np.mean([roc_auc_score(yb, p[idx]) for p in P_rand])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--condition", default="chf")
    ap.add_argument("--task", default="los_gt7d", choices=list(TASKS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    rng = np.random.default_rng(SEED)
    tag = tag_for(args.task, args.condition)
    cols = design_columns() + ["charlson"] + FLAGS

    # ---- eICU training data
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    X, y = e[cols].to_numpy(float), eicu_outcome(args.task, e, E)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
    size_h = pd.Series({h: len(r) for h, r in rows_of.items()})

    # ---- MIMIC-IV evaluation data
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    m = m[~m.reference].reset_index(drop=True)
    m = m.merge(pd.read_parquet(rowlevel / "mimic_m2_features.parquet")[["stay_id", "charlson"]], on="stay_id", how="left")
    dm = [f for f in FLAGS if not f.endswith("_H")]
    hf = [f for f in FLAGS if f.endswith("_H")]
    m = m.merge(pd.read_parquet(rowlevel / "mimic_flags.parquet")[["stay_id"] + dm], on="stay_id", how="left")
    m = m.merge(pd.read_parquet(rowlevel / "mimic_history_flags.parquet")[["stay_id"] + hf], on="stay_id", how="left")
    m[FLAGS] = m[FLAGS].fillna(0).astype(float)
    Xm = m[cols].to_numpy(float)

    # ---- pool and sets
    das = mimic_angles(res, args.condition)
    d_ref = das[das.reference == "ref_2011_2019"].set_index("hospitalid").das
    d_all = das[das.reference == "ref_all_years"].set_index("hospitalid").das
    pool = np.array(sorted(set(d_ref.index) & pool_ok(y, hid, size_h)))
    cm = pd.read_csv(res / "casemix_distance.csv").set_index("hospitalid").casemix_distance
    sets = {"nearest": d_ref.loc[pool].sort_values(kind="stable").index[:K].to_numpy(),
            "nearest_ref_all_years": d_all.reindex(pool).dropna().sort_values(kind="stable").index[:K].to_numpy(),
            "casemix_nearest": cm.reindex(pool).dropna().sort_values(kind="stable").index[:K].to_numpy(),
            "farthest": d_ref.loc[pool].sort_values(ascending=False, kind="stable").index[:K].to_numpy()}
    rand = [rng.choice(pool, K, replace=False) for _ in range(N_RANDOM)]
    size = lambda hs: sum(len(rows_of[h]) for h in hs)
    # W_mimic_aligned: one size for every set, as S_site_selection_aligned (smallest selected set or 5th percentile of random)
    n_target = int(min(min(size(hs) for hs in sets.values()), np.percentile([size(r) for r in rand], 5)))
    print(f"pool {len(pool)}, training size {n_target:,}, MIMIC-IV evaluation {len(m):,} stays", flush=True)
    if args.check:
        return
    ym = mimic_outcome(args.task, m, M)
    print(f"MIMIC-IV outcome: {ym.sum():,} events ({ym.mean():.1%})", flush=True)

    jobs = []
    rng_far = np.random.default_rng(SEED + 7)  # own stream, so the other arms draw exactly as before the farthest arm was added
    for a, hs in sets.items():
        rows = np.concatenate([rows_of[h] for h in hs])
        g = rng_far if a == "farthest" else rng
        for r in range(N_REPEAT):
            jobs.append((a, rows if len(rows) <= n_target else g.choice(rows, n_target, replace=False)))
    for i, hs in enumerate(rand):
        rows = np.concatenate([rows_of[h] for h in hs])
        jobs.append((f"rand_{i}", rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
    preds = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(fit_pred)(X, y, rows, Xm) for _, rows in jobs)
    P = {}
    for (a, _), p in zip(jobs, preds):
        P.setdefault(a, []).append(p)
    P = {a: np.mean(v, axis=0) for a, v in P.items()}
    rnames = [f"rand_{i}" for i in range(N_RANDOM)]
    r_auc = np.array([roc_auc_score(ym, P[n]) for n in rnames])

    brng = np.random.default_rng(SEED + 1)
    idxs = [brng.integers(0, len(ym), len(ym)) for _ in range(N_BOOT)]
    P_rand100 = [P[n] for n in rnames[:N_BOOT_RAND]]
    bd = np.array(Parallel(n_jobs=args.jobs)(delayed(boot_one)(ym, P["nearest"], P_rand100, ix) for ix in idxs))
    bf = np.array(Parallel(n_jobs=args.jobs)(delayed(boot_one)(ym, P["farthest"], P_rand100, ix) for ix in idxs))

    rows = []
    for a in sets:
        auc_a = roc_auc_score(ym, P[a])
        citl, slope = calib(ym, P[a])
        row = dict(arm=a, auroc=auc_a, random_mean=r_auc.mean(), gain_vs_random=auc_a - r_auc.mean(),
                   percentile_among_random=100 * (r_auc < auc_a).mean(), citl=citl, cal_slope=slope,
                   n_train=n_target, hospitals=";".join(map(str, sets[a])))
        if a == "nearest":
            row.update(lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5), p_one_sided=max((bd <= 0).mean(), 1 / N_BOOT),
                       success=bool(auc_a - r_auc.mean() > 0 and np.percentile(bd, 2.5) > 0))
        if a == "farthest":
            row.update(lo=np.percentile(bf, 2.5), hi=np.percentile(bf, 97.5))
        rows.append(row)
    rows.append(dict(arm="random_sd", auroc=r_auc.std(), random_mean=r_auc.mean(), n_train=n_target))
    out = pd.DataFrame(rows)
    out.to_csv(res / f"w1_mimic{tag}.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 220)
    print(out.drop(columns="hospitals").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
