#!/usr/bin/env python
"""Part A task gate: does documentation matter for the task at all?

Settings: configs/screen_settings.yaml (A_task_screen), approved 2026-11-06 before computing.
eICU only, hospital-grouped 5-fold CV, logistic regression (p0 pipeline, C = 1e6).

Tasks
  A3  ICU stay > 7 days            M1 inputs + Charlson + 12 flags
  A4  ICU readmission, same        M1 inputs + index LOS + Charlson + 12 flags,
      hospitalization              among stays discharged alive from the ICU
  A5  in-hospital death            age, sex, Charlson + 12 flags
  A6  hospital stay > 14 days      M1 inputs + Charlson + 12 flags (A_task_screen_ext)
  A7  discharge to a facility      M1 inputs + Charlson + 12 flags, among hospital survivors
      vs home                      discharged home or to a facility (A_task_screen_ext)
The 12 flags: diabetes, hypertension, CHF, AF x diagnosis (D), past history (H),
in-stay medication (M).

Checks (out-of-fold, pooled over held-out hospitals)
  ablation      AUROC(full) - AUROC(without Charlson and flags; A5: age and sex only)
  perturbation  in each training fold every hospital's rate per flag is moved to the
                10th, then the 90th percentile hospital rate by random deletion or
                addition; held-out hospitals keep their records. Loss = AUROC(full) -
                AUROC(perturbed), worst direction; CITL shift = change in
                calibration-in-the-large. 3 repeats averaged.
  pass          ablation >= 0.01 and perturbation >= 0.01, or |CITL shift| >= 0.1

Output (aggregate): results/task_gate.csv (--tasks A6_hospital_los_gt14d A7_discharge_facility
--tag _ext writes results/task_gate_ext.csv)

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with statsmodels python scripts/21_task_gate.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import design_columns  # noqa: E402
from p0_model import make_pipeline  # noqa: E402

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
C, FOLDS, N_REPEAT, MIN_HOSP = 1e6, 5, 3, 100
SEED = 20261109
FACILITY = ["Skilled Nursing Facility", "Nursing Home", "Rehabilitation"]


def citl(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    off = np.log(p / (1 - p))
    return sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=off).fit().params[0]


def perturb(F, hosp, train, pct, rng):
    """Move each training hospital's rate of every flag to the pct-th percentile hospital rate."""
    F = F.copy()
    df = pd.DataFrame(F[train], columns=FLAGS)
    df["h"] = hosp[train]
    sizes = df.groupby("h").size()
    rates = df.groupby("h")[FLAGS].mean()[sizes >= MIN_HOSP]
    goal = rates.quantile(pct / 100)
    for j, f in enumerate(FLAGS):
        col = F[train, j].copy()
        for h, idx in df.groupby("h").indices.items():
            x = col[idx]
            r = x.mean()
            if r > goal[f] and r > 0:  # delete flags among positives
                drop = rng.random(len(idx)) < 1 - goal[f] / r
                x = np.where((x == 1) & drop, 0, x)
            elif r < goal[f] and r < 1:  # add flags among negatives
                add = rng.random(len(idx)) < (goal[f] - r) / (1 - r)
                x = np.where((x == 0) & add, 1, x)
            col[idx] = x
        F[train, j] = col
    return F


def oof(X, y, groups, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = make_pipeline(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--tasks", nargs="+", default=["A3_los_gt7d", "A4_icu_readmission", "A5_comorbidity_only"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    rng = np.random.default_rng(SEED if not args.tag else 20261116)

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    pt = duckdb.execute(f"""
        SELECT a.patientunitstayid, a.unitdischargeoffset / 1440.0 AS los_d, a.unitdischargestatus,
               (a.hospitaldischargeoffset - a.hospitaladmitoffset) / 1440.0 AS hosp_los_d,
               a.hospitaldischargestatus, a.hospitaldischargelocation,
               EXISTS (SELECT 1 FROM '{E}/patient.parquet' b
                       WHERE b.patienthealthsystemstayid = a.patienthealthsystemstayid
                         AND b.unitvisitnumber > a.unitvisitnumber) AS readmit
        FROM '{E}/patient.parquet' a""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    m1 = design_columns()
    age_sex = ["agenum"] + [c for c in m1 if c.startswith("gender=")]

    tasks = {
        "A3_los_gt7d": dict(df=e, y=(e.los_d > 7).astype(int), base=m1),
        "A4_icu_readmission": dict(df=e[e.unitdischargestatus == "Alive"].copy(), y=None, base=m1 + ["los_d"]),
        "A5_comorbidity_only": dict(df=e, y=e.death.astype(int), base=age_sex),
    }
    tasks["A4_icu_readmission"]["y"] = tasks["A4_icu_readmission"]["df"].readmit.astype(int)
    tasks["A6_hospital_los_gt14d"] = dict(df=e, y=(e.hosp_los_d > 14).astype(int), base=m1)
    surv = e[(e.hospitaldischargestatus == "Alive") & e.hospitaldischargelocation.isin(FACILITY + ["Home"])].copy()
    tasks["A7_discharge_facility"] = dict(df=surv, y=surv.hospitaldischargelocation.isin(FACILITY).astype(int), base=m1)
    tasks = {k: tasks[k] for k in args.tasks}

    rows = []
    for name, t in tasks.items():
        d = t["df"].reset_index(drop=True)
        y = np.asarray(t["y"], int)
        hosp = d.hospitalid.to_numpy()
        folds = list(GroupKFold(FOLDS).split(d, y, hosp))
        Xb = d[t["base"]].to_numpy(float)
        Xc = d[["charlson"]].to_numpy(float)
        F = d[FLAGS].to_numpy(float)
        X_full = np.hstack([Xb, Xc, F])
        p_full = oof(X_full, y, hosp, folds)
        p_abl = oof(Xb, y, hosp, folds)
        a_full, a_abl = roc_auc_score(y, p_full), roc_auc_score(y, p_abl)
        c_full = citl(y, p_full)
        row = dict(task=name, n=len(y), events=int(y.sum()), prevalence=y.mean(), auroc_full=a_full,
                   auroc_ablated=a_abl, ablation_loss=a_full - a_abl, citl_full=c_full)
        for pct in (10, 90):
            aucs, cits = [], []
            for _ in range(N_REPEAT):
                p = np.zeros(len(y))
                for tr, te in folds:
                    Fp = perturb(F, hosp, tr, pct, rng)
                    Xp = np.hstack([Xb, Xc, Fp])
                    p[te] = make_pipeline(C).fit(Xp[tr], y[tr]).predict_proba(X_full[te])[:, 1]
                aucs.append(roc_auc_score(y, p))
                cits.append(citl(y, p))
            row[f"pert_p{pct}_loss"] = a_full - np.mean(aucs)
            row[f"pert_p{pct}_citl_shift"] = np.mean(cits) - c_full
        row["perturbation_loss"] = max(row["pert_p10_loss"], row["pert_p90_loss"])
        row["citl_shift_max_abs"] = max(abs(row["pert_p10_citl_shift"]), abs(row["pert_p90_citl_shift"]))
        row["pass"] = bool((row["ablation_loss"] >= 0.01 and row["perturbation_loss"] >= 0.01)
                           or row["citl_shift_max_abs"] >= 0.1)
        rows.append(row)
        print(pd.Series(row).to_string(), "\n", flush=True)
    pd.DataFrame(rows).round(5).to_csv(res / f"task_gate{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
