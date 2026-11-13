#!/usr/bin/env python
"""S_membership_comparator, MIMIC-IV part (configs/screen_settings.yaml, locked 2026-11-08 before computing).

One pool per outcome: eICU hospitals with a combined DHM angle to MIMIC-IV that are eligible as in W_mimic_aligned.
Membership relatedness of each pool hospital to MIMIC-IV: outcome-free Debray membership model (scripts/35), the
hospital's stays against the MIMIC-IV reference subset (25%), M1 patient covariates, 5-fold CV c-statistic.
Arms: das_near / das_far (combined angle), memb_near / memb_far, dasmemb_near (mean of the two ranks), 500 random
sets of 10. Every set subsampled to one n (smallest selected set or 5th percentile of random sizes); selected sets
10 repeats averaged. Scored on the MIMIC-IV evaluation subset: AUROC, gain over the random mean, percentile among
random sets, patient-bootstrap CI against the mean of the first 100 random draws, calibration.

Outputs (aggregate): results/membership_mimic.csv, results/membership_mimic_relatedness.csv

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with statsmodels python scripts/36_membership_mimic.py --config configs/external.yaml
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
from m1_model import design_columns  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s24 = _load("s24", "24_w1_mimic.py")
s35 = _load("s35", "35_membership_eicu.py")
K, N_RANDOM, N_REPEAT, N_BOOT, N_BOOT_RAND = 10, 500, 10, 2000, 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    ms = yaml.safe_load(open(base / "configs" / "screen_settings.yaml"))["S_membership_comparator"]
    assert ms["status"] == "locked"
    FLAGS = s24.FLAGS
    cols = design_columns() + ["charlson"] + FLAGS

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    X, XM = e[cols].to_numpy(float), e[design_columns()].to_numpy(float)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
    size_h = pd.Series({h: len(r) for h, r in rows_of.items()})

    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    ref_M = m[m.reference][design_columns()].to_numpy(float)
    m = m[~m.reference].reset_index(drop=True)
    m = m.merge(pd.read_parquet(rowlevel / "mimic_m2_features.parquet")[["stay_id", "charlson"]], on="stay_id", how="left")
    dm = [f for f in FLAGS if not f.endswith("_H")]
    hf = [f for f in FLAGS if f.endswith("_H")]
    m = m.merge(pd.read_parquet(rowlevel / "mimic_flags.parquet")[["stay_id"] + dm], on="stay_id", how="left")
    m = m.merge(pd.read_parquet(rowlevel / "mimic_history_flags.parquet")[["stay_id"] + hf], on="stay_id", how="left")
    m[FLAGS] = m[FLAGS].fillna(0).astype(float)
    Xm = m[cols].to_numpy(float)

    das = s24.mimic_angles(res, "combined")
    d_ref = das[das.reference == "ref_2011_2019"].set_index("hospitalid").das
    candidates = sorted(set(d_ref.index) & set(rows_of))
    cs = Parallel(n_jobs=args.jobs, batch_size=4)(
        delayed(s35.membership_c)(XM[rows_of[h]], ref_M, int(ms["seed"]) + int(h)) for h in candidates)
    c_all = pd.Series(cs, index=candidates)
    pd.DataFrame({"hospitalid": candidates, "c_statistic": cs, "das_combined": d_ref.loc[candidates].values}) \
        .round(5).to_csv(res / "membership_mimic_relatedness.csv", index=False)
    print(f"membership c-statistic to MIMIC-IV: median {c_all.median():.3f}, range {c_all.min():.3f} to {c_all.max():.3f}; "
          f"Spearman with combined angle {c_all.corr(d_ref.loc[candidates], method='spearman'):.2f}", flush=True)

    out = []
    for ti, task in enumerate(["death", "los_gt7d", "hosp_los_gt14d"]):
        rng = np.random.default_rng([int(ms["seed"]), ti])
        y = s24.eicu_outcome(task, e, E)
        pool = np.array(sorted(set(candidates) & s24.pool_ok(y, hid, size_h)))
        d, c = d_ref.loc[pool], c_all.loc[pool]
        both = pd.concat([d.rank(), c.rank()], axis=1).mean(axis=1)
        sets = {"das_near": d.sort_values(kind="stable").index[:K].to_numpy(),
                "das_far": d.sort_values(ascending=False, kind="stable").index[:K].to_numpy(),
                "memb_near": c.sort_values(kind="stable").index[:K].to_numpy(),
                "memb_far": c.sort_values(ascending=False, kind="stable").index[:K].to_numpy(),
                "dasmemb_near": both.sort_values(kind="stable").index[:K].to_numpy()}
        rand = [rng.choice(pool, K, replace=False) for _ in range(N_RANDOM)]
        size = lambda hs: sum(len(rows_of[h]) for h in hs)
        n_target = int(min(min(size(hs) for hs in sets.values()), np.percentile([size(r) for r in rand], 5)))
        ym = s24.mimic_outcome(task, m, M)
        print(f"{task}: pool {len(pool)}, n {n_target:,}, {ym.sum():,} events", flush=True)
        jobs = []
        for a, hs in sets.items():
            rows = np.concatenate([rows_of[h] for h in hs])
            for _ in range(N_REPEAT):
                jobs.append((a, rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
        for i, hs in enumerate(rand):
            rows = np.concatenate([rows_of[h] for h in hs])
            jobs.append((f"rand_{i}", rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)))
        preds = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(s24.fit_pred)(X, y, rows, Xm) for _, rows in jobs)
        P = {}
        for (a, _), p in zip(jobs, preds):
            P.setdefault(a, []).append(p)
        P = {a: np.mean(v, axis=0) for a, v in P.items()}
        rnames = [f"rand_{i}" for i in range(N_RANDOM)]
        r_auc = np.array([roc_auc_score(ym, P[n]) for n in rnames])
        brng = np.random.default_rng(int(ms["seed"]) + 100 + ti)
        idxs = [brng.integers(0, len(ym), len(ym)) for _ in range(N_BOOT)]
        P100 = [P[n] for n in rnames[:N_BOOT_RAND]]
        for a, hs in sets.items():
            bd = np.array(Parallel(n_jobs=args.jobs)(delayed(s24.boot_one)(ym, P[a], P100, ix) for ix in idxs))
            auc_a = roc_auc_score(ym, P[a])
            citl, slope = s24.calib(ym, P[a])
            out.append(dict(task=task, arm=a, auroc=auc_a, random_mean=r_auc.mean(), random_sd=r_auc.std(),
                            gain_vs_random=auc_a - r_auc.mean(), lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5),
                            percentile_among_random=100 * (r_auc < auc_a).mean(), citl=citl, cal_slope=slope,
                            n_train=n_target, pool=len(pool), mean_c=float(c.loc[hs].mean()),
                            mean_angle=float(d.loc[hs].mean()), hospitals=";".join(map(str, hs))))
        print(pd.DataFrame(out)[lambda x: x.task == task].drop(columns="hospitals").round(4).to_string(index=False), flush=True)
    pd.DataFrame(out).to_csv(res / "membership_mimic.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
