#!/usr/bin/env python
"""W1 against hospital type: does the CHF DHM angle add anything beyond region, teaching and size?

Settings: configs/ceiling_devconfirm_settings.yaml (W1_proxy).
  training    as scripts/24_w1_mimic.py
  mimic type  Northeast, teaching, >= 500 beds
  T1          nearest10 vs type_matched (pool ranked by MIMIC-IV type matches, top 10,
              random tie-breaks, 200 draws)
  T2          nearest10_non_ne vs random10_non_ne (200 draws), non-Northeast pool only
  size        every arm subsampled to one common n; nearest arms 10 repeats averaged
  statistic   paired patient bootstrap (2,000; draw means from the first 100 draws),
              one-sided p, Holm across T1 and T2
Descriptive: northeast_all (every Northeast pool hospital, all stays), arm composition,
calibration.

--check builds the training sets and prints sizes and composition; no model is fitted and
no MIMIC-IV outcome is read.

Output (aggregate): results/w1_proxy.csv

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with statsmodels python scripts/26_w1_proxy.py --config configs/external.yaml
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

FLAGS, K, MIN_POOL, N_REPEAT, N_BOOT, N_BOOT_RAND = s24.FLAGS, 10, s24.MIN_POOL, s24.N_REPEAT, s24.N_BOOT, 100
N_DRAW = 200
SEED = 20261114


def boot_one(y, p_near, P_draw, idx):
    yb = y[idx]
    return roc_auc_score(yb, p_near[idx]) - np.mean([roc_auc_score(yb, p[idx]) for p in P_draw])


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

    # ---- pool, hospital type, sets
    das = s24.mimic_angles(res, args.condition)
    d_ref = das[das.reference == "ref_2011_2019"].set_index("hospitalid").das
    pool = np.array(sorted(set(d_ref.index) & s24.pool_ok(y, hid, size_h)))
    ht = duckdb.execute(f"SELECT * FROM '{E}/hospital.parquet'").df().set_index("hospitalid").reindex(pool)
    ne = (ht.region == "Northeast").to_numpy()
    match = (ne.astype(int) + ht.teachingstatus.fillna(False).astype(bool).to_numpy().astype(int)
             + (ht.numbedscategory == ">= 500").to_numpy().astype(int))
    w1 = pd.read_csv(res / f"w1_mimic{tag}.csv")
    w1n = w1[w1.arm == "nearest"].iloc[0]
    order = d_ref.loc[pool].sort_values(kind="stable").index.to_numpy()
    assert ";".join(map(str, order[:K])) == str(w1n.hospitals), "nearest 10 differ from W1_mimic"
    non_ne = pool[~ne]
    sets = {"nearest10": order[:K], "nearest10_non_ne": d_ref.loc[non_ne].sort_values(kind="stable").index[:K].to_numpy()}
    draws = {"type_matched": [pool[np.lexsort((rng.random(len(pool)), -match))[:K]] for _ in range(N_DRAW)],
             "random10_non_ne": [rng.choice(non_ne, K, replace=False) for _ in range(N_DRAW)]}
    size = lambda hs: sum(len(rows_of[h]) for h in hs)
    n_target = int(min(w1n.n_train, size(sets["nearest10_non_ne"]),
                       *[np.percentile([size(d) for d in v], 5) for v in draws.values()]))
    northeast = pool[ne]

    def comp(hs):
        t = ht.loc[hs]
        return dict(northeast=int((t.region == "Northeast").sum()), teaching=int(t.teachingstatus.fillna(False).sum()),
                    beds_ge500=int((t.numbedscategory == ">= 500").sum()))

    print(f"pool {len(pool)} ({ne.sum()} Northeast), match score counts "
          f"{dict(zip(*np.unique(match, return_counts=True)))}, common n {n_target:,}", flush=True)
    for a, hs in sets.items():
        print(f"  {a:18s} {comp(hs)}", flush=True)
    tm = pd.DataFrame([comp(d) for d in draws["type_matched"]]).mean()
    print(f"  type_matched mean  {tm.round(1).to_dict()}; northeast_all {len(northeast)} hospitals, "
          f"{size(northeast):,} stays", flush=True)
    if args.check:
        return

    jobs = []
    sub = lambda rows: rows if len(rows) <= n_target else rng.choice(rows, n_target, replace=False)
    for a, hs in sets.items():
        rows = np.concatenate([rows_of[h] for h in hs])
        jobs += [(a, sub(rows)) for _ in range(N_REPEAT)]
    for a, v in draws.items():
        jobs += [(f"{a}_{i}", sub(np.concatenate([rows_of[h] for h in hs]))) for i, hs in enumerate(v)]
    jobs.append(("northeast_all", np.concatenate([rows_of[h] for h in northeast])))

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

    preds = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(s24.fit_pred)(X, y, rows, Xm) for _, rows in jobs)
    P, N = {}, {}
    for (a, rows), p in zip(jobs, preds):
        P.setdefault(a, []).append(p)
        N[a] = len(rows)
    P = {a: np.mean(v, axis=0) for a, v in P.items()}
    names = {a: [f"{a}_{i}" for i in range(N_DRAW)] for a in draws}
    d_auc = {a: np.array([roc_auc_score(ym, P[n]) for n in v]) for a, v in names.items()}

    brng = np.random.default_rng(SEED + 1)
    idxs = [brng.integers(0, len(ym), len(ym)) for _ in range(N_BOOT)]
    out = []
    for test, near, comp_arm in [("T1", "nearest10", "type_matched"), ("T2", "nearest10_non_ne", "random10_non_ne")]:
        bd = np.array(Parallel(n_jobs=args.jobs)(
            delayed(boot_one)(ym, P[near], [P[n] for n in names[comp_arm][:N_BOOT_RAND]], ix) for ix in idxs))
        a_near = roc_auc_score(ym, P[near])
        citl, slope = s24.calib(ym, P[near])
        out.append(dict(test=test, arm=near, comparator=comp_arm, auroc=a_near, comparator_mean=d_auc[comp_arm].mean(),
                        comparator_sd=d_auc[comp_arm].std(), gain=a_near - d_auc[comp_arm].mean(),
                        lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5),
                        p_one_sided=max((bd <= 0).mean(), 1 / N_BOOT),
                        percentile_among_draws=100 * (d_auc[comp_arm] < a_near).mean(),
                        citl=citl, cal_slope=slope, n_train=N[near], **comp(sets[near])))
    for a in draws:
        out.append(dict(arm=f"{a}_mean", auroc=d_auc[a].mean(), comparator_sd=d_auc[a].std(), n_train=n_target))
    citl, slope = s24.calib(ym, P["northeast_all"])
    out.append(dict(arm="northeast_all", auroc=roc_auc_score(ym, P["northeast_all"]), citl=citl, cal_slope=slope,
                    n_train=N["northeast_all"], **comp(northeast)))
    out = pd.DataFrame(out)
    t = out.test.notna()
    sm = out[t].sort_values("p_one_sided")
    adj = np.maximum.accumulate([min(1.0, (len(sm) - i) * p) for i, p in enumerate(sm.p_one_sided)])
    out.loc[sm.index, "p_holm"] = adj
    out.loc[t, "success"] = out.loc[t, "p_holm"] < 0.05
    out.to_csv(res / f"w1_proxy{tag}.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
