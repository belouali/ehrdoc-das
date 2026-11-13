#!/usr/bin/env python
"""Case-mix-nearest comparator (the proposal's "clinical-nearest k").

Listed in the proposal as a secondary strategy; specified and run after the
stage 4 and Northeast results were seen, so it is reported as post hoc.

Case mix is measured without outcomes on the MIMIC-IV reference subset (never
used for evaluation) and on each eICU hospital's stays:
  age, male share, one first-day summary per measurement (the median hourly
  vital, the worst lab value, minimum GCS), and the share of stays in cardiac,
  neuro, medical and surgical ICUs (mixed medical-surgical is the remainder).
Each hospital-level mean is standardized by its spread across eligible eICU
hospitals. Distance to MIMIC-IV is the root-mean-square standardized difference
over the features a hospital has.

For each condition and k (5, 10, 20), the k case-mix-nearest eligible hospitals
are trained with the stage 4 pipeline and compared with the stage 4 random draws:
same pool, same k, same size-control targets (results/stage4_performance.csv).
At k = 10 the 95% CIs (2,000 patient bootstraps) are against the random mean
(the first 100 stage 4 draws refitted), DAS-nearest and the Northeast pool.

Outputs: results/casemix_distance.csv, casemix_nearest_performance.csv, casemix_nearest_contrasts.csv

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib \
      python scripts/07_casemix_nearest.py --config configs/external.yaml
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
from p0_model import feature_columns  # noqa: E402

_spec = importlib.util.spec_from_file_location("stage4", HERE / "05_external_models.py")
stage4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage4)

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
KS = [5, 10, 20]
N_RANDOM_BOOT = 100
N_BOOT = 2000
REPEATS = 10
SEED = 20261107
CASEMIX = ["age", "male", "gcs_min", "hr_median", "map_median", "rr_median", "temp_median", "spo2_median",
           "creatinine_max", "bun_max", "sodium_min", "potassium_max", "hgb_min", "wbc_max", "platelets_min",
           "unit_cardiac", "unit_neuro", "unit_medical", "unit_surgical"]
EICU_UNITS = {"CCU-CTICU": "cardiac", "Cardiac ICU": "cardiac", "CSICU": "cardiac", "CTICU": "cardiac",
              "Neuro ICU": "neuro", "MICU": "medical", "SICU": "surgical", "Med-Surg ICU": "mixed"}
MIMIC_UNITS = {"Cardiac Vascular Intensive Care Unit (CVICU)": "cardiac", "Coronary Care Unit (CCU)": "cardiac",
               "Neuro Surgical Intensive Care Unit (Neuro SICU)": "neuro", "Neuro Intermediate": "neuro",
               "Neuro Stepdown": "neuro", "Medical Intensive Care Unit (MICU)": "medical",
               "Surgical Intensive Care Unit (SICU)": "surgical", "Trauma SICU (TSICU)": "surgical",
               "Medical/Surgical Intensive Care Unit (MICU/SICU)": "mixed"}


def add_units(df: pd.DataFrame, unit: pd.Series, mapping: dict) -> pd.DataFrame:
    kind = unit.map(mapping)
    assert kind.notna().all(), sorted(unit[kind.isna()].unique())
    for k in ["cardiac", "neuro", "medical", "surgical"]:
        df[f"unit_{k}"] = (kind == k).astype(float).to_numpy()
    return df


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
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    C = yaml.safe_load(open(res / "p0_locked.yaml"))["C"]
    cols = feature_columns()
    feat = list(range(len(cols)))

    # eICU training data, with unit type for case mix
    e = pd.read_parquet(rowlevel / "eicu_features.parquet")
    e = e[e.death.notna()].reset_index(drop=True)
    eu = duckdb.execute(f"SELECT patientunitstayid, unittype FROM '{E}/patient.parquet'").df()
    e = add_units(e, e[["patientunitstayid"]].merge(eu, how="left").unittype, EICU_UNITS)
    Xtr, ytr = e[cols].to_numpy(float), e.death.to_numpy(int)
    rows_of = {h: np.flatnonzero(e.hospitalid.to_numpy() == h) for h in e.hospitalid.unique()}

    # MIMIC-IV: reference subset for case mix (no outcome), evaluation subset for scoring
    m = pd.read_parquet(rowlevel / "mimic_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    mu = duckdb.execute(f"SELECT stay_id, first_careunit FROM '{M}/icu/icustays.parquet'").df()
    m = add_units(m, m[["stay_id"]].merge(mu, how="left").first_careunit, MIMIC_UNITS)
    ref = m[m.reference]
    ev = m[~m.reference].reset_index(drop=True)
    out = duckdb.execute(f"SELECT hadm_id, hospital_expire_flag death FROM '{M}/hosp/admissions.parquet'").df()
    ev = ev.merge(out, on="hadm_id", how="left")
    Xev, yev = ev[cols].to_numpy(float), ev.death.to_numpy(int)

    # case-mix distance
    das = pd.read_csv(res / "das_to_mimic.csv")
    das = das[(das.construct == "DAM") & (das.reference == "ref_2011_2019")]
    elig_all = sorted(das.hospitalid.unique())
    hmeans = e[e.hospitalid.isin(elig_all)].groupby("hospitalid")[CASEMIX].mean()
    target_vec = ref[CASEMIX].mean()
    z = (hmeans - hmeans.mean()) / hmeans.std()
    zt = (target_vec - hmeans.mean()) / hmeans.std()
    dist = np.sqrt(((z - zt) ** 2).mean(axis=1, skipna=True)).rename("casemix_distance")
    region = duckdb.execute(f"SELECT hospitalid, coalesce(region, '(missing)') region FROM '{E}/hospital.parquet'").df().set_index("hospitalid").region
    dtab = dist.reset_index().merge(region.reset_index(), on="hospitalid", how="left")
    dtab = dtab.merge(das.pivot(index="hospitalid", columns="condition", values="das").add_prefix("das_").reset_index(),
                      on="hospitalid", how="left").sort_values("casemix_distance")
    dtab["casemix_rank"] = np.arange(1, len(dtab) + 1)
    dtab.round(4).to_csv(res / "casemix_distance.csv", index=False)
    print("MIMIC-IV reference case mix vs eICU hospital median:")
    print(pd.DataFrame({"mimic_ref": target_vec, "eicu_hosp_median": hmeans.median(), "z_mimic": zt}).round(3).to_string())

    s4 = pd.read_csv(res / "stage4_performance.csv")
    s4 = s4[s4.config == "main"]
    rd = pd.read_csv(res / "stage4_random_draws.csv")
    rd = rd[(rd.config == "main") & (rd.strategy == "random")]

    rng = np.random.default_rng(SEED)
    tasks = []

    def add(cond, k, arm, version, hids, target, draw=-1, keep=True):
        rows = np.concatenate([rows_of[h] for h in hids if h in rows_of])
        repeats = REPEATS if (target is not None and draw < 0) else 1
        tasks.append(dict(condition=cond, k=k, strategy=arm, version=version, config="casemix", features="P0",
                          draw=draw, n_hospitals=len(hids), hospitals=" ".join(map(str, sorted(hids))),
                          target=target, repeats=repeats, seed=int(rng.integers(1 << 31)), keep=keep,
                          rows=rows, feat=feat))

    overlap = []
    for cond in CONDITIONS:
        pool = das[das.condition == cond]
        cm = dist.reindex(pool.hospitalid).sort_values()
        das_order = pool.sort_values("das").hospitalid.tolist()
        rho = pd.Series(cm.rank()).corr(pool.set_index("hospitalid").das.reindex(cm.index).rank())
        ne = [h for h in pool.hospitalid if region.get(h) == "Northeast"]
        for k in KS:
            near = cm.index[:k].tolist()
            overlap.append(dict(condition=cond, k=k, casemix_nearest=" ".join(map(str, near)),
                                overlap_with_das_nearest=len(set(near) & set(das_order[:k])),
                                northeast_in_casemix_nearest=sum(region.get(h) == "Northeast" for h in near),
                                spearman_casemix_vs_das=rho))
            for version in ["equal_count", "size_controlled"]:
                tgt = None
                if version == "size_controlled":
                    tgt = int(s4[(s4.condition == cond) & (s4.k == k) & (s4.strategy == "nearest") &
                                 (s4.version == version)].target.iloc[0])
                add(cond, k, "casemix_nearest", version, near, tgt)
                if k == 10:
                    add(cond, k, "das_nearest", version, das_order[:k], tgt)
                    add(cond, k, "northeast_pool", version, ne, tgt)
                    draws = rd[(rd.condition == cond) & (rd.k == k) & (rd.version == version)].sort_values("draw")
                    for _, r in draws.head(N_RANDOM_BOOT).iterrows():
                        add(cond, k, "random", version, [int(x) for x in r.hospitals.split()], tgt, draw=int(r.draw))

    print(f"{len(tasks):,} fits on {args.jobs} workers")
    results = Parallel(n_jobs=args.jobs, batch_size=4)(delayed(stage4.run_task)(t, Xtr, ytr, Xev, yev, C) for t in tasks)
    perf = pd.DataFrame([r for r, _ in results]).drop(columns=["keep", "seed", "config", "features"])
    preds = {(r["condition"], r["k"], r["strategy"], r["version"], r["draw"]): p for r, p in results}
    perf[perf.strategy != "random"].round(5).to_csv(res / "casemix_nearest_performance.csv", index=False)

    W = np.random.default_rng(SEED + 1).multinomial(len(yev), np.full(len(yev), 1 / len(yev)), size=N_BOOT).astype(np.int32)
    bauc = lambda p: np.concatenate([stage4.weighted_auc(p, yev, W[i:i + 250]) for i in range(0, N_BOOT, 250)])
    sq = lambda p: (p - yev) ** 2
    rows = []
    for o in overlap:
        cond, k = o["condition"], o["k"]
        for version in ["equal_count", "size_controlled"]:
            cmr = perf[(perf.condition == cond) & (perf.k == k) & (perf.strategy == "casemix_nearest") & (perf.version == version)].iloc[0]
            g = rd[(rd.condition == cond) & (rd.k == k) & (rd.version == version)]
            row = dict(o, version=version, n_train=cmr.n_train, auroc_casemix=cmr.auroc, brier_casemix=cmr.brier,
                       cal_slope_casemix=cmr.cal_slope, auroc_random_mean=g.auroc.mean(),
                       auroc_diff_vs_random=cmr.auroc - g.auroc.mean(), pct_random_beaten=100 * np.mean(g.auroc < cmr.auroc),
                       brier_adv_vs_random=g.brier.mean() - cmr.brier)
            if k == 10:
                pc = preds[(cond, k, "casemix_nearest", version, -1)].astype(float)
                bc = bauc(pc)
                pr = [preds[(cond, k, "random", version, int(d))].astype(float)
                      for d in sorted(g.draw)[:N_RANDOM_BOOT]]
                br = np.mean([bauc(p) for p in pr], axis=0)
                shift = row["auroc_diff_vs_random"] - (roc_auc_score(yev, pc) - np.mean([roc_auc_score(yev, p) for p in pr]))
                row.update(auroc_diff_vs_random_lo=np.percentile(bc - br, 2.5) + shift,
                           auroc_diff_vs_random_hi=np.percentile(bc - br, 97.5) + shift)
                for comp in ["das_nearest", "northeast_pool"]:
                    pp = preds[(cond, k, comp, version, -1)].astype(float)
                    bp = bauc(pp)
                    row.update({f"auroc_{comp}": roc_auc_score(yev, pp),
                                f"auroc_diff_vs_{comp}": roc_auc_score(yev, pc) - roc_auc_score(yev, pp),
                                f"auroc_diff_vs_{comp}_lo": np.percentile(bc - bp, 2.5),
                                f"auroc_diff_vs_{comp}_hi": np.percentile(bc - bp, 97.5),
                                f"brier_adv_vs_{comp}": sq(pp).mean() - sq(pc).mean()})
            rows.append(row)
    con = pd.DataFrame(rows)
    con.round(5).to_csv(res / "casemix_nearest_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(con[["condition", "k", "version", "overlap_with_das_nearest", "northeast_in_casemix_nearest", "n_train",
               "auroc_casemix", "auroc_random_mean", "auroc_diff_vs_random", "pct_random_beaten", "brier_adv_vs_random",
               "cal_slope_casemix"]].round(4).to_string(index=False))
    k10 = con[con.k == 10]
    print(k10[["condition", "version", "auroc_diff_vs_random_lo", "auroc_diff_vs_random_hi", "auroc_das_nearest",
               "auroc_diff_vs_das_nearest", "auroc_diff_vs_das_nearest_lo", "auroc_diff_vs_das_nearest_hi",
               "auroc_northeast_pool", "auroc_diff_vs_northeast_pool", "auroc_diff_vs_northeast_pool_lo",
               "auroc_diff_vs_northeast_pool_hi"]].round(4).to_string(index=False))
    print(dtab.head(12)[["hospitalid", "region", "casemix_distance", "das_af", "das_hypertension"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
