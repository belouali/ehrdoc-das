#!/usr/bin/env python
"""Post hoc check of the Northeast-pool result from stage 4 (not prespecified).

The Northeast pool (6 to 9 eligible hospitals per condition, about 7,000 stays,
two of them 500+ bed teaching hospitals) gave the best external AUROC. Is that
region, hospital type, one hospital, or just the luck of one set?

Every arm is subsampled to a common TARGET training stays (nearest-style
arms 10 repeats averaged; each random draw once). Random draws whose hospitals
hold fewer stays than TARGET are redrawn.

Arms per condition (n = number of eligible Northeast hospitals):
  northeast            the Northeast pool
  ne_teaching_only     its 500+ bed teaching hospitals
  ne_without_teaching  the rest of the pool
  ne_drop_<id>         leave one Northeast hospital out
  random_non_ne        500 sets of n hospitals from eligible non-Northeast hospitals
  random_<region>      500 sets of n hospitals from one other region
  matched_non_ne       500 sets of n non-Northeast hospitals with the same number of
                       500+ bed teaching hospitals as the Northeast pool

Outputs: results/northeast_check_arms.csv, northeast_check_draws.csv, northeast_check_summary.csv

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib \
      python scripts/06_northeast_check.py --config configs/external.yaml
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
TARGET = 4000
N_RANDOM = 500
N_RANDOM_BOOT = 100
N_BOOT = 2000
REPEATS = 10
SEED = 20261106


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
    C = yaml.safe_load(open(res / "p0_locked.yaml"))["C"]
    cols = feature_columns()
    feat = list(range(len(cols)))

    e = pd.read_parquet(rowlevel / "eicu_features.parquet")
    e = e[e.death.notna()].reset_index(drop=True)
    Xtr, ytr = e[cols].to_numpy(float), e.death.to_numpy(int)
    rows_of = {h: np.flatnonzero(e.hospitalid.to_numpy() == h) for h in e.hospitalid.unique()}
    size = lambda hs: sum(len(rows_of.get(h, [])) for h in hs)

    m = pd.read_parquet(rowlevel / "mimic_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    m = m[~m.reference].reset_index(drop=True)
    out = duckdb.execute(f"SELECT hadm_id, hospital_expire_flag death FROM '{cfg['paths']['mimic']}/hosp/admissions.parquet'").df()
    m = m.merge(out, on="hadm_id", how="left")
    Xev, yev = m[cols].to_numpy(float), m.death.to_numpy(int)

    hosp = duckdb.execute(f"""SELECT hospitalid, coalesce(region, '(missing)') region,
        numbedscategory = '>= 500' AND teachingstatus AS big_teaching
        FROM '{cfg['paths']['eicu']}/hospital.parquet'""").df().set_index("hospitalid")
    das = pd.read_csv(res / "das_to_mimic.csv")
    das = das[(das.construct == "DAM") & (das.reference == "ref_2011_2019")]

    rng = np.random.default_rng(SEED)
    tasks = []

    def add(cond, arm, hids, draw=-1, keep=False):
        rows = np.concatenate([rows_of[h] for h in hids if h in rows_of])
        tasks.append(dict(condition=cond, k=len(hids), strategy=arm, version="size_controlled", config="northeast_check",
                          features="P0", draw=draw, n_hospitals=len(hids), hospitals=" ".join(map(str, sorted(hids))),
                          target=TARGET, repeats=REPEATS if draw < 0 else 1, seed=int(rng.integers(1 << 31)),
                          keep=keep, rows=rows, feat=feat))

    def draw_sets(pools, n_each):
        """Random sets: n_each[i] hospitals from pools[i], redrawn until the set holds >= TARGET stays."""
        sets = []
        while len(sets) < N_RANDOM:
            s = [h for pool, n in zip(pools, n_each) for h in rng.choice(pool, size=n, replace=False)]
            if size(s) >= TARGET:
                sets.append(s)
        return sets

    for cond in CONDITIONS:
        elig = das[das.condition == cond].hospitalid.to_numpy()
        reg = hosp.region.reindex(elig).to_numpy()
        big = hosp.big_teaching.reindex(elig).fillna(False).to_numpy(bool)
        ne = elig[reg == "Northeast"].tolist()
        n, n_big = len(ne), int(big[reg == "Northeast"].sum())
        add(cond, "northeast", ne, keep=True)
        add(cond, "ne_teaching_only", [h for h in ne if hosp.big_teaching.get(h)], keep=True)
        rest = [h for h in ne if not hosp.big_teaching.get(h)]
        if size(rest) >= TARGET:
            add(cond, "ne_without_teaching", rest)
        else:   # too small to subsample to TARGET: fit at full size
            add(cond, "ne_without_teaching", rest)
            tasks[-1]["target"] = None
            tasks[-1]["repeats"] = 1
        for h in ne:
            add(cond, f"ne_drop_{h}", [x for x in ne if x != h])
        non_ne = elig[reg != "Northeast"]
        for i, s in enumerate(draw_sets([non_ne], [n])):
            add(cond, "random_non_ne", s, draw=i, keep=i < N_RANDOM_BOOT)
        for region in ["Midwest", "South", "West"]:
            for i, s in enumerate(draw_sets([elig[reg == region]], [n])):
                add(cond, f"random_{region.lower()}", s, draw=i)
        nb = elig[(reg != "Northeast") & big]
        nn = elig[(reg != "Northeast") & ~big]
        for i, s in enumerate(draw_sets([nb, nn], [n_big, n - n_big])):
            add(cond, "matched_non_ne", s, draw=i, keep=i < N_RANDOM_BOOT)

    print(f"{len(tasks):,} fits on {args.jobs} workers (target {TARGET:,} training stays)")
    results = Parallel(n_jobs=args.jobs, batch_size=8)(
        delayed(stage4.run_task)(t, Xtr, ytr, Xev, yev, C) for t in tasks)
    perf = pd.DataFrame([r for r, _ in results]).drop(columns=["keep", "seed", "version", "config", "features"])
    preds = {(r["condition"], r["strategy"], r["draw"]): p for r, p in results if p is not None}

    fixed = perf[perf.draw < 0]
    draws = perf[perf.draw >= 0]
    fixed.round(5).to_csv(res / "northeast_check_arms.csv", index=False)
    draws.round(5).to_csv(res / "northeast_check_draws.csv", index=False)

    W = np.random.default_rng(SEED + 1).multinomial(len(yev), np.full(len(yev), 1 / len(yev)), size=N_BOOT).astype(np.int32)
    boot_auc = lambda p: np.concatenate([stage4.weighted_auc(p, yev, W[i:i + 250]) for i in range(0, N_BOOT, 250)])
    rows = []
    for cond in CONDITIONS:
        f = fixed[fixed.condition == cond].set_index("strategy")
        ne = f.loc["northeast"]
        pn = preds[(cond, "northeast", -1)].astype(float)
        bn = boot_auc(pn)
        for arm, g in draws[draws.condition == cond].groupby("strategy"):
            row = dict(condition=cond, comparator=arm, ne_auroc=ne.auroc, ne_brier=ne.brier, ne_cal_slope=ne.cal_slope,
                       arm_auroc_mean=g.auroc.mean(), arm_auroc_q025=g.auroc.quantile(0.025),
                       arm_auroc_q975=g.auroc.quantile(0.975), arm_cal_slope_mean=g.cal_slope.mean(),
                       auroc_diff=ne.auroc - g.auroc.mean(), pct_draws_beaten=100 * np.mean(g.auroc < ne.auroc),
                       brier_adv=g.brier.mean() - ne.brier, pct_draws_beaten_brier=100 * np.mean(g.brier > ne.brier))
            if (cond, arm, 0) in preds:
                pr = [preds[(cond, arm, i)].astype(float) for i in range(N_RANDOM_BOOT)]
                br = np.mean([boot_auc(p) for p in pr], axis=0)
                shift = row["auroc_diff"] - (roc_auc_score(yev, pn) - np.mean([roc_auc_score(yev, p) for p in pr]))
                row.update(auroc_diff_lo=np.percentile(bn - br, 2.5) + shift, auroc_diff_hi=np.percentile(bn - br, 97.5) + shift)
            rows.append(row)
    summ = pd.DataFrame(rows)
    summ.round(5).to_csv(res / "northeast_check_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(fixed[["condition", "strategy", "n_hospitals", "n_train", "auroc", "brier", "cal_slope"]].round(4).to_string(index=False))
    print(summ[["condition", "comparator", "ne_auroc", "arm_auroc_mean", "arm_auroc_q975", "auroc_diff",
                "auroc_diff_lo", "auroc_diff_hi", "pct_draws_beaten", "brier_adv"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
