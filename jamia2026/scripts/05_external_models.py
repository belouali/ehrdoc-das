#!/usr/bin/env python
"""Stage 4: external mortality comparison of training-site strategies.

Settings are fixed in configs/stage4_settings.yaml (locked before this script
first read a MIMIC-IV outcome). Models are trained on eICU hospitals chosen by
each strategy and evaluated, unchanged, on the MIMIC-IV held-out subset.

Configurations run:
  main        all eligible hospitals, P0, k = 5, 10, 20, equal count and size-controlled
  nogcs       as main without GCS, k = 10
  complete    pool restricted to hospitals with all three sources, k = 10
  comparators pooled, Northeast pool, source-complete random-k (equal count, all k, main features)

Outputs (results/):
  stage4_performance.csv   every non-random model, and random-draw summaries
  stage4_random_draws.csv  every random draw
  stage4_contrasts.csv     nearest vs random (difference, percentile, bootstrap CI at k = 10),
                           nearest vs farthest, pooled and Northeast

Another construct runs with its own locked settings file (--settings); its
outputs carry the file's output_tag, e.g. stage4_DHM_contrasts.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib \
      python scripts/05_external_models.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_model import feature_columns, make_pipeline, metrics  # noqa: E402
from m1_model import C_M1, design_columns  # noqa: E402

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
KS = [5, 10, 20]
N_RANDOM = 500
N_BOOT = 2000
N_RANDOM_BOOT = 100
SIZE_REPEATS = 10
SEED = 20261105


def fit_predict(Xtr, ytr, Xev, feat, C, rows, target, repeats, seed):
    """Fit on the given eICU rows (subsampled to target if larger) and predict MIMIC-IV."""
    rng = np.random.default_rng(seed)
    if target is None or len(rows) <= target:
        repeats, use = 1, [rows]
    else:
        use = [rng.choice(rows, size=target, replace=False) for _ in range(repeats)]
    p = np.zeros(len(Xev))
    Xe = Xev[:, feat]
    for r in use:
        p += make_pipeline(C).fit(Xtr[np.ix_(r, feat)], ytr[r]).predict_proba(Xe)[:, 1]
    p /= len(use)
    n = len(use[0])
    deaths = int(np.mean([ytr[r].sum() for r in use]))
    return p.astype(np.float32), n, deaths


def run_task(task, Xtr, ytr, Xev, yev, C):
    p, n, deaths = fit_predict(Xtr, ytr, Xev, task["feat"], C,
                               task["rows"], task["target"], task["repeats"], task["seed"])
    m = metrics(yev, p)
    out = {k: v for k, v in task.items() if k not in ("rows", "feat")}
    out.update(n_train=n, train_deaths=deaths, **m)
    return out, (p if task["keep"] else None)


def weighted_auc(p, y, W):
    """AUROC for each bootstrap row of W (counts per patient), models fixed. Ties are rare and ignored."""
    order = np.argsort(p, kind="stable")
    ys = y[order].astype(bool)
    Wb = W[:, order]
    below = np.cumsum(Wb * ~ys, axis=1)          # at a positive position: negatives ranked below it
    pos = Wb[:, ys]
    num = (pos * below[:, ys]).sum(axis=1, dtype=np.int64)
    return num / (pos.sum(axis=1).astype(np.int64) * below[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--settings", default="configs/stage4_settings.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    sp = Path(args.settings)
    settings = yaml.safe_load(open(sp if sp.is_absolute() else base / sp))
    tag = settings.get("output_tag", "")
    assert settings["status"] == "locked", "stage 4 settings must be locked before outcomes are read"
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    model = settings.get("model_type", "P0")
    if model in ("M1", "M2"):   # the paper's model (scripts/m1_model.py), M2 adds Charlson (script 11); "P0" names the full feature set
        cols, C = design_columns() + (["charlson"] if model == "M2" else []), C_M1
        feats = {"P0": list(range(len(cols)))}
        e_file, m_file = f"eicu_{model.lower()}_features.parquet", f"mimic_{model.lower()}_features.parquet"
    else:
        C = yaml.safe_load(open(res / "p0_locked.yaml"))["C"]
        cols = feature_columns()
        feats = {"P0": [cols.index(c) for c in cols],
                 "P0_noGCS": [cols.index(c) for c in cols if c != "gcs_min"]}
        e_file, m_file = "eicu_features.parquet", "mimic_features.parquet"

    # eICU training data
    e = pd.read_parquet(rowlevel / e_file)
    e = e[e.death.notna()].reset_index(drop=True)
    Xtr, ytr = e[cols].to_numpy(float), e.death.to_numpy(int)
    rows_of = {h: np.flatnonzero(e.hospitalid.to_numpy() == h) for h in e.hospitalid.unique()}

    # MIMIC-IV evaluation subset; the outcome is read here for the first time
    m = pd.read_parquet(rowlevel / m_file).merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    m = m[~m.reference].reset_index(drop=True)
    out = duckdb.execute(f"""SELECT hadm_id, hospital_expire_flag death
        FROM '{cfg['paths']['mimic']}/hosp/admissions.parquet'""").df()
    m = m.merge(out, on="hadm_id", how="left")
    assert m.death.notna().all()
    Xev, yev = m[cols].to_numpy(float), m.death.to_numpy(int)
    print(f"Evaluation: {len(m):,} MIMIC-IV stays, {yev.sum():,} deaths ({yev.mean():.1%}). "
          f"Training pool: {len(e):,} eICU stays.")

    E = cfg["paths"]["eicu"]
    region = duckdb.execute(f"SELECT hospitalid, region FROM '{E}/hospital.parquet'").df().set_index("hospitalid").region
    if settings.get("ranking_file"):   # a precomputed ranking (e.g. the learned distance, script 10)
        das = pd.read_csv(base / settings["ranking_file"])
    else:
        das = pd.read_csv(res / "das_to_mimic.csv")
        das = das[(das.construct == settings["construct"]) & (das.reference == settings["reference"])]

    rng = np.random.default_rng(SEED)
    tasks = []

    def add(cond, k, strategy, version, config, featset, hids, target=None, draw=-1, keep=False):
        rows = np.concatenate([rows_of[h] for h in hids if h in rows_of])
        repeats = SIZE_REPEATS if (target is not None and strategy in ("nearest", "farthest")) else 1
        tasks.append(dict(condition=cond, k=k, strategy=strategy, version=version, config=config,
                          features=featset, draw=draw, n_hospitals=len(hids), hospitals=" ".join(map(str, sorted(hids))),
                          target=target, repeats=repeats, seed=int(rng.integers(1 << 31)), keep=keep,
                          rows=rows, feat=feats[featset]))

    for cond in CONDITIONS:
        d = das[das.condition == cond].sort_values("das")
        pools = {"all": d, "complete": d[~d.zero_source]}
        add(cond, 0, "pooled", "equal_count", "comparators", "P0", d.hospitalid.tolist(), keep=True)
        ne = [h for h in d.hospitalid if region.get(h) == "Northeast"]
        add(cond, 0, "northeast_pool", "equal_count", "comparators", "P0", ne, keep=True)
        for config, featset, pool, ks in [("main", "P0", "all", KS), ("nogcs", "P0_noGCS", "all", [10]),
                                          ("complete", "P0", "complete", [10])]:
            if featset not in feats:
                continue
            dd = pools[pool]
            hids = dd.hospitalid.to_numpy()
            for k in ks:
                near, far = hids[:k].tolist(), hids[-k:].tolist()
                draws = [rng.choice(hids, size=k, replace=False).tolist() for _ in range(N_RANDOM)]
                size = lambda hs: sum(len(rows_of.get(h, [])) for h in hs)
                target = int(min(size(near), size(far), np.percentile([size(x) for x in draws], 5)))
                for version, tgt in [("equal_count", None), ("size_controlled", target)]:
                    add(cond, k, "nearest", version, config, featset, near, tgt, keep=True)
                    add(cond, k, "farthest", version, config, featset, far, tgt, keep=True)
                    for i, x in enumerate(draws):
                        add(cond, k, "random", version, config, featset, x, tgt, draw=i,
                            keep=(k == 10 and i < N_RANDOM_BOOT))
                if config == "main":
                    comp = d[~d.zero_source].hospitalid.to_numpy()
                    for i in range(N_RANDOM):
                        add(cond, k, "source_complete_random", "equal_count", "comparators", "P0",
                            rng.choice(comp, size=k, replace=False).tolist(), draw=i)

    print(f"{len(tasks):,} model fits on {args.jobs} workers")
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, batch_size=8, verbose=0)(
        delayed(run_task)(t, Xtr, ytr, Xev, yev, C) for t in tasks)
    print(f"fits done in {(time.time() - t0) / 60:.1f} min")
    perf = pd.DataFrame([r for r, _ in results])
    preds = {(r["condition"], r["k"], r["strategy"], r["version"], r["config"], r["draw"]): p
             for r, p in results if p is not None}

    keys = ["condition", "k", "version", "config"]
    rand = perf[perf.strategy.isin(["random", "source_complete_random"])]
    rand.drop(columns=["seed", "keep", "repeats"]).round(5).to_csv(res / f"stage4{tag}_random_draws.csv", index=False)
    summ = (rand.groupby(keys + ["strategy"])[["auroc", "auprc", "brier", "cal_slope", "cal_intercept", "n_train", "train_deaths"]]
            .agg(["mean", "std", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)]))
    summ.columns = [f"{a}_{ {'<lambda_0>': 'q025', '<lambda_1>': 'q975'}.get(b, b)}" for a, b in summ.columns]
    fixed = perf[~perf.strategy.isin(["random", "source_complete_random"])].drop(columns=["seed", "keep", "draw"])
    pd.concat([fixed, summ.reset_index()], ignore_index=True).round(5).to_csv(res / f"stage4{tag}_performance.csv", index=False)

    # contrasts
    boot_rng = np.random.default_rng(SEED + 1)
    W = boot_rng.multinomial(len(yev), np.full(len(yev), 1 / len(yev)), size=N_BOOT).astype(np.int32)
    contrasts = []
    for (cond, k, version, config), g in perf[perf.strategy == "random"].groupby(keys):
        f = perf[(perf.condition == cond) & (perf.k == k) & (perf.version == version) & (perf.config == config)]
        near = f[f.strategy == "nearest"].iloc[0]
        far = f[f.strategy == "farthest"].iloc[0]
        row = dict(condition=cond, k=k, version=version, config=config,
                   auroc_nearest=near.auroc, auroc_random_mean=g.auroc.mean(), auroc_farthest=far.auroc,
                   brier_nearest=near.brier, brier_random_mean=g.brier.mean(), brier_farthest=far.brier,
                   auroc_diff=near.auroc - g.auroc.mean(), brier_adv=g.brier.mean() - near.brier,
                   auroc_pct_beaten=100 * np.mean(g.auroc < near.auroc),
                   brier_pct_beaten=100 * np.mean(g.brier > near.brier),
                   auroc_near_minus_far=near.auroc - far.auroc, brier_far_minus_near=far.brier - near.brier,
                   n_train_nearest=near.n_train, n_train_random_median=g.n_train.median(),
                   n_hospitals_pool=None)
        if k == 10:
            pn = preds[(cond, k, "nearest", version, config, -1)].astype(float)
            pf = preds[(cond, k, "farthest", version, config, -1)].astype(float)
            pr = [preds[(cond, k, "random", version, config, i)].astype(float) for i in range(N_RANDOM_BOOT)]
            auc_n = np.concatenate([weighted_auc(pn, yev, W[i:i + 250]) for i in range(0, N_BOOT, 250)])
            auc_f = np.concatenate([weighted_auc(pf, yev, W[i:i + 250]) for i in range(0, N_BOOT, 250)])
            auc_r = np.mean([np.concatenate([weighted_auc(p, yev, W[i:i + 250]) for i in range(0, N_BOOT, 250)])
                             for p in pr], axis=0)
            sq = lambda p: (p - yev) ** 2
            br_n = W @ sq(pn) / len(yev)
            br_f = W @ sq(pf) / len(yev)
            br_r = W @ np.mean([sq(p) for p in pr], axis=0) / len(yev)
            # centre the 100-draw bootstrap on the 500-draw point estimate
            d_auc = auc_n - auc_r - (roc_auc_score(yev, pn) - np.mean([roc_auc_score(yev, p) for p in pr]))
            d_br = br_r - br_n - (np.mean([sq(p).mean() for p in pr]) - sq(pn).mean())
            row.update(auroc_diff_lo=row["auroc_diff"] + np.percentile(d_auc, 2.5),
                       auroc_diff_hi=row["auroc_diff"] + np.percentile(d_auc, 97.5),
                       brier_adv_lo=row["brier_adv"] + np.percentile(d_br, 2.5),
                       brier_adv_hi=row["brier_adv"] + np.percentile(d_br, 97.5),
                       auroc_near_minus_far_lo=np.percentile(auc_n - auc_f, 2.5),
                       auroc_near_minus_far_hi=np.percentile(auc_n - auc_f, 97.5),
                       brier_far_minus_near_lo=np.percentile(br_f - br_n, 2.5),
                       brier_far_minus_near_hi=np.percentile(br_f - br_n, 97.5))
            if config == "main":
                for comp in ["pooled", "northeast_pool"]:
                    pc = preds[(cond, 0, comp, "equal_count", "comparators", -1)].astype(float)
                    auc_c = np.concatenate([weighted_auc(pc, yev, W[i:i + 250]) for i in range(0, N_BOOT, 250)])
                    br_c = W @ sq(pc) / len(yev)
                    row.update({f"auroc_near_minus_{comp}": roc_auc_score(yev, pn) - roc_auc_score(yev, pc),
                                f"auroc_near_minus_{comp}_lo": np.percentile(auc_n - auc_c, 2.5),
                                f"auroc_near_minus_{comp}_hi": np.percentile(auc_n - auc_c, 97.5),
                                f"brier_{comp}_minus_near": sq(pc).mean() - sq(pn).mean(),
                                f"brier_{comp}_minus_near_lo": np.percentile(br_c - br_n, 2.5),
                                f"brier_{comp}_minus_near_hi": np.percentile(br_c - br_n, 97.5)})
        contrasts.append(row)
    pd.DataFrame(contrasts).drop(columns=["n_hospitals_pool"]).round(5).to_csv(res / f"stage4{tag}_contrasts.csv", index=False)
    print(f"wrote results/stage4{tag}_performance.csv, stage4{tag}_random_draws.csv, stage4{tag}_contrasts.csv")


if __name__ == "__main__":
    main()
