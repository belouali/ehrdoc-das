#!/usr/bin/env python
"""S_membership_comparator, eICU part (configs/screen_settings.yaml, locked 2026-11-08 before computing).

Replays S_site_selection_aligned (scripts/33) for all hospitals at k = 10 with the same seed, so every target has
the same selection half S, evaluation half E, random draws and training size n. The random generator is advanced
through script 33's calls without refitting its arms, and the stored per-target results
(results/site_selection_targets.csv) supply the random mean and the DAS arms for paired contrasts.

New arms, scored on E:
  memb_near / memb_far  10 pool hospitals with the lowest / highest outcome-free membership c-statistic
                        (Debray et al. 2015 without the outcome): L2 logistic regression, standardized M1 patient
                        covariates, pool hospital's stays against the target's S stays, 5-fold stratified CV
  dasmemb_near          10 with the lowest mean of the within-pool ranks of the combined DHM angle and the
                        membership c-statistic
Each new set is subsampled to the script-33 n of that target, 3 repeats averaged, from its own generator.

--check refits near_joined for the first two targets of each outcome and compares with the stored AUROC.

Outputs (aggregate): results/membership_eicu_targets.csv, results/membership_eicu_summary.csv,
results/membership_eicu_relatedness.csv (target, pool hospital, c-statistic).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib python scripts/35_membership_eicu.py --config configs/external.yaml
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import C_M1, design_columns  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s33 = _load("s33", "33_site_selection_aligned.py")
s02, s07, s19 = s33.s02, s33.s07, s33.s19
CONDITIONS, FLAGS = s33.CONDITIONS, s33.FLAGS
K, SAMPLE, N_RAND = 10, "all", s33.N_RAND[10]


def membership_c(Xp, Xt, seed):
    """Cross-validated c-statistic of a model separating pool rows (1) from target rows (0)."""
    X = np.vstack([Xp, Xt])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xt))]
    keep = X.std(axis=0) > 0
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    p = cross_val_predict(model, X[:, keep], y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    allst = yaml.safe_load(open(base / "configs" / "screen_settings.yaml"))
    st, ms = allst["S_site_selection_aligned"], allst["S_membership_comparator"]
    assert st["status"] == "locked" and ms["status"] == "locked"

    # ---- data, exactly as script 33
    import duckdb
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
    XM = e[design_columns()].to_numpy(float)  # membership: patient covariates only
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}

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
    split = pd.read_csv(res / "target_split.csv").set_index("hospitalid").half
    stored = pd.read_csv(res / "site_selection_targets.csv")
    stored = stored[(stored["sample"] == SAMPLE) & (stored.k == K)].set_index(["task", "target"])

    trows, rel = [], []
    for task, yy in tasks.items():
        y = np.asarray(yy, int)
        g = pd.DataFrame({"h": hid, "y": y}).groupby("h").y.agg(["size", "sum"])
        pool_ok = g[(g["size"] >= s33.MIN_POOL) & (g["sum"] > 0) & (g["sum"] < g["size"])].index
        rng = np.random.default_rng([st["seed"], len(task), K, len(SAMPLE)])
        mrng = np.random.default_rng([ms["seed"], len(task)])
        pool_all = np.array(sorted(pool_ok))
        tg = [t for t in split.index if t in g.index and g.loc[t, "sum"] >= s33.MIN_EVENTS]
        if args.check:
            tg = tg[:2]
        print(f"{task}: {len(tg)} targets, pool {len(pool_all)}", flush=True)
        for t in tg:
            # ---- replay script 33 for this target
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
            joined = None
            if ang:
                joined = pd.concat(ang.values(), axis=1).mean(axis=1, skipna=True)
                sets["near_joined"] = joined.sort_values(kind="stable").index[:K].to_numpy()
                sets["far_joined"] = joined.sort_values(ascending=False, kind="stable").index[:K].to_numpy()
            if t in Z.index:
                cm = np.sqrt(((Z.reindex(pool) - Z.loc[t]) ** 2).mean(axis=1, skipna=True)).dropna()
                sets["casemix"] = cm.sort_values(kind="stable").index[:K].to_numpy()
            rand = [rng.choice(pool, K, replace=False) for _ in range(N_RAND)]
            jobs33, n_target = s19.size_jobs(sets, rand, rows_of, rng)  # advances rng as script 33 did
            st_row = stored.loc[(task, t)]
            assert int(st_row.n_train) == n_target, (task, t, st_row.n_train, n_target)

            # ---- membership relatedness of every pool hospital to the target's S stays
            cs = Parallel(n_jobs=args.jobs, batch_size=4)(
                delayed(membership_c)(XM[rows_of[h]], XM[S], int(ms["seed"]) + int(h)) for h in pool)
            c = pd.Series(cs, index=pool)
            rel += [dict(task=task, target=t, hospitalid=h, c_statistic=v) for h, v in c.items()]
            new = {"memb_near": c.sort_values(kind="stable").index[:K].to_numpy(),
                   "memb_far": c.sort_values(ascending=False, kind="stable").index[:K].to_numpy()}
            if joined is not None:
                both = pd.concat([joined.rank(), c.rank()], axis=1, join="inner").mean(axis=1)
                new["dasmemb_near"] = both.sort_values(kind="stable").index[:K].to_numpy()
            if args.check and "near_joined" in sets:
                new["check_near_joined"] = sets["near_joined"]
            jobs = []
            for a, hs in new.items():
                rows = np.concatenate([rows_of[h] for h in hs])
                for _ in range(s19.N_REPEAT):
                    jobs.append((a, rows if len(rows) <= n_target else mrng.choice(rows, n_target, replace=False)))
            preds = Parallel(n_jobs=args.jobs, batch_size=4)(
                delayed(s19.fit_scores)(X, y, rows, X[Ev], C_M1) for _, rows in jobs)
            P = {}
            for (a, _), p in zip(jobs, preds):
                P.setdefault(a, []).append(s19.auc(y[Ev], p))
            row = dict(task=task, target=t, half=split[t], n_train=n_target, random_auc=st_row.random_auc)
            for a, v in P.items():
                row[f"{a}_auc"] = float(np.mean(v))
                row[f"{a}_n_set"] = int(sum(len(rows_of[h]) for h in new[a]))
            for a in [f"near_{c_}" for c_ in CONDITIONS + ["joined"]] + ["far_joined", "casemix"]:
                row[f"{a}_auc"] = st_row.get(f"{a}_auc", np.nan)
            row["memb_near_mean_c"] = float(c.loc[new["memb_near"]].mean())
            row["dasnear_mean_c"] = float(c.reindex(sets.get("near_joined", [])).mean()) if "near_joined" in sets else np.nan
            if args.check and "check_near_joined_auc" in row:
                print(f"  check {task} {t}: refit near_joined {row['check_near_joined_auc']:.4f}, "
                      f"stored {st_row.near_joined_auc:.4f}", flush=True)
            trows.append(row)
        print(f"  done {task}", flush=True)
    if args.check:
        return
    tg = pd.DataFrame(trows)
    tg.round(5).to_csv(res / "membership_eicu_targets.csv", index=False)
    pd.DataFrame(rel).round(5).to_csv(res / "membership_eicu_relatedness.csv", index=False)

    brng = np.random.default_rng(int(ms["seed"]) + 1)
    contrasts = {"memb_near": ("memb_near_auc", "random_auc"), "memb_far": ("memb_far_auc", "random_auc"),
                 "dasmemb_near": ("dasmemb_near_auc", "random_auc"),
                 "dasmemb_near_minus_memb_near": ("dasmemb_near_auc", "memb_near_auc"),
                 "far_joined_minus_memb_far": ("far_joined_auc", "memb_far_auc")}
    contrasts |= {f"near_{c_}_minus_memb_near": (f"near_{c_}_auc", "memb_near_auc") for c_ in CONDITIONS + ["joined"]}
    srows = []
    for (task, half), s in tg.groupby(["task", "half"]):
        for name, (a, b) in contrasts.items():
            ok = s[a].notna() & s[b].notna()
            d = (s[a] - s[b])[ok]
            est, lo, hi = s19.boot_mean(d, brng)
            srows.append(dict(task=task, half=half, contrast=name, targets=int(ok.sum()), estimate=est, lo=lo, hi=hi,
                              share_positive=(d > 0).mean(), passes=bool(lo > 0)))
    sm = pd.DataFrame(srows)
    sm.round(5).to_csv(res / "membership_eicu_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
