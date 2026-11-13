#!/usr/bin/env python
"""Split-half selection ceiling on the v6 site-selection design (2026-11-11, author's request).

Same targets, pools, inputs, outcomes and set sizes as 38_iecv.py --spec v6 (development half). For each target and
outcome, the target's stays are split 50/50 within outcome class into a selection half S and an evaluation half E.
Every arm is scored on E only:
  random           200 random sets of 10 (the reference)
  set_best/worst   the random set with the highest or lowest AUROC on S
  rank_best/worst  each pool hospital's own model ranked by AUROC on S, top or bottom 10 pooled
  near_<measure>   the DAS and membership sets of the v6 loop, for comparison on the same E
Gain = arm AUROC on E minus the random mean on E; standard error from 1,000 stratified bootstrap resamples of E.
Pooled by REML with Hartung-Knapp intervals as 38_iecv.py.

Writes results/v6_2026-11-11/ceiling_targets_development_<sample>.csv (resumable) and ceiling_pooled.csv.
Usage (from mimic_external/):
  bash scripts/ceiling.sh --sample all [--limit 1 --boot 50]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("iecv", HERE / "38_iecv.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

OUT = "v6_2026-11-11"


def solo_fit(X, y, rows, Xt):
    if len(rows) == 0 or y[rows].min() == y[rows].max():
        return None
    return M.s19.fit_scores(X, y, rows, Xt, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["all", "with_medication"], default="all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--pool-only", action="store_true")
    args = ap.parse_args()
    M.TASKS = M.TASKS + ["icu_readmit", "disch_facility"]
    M.DAS_MIN[0] = 30
    M.setup.out = OUT
    cfg, st, res, rowlevel = M.setup("configs/external.yaml")
    base = res / "_test" if args.limit else res
    base.mkdir(exist_ok=True)
    if args.pool_only:
        return pool(res)
    split = pd.read_csv(res / "iecv_split.csv")
    e, X, XM, Y, rows_of, V, no_med = M.load_eicu(cfg, rowlevel)
    f24 = pd.read_parquet(rowlevel / "eicu_flags_24h.parquet")
    g = e[["patientunitstayid"]].merge(f24, on="patientunitstayid", how="left")
    assert (g.patientunitstayid.to_numpy() == e.patientunitstayid.to_numpy()).all()
    Xfit = np.hstack([XM, g[M.FLAGS].fillna(False).astype(float).to_numpy()])
    half = split[split.half == "development"]
    if args.sample == "with_medication":
        half = half[~half.no_medication]
    members = half.hospitalid.to_numpy()
    das_ok = set().union(*[set(V[c].index) for c in M.CONDITIONS])
    out = base / f"ceiling_targets_development_{args.sample}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["target", "task"])
    keys = set(zip(done.target, done.task))
    seed = int(st["seed"]) + 4900
    scode = {"all": 1, "with_medication": 2}[args.sample]
    for t in (members[: args.limit] if args.limit else members):
        if all((t, k) in keys for k in M.TASKS):
            continue
        pool_h = np.array([h for h in members[members != t] if h in das_ok])
        rt = rows_of[t]
        tvec = {c: V[c].loc[t].to_numpy(float) for c in M.CONDITIONS if t in V[c].index}
        ang = M.das_rankings(V, tvec, pool_h)
        cs = Parallel(n_jobs=args.jobs)(delayed(M.s35.membership_c)(XM[rows_of[h]], XM[rt], int(st["seed"]) + int(h))
                                        for h in pool_h)
        base_sets = M.make_sets(ang, pd.Series(cs, index=pool_h))
        base_sets = {k: v for k, v in base_sets.items() if k.startswith("near_")}
        rows = []
        for ti, task in enumerate(M.TASKS):
            if (t, task) in keys:
                continue
            if task in M.ELIG:
                el = M.ELIG[task]
                rtk, rows_k, Xk = rt[el[rt]], {h: rows_of[h][el[rows_of[h]]] for h in pool_h}, M.XDIS[0]
            else:
                rtk, rows_k, Xk = rt, rows_of, Xfit
            y, yt = Y[task], Y[task][rtk]
            memb_s = pd.Series(cs, index=pool_h)
            pool_k, bs = M.task_pool(task, pool_h, base_sets, ang, memb_s)
            row = dict(target=t, task=task, sample=args.sample, stays=len(rtk), events=int(yt.sum()), pool=len(pool_k))
            rng = np.random.default_rng([seed, scode, ti, int(t)])
            pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
            if len(pos) < 2 or len(neg) < 2:
                rows.append({**row, "skipped": "fewer than 2 events or non-events"}); continue
            pos, neg = rng.permutation(pos), rng.permutation(neg)
            S = np.sort(np.r_[pos[: len(pos) // 2], neg[: len(neg) // 2]])
            E = np.sort(np.r_[pos[len(pos) // 2:], neg[len(neg) // 2:]])
            # rank arms: each pool hospital alone, AUROC on S
            # single-hospital models with C = 1.0 (as the earlier ceiling): one hospital is too small for an unpenalized fit
            solo = Parallel(n_jobs=args.jobs)(delayed(solo_fit)(Xk, y, rows_k[h], Xk[rtk[S]]) for h in pool_k)
            sa = pd.Series({h: (M.s19.auc(yt[S], p) if p is not None else np.nan) for h, p in zip(pool_k, solo)}).dropna()
            sets = dict(bs)
            sets["rank_best"] = sa.sort_values(ascending=False, kind="stable").index[: M.K].to_numpy()
            sets["rank_worst"] = sa.sort_values(kind="stable").index[: M.K].to_numpy()
            rand = [rng.choice(pool_k, M.K, replace=False) for _ in range(M.N_RAND["eicu"])]
            jobs, n = M.size_jobs(sets, rand, rows_k, rng, "v6")
            fits = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(M.fit_one)(Xk, y, r, Xk[rtk], yt) for _, r in jobs)
            ok = [i for i, f in enumerate(fits) if f[0] is not None]
            arms = [jobs[i][0] for i in ok]
            P = np.vstack([fits[i][0] for i in ok])
            ri = [i for i, a in enumerate(arms) if a.startswith("rand_")]
            aS = np.array([M.s19.auc(yt[S], P[i, S]) for i in ri])
            best, worst = ri[int(np.argmax(aS))], ri[int(np.argmin(aS))]
            arms_e = arms + ["set_best", "set_worst"]
            PE = np.vstack([P[:, E], P[best, E], P[worst, E]])
            point, rmean, rauc, draws, auc0 = M.score_target(arms_e, PE, yt[E], args.boot or M.N_BOOT["eicu"], rng)
            row.update(n_train=n, random_auc=rmean, random_sd=rauc.std(), random_citl=np.nan, random_slope=np.nan)
            for a in point:
                row.update({f"{a}_auc": point[a], f"{a}_gain": point[a] - rmean, f"{a}_se": draws[a].std(ddof=1),
                            f"{a}_citl": np.nan, f"{a}_slope": np.nan})
            rows.append(row)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(out, index=False, float_format="%.6f")
        print(f"target {t} done ({done.target.nunique()}/{len(members)})", flush=True)


def pool(res):
    allp = []
    for s in ["all", "with_medication"]:
        f = res / f"ceiling_targets_development_{s}.csv"
        if f.exists():
            p = M.pool_file(pd.read_csv(f))
            p.insert(0, "sample", s)
            allp.append(p)
    d = pd.concat(allp, ignore_index=True)
    d = d[d.analysis == "main"].drop(columns=["analysis", "citl", "slope", "random_citl", "random_slope"])
    d.to_csv(res / "ceiling_pooled.csv", index=False, float_format="%.6f")
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
