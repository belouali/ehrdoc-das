#!/usr/bin/env python
"""Percentile of the reported (trim-mean) AUROC among the random sets, added to an existing run (2026-11-12).

38_iecv.py stored each selected set's percentile as the mean, over its 10 trims, of the share of random sets with a lower
AUROC than that trim ({arm}_pct, percentile). The paper reports the gain of the trim-mean AUROC, so it now reports the
share of random sets with a strictly lower AUROC than that trim-mean AUROC ({arm}_pctm, percentile_mean), which 38_iecv.py
also writes from now on. This script computes it from the set-level AUROCs the run saved, after checking that the same
file reproduces every stored gain and stored per-trim percentile. Nothing else in the result files changes.

Usage (from mimic_external/):
  uv run --no-project --with pandas --with numpy python scripts/52_percentile_of_mean.py v6_2026-11-11
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / sys.argv[1]
BACKUP = RES / "_before_percentile_of_mean"
TOL = 1e-6
TOL_PCT = 0.05  # set AUROCs are saved to 6 decimals; rounding can flip one random-set comparison (0.02 points for 500 x 10)


def summarize(sa):
    """Per (keys..., arm): trim-mean AUROC, per-trim percentile, percentile of the mean, random mean."""
    out = {}
    rand_mask = sa.set.str.startswith("rand_")
    keys = [c for c in sa.columns if c not in ("set", "auroc")]
    for k, g in sa.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        rnd = g.loc[rand_mask.loc[g.index], "auroc"].to_numpy()
        for arm, a in g.loc[~rand_mask.loc[g.index]].groupby("set", sort=False):
            v = a.auroc.to_numpy()
            out[k + (arm,)] = dict(mean=v.mean(), rmean=rnd.mean(), n_rand=len(rnd), n_trim=len(v),
                                   pct=100 * np.mean([(rnd < x).mean() for x in v]),
                                   pctm=100 * (rnd < v.mean()).mean())
    return out


def backup(f):
    BACKUP.mkdir(exist_ok=True)
    if not (BACKUP / f.name).exists():
        shutil.copy2(f, BACKUP / f.name)


def do_targets(half, sample):
    f, fs = RES / f"iecv_targets_{half}_{sample}.csv", RES / f"iecv_set_aucs_{half}_{sample}.csv"
    if not (f.exists() and fs.exists()):
        return
    t, S = pd.read_csv(f), summarize(pd.read_csv(fs))
    arms = sorted({k[-1] for k in S})
    worst_g = worst_p = 0.0
    for a in arms:
        col_m = []
        for _, r in t.iterrows():
            s = S.get((r.target, r.task, a))
            if s is None or pd.isna(r.get(f"{a}_gain")):
                col_m.append(np.nan); continue
            worst_g = max(worst_g, abs((s["mean"] - s["rmean"]) - r[f"{a}_gain"]))
            worst_p = max(worst_p, abs(s["pct"] - r[f"{a}_pct"]))
            col_m.append(s["pctm"])
        t[f"{a}_pctm"] = col_m
    assert worst_g < TOL and worst_p < TOL_PCT, f"{f.name}: set AUROCs do not reproduce stored values (gain {worst_g}, pct {worst_p})"
    backup(f)
    t.to_csv(f, index=False, float_format="%.6f")
    print(f"{f.name}: {len(arms)} arms, stored gains and per-trim percentiles reproduced (max diff {worst_g:.1e}, {worst_p:.1e})")


def do_mimic(tag):
    f, fs = RES / f"iecv_mimic{tag}.csv", RES / f"iecv_set_aucs_mimic{tag}.csv"
    if not (f.exists() and fs.exists()):
        return
    m, S = pd.read_csv(f), summarize(pd.read_csv(fs))
    pm, worst_g, worst_p = [], 0.0, 0.0
    for _, r in m.iterrows():
        s = S[(r.task, r.arm)]
        worst_g = max(worst_g, abs((s["mean"] - s["rmean"]) - r.gain))
        worst_p = max(worst_p, abs(s["pct"] - r.percentile))
        pm.append(s["pctm"])
    assert worst_g < TOL and worst_p < TOL_PCT, f"{f.name}: set AUROCs do not reproduce stored values (gain {worst_g}, pct {worst_p})"
    m["percentile_mean"] = pm
    backup(f)
    m.to_csv(f, index=False, float_format="%.6f")
    print(f"{f.name}: stored gains and percentiles reproduced (max diff {worst_g:.1e}, {worst_p:.1e}); percentile_mean added")


if __name__ == "__main__":
    for half in ["development", "held_out"]:
        for sample in ["all", "with_medication"]:
            do_targets(half, sample)
    for tag in ["", "_mean", "_withmed", "_mean_withmed"]:
        do_mimic(tag)
