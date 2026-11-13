#!/usr/bin/env python
"""Percentile of each selected training set among the 200 random sets, per held-out eICU target (added 2026-11-10).

Same definition as the MIMIC-IV percentile in scripts/38_iecv.py (cmd_mimic): the share of random sets with a lower
AUROC than the selected set. Reads the rerun of the held-out loop with --save-random random_aucs, first checking
that the rerun reproduces the stored random mean and every arm's AUROC; then reports, per outcome and arm, the
median percentile across targets with its interquartile range.

Outputs (aggregate): results/iecv_percentiles.csv
Usage (from mimic_external/): uv run --no-project --with pandas python scripts/44_iecv_percentiles.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
ARMS = ["near_mean", "far_mean", "near_membership", "far_membership",
        *[f"{s}_{c}" for c in ["diabetes", "hypertension", "chf", "af"] for s in ("near", "far")]]

stored = pd.read_csv(RES / "iecv_targets_held_out_all.csv")
rerun = pd.read_csv(RES / "random_aucs" / "iecv_targets_held_out_all.csv")
rand = pd.read_csv(RES / "random_aucs" / "iecv_random_aucs_held_out_all.csv").drop_duplicates(["target", "task", "set"])

m = stored.merge(rerun, on=["target", "task"], suffixes=("", "_rerun"))
assert len(m) == len(stored), f"rerun has {len(m)} of {len(stored)} target-outcome rows"
for c in ["random_auc"] + [f"{a}_auc" for a in ARMS]:
    d = np.nanmax(np.abs(m[c] - m[f"{c}_rerun"]))
    assert d < 1e-6, f"{c}: rerun differs from stored by {d}"
g = rand.groupby(["target", "task"]).auroc
chk = stored.set_index(["target", "task"]).join(g.mean().rename("rmean"), how="inner")
assert np.abs(chk.random_auc - chk.rmean).max() < 1e-6, "saved random AUROCs do not reproduce the stored random mean"
assert (g.size() == 200).all(), "not every target has 200 random sets"

rows = []
for (t, task), r in rand.groupby(["target", "task"]):
    s = stored[(stored.target == t) & (stored.task == task)].iloc[0]
    for a in ARMS:
        if pd.notna(s[f"{a}_auc"]):
            rows.append(dict(target=t, task=task, arm=a, percentile=100 * (r.auroc.to_numpy() < s[f"{a}_auc"]).mean()))
per = pd.DataFrame(rows)
out = per.groupby(["task", "arm"]).percentile.agg(
    targets="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
    share_above_median=lambda x: (x > 50).mean()).reset_index()
out.to_csv(RES / "iecv_percentiles.csv", index=False, float_format="%.4f")
print(out[out.arm.isin(["near_mean", "near_membership", "far_mean"])].round(1).to_string(index=False))
