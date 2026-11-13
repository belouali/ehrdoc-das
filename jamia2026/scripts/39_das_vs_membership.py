#!/usr/bin/env python
"""Nearest 10 by mean DAS against nearest 10 by the membership model, paired within target hospitals.

Added comparison (2026-11-09, after the S_site_selection_iecv run). Per target and outcome, the difference of the two
gains equals the difference of the two AUROCs, since both share the same random mean. Targets with both arms are pooled
by the same random-effects meta-analysis as scripts/38 (REML, Hartung-Knapp). The per-target bootstrap draws were not
stored, so the SE of the difference is computed two ways: arms independent, and correlation 0.5 between the arms
(both are scored on the same patients). An unweighted paired Wilcoxon test that uses no SE is the third check.

Outputs (aggregate): results/iecv_das_vs_membership.csv

Usage (from mimic_external/): bash scripts/iecv.sh is for 38; run this with
  uv run --no-project --with pandas --with scipy --with numpy python scripts/39_das_vs_membership.py
"""

from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
if len(sys.argv) > 1:  # results subfolder of a run, e.g. v6_2026-11-11
    RES = RES / sys.argv[1]
A, B = "near_mean", "near_membership"
MIN_EVENTS_SENS = 20


def reml(y, v):
    """Same estimator as scripts/38_iecv.py reml()."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    for _ in range(500):
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        new = max(0.0, ((w ** 2) * ((y - mu) ** 2 - v)).sum() / (w ** 2).sum() + 1 / w.sum())
        if abs(new - tau2) < 1e-12:
            tau2 = new; break
        tau2 = new
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    se = np.sqrt(max((w * (y - mu) ** 2).sum() / (k - 1), 1e-300) / w.sum())
    tq = stats.t.ppf(0.975, k - 1)
    se_pi = np.sqrt(tau2 + 1 / w.sum())
    tp = stats.t.ppf(0.975, k - 2)
    p = 2 * stats.t.sf(abs(mu / se), k - 1)
    return dict(k=k, diff=mu, lo=mu - tq * se, hi=mu + tq * se, p=p, pi_lo=mu - tp * se_pi, pi_hi=mu + tp * se_pi,
                I2=max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, share_das_better=(y > 0).mean())


def main():
    out = []
    for half in ["development", "held_out"]:
        for sample in ["all", "with_medication"]:
            df = pd.read_csv(RES / f"iecv_targets_{half}_{sample}.csv")
            for task, s in df.groupby("task"):
                for analysis, d in [("main", s), ("ge20_events", s[s.events >= MIN_EVENTS_SENS])]:
                    ok = d[f"{A}_gain"].notna() & d[f"{B}_gain"].notna() & (d[f"{A}_se"] > 0) & (d[f"{B}_se"] > 0)
                    d = d[ok]
                    y = d[f"{A}_gain"] - d[f"{B}_gain"]
                    va, vb = d[f"{A}_se"] ** 2, d[f"{B}_se"] ** 2
                    for se_rule, v in [("independent", va + vb), ("corr_0.5", va + vb - np.sqrt(va * vb))]:
                        r = reml(y, v)
                        out.append(dict(half=half, sample=sample, task=task, analysis=analysis, se_rule=se_rule, **r,
                                        wilcoxon_p=stats.wilcoxon(y).pvalue))
    res = pd.DataFrame(out)
    res.to_csv(RES / "iecv_das_vs_membership.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
