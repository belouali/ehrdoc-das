#!/usr/bin/env python
"""Size adjustment of the nearest-versus-random transportability comparison.

Nearest-k training sets tend to contain more stays than random-k sets of the
same number of hospitals. For each condition, k and metric, this script
regresses each held-out hospital's (nearest - random) metric difference on its
(nearest - random) training-stay difference (per 1,000 stays) and reports the
intercept as the size-adjusted difference, with 95% CI.

Reads:  artifacts/<cond>/tables/transportability_results.csv (all hospitals, large)
        artifacts/sensitivity_zero_source/transportability_zero_source_<cond>.csv
Writes: artifacts/general/transport_size_adjustment.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import yaml

from ehrdoc.utils.io import ensure_dir

COND = ["diabetes", "hypertension", "chf", "af"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()
    art = Path(yaml.safe_load(open(args.paths))["outputs"]["artifacts"])
    rows = []
    sources = [("all_hospitals", lambda c: art / c / "tables" / "transportability_results.csv"),
               ("excluding_zero_source", lambda c: art / "sensitivity_zero_source" / f"transportability_zero_source_{c}.csv")]
    for sample, pth in sources:
        for c in COND:
            p = pth(c)
            if not p.exists():
                continue
            d = pd.read_csv(p)
            if "size_group" in d:
                d = d[d["size_group"] == "large"]
            for k in [5, 10, 20]:
                for met in ["AUC", "Brier", "AUPRC"]:
                    cols = [f"near{k}_{met}", f"rand{k}_{met}", f"near{k}_n_train", f"rand{k}_n_train"]
                    if not all(x in d.columns for x in cols):
                        continue
                    dd = d.dropna(subset=cols)
                    if len(dd) < 10:
                        continue
                    sign = -1.0 if met == "Brier" else 1.0  # positive favours nearest
                    dy = sign * (dd[cols[0]] - dd[cols[1]])
                    dx = (dd[cols[2]] - dd[cols[3]]) / 1000.0
                    m = sm.OLS(dy, sm.add_constant(dx)).fit()
                    ci = m.conf_int()
                    rows.append(dict(sample=sample, condition=c, k=k, metric=met, n=len(dd),
                                     mean_diff=dy.mean(), mean_dstays=dx.mean() * 1000,
                                     median_n_train_nearest=dd[cols[2]].median(), median_n_train_random=dd[cols[3]].median(),
                                     slope_per_1000=m.params.iloc[1], slope_p=m.pvalues.iloc[1],
                                     size_adjusted_diff=m.params.iloc[0], size_adjusted_lo=ci.iloc[0, 0],
                                     size_adjusted_hi=ci.iloc[0, 1], size_adjusted_p=m.pvalues.iloc[0],
                                     share_retained=(m.params.iloc[0] / dy.mean()) if dy.mean() != 0 else float("nan")))
    out = ensure_dir(art / "general") / "transport_size_adjustment.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
