#!/usr/bin/env python
"""Multivariable PERMANOVA (marginal tests) and PERMDISP for hospital descriptors.

For each condition and sample (all hospitals; excluding hospitals with an
unpopulated source), fits region + bed category + teaching status jointly to
the DAS matrix and reports each term's marginal pseudo-F, R2 and permutation p
(999 hospital-label permutations), plus the joint model R2. PERMDISP tests
homogeneity of multivariate dispersion for each descriptor separately.

Writes: artifacts/general/permanova_multivariable.csv
        artifacts/general/permdisp.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ehrdoc.distances.permanova import permanova_multivariable, permdisp
from ehrdoc.utils.io import ensure_dir
from ehrdoc.venn.vectors import zero_source_hospitals

COND = ["diabetes", "hypertension", "chf", "af"]
TERMS = ["region", "numbedscategory", "teachingstatus"]
NICE = {"region": "Region", "numbedscategory": "Bed category", "teachingstatus": "Teaching status"}


def _latest(processed: Path, pattern: str) -> Path:
    cand = sorted(processed.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not cand:
        raise FileNotFoundError(pattern)
    return cand[-1]


def hospital_chars(flags: pd.DataFrame) -> pd.DataFrame:
    cols = ["hospitalid"] + [c for c in TERMS if c in flags.columns]
    h = flags[cols].drop_duplicates("hospitalid").set_index("hospitalid")
    if "teachingstatus" in h.columns:
        h["teachingstatus"] = h["teachingstatus"].apply(lambda x: "Teaching" if str(x).lower() in ("t", "true", "1", "yes") else "Non-teaching")
    for c in h.columns:
        h[c] = h[c].where(h[c].astype(str).str.lower() != "unknown")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--n-perm", type=int, default=999)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.paths))
    processed = Path(cfg["outputs"]["processed"]); art = Path(cfg["outputs"]["artifacts"])
    general = ensure_dir(art / "general")
    perm_rows, disp_rows = [], []
    for c in COND:
        z = np.load(_latest(processed, f"dist_{c}_min*.npz"), allow_pickle=True)
        D, ids = z["D"], [int(i) for i in z["ids"]]
        flags = pd.read_parquet(processed / f"flags_{c}.parquet")
        chars = hospital_chars(flags).reindex(ids)
        counts = pd.read_parquet(_latest(processed, f"counts_{c}_min*.parquet"))
        zs = set().union(*[set(v) for v in zero_source_hospitals(counts).values()])
        for sample in ["all_hospitals", "excluding_zero_source"]:
            keep = np.array([True] * len(ids)) if sample == "all_hospitals" else np.array([h not in zs for h in ids])
            Ds = D[np.ix_(keep, keep)]; ch = chars[keep]
            terms = [t for t in TERMS if t in ch.columns]
            res = permanova_multivariable(Ds, ch, terms, n_perm=args.n_perm, seed=0)
            res.insert(0, "sample", sample); res.insert(0, "condition", c)
            res["term"] = res["term"].map(lambda t: NICE.get(t, t))
            perm_rows.append(res)
            for t in terms:
                pdp = permdisp(Ds, ch[t].values, n_perm=args.n_perm, seed=0)
                disp_rows.append({"condition": c, "sample": sample, "characteristic": NICE[t], "n_hospitals": pdp["n_hospitals"],
                                  "n_groups": pdp["n_groups"], "F": pdp["F"], "p_value": pdp["p_value"],
                                  "group_mean_dispersion": "; ".join(f"{k}: {v:.1f} (n={pdp['group_n'][k]})" for k, v in pdp["group_mean_dispersion"].items())})
            print(c, sample, "PERMANOVA:", "; ".join(f"{r.term} R2={r.R2_marginal:.3f} p={r.p_value}" for r in res.itertuples() if r.term not in ("Residual",)))
            print("   PERMDISP:", "; ".join(f"{d['characteristic']} F={d['F']:.2f} p={d['p_value']}" for d in disp_rows[-len(terms):]))
    pd.concat(perm_rows).to_csv(general / "permanova_multivariable.csv", index=False)
    pd.DataFrame(disp_rows).to_csv(general / "permdisp.csv", index=False)
    print("wrote", general / "permanova_multivariable.csv", "and permdisp.csv")


if __name__ == "__main__":
    main()
