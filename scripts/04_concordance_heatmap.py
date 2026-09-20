#!/usr/bin/env python
"""Compute cross-condition concordance (Spearman rho) and plot correlation heatmap.

Inputs (from 03_compute_distances.py):
  data/processed/dist_<condition>_*.npz

Outputs:
  artifacts/concordance/concordance_rho.csv
  artifacts/concordance/concordance_rho.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ehrdoc.distances.concordance import distance_matrix_concordance
from ehrdoc.viz.heatmaps import plot_matrix_heatmap
from ehrdoc.utils.io import ensure_dir


def load_npz(path: Path) -> tuple[np.ndarray, list[int]]:
    z = np.load(path, allow_pickle=True)
    D = z["D"]
    ids = z["ids"].tolist()
    return D, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes", "hypertension", "chf", "af"])
    ap.add_argument("--pattern", default=None,
                    help="Optional glob pattern for dist files, e.g. 'dist_*_min500.npz'.")
    ap.add_argument("--n-perm", type=int, default=1999,
                    help="Mantel permutations (hospital-label permutation test).")
    args = ap.parse_args()

    with open(args.paths, "r") as f:
        paths_cfg = yaml.safe_load(f)
    processed = Path(paths_cfg["outputs"]["processed"])
    art_root = ensure_dir(Path(paths_cfg["outputs"]["artifacts"]) / "concordance")

    # discover distance files
    if args.pattern:
        dist_files = sorted(processed.glob(args.pattern))
    else:
        dist_files = sorted(processed.glob("dist_*.npz"))

    # map condition->best file (last alphabetically tends to pick largest threshold)
    cond_to_file: dict[str, Path] = {}
    for fpath in dist_files:
        name = fpath.name.replace("dist_", "")
        cond = name.split("_min")[0].split(".")[0]
        cond_to_file[cond] = fpath

    conditions = [c for c in args.conditions if c in cond_to_file]
    if len(conditions) < 2:
        raise RuntimeError(f"Need at least 2 conditions with distance matrices. Found: {sorted(cond_to_file)}")

    mats = {}
    for c in conditions:
        mats[c] = load_npz(cond_to_file[c])

    rho = np.full((len(conditions), len(conditions)), np.nan)
    pvals = np.full((len(conditions), len(conditions)), np.nan)
    mantel = np.full((len(conditions), len(conditions)), np.nan)
    n_hosp = np.zeros_like(rho)

    for i, a in enumerate(conditions):
        Da, ida = mats[a]
        for j, b in enumerate(conditions):
            if i == j:
                rho[i, j] = 1.0
                pvals[i, j] = 0.0
                mantel[i, j] = 0.0
                n_hosp[i, j] = len(ida)
                continue
            if j < i:
                # symmetric: reuse the (i, j) computation
                rho[i, j], pvals[i, j], mantel[i, j], n_hosp[i, j] = rho[j, i], pvals[j, i], mantel[j, i], n_hosp[j, i]
                continue
            Db, idb = mats[b]
            out = distance_matrix_concordance(Da, ida, Db, idb, n_perm=args.n_perm)
            rho[i, j] = out.get("rho", np.nan)
            pvals[i, j] = out.get("p", np.nan)
            mantel[i, j] = out.get("mantel_p", np.nan)
            n_hosp[i, j] = out.get("n_hospitals", 0)
            print(f"  {a} vs {b}: rho={rho[i, j]:.3f}, Mantel p={mantel[i, j]:.4f} "
                  f"(n_hospitals={int(n_hosp[i, j])}, {args.n_perm} permutations)")

    df = pd.DataFrame(rho, index=conditions, columns=conditions)
    out_csv = art_root / "concordance_rho.csv"
    df.to_csv(out_csv)

    # Naive pair-level p-values are kept for reference only; report the Mantel p.
    df_p = pd.DataFrame(pvals, index=conditions, columns=conditions)
    out_pcsv = art_root / "concordance_pvalues_naive.csv"
    df_p.to_csv(out_pcsv)

    df_m = pd.DataFrame(mantel, index=conditions, columns=conditions)
    out_mcsv = art_root / "concordance_mantel_p.csv"
    df_m.to_csv(out_mcsv)
    print(f"Wrote {out_mcsv}")

    df_n = pd.DataFrame(n_hosp.astype(int), index=conditions, columns=conditions)
    out_ncsv = art_root / "concordance_n_hospitals.csv"
    df_n.to_csv(out_ncsv)

    fig, _ = plot_matrix_heatmap(rho, labels=conditions,
                                title="Cross-condition concordance of pairwise angular separations\n"
                                      f"(Spearman ρ over hospital pairs; Mantel test, {args.n_perm} permutations)")
    out_png = art_root / "concordance_rho.png"
    fig.savefig(out_png, dpi=300)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_pcsv}")
    print(f"Wrote {out_ncsv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
