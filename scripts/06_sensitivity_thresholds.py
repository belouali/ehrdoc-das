#!/usr/bin/env python
"""Sensitivity analysis: concordance stability across min-patient thresholds.

Runs scripts 02-04 at multiple thresholds (e.g., 100, 200, 300, 500) and
produces a summary table + plot showing how concordance changes with threshold.

Prereqs:
  - Run scripts/01_build_flags.py --conditions all

Outputs:
  artifacts/sensitivity/threshold_concordance.csv
  artifacts/sensitivity/threshold_concordance.png
  artifacts/sensitivity/hospital_counts_by_threshold.csv

Usage:
  python scripts/06_sensitivity_thresholds.py
  python scripts/06_sensitivity_thresholds.py --thresholds 100 200 300 400 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from ehrdoc.venn.vectors import hospital_venn_counts, hospital_venn_vectors
from ehrdoc.distances.matrices import distance_matrix
from ehrdoc.distances.concordance import distance_matrix_concordance
from ehrdoc.utils.io import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--cohorts", default="configs/cohorts.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes", "hypertension", "chf", "af"])
    ap.add_argument("--thresholds", nargs="+", type=int, default=None,
                    help="Override thresholds (else use configs/cohorts.yaml sensitivity_thresholds)")
    ap.add_argument("--n-perm", type=int, default=999, help="Mantel permutations per pair")
    args = ap.parse_args()

    with open(args.paths, "r") as f:
        paths_cfg = yaml.safe_load(f)
    processed = Path(paths_cfg["outputs"]["processed"])

    with open(args.cohorts, "r") as f:
        cohort_cfg = yaml.safe_load(f)

    thresholds = args.thresholds or cohort_cfg.get("sensitivity_thresholds", [100, 200, 300, 500])
    art_dir = ensure_dir(Path(paths_cfg["outputs"]["artifacts"]) / "sensitivity")

    conditions = args.conditions

    # ── Step 1: For each threshold × condition, build vectors & distances ──
    hospital_counts = []  # track how many hospitals at each threshold
    all_results = []

    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold: min {thresh} patients per hospital per condition")
        print(f"{'='*60}")

        dist_mats = {}  # condition -> (D, ids)

        for cond in conditions:
            flags_path = processed / f"flags_{cond}.parquet"
            if not flags_path.exists():
                print(f"  WARNING: {flags_path} not found, skipping {cond}")
                continue

            flags = pd.read_parquet(flags_path)
            counts = hospital_venn_counts(flags, min_patients_per_hospital=thresh)
            n_hosp = counts["hospitalid"].nunique()
            hospital_counts.append({"condition": cond, "threshold": thresh, "n_hospitals": n_hosp})

            if n_hosp < 3:
                print(f"  {cond}: only {n_hosp} hospitals (need ≥3), skipping")
                continue

            vectors = hospital_venn_vectors(counts)
            D, ids = distance_matrix(vectors)
            dist_mats[cond] = (D, ids)
            print(f"  {cond}: {n_hosp} hospitals, {n_hosp*(n_hosp-1)//2} pairs")

        # ── Step 2: Compute concordance for all condition pairs ──
        for (a, b) in combinations(conditions, 2):
            if a not in dist_mats or b not in dist_mats:
                all_results.append({
                    "threshold": thresh, "condition_1": a, "condition_2": b,
                    "n_hospitals_overlap": 0, "n_pairs": 0,
                    "spearman_rho": np.nan, "p_value_naive": np.nan, "mantel_p": np.nan,
                })
                continue

            Da, ida = dist_mats[a]
            Db, idb = dist_mats[b]
            out = distance_matrix_concordance(Da, ida, Db, idb, n_perm=args.n_perm)
            all_results.append({
                "threshold": thresh, "condition_1": a, "condition_2": b,
                "n_hospitals_overlap": out["n_hospitals"],
                "n_pairs": out.get("n_pairs", 0),
                "spearman_rho": out["rho"],
                "p_value_naive": out["p"],
                "mantel_p": out["mantel_p"],
            })
            print(f"  {a} vs {b}: ρ={out['rho']:.3f} (Mantel p={out['mantel_p']:.4f}, "
                  f"n_overlap={out['n_hospitals']})")

    # ── Step 3: Save results ──
    results_df = pd.DataFrame(all_results)
    out_csv = art_dir / "threshold_concordance.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    hosp_df = pd.DataFrame(hospital_counts)
    out_hosp = art_dir / "hospital_counts_by_threshold.csv"
    hosp_df.to_csv(out_hosp, index=False)
    print(f"Wrote {out_hosp}")

    # ── Step 4: Plot concordance vs threshold ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Spearman rho by threshold for each condition pair
    ax = axes[0]
    for (a, b), grp in results_df.groupby(["condition_1", "condition_2"]):
        grp = grp.sort_values("threshold")
        ax.plot(grp["threshold"], grp["spearman_rho"], "o-", label=f"{a} vs {b}", markersize=5)
    ax.set_xlabel("Minimum patients with condition per hospital")
    ax.set_ylabel("Spearman ρ (pairwise angular separation concordance)")
    ax.set_title("Cross-condition concordance stability\nacross minimum patient thresholds")
    ax.legend(fontsize=7, loc="lower right", title="Condition pair")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.6, color="gray", linestyle="--", alpha=0.5, label="ρ=0.6")
    ax.grid(alpha=0.3)

    # Panel B: Number of hospitals by threshold
    ax2 = axes[1]
    for cond, grp in hosp_df.groupby("condition"):
        grp = grp.sort_values("threshold")
        ax2.plot(grp["threshold"], grp["n_hospitals"], "s-", label=cond, markersize=5)
    ax2.set_xlabel("Minimum patients with condition per hospital")
    ax2.set_ylabel("Number of hospitals meeting threshold")
    ax2.set_title("Hospital inclusion\nby condition and threshold")
    ax2.legend(fontsize=8, title="Condition")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png = art_dir / "threshold_concordance.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: Mean concordance (Spearman ρ) by threshold")
    print("="*60)
    summary = results_df.groupby("threshold")["spearman_rho"].agg(["mean", "min", "max", "count"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
