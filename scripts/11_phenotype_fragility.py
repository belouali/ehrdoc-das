#!/usr/bin/env python
"""Phenotype fragility analysis.

Shows that the same phenotype definition captures different proportions
of patients at different hospitals, and that DAS predicts this variation.

For each hospital, computes capture rates under multiple phenotype definitions:
  - Dx only: patient has condition in Diagnosis table
  - PH only: patient has condition in Past History table
  - Rx only: patient has condition in Medication table
  - Dx OR PH: either source
  - Dx OR Rx: either source
  - Full union (Dx OR PH OR Rx): most inclusive
  - 2-of-3: requires evidence in at least 2 sources (higher specificity)
  - All 3: all three sources agree

Then correlates cross-hospital variation in capture rates with DAS.

Prereqs:
  - Run scripts 01-03

Outputs (per condition, in artifacts/<condition>/tables/):
  phenotype_fragility.csv         -- per-hospital capture rates
  phenotype_fragility_corr.csv    -- DAS correlations with capture rates
Figures:
  phenotype_fragility.png         -- scatter: DAS vs capture rate per definition
  phenotype_source_dominance.png  -- which source dominates at each hospital

Usage:
  python scripts/11_phenotype_fragility.py
  python scripts/11_phenotype_fragility.py --conditions diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from ehrdoc.utils.io import ensure_dir


def compute_fragility(flags_df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Compute per-hospital capture rates under different phenotype definitions."""

    # The flags file uses standard column names from build_patient_flags
    col_map = {
        "dx": "exists_in_diagnosis",
        "ph": "exists_in_pastHistory",
        "rx": "exists_in_medication",
    }

    dx_col = col_map["dx"]
    ph_col = col_map["ph"]
    rx_col = col_map["rx"]

    # Check columns exist
    for col in [dx_col, ph_col, rx_col]:
        if col not in flags_df.columns:
            raise ValueError(f"Column {col} not found. Available: {list(flags_df.columns)}")

    # Patient has ANY evidence of the condition (used as denominator)
    flags_df["any_evidence"] = (
        flags_df[dx_col].fillna(0).astype(bool) |
        flags_df[ph_col].fillna(0).astype(bool) |
        flags_df[rx_col].fillna(0).astype(bool)
    ).astype(int)

    # Count sources per patient
    flags_df["n_sources"] = (
        flags_df[dx_col].fillna(0).astype(bool).astype(int) +
        flags_df[ph_col].fillna(0).astype(bool).astype(int) +
        flags_df[rx_col].fillna(0).astype(bool).astype(int)
    )

    # Define phenotype definitions
    definitions = {
        "Dx_only": flags_df[dx_col].fillna(0).astype(bool).astype(int),
        "PH_only": flags_df[ph_col].fillna(0).astype(bool).astype(int),
        "Rx_only": flags_df[rx_col].fillna(0).astype(bool).astype(int),
        "Dx_or_PH": (flags_df[dx_col].fillna(0).astype(bool) |
                      flags_df[ph_col].fillna(0).astype(bool)).astype(int),
        "Dx_or_Rx": (flags_df[dx_col].fillna(0).astype(bool) |
                      flags_df[rx_col].fillna(0).astype(bool)).astype(int),
        "Full_union": flags_df["any_evidence"],
        "Two_of_three": (flags_df["n_sources"] >= 2).astype(int),
        "All_three": (flags_df["n_sources"] == 3).astype(int),
    }

    # Add all definitions as columns
    for name, series in definitions.items():
        flags_df[f"pheno_{name}"] = series

    # Compute per-hospital: N with any evidence, and capture rate for each definition
    # Only include patients with any evidence (denominator)
    has_evidence = flags_df[flags_df["any_evidence"] == 1]

    results = []
    for hid, grp in has_evidence.groupby("hospitalid"):
        row = {"hospitalid": hid, "n_with_any_evidence": len(grp)}
        for name in definitions:
            col = f"pheno_{name}"
            row[f"capture_{name}"] = grp[col].mean()
            row[f"count_{name}"] = grp[col].sum()
        results.append(row)

    return pd.DataFrame(results)


def run_fragility_analysis(condition: str, processed: Path, art_root: Path,
                            ref_hospital: int):
    """Run fragility analysis for one condition."""
    fig_dir = ensure_dir(art_root / condition / "figures")
    tbl_dir = ensure_dir(art_root / condition / "tables")

    # Load patient flags
    flags_file = processed / f"flags_{condition}.parquet"
    if not flags_file.exists():
        # Try CSV
        flags_file = processed / f"flags_{condition}.csv"
    if not flags_file.exists():
        print(f"  No flags file for {condition}, skipping")
        return

    if str(flags_file).endswith(".parquet"):
        flags_df = pd.read_parquet(flags_file)
    else:
        flags_df = pd.read_csv(flags_file)

    print(f"  Loaded {len(flags_df):,} patient flags, {flags_df['hospitalid'].nunique()} hospitals")

    # Compute fragility
    frag_df = compute_fragility(flags_df, condition)
    print(f"  Computed capture rates for {len(frag_df)} hospitals")

    # Load DAS
    dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                        key=lambda p: p.stat().st_mtime)
    if not dist_files:
        print(f"  No distance matrix for {condition}")
        return

    ref_dists = pd.read_csv(dist_files[-1])
    ref_dists = ref_dists[ref_dists["hospital_1"] == ref_hospital][
        ["hospital_2", "angular_separation_deg"]]
    ref_dists.columns = ["hospitalid", "DAS"]
    ref_dists = pd.concat([ref_dists,
                           pd.DataFrame([{"hospitalid": ref_hospital, "DAS": 0.0}])],
                          ignore_index=True).drop_duplicates("hospitalid", keep="first")  # long-form CSV already holds the self row

    # Merge
    merged = frag_df.merge(ref_dists, on="hospitalid", how="inner")
    merged = merged[merged["n_with_any_evidence"] >= 50]  # min patients
    print(f"  {len(merged)} hospitals with DAS and >= 50 patients with evidence")

    # Save
    merged.to_csv(tbl_dir / "phenotype_fragility.csv", index=False)
    print(f"  Wrote {tbl_dir / 'phenotype_fragility.csv'}")

    # ── Correlations: DAS vs capture rate ──
    capture_cols = [c for c in merged.columns if c.startswith("capture_")]
    corr_results = []
    for col in capture_cols:
        name = col.replace("capture_", "")
        valid = merged.dropna(subset=["DAS", col])
        if len(valid) < 10:
            continue
        rho, p = spearmanr(valid["DAS"], valid[col])
        spread = valid[col].max() - valid[col].min()
        cv = valid[col].std() / valid[col].mean() if valid[col].mean() > 0 else np.nan
        corr_results.append({
            "phenotype": name,
            "mean_capture": round(valid[col].mean(), 3),
            "std_capture": round(valid[col].std(), 3),
            "min_capture": round(valid[col].min(), 3),
            "max_capture": round(valid[col].max(), 3),
            "range": round(spread, 3),
            "CV": round(cv, 3),
            "spearman_rho_vs_DAS": round(rho, 3),
            "p_value": round(p, 4),
            "n_hospitals": len(valid),
        })
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {name:15s}: mean={valid[col].mean():.1%}, "
              f"range=[{valid[col].min():.1%}, {valid[col].max():.1%}], "
              f"rho={rho:.3f} {sig}")

    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(tbl_dir / "phenotype_fragility_corr.csv", index=False)
    print(f"  Wrote {tbl_dir / 'phenotype_fragility_corr.csv'}")

    # ── Figure 1: DAS vs capture rate for each definition ──
    definitions_to_plot = ["Dx_only", "PH_only", "Rx_only", "Full_union",
                           "Two_of_three", "All_three"]
    n_plots = len(definitions_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, name in enumerate(definitions_to_plot):
        ax = axes[i]
        col = f"capture_{name}"
        if col not in merged.columns:
            continue

        ax.scatter(merged["DAS"], merged[col], alpha=0.5, s=20)

        # Regression line
        valid = merged.dropna(subset=["DAS", col])
        if len(valid) > 5:
            z = np.polyfit(valid["DAS"], valid[col], 1)
            x_line = np.linspace(valid["DAS"].min(), valid["DAS"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)

            rho, p = spearmanr(valid["DAS"], valid[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.set_title(f"{name.replace('_', ' ')}\n(rho={rho:.3f}{sig})")

        ax.set_xlabel("DAS (degrees)")
        ax.set_ylabel("Capture rate")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f"{condition.capitalize()}: Phenotype capture rate vs documentation similarity (DAS)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "phenotype_fragility.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'phenotype_fragility.png'}")

    # ── Figure 2: Source dominance by hospital, ordered by DAS ──
    merged_sorted = merged.sort_values("DAS")
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(merged_sorted))
    width = 0.8

    ax.bar(x, merged_sorted["capture_Dx_only"], width, label="Diagnosis", color="#e41a1c", alpha=0.7)
    ax.bar(x, merged_sorted["capture_PH_only"] - merged_sorted["capture_Dx_only"].clip(upper=0),
           width, bottom=merged_sorted["capture_Dx_only"],
           label="Past History (additional)", color="#377eb8", alpha=0.7)

    ax.set_xlabel("Hospitals (ordered by DAS, left=low, right=high)")
    ax.set_ylabel("Capture rate")
    ax.set_title(f"{condition.capitalize()}: Source composition by hospital\n(ordered by documentation angular separation)")
    ax.legend()
    ax.set_xticks([])

    # Add DAS gradient
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    n_ticks = 5
    tick_positions = np.linspace(0, len(merged_sorted) - 1, n_ticks).astype(int)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{merged_sorted.iloc[i]['DAS']:.0f}°" for i in tick_positions])
    ax2.set_xlabel("DAS (degrees)")

    fig.tight_layout()
    fig.savefig(fig_dir / "phenotype_source_dominance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'phenotype_source_dominance.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--ref-hospital", type=int, default=73)
    args = ap.parse_args()

    with open(args.paths) as f:
        paths_cfg = yaml.safe_load(f)

    processed = Path(paths_cfg["outputs"]["processed"])
    art_root = Path(paths_cfg["outputs"]["artifacts"])

    if args.conditions:
        conditions = args.conditions
    else:
        conditions = sorted(set(
            p.name.replace("flags_", "").replace(".parquet", "").replace(".csv", "")
            for p in processed.glob("flags_*")
        ))
    if not conditions:
        conditions = ["diabetes"]

    print(f"Conditions: {conditions}")
    warnings.filterwarnings("ignore")

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"PHENOTYPE FRAGILITY: {condition.upper()}")
        print(f"{'='*60}")
        run_fragility_analysis(condition, processed, art_root, args.ref_hospital)

    print("\nDone.")


if __name__ == "__main__":
    main()
