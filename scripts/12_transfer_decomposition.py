#!/usr/bin/env python
"""Transfer coefficient decomposition.

Explains WHY transfer status reverses direction across model specifications:
1. Transfer rate varies across hospitals (documentation/workflow variation)
2. High-transfer hospitals have different mortality rates (Simpson's paradox)
3. Transfer rate correlates with DAS (documentation geometry captures this)

Prereqs: Run scripts 01-03, 08

Outputs per condition:
  tables/  transfer_decomposition.csv, transfer_correlations.csv
  figures/ transfer_decomposition.png

Usage:
  python scripts/12_transfer_decomposition.py
  python scripts/12_transfer_decomposition.py --conditions diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

from ehrdoc.utils.io import ensure_dir


def analyze_transfer(merged_df: pd.DataFrame, condition: str,
                     fig_dir: Path, tbl_dir: Path):
    """Decompose the transfer coefficient sign flip."""

    # Hospital-level aggregates
    hosp = merged_df.groupby("hospitalid").agg(
        n_patients=("actualhospitalmortality", "size"),
        mortality_rate=("actualhospitalmortality", "mean"),
        transfer_rate=("unitstaytype", lambda x: (x == "transfer").mean()),
        readmit_rate=("unitstaytype", lambda x: (x == "readmit").mean()),
        DAS=("Angular_Separation", "first"),
        AS_quartile=("Angular_Separation_Quartile", "first"),
    ).reset_index()

    hosp = hosp[hosp["n_patients"] >= 50]
    print(f"    {len(hosp)} hospitals with >= 50 patients")
    print(f"    Transfer rate: mean={hosp['transfer_rate'].mean():.3f}, "
          f"range=[{hosp['transfer_rate'].min():.3f}, {hosp['transfer_rate'].max():.3f}]")

    # 1. Transfer rate vs mortality rate
    rho_tm, p_tm = spearmanr(hosp["transfer_rate"], hosp["mortality_rate"])
    sig = "***" if p_tm < 0.001 else "**" if p_tm < 0.01 else "*" if p_tm < 0.05 else ""
    print(f"    Transfer rate vs mortality: rho={rho_tm:.3f} (p={p_tm:.4f}) {sig}")

    # 2. Transfer rate vs DAS
    rho_td, p_td = spearmanr(hosp["transfer_rate"], hosp["DAS"])
    sig = "***" if p_td < 0.001 else "**" if p_td < 0.01 else "*" if p_td < 0.05 else ""
    print(f"    Transfer rate vs DAS: rho={rho_td:.3f} (p={p_td:.4f}) {sig}")

    # 3. Mortality rate vs DAS
    rho_md, p_md = spearmanr(hosp["mortality_rate"], hosp["DAS"])
    sig = "***" if p_md < 0.001 else "**" if p_md < 0.01 else "*" if p_md < 0.05 else ""
    print(f"    Mortality rate vs DAS: rho={rho_md:.3f} (p={p_md:.4f}) {sig}")

    # 4. Transfer rate by DAS quartile
    print(f"\n    Transfer rate by DAS quartile:")
    for q in sorted(hosp["AS_quartile"].dropna().unique()):
        sub = hosp[hosp["AS_quartile"] == q]
        print(f"      {q}: mean={sub['transfer_rate'].mean():.3f}, "
              f"median={sub['transfer_rate'].median():.3f}, n={len(sub)}")

    groups = [g["transfer_rate"].values for _, g in hosp.groupby("AS_quartile") if len(g) > 2]
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else ""
        print(f"    Kruskal-Wallis (transfer ~ DAS quartile): H={H:.2f}, p={p_kw:.4f} {sig}")

    # 5. Mortality rate by DAS quartile
    print(f"\n    Mortality rate by DAS quartile:")
    for q in sorted(hosp["AS_quartile"].dropna().unique()):
        sub = hosp[hosp["AS_quartile"] == q]
        print(f"      {q}: mean={sub['mortality_rate'].mean():.3f}, n={len(sub)}")

    # Save tables
    hosp.to_csv(tbl_dir / "transfer_decomposition.csv", index=False)

    results = {
        "transfer_vs_mortality_rho": round(rho_tm, 3),
        "transfer_vs_mortality_p": round(p_tm, 4),
        "transfer_vs_DAS_rho": round(rho_td, 3),
        "transfer_vs_DAS_p": round(p_td, 4),
        "mortality_vs_DAS_rho": round(rho_md, 3),
        "mortality_vs_DAS_p": round(p_md, 4),
    }
    pd.DataFrame([results]).to_csv(tbl_dir / "transfer_correlations.csv", index=False)
    print(f"    Wrote {tbl_dir / 'transfer_decomposition.csv'}")
    print(f"    Wrote {tbl_dir / 'transfer_correlations.csv'}")

    # ── Plot: 2x2 panel ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Transfer rate vs mortality rate
    ax = axes[0, 0]
    ax.scatter(hosp["transfer_rate"], hosp["mortality_rate"], alpha=0.5, s=20)
    z = np.polyfit(hosp["transfer_rate"], hosp["mortality_rate"], 1)
    x_line = np.linspace(hosp["transfer_rate"].min(), hosp["transfer_rate"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
    ax.set_xlabel("Hospital transfer rate")
    ax.set_ylabel("Hospital mortality rate")
    ax.set_title(f"A. Transfer rate vs mortality\n(rho={rho_tm:.3f}, p={p_tm:.4f})")

    # Panel B: Transfer rate vs DAS
    ax = axes[0, 1]
    ax.scatter(hosp["DAS"], hosp["transfer_rate"], alpha=0.5, s=20)
    z = np.polyfit(hosp["DAS"], hosp["transfer_rate"], 1)
    x_line = np.linspace(hosp["DAS"].min(), hosp["DAS"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
    ax.set_xlabel("DAS (degrees)")
    ax.set_ylabel("Hospital transfer rate")
    ax.set_title(f"B. DAS vs transfer rate\n(rho={rho_td:.3f}, p={p_td:.4f})")

    # Panel C: Transfer rate by DAS quartile
    ax = axes[1, 0]
    quartiles = sorted(hosp["AS_quartile"].dropna().unique())
    data_by_q = [hosp[hosp["AS_quartile"] == q]["transfer_rate"].values for q in quartiles]
    bp = ax.boxplot(data_by_q, tick_labels=quartiles, patch_artist=True)
    colors = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
    for patch, color in zip(bp["boxes"], colors[:len(quartiles)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("DAS Quartile")
    ax.set_ylabel("Transfer rate")
    ax.set_title("C. Transfer rate by DAS quartile")

    # Panel D: Mortality rate by DAS quartile
    ax = axes[1, 1]
    data_by_q = [hosp[hosp["AS_quartile"] == q]["mortality_rate"].values for q in quartiles]
    bp = ax.boxplot(data_by_q, tick_labels=quartiles, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(quartiles)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("DAS Quartile")
    ax.set_ylabel("Mortality rate")
    ax.set_title("D. Mortality rate by DAS quartile")

    fig.suptitle(f"{condition.capitalize()}: Transfer coefficient mechanism",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "transfer_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Wrote {fig_dir / 'transfer_decomposition.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--ref-hospital", type=int, default=73)
    args = ap.parse_args()

    with open(args.paths) as f:
        cfg = yaml.safe_load(f)

    processed = Path(cfg["outputs"]["processed"])
    art_root = Path(cfg["outputs"]["artifacts"])
    e = cfg["eicu"]
    if "apachePatientResult" not in e:
        e["apachePatientResult"] = "data/raw/apachePatientResult.csv.gz"

    if args.conditions:
        conditions = args.conditions
    else:
        conditions = sorted(set(
            p.name.replace("dist_", "").split("_min")[0]
            for p in processed.glob("dist_*.npz")
        ))
    if not conditions:
        conditions = ["diabetes"]

    print(f"Conditions: {conditions}")

    # Build cohort
    print("Building base cohort...")
    patient_df = pd.read_csv(e["patient"], low_memory=False)
    apache_df = pd.read_csv(e["apachePatientResult"], low_memory=False)
    apache_iva = apache_df[apache_df["apacheversion"] == "IVa"].copy()
    hospital_df = pd.read_csv(e["hospital"], low_memory=False)
    hospital_df.columns = [c.lower() for c in hospital_df.columns]

    cohort = patient_df.merge(apache_iva, on="patientunitstayid", how="inner")
    cohort = cohort.merge(hospital_df, on="hospitalid", how="inner")

    if "age" in cohort.columns:
        cohort["agenum"] = pd.to_numeric(
            cohort["age"].replace("> 89", "90"), errors="coerce")
    if cohort["actualhospitalmortality"].dtype == object:
        cohort["actualhospitalmortality"] = (
            cohort["actualhospitalmortality"] == "EXPIRED").astype(int)

    print(f"  Cohort: {len(cohort):,} rows, {cohort['hospitalid'].nunique()} hospitals")

    warnings.filterwarnings("ignore")

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"TRANSFER DECOMPOSITION: {condition.upper()}")
        print(f"{'='*60}")

        fig_dir = ensure_dir(art_root / condition / "figures")
        tbl_dir = ensure_dir(art_root / condition / "tables")

        # Load DAS
        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            print(f"  No distances for {condition}")
            continue

        ref_dists = pd.read_csv(dist_files[-1])
        ref_sub = ref_dists[ref_dists["hospital_1"] == args.ref_hospital][
            ["hospital_2", "angular_separation_deg"]]
        ref_sub.columns = ["hospitalid", "Angular_Separation"]
        ref_sub = pd.concat([ref_sub,
                              pd.DataFrame([{"hospitalid": args.ref_hospital,
                                             "Angular_Separation": 0.0}])],
                             ignore_index=True).drop_duplicates("hospitalid", keep="first")  # long-form CSV already holds the self row

        merged = cohort.merge(ref_sub, on="hospitalid", how="left")
        merged = merged.dropna(subset=["Angular_Separation"]).copy()
        merged["Angular_Separation_Quartile"] = pd.qcut(
            merged["Angular_Separation"], q=4,
            labels=["q1", "q2", "q3", "q4"], duplicates="drop")

        print(f"  Merged: {len(merged):,} patients, {merged['hospitalid'].nunique()} hospitals")

        analyze_transfer(merged, condition, fig_dir, tbl_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
