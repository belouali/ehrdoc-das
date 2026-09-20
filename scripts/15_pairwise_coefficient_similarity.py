#!/usr/bin/env python
"""Pairwise coefficient similarity vs documentation similarity.

The cleanest inference transportability test:
  1. Fit a local regression at each hospital (>=500 patients)
  2. For each pair, compute |coef_A - coef_B| on the log-OR scale
  3. Correlate pairwise coefficient difference with pairwise DAS

If documentation-similar hospitals produce similar coefficients,
DAS predicts inference transportability.

Expected:
  - APACHE: no correlation (documentation-insensitive, negative control)
  - Transfer: positive correlation (documentation-sensitive)
  - Race/Ethnicity: may correlate (partially documentation-sensitive)
  - Gender: no correlation (negative control)

Uses Mantel test (matrix correlation) and Spearman on flattened pairs.

Prereqs: Run scripts 01-03

Usage:
  python scripts/15_pairwise_coefficient_similarity.py --conditions diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform
import statsmodels.formula.api as smf

from ehrdoc.utils.io import ensure_dir

MIN_PATIENTS = 500
FORMULA = ("actualhospitalmortality ~ apachescore + agenum + "
           "C(gender) + C(ethnicity) + C(unitstaytype) + C(unitadmitsource)")

TRACKED_COEFS = [
    "apachescore",
    "C(unitstaytype)[T.transfer]",
    "C(ethnicity)[T.Caucasian]",
    "C(unitadmitsource)[T.Floor]",
    "C(gender)[T.Male]",
]
COEF_LABELS = {
    "apachescore": "APACHE",
    "C(unitstaytype)[T.transfer]": "Transfer",
    "C(ethnicity)[T.Caucasian]": "Race/Eth.",
    "C(unitadmitsource)[T.Floor]": "Floor admit",
    "C(gender)[T.Male]": "Gender",
}


def fit_local(df):
    """Fit logistic regression on one hospital's data."""
    try:
        if df["actualhospitalmortality"].nunique() < 2 or len(df) < MIN_PATIENTS:
            return None
        model = smf.logit(FORMULA, data=df)
        return model.fit(disp=0, maxiter=50, method="bfgs")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=None)
    args = ap.parse_args()

    with open(args.paths) as f:
        cfg = yaml.safe_load(f)

    processed = Path(cfg["outputs"]["processed"])
    art_root = Path(cfg["outputs"]["artifacts"])
    e = cfg["eicu"]
    if "apachePatientResult" not in e:
        e["apachePatientResult"] = "data/raw/apachePatientResult.csv.gz"

    conditions = args.conditions or ["diabetes"]
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

    # Drop NaN in needed columns
    needed = ["apachescore", "agenum", "gender", "ethnicity", "unitstaytype",
              "unitadmitsource", "actualhospitalmortality", "hospitalid"]
    clean = cohort.dropna(subset=needed).copy()
    print(f"  Clean cohort: {len(clean):,} rows, {clean['hospitalid'].nunique()} hospitals")

    warnings.filterwarnings("ignore")

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"PAIRWISE COEFFICIENT SIMILARITY: {condition.upper()}")
        print(f"{'='*60}")

        fig_dir = ensure_dir(art_root / condition / "figures")
        tbl_dir = ensure_dir(art_root / condition / "tables")

        # Load pairwise DAS
        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            print(f"  No distances for {condition}")
            continue

        das_df = pd.read_csv(dist_files[-1])
        das_pairs = {}
        for _, r in das_df.iterrows():
            h1, h2 = int(r["hospital_1"]), int(r["hospital_2"])
            das_pairs[(min(h1, h2), max(h1, h2))] = r["angular_separation_deg"]

        # Fit local regressions
        hosp_counts = clean.groupby("hospitalid").size()
        big_hospitals = hosp_counts[hosp_counts >= MIN_PATIENTS].index.tolist()
        # Only hospitals with DAS
        das_hospitals = set()
        for h1, h2 in das_pairs:
            das_hospitals.add(h1)
            das_hospitals.add(h2)
        eligible = [h for h in big_hospitals if h in das_hospitals]
        print(f"  {len(eligible)} hospitals with >= {MIN_PATIENTS} patients and DAS")

        print(f"  Fitting local regressions...")
        local_coefs = {}
        for hid in eligible:
            hdata = clean[clean["hospitalid"] == hid]
            result = fit_local(hdata)
            if result is None:
                continue
            coefs = {}
            for c in TRACKED_COEFS:
                if c in result.params.index:
                    coefs[c] = result.params[c]  # log-OR
            if coefs:
                local_coefs[hid] = coefs

        hospitals_with_coefs = list(local_coefs.keys())
        print(f"  {len(hospitals_with_coefs)} hospitals with successful local fits")

        if len(hospitals_with_coefs) < 10:
            print("  Too few hospitals")
            continue

        # Build pairwise: |coef_A - coef_B| vs DAS(A, B)
        print(f"  Building pairwise comparisons...")
        rows = []
        for i, h1 in enumerate(hospitals_with_coefs):
            for h2 in hospitals_with_coefs[i+1:]:
                key = (min(h1, h2), max(h1, h2))
                if key not in das_pairs:
                    continue
                das = das_pairs[key]

                for coef in TRACKED_COEFS:
                    if coef in local_coefs[h1] and coef in local_coefs[h2]:
                        diff = abs(local_coefs[h1][coef] - local_coefs[h2][coef])
                        rows.append({
                            "hospital_1": h1, "hospital_2": h2,
                            "DAS": das,
                            "coefficient": coef,
                            "coef_label": COEF_LABELS.get(coef, coef),
                            "abs_log_OR_diff": diff,
                            "log_OR_1": local_coefs[h1][coef],
                            "log_OR_2": local_coefs[h2][coef],
                        })

        pairs = pd.DataFrame(rows)
        pairs.to_csv(tbl_dir / "pairwise_coefficient_similarity.csv", index=False)
        n_pairs = len(pairs[pairs["coefficient"] == TRACKED_COEFS[0]])
        print(f"  {n_pairs} hospital pairs, {len(pairs)} total rows")

        # Correlations
        print(f"\n  === Pairwise correlations: |coef difference| vs DAS ===")
        corr_rows = []
        for coef in TRACKED_COEFS:
            label = COEF_LABELS.get(coef, coef)
            sub = pairs[pairs["coefficient"] == coef].dropna()
            if len(sub) < 10:
                continue

            rho, p_rho = spearmanr(sub["DAS"], sub["abs_log_OR_diff"])
            r, p_r = pearsonr(sub["DAS"], sub["abs_log_OR_diff"])
            sig = "***" if p_rho < 0.001 else "**" if p_rho < 0.01 else "*" if p_rho < 0.05 else ""

            print(f"    {label:12s}: Spearman rho={rho:.3f} (p={p_rho:.4f}) {sig}  "
                  f"Pearson r={r:.3f} (p={p_r:.4f})  n={len(sub)} pairs")

            corr_rows.append({
                "coefficient": label,
                "n_pairs": len(sub),
                "spearman_rho": round(rho, 3),
                "spearman_p": round(p_rho, 4),
                "pearson_r": round(r, 3),
                "pearson_p": round(p_r, 4),
            })

        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(tbl_dir / "pairwise_coef_das_correlation.csv", index=False)

        # ── Figure: scatter per coefficient ──
        n_coefs = len(TRACKED_COEFS)
        fig, axes = plt.subplots(1, n_coefs, figsize=(4 * n_coefs, 4))
        if n_coefs == 1:
            axes = [axes]

        for idx, coef in enumerate(TRACKED_COEFS):
            ax = axes[idx]
            label = COEF_LABELS.get(coef, coef)
            sub = pairs[pairs["coefficient"] == coef].dropna()

            if len(sub) < 10:
                ax.text(0.5, 0.5, "n<10", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            ax.scatter(sub["DAS"], sub["abs_log_OR_diff"], alpha=0.15, s=8, color="steelblue")
            rho, p = spearmanr(sub["DAS"], sub["abs_log_OR_diff"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

            ax.set_xlabel("Pairwise DAS (degrees)")
            ax.set_ylabel("|log OR difference|")
            ax.set_title(f"{label}\nrho={rho:.3f} (p={p:.4f}) {sig}")

        fig.suptitle(f"{condition.capitalize()}: Do documentation-similar hospitals "
                     "produce similar coefficients?", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "pairwise_coefficient_similarity.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {fig_dir / 'pairwise_coefficient_similarity.png'}")

        # ── Summary interpretation ──
        print(f"\n  === Interpretation ===")
        for _, cr in corr_df.iterrows():
            if cr["spearman_p"] < 0.05:
                print(f"    {cr['coefficient']}: DAS predicts coefficient divergence "
                      f"(rho={cr['spearman_rho']}, p={cr['spearman_p']})")
            else:
                print(f"    {cr['coefficient']}: coefficient stable across DAS "
                      f"(rho={cr['spearman_rho']}, p={cr['spearman_p']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
