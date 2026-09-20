#!/usr/bin/env python
"""APACHE sensitivity analysis: do DAS findings hold without restricting to APACHE-available patients?

Compares main analysis (APACHE inner join, ~148K patients) with full cohort
(~200K patients, APACHE imputed to median + missing flag as covariate).

Prereqs:
  - Run scripts 01-03 (distance matrices)

Outputs (per condition, in artifacts/<condition>/tables/):
  sensitivity_apache_imputed.csv

Usage:
  python scripts/10_apache_sensitivity.py
  python scripts/10_apache_sensitivity.py --conditions diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

from ehrdoc.utils.io import ensure_dir

BASE_COVARIATES = ("apachescore + agenum + C(gender) + C(ethnicity) "
                   "+ C(unitstaytype) + C(unitadmitsource)")

TRACKED_COEFFICIENTS = [
    "apachescore",
    "C(ethnicity)[T.Caucasian]",
    "C(gender)[T.Male]",
    "C(unitstaytype)[T.transfer]",
    "C(unitstaytype)[T.readmit]",
    "C(unitadmitsource)[T.Floor]",
    "C(unitadmitsource)[T.Operating Room]",
    "apache_missing",
]


def _fit_and_extract(formula, data, label):
    """Fit logit model and extract key results."""
    try:
        m = smf.logit(formula, data=data).fit(method="lbfgs", maxiter=1000, disp=False)
    except Exception as e:
        print(f"    {label}: FAILED - {e}")
        return None

    # Prediction metrics
    try:
        y_true = m.model.endog
        y_pred = m.predict()
        brier = brier_score_loss(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
    except Exception:
        brier, auc, auprc = np.nan, np.nan, np.nan

    # Extract coefficients
    try:
        conf = m.conf_int()
    except Exception:
        conf = None

    try:
        pvals = m.pvalues
    except Exception:
        pvals = pd.Series(dtype=float)

    rows = []
    for coef_name in TRACKED_COEFFICIENTS:
        if coef_name not in m.params.index:
            continue
        coef_val = float(m.params[coef_name])

        lo, hi, p = np.nan, np.nan, np.nan
        if conf is not None and coef_name in conf.index:
            row_vals = conf.loc[coef_name]
            if hasattr(row_vals, 'values') and len(row_vals.values) >= 2:
                lo, hi = float(row_vals.values[0]), float(row_vals.values[1])
        if coef_name in pvals.index:
            p = float(pvals[coef_name])

        rows.append({
            "model": label,
            "coefficient": coef_name,
            "coef_value": coef_val,
            "OR": np.exp(coef_val),
            "lower_CI": np.exp(lo) if not np.isnan(lo) else np.nan,
            "upper_CI": np.exp(hi) if not np.isnan(hi) else np.nan,
            "p_value": p,
            "AIC": m.aic,
            "BIC": m.bic,
            "pseudo_R2": m.prsquared,
            "n_params": int(m.df_model + 1),
            "df": int(m.df_model),
            "brier": brier,
            "AUC": auc,
            "AUPRC": auprc,
            "n_patients": len(data),
            "n_hospitals": data["hospitalid"].nunique(),
        })

    print(f"    {label}: AIC={m.aic:.1f}, Brier={brier:.4f}, AUC={auc:.4f}, "
          f"n={len(data):,}, hospitals={data['hospitalid'].nunique()}")

    return rows


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
    e = paths_cfg["eicu"]
    if "apachePatientResult" not in e:
        e["apachePatientResult"] = "data/raw/apachePatientResult.csv.gz"

    # Discover conditions
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

    # ── Build both cohorts ──
    print("=" * 60)
    print("Building cohorts")
    print("=" * 60)

    patient_df = pd.read_csv(e["patient"], low_memory=False)
    apache_df = pd.read_csv(e["apachePatientResult"], low_memory=False)
    apache_iva = apache_df[apache_df["apacheversion"] == "IVa"].copy()
    hospital_df = pd.read_csv(e["hospital"], low_memory=False)
    hospital_df.columns = [c.lower() for c in hospital_df.columns]

    # Primary cohort: APACHE inner join (same as main analysis)
    primary = patient_df.merge(apache_iva, on="patientunitstayid", how="inner")
    primary = primary.merge(hospital_df, on="hospitalid", how="inner")

    # Full cohort: APACHE left join with imputation
    full = patient_df.merge(apache_iva, on="patientunitstayid", how="left")
    full["apache_missing"] = full["apachescore"].isna().astype(int)
    apache_median = full["apachescore"].median()
    full["apachescore"] = full["apachescore"].fillna(apache_median)
    full = full.merge(hospital_df, on="hospitalid", how="inner")

    # Clean both
    for df in [primary, full]:
        if "age" in df.columns:
            df["agenum"] = pd.to_numeric(
                df["age"].replace("> 89", "90"), errors="coerce")
        if df["actualhospitalmortality"].dtype == object:
            df["actualhospitalmortality"] = (
                df["actualhospitalmortality"] == "EXPIRED").astype(int)
        for col in ["region", "numbedscategory", "teachingstatus"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

    # Add apache_missing = 0 to primary (for formula compatibility)
    primary["apache_missing"] = 0

    n_primary = len(primary)
    n_full = len(full)
    n_imputed = (full["apache_missing"] == 1).sum()
    print(f"  Primary cohort: {n_primary:,} rows ({primary['hospitalid'].nunique()} hospitals)")
    print(f"  Full cohort: {n_full:,} rows ({full['hospitalid'].nunique()} hospitals)")
    print(f"  Imputed APACHE (median={apache_median:.0f}): {n_imputed:,} patients")
    print(f"  Mortality rate - primary: {primary['actualhospitalmortality'].mean():.3f}")
    print(f"  Mortality rate - full: {full['actualhospitalmortality'].mean():.3f}")

    warnings.filterwarnings("ignore")

    OBS = "C(region) + C(numbedscategory) + C(teachingstatus)"

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"APACHE SENSITIVITY: {condition.upper()}")
        print(f"{'='*60}")

        tbl_dir = ensure_dir(art_root / condition / "tables")

        # Load AS
        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            print(f"  No distance matrix for {condition}, skipping")
            continue

        ref_dists = pd.read_csv(dist_files[-1])
        ref_dists = ref_dists[ref_dists["hospital_1"] == args.ref_hospital][
            ["hospital_2", "angular_separation_deg"]]
        ref_dists.columns = ["hospitalid", "Angular_Separation"]
        ref_dists = pd.concat([ref_dists,
                               pd.DataFrame([{"hospitalid": args.ref_hospital,
                                              "Angular_Separation": 0.0}])],
                              ignore_index=True).drop_duplicates("hospitalid", keep="first")  # long-form CSV already holds the self row

        all_rows = []

        # ── Primary cohort models (replicate main analysis for comparison) ──
        print("\n  Primary cohort (APACHE available only):")
        p_cohort = primary.merge(ref_dists, on="hospitalid", how="left")
        p_merged = p_cohort.dropna(subset=["Angular_Separation"]).copy()

        models_primary = [
            ("Primary: Patient-level",
             f"actualhospitalmortality ~ {BASE_COVARIATES}", p_cohort),
            ("Primary: + DAS",
             f"actualhospitalmortality ~ {BASE_COVARIATES} + Angular_Separation", p_merged),
            ("Primary: + Hosp chars + DAS",
             f"actualhospitalmortality ~ {BASE_COVARIATES} + {OBS} + Angular_Separation", p_merged),
        ]

        for label, formula, data in models_primary:
            rows = _fit_and_extract(formula, data, label)
            if rows:
                all_rows.extend(rows)

        # ── Full cohort models (with imputation + missing flag) ──
        print("\n  Full cohort (APACHE imputed + missing flag):")
        SENS_BASE = BASE_COVARIATES + " + apache_missing"
        f_cohort = full.merge(ref_dists, on="hospitalid", how="left")
        f_merged = f_cohort.dropna(subset=["Angular_Separation"]).copy()

        models_full = [
            ("Full: Patient-level + missing flag",
             f"actualhospitalmortality ~ {SENS_BASE}", f_cohort),
            ("Full: + DAS + missing flag",
             f"actualhospitalmortality ~ {SENS_BASE} + Angular_Separation", f_merged),
            ("Full: + Hosp chars + DAS + missing flag",
             f"actualhospitalmortality ~ {SENS_BASE} + {OBS} + Angular_Separation", f_merged),
        ]

        for label, formula, data in models_full:
            rows = _fit_and_extract(formula, data, label)
            if rows:
                all_rows.extend(rows)

        # ── Full cohort WITHOUT APACHE (DAS without severity adjustment) ──
        print("\n  Full cohort (no APACHE, DAS only):")
        NO_APACHE_BASE = "agenum + C(gender) + C(ethnicity) + C(unitstaytype) + C(unitadmitsource)"

        models_no_apache = [
            ("Full_noAPACHE: Patient-level",
             f"actualhospitalmortality ~ {NO_APACHE_BASE}", f_cohort),
            ("Full_noAPACHE: + DAS",
             f"actualhospitalmortality ~ {NO_APACHE_BASE} + Angular_Separation", f_merged),
            ("Full_noAPACHE: + Hosp chars + DAS",
             f"actualhospitalmortality ~ {NO_APACHE_BASE} + {OBS} + Angular_Separation", f_merged),
        ]

        for label, formula, data in models_no_apache:
            rows = _fit_and_extract(formula, data, label)
            if rows:
                all_rows.extend(rows)

        # Save
        if all_rows:
            results_df = pd.DataFrame(all_rows)
            results_df["condition"] = condition
            out = tbl_dir / "sensitivity_apache_imputed.csv"
            results_df.to_csv(out, index=False)
            print(f"\n  Wrote {out}")

            # Summary comparison
            print(f"\n  Summary: DAS effect on AIC")
            for cohort_label in ["Primary", "Full", "Full_noAPACHE"]:
                base_rows = results_df[
                    (results_df["model"].str.startswith(cohort_label + ":")) &
                    (results_df["model"].str.contains("Patient-level"))
                ]
                das_rows = results_df[
                    (results_df["model"].str.startswith(cohort_label + ":")) &
                    (results_df["model"].str.contains("Hosp chars.*DAS", regex=True))
                ]
                if not base_rows.empty and not das_rows.empty:
                    base_aic = base_rows.iloc[0]["AIC"]
                    das_aic = das_rows.iloc[0]["AIC"]
                    delta = das_aic - base_aic
                    print(f"    {cohort_label:20s}: baseline AIC={base_aic:.0f}, "
                          f"+ chars + DAS AIC={das_aic:.0f}, delta={delta:.0f}")

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
