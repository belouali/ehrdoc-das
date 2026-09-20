#!/usr/bin/env python
"""Build regression cohort and run all model specifications.

Builds the analysis cohort by merging patient, APACHE, hospital, and angular
separation data. Runs all 9+ regression models and extracts coefficients for
APACHE, ethnicity, gender, and transfer across all specifications.

Prereqs:
  - eICU CSV files (patient, apachePatientResult, hospital)
  - Angular separation distance matrix from scripts 01-03

Outputs:
  data/processed/regression_cohort.parquet
  artifacts/regression/all_model_results.csv
  artifacts/regression/coefficient_tracking.csv  (key coefficients across models)
  artifacts/regression/apache_missingness.csv
  artifacts/regression/forest_plot_ethnicity_gender_transfer.png
  artifacts/regression/aic_comparison.png

Usage:
  python scripts/08_regression_models.py
  python scripts/08_regression_models.py --ref-hospital 73 --condition diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm

from ehrdoc.utils.io import ensure_dir
from ehrdoc.data.cohort import analytic_sample


# ── Cohort construction ──────────────────────────────────────────────────────

def build_cohort(paths_cfg: dict, processed: Path, condition: str = "diabetes",
                 ref_hospital: int = 73) -> pd.DataFrame:
    """Build the regression cohort: patient + APACHE IVa + hospital + angular separation."""
    e = paths_cfg["eicu"]

    print("Loading tables...")
    patient_df = pd.read_csv(e["patient"], low_memory=False)
    print(f"  patient: {len(patient_df):,} rows")

    apache_df = pd.read_csv(e.get("apachePatientResult", "data/raw/apachePatientResult.csv.gz"),
                            low_memory=False)
    print(f"  apachePatientResult: {len(apache_df):,} rows")

    hospital_df = pd.read_csv(e["hospital"], low_memory=False)
    hospital_df.columns = [c.lower() for c in hospital_df.columns]
    print(f"  hospital: {len(hospital_df):,} rows")

    # Filter APACHE to version IVa
    apache_iva = apache_df[apache_df["apacheversion"] == "IVa"].copy()
    print(f"  APACHE IVa: {len(apache_iva):,} rows")

    # Merge: patient + APACHE (inner) + hospital (inner)
    cohort = patient_df.merge(apache_iva, on="patientunitstayid", how="inner")
    n_after_apache = len(cohort)
    cohort = cohort.merge(hospital_df, on="hospitalid", how="inner")
    n_after_hospital = len(cohort)

    print(f"\n  APACHE missingness:")
    print(f"    Total patients: {len(patient_df):,}")
    print(f"    With APACHE IVa: {n_after_apache:,} ({100*n_after_apache/len(patient_df):.1f}%)")
    print(f"    After hospital merge: {n_after_hospital:,}")

    # Clean age: eICU stores "> 89" as string
    if "age" in cohort.columns:
        cohort["agenum"] = pd.to_numeric(cohort["age"].replace("> 89", "90"), errors="coerce")
    elif "agenum" not in cohort.columns:
        cohort["agenum"] = np.nan

    # Ensure binary outcome
    if cohort["actualhospitalmortality"].dtype == object:
        cohort["actualhospitalmortality"] = (cohort["actualhospitalmortality"] == "EXPIRED").astype(int)

    # Load angular separation distances
    dist_csv_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
    if dist_csv_files:
        dist_df = pd.read_csv(dist_csv_files[-1])
        # Get distance from reference hospital for each hospital
        ref_dists = dist_df[dist_df["hospital_1"] == ref_hospital][["hospital_2", "angular_separation_deg"]]
        ref_dists.columns = ["hospitalid", "Angular_Separation"]
        # Add self-distance
        ref_dists = pd.concat([ref_dists, pd.DataFrame([{"hospitalid": ref_hospital, "Angular_Separation": 0.0}])],
                              ignore_index=True).drop_duplicates("hospitalid", keep="first")  # long-form CSV already holds the self row
        cohort = cohort.merge(ref_dists, on="hospitalid", how="left")
        n_with_as = cohort["Angular_Separation"].notna().sum()
        print(f"  Angular separation: {n_with_as:,} rows matched ({100*n_with_as/len(cohort):.1f}%)")

        # Create quartiles
        cohort["Angular_Separation_Quartile"] = pd.qcut(
            cohort["Angular_Separation"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop"
        )
    else:
        print("  WARNING: No distance CSV found. Angular separation models will be skipped.")
        cohort["Angular_Separation"] = np.nan
        cohort["Angular_Separation_Quartile"] = np.nan

    print(f"\n  Final cohort: {len(cohort):,} rows, {cohort['hospitalid'].nunique()} hospitals")
    return cohort


# ── APACHE missingness analysis ──────────────────────────────────────────────

def analyze_apache_missingness(paths_cfg: dict, cohort: pd.DataFrame,
                               art_dir: Path) -> pd.DataFrame:
    """Analyze whether APACHE availability varies by hospital and correlates with AS.
    
    Key question: Are hospitals with atypical documentation (high AS) also less likely
    to have APACHE scores? If yes, the regression cohort is biased.
    If no, the APACHE inner join doesn't introduce documentation-related selection bias.
    """
    e = paths_cfg["eicu"]
    patient_df = pd.read_csv(e["patient"], usecols=["patientunitstayid", "hospitalid"], low_memory=False)
    apache_df = pd.read_csv(e.get("apachePatientResult", "data/raw/apachePatientResult.csv.gz"),
                            usecols=["patientunitstayid", "apacheversion"], low_memory=False)

    apache_pids = set(apache_df[apache_df["apacheversion"] == "IVa"]["patientunitstayid"])
    patient_df["has_apache"] = patient_df["patientunitstayid"].isin(apache_pids)

    # Overall missingness
    total = len(patient_df)
    with_apache = patient_df["has_apache"].sum()
    print(f"  Overall: {with_apache:,}/{total:,} stays have APACHE IVa ({100*with_apache/total:.1f}%)")

    # Hospital-level completion rates
    hosp_rates = patient_df.groupby("hospitalid").agg(
        total_stays=("patientunitstayid", "count"),
        apache_stays=("has_apache", "sum"),
    ).reset_index()
    hosp_rates["apache_completion_rate"] = hosp_rates["apache_stays"] / hosp_rates["total_stays"]

    print(f"  Hospital-level completion rates:")
    print(f"    Mean:   {hosp_rates['apache_completion_rate'].mean():.3f}")
    print(f"    Median: {hosp_rates['apache_completion_rate'].median():.3f}")
    print(f"    Min:    {hosp_rates['apache_completion_rate'].min():.3f}")
    print(f"    Max:    {hosp_rates['apache_completion_rate'].max():.3f}")
    print(f"    Hospitals with <50% completion: {(hosp_rates['apache_completion_rate'] < 0.5).sum()}")
    print(f"    Hospitals with 0% completion: {(hosp_rates['apache_completion_rate'] == 0).sum()}")

    # Merge with angular separation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Distribution of APACHE completion rates
    ax = axes[0]
    ax.hist(hosp_rates["apache_completion_rate"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("APACHE IVa completion rate")
    ax.set_ylabel("Number of hospitals")
    ax.set_title("Distribution of APACHE completion rates\nacross hospitals")
    ax.axvline(hosp_rates["apache_completion_rate"].median(), color="red", linestyle="--",
               label=f"Median: {hosp_rates['apache_completion_rate'].median():.2f}")
    ax.legend()

    # Panel B: APACHE completion vs Angular Separation
    ax2 = axes[1]
    if "Angular_Separation" in cohort.columns:
        as_per_hosp = cohort.groupby("hospitalid")["Angular_Separation"].first().reset_index()
        hosp_rates = hosp_rates.merge(as_per_hosp, on="hospitalid", how="left")

        valid = hosp_rates.dropna(subset=["Angular_Separation", "apache_completion_rate"])
        if len(valid) > 5:
            from scipy.stats import spearmanr
            rho, p = spearmanr(valid["Angular_Separation"], valid["apache_completion_rate"])
            print(f"\n  APACHE completion vs Angular Separation:")
            print(f"    Spearman ρ = {rho:.3f}, p = {p:.4f}")
            if abs(rho) < 0.2:
                print(f"    → Weak correlation: APACHE availability is largely independent of documentation behavior")
            elif abs(rho) < 0.4:
                print(f"    → Moderate correlation: some relationship between documentation behavior and APACHE availability")
            else:
                print(f"    → Strong correlation: documentation behavior significantly predicts APACHE availability")

            ax2.scatter(valid["Angular_Separation"], valid["apache_completion_rate"],
                       s=20, alpha=0.6, edgecolors="k", linewidths=0.3)
            ax2.set_xlabel("Angular separation from reference (degrees)")
            ax2.set_ylabel("APACHE IVa completion rate")
            ax2.set_title(f"APACHE completion vs documentation distance\n"
                         f"Spearman ρ={rho:.3f}, p={p:.4f}")
            
            # Add trend line
            z = np.polyfit(valid["Angular_Separation"], valid["apache_completion_rate"], 1)
            x_line = np.linspace(valid["Angular_Separation"].min(), valid["Angular_Separation"].max(), 100)
            ax2.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5)
        else:
            ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No angular separation data", ha="center", va="center", transform=ax2.transAxes)

    fig.tight_layout()
    out_png = art_dir / "apache_missingness.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_png}")

    out = art_dir / "apache_missingness.csv"
    hosp_rates.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return hosp_rates


# ── Model fitting ────────────────────────────────────────────────────────────

TRACKED_COEFFICIENTS = [
    "apachescore",
    "C(gender)[T.Male]",
    "C(ethnicity)[T.Caucasian]",
    "C(unitstaytype)[T.readmit]",
    "C(unitstaytype)[T.transfer]",
    "C(unitadmitsource)[T.Floor]",
    "C(unitadmitsource)[T.Operating Room]",
]

BASE_COVARIATES = "apachescore + agenum + C(gender) + C(ethnicity) + C(unitstaytype) + C(unitadmitsource)"


def _extract_coefficients(result, model_name: str, model_type: str = "logit",
                          aic: float = None, bic: float = None,
                          pseudo_r2: float = None, n_params: int = None,
                          brier: float = None, auc: float = None,
                          auprc: float = None) -> list[dict]:
    """Extract tracked coefficients from a fitted model."""
    rows = []
    try:
        params = result.params
        pvals = result.pvalues

        # Get confidence intervals — handle different formats
        try:
            conf = result.conf_int()
        except Exception:
            conf = None

    except Exception:
        return rows

    for coef_name in TRACKED_COEFFICIENTS:
        if coef_name in params.index:
            coef_val = params[coef_name]

            # Extract CI robustly
            lo, hi = np.nan, np.nan
            if conf is not None and coef_name in conf.index:
                row_vals = conf.loc[coef_name]
                if hasattr(row_vals, 'values') and len(row_vals.values) >= 2:
                    lo = float(row_vals.values[0])
                    hi = float(row_vals.values[1])
                elif hasattr(row_vals, 'iloc'):
                    lo = float(row_vals.iloc[0])
                    hi = float(row_vals.iloc[1])

            # Extract p-value robustly
            p = np.nan
            if coef_name in pvals.index:
                p = float(pvals[coef_name])

            row = {
                "model": model_name,
                "model_type": model_type,
                "coefficient": coef_name,
                "coef_value": float(coef_val),
                "OR": np.exp(float(coef_val)),
                "lower_CI": np.exp(lo) if not np.isnan(lo) else np.nan,
                "upper_CI": np.exp(hi) if not np.isnan(hi) else np.nan,
                "p_value": p,
                "AIC": aic,
                "BIC": bic,
                "pseudo_R2": pseudo_r2,
                "n_params": n_params,
                "brier": brier,
                "AUC": auc,
                "AUPRC": auprc,
            }
            rows.append(row)
    return rows


def _fit_logit_core(formula, data):
    """Fit a logit to full convergence.

    Hospital fixed-effect models (~150 parameters) are fit with Newton-Raphson.
    L-BFGS with the default iteration budget stops well short of the optimum for
    these models, and the leftover log-likelihood gap (about 19 nats on the
    diabetes cohort) is larger than the AIC differences the paper reports, so
    comparisons across optimizers are meaningless. If Newton does not converge
    (e.g. quasi-separation), the fit with the higher log-likelihood is kept.
    """
    model = smf.logit(formula, data=data)
    best = None
    for method, kw in (("newton", dict(maxiter=300, tol=1e-6)),
                       ("lbfgs", dict(maxiter=20000))):
        try:
            r = model.fit(method=method, disp=False, **kw)
        except Exception:
            continue
        if best is None or r.llf > best.llf + 1e-6:
            best = r
        if r.mle_retvals.get("converged"):
            break
    if best is None:
        raise RuntimeError("all optimizers failed")
    return best


class _RIResult:
    """Minimal results shim so random-intercept fits flow through _extract_coefficients."""

    def __init__(self, names, mean, sd):
        self.params = pd.Series(np.asarray(mean, dtype=float), index=names)
        self.bse = pd.Series(np.asarray(sd, dtype=float), index=names)
        z = self.params / self.bse
        self.pvalues = pd.Series(2 * norm.sf(np.abs(z.values)), index=names)

    def conf_int(self, alpha=0.05):
        q = norm.ppf(1 - alpha / 2)
        return pd.DataFrame({0: self.params - q * self.bse, 1: self.params + q * self.bse})


def _fit_ri(formula, data, label, group="hospitalid"):
    """Logistic random-intercept model (hospital as a random effect).

    This is the correct way to ask whether a hospital-level covariate (DAS)
    explains between-hospital variation: a hospital-level covariate cannot be
    added to a hospital fixed-effect model (perfect collinearity), but it can be
    added to a random-intercept model, where its contribution shows up as a
    reduction in the random-intercept variance. Fit by variational Bayes
    (statsmodels BinomialBayesMixedGLM) with weak priors.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

    def _metrics(model, fe_mean, vc_mean):
        eta = np.asarray(model.exog) @ np.asarray(fe_mean) + model.exog_vc.dot(np.asarray(vc_mean))
        p = 1.0 / (1.0 + np.exp(-eta))
        y = np.asarray(model.endog)
        return brier_score_loss(y, p), roc_auc_score(y, p), average_precision_score(y, p)

    try:
        model = BinomialBayesMixedGLM.from_formula(
            formula, {"hosp": f"0 + C({group})"}, data, vcp_p=2.0, fe_p=10.0)
        # Warm start from the plain logit solution: the default zero start can
        # leave the VB optimizer stuck at its initial point (SD exactly 1, AUC 0.5).
        start_fe = smf.logit(formula, data=data).fit(method="lbfgs", maxiter=5000, disp=False).params.values
        k_vc = model.exog_vc.shape[1]
        mean0 = np.concatenate([start_fe, np.zeros(k_vc), np.array([np.log(0.3)])])
        sd0 = np.concatenate([np.full(len(start_fe), 0.05), np.full(k_vc, 0.3), np.array([0.3])])
        r = model.fit_vb(mean=mean0, sd=sd0, verbose=False)
        brier, auc, auprc = _metrics(model, r.fe_mean, r.vc_mean)
        fit_method = "vb"
        if not np.isfinite(auc) or auc < 0.6 or np.allclose(r.fe_mean, 0):
            print(f"    (VB did not move from start; falling back to Laplace/MAP)")
            r = model.fit_map(method="BFGS", minim_opts={"maxiter": 500})
            brier, auc, auprc = _metrics(model, r.fe_mean, r.vc_mean)
            fit_method = "map"
        if not np.isfinite(auc) or auc < 0.6:
            raise RuntimeError(f"random-intercept fit did not converge (AUC={auc:.3f})")
    except Exception as e:
        print(f"    {label}: FAILED — {e}")
        return None, str(e)

    names = list(model.exog_names)
    res = _RIResult(names, r.fe_mean, r.fe_sd)
    re_sd = float(np.exp(r.vcp_mean[0]))
    re_var = re_sd ** 2
    icc_latent = re_var / (re_var + np.pi ** 2 / 3)

    info = {"AIC": None, "BIC": None, "pseudo_R2": None, "n_params": len(names) + 1,
            "llf": None, "df_model": len(names) - 1,
            "re_sd": re_sd, "re_var": re_var, "icc_latent": icc_latent,
            "n_groups": int(data[group].nunique()), "fit_method": fit_method,
            "brier": brier, "AUC": auc, "AUPRC": auprc}
    print(f"    random-intercept SD={re_sd:.3f} (var={re_var:.3f}, latent ICC={icc_latent:.3f}, {fit_method})")
    return res, info


def _fit_logit(formula, data, label):
    """Fit a logit model and return (result, info_dict) or (None, error_string).
    
    For models with many parameters (hospital FE), tries multiple optimizers
    to recover CIs when the default fails.
    """
    methods_to_try = ["lbfgs"]
    
    # First fit with lbfgs (fast, reliable for coefficients)
    try:
        m = _fit_logit_core(formula, data)
    except Exception as e:
        print(f"    {label}: FAILED — {e}")
        return None, str(e)
    if not m.mle_retvals.get("converged", True):
        print(f"    (warning: optimizer did not report convergence; llf={m.llf:.3f})")

    # Check if CIs are available
    try:
        conf = m.conf_int()
        has_nans = conf.isna().any().any()
    except Exception:
        has_nans = True

    # If CIs have NaNs, try recovery
    if has_nans:
        # Tier 1: Try alternative optimizers (only for models with < 100 df)
        if m.df_model < 100:
            for alt_method in ["bfgs", "newton"]:
                try:
                    m_alt = smf.logit(formula, data=data).fit(
                        method=alt_method, maxiter=2000, disp=False)
                    conf_alt = m_alt.conf_int()
                    if not conf_alt.isna().any().any():
                        print(f"    (recovered CIs with method='{alt_method}')")
                        m = m_alt
                        has_nans = False
                        break
                except Exception:
                    continue

    # Tier 2: If still NaN, try robust SEs (works for large models)
    if has_nans:
        try:
            m_robust = smf.logit(formula, data=data).fit(
                method="lbfgs", maxiter=1000, disp=False,
                cov_type="HC0")
            conf_robust = m_robust.conf_int()
            if not conf_robust.loc[conf_robust.index.isin(TRACKED_COEFFICIENTS)].isna().any().any():
                print(f"    (recovered tracked CIs with robust SEs)")
                m = m_robust
                has_nans = False
        except Exception:
            pass

    if has_nans:
        print(f"    (CIs unavailable for some coefficients, df={int(m.df_model)})")

    # Compute prediction metrics
    from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
    try:
        y_true = m.model.endog
        y_pred = m.predict()
        brier = brier_score_loss(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
    except Exception:
        brier, auc, auprc = np.nan, np.nan, np.nan

    return m, {"AIC": m.aic, "BIC": m.bic, "pseudo_R2": m.prsquared,
                "n_params": int(m.df_model + 1), "llf": m.llf, "df_model": int(m.df_model),
                "brier": brier, "AUC": auc, "AUPRC": auprc}


def _fit_gee(formula, data, groups, label):
    """Fit a GEE model and return (result, info_dict) or (None, error_string)."""
    try:
        family = sm.families.Binomial()
        m = smf.gee(formula, data=data, groups=groups, family=family).fit()
        return m, {"AIC": None, "BIC": None, "pseudo_R2": None, "n_params": None, "llf": None}
    except Exception as e:
        print(f"    {label}: FAILED — {e}")
        return None, str(e)


def run_all_models(cohort: pd.DataFrame, merged_df: pd.DataFrame = None,
                   mean_as_df: pd.DataFrame = None,
                   include_gee: bool = False,
                   include_ri: bool = True,
                   model_filter: list[str] = None) -> tuple[pd.DataFrame, dict]:
    """Run all model specifications and extract coefficients.

    All models run on merged_df (the analytic sample with DAS available)
    to ensure AICs are comparable across the same sample.

    Returns (coef_dataframe, fitted_models_dict) where fitted_models_dict
    maps model label -> fitted result for LRT computation.
    """
    if merged_df is None:
        merged_df = cohort.dropna(subset=["Angular_Separation"])

    all_rows = []
    fitted = {}  # label -> (result, info)

    OBS = "C(region) + C(numbedscategory) + C(teachingstatus)"

    # All models run on merged_df ("as") to ensure same sample.
    #
    # NOTE on what is deliberately absent: any model that adds a hospital-level
    # covariate (DAS, DAS quartile, region, bed size, teaching status) on top of
    # hospital fixed effects. DAS is constant within hospital, so such a design
    # is rank-deficient (verified: 150 columns, rank 149 on the diabetes cohort);
    # the likelihood cannot improve and any apparent AIC gain is optimizer noise.
    # The question "does DAS explain between-hospital variation?" is answered by
    # the random-intercept models (RI) below via the reduction in the hospital
    # random-intercept variance, see compute_variance_decomposition().
    models = [
        # (label, formula_suffix, dataset_key, method)
        ("M1: Patient-level",                    "",                                                "as",      "logit"),
        ("M2: + Hospital FE",                    " + C(hospitalid)",                                "as",      "logit"),
        ("M3: + AS continuous",                  " + Angular_Separation",                           "as",      "logit"),
        ("M4: + AS quartile",                    " + C(Angular_Separation_Quartile)",               "as",      "logit"),
        ("M9: + Hospital chars",                 f" + {OBS}",                                       "as",      "logit"),
        ("M10: + Hosp chars + AS quartile",      f" + {OBS} + C(Angular_Separation_Quartile)",      "as",      "logit"),
        ("M13: + Hosp chars + AS continuous",    f" + {OBS} + Angular_Separation",                  "as",      "logit"),
        # Random-intercept (hospital) models: variational Bayes logistic GLMM
        ("M6: + Hospital RI",                    "",                                                "as",      "ri"),
        ("M8: + AS continuous + Hospital RI",    " + Angular_Separation",                           "as",      "ri"),
        ("M18: + Hosp chars + Hospital RI",      f" + {OBS}",                                       "as",      "ri"),
        ("M19: + Hosp chars + AS continuous + Hospital RI", f" + {OBS} + Angular_Separation",       "as",      "ri"),
        # GEE (population-averaged) versions, optional
        ("M7: + Hospital GEE",                   "",                                                "as",      "gee_hosp"),
        ("M12: + AS quartile + Hospital GEE",    " + C(Angular_Separation_Quartile)",               "as",      "gee_hosp"),
    ]

    # Add M14 if mean_as_df is available
    if mean_as_df is not None and "Mean_Angular_Separation" in mean_as_df.columns:
        models.append(
            ("M14: + Mean AS (all conditions)",  " + Mean_Angular_Separation",                      "mean_as", "logit"),
        )

    # Filter out GEE models unless requested
    if not include_gee:
        gee_methods = {"gee_hosp", "gee_asq"}
        skipped = [label for label, _, _, method in models if method in gee_methods]
        models = [(label, fs, ds, method) for label, fs, ds, method in models if method not in gee_methods]
        if skipped:
            print(f"  Skipping GEE models: {', '.join(skipped)} (use --include-gee to enable)")
    if not include_ri:
        models = [(label, fs, ds, method) for label, fs, ds, method in models if method != "ri"]
        print("  Skipping random-intercept models (--skip-ri)")

    # Filter to specific models if requested
    if model_filter:
        models = [(label, fs, ds, method) for label, fs, ds, method in models
                  if any(mf in label for mf in model_filter)]
        print(f"  Running {len(models)} models matching filter: {model_filter}")

    print(f"  Analytic sample: {len(merged_df):,} patients, {merged_df['hospitalid'].nunique()} hospitals")

    for label, formula_suffix, ds_key, method in models:
        print(f"\n  {label}")
        base_formula = f"actualhospitalmortality ~ {BASE_COVARIATES}{formula_suffix}"

        # Select dataset
        if ds_key == "full":
            data = cohort
        elif ds_key == "mean_as":
            data = mean_as_df
        else:
            data = merged_df

        if method == "logit":
            m, info = _fit_logit(base_formula, data, label)
        elif method == "gee_hosp":
            m, info = _fit_gee(base_formula, data, data["hospitalid"], label)
        elif method == "gee_asq":
            m, info = _fit_gee(base_formula, data, data["Angular_Separation_Quartile"], label)
        elif method == "ri":
            m, info = _fit_ri(base_formula, data, label)
        else:
            m, info = None, "Unknown method"

        if m is not None:
            aic = info.get("AIC")
            bic = info.get("BIC")
            pr2 = info.get("pseudo_R2")
            npar = info.get("n_params")
            brier = info.get("brier")
            auc_val = info.get("AUC")
            auprc_val = info.get("AUPRC")
            all_rows.extend(_extract_coefficients(m, label, method, aic, bic, pr2, npar,
                                                  brier, auc_val, auprc_val))
            fitted[label] = (m, info)
            if aic is not None:
                brier_str = f", Brier={brier:.4f}" if brier is not None else ""
                auc_str = f", AUC={auc_val:.4f}" if auc_val is not None else ""
                print(f"    AIC={aic:.1f}, pseudo-R²={pr2:.4f}, params={npar}{brier_str}{auc_str}")
            elif method == "ri":
                print(f"    Converged (random intercept — no AIC; AUC={auc_val:.4f}, Brier={brier:.4f})")
            else:
                print(f"    Converged (GEE — no AIC)")

    return pd.DataFrame(all_rows), fitted


# ── Likelihood Ratio Tests ───────────────────────────────────────────────────

def compute_lrt(fitted: dict) -> pd.DataFrame:
    """Compute likelihood ratio tests for nested model pairs."""
    from scipy.stats import chi2

    # Only genuinely nested pairs where the added term is identifiable.
    # (Adding DAS or hospital characteristics to a hospital-FE model is not a
    # valid comparison: the added term is collinear with the hospital dummies.)
    nested_pairs = [
        ("M1: Patient-level",               "M2: + Hospital FE",               "Hospital FE improve on patient-level?"),
        ("M1: Patient-level",               "M3: + AS continuous",             "AS continuous improve on patient-level?"),
        ("M1: Patient-level",               "M4: + AS quartile",               "AS quartile improve on patient-level?"),
        ("M1: Patient-level",               "M9: + Hospital chars",            "Hospital chars improve on patient-level?"),
        ("M3: + AS continuous",             "M13: + Hosp chars + AS continuous","Hosp chars add to AS continuous?"),
        ("M4: + AS quartile",               "M10: + Hosp chars + AS quartile", "Hosp chars add to AS quartile?"),
        ("M9: + Hospital chars",            "M10: + Hosp chars + AS quartile", "AS quartile add to hospital chars?"),
        ("M9: + Hospital chars",            "M13: + Hosp chars + AS continuous","AS continuous add to hospital chars?"),
        ("M3: + AS continuous",             "M2: + Hospital FE",               "(not nested; listed for AIC only)"),
    ]

    rows = []
    print("\n  Likelihood Ratio Tests:")
    for restricted_label, full_label, description in nested_pairs:
        if restricted_label not in fitted or full_label not in fitted:
            continue
        r_model, r_info = fitted[restricted_label]
        f_model, f_info = fitted[full_label]

        # LRT only works for models with log-likelihoods (logit, not GEE)
        if r_info.get("llf") is None or f_info.get("llf") is None:
            continue

        if description.startswith("(not nested"):
            continue
        lr_stat = -2 * (r_info["llf"] - f_info["llf"])
        df_diff = f_info["df_model"] - r_info["df_model"]
        if df_diff <= 0:
            continue

        p_value = chi2.sf(lr_stat, df_diff)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        rows.append({
            "restricted": restricted_label,
            "full": full_label,
            "description": description,
            "LR_statistic": round(lr_stat, 2),
            "df": df_diff,
            "p_value": p_value,
            "significant": sig,
        })
        print(f"    {description}")
        print(f"      LR={lr_stat:.2f}, df={df_diff}, p={p_value:.2e} {sig}")

    return pd.DataFrame(rows)


# ── Clustered SEs sensitivity ────────────────────────────────────────────────

def run_clustered_se_models(cohort: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    """Re-run key models with SEs clustered on uniquepid."""
    print("\n  Clustered SEs (on uniquepid):")
    all_rows = []

    pid_col = "uniquepid" if "uniquepid" in cohort.columns else "patientunitstayid"

    # Columns used in the base formula — need to drop NaN on these
    formula_cols = ["actualhospitalmortality", "apachescore", "agenum", "gender",
                    "ethnicity", "unitstaytype", "unitadmitsource"]

    specs = [
        ("M1 (clustered)", f"actualhospitalmortality ~ {BASE_COVARIATES}",
         merged_df, []),
        ("M4 (clustered)", f"actualhospitalmortality ~ {BASE_COVARIATES} + C(Angular_Separation_Quartile)",
         merged_df, ["Angular_Separation_Quartile"]),
    ]

    for label, formula, data, extra_cols in specs:
        try:
            # Drop rows with NaN in any formula variable so groups align
            keep_cols = formula_cols + extra_cols + [pid_col]
            clean = data[keep_cols].dropna()
            # Rebuild data with only clean rows
            clean_data = data.loc[clean.index].copy()
            groups = clean_data[pid_col].values

            m = smf.logit(formula, data=clean_data).fit(
                method="lbfgs", maxiter=1000, disp=False,
                cov_type="cluster", cov_kwds={"groups": groups},
            )
            from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
            try:
                y_t = m.model.endog; y_p = m.predict()
                br = brier_score_loss(y_t, y_p); au = roc_auc_score(y_t, y_p); ap = average_precision_score(y_t, y_p)
            except Exception:
                br, au, ap = np.nan, np.nan, np.nan
            all_rows.extend(_extract_coefficients(m, label, "logit_clustered",
                                                  m.aic, m.bic, m.prsquared, int(m.df_model + 1),
                                                  br, au, ap))
            print(f"    {label}: done (n={len(clean_data):,})")
        except Exception as e:
            print(f"    {label}: FAILED — {e}")

    return pd.DataFrame(all_rows)


# ── Stratified regressions ───────────────────────────────────────────────────

def run_stratified_models(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Run regressions stratified by AS quartile."""
    print("\n  Stratified regressions by AS quartile:")
    all_rows = []
    formula = f"actualhospitalmortality ~ {BASE_COVARIATES}"

    for q in sorted(merged_df["Angular_Separation_Quartile"].dropna().unique()):
        sub = merged_df[merged_df["Angular_Separation_Quartile"] == q]
        try:
            m = smf.logit(formula, data=sub).fit(method="lbfgs", maxiter=1000, disp=False)
            rows = _extract_coefficients(m, f"Stratified: {q}", "logit",
                                         m.aic, m.bic, m.prsquared, m.df_model + 1)
            for r in rows:
                r["quartile"] = q
                r["n_patients"] = len(sub)
                r["n_hospitals"] = sub["hospitalid"].nunique()
            all_rows.extend(rows)
            print(f"    {q}: n={len(sub):,}, {sub['hospitalid'].nunique()} hospitals, AIC={m.aic:.1f}")
        except Exception as e:
            print(f"    {q}: FAILED — {e}")

    return pd.DataFrame(all_rows)


# ── Per-Hospital Models with Averaged Betas (Harold's Item 3) ────────────────

def run_per_hospital_models(merged_df: pd.DataFrame, tbl_dir: Path) -> pd.DataFrame:
    """Run regressions within each hospital, then average betas.

    Three averaging strategies (Harold's items 3-5):
      1. Naive: equal-weight average across all hospitals
      2. AS-weighted: weight by inverse angular separation (similar hospitals count more)
      3. AS-clustered: average within AS quartile first, then combine

    Averaging is done on the log-odds (coefficient) scale, then exponentiated
    to OR. Hospitals with extreme coefficients (|coef| > 5, i.e. OR > 148 or < 0.007)
    are excluded as likely due to complete separation.
    """
    print("\n  Per-hospital models (Harold's item 3):")
    formula = f"actualhospitalmortality ~ {BASE_COVARIATES}"

    COEF_CAP = 5.0  # |log-odds| > 5 indicates separation

    hospital_coefs = []
    hospitals = merged_df["hospitalid"].unique()
    n_skipped_separation = 0

    for hid in hospitals:
        sub = merged_df[merged_df["hospitalid"] == hid]
        if len(sub) < 50:
            continue
        if sub["actualhospitalmortality"].nunique() < 2:
            continue
        try:
            m = smf.logit(formula, data=sub).fit(method="lbfgs", maxiter=500, disp=False)
            for coef_name in TRACKED_COEFFICIENTS:
                if coef_name in m.params.index:
                    coef_val = float(m.params[coef_name])
                    if abs(coef_val) > COEF_CAP:
                        n_skipped_separation += 1
                        continue
                    hospital_coefs.append({
                        "hospitalid": hid,
                        "coefficient": coef_name,
                        "coef_value": coef_val,
                        "OR": np.exp(coef_val),
                        "n_patients": len(sub),
                        "AS": float(sub["Angular_Separation"].iloc[0]),
                        "AS_quartile": str(sub["Angular_Separation_Quartile"].iloc[0]),
                    })
        except Exception:
            pass

    if not hospital_coefs:
        print("    No hospitals with sufficient data")
        return pd.DataFrame()

    hc_df = pd.DataFrame(hospital_coefs)
    n_hospitals_fit = hc_df["hospitalid"].nunique()
    print(f"    Fit models for {n_hospitals_fit} hospitals (min 50 patients, outcome variance)")
    if n_skipped_separation > 0:
        print(f"    Excluded {n_skipped_separation} extreme coefficients (|log-odds| > {COEF_CAP})")

    # ── Strategy 1: Naive averaging on log-odds scale ──
    naive_rows = []
    for coef_name, grp in hc_df.groupby("coefficient"):
        coefs = grp["coef_value"].values
        mean_coef = np.mean(coefs)
        naive_rows.append({
            "coefficient": coef_name,
            "mean_coef": mean_coef,
            "mean_OR": np.exp(mean_coef),
            "median_OR": np.exp(np.median(coefs)),
            "std_coef": np.std(coefs),
            "n_hospitals": len(grp),
            "strategy": "naive_equal_weight",
        })
    naive_avg = pd.DataFrame(naive_rows)

    # ── Strategy 2: AS-weighted averaging on log-odds scale ──
    hc_df["weight"] = 1.0 / (1.0 + hc_df["AS"])
    weighted_rows = []
    for coef_name, grp in hc_df.groupby("coefficient"):
        w = grp["weight"].values
        coefs = grp["coef_value"].values
        w_norm = w / w.sum()
        weighted_coef = np.average(coefs, weights=w_norm)
        weighted_rows.append({
            "coefficient": coef_name,
            "mean_coef": weighted_coef,
            "mean_OR": np.exp(weighted_coef),
            "n_hospitals": len(grp),
            "strategy": "AS_weighted",
        })
    weighted_avg = pd.DataFrame(weighted_rows)

    # ── Strategy 3: AS-clustered averaging on log-odds scale ──
    clustered_rows = []
    for coef_name, grp in hc_df.groupby("coefficient"):
        q_means = grp.groupby("AS_quartile")["coef_value"].mean()
        clustered_coef = q_means.mean()
        clustered_rows.append({
            "coefficient": coef_name,
            "mean_coef": clustered_coef,
            "mean_OR": np.exp(clustered_coef),
            "n_hospitals": len(grp),
            "n_quartiles": len(q_means),
            "strategy": "AS_clustered",
        })
    clustered_avg = pd.DataFrame(clustered_rows)

    # Combine and save
    summary = pd.concat([naive_avg, weighted_avg, clustered_avg], ignore_index=True)
    out = tbl_dir / "per_hospital_averaged_betas.csv"
    summary.to_csv(out, index=False)
    print(f"    Wrote {out}")

    # Also save the raw per-hospital coefficients
    hc_df.to_csv(tbl_dir / "per_hospital_coefficients.csv", index=False)
    print(f"    Wrote {tbl_dir / 'per_hospital_coefficients.csv'}")

    # Print key results
    for strategy in ["naive_equal_weight", "AS_weighted", "AS_clustered"]:
        sub = summary[summary["strategy"] == strategy]
        eth = sub[sub["coefficient"] == "C(ethnicity)[T.Caucasian]"]
        if not eth.empty:
            print(f"    {strategy}: Ethnicity OR = {eth['mean_OR'].values[0]:.3f}")

    return hc_df


# ── Cluster-Based Models (Harold's Item 6) ───────────────────────────────────

def run_cluster_based_models(merged_df: pd.DataFrame, D: np.ndarray, ids: list[int],
                             tbl_dir: Path) -> pd.DataFrame:
    """Run models using data-driven clustering of hospitals (kNN, k-means).

    Harold's item 6: cluster on some other basis beyond quartiles.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    print("\n  Cluster-based models (Harold's item 6):")

    all_rows = []
    formula_base = f"actualhospitalmortality ~ {BASE_COVARIATES}"

    # Map hospital IDs to distance matrix indices
    id_to_idx = {hid: i for i, hid in enumerate(ids)}

    for n_clusters in [3, 4, 5, 8]:
        for method_name, clusterer in [
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
            ("hierarchical", AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                                     linkage="average")),
        ]:
            try:
                if method_name == "kmeans":
                    # KMeans needs feature matrix, use MDS to get 2D coordinates from distances
                    from sklearn.manifold import MDS
                    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                                 n_init=2, max_iter=200).fit_transform(D)
                    labels = clusterer.fit_predict(coords)
                else:
                    labels = clusterer.fit_predict(D)

                # Map cluster labels to hospitals in merged_df
                cluster_map = {ids[i]: int(labels[i]) for i in range(len(ids))}
                col_name = f"cluster_{method_name}_k{n_clusters}"
                merged_df[col_name] = merged_df["hospitalid"].map(cluster_map)

                # Fit model with cluster as covariate
                formula = f"{formula_base} + C({col_name})"
                m = smf.logit(formula, data=merged_df.dropna(subset=[col_name])).fit(
                    method="lbfgs", maxiter=1000, disp=False)

                rows = _extract_coefficients(
                    m, f"+ {method_name}(k={n_clusters})", "logit",
                    m.aic, m.bic, m.prsquared, int(m.df_model + 1))
                all_rows.extend(rows)
                print(f"    {method_name} k={n_clusters}: AIC={m.aic:.1f}, params={int(m.df_model+1)}")

            except Exception as e:
                print(f"    {method_name} k={n_clusters}: FAILED — {e}")

    results_df = pd.DataFrame(all_rows)
    if not results_df.empty:
        out = tbl_dir / "cluster_based_models.csv"
        results_df.to_csv(out, index=False)
        print(f"    Wrote {out}")

    return results_df


# ── Variance Decomposition ───────────────────────────────────────────────────

def compute_variance_decomposition(cohort: pd.DataFrame, merged_df: pd.DataFrame,
                                   tbl_dir: Path, fitted: dict | None = None) -> pd.DataFrame:
    """Variance decomposition.

    Primary: hospital random-intercept variance from the logistic GLMMs fitted in
    run_all_models (M6, M8, M18, M19). The share of between-hospital variance
    explained by a hospital-level covariate is 1 - var(with) / var(without).
    Secondary (descriptive): between-hospital variance of predicted probabilities.
    """
    print("\n  Variance decomposition:")
    results = []

    # ── Primary: random-intercept variances from the fitted GLMMs ──
    ri = {label: info for label, (m, info) in (fitted or {}).items() if "re_var" in info}
    base_label = "M6: + Hospital RI"
    base_var = ri.get(base_label, {}).get("re_var", np.nan)
    chars_var = ri.get("M18: + Hosp chars + Hospital RI", {}).get("re_var", np.nan)
    for label, info in ri.items():
        ref_var = chars_var if ("Hosp chars" in label and "AS" in label) else base_var
        red = 1 - info["re_var"] / ref_var if np.isfinite(ref_var) and ref_var > 0 else np.nan
        results.append({
            "model": label, "approach": "glmm_random_intercept",
            "n_hospitals": info.get("n_groups"),
            "re_sd": round(info["re_sd"], 4),
            "re_var": round(info["re_var"], 4),
            "ICC_latent": round(info["icc_latent"], 4),
            "reference_model": base_label if ref_var is base_var else "M18: + Hosp chars + Hospital RI",
            "pct_between_hospital_var_explained": round(100 * red, 1) if np.isfinite(red) else np.nan,
        })
        print(f"    {label}: RI var={info['re_var']:.4f}, ICC={info['icc_latent']:.3f}, "
              f"explained vs reference={100*red:.1f}%" if np.isfinite(red) else
              f"    {label}: RI var={info['re_var']:.4f}, ICC={info['icc_latent']:.3f}")

    # ── Secondary: between-hospital variance of predicted probabilities ──
    models_to_compare = [
        ("No site adjustment", f"actualhospitalmortality ~ {BASE_COVARIATES}", merged_df),
        ("+ Hospital FE", f"actualhospitalmortality ~ {BASE_COVARIATES} + C(hospitalid)", merged_df),
        ("+ AS continuous", f"actualhospitalmortality ~ {BASE_COVARIATES} + Angular_Separation", merged_df),
        ("+ AS quartile", f"actualhospitalmortality ~ {BASE_COVARIATES} + C(Angular_Separation_Quartile)", merged_df),
    ]
    for label, formula, data in models_to_compare:
        try:
            key = {"No site adjustment": "M1: Patient-level", "+ Hospital FE": "M2: + Hospital FE",
                   "+ AS continuous": "M3: + AS continuous", "+ AS quartile": "M4: + AS quartile"}[label]
            if fitted and key in fitted:
                m = fitted[key][0]
            else:
                m = _fit_logit_core(formula, data)
            pred = m.predict(data)
            data_tmp = data.copy()
            data_tmp["pred"] = pred
            hosp_means = data_tmp.groupby("hospitalid")["pred"].mean()
            between_var = hosp_means.var()
            overall_var = pred.var()
            icc_approx = between_var / overall_var if overall_var > 0 else np.nan
            results.append({
                "model": label, "approach": "logit_predicted_probs",
                "between_hospital_var": round(between_var, 6),
                "total_pred_var": round(overall_var, 6),
                "ICC_approx": round(icc_approx, 4),
            })
            print(f"    {label}: between-hosp var of predictions={between_var:.6f}")
        except Exception as e:
            print(f"    {label}: FAILED — {e}")

    results_df = pd.DataFrame(results)
    out = tbl_dir / "variance_decomposition.csv"
    results_df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return results_df


def write_model_comparison_table(all_results: pd.DataFrame, lrt_df: pd.DataFrame,
                                 fitted: dict, tbl_dir: Path) -> pd.DataFrame:
    """One row per model: parameters, fit statistics, and key ORs (95% CI)."""
    main = all_results[~all_results["model"].str.contains(
        "clustered|Stratified|kmeans|hierarchical", case=False, na=False)]
    key_coefs = [("apachescore", "APACHE"), ("C(ethnicity)[T.Caucasian]", "RaceEth_Caucasian_vs_AA"),
                 ("C(unitstaytype)[T.transfer]", "Transfer_vs_admit"), ("C(gender)[T.Male]", "Male_vs_Female")]
    rows = []
    for model, g in main.groupby("model", sort=False):
        first = g.iloc[0]
        info = fitted.get(model, (None, {}))[1] if fitted else {}
        row = {"model": model, "n_params": first["n_params"], "AIC": first["AIC"], "BIC": first["BIC"],
               "AUC": first["AUC"], "Brier": first["brier"],
               "RI_sd": info.get("re_sd", np.nan) if isinstance(info, dict) else np.nan}
        for coef, nice in key_coefs:
            r = g[g["coefficient"] == coef]
            if r.empty:
                row[f"{nice}_OR"] = ""
                row[f"{nice}_p"] = np.nan
            else:
                r = r.iloc[0]
                row[f"{nice}_OR"] = f"{r['OR']:.3f} ({r['lower_CI']:.3f}, {r['upper_CI']:.3f})"
                row[f"{nice}_p"] = r["p_value"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if lrt_df is not None and not lrt_df.empty:
        lrt_map = {}
        for r in lrt_df.itertuples():
            lrt_map.setdefault(r.full, []).append(f"vs {r.restricted.split(':')[0]}: p={r.p_value:.2g}")
        df["LRT"] = df["model"].map(lambda m: "; ".join(lrt_map.get(m, [])))
    out = tbl_dir / "model_comparison.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return df


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_forest(coef_df: pd.DataFrame, art_dir: Path):
    """Forest plot comparing ethnicity, gender, and transfer ORs across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

    targets = [
        ("C(ethnicity)[T.Caucasian]", "Ethnicity: Caucasian vs AA", axes[0]),
        ("C(gender)[T.Male]", "Gender: Male vs Female", axes[1]),
        ("C(unitstaytype)[T.transfer]", "Unit Stay: Transfer vs Admit", axes[2]),
    ]

    # Filter to main models only (not clustered or stratified)
    main_models = coef_df[~coef_df["model"].str.contains("clustered|Stratified", case=False, na=False)]

    for coef_name, title, ax in targets:
        sub = main_models[main_models["coefficient"] == coef_name].copy()
        if sub.empty:
            ax.set_title(f"{title}\n(no data)")
            continue

        sub = sub.sort_values("model")
        y_pos = range(len(sub))

        for j, (_, row) in enumerate(sub.iterrows()):
            has_ci = pd.notna(row["lower_CI"]) and pd.notna(row["upper_CI"])
            if has_ci:
                ax.errorbar(row["OR"], j,
                            xerr=[[row["OR"] - row["lower_CI"]], [row["upper_CI"] - row["OR"]]],
                            fmt="o", capsize=3, color="darkblue", markersize=5)
            else:
                ax.plot(row["OR"], j, "o", color="darkblue", markersize=5)

            # p-value annotation
            p = row["p_value"]
            x_annot = row["upper_CI"] if has_ci else row["OR"] + 0.02
            if pd.notna(p):
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.annotate(f"p={p:.3f}{sig}", (x_annot + 0.01, j),
                            fontsize=7, va="center")

        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["model"], fontsize=8)
        ax.set_xlabel("Odds Ratio (95% CI)")
        ax.set_title(title, fontsize=11)

    fig.suptitle("Coefficient stability across model specifications\n"
                 "(Ethnicity shifts, Gender stable, Transfer reverses)", fontsize=13)
    fig.tight_layout()
    out = art_dir / "forest_plot_ethnicity_gender_transfer.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_aic_comparison(coef_df: pd.DataFrame, art_dir: Path):
    """Dot plot of AIC across models."""
    aic_df = coef_df[["model", "AIC"]].drop_duplicates().dropna(subset=["AIC"])
    if aic_df.empty:
        print("  No AIC values to plot")
        return

    aic_df = aic_df.sort_values("AIC")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(aic_df))
    colors = ["#d73027" if aic == aic_df["AIC"].min() else "#4575b4" for aic in aic_df["AIC"]]

    ax.barh(list(y_pos), aic_df["AIC"], color=colors, height=0.6, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(aic_df["model"], fontsize=9)
    ax.set_xlabel("AIC (lower = better fit)")
    ax.set_title("Model comparison by AIC\n(red = best model)", fontsize=12)

    # Annotate values
    for j, (_, row) in enumerate(aic_df.iterrows()):
        ax.annotate(f" {row['AIC']:.0f}", (row["AIC"], j), va="center", fontsize=8)

    fig.tight_layout()
    out = art_dir / "aic_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_stratified_coefficients(strat_df: pd.DataFrame, art_dir: Path):
    """Plot APACHE and ethnicity ORs by AS quartile."""
    if strat_df.empty:
        print("  No stratified results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for coef_name, title, ax in [
        ("apachescore", "APACHE Score OR by AS Quartile", axes[0]),
        ("C(ethnicity)[T.Caucasian]", "Ethnicity OR (Caucasian vs AA) by AS Quartile", axes[1]),
    ]:
        sub = strat_df[strat_df["coefficient"] == coef_name]
        if sub.empty:
            continue
        sub = sub.sort_values("quartile")
        x = range(len(sub))
        ax.errorbar(list(x), sub["OR"],
                    yerr=[sub["OR"] - sub["lower_CI"], sub["upper_CI"] - sub["OR"]],
                    fmt="o-", capsize=5, color="darkblue", markersize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["quartile"])
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Angular Separation Quartile")
        ax.set_ylabel("Odds Ratio (95% CI)")
        ax.set_title(title, fontsize=10)

    fig.suptitle("Stratified regressions: coefficient variation across documentation similarity groups",
                 fontsize=12)
    fig.tight_layout()
    out = art_dir / "stratified_coefficients.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Conditions to run regressions for. Default: all available.")
    ap.add_argument("--ref-hospital", type=int, default=73)
    ap.add_argument("--include-gee", action="store_true",
                    help="Include GEE models (M6, M7, M8, M12). Slow, skipped by default.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Specific model labels to run, e.g. --models M1 M3 M9 M13. Default: all.")
    ap.add_argument("--skip-extras", action="store_true",
                    help="Skip per-hospital, cluster, variance decomposition, and sensitivity analyses.")
    ap.add_argument("--skip-ri", action="store_true",
                    help="Skip the logistic random-intercept (GLMM) models.")
    args = ap.parse_args()

    with open(args.paths) as f:
        paths_cfg = yaml.safe_load(f)

    processed = Path(paths_cfg["outputs"]["processed"])
    art_root = Path(paths_cfg["outputs"]["artifacts"])

    if "apachePatientResult" not in paths_cfg["eicu"]:
        paths_cfg["eicu"]["apachePatientResult"] = "data/raw/apachePatientResult.csv.gz"

    # Discover conditions
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = sorted([p.name.replace("dist_", "").split("_min")[0]
                             for p in processed.glob("dist_*.npz")])
        conditions = sorted(set(conditions))

    if not conditions:
        conditions = ["diabetes"]

    print(f"Conditions: {conditions}")

    # ── Step 1: Build base cohort (once — without AS) ──
    print("=" * 60)
    print("STEP 1: Building base regression cohort")
    print("=" * 60)
    e = paths_cfg["eicu"]

    print("  Loading tables...")
    patient_df = pd.read_csv(e["patient"], low_memory=False)
    print(f"    patient: {len(patient_df):,} rows")

    apache_df = pd.read_csv(e["apachePatientResult"], low_memory=False)
    apache_iva = apache_df[apache_df["apacheversion"] == "IVa"].copy()
    print(f"    APACHE IVa: {len(apache_iva):,} rows")

    hospital_df = pd.read_csv(e["hospital"], low_memory=False)
    hospital_df.columns = [c.lower() for c in hospital_df.columns]
    print(f"    hospital: {len(hospital_df):,} rows")

    # Merge: patient + APACHE (inner join) + hospital (inner)
    base_cohort = patient_df.merge(apache_iva, on="patientunitstayid", how="inner")
    n_after_apache = len(base_cohort)
    base_cohort = base_cohort.merge(hospital_df, on="hospitalid", how="inner")
    print(f"    After APACHE merge: {n_after_apache:,}")
    print(f"    After hospital merge: {len(base_cohort):,}")

    # Clean age
    if "age" in base_cohort.columns:
        base_cohort["agenum"] = pd.to_numeric(
            base_cohort["age"].replace("> 89", "90"), errors="coerce")

    # Ensure binary outcome
    if base_cohort["actualhospitalmortality"].dtype == object:
        base_cohort["actualhospitalmortality"] = (
            base_cohort["actualhospitalmortality"] == "EXPIRED").astype(int)

    # Fill missing hospital characteristics with "Unknown" so all models
    # run on the same sample (statsmodels drops NaN rows silently)
    for col in ["region", "numbedscategory", "teachingstatus"]:
        if col in base_cohort.columns:
            n_missing = base_cohort[col].isna().sum()
            if n_missing > 0:
                base_cohort[col] = base_cohort[col].fillna("Unknown")
                print(f"    Filled {n_missing:,} missing {col} with 'Unknown'")

    print(f"    Base cohort: {len(base_cohort):,} rows, "
          f"{base_cohort['hospitalid'].nunique()} hospitals")

    # ── Step 2: APACHE missingness (once, general) ──
    print("\n" + "=" * 60)
    print("STEP 2: APACHE missingness analysis")
    print("=" * 60)
    general_dir = ensure_dir(art_root / "general")

    # Compute average angular separation across all conditions per hospital
    all_as_per_hospital = {}  # hospitalid -> {condition: AS}
    for condition in conditions:
        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            continue
        dist_df = pd.read_csv(dist_files[-1])
        ref_dists = dist_df[dist_df["hospital_1"] == args.ref_hospital][
            ["hospital_2", "angular_separation_deg"]]
        for _, row in ref_dists.iterrows():
            hid = int(row["hospital_2"])
            if hid not in all_as_per_hospital:
                all_as_per_hospital[hid] = {}
            all_as_per_hospital[hid][condition] = row["angular_separation_deg"]

    # Build hospital-level AS summary
    as_summary = []
    for hid, cond_vals in all_as_per_hospital.items():
        row = {"hospitalid": hid, "n_conditions": len(cond_vals), "mean_AS": np.mean(list(cond_vals.values()))}
        for c, v in cond_vals.items():
            row[f"AS_{c}"] = v
        as_summary.append(row)
    as_summary_df = pd.DataFrame(as_summary) if as_summary else pd.DataFrame()

    # Add average AS to a temp cohort for the missingness analysis
    temp_cohort = base_cohort.copy()
    if not as_summary_df.empty:
        temp_cohort = temp_cohort.merge(
            as_summary_df[["hospitalid", "mean_AS"]].rename(columns={"mean_AS": "Angular_Separation"}),
            on="hospitalid", how="left")
    else:
        temp_cohort["Angular_Separation"] = np.nan

    analyze_apache_missingness(paths_cfg, temp_cohort, general_dir)

    # Per-condition APACHE-AS correlations
    if not as_summary_df.empty:
        from scipy.stats import spearmanr
        # Get hospital-level APACHE rates
        all_patients = pd.read_csv(e["patient"], usecols=["patientunitstayid", "hospitalid"], low_memory=False)
        apache_pids = set(apache_df[apache_df["apacheversion"] == "IVa"]["patientunitstayid"])
        all_patients["has_apache"] = all_patients["patientunitstayid"].isin(apache_pids)
        hosp_rates = all_patients.groupby("hospitalid").agg(
            total=("patientunitstayid", "count"), with_apache=("has_apache", "sum")
        ).reset_index()
        hosp_rates["completion_rate"] = hosp_rates["with_apache"] / hosp_rates["total"]

        apache_as_corr = []
        for condition in conditions:
            col = f"AS_{condition}"
            if col not in as_summary_df.columns:
                continue
            merged = hosp_rates.merge(as_summary_df[["hospitalid", col]], on="hospitalid", how="inner")
            valid = merged.dropna(subset=[col, "completion_rate"])
            if len(valid) > 5:
                rho, p = spearmanr(valid[col], valid["completion_rate"])
                apache_as_corr.append({"condition": condition, "spearman_rho": round(rho, 3),
                                       "p_value": round(p, 4), "n_hospitals": len(valid)})
                print(f"  APACHE completion vs {condition} AS: ρ={rho:.3f}, p={p:.4f}")

        # Average AS
        merged = hosp_rates.merge(as_summary_df[["hospitalid", "mean_AS"]], on="hospitalid", how="inner")
        valid = merged.dropna(subset=["mean_AS", "completion_rate"])
        if len(valid) > 5:
            rho, p = spearmanr(valid["mean_AS"], valid["completion_rate"])
            apache_as_corr.append({"condition": "mean_across_conditions", "spearman_rho": round(rho, 3),
                                   "p_value": round(p, 4), "n_hospitals": len(valid)})
            print(f"  APACHE completion vs MEAN AS: ρ={rho:.3f}, p={p:.4f}")

        if apache_as_corr:
            pd.DataFrame(apache_as_corr).to_csv(general_dir / "apache_vs_as_correlations.csv", index=False)
            print(f"  Wrote {general_dir / 'apache_vs_as_correlations.csv'}")

    # ── Step 3+: Per-condition regressions ──
    warnings.filterwarnings("ignore")

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"REGRESSIONS: {condition.upper()}")
        print(f"{'='*60}")

        cond_dir = ensure_dir(art_root / condition)
        fig_dir = ensure_dir(cond_dir / "figures")
        tbl_dir = ensure_dir(cond_dir / "tables")

        # Load this condition's angular separation
        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            print(f"  No distance matrix for {condition}, skipping AS-dependent models")
            continue

        ref_dists = pd.read_csv(dist_files[-1])
        ref_dists = ref_dists[ref_dists["hospital_1"] == args.ref_hospital][
            ["hospital_2", "angular_separation_deg"]]
        ref_dists.columns = ["hospitalid", "Angular_Separation"]
        ref_dists = pd.concat([ref_dists,
                               pd.DataFrame([{"hospitalid": args.ref_hospital, "Angular_Separation": 0.0}])],
                              ignore_index=True).drop_duplicates("hospitalid", keep="first")  # long-form CSV already holds the self row

        cohort = base_cohort.merge(ref_dists, on="hospitalid", how="left")
        cohort["Angular_Separation_Quartile"] = pd.qcut(
            cohort["Angular_Separation"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop"
        )

        merged_df = cohort.dropna(subset=["Angular_Separation"]).copy()
        print(f"  Cohort with {condition} AS: {len(merged_df):,} rows, "
              f"{merged_df['hospitalid'].nunique()} hospitals")

        # Analytic sample: complete cases on the model variables, and hospitals
        # with outcome variation (a hospital with zero deaths has no finite fixed
        # effect). Every model below is fit on exactly this sample so that AIC,
        # LRT and reported n are comparable and honest.
        merged_df, sample_info = analytic_sample(merged_df, drop_no_variation=True)
        merged_df["Angular_Separation_Quartile"] = pd.qcut(
            merged_df["Angular_Separation"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
        print(f"  Analytic sample: {sample_info['n_analytic']:,} stays, "
              f"{sample_info['hospitals_analytic']} hospitals "
              f"(dropped {sample_info['n_before'] - sample_info['n_complete_case']:,} incomplete rows; "
              f"hospitals without outcome variation: {sample_info['hospitals_no_outcome_variation']})")
        import json
        with open(ensure_dir(art_root / condition / "tables") / "analytic_sample_info.json", "w") as f:
            json.dump(sample_info, f, indent=2)

        # Save per-condition cohort
        cohort.to_parquet(processed / f"regression_cohort_{condition}.parquet", index=False)

        # Build mean AS dataframe for M14
        mean_as_df = None
        if not as_summary_df.empty and "mean_AS" in as_summary_df.columns:
            mean_as_merge = as_summary_df[["hospitalid", "mean_AS"]].copy()
            mean_as_merge = mean_as_merge.rename(columns={"mean_AS": "Mean_Angular_Separation"})
            mean_as_df = merged_df.merge(mean_as_merge, on="hospitalid", how="inner")
            mean_as_df["Mean_AS_Quartile"] = pd.qcut(
                mean_as_df["Mean_Angular_Separation"], q=4,
                labels=["q1", "q2", "q3", "q4"], duplicates="drop"
            )
            print(f"  Mean AS cohort: {len(mean_as_df):,} rows")

        # Run models
        print(f"\n  Running models...")
        coef_df, fitted = run_all_models(cohort, merged_df, mean_as_df,
                                         include_gee=args.include_gee,
                                         include_ri=not args.skip_ri,
                                         model_filter=args.models)

        # LRT
        print(f"\n  Likelihood Ratio Tests...")
        lrt_df = compute_lrt(fitted)

        if not args.skip_extras:
            # Clustered SEs
            print(f"\n  Clustered SEs...")
            clustered_df = run_clustered_se_models(cohort, merged_df)

            # Stratified
            print(f"\n  Stratified regressions...")
            strat_df = run_stratified_models(merged_df)

            # Variance decomposition
            print(f"\n  Variance decomposition...")
            var_df = compute_variance_decomposition(cohort, merged_df, tbl_dir, fitted=fitted)

            # Per-hospital models with averaged betas (Harold's item 3)
            print(f"\n  Per-hospital averaged betas...")
            ph_df = run_per_hospital_models(merged_df, tbl_dir)

            # Cluster-based models (Harold's item 6)
            dist_npz_files = sorted(processed.glob(f"dist_{condition}*.npz"),
                                    key=lambda p: p.stat().st_mtime)
            if dist_npz_files:
                z = np.load(dist_npz_files[-1], allow_pickle=True)
                D_cond = z["D"]
                ids_cond = z["ids"].tolist()
                print(f"\n  Cluster-based models...")
                cluster_df = run_cluster_based_models(merged_df, D_cond, ids_cond, tbl_dir)
            else:
                cluster_df = pd.DataFrame()
        else:
            clustered_df = pd.DataFrame()
            strat_df = pd.DataFrame()
            cluster_df = pd.DataFrame()
            print("  Skipping extras (clustered SEs, stratified, variance decomp, per-hospital, clusters)")

        # Combine and save
        all_parts = [coef_df, clustered_df, strat_df]
        if not cluster_df.empty:
            all_parts.append(cluster_df)
        all_results = pd.concat(all_parts, ignore_index=True)
        all_results["condition"] = condition
        all_results.to_csv(tbl_dir / "all_model_results.csv", index=False)
        print(f"  Wrote {tbl_dir / 'all_model_results.csv'} ({len(all_results)} rows)")

        # LRT results
        if not lrt_df.empty:
            lrt_df.to_csv(tbl_dir / "likelihood_ratio_tests.csv", index=False)
            print(f"  Wrote {tbl_dir / 'likelihood_ratio_tests.csv'}")

        # One-row-per-model comparison table (replaces the old hard-coded table)
        write_model_comparison_table(all_results, lrt_df, fitted, tbl_dir)

        # Key coefficient tracking
        key_coefs = all_results[
            all_results["coefficient"].isin([
                "C(ethnicity)[T.Caucasian]", "C(gender)[T.Male]",
                "C(unitstaytype)[T.transfer]", "apachescore",
            ])
        ][["model", "coefficient", "OR", "lower_CI", "upper_CI", "p_value", "AIC"]].copy()
        key_coefs.to_csv(tbl_dir / "coefficient_tracking.csv", index=False)

        # Figures
        print(f"\n  Generating figures...")
        plot_forest(coef_df, fig_dir)
        plot_aic_comparison(coef_df, fig_dir)
        plot_stratified_coefficients(strat_df, fig_dir)

        print(f"  Done with {condition}")

    print(f"\nAll done. Check {art_root}/")


if __name__ == "__main__":
    main()
