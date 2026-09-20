#!/usr/bin/env python
"""Inference transportability: does DAS-guided site selection recover
local coefficient estimates better than dissimilar-site selection?

For each held-out hospital with enough data:
  1. Estimate coefficients locally (ground truth)
  2. Estimate from nearest-k hospitals (pooled, no FE)
  3. Estimate from farthest-k hospitals (pooled, no FE)
  4. Estimate from nearest-k with FE
  5. Estimate from farthest-k with FE
  6. Compare: relative bias, coverage, sign stability

Prereqs: Run scripts 01-03

Usage:
  python scripts/14_inference_transportability.py --conditions diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
import statsmodels.formula.api as smf

from ehrdoc.utils.io import ensure_dir

MIN_LOCAL = 500       # held-out hospital needs enough patients for stable local estimate
MIN_TRAIN = 100       # large hospitals for training pool
FORMULA_BASE = ("actualhospitalmortality ~ apachescore + agenum + "
                "C(gender) + C(ethnicity) + C(unitstaytype) + C(unitadmitsource)")
FORMULA_FE = FORMULA_BASE + " + C(hospitalid)"

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


def fit_logit(df, formula):
    """Fit logistic regression, return results or None."""
    try:
        if df["actualhospitalmortality"].nunique() < 2:
            return None
        model = smf.logit(formula, data=df)
        try:
            result = model.fit(disp=0, maxiter=50, method="bfgs")
        except Exception:
            result = model.fit(disp=0, maxiter=50, method="newton")
        return result
    except Exception:
        return None


def extract_coefs(result, coefs):
    """Extract OR, CI, p-value for tracked coefficients."""
    if result is None:
        return {}
    out = {}
    for c in coefs:
        if c in result.params.index:
            or_val = np.exp(result.params[c])
            ci = np.exp(result.conf_int().loc[c])
            out[c] = {
                "OR": or_val,
                "lower_CI": ci[0],
                "upper_CI": ci[1],
                "p_value": result.pvalues[c],
                "log_OR": result.params[c],
                "log_SE": result.bse[c],
            }
    return out


def get_pairwise_das(full_dists, hid):
    rows = full_dists[
        (full_dists["hospital_1"] == hid) | (full_dists["hospital_2"] == hid)
    ]
    result = {}
    for _, r in rows.iterrows():
        other = r["hospital_2"] if r["hospital_1"] == hid else r["hospital_1"]
        result[int(other)] = r["angular_separation_deg"]
    return result


def prepare_data(df):
    needed = ["apachescore", "agenum", "gender", "ethnicity", "unitstaytype",
              "unitadmitsource", "actualhospitalmortality", "hospitalid",
              "Angular_Separation"]
    return df.dropna(subset=needed).copy()


def run_experiment(df, condition, dist_csv, fig_dir, tbl_dir):
    print(f"\n  Inference transportability ({condition}):")

    clean = prepare_data(df)
    full_dists = pd.read_csv(dist_csv)

    hosp_counts = clean.groupby("hospitalid").size()
    large = hosp_counts[hosp_counts >= MIN_TRAIN].index.tolist()
    evaluable = hosp_counts[hosp_counts >= MIN_LOCAL].index.tolist()
    evaluable = [h for h in evaluable if h in large]  # must also be in distance matrix
    print(f"    {len(large)} large hospitals (training pool)")
    print(f"    {len(evaluable)} hospitals with >= {MIN_LOCAL} patients (evaluable)")

    results = []

    for i, hid in enumerate(evaluable):
        holdout = clean[clean["hospitalid"] == hid]
        if holdout["actualhospitalmortality"].nunique() < 2:
            continue

        # Local estimate (ground truth)
        local_result = fit_logit(holdout, FORMULA_BASE)
        if local_result is None:
            continue
        local_coefs = extract_coefs(local_result, TRACKED_COEFS)
        if not local_coefs:
            continue

        # Get DAS to other large hospitals
        das_to_others = get_pairwise_das(full_dists, hid)
        pool = [h for h in large if h != hid and h in das_to_others]
        if len(pool) < 20:
            continue

        sorted_pool = sorted(pool, key=lambda h: das_to_others[h])

        for k in [1, 5, 10, 20]:
            nearest_hids = sorted_pool[:k]
            farthest_hids = sorted_pool[-k:]

            for group_name, group_hids in [("nearest", nearest_hids),
                                            ("farthest", farthest_hids)]:
                group_data = clean[clean["hospitalid"].isin(group_hids)]
                if len(group_data) < 50 or group_data["actualhospitalmortality"].nunique() < 2:
                    continue

                # ── Strategy 1: Pooled no-FE ──
                res_nofe = fit_logit(group_data, FORMULA_BASE)
                coefs_nofe = extract_coefs(res_nofe, TRACKED_COEFS)

                # ── Strategy 2: No FE models (skip — too slow, inconsistent results) ──
                coefs_fe = {}

                # ── Strategy 3: Meta-analytic (fit each hospital, IV-weighted) ──
                meta_coefs = {}
                if k > 1:
                    per_hosp_estimates = {c: [] for c in TRACKED_COEFS}
                    for gh in group_hids:
                        gh_data = clean[clean["hospitalid"] == gh]
                        if len(gh_data) < 50 or gh_data["actualhospitalmortality"].nunique() < 2:
                            continue
                        gh_result = fit_logit(gh_data, FORMULA_BASE)
                        if gh_result is None:
                            continue
                        gh_coefs = extract_coefs(gh_result, TRACKED_COEFS)
                        for c in TRACKED_COEFS:
                            if c in gh_coefs and gh_coefs[c]["log_SE"] > 0:
                                per_hosp_estimates[c].append({
                                    "log_OR": gh_coefs[c]["log_OR"],
                                    "log_SE": gh_coefs[c]["log_SE"],
                                })

                    for c in TRACKED_COEFS:
                        ests = per_hosp_estimates[c]
                        if len(ests) >= 2:
                            weights = [1.0 / (e["log_SE"] ** 2) for e in ests]
                            w_sum = sum(weights)
                            meta_log_or = sum(w * e["log_OR"] for w, e in zip(weights, ests)) / w_sum
                            meta_se = np.sqrt(1.0 / w_sum)
                            meta_or = np.exp(meta_log_or)
                            meta_lower = np.exp(meta_log_or - 1.96 * meta_se)
                            meta_upper = np.exp(meta_log_or + 1.96 * meta_se)
                            meta_coefs[c] = {
                                "OR": meta_or, "log_OR": meta_log_or, "log_SE": meta_se,
                                "lower_CI": meta_lower, "upper_CI": meta_upper,
                            }
                elif k == 1:
                    # k=1: meta = pooled (same thing)
                    meta_coefs = coefs_nofe

                for coef in TRACKED_COEFS:
                    if coef not in local_coefs:
                        continue

                    local = local_coefs[coef]
                    nofe = coefs_nofe.get(coef, {})
                    fe = coefs_fe.get(coef, {})
                    meta = meta_coefs.get(coef, {})

                    if not nofe:
                        continue

                    # Absolute bias (avoids division-by-near-zero)
                    abs_bias_nofe = abs(nofe["log_OR"] - local["log_OR"])
                    abs_bias_meta = abs(meta["log_OR"] - local["log_OR"]) if meta else np.nan

                    # Relative bias
                    rel_bias_nofe = abs_bias_nofe / abs(local["log_OR"]) \
                        if local["log_OR"] != 0 else np.nan

                    # Coverage: does CI contain local OR?
                    coverage_nofe = 1 if (nofe["lower_CI"] <= local["OR"] <= nofe["upper_CI"]) else 0
                    coverage_meta = 1 if (meta and meta["lower_CI"] <= local["OR"] <= meta["upper_CI"]) else np.nan

                    # Sign agreement
                    sign_agree_nofe = 1 if ((nofe["OR"] > 1) == (local["OR"] > 1)) else 0
                    sign_agree_meta = 1 if (meta and (meta["OR"] > 1) == (local["OR"] > 1)) else np.nan

                    row = {
                        "hospitalid": hid,
                        "n_holdout": len(holdout),
                        "k": k,
                        "group": group_name,
                        "coefficient": coef,
                        "coef_label": COEF_LABELS.get(coef, coef),
                        "local_OR": local["OR"],
                        "local_log_OR": local["log_OR"],
                        # Pooled no-FE
                        "pooled_OR_noFE": nofe["OR"],
                        "pooled_log_OR_noFE": nofe["log_OR"],
                        "pooled_lower_CI_noFE": nofe["lower_CI"],
                        "pooled_upper_CI_noFE": nofe["upper_CI"],
                        "abs_bias_noFE": abs_bias_nofe,
                        "relative_bias_noFE": rel_bias_nofe,
                        "coverage_noFE": coverage_nofe,
                        "sign_agree_noFE": sign_agree_nofe,
                        # Meta-analytic
                        "meta_OR": meta.get("OR", np.nan),
                        "meta_log_OR": meta.get("log_OR", np.nan),
                        "meta_lower_CI": meta.get("lower_CI", np.nan),
                        "meta_upper_CI": meta.get("upper_CI", np.nan),
                        "abs_bias_meta": abs_bias_meta,
                        "coverage_meta": coverage_meta,
                        "sign_agree_meta": sign_agree_meta,
                    }

                    if fe and coef in coefs_fe:
                        rel_bias_fe = (fe["log_OR"] - local["log_OR"]) / abs(local["log_OR"]) \
                            if local["log_OR"] != 0 else np.nan
                        coverage_fe = 1 if (fe["lower_CI"] <= local["OR"] <= fe["upper_CI"]) else 0
                        sign_agree_fe = 1 if ((fe["OR"] > 1) == (local["OR"] > 1)) else 0

                        row["pooled_OR_FE"] = fe["OR"]
                        row["pooled_log_OR_FE"] = fe["log_OR"]
                        row["relative_bias_FE"] = rel_bias_fe
                        row["coverage_FE"] = coverage_fe
                        row["sign_agree_FE"] = sign_agree_fe

                        # Flip: sign changes between noFE and FE
                        row["sign_flip"] = 1 if ((nofe["OR"] > 1) != (fe["OR"] > 1)) else 0
                    else:
                        row["sign_flip"] = 0

                    results.append(row)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(evaluable)}...")

    if not results:
        print("    No valid results")
        return

    res = pd.DataFrame(results)
    res.to_csv(tbl_dir / "inference_transportability.csv", index=False)
    print(f"    {len(res)} rows, {res['hospitalid'].nunique()} hospitals")

    # ── Summary ──
    print(f"\n    === Panel A: Coefficient recovery ===")
    summary_rows = []

    for k in [1, 5, 10, 20]:
        print(f"\n    k={k}:")
        for coef in TRACKED_COEFS:
            label = COEF_LABELS.get(coef, coef)
            near = res[(res["k"] == k) & (res["group"] == "nearest") &
                       (res["coefficient"] == coef)].copy()
            far = res[(res["k"] == k) & (res["group"] == "farthest") &
                      (res["coefficient"] == coef)].copy()

            if len(near) < 5 or len(far) < 5:
                continue

            # Merge on hospitalid for paired comparison
            merge_cols_pooled = ["hospitalid", "abs_bias_noFE", "relative_bias_noFE",
                                  "coverage_noFE", "sign_agree_noFE"]
            merge_cols_meta = ["hospitalid", "abs_bias_meta", "coverage_meta",
                                "sign_agree_meta"]

            merged = near[merge_cols_pooled + ["abs_bias_meta", "coverage_meta", "sign_agree_meta"]].merge(
                far[merge_cols_pooled + ["abs_bias_meta", "coverage_meta", "sign_agree_meta"]],
                on="hospitalid", suffixes=("_near", "_far"))

            if len(merged) < 5:
                continue
            n = len(merged)

            # ── Pooled no-FE ──
            abs_bias_near_p = merged["abs_bias_noFE_near"].mean()
            abs_bias_far_p = merged["abs_bias_noFE_far"].mean()
            near_closer_p = (merged["abs_bias_noFE_near"] < merged["abs_bias_noFE_far"]).sum()
            try:
                diff = merged["abs_bias_noFE_near"] - merged["abs_bias_noFE_far"]
                diff = diff[diff != 0]
                _, p_pooled = wilcoxon(diff)
            except Exception:
                p_pooled = np.nan

            cov_near_p = merged["coverage_noFE_near"].mean()
            cov_far_p = merged["coverage_noFE_far"].mean()
            sign_near_p = merged["sign_agree_noFE_near"].mean()
            sign_far_p = merged["sign_agree_noFE_far"].mean()

            # ── Meta-analytic ──
            m_valid = merged.dropna(subset=["abs_bias_meta_near", "abs_bias_meta_far"])
            if len(m_valid) >= 5:
                abs_bias_near_m = m_valid["abs_bias_meta_near"].mean()
                abs_bias_far_m = m_valid["abs_bias_meta_far"].mean()
                near_closer_m = (m_valid["abs_bias_meta_near"] < m_valid["abs_bias_meta_far"]).sum()
                try:
                    diff_m = m_valid["abs_bias_meta_near"] - m_valid["abs_bias_meta_far"]
                    diff_m = diff_m[diff_m != 0]
                    _, p_meta = wilcoxon(diff_m)
                except Exception:
                    p_meta = np.nan
                cov_near_m = m_valid["coverage_meta_near"].mean()
                cov_far_m = m_valid["coverage_meta_far"].mean()
                n_meta = len(m_valid)
            else:
                abs_bias_near_m = abs_bias_far_m = near_closer_m = p_meta = np.nan
                cov_near_m = cov_far_m = np.nan
                n_meta = 0

            sig_p = "***" if p_pooled < 0.001 else "**" if p_pooled < 0.01 else "*" if p_pooled < 0.05 else ""
            sig_m = "***" if p_meta < 0.001 else "**" if p_meta < 0.01 else "*" if p_meta < 0.05 else "" if not np.isnan(p_meta) else ""

            print(f"      {label:12s} (n={n}):")
            print(f"        Pooled:  |bias| near={abs_bias_near_p:.3f} far={abs_bias_far_p:.3f} "
                  f"closer={near_closer_p}/{n} ({100*near_closer_p/n:.0f}%) p={p_pooled:.4f} {sig_p} "
                  f"| cov near={cov_near_p:.0%} far={cov_far_p:.0%}")
            if n_meta >= 5:
                print(f"        Meta:    |bias| near={abs_bias_near_m:.3f} far={abs_bias_far_m:.3f} "
                      f"closer={near_closer_m}/{n_meta} ({100*near_closer_m/n_meta:.0f}%) p={p_meta:.4f} {sig_m} "
                      f"| cov near={cov_near_m:.0%} far={cov_far_m:.0%}")

            summary_rows.append({
                "k": k, "coefficient": label, "n_pooled": n, "n_meta": n_meta,
                "abs_bias_nearest_pooled": round(abs_bias_near_p, 4),
                "abs_bias_farthest_pooled": round(abs_bias_far_p, 4),
                "nearest_closer_pooled_pct": round(100 * near_closer_p / n, 1),
                "wilcoxon_p_pooled": round(p_pooled, 4),
                "coverage_nearest_pooled": round(cov_near_p, 3),
                "coverage_farthest_pooled": round(cov_far_p, 3),
                "abs_bias_nearest_meta": round(abs_bias_near_m, 4) if not np.isnan(abs_bias_near_m) else np.nan,
                "abs_bias_farthest_meta": round(abs_bias_far_m, 4) if not np.isnan(abs_bias_far_m) else np.nan,
                "nearest_closer_meta_pct": round(100 * near_closer_m / n_meta, 1) if n_meta > 0 else np.nan,
                "wilcoxon_p_meta": round(p_meta, 4) if not np.isnan(p_meta) else np.nan,
                "coverage_nearest_meta": round(cov_near_m, 3) if not np.isnan(cov_near_m) else np.nan,
                "coverage_farthest_meta": round(cov_far_m, 3) if not np.isnan(cov_far_m) else np.nan,
            })

    # ── Panel B: Sign stability (flip rate with vs without FE) ──
    print(f"\n    === Panel B: Sign stability (flip rate no-FE vs FE) ===")
    for k in [5, 10, 20]:
        print(f"\n    k={k}:")
        for coef in TRACKED_COEFS:
            label = COEF_LABELS.get(coef, coef)
            for group in ["nearest", "farthest"]:
                sub = res[(res["k"] == k) & (res["group"] == group) &
                          (res["coefficient"] == coef) &
                          (res["sign_flip"].notna())]
                if len(sub) < 5:
                    continue
                flip_rate = sub["sign_flip"].mean()
                print(f"      {label:12s} {group:8s}: flip rate = {flip_rate:.0%} "
                      f"({int(sub['sign_flip'].sum())}/{len(sub)})")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(tbl_dir / "inference_transportability_summary.csv", index=False)
        print(f"\n    Wrote {tbl_dir / 'inference_transportability_summary.csv'}")

    # ── Figure: bias comparison for k=1 and k=5 ──
    for k_plot in [1, 5]:
        fig, axes = plt.subplots(1, len(TRACKED_COEFS), figsize=(4 * len(TRACKED_COEFS), 5))
        if len(TRACKED_COEFS) == 1:
            axes = [axes]

        for idx, coef in enumerate(TRACKED_COEFS):
            ax = axes[idx]
            label = COEF_LABELS.get(coef, coef)

            near = res[(res["k"] == k_plot) & (res["group"] == "nearest") &
                       (res["coefficient"] == coef)]
            far = res[(res["k"] == k_plot) & (res["group"] == "farthest") &
                      (res["coefficient"] == coef)]

            merged = near[["hospitalid", "relative_bias_noFE"]].merge(
                far[["hospitalid", "relative_bias_noFE"]],
                on="hospitalid", suffixes=("_near", "_far"))

            if len(merged) < 5:
                ax.text(0.5, 0.5, "n<5", ha="center", va="center", transform=ax.transAxes)
                continue

            ax.scatter(merged["relative_bias_noFE_far"].abs(),
                       merged["relative_bias_noFE_near"].abs(),
                       alpha=0.5, s=20)
            lim = max(merged["relative_bias_noFE_far"].abs().max(),
                      merged["relative_bias_noFE_near"].abs().max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
            ax.set_xlabel(f"|Relative bias| farthest-{k_plot}")
            ax.set_ylabel(f"|Relative bias| nearest-{k_plot}")
            near_closer = (merged["relative_bias_noFE_near"].abs() <
                          merged["relative_bias_noFE_far"].abs()).sum()
            ax.set_title(f"{label}\nNearest closer: {near_closer}/{len(merged)}")

        fig.suptitle(f"{condition.capitalize()}: Inference transportability (k={k_plot})",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / f"inference_transportability_k{k_plot}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Wrote {fig_dir / f'inference_transportability_k{k_plot}.png'}")


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
        print(f"INFERENCE TRANSPORTABILITY: {condition.upper()}")
        print(f"{'='*60}")

        fig_dir = ensure_dir(art_root / condition / "figures")
        tbl_dir = ensure_dir(art_root / condition / "tables")

        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            print(f"  No distances for {condition}")
            continue

        ref_dists = pd.read_csv(dist_files[-1])
        ref_hospital = 73
        ref_sub = ref_dists[ref_dists["hospital_1"] == ref_hospital][
            ["hospital_2", "angular_separation_deg"]]
        ref_sub.columns = ["hospitalid", "Angular_Separation"]
        ref_sub = pd.concat([ref_sub,
                              pd.DataFrame([{"hospitalid": ref_hospital,
                                             "Angular_Separation": 0.0}])],
                             ignore_index=True).drop_duplicates("hospitalid", keep="first")  # long-form CSV already holds the self row

        merged = cohort.merge(ref_sub, on="hospitalid", how="left")
        merged = merged.dropna(subset=["Angular_Separation"]).copy()
        print(f"  Merged: {len(merged):,} patients, {merged['hospitalid'].nunique()} hospitals")

        run_experiment(merged, condition, dist_files[-1], fig_dir, tbl_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
