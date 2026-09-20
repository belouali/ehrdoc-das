#!/usr/bin/env python
"""Ensemble (voting) prediction: nearest vs farthest hospitals.

Instead of pooling data, trains separate models per hospital and
combines predictions via ensemble (unweighted and size-weighted).
Also tests with and without hospital characteristics to isolate
the documentation effect.

Design:
  For each held-out hospital:
    1. Train one model per hospital in the nearest-k and farthest-k groups
    2. Predict on held-out hospital using each model
    3. Combine predictions: unweighted average, size-weighted average
    4. Repeat with hospital characteristics (region, bed size, teaching)
       added to the model to control for observable hospital differences

Prereqs: Run scripts 01-03

Usage:
  python scripts/16_ensemble_prediction.py --conditions diabetes
  python scripts/16_ensemble_prediction.py  # all conditions
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from ehrdoc.utils.io import ensure_dir

MIN_TRAIN = 100
K_VALUES = [1, 5, 10, 20]

PATIENT_FEATURES = [
    "apachescore", "agenum", "gender", "ethnicity",
    "unitstaytype", "unitadmitsource",
]
HOSP_FEATURES = [
    "region", "numbedscategory", "teachingstatus",
]

CAT_PATIENT = ["gender", "ethnicity", "unitstaytype", "unitadmitsource"]
CAT_HOSP = ["region", "numbedscategory", "teachingstatus"]
NUM_FEATURES = ["apachescore", "agenum"]

TARGET = "actualhospitalmortality"


def build_pipeline(use_hosp_chars: bool):
    """Build a logistic regression pipeline with appropriate preprocessing."""
    cat_cols = CAT_PATIENT + (CAT_HOSP if use_hosp_chars else [])
    num_cols = NUM_FEATURES

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"), cat_cols),
    ])

    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs")),
    ])


def train_single_hospital(data, hid, features, use_hosp_chars):
    """Train a model on one hospital's data with CV-based regularization selection."""
    from sklearn.model_selection import GridSearchCV

    hdata = data[data["hospitalid"] == hid]
    if len(hdata) < 50 or hdata[TARGET].nunique() < 2:
        return None, 0

    X = hdata[features]
    y = hdata[TARGET]
    try:
        pipe = build_pipeline(use_hosp_chars)
        n_events = y.sum()
        if n_events >= 15 and len(hdata) >= 150:
            # Enough data for CV-based selection
            grid = GridSearchCV(
                pipe,
                param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
                cv=3,
                scoring="roc_auc",
                n_jobs=1,
                refit=True,
            )
            grid.fit(X, y)
            return grid.best_estimator_, len(hdata)
        else:
            # Too few events for CV; use conservative regularization
            pipe.set_params(clf__C=0.1)
            pipe.fit(X, y)
            return pipe, len(hdata)
    except Exception:
        return None, 0


def ensemble_predict(models, size_weights, das_distances, X):
    """Average predicted probabilities from multiple models.
    
    Args:
        models: list of trained models
        size_weights: list of hospital patient counts
        das_distances: list of DAS distances from held-out hospital (lower = more similar)
        X: features for held-out hospital
    """
    preds = []
    valid_size_weights = []
    valid_das_distances = []
    for model, sw, dd in zip(models, size_weights, das_distances):
        try:
            p = model.predict_proba(X)[:, 1]
            preds.append(p)
            valid_size_weights.append(sw)
            valid_das_distances.append(dd)
        except Exception:
            continue

    if not preds:
        return None

    preds = np.array(preds)
    valid_size_weights = np.array(valid_size_weights, dtype=float)
    valid_das_distances = np.array(valid_das_distances, dtype=float)

    # Unweighted
    unweighted = preds.mean(axis=0)

    # Size-weighted
    sw_norm = valid_size_weights / valid_size_weights.sum()
    size_weighted = (preds * sw_norm[:, None]).sum(axis=0)

    # DAS-weighted: inverse distance (closer = more weight)
    # Add small epsilon to avoid division by zero for very similar hospitals
    inv_das = 1.0 / (valid_das_distances + 1.0)
    dw_norm = inv_das / inv_das.sum()
    das_weighted = (preds * dw_norm[:, None]).sum(axis=0)

    return {"unweighted": unweighted, "size_weighted": size_weighted, "das_weighted": das_weighted}


def evaluate(y_true, y_pred):
    """Compute AUC, Brier, AUPRC."""
    if y_pred is None or len(np.unique(y_true)) < 2:
        return {"AUC": np.nan, "Brier": np.nan, "AUPRC": np.nan}
    return {
        "AUC": roc_auc_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_pred),
        "AUPRC": average_precision_score(y_true, y_pred),
    }


def run_experiment(merged, condition, dist_file, fig_dir, tbl_dir):
    """Run ensemble prediction experiment."""

    # Load pairwise DAS
    das_df = pd.read_csv(dist_file)
    das_lookup = {}
    for _, r in das_df.iterrows():
        h1, h2 = int(r["hospital_1"]), int(r["hospital_2"])
        das_lookup[(h1, h2)] = r["angular_separation_deg"]
        das_lookup[(h2, h1)] = r["angular_separation_deg"]

    # Needed columns
    all_features = NUM_FEATURES + CAT_PATIENT
    needed = all_features + [TARGET, "hospitalid"]
    clean = merged.dropna(subset=needed).copy()

    hosp_counts = clean.groupby("hospitalid").size()
    large = hosp_counts[hosp_counts >= MIN_TRAIN].index.tolist()

    das_hospitals = set()
    for h1, h2 in das_lookup:
        das_hospitals.add(h1)
    large = [h for h in large if h in das_hospitals]

    print(f"    {len(large)} large hospitals (>= {MIN_TRAIN}, with DAS)")

    if len(large) < 20:
        print("    Too few hospitals")
        return

    # Pre-train all individual hospital models (patient features only)
    print(f"    Pre-training individual hospital models...")
    models = {}  # hid -> (model, n_train)
    for hid in large:
        m, n = train_single_hospital(clean, hid, all_features, False)
        if m is not None:
            models[hid] = (m, n)

    print(f"    {len(models)} patient-only models")

    results = []

    for i, hid in enumerate(large):
        holdout = clean[clean["hospitalid"] == hid]
        if len(holdout) < 50 or holdout[TARGET].nunique() < 2:
            continue

        # Get DAS to all other hospitals
        das_to_others = {}
        for h in large:
            if h == hid:
                continue
            key = (hid, h)
            if key in das_lookup:
                das_to_others[h] = das_lookup[key]

        pool = [h for h in large if h != hid and h in das_to_others]
        if len(pool) < 20:
            continue

        sorted_pool = sorted(pool, key=lambda h: das_to_others[h])

        y_true = holdout[TARGET].values
        X_holdout = holdout[all_features]

        for k in K_VALUES:
            nearest_hids = sorted_pool[:k]
            farthest_hids = sorted_pool[-k:]

            for group_name, group_hids in [("nearest", nearest_hids),
                                            ("farthest", farthest_hids)]:
                # Collect models, sizes, and DAS distances for this group
                group_models = []
                group_sizes = []
                group_das = []
                for gh in group_hids:
                    if gh in models:
                        m, n = models[gh]
                        group_models.append(m)
                        group_sizes.append(n)
                        group_das.append(das_to_others[gh])

                if not group_models:
                    continue

                # Ensemble predictions
                ens = ensemble_predict(group_models, group_sizes, group_das, X_holdout)
                if ens is None:
                    continue

                for ens_type in ["unweighted", "size_weighted", "das_weighted"]:
                    met = evaluate(y_true, ens[ens_type])
                    results.append({
                        "hospitalid": hid,
                        "n_holdout": len(holdout),
                        "k": k,
                        "group": group_name,
                        "ensemble": ens_type,
                        "AUC": met["AUC"],
                        "Brier": met["Brier"],
                        "AUPRC": met["AUPRC"],
                        "n_models": len(group_models),
                        "total_train": sum(group_sizes),
                    })

        if (i + 1) % 30 == 0:
            print(f"    Processed {i+1}/{len(large)}...")

    if not results:
        print("    No results")
        return

    res = pd.DataFrame(results)
    res.to_csv(tbl_dir / "ensemble_prediction_results.csv", index=False)
    print(f"    {len(res)} rows, {res['hospitalid'].nunique()} hospitals")

    # ── Summary ──
    print(f"\n    === Ensemble prediction summary ===")
    summary_rows = []

    for ensemble in ["unweighted", "size_weighted", "das_weighted"]:
        print(f"\n    Ensemble: {ensemble}")
        for k in K_VALUES:
            near = res[(res["k"] == k) & (res["group"] == "nearest") &
                       (res["ensemble"] == ensemble)]
            far = res[(res["k"] == k) & (res["group"] == "farthest") &
                      (res["ensemble"] == ensemble)]

            if len(near) < 10 or len(far) < 10:
                continue

            merged_df = near[["hospitalid", "AUC", "Brier", "AUPRC"]].merge(
                far[["hospitalid", "AUC", "Brier", "AUPRC"]],
                on="hospitalid", suffixes=("_near", "_far"))

            if len(merged_df) < 10:
                continue

            n = len(merged_df)
            row_summary = {"k": k, "ensemble": ensemble, "n": n}

            for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True)]:
                near_vals = merged_df[f"{metric}_near"]
                far_vals = merged_df[f"{metric}_far"]

                if higher_better:
                    delta = near_vals - far_vals
                    wins = (delta > 0).sum()
                else:
                    delta = far_vals - near_vals
                    wins = (delta > 0).sum()

                mean_delta = delta.mean()
                median_delta = delta.median()

                try:
                    d = delta[delta != 0]
                    _, p = wilcoxon(d)
                except Exception:
                    p = np.nan

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                row_summary[f"{metric}_near_mean"] = round(near_vals.mean(), 4)
                row_summary[f"{metric}_far_mean"] = round(far_vals.mean(), 4)
                row_summary[f"{metric}_wins"] = wins
                row_summary[f"{metric}_win_pct"] = round(100 * wins / n, 1)
                row_summary[f"{metric}_p"] = round(p, 4)
                row_summary[f"{metric}_mean_delta"] = round(mean_delta, 5)
                row_summary[f"{metric}_median_delta"] = round(median_delta, 5)

                print(f"      k={k:2d}: {metric:6s} near={near_vals.mean():.4f} "
                      f"far={far_vals.mean():.4f} wins={wins}/{n} "
                      f"({100*wins/n:.0f}%) p={p:.4f} {sig} "
                      f"mean\u0394={mean_delta:+.4f} med\u0394={median_delta:+.4f}")

            summary_rows.append(row_summary)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(tbl_dir / "ensemble_prediction_summary.csv", index=False)
        print(f"\n    Wrote {tbl_dir / 'ensemble_prediction_summary.csv'}")

    # ── Comparison figure ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for col, metric in enumerate(["AUC", "Brier", "AUPRC"]):
        ax = axes[col]
        for ens_type, marker, color in [("unweighted", "o", "steelblue"),
                                          ("size_weighted", "s", "coral"),
                                          ("das_weighted", "D", "forestgreen")]:
            k_vals = []
            win_pcts = []
            for k in K_VALUES:
                sub = [r for r in summary_rows
                       if r["k"] == k and r["ensemble"] == ens_type]
                if sub:
                    k_vals.append(k)
                    win_pcts.append(sub[0][f"{metric}_win_pct"])

            if k_vals:
                ax.plot(k_vals, win_pcts, marker=marker, color=color,
                        label=ens_type, linewidth=1.5)

        ax.axhline(50, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("k")
        ax.set_ylabel(f"{metric} nearest win %")
        ax.set_title(f"{metric}")
        ax.legend(fontsize=8)
        ax.set_ylim(30, 90)

    fig.suptitle(f"{condition.capitalize()}: Ensemble nearest vs farthest", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "ensemble_prediction_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Wrote {fig_dir / 'ensemble_prediction_comparison.png'}")


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

    # Detect conditions
    all_conditions = args.conditions
    if all_conditions is None:
        dist_files = sorted(processed.glob("dist_*.csv"))
        all_conditions = list({f.stem.split("_")[1] for f in dist_files})
    print(f"Conditions: {all_conditions}")

    warnings.filterwarnings("ignore")

    for condition in sorted(all_conditions):
        print(f"\n{'='*60}")
        print(f"ENSEMBLE PREDICTION: {condition.upper()}")
        print(f"{'='*60}")

        fig_dir = ensure_dir(art_root / condition / "figures")
        tbl_dir = ensure_dir(art_root / condition / "tables")

        # Load flags and merge DAS
        flags_file = processed / f"flags_{condition}.parquet"
        if not flags_file.exists():
            print(f"  No flags for {condition}")
            continue

        flags = pd.read_parquet(flags_file)
        flag_cols = [c for c in flags.columns
                     if c not in cohort.columns or c == "patientunitstayid"]
        merged = cohort.merge(flags[flag_cols], on="patientunitstayid", how="inner")

        # Load distances
        dist_files = sorted(processed.glob(f"dist_{condition}*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not dist_files:
            print(f"  No distances for {condition}")
            continue

        # Merge Angular_Separation from distances if needed
        das_df = pd.read_csv(dist_files[-1])

        print(f"  Merged: {len(merged):,} patients, {merged['hospitalid'].nunique()} hospitals")

        run_experiment(merged, condition, dist_files[-1], fig_dir, tbl_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
