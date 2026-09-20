#!/usr/bin/env python
"""Small hospital transportability experiment.

Computes Venn diagram vectors for ALL hospitals (no min-patient threshold),
then tests: can DAS guide which large hospitals to train on when deploying
to a small hospital that can't build its own model?

Design:
  - Training pool: large hospitals (>=100 patients)
  - Test set: small hospitals (20-99 patients)
  - For each small hospital, compute its DAS to each large hospital
  - Train on k nearest vs k farthest large hospitals
  - Compare AUC, Brier, AUPRC

Prereqs: Run scripts 01-02 (patient flags + vectors)

Usage:
  python scripts/13_small_hospital_experiment.py --conditions diabetes
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
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ehrdoc.utils.io import ensure_dir

MIN_LARGE = 100
MIN_SMALL = 20
MAX_SMALL = 99
NUMERIC_COLS = ["apachescore", "agenum"]
CATEGORICAL_COLS = ["gender", "ethnicity", "unitstaytype", "unitadmitsource"]


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)),
    ])


def evaluate(y_true, y_prob):
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return {"AUC": np.nan, "Brier": np.nan, "AUPRC": np.nan}
    try:
        return {
            "AUC": roc_auc_score(y_true, y_prob),
            "Brier": brier_score_loss(y_true, y_prob),
            "AUPRC": average_precision_score(y_true, y_prob),
        }
    except Exception:
        return {"AUC": np.nan, "Brier": np.nan, "AUPRC": np.nan}


def prepare_data(df):
    needed = NUMERIC_COLS + CATEGORICAL_COLS + ["actualhospitalmortality", "hospitalid"]
    return df.dropna(subset=needed).copy()


def fit_and_predict(clean, train_hids, holdout, features, target):
    train = clean[clean["hospitalid"].isin(train_hids)]
    if len(train) < 50 or train[target].nunique() < 2:
        return {"AUC": np.nan, "Brier": np.nan, "AUPRC": np.nan, "n_train": len(train)}
    try:
        pipe = build_pipeline()
        pipe.fit(train[features], train[target])
        met = evaluate(holdout[target].values,
                       pipe.predict_proba(holdout[features])[:, 1])
        met["n_train"] = len(train)
        return met
    except Exception:
        return {"AUC": np.nan, "Brier": np.nan, "AUPRC": np.nan, "n_train": len(train)}


def compute_das(vec1, vec2):
    """Angular separation in degrees between two normalized vectors."""
    # Normalize
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    v1 = v1 / n1
    v2 = v2 / n2
    cos_sim = np.clip(np.dot(v1, v2), -1, 1)
    return np.degrees(np.arccos(cos_sim))


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
    for col in ["region", "numbedscategory", "teachingstatus"]:
        if col in cohort.columns:
            cohort[col] = cohort[col].fillna("Unknown")

    print(f"  Cohort: {len(cohort):,} rows, {cohort['hospitalid'].nunique()} hospitals")
    warnings.filterwarnings("ignore")

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"SMALL HOSPITAL EXPERIMENT: {condition.upper()}")
        print(f"{'='*60}")

        fig_dir = ensure_dir(art_root / condition / "figures")
        tbl_dir = ensure_dir(art_root / condition / "tables")

        # Load patient flags for this condition
        flags_path = processed / f"flags_{condition}.parquet"
        if not flags_path.exists():
            print(f"  ERROR: {flags_path} not found. Run script 01 first.")
            continue
        cond_flags = pd.read_parquet(flags_path)
        print(f"  Patient flags: {len(cond_flags):,} rows")

        # ── Compute Venn vectors for ALL hospitals (no threshold) ──
        dx_col = "exists_in_diagnosis"
        ph_col = "exists_in_pastHistory"
        rx_col = "exists_in_medication"

        # Merge with cohort to get hospitalid
        # flags already carry hospitalid (from script 01); keep the cohort's copy
        cond_merged = cond_flags.drop(columns=["hospitalid"], errors="ignore").merge(
            cohort[["patientunitstayid", "hospitalid"]].drop_duplicates(),
            on="patientunitstayid", how="inner"
        )

        # Compute 7-lobe vectors per hospital
        print(f"  Computing Venn vectors for all hospitals...")
        lobe_cols = {
            "DxYPhYRxY": (True, True, True),
            "DxYPhYRxN": (True, True, False),
            "DxYPhNRxY": (True, False, True),
            "DxYPhNRxN": (True, False, False),
            "DxNPhYRxY": (False, True, True),
            "DxNPhYRxN": (False, True, False),
            "DxNPhNRxY": (False, False, True),
        }

        vectors = {}
        hosp_patient_counts = {}
        for hid, grp in cond_merged.groupby("hospitalid"):
            n = len(grp)
            if n < MIN_SMALL:  # need at least 20 patients
                continue
            vec = []
            for lobe_name, (dx, ph, rx) in lobe_cols.items():
                mask = (grp[dx_col] == dx) & (grp[ph_col] == ph) & (grp[rx_col] == rx)
                vec.append(mask.sum())
            vec = np.array(vec, dtype=float)
            total = vec.sum()
            if total > 0:
                vec = vec / total
            vectors[hid] = vec
            hosp_patient_counts[hid] = n

        print(f"  {len(vectors)} hospitals with >= {MIN_SMALL} patients and valid vectors")

        # Split into large and small
        clean = prepare_data(cohort)
        hosp_sizes = clean.groupby("hospitalid").size()

        large_hids = [h for h in vectors if hosp_sizes.get(h, 0) >= MIN_LARGE]
        small_hids = [h for h in vectors if MIN_SMALL <= hosp_sizes.get(h, 0) <= MAX_SMALL]
        print(f"  Large (training): {len(large_hids)} hospitals")
        print(f"  Small (testing): {len(small_hids)} hospitals")

        if len(small_hids) < 5:
            print(f"  Too few small hospitals to test")
            continue

        # ── Compute pairwise DAS: each small hospital to each large hospital ──
        features = NUMERIC_COLS + CATEGORICAL_COLS
        target = "actualhospitalmortality"

        results = []
        for i, hid in enumerate(small_hids):
            holdout = clean[clean["hospitalid"] == hid]
            if holdout[target].nunique() < 2 or len(holdout) < MIN_SMALL:
                continue

            # DAS from this small hospital to each large hospital
            das_to_large = {}
            for lh in large_hids:
                d = compute_das(vectors[hid], vectors[lh])
                if not np.isnan(d):
                    das_to_large[lh] = d

            if len(das_to_large) < 20:
                continue

            sorted_by_das = sorted(das_to_large.keys(), key=lambda h: das_to_large[h])

            # Baseline: all large hospitals
            all_met = fit_and_predict(clean, large_hids, holdout, features, target)

            row = {
                "hospitalid": hid,
                "n_holdout": len(holdout),
                "mortality_rate": holdout[target].mean(),
                "mean_das_to_large": np.mean(list(das_to_large.values())),
                "min_das_to_large": min(das_to_large.values()),
                "all_AUC": all_met["AUC"], "all_Brier": all_met["Brier"],
                "all_AUPRC": all_met["AUPRC"], "all_n_train": all_met["n_train"],
            }

            for k in [5, 10, 20]:
                near_met = fit_and_predict(clean, sorted_by_das[:k], holdout, features, target)
                far_met = fit_and_predict(clean, sorted_by_das[-k:], holdout, features, target)
                for prefix, met in [("near", near_met), ("far", far_met)]:
                    row[f"{prefix}{k}_AUC"] = met["AUC"]
                    row[f"{prefix}{k}_Brier"] = met["Brier"]
                    row[f"{prefix}{k}_AUPRC"] = met["AUPRC"]
                    row[f"{prefix}{k}_n_train"] = met["n_train"]

            results.append(row)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(small_hids)}...")

        if not results:
            print("    No valid results for small hospitals")
            continue

        res = pd.DataFrame(results)
        res.to_csv(tbl_dir / "small_hospital_transportability.csv", index=False)
        print(f"\n  Completed {len(res)} small hospitals")

        # ── Summary ──
        print(f"\n  === Small Hospital Results ===")
        summary_rows = []
        for k in [5, 10, 20]:
            v = res.dropna(subset=[f"near{k}_AUC", f"far{k}_AUC"])
            if len(v) < 5:
                continue

            print(f"\n  k={k} (n={len(v)}):")
            for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True)]:
                near = v[f"near{k}_{metric}"]
                far = v[f"far{k}_{metric}"]
                all_val = v[f"all_{metric}"]

                wins = (near > far).sum() if higher_better else (near < far).sum()
                try:
                    diff = near - far
                    diff = diff[diff != 0]
                    _, p = wilcoxon(diff)
                except Exception:
                    p = np.nan
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                print(f"    {metric:6s}: near={near.mean():.4f}, far={far.mean():.4f}, "
                      f"all={all_val.mean():.4f} | wins={wins}/{len(v)} ({100*wins/len(v):.0f}%) "
                      f"p={p:.4f} {sig}")

                summary_rows.append({
                    "k": k, "metric": metric,
                    "mean_nearest": round(near.mean(), 4),
                    "mean_farthest": round(far.mean(), 4),
                    "mean_all": round(all_val.mean(), 4),
                    "nearest_wins": wins, "n": len(v),
                    "wilcoxon_p": round(p, 4),
                })

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                tbl_dir / "small_hospital_summary.csv", index=False)

        # ── Figure: paired-difference for small hospitals ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for col_idx, k in enumerate([5, 10, 20]):
            v = res.dropna(subset=[f"near{k}_AUC", f"far{k}_AUC",
                                    f"near{k}_Brier", f"far{k}_Brier"])
            if len(v) < 5:
                continue

            # ΔAUC
            ax = axes[0, col_idx]
            delta = v[f"near{k}_AUC"] - v[f"far{k}_AUC"]
            ax.scatter(range(len(delta)), delta.sort_values().values,
                       alpha=0.6, s=15, color=["#2c7bb6" if d > 0 else "#d7191c"
                                                for d in delta.sort_values().values])
            ax.axhline(0, color="black", linewidth=1)
            ax.axhline(delta.mean(), color="#2c7bb6", ls="--", lw=2,
                       label=f"Mean Δ={delta.mean():+.4f}")
            wins = (delta > 0).sum()
            try:
                d = delta[delta != 0]; _, p = wilcoxon(d)
            except: p = np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_title(f"k={k}: ΔAUC | {wins}/{len(v)} ({100*wins/len(v):.0f}%) p={p:.4f} {sig}")
            ax.set_ylabel("ΔAUC (near − far)")
            ax.legend(fontsize=8)

            # ΔBrier
            ax = axes[1, col_idx]
            delta = v[f"near{k}_Brier"] - v[f"far{k}_Brier"]
            ax.scatter(range(len(delta)), delta.sort_values().values,
                       alpha=0.6, s=15, color=["#2c7bb6" if d < 0 else "#d7191c"
                                                for d in delta.sort_values().values])
            ax.axhline(0, color="black", linewidth=1)
            ax.axhline(delta.mean(), color="#2c7bb6", ls="--", lw=2,
                       label=f"Mean Δ={delta.mean():+.4f}")
            wins = (delta < 0).sum()
            try:
                d = delta[delta != 0]; _, p = wilcoxon(d)
            except: p = np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_title(f"k={k}: ΔBrier | {wins}/{len(v)} ({100*wins/len(v):.0f}%) p={p:.4f} {sig}")
            ax.set_ylabel("ΔBrier (near − far)")
            ax.legend(fontsize=8)

        fig.suptitle(f"{condition.capitalize()}: Small hospitals ({MIN_SMALL}-{MAX_SMALL} patients) — "
                     f"trained on documentation-similar vs dissimilar large hospitals",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "small_hospital_transportability.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {fig_dir / 'small_hospital_transportability.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
