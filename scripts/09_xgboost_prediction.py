#!/usr/bin/env python
"""DAS-guided training set selection for model transportability.

For each held-out hospital, compares model performance when trained on:
  - ALL large hospitals (baseline)
  - K nearest large hospitals by DAS
  - K farthest large hospitals by DAS
  - Same DAS quartile large hospitals

Two evaluation cohorts:
  - Large hospitals (>=100 patients): LOHO, each takes a turn as held-out
  - Small hospitals (20-99 patients): external evaluation only, always
    trained on large hospitals. This is the practical scenario: small
    hospitals borrow models from the network.

Prereqs: Run scripts 01-03

Usage:
  python scripts/09_xgboost_prediction.py --conditions diabetes
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
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ehrdoc.utils.io import ensure_dir

MIN_TRAIN = 100       # large hospitals used for training
MIN_SMALL = 20        # small hospitals for external eval
N_RANDOM = 20         # random draws of k hospitals per held-out hospital (random-k baseline)
METRICS = ("AUC", "Brier", "AUPRC", "cal_slope", "cal_intercept")
NUMERIC_COLS = ["apachescore", "agenum"]
CATEGORICAL_COLS = ["gender", "ethnicity", "unitstaytype", "unitadmitsource"]
HOSP_CHAR_COLS = ["region", "numbedscategory", "teachingstatus"]


def build_pipeline(use_hosp_chars=False):
    cat_cols = CATEGORICAL_COLS + (HOSP_CHAR_COLS if use_hosp_chars else [])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)),
    ])


def _nan_metrics():
    return {m: np.nan for m in METRICS}


def evaluate(y_true, y_prob) -> dict:
    """Discrimination (AUC, AUPRC), overall probabilistic accuracy (Brier) and
    calibration: slope of logit(y) on logit(p), and calibration-in-the-large as
    logit(mean observed) - logit(mean predicted) (0 = calibrated on average)."""
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return _nan_metrics()
    try:
        out = {
            "AUC": roc_auc_score(y_true, y_prob),
            "Brier": brier_score_loss(y_true, y_prob),
            "AUPRC": average_precision_score(y_true, y_prob),
        }
        p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
        lp = np.log(p / (1 - p)).reshape(-1, 1)
        y = np.asarray(y_true)
        try:
            cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(lp, y)
            out["cal_slope"] = float(cal.coef_[0][0])
        except Exception:
            out["cal_slope"] = np.nan
        ybar, pbar = y.mean(), p.mean()
        out["cal_intercept"] = float(np.log(ybar / (1 - ybar)) - np.log(pbar / (1 - pbar))) if 0 < ybar < 1 else np.nan
        return out
    except Exception:
        return _nan_metrics()


def prepare_data(df):
    needed = NUMERIC_COLS + CATEGORICAL_COLS + HOSP_CHAR_COLS + [
        "actualhospitalmortality", "hospitalid", "Angular_Separation"]
    return df.dropna(subset=needed).copy()


def get_pairwise_das(full_dists, hid):
    rows = full_dists[
        (full_dists["hospital_1"] == hid) | (full_dists["hospital_2"] == hid)
    ]
    result = {}
    for _, r in rows.iterrows():
        other = r["hospital_2"] if r["hospital_1"] == hid else r["hospital_1"]
        result[int(other)] = r["angular_separation_deg"]
    return result


def fit_and_predict(clean, train_hids, holdout, features, target, use_hosp_chars=False):
    train = clean[clean["hospitalid"].isin(train_hids)]
    if len(train) < 50 or train[target].nunique() < 2:
        return {**_nan_metrics(), "n_train": len(train)}
    try:
        pipe = build_pipeline(use_hosp_chars=use_hosp_chars)
        pipe.fit(train[features], train[target])
        met = evaluate(holdout[target].values,
                       pipe.predict_proba(holdout[features])[:, 1])
        met["n_train"] = len(train)
        return met
    except Exception:
        return {**_nan_metrics(), "n_train": len(train)}


def summarize_comparisons(res, size_label, tbl_dir):
    """Print and return summary for one size group."""
    print(f"\n    --- {size_label} hospitals (n={len(res)}) ---")
    rows = []
    for k in [1, 5, 10, 20]:
        v = res.dropna(subset=[f"near{k}_AUC", f"far{k}_AUC"])
        if len(v) < 10:
            print(f"    k={k}: too few ({len(v)})")
            continue

        print(f"\n    k={k} (n={len(v)}):")
        for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True), ("cal_slope", None), ("cal_intercept", None)]:
            if higher_better is None:
                # calibration: closer to the ideal (slope 1, intercept 0) is better
                ideal = 1.0 if metric == "cal_slope" else 0.0
                vv = v.dropna(subset=[f"near{k}_{metric}", f"far{k}_{metric}"])
                if len(vv) < 10:
                    continue
                near_e = (vv[f"near{k}_{metric}"] - ideal).abs(); far_e = (vv[f"far{k}_{metric}"] - ideal).abs()
                d = far_e - near_e
                try:
                    _, pc = wilcoxon(d[d != 0])
                except Exception:
                    pc = np.nan
                rows.append({"size_group": size_label, "k": k, "metric": metric,
                             "mean_nearest": round(vv[f"near{k}_{metric}"].mean(), 4), "mean_farthest": round(vv[f"far{k}_{metric}"].mean(), 4),
                             "mean_all": round(vv[f"all_{metric}"].mean(), 4) if f"all_{metric}" in vv else np.nan,
                             "mean_random": round(vv[f"rand{k}_{metric}"].mean(), 4) if f"rand{k}_{metric}" in vv else np.nan,
                             "mean_delta": round(d.mean(), 5), "median_delta": round(d.median(), 5),
                             "nearest_wins": int((d > 0).sum()), "n": len(vv), "wilcoxon_p": round(pc, 4) if np.isfinite(pc) else np.nan})
                continue
            near = v[f"near{k}_{metric}"]
            far = v[f"far{k}_{metric}"]
            all_val = v[f"all_{metric}"]

            # Delta: positive = nearest better
            if higher_better:
                delta = near - far
                wins = (delta > 0).sum()
            else:
                delta = far - near
                wins = (delta > 0).sum()

            mean_delta = delta.mean()
            median_delta = delta.median()

            try:
                d = delta[delta != 0]
                _, p = wilcoxon(d)
            except Exception:
                p = np.nan

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"      {metric:6s}: near={near.mean():.4f}, far={far.mean():.4f}, "
                  f"all={all_val.mean():.4f} | wins={wins}/{len(v)} ({100*wins/len(v):.0f}%) "
                  f"p={p:.4f} {sig} | mean\u0394={mean_delta:+.4f} med\u0394={median_delta:+.4f}")

            extra = {}
            for tag, col in (("random", f"rand{k}_{metric}"), ("all", f"all_{metric}")):
                if col not in v.columns or v[col].isna().all():
                    continue
                vv = v.dropna(subset=[col])
                d2 = (vv[f"near{k}_{metric}"] - vv[col]) if higher_better else (vv[col] - vv[f"near{k}_{metric}"])
                try:
                    _, p2 = wilcoxon(d2[d2 != 0])
                except Exception:
                    p2 = np.nan
                extra[f"mean_{tag}"] = round(vv[col].mean(), 4)
                extra[f"delta_vs_{tag}"] = round(d2.mean(), 5)
                extra[f"wins_vs_{tag}"] = int((d2 > 0).sum())
                extra[f"n_vs_{tag}"] = len(vv)
                extra[f"p_vs_{tag}"] = round(p2, 4) if np.isfinite(p2) else np.nan
            rows.append({
                "size_group": size_label, "k": k, "metric": metric,
                "mean_nearest": round(near.mean(), 4),
                "mean_farthest": round(far.mean(), 4),
                "mean_all": round(all_val.mean(), 4),
                "mean_delta": round(mean_delta, 5),
                "median_delta": round(median_delta, 5),
                "nearest_wins": wins, "n": len(v),
                "wilcoxon_p": round(p, 4),
                "n_train_nearest": float(v[f"near{k}_n_train"].median()),
                "n_train_farthest": float(v[f"far{k}_n_train"].median()),
                "n_train_random": float(v[f"rand{k}_n_train"].median()) if f"rand{k}_n_train" in v.columns else np.nan,
                "n_train_all": float(v["all_n_train"].median()),
                **extra,
            })

    # Quartile
    vq = res.dropna(subset=["quartile_AUC", "all_AUC"])
    if len(vq) >= 10:
        print(f"\n    Quartile-matched vs All (n={len(vq)}):")
        for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True)]:
            q_val = vq[f"quartile_{metric}"]
            a_val = vq[f"all_{metric}"]
            if higher_better:
                wins = (q_val > a_val).sum()
            else:
                wins = (q_val < a_val).sum()
            try:
                diff = q_val - a_val
                diff = diff[diff != 0]
                _, p = wilcoxon(diff)
            except Exception:
                p = np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"      {metric:6s}: quartile={q_val.mean():.4f}, all={a_val.mean():.4f} "
                  f"| wins={wins}/{len(vq)} p={p:.4f} {sig}")

    return pd.DataFrame(rows)


def run_experiment(df, condition, dist_csv, fig_dir, tbl_dir):
    print(f"\n  DAS-guided training selection ({condition}):")

    clean = prepare_data(df)
    features_patient = NUMERIC_COLS + CATEGORICAL_COLS
    features_full = NUMERIC_COLS + CATEGORICAL_COLS + HOSP_CHAR_COLS
    target = "actualhospitalmortality"
    full_dists = pd.read_csv(dist_csv)

    hosp_counts = clean.groupby("hospitalid").size()
    large = hosp_counts[hosp_counts >= MIN_TRAIN].index.tolist()
    small = hosp_counts[(hosp_counts >= MIN_SMALL) & (hosp_counts < MIN_TRAIN)].index.tolist()
    print(f"    {len(large)} large hospitals (>= {MIN_TRAIN}, training pool + LOHO)")
    print(f"    {len(small)} small hospitals ({MIN_SMALL}-{MIN_TRAIN-1}, external eval)")

    # DAS quartiles
    hosp_das = df.groupby("hospitalid")["Angular_Separation"].first().reset_index()
    hosp_das["DAS_quartile"] = pd.qcut(
        hosp_das["Angular_Separation"], q=4,
        labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    qmap = hosp_das.set_index("hospitalid")["DAS_quartile"].to_dict()

    # Build holdout list: large (LOHO) + small (external)
    holdout_list = [(h, "large") for h in large] + [(h, "small") for h in small]

    results = []
    n_total = len(holdout_list)
    for i, (hid, size_grp) in enumerate(holdout_list):
        holdout = clean[clean["hospitalid"] == hid]
        if holdout[target].nunique() < 2:
            continue

        das_to_others = get_pairwise_das(full_dists, hid)

        # Training pool: large hospitals only (exclude self if large)
        pool = [h for h in large if h != hid and h in das_to_others]
        if len(pool) < 20:
            continue

        sorted_pool = sorted(pool, key=lambda h: das_to_others[h])

        # Baseline: all large hospitals (patient-only)
        all_met = fit_and_predict(clean, pool, holdout, features_patient, target)

        # Quartile-matched (patient-only)
        my_q = qmap.get(hid)
        q_pool = [h for h in pool if qmap.get(h) == my_q]
        q_met = fit_and_predict(clean, q_pool, holdout, features_patient, target) \
            if len(q_pool) >= 5 else {"AUC": np.nan, "Brier": np.nan, "AUPRC": np.nan, "n_train": 0}

        row = {
            "hospitalid": hid,
            "size_group": size_grp,
            "n_holdout": len(holdout),
            "mortality_rate": holdout[target].mean(),
            "DAS_from_ref": holdout["Angular_Separation"].iloc[0],
            "DAS_quartile": my_q,
            "all_AUC": all_met["AUC"], "all_Brier": all_met["Brier"],
            "all_AUPRC": all_met["AUPRC"], "all_n_train": all_met["n_train"],
            "all_cal_slope": all_met.get("cal_slope", np.nan), "all_cal_intercept": all_met.get("cal_intercept", np.nan),
            "quartile_AUC": q_met["AUC"], "quartile_Brier": q_met["Brier"],
            "quartile_AUPRC": q_met.get("AUPRC"), "quartile_n_train": q_met.get("n_train", 0),
        }

        for k in [1, 5, 10, 20]:
            # Patient-only pooled
            near_met = fit_and_predict(clean, sorted_pool[:k], holdout, features_patient, target)
            far_met = fit_and_predict(clean, sorted_pool[-k:], holdout, features_patient, target)
            for prefix, met in [("near", near_met), ("far", far_met)]:
                for m in METRICS:
                    row[f"{prefix}{k}_{m}"] = met.get(m, np.nan)
                row[f"{prefix}{k}_n_train"] = met["n_train"]

            # Random-k baseline: N_RANDOM draws of k hospitals from the same pool,
            # metrics averaged over draws (seeded per held-out hospital)
            if k >= 5 and len(pool) > k:
                rng = np.random.default_rng(int(hid))
                rmets = [fit_and_predict(clean, list(rng.choice(pool, size=k, replace=False)), holdout,
                                         features_patient, target) for _ in range(N_RANDOM)]
                for m in METRICS:
                    row[f"rand{k}_{m}"] = np.nanmean([x.get(m, np.nan) for x in rmets])
                row[f"rand{k}_n_train"] = float(np.median([x["n_train"] for x in rmets]))

            # CV-tuned deployment for k=1: train with grid search, then predict
            if k == 1:
                for prefix, hids in [("near", sorted_pool[:1]), ("far", sorted_pool[-1:])]:
                    train_data = clean[clean["hospitalid"].isin(hids)]
                    if len(train_data) >= 50 and train_data[target].nunique() >= 2:
                        try:
                            from sklearn.model_selection import GridSearchCV
                            pipe = build_pipeline(use_hosp_chars=False)
                            n_events = train_data[target].sum()
                            if n_events >= 15 and len(train_data) >= 150:
                                grid = GridSearchCV(
                                    pipe,
                                    param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
                                    cv=3, scoring="roc_auc", n_jobs=1, refit=True)
                                grid.fit(train_data[features_patient], train_data[target])
                                best_model = grid.best_estimator_
                            else:
                                pipe.set_params(clf__C=0.1)
                                pipe.fit(train_data[features_patient], train_data[target])
                                best_model = pipe
                            cv_met = evaluate(
                                holdout[target].values,
                                best_model.predict_proba(holdout[features_patient])[:, 1])
                            row[f"{prefix}1_cv_AUC"] = cv_met["AUC"]
                            row[f"{prefix}1_cv_Brier"] = cv_met["Brier"]
                            row[f"{prefix}1_cv_AUPRC"] = cv_met["AUPRC"]
                        except Exception:
                            row[f"{prefix}1_cv_AUC"] = np.nan
                            row[f"{prefix}1_cv_Brier"] = np.nan
                            row[f"{prefix}1_cv_AUPRC"] = np.nan
                    else:
                        row[f"{prefix}1_cv_AUC"] = np.nan
                        row[f"{prefix}1_cv_Brier"] = np.nan
                        row[f"{prefix}1_cv_AUPRC"] = np.nan

            # Hospital-characteristics-adjusted pooled (only for k >= 5)
            if k >= 5:
                near_adj = fit_and_predict(clean, sorted_pool[:k], holdout,
                                           features_full, target, use_hosp_chars=True)
                far_adj = fit_and_predict(clean, sorted_pool[-k:], holdout,
                                          features_full, target, use_hosp_chars=True)
                for prefix, met in [("near", near_adj), ("far", far_adj)]:
                    row[f"{prefix}{k}_adj_AUC"] = met["AUC"]
                    row[f"{prefix}{k}_adj_Brier"] = met["Brier"]
                    row[f"{prefix}{k}_adj_AUPRC"] = met["AUPRC"]

        # Size-matched farthest for k=1:
        # Find the farthest hospital with similar patient count (±50%) to nearest-1
        nearest_hid = sorted_pool[0]
        nearest_size = hosp_counts.get(nearest_hid, 0)
        lo, hi = nearest_size * 0.5, nearest_size * 1.5
        # Search from farthest end for a size-matched hospital
        far_matched_hid = None
        for h in reversed(sorted_pool):
            h_size = hosp_counts.get(h, 0)
            if lo <= h_size <= hi and h != nearest_hid:
                far_matched_hid = h
                break
        if far_matched_hid is not None:
            far_matched_met = fit_and_predict(clean, [far_matched_hid], holdout, features_patient, target)
            row["far1_matched_AUC"] = far_matched_met["AUC"]
            row["far1_matched_Brier"] = far_matched_met["Brier"]
            row["far1_matched_AUPRC"] = far_matched_met["AUPRC"]
            row["far1_matched_n_train"] = far_matched_met["n_train"]
            row["far1_matched_DAS"] = das_to_others[far_matched_hid]
        else:
            row["far1_matched_AUC"] = np.nan
            row["far1_matched_Brier"] = np.nan
            row["far1_matched_AUPRC"] = np.nan
            row["far1_matched_n_train"] = np.nan
            row["far1_matched_DAS"] = np.nan

        results.append(row)

        if (i + 1) % 30 == 0:
            print(f"    Processed {i+1}/{n_total}...")

    if not results:
        print("    No results")
        return

    res = pd.DataFrame(results)
    res.to_csv(tbl_dir / "transportability_results.csv", index=False)
    print(f"    Completed {len(res)} hospitals ({(res['size_group']=='large').sum()} large, "
          f"{(res['size_group']=='small').sum()} small)")

    # ── Summaries by size group ──
    all_summaries = []
    has_small = (res["size_group"] == "small").sum() > 0
    for grp in ["large", "small"]:
        sub = res[res["size_group"] == grp]
        if len(sub) >= 10:
            s = summarize_comparisons(sub, grp, tbl_dir)
            all_summaries.append(s)

    # Combined only if both groups present
    if has_small:
        s = summarize_comparisons(res, "all", tbl_dir)
        all_summaries.append(s)

    if all_summaries:
        pd.concat(all_summaries).to_csv(tbl_dir / "transportability_summary.csv", index=False)
        print(f"\n    Wrote {tbl_dir / 'transportability_summary.csv'}")

    # ── Size-matched k=1 sensitivity ──
    v_matched = res.dropna(subset=["near1_AUC", "far1_matched_AUC"])
    if len(v_matched) >= 10:
        print(f"\n    === Size-matched k=1 (n={len(v_matched)}) ===")
        print(f"    Nearest-1 train size: median={v_matched['near1_n_train'].median():.0f}")
        print(f"    Farthest-1 (matched) train size: median={v_matched['far1_matched_n_train'].median():.0f}")
        print(f"    Farthest-1 (matched) DAS: median={v_matched['far1_matched_DAS'].median():.1f}")
        for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True)]:
            near = v_matched[f"near1_{metric}"]
            far = v_matched[f"far1_matched_{metric}"]
            wins = (near > far).sum() if higher_better else (near < far).sum()
            try:
                diff = near - far
                diff = diff[diff != 0]
                _, p = wilcoxon(diff)
            except Exception:
                p = np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"      {metric:6s}: near={near.mean():.4f}, far_matched={far.mean():.4f} "
                  f"| wins={wins}/{len(v_matched)} ({100*wins/len(v_matched):.0f}%) p={p:.4f} {sig}")

    # ── CV-tuned k=1 deployment results ──
    cv_cols = [c for c in res.columns if "1_cv_" in c]
    if cv_cols:
        v_cv = res.dropna(subset=["near1_cv_AUC", "far1_cv_AUC"])
        if len(v_cv) >= 10:
            print(f"\n    === CV-tuned k=1 deployment (n={len(v_cv)}) ===")
            for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True)]:
                near = v_cv[f"near1_cv_{metric}"]
                far = v_cv[f"far1_cv_{metric}"]
                if higher_better:
                    delta = near - far
                    wins = (delta > 0).sum()
                else:
                    delta = far - near
                    wins = (delta > 0).sum()
                try:
                    d = delta[delta != 0]
                    _, p = wilcoxon(d)
                except Exception:
                    p = np.nan
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"      {metric:6s}: near={near.mean():.4f}, far={far.mean():.4f} "
                      f"| wins={wins}/{len(v_cv)} ({100*wins/len(v_cv):.0f}%) p={p:.4f} {sig} "
                      f"| mean\u0394={delta.mean():+.4f}")

    # ── Hospital-characteristics-adjusted results ──
    adj_cols_check = [c for c in res.columns if "_adj_" in c]
    if adj_cols_check:
        print(f"\n    === Hospital-characteristics-adjusted pooled models ===")
        for k in [5, 10, 20]:
            near_col = f"near{k}_adj_AUC"
            far_col = f"far{k}_adj_AUC"
            if near_col not in res.columns:
                continue
            v = res.dropna(subset=[near_col, far_col])
            if len(v) < 10:
                continue

            n = len(v)
            print(f"\n      k={k} (adjusted, n={n}):")
            for metric, higher_better in [("AUC", True), ("Brier", False), ("AUPRC", True)]:
                near_vals = v[f"near{k}_adj_{metric}"]
                far_vals = v[f"far{k}_adj_{metric}"]
                # Also get unadjusted for comparison
                near_unadj = v[f"near{k}_{metric}"]
                far_unadj = v[f"far{k}_{metric}"]

                if higher_better:
                    wins_adj = (near_vals > far_vals).sum()
                    wins_unadj = (near_unadj > far_unadj).sum()
                    delta_adj = near_vals - far_vals
                    delta_unadj = near_unadj - far_unadj
                else:
                    wins_adj = (near_vals < far_vals).sum()
                    wins_unadj = (near_unadj < far_unadj).sum()
                    delta_adj = far_vals - near_vals
                    delta_unadj = far_unadj - near_unadj

                try:
                    diff = delta_adj[delta_adj != 0]
                    _, p = wilcoxon(diff)
                except Exception:
                    p = np.nan
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                print(f"        {metric:6s}: adj wins={wins_adj}/{n} ({100*wins_adj/n:.0f}%) p={p:.4f} {sig} "
                      f"mean\u0394={delta_adj.mean():+.4f}"
                      f" | unadj wins={wins_unadj}/{n} ({100*wins_unadj/n:.0f}%) "
                      f"mean\u0394={delta_unadj.mean():+.4f}")

    # ── Figure: paired-difference plots for each k ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col_idx, k in enumerate([1, 5, 10, 20]):
        v = res.dropna(subset=[f"near{k}_AUC", f"far{k}_AUC",
                                f"near{k}_Brier", f"far{k}_Brier"])
        if len(v) < 10:
            continue

        # Panel top: ΔAUC = nearest - farthest (positive = nearest better)
        ax = axes[0, col_idx]
        delta_auc = v[f"near{k}_AUC"] - v[f"far{k}_AUC"]
        ax.scatter(range(len(delta_auc)), delta_auc.sort_values().values,
                   alpha=0.5, s=12, color=["#2c7bb6" if d > 0 else "#d7191c"
                                            for d in delta_auc.sort_values().values])
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(delta_auc.mean(), color="#2c7bb6", ls="--", linewidth=2,
                   label=f"Mean Δ = {delta_auc.mean():+.4f}")

        wins = (delta_auc > 0).sum()
        try:
            d = delta_auc[delta_auc != 0]
            _, p = wilcoxon(d)
        except Exception:
            p = np.nan
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        ax.set_xlabel("Hospitals (sorted by Δ)")
        ax.set_ylabel("ΔAUC (nearest − farthest)")
        ax.set_title(f"k={k}: ΔAUC | wins {wins}/{len(v)} ({100*wins/len(v):.0f}%) p={p:.4f} {sig}")
        ax.legend(fontsize=8, loc="upper left")

        # Panel bottom: ΔBrier = nearest - farthest (negative = nearest better)
        ax = axes[1, col_idx]
        delta_brier = v[f"near{k}_Brier"] - v[f"far{k}_Brier"]
        ax.scatter(range(len(delta_brier)), delta_brier.sort_values().values,
                   alpha=0.5, s=12, color=["#2c7bb6" if d < 0 else "#d7191c"
                                            for d in delta_brier.sort_values().values])
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(delta_brier.mean(), color="#2c7bb6", ls="--", linewidth=2,
                   label=f"Mean Δ = {delta_brier.mean():+.4f}")

        wins_b = (delta_brier < 0).sum()
        try:
            d = delta_brier[delta_brier != 0]
            _, p_b = wilcoxon(d)
        except Exception:
            p_b = np.nan
        sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else "n.s."

        ax.set_xlabel("Hospitals (sorted by Δ)")
        ax.set_ylabel("ΔBrier (nearest − farthest)")
        ax.set_title(f"k={k}: ΔBrier | wins {wins_b}/{len(v)} ({100*wins_b/len(v):.0f}%) p={p_b:.4f} {sig_b}")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{condition.capitalize()}: Documentation-similar vs dissimilar training",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "transportability_nearest_vs_farthest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Wrote {fig_dir / 'transportability_nearest_vs_farthest.png'}")


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

    conditions = args.conditions or sorted(set(
        p.name.replace("dist_", "").split("_min")[0]
        for p in processed.glob("dist_*.npz")
    )) or ["diabetes"]

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
        print(f"TRANSPORTABILITY: {condition.upper()}")
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
