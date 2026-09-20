#!/usr/bin/env python
"""Sensitivity analysis: exclude hospitals with an empty evidence source.

A block of eICU hospitals contributes no medication records at all. For every
condition those hospitals sit at ~88-90 degrees from every hospital that does,
which (a) inflates cross-condition concordance and (b) makes "documentation-
dissimilar" partly synonymous with "no medication table". This script re-runs
the three core analyses with those hospitals removed:

  1. Cross-condition concordance (Mantel test)       -> concordance_zero_source.csv
  2. Regression model set (M1-M4, M9, M13, RI models) -> regression_zero_source.csv,
                                                          lrt_zero_source.csv,
                                                          variance_zero_source.csv
  3. Leave-one-hospital-out nearest-vs-farthest       -> transportability_zero_source_<cond>.csv,
     (pooled logistic regression, k nearest/farthest)    transportability_zero_source_summary.csv

Prereqs: scripts 01-03 (flags, vectors, distances); 08/09 for the main-analysis
comparison columns.

Usage:
  python scripts/17_zero_source_sensitivity.py
  python scripts/17_zero_source_sensitivity.py --conditions diabetes --skip-loho
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from ehrdoc.data.cohort import load_base_cohort, attach_das, analytic_sample
from ehrdoc.distances.concordance import distance_matrix_concordance
from ehrdoc.distances.matrices import distance_matrix
from ehrdoc.utils.io import ensure_dir
from ehrdoc.venn.vectors import hospital_venn_counts, hospital_venn_vectors, zero_source_hospitals

SCRIPTS = Path(__file__).resolve().parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dist_csv(processed: Path, condition: str) -> Path:
    cand = sorted(processed.glob(f"dist_{condition}_min*.csv"), key=lambda p: p.stat().st_mtime)
    if not cand:
        raise FileNotFoundError(f"No dist_{condition}_min*.csv in {processed}")
    return cand[-1]


def loho_nearest_vs_farthest(clean: pd.DataFrame, D: np.ndarray, ids: list[int],
                             ks: list[int], s09, min_train: int = 100) -> pd.DataFrame:
    """Lean LOHO: pooled logistic regression on k nearest vs k farthest hospitals."""
    features = s09.NUMERIC_COLS + s09.CATEGORICAL_COLS
    target = "actualhospitalmortality"
    idx = {h: i for i, h in enumerate(ids)}
    counts = clean.groupby("hospitalid").size()
    large = [h for h in ids if counts.get(h, 0) >= min_train]
    rows = []
    for n_done, hid in enumerate(large):
        holdout = clean[clean["hospitalid"] == hid]
        if holdout[target].nunique() < 2:
            continue
        pool = [h for h in large if h != hid]
        if len(pool) < max(ks):
            continue
        pool = sorted(pool, key=lambda h: D[idx[hid], idx[h]])
        row = {"hospitalid": hid, "n_holdout": len(holdout),
               "mortality_rate": holdout[target].mean()}
        allm = s09.fit_and_predict(clean, pool, holdout, features, target)
        for m in s09.METRICS:
            row[f"all_{m}"] = allm.get(m, np.nan)
        row["all_n_train"] = allm["n_train"]
        rng = np.random.default_rng(int(hid))
        for k in ks:
            near = s09.fit_and_predict(clean, pool[:k], holdout, features, target)
            far = s09.fit_and_predict(clean, pool[-k:], holdout, features, target)
            for prefix, met in (("near", near), ("far", far)):
                for m in s09.METRICS:
                    row[f"{prefix}{k}_{m}"] = met.get(m, np.nan)
                row[f"{prefix}{k}_n_train"] = met["n_train"]
            rmets = [s09.fit_and_predict(clean, list(rng.choice(pool, size=k, replace=False)), holdout, features, target)
                     for _ in range(s09.N_RANDOM)]
            for m in s09.METRICS:
                row[f"rand{k}_{m}"] = np.nanmean([x.get(m, np.nan) for x in rmets])
            row[f"rand{k}_n_train"] = float(np.median([x["n_train"] for x in rmets]))
        rows.append(row)
        if (n_done + 1) % 25 == 0:
            print(f"      {n_done + 1}/{len(large)} hospitals")
    return pd.DataFrame(rows)


def summarize_loho(res: pd.DataFrame, ks: list[int], condition: str) -> pd.DataFrame:
    rows = []
    for k in ks:
        v = res.dropna(subset=[f"near{k}_AUC", f"far{k}_AUC"])
        if len(v) < 10:
            continue
        for metric, higher_better in (("AUC", True), ("Brier", False), ("AUPRC", True)):
            near, far = v[f"near{k}_{metric}"], v[f"far{k}_{metric}"]
            delta = (near - far) if higher_better else (far - near)
            wins = int((delta > 0).sum())
            try:
                _, p = wilcoxon(delta[delta != 0])
            except Exception:
                p = np.nan
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
                extra.update({f"mean_{tag}": round(vv[col].mean(), 4), f"delta_vs_{tag}": round(d2.mean(), 5),
                              f"wins_vs_{tag}": int((d2 > 0).sum()), f"n_vs_{tag}": len(vv),
                              f"p_vs_{tag}": round(p2, 4) if np.isfinite(p2) else np.nan})
            rows.append({"condition": condition, "sample": "excluding_zero_source", "k": k, "metric": metric,
                         "mean_nearest": round(near.mean(), 4), "mean_farthest": round(far.mean(), 4),
                         "mean_delta": round(delta.mean(), 5), "median_delta": round(delta.median(), 5),
                         "nearest_wins": wins, "n": len(v), "win_pct": round(100 * wins / len(v), 1),
                         "wilcoxon_p": round(p, 4) if np.isfinite(p) else np.nan,
                         "n_train_nearest": float(v[f"near{k}_n_train"].median()), "n_train_farthest": float(v[f"far{k}_n_train"].median()),
                         "n_train_random": float(v[f"rand{k}_n_train"].median()) if f"rand{k}_n_train" in v else np.nan,
                         "n_train_all": float(v["all_n_train"].median()) if "all_n_train" in v else np.nan, **extra})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--cohorts", default="configs/cohorts.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes", "hypertension", "chf", "af"])
    ap.add_argument("--ref-hospital", type=int, default=73)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--skip-loho", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    ap.add_argument("--skip-ri", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    with open(args.paths) as f:
        cfg = yaml.safe_load(f)
    with open(args.cohorts) as f:
        thresholds = (yaml.safe_load(f) or {}).get("phenotype_thresholds", {})
    processed = Path(cfg["outputs"]["processed"])
    art_root = Path(cfg["outputs"]["artifacts"])
    out_dir = ensure_dir(art_root / "sensitivity_zero_source")
    e = cfg["eicu"]
    e.setdefault("apachePatientResult", "data/raw/apachePatientResult.csv.gz")

    # ── 1. Identify zero-source hospitals and rebuild distance matrices ──
    mats, excluded_by_cond, zs_rows = {}, {}, []
    for cond in args.conditions:
        flags = pd.read_parquet(processed / f"flags_{cond}.parquet")
        k = int(thresholds.get(cond, 100) or 100)
        counts = hospital_venn_counts(flags, min_patients_per_hospital=k)
        zs = zero_source_hospitals(counts)
        excluded = sorted(set(zs["Dx"]) | set(zs["Hx"]) | set(zs["Med"]))
        excluded_by_cond[cond] = excluded
        zs_rows.append({"condition": cond, "min_patients": k, "n_hospitals": len(counts),
                        "n_zero_diagnosis": len(zs["Dx"]), "n_zero_past_history": len(zs["Hx"]),
                        "n_zero_medication": len(zs["Med"]), "n_excluded": len(excluded),
                        "excluded_hospitals": " ".join(map(str, excluded))})
        D_all, ids_all = distance_matrix(hospital_venn_vectors(counts))
        keep = counts[~counts["hospitalid"].isin(excluded)]
        D_ex, ids_ex = distance_matrix(hospital_venn_vectors(keep))
        ids_all = [int(i) for i in ids_all]; ids_ex = [int(i) for i in ids_ex]
        mats[cond] = {"all": (D_all, ids_all), "excl": (D_ex, ids_ex)}
        np.savez_compressed(out_dir / f"dist_{cond}_nozero.npz", D=D_ex, ids=np.array(ids_ex))
        print(f"{cond}: {len(counts)} hospitals; zero-medication={len(zs['Med'])}, "
              f"zero-diagnosis={len(zs['Dx'])}, zero-history={len(zs['Hx'])} -> excluding {len(excluded)}")
    pd.DataFrame(zs_rows).to_csv(out_dir / "zero_source_hospitals.csv", index=False)

    # ── 1b. Do hospital descriptors explain DAS once zero-source hospitals are gone? ──
    from ehrdoc.distances.permanova import permanova
    perm_rows = []
    nice = {"region": "Region", "numbedscategory": "Bed category", "teachingstatus": "Teaching status"}
    for cond in args.conditions:
        flags = pd.read_parquet(processed / f"flags_{cond}.parquet")
        chars = flags.drop_duplicates("hospitalid").set_index("hospitalid")
        for variant in ("all", "excl"):
            D, ids = mats[cond][variant]
            for col, label in nice.items():
                if col not in chars.columns:
                    continue
                vals = [chars[col].get(h, np.nan) for h in ids]
                vals = [np.nan if (isinstance(v, float) and np.isnan(v)) or str(v) == "Unknown" else str(v) for v in vals]
                pv = permanova(D, vals, n_perm=args.n_perm, seed=0)
                perm_rows.append({"condition": cond, "sample": "all_hospitals" if variant == "all" else "excluding_zero_source",
                                  "characteristic": label, "n_hospitals": pv["n_hospitals"], "n_groups": pv["n_groups"],
                                  "permanova_R2": round(pv["R2"], 4), "permanova_p": pv["p_value"]})
    pd.DataFrame(perm_rows).to_csv(out_dir / "permanova_zero_source.csv", index=False)
    for r in perm_rows:
        print(f"  PERMANOVA {r['condition']:12s} {r['characteristic']:16s} [{r['sample']}]: R2={r['permanova_R2']:.3f} p={r['permanova_p']:.4f}")

    # ── 2. Concordance with and without ──
    conc = []
    for a, b in combinations(args.conditions, 2):
        for variant in ("all", "excl"):
            Da, ia = mats[a][variant]; Db, ib = mats[b][variant]
            out = distance_matrix_concordance(Da, ia, Db, ib, n_perm=args.n_perm)
            conc.append({"condition_1": a, "condition_2": b,
                         "sample": "all_hospitals" if variant == "all" else "excluding_zero_source",
                         "n_hospitals": out["n_hospitals"], "n_pairs": out["n_pairs"],
                         "spearman_rho": round(out["rho"], 4), "mantel_p": out["mantel_p"]})
            print(f"  {a} vs {b} [{variant}]: rho={out['rho']:.3f}, Mantel p={out['mantel_p']:.4f}, n={out['n_hospitals']}")
    pd.DataFrame(conc).to_csv(out_dir / "concordance_zero_source.csv", index=False)

    if args.skip_regression and args.skip_loho:
        print("Done (concordance only).")
        return

    print("\nBuilding regression cohort...")
    cohort = load_base_cohort(e)
    s08 = _load_script("08_regression_models.py") if not args.skip_regression else None
    s09 = _load_script("09_xgboost_prediction.py") if not args.skip_loho else None

    reg_rows, lrt_rows, var_rows, loho_summaries = [], [], [], []
    for cond in args.conditions:
        excluded = excluded_by_cond[cond]
        merged = attach_das(cohort, _dist_csv(processed, cond), args.ref_hospital)
        merged = merged[~merged["hospitalid"].isin(excluded)].copy()
        merged, info = analytic_sample(merged, drop_no_variation=True)
        merged["Angular_Separation_Quartile"] = pd.qcut(
            merged["Angular_Separation"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
        print(f"\n{cond}: analytic sample excluding zero-source hospitals: "
              f"{info['n_analytic']:,} stays, {info['hospitals_analytic']} hospitals")
        with open(out_dir / f"analytic_sample_info_{cond}.json", "w") as f:
            json.dump(info, f, indent=2)

        # ── Regression ──
        if s08 is not None:
            keep_models = ["M1:", "M2:", "M3:", "M4:", "M9:", "M13:"] + ([] if args.skip_ri else ["M6:", "M8:"])
            coef_df, fitted = s08.run_all_models(merged, merged, None, include_gee=False,
                                                 include_ri=not args.skip_ri, model_filter=keep_models)
            coef_df["condition"] = cond
            coef_df["sample"] = "excluding_zero_source"
            reg_rows.append(coef_df)
            lrt = s08.compute_lrt(fitted)
            if not lrt.empty:
                lrt["condition"] = cond
                lrt_rows.append(lrt)
            for label, (m, inf) in fitted.items():
                if "re_var" in inf:
                    var_rows.append({"condition": cond, "model": label, "re_sd": inf["re_sd"],
                                     "re_var": inf["re_var"], "icc_latent": inf["icc_latent"]})
            s08.write_model_comparison_table(coef_df, lrt, fitted, ensure_dir(out_dir / cond))

        # ── LOHO ──
        if s09 is not None:
            print(f"  LOHO nearest vs farthest (k={args.k})...")
            clean = s09.prepare_data(merged)
            D_ex, ids_ex = mats[cond]["excl"]
            res = loho_nearest_vs_farthest(clean, D_ex, ids_ex, args.k, s09)
            res.to_csv(out_dir / f"transportability_zero_source_{cond}.csv", index=False)
            summ = summarize_loho(res, args.k, cond)
            # attach the main-analysis (all hospitals) summary for side-by-side comparison
            main_path = art_root / cond / "tables" / "transportability_summary.csv"
            if main_path.exists():
                ms = pd.read_csv(main_path)
                ms = ms[(ms["size_group"] == "large") & (ms["k"].isin(args.k)) & (ms["metric"].isin(["AUC", "Brier", "AUPRC"]))].copy()
                ms["condition"] = cond; ms["sample"] = "all_hospitals"
                ms["win_pct"] = (100 * ms["nearest_wins"] / ms["n"]).round(1)
                summ = pd.concat([summ, ms[[c for c in summ.columns if c in ms.columns]]], ignore_index=True)
            loho_summaries.append(summ)
            for r in summ[summ["sample"] == "excluding_zero_source"].itertuples():
                print(f"    k={r.k} {r.metric}: near={r.mean_nearest:.4f} far={r.mean_farthest:.4f} "
                      f"wins={r.nearest_wins}/{r.n} ({r.win_pct:.0f}%) p={r.wilcoxon_p}")

    if reg_rows:
        pd.concat(reg_rows, ignore_index=True).to_csv(out_dir / "regression_zero_source.csv", index=False)
    if lrt_rows:
        pd.concat(lrt_rows, ignore_index=True).to_csv(out_dir / "lrt_zero_source.csv", index=False)
    if var_rows:
        vr = pd.DataFrame(var_rows)
        base = vr[vr["model"].str.startswith("M6:")].set_index("condition")["re_var"]
        vr["pct_between_hospital_var_explained_by_DAS"] = [
            round(100 * (1 - r.re_var / base[r.condition]), 1) if r.model.startswith("M8:") and r.condition in base else np.nan
            for r in vr.itertuples()]
        vr.to_csv(out_dir / "variance_zero_source.csv", index=False)
    if loho_summaries:
        pd.concat(loho_summaries, ignore_index=True).to_csv(
            out_dir / "transportability_zero_source_summary.csv", index=False)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
