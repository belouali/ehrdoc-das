#!/usr/bin/env python
"""Reference-hospital sensitivity for the scalar DAS covariate.

Pairwise DAS needs no reference, but the regression covariate, the quartiles
and the capture-rate correlations use one number per hospital: DAS from the
reference hospital 73. This script repeats those analyses with three
alternative scalars:

  ref73     : DAS from hospital 73 (primary, as in the paper)
  centroid  : angle between the hospital's vector and the centroid (mean) of
              all hospitals' normalized vectors
  mean_all  : mean DAS to every other hospital
  ref_alt   : DAS from the second-largest hospital by condition patients

Outputs (artifacts/sensitivity_reference/):
  scalar_correlations.csv      Spearman correlations among the four scalars
  regression_reference.csv     coefficient rows for M3 / M13 / M8 under each scalar
  lrt_reference.csv            LRTs (M1->M3, M9->M13) under each scalar
  variance_reference.csv       random-intercept variance explained under each scalar
  capture_corr_reference.csv   capture rate vs scalar (Spearman) under each scalar

Usage:
  python scripts/21_reference_sensitivity.py --conditions diabetes hypertension chf af
"""
from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from ehrdoc.data.cohort import load_base_cohort, analytic_sample
from ehrdoc.distances.metrics import angular_separation
from ehrdoc.utils.io import ensure_dir

SCRIPTS = Path(__file__).resolve().parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hospital_scalars(processed: Path, condition: str, ref: int) -> pd.DataFrame:
    vec = sorted(processed.glob(f"vectors_{condition}_min*.parquet"), key=lambda p: p.stat().st_mtime)[-1]
    cnt = sorted(processed.glob(f"counts_{condition}_min*.parquet"), key=lambda p: p.stat().st_mtime)[-1]
    npz = sorted(processed.glob(f"dist_{condition}_min*.npz"), key=lambda p: p.stat().st_mtime)[-1]
    vectors = pd.read_parquet(vec)
    counts = pd.read_parquet(cnt)
    z = np.load(npz, allow_pickle=True)
    D, ids = z["D"], [int(i) for i in z["ids"]]
    idx = {h: i for i, h in enumerate(ids)}

    lobes = [c for c in vectors.columns if c != "hospitalid"]
    X = vectors[lobes].to_numpy(dtype=float)
    centroid = X.mean(axis=0)
    hid = vectors["hospitalid"].astype(int).tolist()

    totals = counts.set_index("hospitalid")[[c for c in counts.columns if c != "hospitalid"]].sum(axis=1)
    ref_alt = int(totals.drop(index=ref, errors="ignore").idxmax())

    rows = []
    for h, v in zip(hid, X):
        i = idx[h]
        rows.append({"hospitalid": h,
                     "ref73": float(D[i, idx[ref]]) if ref in idx else np.nan,
                     "centroid": angular_separation(v, centroid),
                     "mean_all": float(np.delete(D[i], i).mean()),
                     "ref_alt": float(D[i, idx[ref_alt]])})
    out = pd.DataFrame(rows)
    out.attrs["ref_alt"] = ref_alt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes", "hypertension", "chf", "af"])
    ap.add_argument("--ref-hospital", type=int, default=73)
    ap.add_argument("--skip-ri", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = yaml.safe_load(open(args.paths))
    processed = Path(cfg["outputs"]["processed"])
    art_root = Path(cfg["outputs"]["artifacts"])
    out_dir = ensure_dir(art_root / "sensitivity_reference")
    e = cfg["eicu"]; e.setdefault("apachePatientResult", "data/raw/apachePatientResult.csv.gz")
    SCALARS = ["ref73", "centroid", "mean_all", "ref_alt"]

    s08 = _load_script("08_regression_models.py")
    cohort = load_base_cohort(e)

    corr_rows, reg_rows, lrt_rows, var_rows, cap_rows = [], [], [], [], []
    for cond in args.conditions:
        sc = hospital_scalars(processed, cond, args.ref_hospital)
        ref_alt = sc.attrs["ref_alt"]
        print(f"\n=== {cond}: {len(sc)} hospitals; alternative reference = hospital {ref_alt}")

        # ── correlations among scalars ──
        for i, a in enumerate(SCALARS):
            for b in SCALARS[i + 1:]:
                rho, p = spearmanr(sc[a], sc[b])
                corr_rows.append({"condition": cond, "scalar_a": a, "scalar_b": b,
                                  "spearman_rho": round(rho, 3), "p": p, "n": len(sc)})
                print(f"  {a:9s} vs {b:9s}: rho={rho:.3f}")

        # ── capture-rate correlations ──
        frag = art_root / cond / "tables" / "phenotype_fragility.csv"
        if frag.exists():
            fr = pd.read_csv(frag).drop_duplicates("hospitalid").merge(sc, on="hospitalid")
            for s in SCALARS:
                for col in ["capture_Dx_only", "capture_PH_only", "capture_Rx_only"]:
                    rho, p = spearmanr(fr[s], fr[col])
                    cap_rows.append({"condition": cond, "scalar": s, "phenotype": col.replace("capture_", ""),
                                     "spearman_rho": round(rho, 3), "p": p, "n": len(fr)})

        # ── regression ──
        merged = cohort.merge(sc, on="hospitalid", how="inner")
        merged, info = analytic_sample(merged, drop_no_variation=True)
        print(f"  analytic sample: {info['n_analytic']:,} stays, {info['hospitals_analytic']} hospitals")

        base_filter = ["M1:", "M9:"] + ([] if args.skip_ri else ["M6:"])
        merged["Angular_Separation"] = merged["ref73"]
        merged["Angular_Separation_Quartile"] = pd.qcut(merged["Angular_Separation"], 4,
                                                        labels=["q1", "q2", "q3", "q4"], duplicates="drop")
        coef_base, fitted_base = s08.run_all_models(merged, merged, None, include_gee=False,
                                                    include_ri=not args.skip_ri, model_filter=base_filter)
        base_var = fitted_base.get("M6: + Hospital RI", (None, {}))[1].get("re_var", np.nan)
        for r in coef_base.to_dict("records"):
            reg_rows.append({"condition": cond, "scalar": "(none)", **r})

        for s in SCALARS:
            merged["Angular_Separation"] = merged[s]
            filt = ["M3:", "M13:"] + ([] if args.skip_ri else ["M8:"])
            coef, fitted = s08.run_all_models(merged, merged, None, include_gee=False,
                                              include_ri=not args.skip_ri, model_filter=filt)
            for r in coef.to_dict("records"):
                reg_rows.append({"condition": cond, "scalar": s, **r})
            allfit = {**fitted_base, **fitted}
            lrt = s08.compute_lrt(allfit)
            for r in lrt.to_dict("records"):
                lrt_rows.append({"condition": cond, "scalar": s, **r})
            ri = fitted.get("M8: + AS continuous + Hospital RI", (None, {}))[1]
            if "re_var" in ri:
                var_rows.append({"condition": cond, "scalar": s, "re_var_without_DAS": base_var,
                                 "re_var_with_DAS": ri["re_var"],
                                 "pct_between_hospital_var_explained": round(100 * (1 - ri["re_var"] / base_var), 1)})
            m3 = fitted.get("M3: + AS continuous", (None, {}))[1]
            m13 = fitted.get("M13: + Hosp chars + AS continuous", (None, {}))[1]
            print(f"  [{s:9s}] M3 AIC={m3.get('AIC', float('nan')):.1f}  M13 AIC={m13.get('AIC', float('nan')):.1f}"
                  + (f"  RI var explained={var_rows[-1]['pct_between_hospital_var_explained']}%" if var_rows and var_rows[-1]['scalar'] == s and var_rows[-1]['condition'] == cond else ""))

    pd.DataFrame(corr_rows).to_csv(out_dir / "scalar_correlations.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(out_dir / "regression_reference.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(out_dir / "lrt_reference.csv", index=False)
    pd.DataFrame(var_rows).to_csv(out_dir / "variance_reference.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(out_dir / "capture_corr_reference.csv", index=False)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
