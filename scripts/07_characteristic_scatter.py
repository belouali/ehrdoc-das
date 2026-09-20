#!/usr/bin/env python
"""Generate per-condition figure sets: scatter plots colored by hospital characteristics.

For each condition, produces scatter plots colored by region, bed size, and teaching
status, highlighting the same hospitals used in that condition's triptych.

Also runs Kruskal-Wallis tests per condition to check if AS correlates with observables.

Prereqs:
  - Run scripts 01-03 (flags, vectors, distances)
  - Flags parquet must include hospital characteristic columns (hospital.csv merged)

Outputs (per condition, in artifacts/figures/<condition>/):
  scatter_quartile.png
  scatter_by_region.png
  scatter_by_numbedscategory.png
  scatter_by_teachingstatus.png
  characteristic_tests.csv

Usage:
  python scripts/07_characteristic_scatter.py
  python scripts/07_characteristic_scatter.py --conditions diabetes hypertension chf af
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import kruskal

from ehrdoc.distances.permanova import permanova
from ehrdoc.utils.io import ensure_dir

N_PERM = 999


def _load_npz(path: Path) -> tuple[np.ndarray, list[int]]:
    z = np.load(path, allow_pickle=True)
    return z["D"], z["ids"].tolist()


def _get_hospital_chars(flags: pd.DataFrame, hospital_ids: list[int]) -> pd.DataFrame:
    """Extract one row per hospital with characteristics."""
    char_cols = ["hospitalid"]
    for col in ["numbedscategory", "teachingstatus", "region"]:
        if col in flags.columns:
            char_cols.append(col)

    hosp = flags[flags["hospitalid"].isin(hospital_ids)][char_cols].drop_duplicates("hospitalid")
    if "teachingstatus" in hosp.columns:
        hosp["teachingstatus"] = hosp["teachingstatus"].apply(
            lambda x: "Teaching" if str(x).lower() in ("t", "true", "1", "yes") else "Non-teaching"
        )
    return hosp.set_index("hospitalid")


def _hospital_has_all_three_sources(flags: pd.DataFrame, hid: int, min_per_source: int = 5) -> bool:
    pid = "uniquepid" if "uniquepid" in flags.columns else "patientunitstayid"
    sub = flags[flags["hospitalid"] == hid]
    n_med = sub[sub["exists_in_medication"].astype(bool)][pid].nunique()
    n_hx = sub[sub["exists_in_pastHistory"].astype(bool)][pid].nunique()
    n_dx = sub[sub["exists_in_diagnosis"].astype(bool)][pid].nunique()
    return n_med >= min_per_source and n_hx >= min_per_source and n_dx >= min_per_source


def _pick_triptych_hospitals(D, ids, flags, ref_hospital, min_per_source=5):
    """Pick ref + closest-different-chars + farthest-same-chars."""
    if ref_hospital not in ids:
        ref_hospital = ids[0]
    ref_idx = ids.index(ref_hospital)
    dists = D[ref_idx, :].copy()

    char_cols = ["region", "numbedscategory", "teachingstatus"]
    ref_sub = flags[flags["hospitalid"] == ref_hospital].head(1)
    ref_chars = {}
    for col in char_cols:
        if col in ref_sub.columns and not ref_sub[col].isna().all():
            ref_chars[col] = str(ref_sub[col].iloc[0])
    has_chars = len(ref_chars) == len(char_cols)

    # Closest with different characteristics
    order_asc = np.argsort(dists)
    closest_diff = None
    closest_any = None
    for idx in order_asc:
        hid = ids[idx]
        if hid == ref_hospital or not _hospital_has_all_three_sources(flags, hid, min_per_source):
            continue
        if closest_any is None:
            closest_any = hid
        if has_chars and closest_diff is None:
            hid_sub = flags[flags["hospitalid"] == hid].head(1)
            hid_chars = {}
            for c in char_cols:
                if c in hid_sub.columns and not hid_sub[c].isna().all():
                    hid_chars[c] = str(hid_sub[c].iloc[0])
            comparable = sum(1 for c in char_cols if c in ref_chars and c in hid_chars)
            matches = sum(1 for c in char_cols if c in ref_chars and c in hid_chars
                         and ref_chars[c] == hid_chars[c])
            if comparable >= 2 and matches <= 1:
                closest_diff = hid
                break
    closest_id = closest_diff or closest_any or ids[1]

    # Farthest with same characteristics
    order_desc = np.argsort(-dists)
    farthest_same = None
    farthest_any = None
    for idx in order_desc:
        hid = ids[idx]
        if hid == ref_hospital or not _hospital_has_all_three_sources(flags, hid, min_per_source):
            continue
        if farthest_any is None:
            farthest_any = hid
        if has_chars and farthest_same is None:
            hid_sub = flags[flags["hospitalid"] == hid].head(1)
            hid_chars = {}
            for c in char_cols:
                if c in hid_sub.columns and not hid_sub[c].isna().all():
                    hid_chars[c] = str(hid_sub[c].iloc[0])
            comparable = sum(1 for c in char_cols if c in ref_chars and c in hid_chars)
            matches = sum(1 for c in char_cols if c in ref_chars and c in hid_chars
                         and ref_chars[c] == hid_chars[c])
            if matches == 3 and comparable == 3:
                farthest_same = hid
                break
    farthest_id = farthest_same or farthest_any or ids[-1]

    return [ref_hospital, closest_id, farthest_id]


def _make_scatter(D, ids, highlight, ref_hospital, color_values, color_label,
                  condition, ax):
    """Generic scatter plot on the direct-distance axes."""
    ref_idx = ids.index(ref_hospital)
    x = D[ref_idx, :]
    y = D.mean(axis=1)

    df = pd.DataFrame({"hospitalid": ids, "x": x, "y": y, "color": color_values})
    df["color"] = df["color"].fillna("Unknown")

    categories = sorted(df["color"].unique())
    cmap = plt.get_cmap("tab10", max(len(categories), 1))
    colors = {cat: cmap(i) for i, cat in enumerate(categories)}

    for cat in categories:
        grp = df[df["color"] == cat]
        ax.scatter(grp["x"], grp["y"], label=cat, s=35, alpha=0.7,
                   color=colors[cat], edgecolors="k", linewidths=0.3)

    # Highlight triptych hospitals
    for hid in highlight:
        if hid in set(ids):
            row = df[df["hospitalid"] == hid].iloc[0]
            ax.scatter([row["x"]], [row["y"]], s=120, marker="x",
                       color="black", linewidths=2, zorder=10)

    # Smart label placement
    highlight_pts = [(hid, df[df["hospitalid"]==hid].iloc[0]["x"], df[df["hospitalid"]==hid].iloc[0]["y"])
                     for hid in highlight if hid in set(ids)]
    for i, (hid, hx, hy) in enumerate(highlight_pts):
        dx_off, dy_off = 10, -12
        for j, (_, hx2, hy2) in enumerate(highlight_pts[:i]):
            xr = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1)
            yr = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1)
            if abs(hx - hx2) / xr < 0.08 and abs(hy - hy2) / yr < 0.08:
                dx_off, dy_off = 20, 20
        ax.annotate(f"ID:{hid}", (hx, hy), fontsize=8, fontweight="bold",
                    xytext=(dx_off, dy_off), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8) if abs(dx_off) > 15 else None,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax.set_xlabel(f"Angular separation from Hospital {ref_hospital} (degrees)")
    ax.set_ylabel("Mean angular separation (degrees)")
    ax.legend(title=color_label, loc="upper left", fontsize=7, title_fontsize=8)


def process_condition(condition, processed, flags, ref_hospital, fig_dir, tbl_dir):
    """Generate all scatter plots for one condition."""
    print(f"\n  {'─'*50}")
    print(f"  {condition.upper()}")
    print(f"  {'─'*50}")

    # Load distance matrix
    cand = sorted(processed.glob(f"dist_{condition}*.npz"), key=lambda p: p.stat().st_mtime)
    if not cand:
        print(f"    No distance matrix found, skipping")
        return
    D, ids = _load_npz(cand[-1])
    print(f"    {len(ids)} hospitals in distance matrix")

    # Load triptych hospitals from script 05's saved selection (ensures consistency)
    import json
    triptych_json = tbl_dir / "triptych_hospitals.json"
    if triptych_json.exists():
        with open(triptych_json) as f:
            tri = json.load(f)
        highlight = [tri["ref"], tri["closest_diff"], tri["farthest_same"]]
        print(f"    Loaded triptych from {triptych_json}: {highlight}")
    else:
        highlight = _pick_triptych_hospitals(D, ids, flags, ref_hospital)
        print(f"    Computed triptych: ref={highlight[0]}, closest={highlight[1]}, farthest={highlight[2]}")

    # Hospital characteristics
    hosp_chars = _get_hospital_chars(flags, ids)
    char_cols = [c for c in ["region", "numbedscategory", "teachingstatus"] if c in hosp_chars.columns]

    if not char_cols:
        print(f"    WARNING: No hospital characteristics found in flags. Skipping characteristic plots.")
        print(f"    Available columns: {list(flags.columns)}")

    # Quartile colors
    mean_d = D.mean(axis=1)
    n_bins = min(4, len(ids))
    quartiles = pd.qcut(mean_d, n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates="drop")

    # ── Generate figures ──
    all_color_specs = [("Documentation similarity quartile", quartiles.astype(str), "scatter_quartile")]
    for col in char_cols:
        nice = {"region": "Region", "numbedscategory": "Bed category", "teachingstatus": "Teaching status"}[col]
        vals = [hosp_chars.loc[hid, col] if hid in hosp_chars.index else "Unknown" for hid in ids]
        all_color_specs.append((nice, vals, f"scatter_by_{col}"))

    for color_label, color_values, filename in all_color_specs:
        fig, ax = plt.subplots(figsize=(10, 7))
        _make_scatter(D, ids, highlight, ref_hospital, color_values, color_label, condition, ax)
        ax.set_title(f"{condition.capitalize()} — Hospital documentation landscape\n"
                     f"Colored by {color_label} (n={len(ids)} hospitals)", fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"{filename}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Wrote {out}")

    # ── Do observable hospital characteristics explain the distance structure? ──
    # Primary test: one-way PERMANOVA on the full DAS matrix (pseudo-F with a
    # hospital-label permutation null; R2 = share of squared distance explained).
    # The Kruskal-Wallis test on each hospital's mean DAS is kept as a secondary,
    # coarser summary.
    test_results = []
    mean_as = pd.DataFrame({"hospitalid": ids, "mean_angular_sep": D.mean(axis=1)})
    mean_as = mean_as.merge(hosp_chars.reset_index(), on="hospitalid", how="left")

    nice_names = {"region": "Region", "numbedscategory": "Bed category", "teachingstatus": "Teaching status"}
    factor_sets = [(col, nice_names[col]) for col in char_cols]
    if len(char_cols) >= 2:
        mean_as["_combined"] = mean_as[char_cols].astype(str).agg(" | ".join, axis=1)
        factor_sets.append(("_combined", "All three combined"))

    for col, nice in factor_sets:
        labels = mean_as[col].where(mean_as[col].astype(str) != "Unknown")
        pv = permanova(D, labels.values, n_perm=N_PERM, seed=0)

        groups = [grp["mean_angular_sep"].dropna().values for _, grp in mean_as.groupby(col) if len(grp) >= 2]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            k = len(groups)
            eta_sq = max((stat - k + 1) / (n_total - k), 0) if n_total > k else np.nan
        else:
            stat, p, eta_sq = np.nan, np.nan, np.nan

        test_results.append({
            "condition": condition, "characteristic": nice,
            "n_hospitals": pv["n_hospitals"], "n_groups": pv["n_groups"],
            "permanova_pseudo_F": round(pv["pseudo_F"], 3) if np.isfinite(pv["pseudo_F"]) else np.nan,
            "permanova_R2": round(pv["R2"], 4) if np.isfinite(pv["R2"]) else np.nan,
            "permanova_p": round(pv["p_value"], 4) if np.isfinite(pv["p_value"]) else np.nan,
            "permanova_n_perm": N_PERM,
            "kruskal_wallis_H": round(stat, 3) if np.isfinite(stat) else np.nan,
            "kw_p_value": round(p, 4) if np.isfinite(p) else np.nan,
            "kw_eta_squared": round(eta_sq, 4) if np.isfinite(eta_sq) else np.nan,
            "significant_0.05": bool(np.isfinite(pv["p_value"]) and pv["p_value"] < 0.05),
        })
        print(f"    {nice}: PERMANOVA F={pv['pseudo_F']:.3f}, R²={pv['R2']:.3f}, p={pv['p_value']:.4f} "
              f"| KW H={stat:.3f}, p={p:.4f}")

    if test_results:
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(tbl_dir / "characteristic_tests.csv", index=False)

    return test_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Conditions to process. Default: all available.")
    ap.add_argument("--ref-hospital", type=int, default=73)
    args = ap.parse_args()

    with open(args.paths) as f:
        paths_cfg = yaml.safe_load(f)
    processed = Path(paths_cfg["outputs"]["processed"])
    art_root = Path(paths_cfg["outputs"]["artifacts"])

    # Discover conditions
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = sorted([p.name.replace("flags_", "").replace(".parquet", "")
                             for p in processed.glob("flags_*.parquet")])

    print(f"Conditions: {conditions}")
    print(f"Reference hospital: {args.ref_hospital}")

    all_tests = []

    for condition in conditions:
        flags_path = processed / f"flags_{condition}.parquet"
        if not flags_path.exists():
            print(f"  {condition}: flags not found, skipping")
            continue

        flags = pd.read_parquet(flags_path)
        fig_dir = ensure_dir(art_root / condition / "figures")
        tbl_dir = ensure_dir(art_root / condition / "tables")
        results = process_condition(condition, processed, flags, args.ref_hospital, fig_dir, tbl_dir)
        if results:
            all_tests.extend(results)

    # Save combined test results
    if all_tests:
        combined = pd.DataFrame(all_tests)
        out = ensure_dir(art_root / "general") / "all_characteristic_tests.csv"
        combined.to_csv(out, index=False)
        print(f"\nWrote {out}")

        print("\n" + "=" * 60)
        print("SUMMARY: Do hospital characteristics explain angular separation?")
        print("=" * 60)
        for _, row in combined.iterrows():
            sig = "YES" if row["significant_0.05"] else "NO"
            print(f"  {row['condition']:15s} × {row['characteristic']:20s}: "
                  f"PERMANOVA R²={row['permanova_R2']:.3f}, p={row['permanova_p']:.4f}, sig={sig}")


if __name__ == "__main__":
    main()
