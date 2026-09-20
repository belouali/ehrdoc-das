#!/usr/bin/env python
"""Create publication-ready assets (figures + tables).

Outputs (under artifacts/amia/):
  Figure1_venn_triptych_<condition>.png  (for each condition)
  Figure1b_hospital_embedding_<method>.png
  Figure2_condition_concordance_heatmap.png
  Table1_model_comparison_large_cohort.csv

Usage:
  python scripts/05_make_amia_assets.py --concordance-csv artifacts/concordance/concordance_rho.csv
  python scripts/05_make_amia_assets.py --all-conditions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from ehrdoc.utils.io import ensure_dir
from ehrdoc.distances.metrics import angular_separation
from ehrdoc.venn.vectors import hospital_venn_counts, hospital_venn_vectors
from ehrdoc.viz.venn_plots import plot_triptych
from ehrdoc.viz.heatmaps import plot_matrix_heatmap


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_npz(path: Path) -> tuple[np.ndarray, list[int]]:
    z = np.load(path, allow_pickle=True)
    return z["D"], z["ids"].tolist()


def _compute_angular_distances_from_ref(
    flags: pd.DataFrame, ref_hospital: int, target_hospitals: list[int],
) -> dict[int, float]:
    """Compute angular separation between ref and targets directly from flags."""
    all_hids = list(set([ref_hospital] + target_hospitals))
    counts = hospital_venn_counts(flags, min_patients_per_hospital=None)
    counts = counts[counts["hospitalid"].isin(all_hids)]
    vectors = hospital_venn_vectors(counts)

    ref_row = vectors[vectors["hospitalid"] == ref_hospital]
    if ref_row.empty:
        return {}

    lobe_cols = [c for c in vectors.columns if c != "hospitalid"]
    ref_vec = ref_row[lobe_cols].values[0]

    distances = {}
    for hid in target_hospitals:
        if hid == ref_hospital:
            continue
        row = vectors[vectors["hospitalid"] == hid]
        if row.empty:
            continue
        vec = row[lobe_cols].values[0]
        distances[hid] = angular_separation(ref_vec, vec)
    return distances


def _hospital_has_all_three_sources(flags: pd.DataFrame, hid: int, min_per_source: int = 5) -> bool:
    """Check if a hospital has at least min_per_source patients in each Venn circle."""
    pid = "uniquepid" if "uniquepid" in flags.columns else "patientunitstayid"
    sub = flags[flags["hospitalid"] == hid]
    n_med = sub[sub["exists_in_medication"].astype(bool)][pid].nunique()
    n_hx = sub[sub["exists_in_pastHistory"].astype(bool)][pid].nunique()
    n_dx = sub[sub["exists_in_diagnosis"].astype(bool)][pid].nunique()
    return n_med >= min_per_source and n_hx >= min_per_source and n_dx >= min_per_source


def _get_hosp_chars_tuple(flags: pd.DataFrame, hid: int) -> dict:
    """Get hospital characteristics dict."""
    sub = flags[flags["hospitalid"] == hid].head(1)
    chars = {}
    for col in ["region", "numbedscategory", "teachingstatus"]:
        if col in sub.columns and not sub[col].isna().all():
            chars[col] = str(sub[col].iloc[0])
        else:
            chars[col] = None
    return chars


def _chars_match(c1: dict, c2: dict) -> tuple[int, int]:
    """Return (n_matching, n_comparable) between two characteristic dicts.
    
    Only counts characteristics where BOTH hospitals have non-None values.
    For a 'same characteristics' match, caller should require 
    n_matching == n_comparable == 3 (all present and all match).
    """
    cols = ["region", "numbedscategory", "teachingstatus"]
    comparable = sum(1 for c in cols if c1.get(c) is not None and c2.get(c) is not None)
    matches = sum(1 for c in cols if c1.get(c) is not None and c2.get(c) is not None
                  and c1.get(c) == c2.get(c))
    return matches, comparable


def _pick_triptych_hospitals(
    D: np.ndarray, ids: list[int], flags: pd.DataFrame,
    ref_hospital: int, min_per_source: int = 5,
) -> tuple[int, int, int]:
    """Pick ref + closest-with-different-chars + farthest-with-same-chars.

    Panel 1 (ref): The reference hospital
    Panel 2 (closest-different): Similar documentation but different hospital type
        → Shows documentation behavior is independent of observables
    Panel 3 (farthest-same): Different documentation but same hospital type
        → Shows observables don't determine documentation behavior
    """
    if ref_hospital not in ids:
        print(f"  WARNING: ref hospital {ref_hospital} not in distance matrix. Using first hospital.")
        ref_hospital = ids[0]

    ref_idx = ids.index(ref_hospital)
    dists = D[ref_idx, :].copy()
    ref_chars = _get_hosp_chars_tuple(flags, ref_hospital)
    char_cols = ["region", "numbedscategory", "teachingstatus"]
    has_chars = all(ref_chars.get(c) is not None for c in char_cols)

    print(f"  Reference hospital {ref_hospital}: {ref_chars}")

    # ── CLOSEST with DIFFERENT characteristics ──
    # Sort ascending (closest first), find first with all 3 sources and ≤1 matching char
    order_asc = np.argsort(dists)
    closest_diff_id = None
    closest_any_id = None  # fallback: absolute closest with all 3 sources

    for idx in order_asc:
        hid = ids[idx]
        if hid == ref_hospital:
            continue
        if not _hospital_has_all_three_sources(flags, hid, min_per_source):
            continue

        # Track absolute closest as fallback
        if closest_any_id is None:
            closest_any_id = hid

        if has_chars and closest_diff_id is None:
            hid_chars = _get_hosp_chars_tuple(flags, hid)
            matches, comparable = _chars_match(ref_chars, hid_chars)
            if comparable >= 2 and matches <= 1:  # at least 2 chars comparable, at most 1 matches
                closest_diff_id = hid
                print(f"  Closest with DIFFERENT chars: Hospital {hid} ({dists[idx]:.2f}°) — {hid_chars} "
                      f"({matches}/{comparable} match)")
                break

    if closest_diff_id is None:
        closest_diff_id = closest_any_id
        if closest_diff_id is not None:
            print(f"  → No characteristic-different close hospital found. "
                  f"Using absolute closest: Hospital {closest_diff_id} ({dists[ids.index(closest_diff_id)]:.2f}°)")
        else:
            closest_diff_id = ids[int(np.argmin(dists[dists > 0]))]
            print(f"  → Fallback to absolute closest: Hospital {closest_diff_id}")

    # ── FARTHEST with tiered characteristic matching ──
    # Tier 1: All 3 characteristics match
    # Tier 2: 2 of 3 match (if tier 1 farthest < 70°)
    # Tier 3: Region only matches (if tier 2 farthest < 70°)
    # Tier 4: Absolute farthest with all 3 Venn sources (no char matching)
    MIN_DRAMATIC_ANGLE = 70.0

    order_desc = np.argsort(-dists)

    # Build list of valid candidates (all 3 Venn sources populated)
    valid_candidates = []  # (hid, distance, chars_dict)
    farthest_any_id = None

    for idx in order_desc:
        hid = ids[idx]
        if hid == ref_hospital:
            continue
        if not _hospital_has_all_three_sources(flags, hid, min_per_source):
            print(f"  Skipping Hospital {hid} ({dists[idx]:.2f}°) — missing a Venn circle")
            continue
        if farthest_any_id is None:
            farthest_any_id = hid
            print(f"  Farthest with all 3 sources: Hospital {hid} ({dists[idx]:.2f}°)")
        hid_chars = _get_hosp_chars_tuple(flags, hid) if has_chars else {}
        valid_candidates.append((hid, dists[idx], hid_chars))

    farthest_same_id = None

    if has_chars and valid_candidates:
        # Tier 1: Full match (all 3 characteristics)
        for hid, d, hc in valid_candidates:
            matches, comparable = _chars_match(ref_chars, hc)
            if matches == 3 and comparable == 3:
                farthest_same_id = hid
                print(f"  Tier 1 (3/3 match): Hospital {hid} ({d:.2f}°) — {hc}")
                break

        # Tier 2: 2-of-3 match if tier 1 < 70°
        if farthest_same_id is None or dists[ids.index(farthest_same_id)] < MIN_DRAMATIC_ANGLE:
            tier1_id = farthest_same_id
            tier1_d = dists[ids.index(farthest_same_id)] if farthest_same_id else 0
            for hid, d, hc in valid_candidates:
                matches, comparable = _chars_match(ref_chars, hc)
                if comparable >= 2 and matches >= 2 and d > tier1_d:
                    farthest_same_id = hid
                    print(f"  Tier 2 (2/3 match, upgrading from {tier1_d:.1f}°): Hospital {hid} ({d:.2f}°) — {hc}")
                    break

        # Tier 3: Region only if tier 2 < 70°
        if farthest_same_id is None or dists[ids.index(farthest_same_id)] < MIN_DRAMATIC_ANGLE:
            tier2_id = farthest_same_id
            tier2_d = dists[ids.index(farthest_same_id)] if farthest_same_id else 0
            for hid, d, hc in valid_candidates:
                if (ref_chars.get("region") is not None and
                    hc.get("region") is not None and
                    hc["region"] == ref_chars["region"] and
                    d > tier2_d):
                    farthest_same_id = hid
                    print(f"  Tier 3 (region only, upgrading from {tier2_d:.1f}°): Hospital {hid} ({d:.2f}°) — {hc}")
                    break

        # Tier 4: Absolute farthest if still < 70°
        if farthest_same_id is None or dists[ids.index(farthest_same_id)] < MIN_DRAMATIC_ANGLE:
            tier3_d = dists[ids.index(farthest_same_id)] if farthest_same_id else 0
            if farthest_any_id is not None:
                farthest_same_id = farthest_any_id
                print(f"  Tier 4 (no char match, upgrading from {tier3_d:.1f}°): "
                      f"Hospital {farthest_any_id} ({dists[ids.index(farthest_any_id)]:.2f}°)")

    if farthest_same_id is None:
        farthest_same_id = farthest_any_id if farthest_any_id else ids[int(np.argmax(dists))]
        print(f"  → Fallback: Hospital {farthest_same_id}")

    closest_d = dists[ids.index(closest_diff_id)]
    farthest_d = dists[ids.index(farthest_same_id)]
    farthest_chars = _get_hosp_chars_tuple(flags, farthest_same_id) if has_chars else {}
    print(f"  Triptych: ref={ref_hospital}, "
          f"closest-diff={closest_diff_id} ({closest_d:.2f}°), "
          f"farthest={farthest_same_id} ({farthest_d:.2f}°) — {farthest_chars}")

    return ref_hospital, closest_diff_id, farthest_same_id


# ── Embedding ────────────────────────────────────────────────────────────────

def _make_embedding(D, ids, highlight, method="direct", ref_hospital=None, condition="diabetes"):
    method_label = method.upper()

    if method == "direct":
        if ref_hospital is None or ref_hospital not in ids:
            print(f"  WARNING: ref {ref_hospital} not in ids, falling back to MDS")
            method = "mds"
        else:
            ref_idx = ids.index(ref_hospital)
            x = D[ref_idx, :]
            y = D.mean(axis=1)
            X = np.column_stack([x, y])
            method_label = "Direct"

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(metric="precomputed", n_components=2, random_state=42,
                                n_neighbors=min(15, len(ids) - 1))
            X = reducer.fit_transform(D)
            method_label = "UMAP"
        except ImportError:
            method = "mds"

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, metric="precomputed", random_state=42,
                     perplexity=min(30, len(ids) - 1)).fit_transform(D)
            method_label = "t-SNE"
        except Exception:
            method = "mds"

    if method == "mds":
        X = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                n_init=4, max_iter=300).fit_transform(D)
        method_label = "MDS"

    df = pd.DataFrame({"hospitalid": ids, "x": X[:, 0], "y": X[:, 1]})
    mean_d = D.mean(axis=1)
    n_bins = min(4, len(ids))
    q = pd.qcut(mean_d, n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates="drop")
    df["quartile"] = q.astype(str)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"Q1": "#4575b4", "Q2": "#fee090", "Q3": "#74add1", "Q4": "#d73027"}
    for label, grp in df.groupby("quartile"):
        ax.scatter(grp["x"], grp["y"], label=label, s=40, alpha=0.8,
                   color=colors.get(label), edgecolors="k", linewidths=0.3)

    # Annotate highlighted hospitals with smart offsets
    highlight_points = []
    for hid in highlight:
        if hid in set(ids):
            row = df[df["hospitalid"] == hid].iloc[0]
            ax.scatter([row["x"]], [row["y"]], s=120, marker="x", color="black", linewidths=2)
            highlight_points.append((hid, row["x"], row["y"]))

    for i, (hid, x, y) in enumerate(highlight_points):
        dx_off, dy_off = 12, -15
        for j, (_, x2, y2) in enumerate(highlight_points[:i]):
            xr = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1)
            yr = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1)
            if abs(x - x2) / xr < 0.08 and abs(y - y2) / yr < 0.08:
                dx_off, dy_off = 25, 25
        ax.annotate(
            f"ID: {hid}", (x, y), fontsize=10, fontweight="bold",
            xytext=(dx_off, dy_off), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8) if abs(dx_off) > 15 else None,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    if method_label == "Direct":
        ax.set_title(f"Hospital documentation distance landscape ({condition.capitalize()})\n"
                     f"Each point = one hospital (n={len(ids)}); reference: Hospital {ref_hospital}",
                     fontsize=12)
        ax.set_xlabel(f"Angular separation from Hospital {ref_hospital} (degrees)\n"
                      f"0° = identical documentation pattern, 90° = maximally different")
        ax.set_ylabel("Mean angular separation to all other hospitals (degrees)\n"
                      "Higher = more atypical documentation overall")
    else:
        ax.set_title(f"Hospital embedding ({method_label}) — {condition.capitalize()}", fontsize=12)
        ax.set_xlabel(f"{method_label}-1")
        ax.set_ylabel(f"{method_label}-2")

    ax.legend(title="Documentation similarity quartile\n(Q1=most typical, Q4=most atypical)",
              loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--condition", default="diabetes")
    ap.add_argument("--all-conditions", action="store_true",
                    help="Generate triptychs for all available conditions (supplementary)")
    ap.add_argument("--dist-npz", default=None)
    ap.add_argument("--ref-hospital", type=int, default=73)
    ap.add_argument("--hospitals", nargs="+", type=int, default=None)
    ap.add_argument("--concordance-csv", default=None)
    ap.add_argument("--embedding-method", default="direct", choices=["direct", "mds", "umap", "tsne"])
    args = ap.parse_args()

    with open(args.paths, "r") as f:
        paths_cfg = yaml.safe_load(f)

    processed = Path(args.processed) if args.processed else Path(paths_cfg["outputs"]["processed"])
    art_root = Path(paths_cfg["outputs"]["artifacts"])
    general_dir = ensure_dir(art_root / "general")

    # ── Determine which conditions to process ──
    if args.all_conditions:
        conditions = sorted([p.name.replace("flags_", "").replace(".parquet", "")
                             for p in processed.glob("flags_*.parquet")])
    else:
        conditions = [args.condition]

    print(f"Conditions to process: {conditions}")

    for cond_idx, condition in enumerate(conditions):
        print(f"\n{'='*60}")
        print(f"Processing: {condition}")
        print(f"{'='*60}")

        # Load flags
        flags_path = processed / f"flags_{condition}.parquet"
        if not flags_path.exists():
            print(f"  WARNING: {flags_path} not found, skipping.")
            continue
        flags = pd.read_parquet(flags_path)

        # Debug: check what columns are in flags
        print(f"  Flags columns: {list(flags.columns)}")
        print(f"  Flags shape: {flags.shape}")

        # Debug: check hospital characteristics availability
        for col in ["numbedscategory", "teachingstatus", "region"]:
            if col in flags.columns:
                print(f"  {col}: {flags[col].nunique()} unique values, {flags[col].isna().sum()} NaN")
            else:
                print(f"  {col}: NOT FOUND in flags")

        # Load distance matrix
        if args.dist_npz and cond_idx == 0:
            dist_path = Path(args.dist_npz)
        else:
            cand = sorted(processed.glob(f"dist_{condition}*.npz"),
                          key=lambda p: p.stat().st_mtime)
            if not cand:
                print(f"  WARNING: No dist_{condition}*.npz found, skipping.")
                continue
            dist_path = cand[-1]

        D, ids = _load_npz(dist_path)
        print(f"  Distance matrix: {len(ids)} hospitals")

        # ── Pick triptych hospitals ──
        if args.hospitals and cond_idx == 0:
            triptych_hospitals = args.hospitals
            ref_hospital = triptych_hospitals[0]
        else:
            ref_hospital = args.ref_hospital
            ref_hospital, closest_id, farthest_id = _pick_triptych_hospitals(
                D, ids, flags, ref_hospital, min_per_source=5,
            )
            triptych_hospitals = [ref_hospital, closest_id, farthest_id]

        # Debug: print Venn circle sizes for each triptych hospital
        pid = "uniquepid" if "uniquepid" in flags.columns else "patientunitstayid"
        for hid in triptych_hospitals:
            sub = flags[flags["hospitalid"] == hid]
            n_med = sub[sub["exists_in_medication"].astype(bool)][pid].nunique()
            n_hx = sub[sub["exists_in_pastHistory"].astype(bool)][pid].nunique()
            n_dx = sub[sub["exists_in_diagnosis"].astype(bool)][pid].nunique()
            print(f"  Hospital {hid}: Medication={n_med}, PastHistory={n_hx}, Diagnosis={n_dx}")

        # ── Triptych ──
        angular_dists = _compute_angular_distances_from_ref(flags, ref_hospital, triptych_hospitals)

        fig_tri = plot_triptych(
            flags, triptych_hospitals,
            condition=condition,
            angular_distances=angular_dists,
            ref_hospital=ref_hospital,
        )

        # Output to per-condition folder
        cond_fig_dir = ensure_dir(art_root / condition / "figures")
        cond_tbl_dir = ensure_dir(art_root / condition / "tables")
        out_tri = cond_fig_dir / "venn_triptych.png"
        fig_tri.savefig(out_tri, dpi=300, bbox_inches="tight")
        plt.close(fig_tri)
        print(f"  Wrote {out_tri}")

        # Save triptych selection for downstream scripts (07, 08)
        import json
        triptych_info = {"ref": triptych_hospitals[0],
                         "closest_diff": triptych_hospitals[1],
                         "farthest_same": triptych_hospitals[2]}
        triptych_path = cond_tbl_dir / "triptych_hospitals.json"
        with open(triptych_path, "w") as f:
            json.dump(triptych_info, f)
        print(f"  Wrote {triptych_path}")

    # ── Scatter plot (primary condition only) ──
    print(f"\n{'='*60}")
    print(f"Generating scatter plot for {args.condition}")
    print(f"{'='*60}")

    # Reload primary condition
    flags = pd.read_parquet(processed / f"flags_{args.condition}.parquet")
    cand = sorted(processed.glob(f"dist_{args.condition}*.npz"), key=lambda p: p.stat().st_mtime)
    D, ids = _load_npz(cand[-1])

    # Re-pick for primary condition
    ref_hospital = args.ref_hospital
    if args.hospitals:
        triptych_hospitals = args.hospitals
    else:
        ref_hospital, closest_id, farthest_id = _pick_triptych_hospitals(
            D, ids, flags, ref_hospital, min_per_source=5,
        )
        triptych_hospitals = [ref_hospital, closest_id, farthest_id]

    fig_emb = _make_embedding(D, ids, highlight=triptych_hospitals, method=args.embedding_method,
                              ref_hospital=ref_hospital, condition=args.condition)
    cond_fig_dir = ensure_dir(art_root / args.condition / "figures")
    out_emb = cond_fig_dir / f"scatter_quartile.png"
    fig_emb.savefig(out_emb, dpi=300, bbox_inches="tight")
    plt.close(fig_emb)
    print(f"  Wrote {out_emb}")

    # ── Concordance heatmap ──
    if args.concordance_csv:
        rho_df = pd.read_csv(args.concordance_csv, index_col=0)
        mat = rho_df.to_numpy(dtype=float)
        labels = rho_df.index.tolist()
        fig, _ = plot_matrix_heatmap(mat, labels=labels,
                                     title="Cross-condition concordance of pairwise angular separations\n"
                                           "(Spearman \u03c1 between hospital-pair distance rankings)")
        out_fig2 = general_dir / "concordance_heatmap.png"
        fig.savefig(out_fig2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_fig2}")

    # The model-comparison table is written by scripts/08_regression_models.py
    # (artifacts/<condition>/tables/model_comparison.csv) from the fitted models.
    # A hard-coded copy of an earlier run used to live here; it was removed so
    # that every reported number traces back to the frozen pipeline run.
    stale = general_dir / "model_comparison_large_cohort.csv"
    if stale.exists():
        stale.unlink()
        print(f"  Removed stale hard-coded table {stale}")

    print(f"\nDone. Check {art_root}/")


if __name__ == "__main__":
    main()
