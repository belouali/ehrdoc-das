#!/usr/bin/env python
"""Figure 1 composite in the AMIA-abstract layout, built from the frozen run.

Three panels (reference hospital, documentation-nearest hospital with different
characteristics, documentation-distant hospital with the same characteristics),
each with a header (hospital ID, DAS, n, beds | teaching | region), a 3-source
Venn diagram of stay-level evidence, and the normalized 7-lobe vector strip.

Reads: data/processed/flags_<cond>.parquet, dist_<cond>_min*.csv,
       artifacts/<cond>/tables/triptych_hospitals.json (from script 05)
Writes: artifacts/<cond>/figures/figure1_composite.png

Usage:
  python scripts/22_figure1_composite.py --conditions diabetes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib_venn import venn3
import numpy as np
import pandas as pd
import yaml

from ehrdoc.utils.io import ensure_dir

# strip order and labels exactly as in the abstract: M, M+H, M+D, all 3, H, H+D, D
STRIP = [("M", "100"), ("M+H", "110"), ("M+D", "101"), ("all 3", "111"), ("H", "010"), ("H+D", "011"), ("D", "001")]
NICE = {"diabetes": "diabetes", "hypertension": "hypertension", "chf": "heart failure", "af": "atrial fibrillation"}


def patient_sets(flags: pd.DataFrame, hid: int):
    pid = "patientunitstayid"  # ICU stays: the unit on which the DAS vectors are computed
    sub = flags[flags["hospitalid"] == hid]
    M = set(sub.loc[sub["exists_in_medication"].astype(bool), pid])
    H = set(sub.loc[sub["exists_in_pastHistory"].astype(bool), pid])
    D = set(sub.loc[sub["exists_in_diagnosis"].astype(bool), pid])
    return M, H, D


def lobe_props(M, H, D):
    allp = M | H | D
    counts = {}
    for p in allp:
        key = f"{int(p in M)}{int(p in H)}{int(p in D)}"
        counts[key] = counts.get(key, 0) + 1
    n = len(allp)
    return {k: counts.get(k, 0) / n for k in [c for _, c in STRIP]}, n


def hospital_chars(flags: pd.DataFrame, hid: int):
    sub = flags[flags["hospitalid"] == hid].head(1)
    beds = str(sub["numbedscategory"].iloc[0]) if "numbedscategory" in sub and pd.notna(sub["numbedscategory"].iloc[0]) else "n/a"
    beds = beds.replace(">= 500", "≥500").replace(" - ", "-")
    t = str(sub["teachingstatus"].iloc[0]).lower() if "teachingstatus" in sub else ""
    teaching = "Yes" if t in ("t", "true", "1", "yes") else "No"
    region = str(sub["region"].iloc[0]) if "region" in sub and pd.notna(sub["region"].iloc[0]) else "n/a"
    return beds, teaching, region


def draw_strip(ax, props):
    ax.set_xlim(0, 7); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.02), 6.9, 0.96, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="#f5f8fc", edgecolor="#c9d6e6", linewidth=1.0, transform=ax.transData))
    ax.text(3.5, 0.84, "NORMALIZED 7-LOBE VECTOR", ha="center", va="center", fontsize=7.5, color="#6b7a8c")
    vals = [props[c] for _, c in STRIP]
    vmax = max(vals) if max(vals) > 0 else 1
    for i, ((lab, _), v) in enumerate(zip(STRIP, vals)):
        x = i + 0.5
        strength = v / vmax
        if v >= 0.03:
            fc = (0.11 + (1 - strength) * 0.55, 0.35 + (1 - strength) * 0.45, 0.65 + (1 - strength) * 0.3)
            ax.add_patch(FancyBboxPatch((x - 0.42, 0.36), 0.84, 0.30, boxstyle="round,pad=0.01,rounding_size=0.06",
                                        facecolor=fc, edgecolor="none"))
            ax.text(x, 0.51, f"{v:.2f}", ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if strength > 0.45 else "#1b2a3a")
        else:
            ax.text(x, 0.51, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#6b7a8c")
        ax.text(x, 0.17, lab, ha="center", va="center", fontsize=7.5, color="#6b7a8c")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes"])
    ap.add_argument("--ref-hospital", type=int, default=73)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.paths))
    processed = Path(cfg["outputs"]["processed"]); art = Path(cfg["outputs"]["artifacts"])

    for cond in args.conditions:
        flags = pd.read_parquet(processed / f"flags_{cond}.parquet")
        tri = json.load(open(art / cond / "tables" / "triptych_hospitals.json"))
        hosps = [tri["ref"], tri["closest_diff"], tri["farthest_same"]]
        dist = sorted(processed.glob(f"dist_{cond}_min*.csv"), key=lambda p: p.stat().st_mtime)[-1]
        d = pd.read_csv(dist)
        das = {int(r.hospital_2): float(r.angular_separation_deg) for r in d[d.hospital_1 == args.ref_hospital].itertuples()}

        fig = plt.figure(figsize=(13.5, 5.6), dpi=300)
        gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 3.6, 0.9], hspace=0.05, wspace=0.08)
        for j, hid in enumerate(hosps):
            M, H, D = patient_sets(flags, hid)
            props, n = lobe_props(M, H, D)
            beds, teaching, region = hospital_chars(flags, hid)
            axh = fig.add_subplot(gs[0, j]); axh.axis("off")
            title = f"Hospital ID {hid}" + (" (reference)" if hid == args.ref_hospital else "")
            axh.text(0.5, 0.85, title, ha="center", va="center", fontsize=11, fontweight="bold", transform=axh.transAxes)
            if hid == args.ref_hospital:
                axh.text(0.5, 0.45, f"n = {n:,} stays", ha="center", va="center", fontsize=9.5, color="#333", transform=axh.transAxes)
            else:
                axh.text(0.5, 0.45, f"DAS = {das[hid]:.2f}°", ha="right", va="center", fontsize=9.5, fontweight="bold", color="#c0392b", transform=axh.transAxes)
                axh.text(0.5, 0.45, f"  |  n = {n:,} stays", ha="left", va="center", fontsize=9.5, color="#333", transform=axh.transAxes)
            axh.text(0.5, 0.08, f"Beds: {beds}  |  Teaching: {teaching}  |  Region: {region}", ha="center", va="center", fontsize=8.5, color="#777", transform=axh.transAxes)

            axv = fig.add_subplot(gs[1, j])
            v = venn3([M, H, D], ("Medication", "Past History", "Diagnosis"), ax=axv, alpha=0.55)
            for t in v.set_labels:
                if t: t.set_fontsize(11)
            for t in v.subset_labels:
                if t: t.set_fontsize(8.5); t.set_fontweight("bold")

            axs = fig.add_subplot(gs[2, j]); draw_strip(axs, props)

        out = ensure_dir(art / cond / "figures") / "figure1_composite.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
