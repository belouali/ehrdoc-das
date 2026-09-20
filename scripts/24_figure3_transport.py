#!/usr/bin/env python
"""Figure 3: leave-one-hospital-out transportability by training-set choice.

For each condition, mean held-out AUC (top row) and Brier score (bottom row)
of models trained on the k documentation-nearest hospitals, k random hospitals
(mean of 20 draws), and the k farthest hospitals, for k = 5, 10, 20, with the
model pooled over all other hospitals as a dashed reference line.

Reads:  artifacts/<cond>/tables/transportability_summary.csv (all hospitals)
        artifacts/sensitivity_zero_source/transportability_zero_source_summary.csv
Writes: artifacts/general/figure3_transport.png            (all hospitals)
        artifacts/general/figure3_transport_nozero.png     (excluding no-medication hospitals)

Usage:
  python scripts/24_figure3_transport.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ehrdoc.utils.io import ensure_dir

COND = ["diabetes", "hypertension", "chf", "af"]
NICE = {"diabetes": "Diabetes", "hypertension": "Hypertension", "chf": "Heart failure", "af": "Atrial fibrillation"}
SERIES = [("nearest", "mean_nearest", "#2a78d6", "o", "k nearest (DAS)"),
          ("random", "mean_random", "#eda100", "s", "k random (mean of 20 draws)"),
          ("farthest", "mean_farthest", "#eb6834", "^", "k farthest (DAS)")]
KS = [5, 10, 20]


def draw(df_by_cond: dict, out: Path, subtitle: str):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), dpi=300, sharex=True)
    fig.patch.set_facecolor("white")
    for j, c in enumerate(COND):
        d = df_by_cond.get(c)
        for i, (metric, ylab) in enumerate([("AUC", "Held-out AUC"), ("Brier", "Held-out Brier score")]):
            ax = axes[i, j]
            if d is None:
                ax.axis("off"); continue
            sub = d[d["metric"] == metric].set_index("k").reindex(KS)
            for key, col, color, mk, lab in SERIES:
                if col in sub.columns and sub[col].notna().any():
                    ax.plot(KS, sub[col].values, marker=mk, ms=5, lw=1.6, color=color, label=lab)
            if "mean_all" in sub.columns and sub["mean_all"].notna().any():
                ax.axhline(sub["mean_all"].mean(), ls="--", lw=1.2, color="#52514e", label="all other hospitals (pooled)")
            ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
            ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(NICE[c], fontsize=10, pad=6)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("k training hospitals")
            if metric == "Brier":
                ax.invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Transportability of ICU-mortality models by training-set choice ({subtitle})", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()
    art = Path(yaml.safe_load(open(args.paths))["outputs"]["artifacts"])
    general = ensure_dir(art / "general")

    by_cond = {}
    for c in COND:
        p = art / c / "tables" / "transportability_summary.csv"
        if p.exists():
            d = pd.read_csv(p); by_cond[c] = d[(d["size_group"] == "large")]
    draw(by_cond, general / "figure3_transport.png", "all hospitals")

    p = art / "sensitivity_zero_source" / "transportability_zero_source_summary.csv"
    if p.exists():
        d = pd.read_csv(p); d = d[d["sample"] == "excluding_zero_source"]
        draw({c: d[d["condition"] == c] for c in COND if (d["condition"] == c).any()},
             general / "figure3_transport_nozero.png", "excluding hospitals with no medication records")


if __name__ == "__main__":
    main()
