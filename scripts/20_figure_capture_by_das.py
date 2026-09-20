#!/usr/bin/env python
"""Figure: phenotype capture under single-source rules, hospitals ordered by DAS.

Reads artifacts/<condition>/tables/phenotype_fragility.csv (script 11) and writes
artifacts/<condition>/figures/capture_by_das.png. Hospitals with no medication
records are shaded as a data-availability stratum.

Usage:
  python scripts/20_figure_capture_by_das.py --conditions diabetes hypertension chf af
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

NICE = {"diabetes": "diabetes", "hypertension": "hypertension",
        "chf": "heart failure", "af": "atrial fibrillation"}
SERIES = [
    ("capture_Dx_only", "Diagnosis table only", "#2a78d6", "o"),
    ("capture_PH_only", "Past history only", "#eb6834", "s"),
    ("capture_Rx_only", "Medication table only", "#1baf7a", "^"),
]


def make_figure(df: pd.DataFrame, cond: str, out: str) -> None:
    df = df.drop_duplicates("hospitalid").sort_values("DAS").reset_index(drop=True)
    x = np.arange(len(df))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

    for col, label, color, mk in SERIES:
        rho, _ = spearmanr(df["DAS"], df[col])
        ax.plot(x, df[col], marker=mk, ms=4.2, lw=0, color=color, alpha=0.9,
                markeredgecolor="#fcfcfb", markeredgewidth=0.4,
                label=f"{label}  (ρ vs DAS = {rho:+.2f})")

    ax2 = ax.twiny()
    ax2.set_xlim(-1, len(df))
    ticks = [0, 10, 20, 30, 50, 88]
    pos = [int(np.argmin(np.abs(df["DAS"].values - t))) for t in ticks]
    ax2.set_xticks(pos); ax2.set_xticklabels([f"{t}°" for t in ticks])
    ax2.set_xlabel("Documentation angular separation (DAS) from reference hospital 73",
                   color="#52514e", labelpad=6)
    ax2.tick_params(colors="#52514e", length=3)
    for s in ax2.spines.values():
        s.set_visible(False)

    ax.set_xlim(-1, len(df))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks([])
    ax.set_xlabel(f"Hospitals (n = {len(df)}), ordered by DAS", color="#52514e")
    ax.set_ylabel(f"Share of {NICE.get(cond, cond)} patients captured", color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8.5, handlelength=1.6)

    zero = df["capture_Rx_only"] == 0
    if zero.any():
        first = int(np.argmax(zero.values))
        ax.axvspan(first - 0.5, len(df) - 0.5, color="#0b0b0b", alpha=0.04, lw=0)
        ax.text(first + 0.5, 0.06, f"{int(zero.sum())} hospitals with\nno medication records",
                fontsize=7.5, color="#52514e", va="bottom")

    ax.set_title("Same phenotype rule, different capture: single-source definitions "
                 f"across eICU hospitals ({NICE.get(cond, cond)})",
                 fontsize=10.5, color="#0b0b0b", loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes", "hypertension", "chf", "af"])
    args = ap.parse_args()
    art = yaml.safe_load(open(args.paths))["outputs"]["artifacts"]
    for cond in args.conditions:
        p = os.path.join(art, cond, "tables", "phenotype_fragility.csv")
        if not os.path.exists(p):
            print("missing", p)
            continue
        make_figure(pd.read_csv(p), cond, os.path.join(art, cond, "figures", "capture_by_das.png"))


if __name__ == "__main__":
    main()
