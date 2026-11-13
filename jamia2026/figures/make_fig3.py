"""DRAFT figure: split-half selection ceiling on the v6 design (development set, all hospitals).

Reads mimic_external/results/v6_2026-11-11/ceiling_pooled.csv (scripts/49_ceiling_iecv.py). Every arm scored on the half
of each target's stays the oracle did not see. Writes ceiling_figure_v6_<date>.png/.svg.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "v6_2026-11-11"
SUB = HERE.parent / "output"  # upload folder
OUT = SUB / "figures" / "Figure3.png"
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else "all"

INK, MUTED, GRID, ORACLE = "#222222", "#777777", "#DDDDDD", "#9A9A9A"
BLUE, LBLUE, MEMB, GREY, GREEN = "#1F5A96", "#8DB8DE", "#6B6B6B", "#B5B5B5", "#4E9A6A"  # palette: DAS blue, condition DAS light blue, membership dark grey, oracles light grey, no-medication green
TASKS = [("mortality", "Death"), ("A7_los_gt5d", "ICU stay\nover 5 days"), ("A3_los_gt7d", "ICU stay\nover 7 days"),
         ("A6_hospital_los_gt14d", "Hospital stay\nover 14 days"), ("icu_readmit", "ICU\nreadmission")]
ROWS = [("set_best", "Best random set", "oracle"), ("rank_best", "Best 10 single hospitals", "oracle"),
        ("near_membership", "Membership model", "memb"), ("near_mean", "Mean DAS", "das"),
        ("near_diabetes", "Diabetes DAS", "das"), ("near_hypertension", "Hypertension DAS", "das"),
        ("near_chf", "Heart failure DAS", "das"), ("near_af", "Atrial fibrillation DAS", "das"),
        ("set_worst", "Worst random set", "oracle"), ("rank_worst", "Worst 10 single hospitals", "oracle")]
GAPS = {2, 8}  # blank line before these rows
XL = (-0.09, 0.045)

d = pd.read_csv(RES / "ceiling_pooled.csv")
d = d[d["sample"] == SAMPLE].set_index(["task", "arm"])
ys, y = [], 0
for i in range(len(ROWS)):
    y += 1 if i in GAPS else 0
    ys.append(y); y += 1

plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "xtick.major.size": 2.5, "xtick.color": INK, "axes.edgecolor": INK})
fig = plt.figure(figsize=(8.0, 4.2), dpi=300)
gs = fig.add_gridspec(1, len(TASKS), wspace=0.1, left=0.17, right=0.985, top=0.83, bottom=0.14)
for j, (task, ttl) in enumerate(TASKS):
    ax = fig.add_subplot(gs[0, j])
    ax.set_xlim(*XL); ax.set_ylim(ys[-1] + 0.7, -0.7)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for x in (-0.09, -0.06, -0.03, 0.03):
        ax.axvline(x, color=GRID, lw=0.4, zorder=0)
    ax.set_xticks([-0.06, 0, 0.03]); ax.set_xticklabels(["−0.06", "0", "+0.03"])
    ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in ROWS] if j == 0 else [])
    ax.tick_params(axis="y", length=0, pad=4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(ttl, fontsize=7.5, weight="bold", color=INK, pad=4)
    for (arm, _, kind), yy in zip(ROWS, ys):
        if (task, arm) not in d.index:
            continue
        r = d.loc[(task, arm)]
        col = {"oracle": GREY, "memb": MEMB, "das": BLUE if arm == "near_mean" else LBLUE}[kind]
        mk = {"oracle": "D", "memb": "o", "das": "o"}[kind]
        ax.plot([max(r.lo, XL[0]), min(r.hi, XL[1])], [yy, yy], color=col, lw=1.0, zorder=3, solid_capstyle="butt")
        ax.plot(r.pooled_gain, yy, mk, ms=4.0 if kind == "oracle" else 4.2,
                mfc="white" if kind == "memb" else col, mec=col, mew=0.9, zorder=4)
    if j == 2:
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.set_xlabel("Pooled ΔAUROC on the unseen half, training set − random sets of 10 (95% CI)",
                      fontsize=7.5, color=INK, labelpad=3)
ntar = pd.read_csv(RES / f"ceiling_targets_development_{SAMPLE}.csv").target.nunique()
lab = "all hospitals" if SAMPLE == "all" else "hospitals with medication records"
fig.text(0.012, 0.95, f"Selection ceiling, development set, {lab}, {ntar} target hospitals", fontsize=8.5,
         weight="bold", color=INK)
fig.text(0.012, 0.905, "Grey diamonds: oracles that pick with the target's outcomes on one half of its stays.",
         fontsize=7.5, color=INK)
fig.savefig(OUT, dpi=300, facecolor="white")
plt.rcParams["svg.fonttype"] = "none"
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT)
