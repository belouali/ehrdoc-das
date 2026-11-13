"""Figure 5: MIMIC-IV external test, candidates from all eICU hospitals (v6 run), two views of the same result
(author's request 2026-11-12): (A) dAUROC against the random mean with 95% CI, on the scale of Figure 4;
(B) percentile among the 500 random sets. No shaded band."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "v6_2026-11-11"
OUT = HERE.parent / "output" / "figures" / "Figure5.png"
INK, MUTED, GRID = "#222222", "#8A8A8A", "#DDDDDD"
BLUE, MEMB = "#1F5A96", "#6B6B6B"  # palette as Figures 3 and 4: mean DAS dark blue, membership dark grey
TASKS = [("mortality", "Death"), ("A7_los_gt5d", "ICU stay\nover 5 days"), ("A3_los_gt7d", "ICU stay\nover 7 days"),
         ("A6_hospital_los_gt14d", "Hospital stay\nover 14 days"), ("icu_readmit", "ICU\nreadmission")]
ARMS = [("near_mean", "Nearest 10, mean DAS", BLUE, "o", BLUE), ("near_membership", "Nearest 10, membership", MEMB, "o", "white"),
        ("far_mean", "Farthest 10, mean DAS", BLUE, "s", BLUE), ("far_membership", "Farthest 10, membership", MEMB, "s", "white")]
mim = pd.read_csv(RES / "iecv_mimic_mean.csv").set_index(["task", "arm"])
XL = (-0.072, 0.045)
PANELS = [("A", "ΔAUROC against the mean of 500 random sets of 10 (95% CI)", "gain"),
          ("B", "Percentile among the 500 random sets", "percentile")]

plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "xtick.major.size": 2.5, "xtick.color": INK, "axes.edgecolor": INK})
fig = plt.figure(figsize=(8.6, 4.9), dpi=300)
gs = fig.add_gridspec(2, len(TASKS), wspace=0.12, hspace=0.62, left=0.19, right=0.985, top=0.83, bottom=0.1)
n = len(ARMS)
for pi, (letter, xlabel, kind) in enumerate(PANELS):
    for j, (task, ttl) in enumerate(TASKS):
        ax = fig.add_subplot(gs[pi, j])
        ax.set_ylim(n - 0.4, -0.6)
        if kind == "gain":
            ax.set_xlim(*XL)
            ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
            for x in (-0.06, -0.03, 0.03):
                ax.axvline(x, color=GRID, lw=0.4, zorder=0)
            ax.set_xticks([-0.06, -0.03, 0, 0.03]); ax.set_xticklabels(["−0.06", "−0.03", "0", "+0.03"])
        else:
            ax.set_xlim(0, 100)
            ax.axvline(50, color=MUTED, lw=0.6, zorder=1)
            for x in (25, 75):
                ax.axvline(x, color=GRID, lw=0.4, zorder=0)
            ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100"]); ax.set_xlim(-4, 104)
        ax.set_yticks(range(n)); ax.set_yticklabels([a[1] for a in ARMS] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=4)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        if pi == 0:
            ax.set_title(ttl, fontsize=7.5, weight="bold", color=INK, pad=6)
        for y, (a, lab, col, mk, fc) in enumerate(ARMS):
            r = mim.loc[(task, a)]
            if kind == "gain":
                ax.plot([max(r.lo, XL[0]), min(r.hi, XL[1])], [y, y], color=col, lw=1.0, zorder=3, solid_capstyle="butt")
                ax.plot(min(max(r.gain, XL[0]), XL[1]), y, mk, ms=4.0, mfc=fc, mec=col, mew=0.9, zorder=4)
            else:
                ax.plot([50, r.percentile_mean], [y, y], color=col, lw=0.8, alpha=0.6, zorder=2)
                ax.plot(r.percentile_mean, y, mk, ms=4.0, mfc=fc, mec=col, mew=0.9, zorder=4)
        if j == 0:
            ax.text(-1.05, 1.0 + 0.9 / n, f"{letter}  {xlabel}", transform=ax.transAxes, fontsize=8, weight="bold",
                    color=INK, ha="left", va="bottom", clip_on=False)
fig.text(0.012, 0.95, "MIMIC-IV external test, all ICU stays as one target, candidates from all eICU hospitals",
         fontsize=8.5, weight="bold", color=INK)
fig.savefig(OUT, dpi=300, facecolor="white")
plt.rcParams["svg.fonttype"] = "none"
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT)
