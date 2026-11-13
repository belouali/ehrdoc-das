"""Figure 2 (S_das_variation): A, hospital map; B, DAS between hospitals against the noise floor.

Reads mimic_external/results/partA_*.csv. Writes partA_figure_<date>.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
SUB = HERE.parent / "output"  # upload folder
OUT = SUB / "figures" / "Figure2.png"
INK, MUTED, LIGHT, GRID, SHADE = "#222222", "#777777", "#BBBBBB", "#DDDDDD", "#E9EFF3"
BLUE, LBLUE, MEMB, GREY, GREEN = "#1F5A96", "#8DB8DE", "#E08A2C", "#8C8C8C", "#4E9A6A"  # palette: DAS blue, condition DAS light blue, membership dark grey, oracles light grey, no-medication green
MEAS = [("mean", "Mean DAS"), ("diabetes", "Diabetes"), ("hypertension", "Hypertension"), ("chf", "Heart failure"),
        ("af", "Atrial fibrillation")]
TASKS = [("mortality", "Death"), ("A7_los_gt5d", "ICU stay >5 days"), ("A3_los_gt7d", "ICU stay >7 days"),
         ("A6_hospital_los_gt14d", "Hospital stay >14 days")]
MODELS = [("+ das", "DAS position", INK), ("+ characteristics", "Hospital characteristics and case mix", MUTED),
          ("+ both", "Both", LIGHT)]

pos = pd.read_csv(RES / "partA_positions.csv")
pos = pos[pos["sample"] == "all"]
axes = pd.read_csv(RES / "partA_pcoa_axes.csv").set_index("sample").loc["all"]
spread = pd.read_csv(RES / "partA_spread.csv").set_index(["sample", "measure"])

plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig = plt.figure(figsize=(6.5, 3.9), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.42, left=0.09, right=0.98, top=0.92, bottom=0.30)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# A: hospital map
ax = fig.add_subplot(gs[0, 0])
h = pos[pos.unit != "MIMIC-IV"]
med, nomed = h[~h.no_medication], h[h.no_medication]
ax.scatter(med.axis1, med.axis2, s=14, c=BLUE, edgecolors="white", linewidths=0.4, zorder=3,
           label=f"Hospitals with medication records ({len(med)})")
ax.scatter(nomed.axis1, nomed.axis2, s=14, facecolors="white", edgecolors=GREEN, linewidths=0.9, zorder=3,
           label=f"Hospitals with no medication records ({len(nomed)})")
m = pos[pos.unit == "MIMIC-IV"].iloc[0]
ax.scatter(m.axis1, m.axis2, s=46, marker="D", c="white", edgecolors=INK, linewidths=1.2, zorder=4, label="MIMIC-IV")
ax.annotate("MIMIC-IV", (m.axis1, m.axis2), xytext=(6, 6), textcoords="offset points", fontsize=7, color=INK)
# the Figure 1 example hospitals (A, B, C), in the order of paper_venn_counts.csv
ex = pd.read_csv(RES / "paper_venn_counts.csv").hospitalid.astype(str).tolist()
P = {lab: h[h.unit.astype(str) == hid].iloc[0] for lab, hid in zip("ABC", ex)}
for r in P.values():
    ax.scatter(r.axis1, r.axis2, s=48, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=5)
ax.annotate("A, B", ((P["A"].axis1 + P["B"].axis1) / 2, (P["A"].axis2 + P["B"].axis2) / 2), xytext=(-58, 30),
            textcoords="offset points", fontsize=7, weight="bold", color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkB=4))
ax.annotate("C", (P["C"].axis1, P["C"].axis2), xytext=(7, -2), textcoords="offset points", fontsize=7, weight="bold", color=INK)
ax.axhline(0, color=GRID, lw=0.5, zorder=0)
ax.axvline(0, color=GRID, lw=0.5, zorder=0)
ax.set_xlabel(f"Documentation difference, first axis ({100 * axes.axis1_share:.0f}% of variation)")
ax.set_ylabel(f"Second axis ({100 * axes.axis2_share:.0f}% of variation)")
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.17), frameon=False, fontsize=6.4, handletextpad=0.3)
clean(ax)
ax.set_title("A  Hospital positions on mean DAS", loc="left", fontsize=8, weight="bold", color=INK, x=-0.12)

# B: DAS between hospitals against noise
ax = fig.add_subplot(gs[0, 1])
for i, (k, lab) in enumerate(MEAS):
    for j, (smp, col, off) in enumerate([("all", BLUE, -0.16), ("with_medication", LBLUE, 0.16)]):
        r = spread.loc[(smp, k)]
        y = i + off
        ax.plot([r.das_q25, r.das_q75], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot(r.das_median, y, "o", ms=4, mfc="white", mec=col, mew=1.0, zorder=4)
        ax.plot(r.noise_mean, y, "|", ms=7, mew=1.4, color=col, zorder=4)
        ax.text(92, y, f"{100 * r.share_pairs_beyond_noise:.0f}%", fontsize=6.6, color=col, va="center", ha="right")
ax.set_yticks(range(len(MEAS)))
ax.set_yticklabels([l for _, l in MEAS])
ax.set_ylim(len(MEAS) - 0.5, -0.9)
ax.set_xlim(0, 93)
ax.set_xlabel("DAS between hospitals (degrees)")
ax.text(92, -0.75, "pairs beyond\nnoise", fontsize=6.4, color=MUTED, ha="right", va="center", linespacing=1.1)
ax.tick_params(axis="y", length=0)
clean(ax)
ax.spines["left"].set_visible(False)
ax.set_title("B  How much hospitals differ", loc="left", fontsize=8, weight="bold", color=INK, x=-0.38)
h1, = ax.plot([], [], color=BLUE, lw=2.2, label="All hospitals (IQR, median)")
h2, = ax.plot([], [], color=LBLUE, lw=2.2, label="Without no-medication hospitals")
h3, = ax.plot([], [], "|", color=INK, ms=7, mew=1.4, label="Split-half noise within hospitals")
ax.legend(handles=[h1, h2, h3], loc="upper left", bbox_to_anchor=(-0.02, -0.17), frameon=False, fontsize=6.4,
          handletextpad=0.4, ncol=1)

fig.savefig(OUT, dpi=300, facecolor="white")
plt.rcParams["svg.fonttype"] = "none"  # keep text editable in Illustrator, Inkscape or Affinity
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT)
