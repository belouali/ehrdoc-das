"""Figure 1: diabetes documentation at three hospitals, the seven-lobe vectors and the angles between them.

Reads mimic_external/results/paper_venn_counts.csv and paper_venn_das.csv (scripts/42_venn_data.py).
Writes venn_figure_<date>.png at print size (6.5 in wide). Palette: neutral greys as in Figures 2, 3 and S1,
with medication dark grey, past history light grey and diagnosis muted blue.
Hospitals are labelled A, B and C; the eICU IDs go in a muted line under each label.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib_venn import venn3
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
SUB = HERE.parent / "output"  # upload folder
OUT = SUB / "figures" / "Figure1.png"
INK, MUTED, GRID, SHADE = "#222222", "#777777", "#DDDDDD", "#E9EFF3"
BLUE, LBLUE, MEMB, GREY, GREEN = "#1F5A96", "#8DB8DE", "#E08A2C", "#8C8C8C", "#4E9A6A"  # palette: DAS blue, condition DAS light blue, membership dark grey, oracles light grey, no-medication green
SRC = {"M": "#5A5A5A", "H": "#BDBDBD", "D": BLUE}  # medication dark grey, past history light grey, diagnosis blue

counts = pd.read_csv(RES / "paper_venn_counts.csv")
das = pd.read_csv(RES / "paper_venn_das.csv")
lobe = {c.split("_", 2)[2]: c for c in counts.columns if c.startswith("lobe_")}
# venn3 subsets with A = medication, B = past history, C = diagnosis: Abc, aBc, ABc, abC, AbC, aBC, ABC
VENN = ["M only", "H only", "H∩M", "D only", "D∩M", "D∩H", "D∩H∩M"]
STRIP = [("M only", "M"), ("H∩M", "MH"), ("D∩M", "MD"), ("D∩H∩M", "MHD"), ("H only", "H"), ("D∩H", "HD"), ("D only", "D")]
BEDS = {"<100": "<100", "100 - 249": "100-249", "250 - 499": "250-499", ">= 500": "≥500"}
LETTERS = "ABC"


def pair(a, b):
    r = das[((das.a == a) & (das.b == b)) | ((das.a == b) & (das.b == a))].iloc[0]
    return float(r.das)


plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK})
fig = plt.figure(figsize=(6.5, 3.6), dpi=300)
hs = counts.hospitalid.tolist()
x0s, w = [0.025, 0.255, 0.485], 0.22

for i, (_, r) in enumerate(counts.iterrows()):
    x = x0s[i]
    tot = sum(int(r[lobe[k]]) for k in VENN)
    fig.text(x + w / 2, 0.965, f"Hospital {LETTERS[i]}", ha="center", fontsize=8.5, weight="bold", color=INK)
    fig.text(x + w / 2, 0.925, f"{tot:,} stays with diabetes evidence", ha="center", fontsize=6.6, color=INK)
    fig.text(x + w / 2, 0.89, f"{BEDS.get(r.beds, r.beds)} beds, {'teaching' if r.teaching else 'non-teaching'}, {r.region}",
             ha="center", fontsize=6.2, color=MUTED)
    ax = fig.add_axes([x + 0.01, 0.40, w - 0.02, 0.47])
    sub = [int(r[lobe[k]]) for k in VENN]
    v = venn3(subsets=sub, set_labels=("Medication", "Past history", "Diagnosis"),
              set_colors=(SRC["M"], SRC["H"], SRC["D"]), alpha=0.7, ax=ax)
    for t in v.set_labels:
        if t: t.set_fontsize(6.4); t.set_color(INK)
    for t, k in zip(v.subset_labels, VENN):
        if t:
            n = int(r[lobe[k]])
            t.set_text(f"{n:,}" if n / tot >= 0.03 else "")
            t.set_fontsize(6.2); t.set_color(INK)
    for p in v.patches:
        if p: p.set_edgecolor("white"); p.set_linewidth(0.6)
    # seven-lobe vector as bars, source membership shown by dots under each bar
    bx = fig.add_axes([x + 0.025, 0.115, w - 0.05, 0.22])
    vals = [int(r[lobe[k]]) / tot for k, _ in STRIP]
    bx.bar(range(7), vals, width=0.68, color=INK, zorder=3)
    for j, val in enumerate(vals):
        if val >= 0.05:
            bx.text(j, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=5.0, color=INK)
    bx.set_ylim(0, 0.9); bx.set_xlim(-0.6, 6.6)
    bx.set_yticks([0, 0.4, 0.8]); bx.tick_params(axis="y", labelsize=5.6, length=2, width=0.5, pad=1)
    bx.set_xticks([])
    bx.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    for s in ("top", "right"): bx.spines[s].set_visible(False)
    bx.spines["left"].set_linewidth(0.5)
    if i == 0:
        bx.set_ylabel("Share of stays", fontsize=6, color=INK, labelpad=2)
    for j, (_, code) in enumerate(STRIP):
        for k, s in enumerate(code):
            bx.scatter(j, -0.09 - 0.085 * k, s=7, color=SRC[s], clip_on=False, zorder=4, linewidths=0)
    fig.text(x + w / 2, 0.015, f"eICU hospital {int(r.hospitalid)}", ha="center", fontsize=5.8, color=MUTED)

# angle panel: each hospital's vector drawn from one origin, angles measured from hospital A
ax = fig.add_axes([0.72, 0.30, 0.27, 0.60])
ax.set_xlim(-0.08, 1.12); ax.set_ylim(-0.08, 1.12); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=90, color=GRID, lw=0.6))
dab, dac, dbc = pair(hs[0], hs[1]), pair(hs[0], hs[2]), pair(hs[1], hs[2])
for lab, ang, off in [("A", 0, (0.07, -0.02)), ("B", dab, (0.07, 0.035)), ("C", dac, (0.0, 0.07))]:
    t = math.radians(ang)
    ax.add_patch(FancyArrowPatch((0, 0), (math.cos(t), math.sin(t)), arrowstyle="-|>", mutation_scale=7, color=INK, lw=1.0,
                                shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round"))
    ax.text(math.cos(t) + off[0], math.sin(t) + off[1], lab, fontsize=8, weight="bold", color=INK, ha="center", va="center")
ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=dac, color=MUTED, lw=0.7))
ax.text(0.30, 0.30, f"A to C\n{dac:.1f}°", fontsize=7, weight="bold", color=INK, ha="center", va="center", linespacing=1.15)
ax.text(0.62, 0.13, f"A to B {dab:.1f}°", fontsize=7, weight="bold", color=INK, ha="center", va="center")
ax.text(0.5, -0.09, f"B to C {dbc:.1f}°", fontsize=7, weight="bold", color=INK, ha="center", va="center")
ax.set_title("Diabetes DAS between hospitals", fontsize=7.5, weight="bold", color=INK, pad=4)
for k, (s_, name) in enumerate([("M", "Medication"), ("H", "Past history"), ("D", "Diagnosis")]):
    fig.text(0.86 - 0.0, 0.215 - 0.035 * k, "●", color=SRC[s_], fontsize=6, ha="right", va="center")
    fig.text(0.865, 0.215 - 0.035 * k, name, color=INK, fontsize=6.2, ha="left", va="center")
fig.text(0.78, 0.215, "Dots under\nbars: sources", fontsize=5.8, color=MUTED, ha="right", va="top", linespacing=1.2)
fig.text(0.855, 0.06, f"A and B differ {counts.stays_with_evidence.iloc[0] / counts.stays_with_evidence.iloc[1]:.0f}-fold in size\n"
         "but document diabetes alike", ha="center", fontsize=6.4, color=INK, style="italic", linespacing=1.2)
fig.savefig(OUT, dpi=300, facecolor="white")
plt.rcParams["svg.fonttype"] = "none"  # keep text editable in Illustrator, Inkscape or Affinity
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT, f"DAS A-B {dab:.2f}, A-C {dac:.2f}, B-C {dbc:.2f}")
