"""Design of the DAS paper: one cohort of 141 eICU hospitals, two analyses.

Left: documentation variation and model inference across all 141 hospitals. Right: DAS-guided training-site selection and generalizability
(internal-external cross-validation, held-out test, MIMIC-IV). Writes das_paper_design_flowchart_<date>.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "output" / "figures" / "FigureS1.png"
# AToP flow-diagram style: one light grey-blue fill, thin dark-grey edges and arrows, Arial, no colour coding
FILL, EDGE = "#E9EFF3", "#555555"
G = L = P = T = (FILL, EDGE, "#222222", "#333333")
LC, INK = EDGE, "#222222"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42})
W = 1000
fig, ax = plt.subplots(figsize=(10, 14), dpi=250)
ax.set_xlim(0, W)
ax.axis("off")


def box(x, y, w, h, title, sub, c, fs=9.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=c[0], ec=c[1], lw=0.6,
                                ls=ls, zorder=3))
    if sub:
        k = sub.count("\n")
        ax.text(x + w / 2, y + h / 2 - 9 - 6 * k, title, ha="center", va="center", fontsize=fs, weight="bold",
                color=c[2], zorder=4)
        ax.text(x + w / 2, y + h / 2 + 10 + 2 * k, sub, ha="center", va="center", fontsize=7.8, color=c[3], zorder=4,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold", color=c[2], zorder=4)


def path(pts, col=LC, dashed=False, head=True):
    ls = (0, (4, 3)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=0.7, ls=ls, zorder=2, solid_capstyle="butt")
    a, b = pts[-2], pts[-1]
    if head:
        ax.annotate("", xy=b, xytext=a, zorder=2, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7,
                    mutation_scale=10, linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=0.7, ls=ls, zorder=2)


def label(x, y, s, col=LC, ha="center", fs=7.4):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=col, zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))


def panel(x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="white", ec=LC, lw=0.6,
                                zorder=1))
    ax.text(x + 16, y + 19, title, fontsize=9.6, weight="bold", color=INK, va="center", zorder=4)
    ax.text(x + 16, y + 36, sub, fontsize=8, color=LC, va="center", zorder=4)


def diamond(cx, cy, hw, hh, text, fs=8.6):
    ax.add_patch(Polygon([(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)], closed=True, fc="white",
                         ec=LC, lw=0.6, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight="bold", color=INK, zorder=4, linespacing=1.3)


# ---------------------------------------------------------------- shared cohort
CX = 500
box(360, 12, 280, 48, "eICU: 187 hospitals", "87,785 ICU stays", G, fs=10.5)
box(700, 62, 170, 56, "46 not used", "fewer than 100 stays", L, fs=9)
path([(CX, 60), (CX, 90), (700, 90)])
path([(CX, 90), (CX, 122)])
box(320, 122, 360, 84, "141 eligible hospitals", "≥100 stays each; outcomes: death, ICU stay\n>5 and >7 days, hospital stay >14 days;\nICU readmission", G)
AX, BX = 190, 690           # centres of part A and part B
path([(CX, 206), (CX, 222)], head=False)
path([(CX, 222), (AX, 222), (AX, 250)])
path([(CX, 222), (BX, 222), (BX, 250)])
label(AX + 8, 238, "all 141 hospitals together", ha="left")
label(BX + 8, 238, "every hospital a target in turn", ha="left")

# ---------------------------------------------------------------- part A: documentation variation
ya = 250
panel(15, ya, 350, 1110, "A. Variation and inference", "how hospitals differ, and what goes with it")
box(35, ya + 58, 310, 98, "DAS for every hospital pair", "main: mean DAS\nsecondary: diabetes, hypertension,\nCHF, AF; no reference hospital", G, fs=9.2)
path([(AX, ya + 156), (AX, ya + 186)])
ax.add_patch(FancyBboxPatch((28, ya + 186), 324, 680, boxstyle="square,pad=0", fc="white", ec="#BBBBBB", lw=0.6, zorder=1.5))
ax.text(AX, ya + 204, "Separate analyses of the DAS matrix", ha="center", va="center", fontsize=8.6, weight="bold", color=INK, zorder=4)
yy = ya + 222
for h, title, sub, ls in [
        (84, "Hospital positions", "principal coordinates of the DAS\ndistances between hospitals;\nMIMIC-IV placed as one more point", "-"),
        (110, "How much hospitals differ", "DAS between hospitals against a\nsplit-half noise floor within hospitals;\nsame hospitals stand out for\nevery condition? (Mantel test)", "-"),
        (110, "What goes with DAS", "region, bed size, teaching status,\nand case mix (APACHE score,\nED admissions, transfers);\nPERMANOVA with a dispersion check", "-"),
        (154, "Outcome variation", "mean DAS, each of the 5 outcomes:\npatient-level model + hospital intercept;\nadd DAS position, characteristics, both;\nshare of between-hospital variation\nexplained, compared across outcomes;\ndo alike hospitals have alike intercepts?", "-"),
        (84, "Sensitivity analyses", "the analyses above without hospitals\nwith no medication records;\nstay outcomes among survivors", "--")]:
    box(45, yy, 290, h, title, sub, G, fs=9.2 if ls == "-" else 9, ls=ls)
    yy += h + 20
yy = ya + 866
ax.text(AX, yy + 40, "This analysis describes all 141 hospitals\nat once, with no split and no screening. It asks\nwhether documentation differs beyond noise,\nand what those differences line up with.",
        ha="center", va="center", fontsize=7.8, color=LC, style="italic", linespacing=1.35)

# ---------------------------------------------------------------- part B: site selection
xb, wb = 385, 600
panel(xb, ya, wb, 1110, "B. Training sites and generalizability",
      "can DAS pick the hospitals to train a model for a new hospital?")
box(xb + 20, ya + 58, 560, 44, "Random split, balanced by region and size", None, G, fs=9)
path([(BX, ya + 102), (BX, ya + 116)], head=False)
path([(BX, ya + 116), (BX - 150, ya + 116), (BX - 150, ya + 130)])
path([(BX, ya + 116), (BX + 150, ya + 116), (BX + 150, ya + 130)])
box(xb + 30, ya + 130, 250, 48, "71 development set", "used to choose the method", G)
box(xb + 320, ya + 130, 250, 48, "70 held-out test set", "untouched until the held-out test", G, ls="--")

# B1 loop
y1 = ya + 200
path([(BX - 150, ya + 178), (BX - 150, y1 + 10)])
ax.add_patch(FancyBboxPatch((xb + 15, y1 + 10), wb - 30, 330, boxstyle="square,pad=0", fc="white",
                            ec="#BBBBBB", lw=0.6, zorder=1.5))
ax.text(xb + 30, y1 + 28, "Development loop", fontsize=9, weight="bold", color=INK, va="center", zorder=4)
ax.text(xb + 30, y1 + 44, "each development hospital takes a turn as the target", fontsize=7.8, color=LC, va="center", zorder=4)
box(xb + 30, y1 + 60, 260, 46, "Candidate pool (70)", "the other development hospitals", P, fs=9)
box(xb + 310, y1 + 60, 260, 46, "Target hospital", "next development hospital", T, fs=9)
box(xb + 30, y1 + 128, 260, 46, "Rank candidates 6 ways", "5 DAS measures, 1 membership", P, fs=9)
box(xb + 22, y1 + 196, 276, 58, "Training set: 10 hospitals", "10 nearest, 10 farthest per ranking,\n200 random sets, same size", P, fs=9)
box(xb + 310, y1 + 200, 260, 50, "Score on all its stays", "gain over random sets, and its SE", T, fs=9)
path([(xb + 160, y1 + 106), (xb + 160, y1 + 128)])
path([(xb + 160, y1 + 174), (xb + 160, y1 + 196)])
path([(xb + 290, y1 + 225), (xb + 310, y1 + 225)])
path([(xb + 440, y1 + 106), (xb + 440, y1 + 200)])
path([(xb + 440, y1 + 151), (xb + 290, y1 + 151)], col=EDGE, dashed=True)
label(xb + 368, y1 + 140, "records only", col=EDGE, fs=7)
label(xb + 440, y1 + 180, "its outcomes", col=EDGE, fs=7)
diamond(xb + 440, y1 + 285, 70, 24, "All 71 done?", fs=8.4)
path([(xb + 440, y1 + 250), (xb + 440, y1 + 261)])
path([(xb + 510, y1 + 285), (xb + 578, y1 + 285), (xb + 578, y1 + 83), (xb + 570, y1 + 83)])
label(xb + 528, y1 + 275, "no", fs=7.6)
path([(xb + 440, y1 + 309), (xb + 440, y1 + 318), (BX - 30, y1 + 318), (BX - 30, y1 + 360)])
label(xb + 455, y1 + 318, "yes", fs=7.6, ha="left")

# meta-analysis and screening
y2 = y1 + 360
box(xb + 60, y2, 480, 50, "Meta-analysis for each arm and outcome", "pooled gain, 95% CI, prediction interval", G, fs=9.2)
path([(BX - 30, y2 + 50), (BX - 30, y2 + 70)])
box(xb + 60, y2 + 70, 480, 58, "Development results", "5 DAS measures × 5 outcomes, nearest and farthest;\nmembership model as comparator", G, fs=9.2)
path([(BX - 30, y2 + 128), (BX - 30, y2 + 140)])
box(xb + 60, y2 + 140, 480, 58, "Selection ceiling", "each target's stays split in half; oracles choose with\none half's outcomes, every set scored on the other", G, fs=9.2)
path([(BX - 30, y2 + 198), (BX - 30, y2 + 216)])
box(xb + 110, y2 + 216, 380, 40, "Main measure: mean DAS", None, G, fs=9.2)

# held-out test
y3 = y2 + 280
path([(BX - 30, y2 + 256), (BX - 30, y3)])
box(xb + 30, y3, 540, 54, "Held-out test: the same loop", "each of the 70 held-out hospitals as target, candidate pool = other 69", T, fs=9.2)
path([(BX + 150, ya + 178), (BX + 150, ya + 190), (xb + 590, ya + 190), (xb + 590, y3 + 27), (xb + 570, y3 + 27)])
path([(BX - 30, y3 + 54), (BX - 30, y3 + 68)], head=False)
path([(BX - 30, y3 + 68), (xb + 100, y3 + 68), (xb + 100, y3 + 80)])
path([(BX - 30, y3 + 68), (BX - 30, y3 + 80)])
path([(BX - 30, y3 + 68), (xb + 480, y3 + 68), (xb + 480, y3 + 80)])
box(xb + 20, y3 + 80, 165, 64, "Main result", "pooled by\nmeta-analysis", G, fs=9)
box(xb + 195, y3 + 80, 170, 64, "Sensitivity", "targets with\n≥20 events", G, fs=9, ls="--")
box(xb + 375, y3 + 80, 205, 64, "Sensitivity", "without hospitals with\nno medication records", G, fs=9, ls="--")

# B4 external
y4 = y3 + 180
path([(xb + 102, y3 + 144), (xb + 102, y3 + 156), (BX - 30, y3 + 156), (BX - 30, y4)])
label(BX - 16, y3 + 165, "same measures and outcomes", ha="left")
box(xb + 20, y4, 560, 62, "External test: MIMIC-IV", "all ICU stays as one target; candidates: all 141 eICU hospitals or the 117 with\nmedication records; nearest, farthest, 500 random sets", T, fs=9.2)

# key: the one line style that carries meaning
yg = ya + 1132
ax.plot([250, 280], [yg, yg], color=EDGE, lw=0.7, ls=(0, (4, 3)))
ax.text(288, yg, "dashed arrow: records of the target hospital, never its outcomes", fontsize=8, va="center", color=INK, style="italic")
H = yg + 16
ax.set_ylim(H, 0)
fig.set_size_inches(10, H / 100)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.rcParams["svg.fonttype"] = "none"  # keep text editable in Illustrator, Inkscape or Affinity
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
print(OUT)
