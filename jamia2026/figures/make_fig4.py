"""Figure 4: held-out eICU test (v6 run, mean DAS as the main measure).

Panels: (A) all hospitals; (B) hospitals with medication records. Rows: nearest and farthest 10 by mean DAS and by the
membership model, and the nearest 10 by each condition's DAS. ΔAUROC against random sets of 10 (95% CI).
MIMIC-IV is Figure 5 (make_fig5.py).
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
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = SUB / "figures" / "Figure4.png"
PENDING = "--pending-facility" in sys.argv

INK, MUTED, GRID = "#222222", "#8A8A8A", "#DDDDDD"
BLUE, LBLUE, MEMB, GREY, GREEN = "#1F5A96", "#8DB8DE", "#6B6B6B", "#B5B5B5", "#4E9A6A"  # palette: DAS blue, condition DAS light blue, membership dark grey, oracles light grey, no-medication green
TASKS = [("mortality", "Death"), ("A7_los_gt5d", "ICU stay\nover 5 days"), ("A3_los_gt7d", "ICU stay\nover 7 days"),
         ("A6_hospital_los_gt14d", "Hospital stay\nover 14 days"), ("icu_readmit", "ICU\nreadmission")]
ARMS = [("near_mean", "Nearest 10, mean DAS", BLUE, "o", BLUE), ("near_membership", "Nearest 10, membership", MEMB, "o", "white"),
        ("far_mean", "Farthest 10, mean DAS", BLUE, "s", BLUE), ("far_membership", "Farthest 10, membership", MEMB, "s", "white")]
XL = (-0.072, 0.045)
COND = [(f"near_{k}", f"Nearest 10, {l} DAS", LBLUE, "^", LBLUE) for k, l in
        [("diabetes", "diabetes"), ("hypertension", "hypertension"), ("chf", "heart failure"), ("af", "AF")]]
ARMS_H = ARMS[:1] + COND + ARMS[1:]  # held-out panels also show the condition measures (2026-11-12)

pooled = pd.read_csv(RES / "iecv_pooled.csv")
pooled = pooled[(pooled.analysis == "main") & (pooled.half == "held_out")].set_index(["sample", "task", "arm"])
mim = {s: pd.read_csv(RES / f"iecv_mimic_mean{suf}.csv").set_index(["task", "arm"]) for s, suf in
       [("all", ""), ("with_medication", "_withmed")]}
ntar = lambda s: pd.read_csv(RES / f"iecv_targets_held_out_{s}.csv").target.nunique()

# rows: (panel, label, getter)
panels = []
for s, lab in [("all", "all hospitals"), ("with_medication", "hospitals with medication records")]:
    rows = [(a, l, c, mk, fc, (lambda t, a=a, s=s: pooled.loc[(s, t, a)][["pooled_gain", "lo", "hi"]].tolist()
                               if (s, t, a) in pooled.index else None)) for a, l, c, mk, fc in ARMS_H]
    panels.append((f"Held-out eICU test, {lab}, {ntar(s)} target hospitals", rows, None))
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "xtick.major.size": 2.5, "xtick.color": INK, "axes.edgecolor": INK})
heights = [len(p[1]) + 0.6 for p in panels]
fig = plt.figure(figsize=(8.6, 6.4), dpi=300)
gs = fig.add_gridspec(len(panels), len(TASKS), height_ratios=heights, wspace=0.12, hspace=0.5,
                      left=0.215, right=0.985, top=0.88, bottom=0.08)
for pi, (title, rows, is_m) in enumerate(panels):
    for j, (task, ttl) in enumerate(TASKS):
        ax = fig.add_subplot(gs[pi, j])
        n = len(rows)
        ax.set_xlim(*XL); ax.set_ylim(n - 0.4, -0.6)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        for x in (-0.06, -0.03, 0.03):
            ax.axvline(x, color=GRID, lw=0.4, zorder=0)
        ax.set_xticks([-0.06, -0.03, 0, 0.03]); ax.set_xticklabels(["−0.06", "−0.03", "0", "+0.03"])
        ax.set_yticks(range(n)); ax.set_yticklabels([r[1] for r in rows] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=4)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        if is_m:
            ax.axhline(3.5, color=GRID, lw=0.6)
        if pi == 0:
            ax.set_title(ttl, fontsize=7.5, weight="bold", color=INK, pad=6)
        if j == 0:
            ax.text(-1.25, 1.0 + (1.6 if pi == 0 else 0.6) / n, f"{'ABC'[pi]}  {title}", transform=ax.transAxes,
                    fontsize=8, weight="bold", color=INK, ha="left", va="bottom", clip_on=False)
        if PENDING and task == "disch_facility":
            ax.text(sum(XL) / 2, (n - 1) / 2, "rerun in\nprogress", ha="center", va="center", fontsize=7, color=MUTED,
                    bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
            continue
        for y, r in enumerate(rows):
            v = r[5](task)
            if v is None:
                continue
            g, lo, hi = v
            ax.plot([max(lo, XL[0]), min(hi, XL[1])], [y, y], color=r[2], lw=1.0, zorder=3, solid_capstyle="butt")
            ax.plot(min(max(g, XL[0]), XL[1]), y, r[3], ms=4.0, mfc=r[4], mec=r[2], mew=0.9, zorder=4)
            if is_m:
                ax.text(XL[1], y, f"{r[6](task):.0f}", ha="right", va="center", fontsize=6.3, color=MUTED)
        if pi == len(panels) - 1 and j == 2:
            ax.xaxis.set_label_coords(0.5, -0.16)
            ax.set_xlabel("ΔAUROC, training set − mean of random sets of 10 (95% CI)", fontsize=7.5, color=INK)
fig.savefig(OUT, dpi=300, facecolor="white")
plt.rcParams["svg.fonttype"] = "none"
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT)
