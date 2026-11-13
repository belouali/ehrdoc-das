#!/usr/bin/env python
"""Summary table and two figures for the follow-up paper, from aggregate results only.

  results/summary_table.csv   one row per test: model, construct, direction, condition, gain, CI
  figures/fig_mimic_position.png   eICU hospitals' DHM angle to MIMIC-IV, per condition
  figures/fig_gain_vs_eicu.png     nearest-10 minus random-10 AUROC: eICU targets against MIMIC-IV

Usage (from mimic_external/):
  uv run --no-project --with pandas --with matplotlib python scripts/14_summary_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES, FIG = BASE / "results", BASE / "figures"
CONDS = ["diabetes", "hypertension", "chf", "af"]
LABEL = {"diabetes": "Diabetes", "hypertension": "Hypertension*", "chf": "Heart failure", "af": "Atrial fibrillation"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def stage4_main(tag):
    c = pd.read_csv(RES / f"stage4{tag}_contrasts.csv")
    return c[(c.k == 10) & (c.version == "size_controlled") & (c.config == "main")].set_index("condition")


def summary_table():
    rows = []
    for model, tag, construct in [("P0", "", "DAM"), ("P0", "_DHM", "DHM"), ("P0", "_DHAM", "DHAM"),
                                  ("M1", "_M1_DHM", "DHM"), ("M1", "_M1_DAM", "DAM"), ("M1", "_M1_DHAM", "DHAM"),
                                  ("M1", "_M1_LEARNED", "learned"), ("M2", "_M2_DHM", "DHM"), ("M2", "_M2_DAM", "DAM")]:
        c = stage4_main(tag)
        for cond in CONDS:
            r = c.loc[cond]
            rows.append(dict(test="eICU to MIMIC-IV, nearest-10 minus random-10", model=model, construct=construct,
                             condition=cond, estimate=r.auroc_diff, lo=r.auroc_diff_lo, hi=r.auroc_diff_hi,
                             pct_random_beaten=r.auroc_pct_beaten))
    t = pd.read_csv(RES / "eicu_transport_dham_summary.csv")
    t = t[t.version == "size_controlled"].set_index("condition")
    for cname in ["DHM", "DHAM"]:
        for cond in CONDS:
            r = t.loc[cond]
            rows.append(dict(test="eICU targets, nearest-10 minus random-10 (mean over targets)", model="M1",
                             construct=cname, condition=cond, estimate=r[f"diff_{cname}"], lo=r[f"diff_{cname}_lo"],
                             hi=r[f"diff_{cname}_hi"], share_targets_positive=r[f"diff_{cname}_share_pos"]))
    s = pd.read_csv(RES / "reverse_summary.csv")
    s = s[(s["sample"] == "primary") & (s.loss == "loss_auroc")].set_index("distance")
    for cname in ["DHM", "DHAM", "DAM"]:
        for cond in CONDS:
            r = s.loc[f"{cname}_{cond}"]
            rows.append(dict(test="MIMIC-IV to eICU, Spearman(DAS, AUROC loss)", model="M1", construct=cname,
                             condition=cond, estimate=r.spearman, lo=r.spearman_lo, hi=r.spearman_hi,
                             adjusted_per_sd=r.adj_das_per_sd, adjusted_p=r.adj_das_p))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(RES / "summary_table.csv", index=False)
    return out


def style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_position():
    das = pd.read_csv(RES / "das_to_mimic.csv")
    das = das[(das.construct == "DHM") & (das.reference == "ref_2011_2019")]
    summ = pd.read_csv(RES / "das_summary.csv")
    summ = summ[(summ.construct == "DHM") & (summ.reference == "ref_2011_2019") & (summ["sample"] == "all_hospitals")].set_index("condition")
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for i, cond in enumerate(CONDS):
        v = das[das.condition == cond].das.to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=BLUE, alpha=0.55, linewidths=0)
        med = summ.loc[cond, "eicu_pairwise_median"]
        ax.hlines(med, i - 0.3, i + 0.3, color=INK, linewidth=2)
    ax.set_xticks(range(len(CONDS)), [f"{LABEL[c]}\nMIMIC-IV: {summ.loc[c, 'mimic_kth_percentile_k10']:.0f}th pct" for c in CONDS], fontsize=9)
    ax.set_ylabel("Angle to MIMIC-IV, degrees (DHM)", fontsize=9, color=INK)
    ax.set_title("Each dot is one eICU hospital. Bar: median angle between eICU hospitals.", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.set_ylim(0, None)
    fig.text(0.01, 0.01, "*Hypertension flagged. Percentile: MIMIC-IV's 10th-nearest angle among eICU hospitals' own 10th-nearest angles.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig_mimic_position.png")
    plt.close(fig)


def fig_gain():
    tg = pd.read_csv(RES / "eicu_transport_dham_targets.csv")
    tg = tg[tg.version == "size_controlled"]
    mim = {"DHM": stage4_main("_M1_DHM"), "DHAM": stage4_main("_M1_DHAM")}
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for i, cond in enumerate(CONDS):
        for j, (cname, col) in enumerate([("DHM", BLUE), ("DHAM", ORANGE)]):
            x0 = i + (j - 0.5) * 0.36
            v = tg[tg.condition == cond][f"diff_{cname}"].to_numpy()
            ax.scatter(x0 + rng.uniform(-0.1, 0.1, len(v)), v, s=8, color=col, alpha=0.45, linewidths=0)
            ax.hlines(v.mean(), x0 - 0.14, x0 + 0.14, color=INK, linewidth=2)
            ax.scatter([x0], [mim[cname].loc[cond, "auroc_diff"]], marker="D", s=42, color=col,
                       edgecolors=INK, linewidths=1.2, zorder=5)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(CONDS)), [LABEL[c] for c in CONDS], fontsize=9)
    ax.set_ylabel("AUROC, nearest-10 minus random-10", fontsize=9, color=INK)
    style(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=BLUE, label="7 lobes (DHM), eICU target"),
               Line2D([], [], marker="o", linestyle="", color=ORANGE, label="15 lobes (DHAM), eICU target"),
               Line2D([], [], marker="D", linestyle="", color="white", markeredgecolor=INK, label="MIMIC-IV target"),
               Line2D([], [], color=INK, linewidth=2, label="Mean over eICU targets")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.text(0.01, 0.01, "Paper's model (M1), size-controlled. *Hypertension flagged.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig_gain_vs_eicu.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    t = summary_table()
    fig_position()
    fig_gain()
    print(f"wrote {RES / 'summary_table.csv'} ({len(t)} rows), {FIG / 'fig_mimic_position.png'}, {FIG / 'fig_gain_vs_eicu.png'}")


if __name__ == "__main__":
    main()
