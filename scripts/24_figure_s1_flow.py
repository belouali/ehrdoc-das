"""Figure S1, cohort flow for one condition, in the JAMA-type flow-diagram style.

Reads artifacts/general/cohort_flow.csv (written by scripts/18_cohort_flow_table1.py) and the
zero-source table (hospitals with a documentation vector), so no count is typed by hand.

Style: square boxes with a light grey fill and a thin dark-grey edge, a bold lead line (the count)
over a regular description, thin grey arrows, exclusions in a right-hand column. No colour coding.
The same constants as the AToP and DepAnx flow diagrams.

Usage:
    python scripts/24_figure_s1_flow.py [--condition diabetes] [--artifacts ../artifacts] [--dpi 300]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
})

FILL, EDGE = "#E9EFF3", "#555555"
LEAD_PT, TEXT_PT = 8.2, 7.8
LINE_H = 0.235
NICE = {"diabetes": "diabetes", "hypertension": "hypertension", "chf": "congestive heart failure", "af": "atrial fibrillation"}


def _box(ax, x, top, w, lead, text="", pad=0.14):
    n_lines = text.count("\n") + 1 if text else 0
    h = pad + LINE_H * 1.25 + (LINE_H * n_lines + 0.06 if text else 0) + pad
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="square,pad=0", fc=FILL, ec=EDGE, lw=0.6, zorder=2))
    ax.text(x + 0.14, top - pad, lead, va="top", ha="left", fontsize=LEAD_PT, fontweight="bold", zorder=3)
    if text:
        ax.text(x + 0.14, top - pad - LINE_H * 1.25 - 0.04, text, va="top", ha="left", fontsize=TEXT_PT, linespacing=1.4, zorder=3)
    return top - h


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=EDGE, shrinkA=0, shrinkB=0), zorder=1)


def n(v):
    return f"{int(v):,}"


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S1 (cohort flow).")
    p.add_argument("--condition", default="diabetes")
    p.add_argument("--artifacts", default="../artifacts")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path", default=None)
    args = p.parse_args()
    art = Path(args.artifacts)
    c = args.condition
    flow = pd.read_csv(art / "general" / "cohort_flow.csv")
    flow = flow[flow["condition"] == c].reset_index(drop=True)
    st = lambda prefix: flow[flow["stage"].str.startswith(prefix)].iloc[0]
    s_all, s_ap, s_das, s_cov, s_out = (st("All eICU"), st("With APACHE"), st("In hospitals with"),
                                        st("Complete model"), st("Hospitals with outcome"))
    zs = pd.read_csv(art / "sensitivity_zero_source" / "zero_source_hospitals.csv")
    n_vec = int(zs.loc[zs["condition"] == c, "n_hospitals"].iloc[0])
    thr = int(zs.loc[zs["condition"] == c, "min_patients"].iloc[0])
    assert int(s_cov.n_stays) == int(s_out.n_stays) and int(s_cov.n_hospitals) == int(s_out.n_hospitals), "outcome-variation step now excludes stays; add a box"
    out = Path(args.output_path) if args.output_path else art / "general" / f"cohort_flow_{c}.png"
    cond = NICE.get(c, c)

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.axis("off"); ax.set_xlim(0, 11.9); ax.set_ylim(0, 9.8)
    MX, MW = 0.4, 6.4          # main column
    EX, EW = 7.5, 4.2          # exclusion column
    GAP = 1.05                 # room for the exclusion box on each connector
    cx = MX + MW / 2

    def step(top, lead, text, excl_lead=None, excl_text=""):
        """Draw a main box at `top`; return (bottom, next_top). An exclusion box sits on the connector below it."""
        b = _box(ax, MX, top, MW, lead, text)
        if excl_lead is None:
            return b, b - 0.55
        t_next = b - GAP
        ymid = (b + t_next) / 2
        _box(ax, EX, ymid + 0.42, EW, excl_lead, excl_text)
        _arrow(ax, cx, ymid, EX, ymid)
        _arrow(ax, cx, b, cx, t_next)
        return b, t_next

    top = 9.7
    b1, t2 = step(top, f"{n(s_all.n_stays)} ICU stays in eICU-CRD v2.0",
                  f"{n(s_all.n_patients)} patients, {n(s_all.n_hospitals)} hospitals, 2014 to 2015\n"
                  f"Documentation vectors use all stays: {n_vec} hospitals\n"
                  f"have at least {thr} stays with {cond} evidence",
                  f"{n(s_all.n_stays - s_ap.n_stays)} ICU stays excluded ({100 * (s_all.n_stays - s_ap.n_stays) / s_all.n_stays:.1f}%)",
                  f"No APACHE IVa score\n{n(s_all.n_hospitals - s_ap.n_hospitals)} hospitals with no scored stay")
    b2, t3 = step(t2, f"{n(s_ap.n_stays)} ICU stays with an APACHE IVa score",
                  f"{n(s_ap.n_patients)} patients, {n(s_ap.n_hospitals)} hospitals",
                  f"{n(s_ap.n_stays - s_das.n_stays)} ICU stays excluded ({100 * (s_ap.n_stays - s_das.n_stays) / s_ap.n_stays:.1f}%)",
                  f"In {n(s_ap.n_hospitals - s_das.n_hospitals)} hospitals with fewer than {thr} scored\nstays with {cond} evidence, so no\ndocumentation vector")
    b3, t4 = step(t3, f"{n(s_das.n_stays)} ICU stays in hospitals with a documentation vector",
                  f"{n(s_das.n_patients)} patients, {n(s_das.n_hospitals)} hospitals; DAS computable for every hospital",
                  f"{n(s_das.n_stays - s_cov.n_stays)} ICU stays excluded ({100 * (s_das.n_stays - s_cov.n_stays) / s_das.n_stays:.1f}%)",
                  "Incomplete model covariates\n(age, sex, race/ethnicity, unit stay type,\nadmission source)")
    b4, _ = step(t4, f"{n(s_out.n_stays)} ICU stays in the mortality-model sample",
                 f"{n(s_out.n_patients)} patients, {n(s_out.n_hospitals)} hospitals\n"
                 f"Every hospital had both outcomes, so none was dropped\nfor lack of outcome variation")
    ax.set_ylim(b4 - 0.2, 9.8)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
