#!/usr/bin/env python
"""Highlight table and figures for the screen positive controls (P1-P8) and shared habit (P9-P12).

  results/screen_highlights.csv        one row per locked prediction: estimate, CI, Holm p, verdict
  figures/fig_screen_severity.png      documentation rate by objective stage (obesity, four lab conditions)
  figures/fig_screen_race_egfr.png     CKD documentation by race in the eGFR bands (B5)
  figures/fig_screen_habit.png         hospital habit: correlations, PC1 share, diagnosis fill, DAS

The habit panels refit the hospital effects of script 16 (primary version). Hospital-level values
go to the external drive only. Aggregates only in results/ and figures/.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scipy \
      --with statsmodels --with matplotlib python scripts/17_screen_figures.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s16", HERE / "16_shared_habit.py")
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)
s15 = s16.s15
RES, FIG = s15.RES, HERE.parent / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
STAGE = ["#86b6ef", "#2a78d6", "#104281"]
NAME = {"obesity": "Obesity", "aki": "Acute kidney injury", "hyponatremia": "Hyponatremia",
        "anemia": "Anemia", "thrombocytopenia": "Thrombocytopenia"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def highlights():
    pc = pd.read_csv(RES / "positive_controls.csv")
    sh = pd.read_csv(RES / "shared_habit.csv")
    pc, sh = pc[pc.in_holm], sh[sh.in_holm & (sh.version == "primary")]
    ask = {"P1": "Race-based eGFR left Black patients with eGFR 45-60 under-documented for CKD",
           "P2": "That gap is larger in the 45-60 band than below 30",
           "P4": "eICU hospitals with more Black patients sit farther away on CKD DAS",
           "P5": "Obesity documentation rises with BMI class",
           "P6": "Hospitals differ in obesity documentation at the same BMI",
           "P7": "Documentation rises with lab-defined stage",
           "P8": "Hospitals differ more at the mild stage than at the severe stage",
           "P9": "A hospital's documentation effects agree across conditions",
           "P10": "One shared factor explains the hospital effects",
           "P12": "Hospitals with an extreme habit sit farther away on DAS"}
    rows = []
    for _, r in pd.concat([pc, sh]).iterrows():
        cond = r.test.split(":")[0] if r.prediction in ("P7", "P8") else ""
        rows.append(dict(prediction=r.prediction, condition=cond, question=ask[r.prediction],
                         data=r.get("data", "eICU") if isinstance(r.get("data"), str) else "eICU",
                         measure=r.test.split(": ")[-1], estimate=r.estimate, lo=r.lo, hi=r.hi,
                         p_holm=r.p_holm, verdict="pass" if r.passed == 1 else "fail"))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "screen_highlights.csv", index=False)
    return t


def fig_severity():
    d = pd.read_csv(RES / "positive_controls_descriptive.csv")
    pc = pd.read_csv(RES / "positive_controls.csv")
    trend = {"obesity": pc[pc.prediction == "P5"].estimate.iloc[0]}
    for c in ["aki", "hyponatremia", "anemia", "thrombocytopenia"]:
        trend[c] = pc[(pc.prediction == "P7") & pc.test.str.startswith(c)].estimate.iloc[0]
    fig, axs = plt.subplots(1, 5, figsize=(11, 2.9), sharey=True)
    for ax, c in zip(axs, NAME):
        s = d[(d.block.isin(["B6", "B7"])) & (d.group == c)]
        y = s.documented.to_numpy() * 100
        lab = ["BMI 30-35", "35-40", "40+"] if c == "obesity" else ["Stage 1", "2", "3"]
        ax.bar(range(3), y, color=STAGE, width=0.7, edgecolor="white", linewidth=2)
        for i, v in enumerate(y):
            ax.text(i, v + 1.2, f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(3), lab)
        ax.set_title(f"{NAME[c]}\ntrend OR {trend[c]:.1f}", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axs[0].set_ylabel("Patients with the condition\ndocumented (%)")
    axs[0].set_ylim(0, 58)
    fig.suptitle("Documentation rises with objective severity, but most patients stay undocumented (eICU)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_screen_severity.png", dpi=200)
    plt.close(fig)


def fig_race_egfr():
    d = pd.read_csv(RES / "positive_controls_descriptive.csv")
    d = d[d.block == "B5"]
    pc = pd.read_csv(RES / "positive_controls.csv")
    orp = pc[pc.prediction == "P1"].set_index("data")
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
    for ax, data in zip(axs, ["MIMIC-IV 3.1", "eICU"]):
        s = d[d.data == data]
        for j, (lev, col) in enumerate([("non-Black", BLUE), ("Black", ORANGE)]):
            for i, band in enumerate(["band 45-60", "band <30"]):
                r = s[(s.group == band) & (s.level == lev)].iloc[0]
                x = i + (j - 0.5) * 0.36
                ax.bar(x, r.documented * 100, width=0.34, color=col, edgecolor="white", linewidth=2,
                       label=lev if i == 0 else None)
                ax.text(x, r.documented * 100 + 1.5, f"{r.documented * 100:.0f}%", ha="center", fontsize=8)
        o = orp.loc[data]
        ax.set_xticks([0, 1], ["eGFR 45-60\n(race term flips stage)", "eGFR <30\n(control band)"])
        ax.set_title(f"{data}\nBlack vs non-Black at 45-60: OR {o.estimate:.2f} ({o.lo:.2f}-{o.hi:.2f})",
                     fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axs[0].set_ylabel("CKD documented (%)")
    axs[0].set_ylim(0, 105)
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Race-based eGFR control: predicted lower documentation for Black patients, not seen",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_screen_race_egfr.png", dpi=200)
    plt.close(fig)


def habit_values():
    e = pd.read_parquet(s15.ROW / "eicu_positive_controls.parquet")
    fr = s16.frames(e)
    elig = set.intersection(*[set(c[c >= s16.MIN_STAYS].index) for c in (d.hospitalid.value_counts() for d in fr.values())])
    U = pd.DataFrame({c: s16.hospital_modes(d, ["sev", "age10", "female"])[0] for c, d in fr.items()}).loc[sorted(elig)]
    Z = (U - U.mean()) / U.std()
    _, load = s16.pc1(Z.to_numpy())
    h = pd.DataFrame(dict(pc1=Z.to_numpy() @ load), index=U.index)
    con = duckdb.connect()
    con.register("ids", e[["patientunitstayid", "hospitalid"]])
    dx = con.sql(f"""select i.hospitalid, count(d.diagnosisid) * 1.0 / count(distinct i.patientunitstayid) dx
                     from ids i left join '{s15.E}/diagnosis.parquet' d using (patientunitstayid)
                     group by 1""").df().set_index("hospitalid").dx
    h["dx_per_stay"] = dx.reindex(h.index)
    h["das"] = s16.das_published(set(e.patientunitstayid)).reindex(h.index)
    h.to_parquet(s15.ROW / "eicu_habit_hospital_values.parquet")
    return h


def fig_habit(h):
    corr = pd.read_csv(RES / "shared_habit_correlations.csv", index_col=0)
    sh = pd.read_csv(RES / "shared_habit.csv")
    sh = sh[(sh.version == "primary") & sh.in_holm].set_index("prediction")
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw=dict(width_ratios=[1.25, 0.7, 1, 1]))
    ax = axs[0]
    m = corr.to_numpy()
    im = ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#cde2fb", "#104281"]),
                   vmin=0, vmax=1)
    lab = [NAME[c].replace("Acute kidney injury", "AKI").replace("Thrombocytopenia", "Low platelets") for c in corr.index]
    ax.set_xticks(range(5), lab, rotation=40, ha="right")
    ax.set_yticks(range(5), lab)
    for i in range(5):
        for j in range(5):
            if i != j:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if m[i, j] > 0.6 else INK)
    ax.set_title(f"a  Hospital effects agree\nmedian rho {sh.loc['P9'].estimate:.2f} "
                 f"({sh.loc['P9'].lo:.2f}-{sh.loc['P9'].hi:.2f})", fontsize=9, loc="left")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax = axs[1]
    p10 = sh.loc["P10"]
    ax.bar([0, 1], [p10.null_median * 100, p10.estimate * 100], color=[GRID, BLUE], width=0.6)
    ax.errorbar(0, p10.null_median * 100, yerr=[[(p10.null_median - p10.lo) * 100], [(p10.hi - p10.null_median) * 100]],
                color=MUTED, capsize=4, linewidth=1)
    ax.text(1, p10.estimate * 100 + 2, f"{p10.estimate * 100:.0f}%", ha="center", fontsize=8)
    ax.text(0, p10.hi * 100 + 2, f"{p10.null_median * 100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["Chance\n(permuted)", "Observed"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Variance explained by one factor (%)")
    ax.set_title("b  One shared factor", fontsize=9, loc="left")
    ax.tick_params(length=0)
    for ax, x, xl, title in [(axs[2], h.dx_per_stay, "Diagnosis rows per stay", "c  It tracks problem-list fill"),
                             (axs[3], h.pc1, "Habit score (higher = documents more)", "d  It moves hospitals on DAS")]:
        y = h.pc1 if ax is axs[2] else h.das
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=22, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.8)
        rho = stats.spearmanr(x[ok], y[ok]).statistic
        ax.set_xlabel(xl)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        extra = f"\n|score| vs DAS rho {sh.loc['P12'].estimate:.2f} (tested)" if ax is axs[3] else ""
        ax.set_title(f"{title}\nrho {rho:.2f}{extra}", fontsize=9, loc="left")
    axs[2].set_xscale("log")
    axs[2].set_ylabel("Habit score")
    axs[3].set_ylabel("Median DHM angle to pooled eICU (deg)")
    fig.suptitle(f"Hospitals share one documentation habit across conditions ({len(h)} eICU hospitals)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_screen_habit.png", dpi=200)
    plt.close(fig)


def main():
    t = highlights()
    pd.set_option("display.width", 220)
    print(t.round(4).to_string())
    fig_severity()
    fig_race_egfr()
    fig_habit(habit_values())


if __name__ == "__main__":
    main()
