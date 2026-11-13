#!/usr/bin/env python
"""Aggregate inputs for the rebuilt JAMIA Open manuscript (S_site_selection_aligned and the MIMIC-IV stay tests).

Reads only aggregate results, except the cohort block, which counts stays and events per task from the
row-level eICU files and writes counts only. Writes to ../artifacts/aligned/:
  site_selection_summary.csv   copy of results/site_selection_summary.csv (script 33)
  site_selection_cohort.csv    per task: stays, hospitals, event rate, targets per half, pool size, training stays
  mimic_stay_tests.csv         CHF nearest 10 in MIMIC-IV for both stay tasks (scripts 24-26), one row per test
  mimic_mortality.csv          nearest 10 vs random 10 for death in MIMIC-IV (stage 4, P0 and M1, DHM)
  los_practice_summary.csv     results/los_practice_summary.csv (script 27) plus the membership-nearest set
  membership_eicu_summary.csv  copy of results/membership_eicu_summary.csv (script 35)
  mimic_angles.csv             MIMIC-IV nearest and farthest 10 by every angle and by membership, three outcomes
  los_practice_hospitals.csv   hospital-level angle and O/E columns of results/los_practice_hospitals.csv
  figure3_site_selection.png   old layout (both halves, CHF point), kept for comparison
  figure3_site_selection_v2.png  held-out half, nearest and farthest by angle and membership, MIMIC-IV diamonds
  figure4_los_practice.png     observed/expected ICU stay > 7 days against the CHF angle to MIMIC-IV

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with matplotlib --with scipy \
      python scripts/34_paper_inputs.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEV, HELD, MIMIC = "#9aa5b1", "#2a78d6", "#eb6834"
TASKS = {"mortality": "In-hospital death", "A3_los_gt7d": "ICU stay > 7 days",
         "A6_hospital_los_gt14d": "Hospital stay > 14 days"}
ARMS = {"near_diabetes": "Diabetes", "near_hypertension": "Hypertension", "near_chf": "CHF", "near_af": "AF",
        "near_joined": "Joined (mean angle)", "casemix": "Case mix"}

MSFX = {"mortality": "_death", "A3_los_gt7d": "", "A6_hospital_los_gt14d": "_a6"}
MTASK = {"mortality": "death", "A3_los_gt7d": "los_gt7d", "A6_hospital_los_gt14d": "hosp_los_gt14d"}


def cohort(cfg, res):
    rowlevel, E = Path(cfg["paths"]["rowlevel"]), cfg["paths"]["eicu"]
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet", columns=["patientunitstayid", "hospitalid", "death"])
    pt = duckdb.execute(f"""SELECT patientunitstayid, unitdischargeoffset / 1440.0 AS los_d,
                                   (hospitaldischargeoffset - hospitaladmitoffset) / 1440.0 AS hosp_los_d
                            FROM '{E}/patient.parquet'""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    y = {"mortality": e.death.astype(int), "A3_los_gt7d": (e.los_d > 7).astype(int),
         "A6_hospital_los_gt14d": (e.hosp_los_d > 14).astype(int)}
    split = pd.read_csv(res / "target_split.csv").set_index("hospitalid").half
    tg = pd.read_csv(res / "site_selection_targets.csv")
    tg = tg[(tg["sample"] == "all") & (tg.k == 10)]
    no_med = pd.read_parquet(rowlevel / "eicu_flags.parquet", columns=["hospitalid", "any_M"]) \
        .groupby("hospitalid").any_M.max().eq(0).sum()
    rows = []
    for t, yy in y.items():
        g = pd.DataFrame({"h": e.hospitalid, "y": yy}).groupby("h").y.agg(["size", "sum"])
        pool = g[(g["size"] >= 100) & (g["sum"] > 0) & (g["sum"] < g["size"])]
        s = tg[tg.task == t]
        rows.append(dict(task=t, stays=len(e), hospitals=e.hospitalid.nunique(), events=int(yy.sum()),
                         event_rate=yy.mean(), pool_hospitals=len(pool), targets=len(split),
                         targets_scored=len(s), targets_development=int((s.half == "development").sum()),
                         targets_confirmation=int((s.half == "confirmation").sum()),
                         train_stays_median=s.n_train.median(), no_medication_hospitals=int(no_med)))
    return pd.DataFrame(rows)


def mimic_tests(res):
    rows = []
    for t, sfx in MSFX.items():
        m = pd.read_csv(res / f"w1_mimic{sfx}.csv").set_index("arm")
        p = pd.read_csv(res / f"w1_pooled{sfx}.csv").set_index("arm")
        x = pd.read_csv(res / f"w1_proxy{sfx}.csv").dropna(subset=["test"]).set_index("test")
        n = m.loc["nearest"]
        rows.append(dict(task=t, test="vs_random10", auroc=n.auroc, comparator_auroc=n.random_mean,
                         gain=n.gain_vs_random, lo=n.lo, hi=n.hi, percentile=n.percentile_among_random,
                         citl=n.citl, slope=n.cal_slope, n_train=n.n_train))
        c = m.loc["casemix_nearest"] if "casemix_nearest" in m.index else None
        if c is not None:
            rows.append(dict(task=t, test="casemix_vs_random10", auroc=c.auroc, comparator_auroc=c.random_mean,
                             gain=c.gain_vs_random, percentile=c.percentile_among_random, citl=c.citl,
                             slope=c.cal_slope, n_train=c.n_train))
        a, b = p.loc["nearest10"], p.loc["pooled_all"]
        rows.append(dict(task=t, test="vs_pooled_all", auroc=a.auroc, comparator_auroc=b.auroc,
                         gain=a.minus_pooled_all, lo=a.lo, hi=a.hi, citl=a.citl, slope=a.cal_slope,
                         n_train=a.n_train, comparator_n_train=b.n_train, comparator_citl=b.citl,
                         comparator_slope=b.cal_slope))
        for test, name in (("T1", "vs_type_matched"), ("T2", "non_northeast_vs_random10")):
            r = x.loc[test]
            rows.append(dict(task=t, test=name, auroc=r.auroc, comparator_auroc=r.comparator_mean, gain=r.gain,
                             lo=r.lo, hi=r.hi, percentile=r.percentile_among_draws, citl=r.citl,
                             slope=r.cal_slope, n_train=r.n_train))
    return pd.DataFrame(rows)


def mimic_mortality(res):
    out = []
    for model, f in (("P0", "stage4_DHM_contrasts.csv"), ("M1", "stage4_M1_DHM_contrasts.csv")):
        c = pd.read_csv(res / f)
        c = c[(c.version == "size_controlled") & (c.config == "main")]
        c = c[["condition", "k", "auroc_nearest", "auroc_random_mean", "auroc_diff", "auroc_diff_lo",
               "auroc_diff_hi", "auroc_pct_beaten"]].assign(model=model)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def figure3(sm, mt, path):
    sm = sm[(sm["sample"] == "all") & (sm.k == 10)]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9), sharey=True)
    arms = list(ARMS)
    for ax, (t, lab) in zip(axes, TASKS.items()):
        for i, a in enumerate(arms):
            yv = len(arms) - 1 - i
            for half, col, dy, mk in (("development", DEV, 0.17, "o"), ("confirmation", HELD, -0.17, "o")):
                r = sm[(sm.task == t) & (sm.arm == a) & (sm.half == half)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([r.lo, r.hi], [yv + dy] * 2, color=col, lw=1.4, solid_capstyle="round")
                ax.plot(r.gain_vs_random, yv + dy, mk, ms=4.5, color=col,
                        mfc=col if half == "confirmation" else "white", mew=1.2)
            if a == "near_chf" and t != "mortality":
                m = mt[(mt.task == t) & (mt.test == "vs_random10")].iloc[0]
                ax.plot(m.gain, yv, "D", ms=5, color=MIMIC, mec="white", mew=0.6, zorder=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(lab, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors=INK, length=2)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-0.012, 0.052)
        ax.set_xticks([0, 0.02, 0.04])
    axes[0].set_yticks(range(len(arms)), [ARMS[a] for a in reversed(arms)], fontsize=7.5)
    axes[1].set_xlabel("AUROC gain over 10 random hospitals", fontsize=8, color=INK)
    h = [plt.Line2D([], [], marker="o", color=DEV, mfc="white", lw=1.4, ms=4.5, label="eICU development half"),
         plt.Line2D([], [], marker="o", color=HELD, lw=1.4, ms=4.5, label="eICU held-out half"),
         plt.Line2D([], [], marker="D", color=MIMIC, lw=0, ms=5, label="MIMIC-IV (CHF angle)")]
    fig.legend(handles=h, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


NEAR_C, FAR_C, CM_C, MIM_C = "#5b4ea3", "#a87a0c", "#4a4a4a", "#1a1a1a"  # neutral pair, validated with the dataviz palette checker




def mimic_angles(res):
    """MIMIC-IV nearest and farthest 10 by every angle and by membership, for the three outcomes (W_mimic_aligned,
    S_membership_comparator)."""
    rows = []
    for t, sfx in MSFX.items():
        for key in ("diabetes", "hypertension", "chf", "af", "combined"):
            m = pd.read_csv(res / f"w1_mimic{sfx}{'' if key == 'chf' else '_' + key}.csv").set_index("arm")
            for arm in ("nearest", "farthest"):
                r = m.loc[arm]
                rows.append(dict(task=t, angle=key, arm=arm, auroc=r.auroc, random_mean=r.random_mean,
                                 gain=r.gain_vs_random, lo=r.lo, hi=r.hi, percentile=r.percentile_among_random,
                                 citl=r.citl, slope=r.cal_slope, n_train=r.n_train))
        mb = pd.read_csv(res / "membership_mimic.csv")
        for a, arm in (("memb_near", "nearest"), ("memb_far", "farthest"), ("dasmemb_near", "das_membership_nearest")):
            r = mb[(mb.task == MTASK[t]) & (mb.arm == a)].iloc[0]
            rows.append(dict(task=t, angle="membership", arm=arm, auroc=r.auroc, random_mean=r.random_mean,
                             gain=r.gain_vs_random, lo=r.lo, hi=r.hi, percentile=r.percentile_among_random,
                             citl=r.citl, slope=r.cal_slope, n_train=r.n_train))
    return pd.DataFrame(rows)


def membership_los(hosp, res):
    """Stay-weighted O/E of the 10 membership-nearest hospitals to MIMIC-IV, and Spearman of the membership
    c-statistic with the O/E gap (same definitions as scripts/27)."""
    from scipy.stats import spearmanr
    c = pd.read_csv(res / "membership_mimic_relatedness.csv").set_index("hospitalid").c_statistic
    h = hosp.set_index("hospitalid").join(c, how="inner").dropna(subset=["oe"])
    s = h.loc[h.c_statistic.sort_values(kind="stable").index[:10]]
    rho, p = spearmanr(h.c_statistic, h.dist_logoe)
    rel = pd.read_csv(res / "membership_mimic_relatedness.csv")
    r2, p2 = spearmanr(rel.c_statistic, rel.das_combined)
    return pd.DataFrame([dict(item="membership_nearest10", hospitals=len(s),
                              observed=(s.observed * s.stays).sum() / s.stays.sum(),
                              oe=(s.observed * s.stays).sum() / (s.expected * s.stays).sum()),
                         dict(item="spearman_membership_c_vs_oe_distance", hospitals=len(h), rho=rho, p=p),
                         dict(item="spearman_membership_c_vs_combined_angle", hospitals=len(rel), rho=r2, p=p2,
                              c_median=rel.c_statistic.median(), c_min=rel.c_statistic.min(), c_max=rel.c_statistic.max())])


def figure3_v2(sm, mb, ma, path):
    """Held-out half, k = 10: nearest (purple) and farthest (ochre) set per angle and by membership; filled circles
    eICU with 95% target-bootstrap intervals, open diamonds MIMIC-IV."""
    sm = sm[(sm["sample"] == "all") & (sm.k == 10) & (sm.half == "confirmation")]
    mb = mb[mb.half == "confirmation"]
    rows = [("diabetes", "Diabetes"), ("hypertension", "Hypertension"), ("chf", "CHF"), ("af", "AF"),
            ("joined", "Combined angle"), ("membership", "Membership model")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.1), sharey=True)
    for ax, (t, lab) in zip(axes, TASKS.items()):
        for i, (key, _) in enumerate(rows):
            yv = len(rows) - 1 - i
            for arm, col, dy in (("near", NEAR_C, 0.13), ("far", FAR_C, -0.13)):
                if key == "membership":
                    r = mb[(mb.task == t) & (mb.contrast == f"memb_{arm}")].iloc[0]
                    g, lo, hi = r.estimate, r.lo, r.hi
                else:
                    r = sm[(sm.task == t) & (sm.arm == f"{arm}_{key}")].iloc[0]
                    g, lo, hi = r.gain_vs_random, r.lo, r.hi
                ax.plot([lo, hi], [yv + dy] * 2, color=col, lw=1.5, solid_capstyle="round", zorder=2)
                ax.plot(g, yv + dy, "o", ms=4.2, color=col, mew=1.2, zorder=3)
                mk = "combined" if key == "joined" else key
                m = ma[(ma.task == t) & (ma.angle == mk) & (ma.arm == ("nearest" if arm == "near" else "farthest"))]
                ax.plot(m.gain.iloc[0], yv + dy, "D", ms=4.4, color=col, mfc="white", mew=1.2, zorder=4)
        ax.axvline(0, color=INK, lw=0.9, zorder=1)
        ax.set_title(lab, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors=INK, length=2)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(-0.18, 0.075)
        ax.set_xticks([-0.15, -0.10, -0.05, 0, 0.05], ["−0.15", "−0.10", "−0.05", "0", "+0.05"], fontsize=6)
        ax.text(0.002, len(rows) - 0.45, "random 10", fontsize=6, color=MUTED, va="bottom")
    axes[0].set_yticks(range(len(rows)), [l for _, l in reversed(rows)], fontsize=7.5)
    axes[0].set_ylim(-0.6, len(rows) - 0.2)
    axes[1].set_xlabel("AUROC difference from 10 random hospitals", fontsize=8, color=INK)
    h = [plt.Line2D([], [], marker="o", color=NEAR_C, lw=1.5, ms=4.2, label="10 nearest, eICU held-out half"),
         plt.Line2D([], [], marker="o", color=FAR_C, lw=1.5, ms=4.2, label="10 farthest, eICU held-out half"),
         plt.Line2D([], [], marker="D", color=NEAR_C, mfc="white", mew=1.2, lw=0, ms=4.4, label="10 nearest, MIMIC-IV"),
         plt.Line2D([], [], marker="D", color=FAR_C, mfc="white", mew=1.2, lw=0, ms=4.4, label="10 farthest, MIMIC-IV")]
    fig.legend(handles=h, fontsize=7, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure4(hosp, summ, nearest_ids, path):
    s = summ.set_index("item")
    h = hosp.dropna(subset=["angle_chf", "oe"])
    near = h[h.hospitalid.isin(nearest_ids)]
    assert len(near) == 10
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    a.scatter(h.angle_chf, h.oe, s=9, color=DEV, edgecolor="none", label="eICU hospital")
    a.scatter(near.angle_chf, near.oe, s=16, color=NEAR_C, edgecolor="white", lw=0.4, label="10 nearest by CHF angle")
    a.axhline(s.loc["mimic_iv", "oe"], color=MIM_C, lw=1.1, ls="--", label=f"MIMIC-IV ({s.loc['mimic_iv', 'oe']:.2f})")
    a.axhline(1, color=MUTED, lw=0.6)
    a.set_yscale("log")
    a.set_yticks([0.1, 0.25, 0.5, 1, 2, 4], ["0.1", "0.25", "0.5", "1", "2", "4"])
    a.minorticks_off()
    a.set_xlabel("CHF angle to MIMIC-IV, degrees", fontsize=8, color=INK)
    a.set_ylabel("Observed / expected, ICU stay > 7 days", fontsize=8, color=INK)
    a.set_title("A", fontsize=9, color=INK, loc="left", fontweight="bold")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")
    items = [("mimic_iv", "MIMIC-IV"), ("chf_nearest10", "CHF nearest 10"), ("diabetes_nearest10", "Diabetes nearest 10"),
             ("chf_random10_mean", "Random 10 (mean)"), ("membership_nearest10", "Membership nearest 10")]
    vals = [s.loc[k, "oe"] for k, _ in items]
    cols = [MIM_C, NEAR_C, NEAR_C, DEV, DEV]
    b.barh(range(len(items))[::-1], vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        b.text(min(v, 1) + 0.04 if v < 1 else v + 0.02, len(items) - 1 - i, f"{v:.2f}", va="center", fontsize=7, color=INK,
               bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    b.set_yticks(range(len(items))[::-1], [l for _, l in items], fontsize=7)
    b.axvline(1, color=MUTED, lw=0.6)
    b.set_xlim(0, 1.25)
    b.set_xlabel("Observed / expected", fontsize=8, color=INK)
    b.set_title("B", fontsize=9, color=INK, loc="left", fontweight="bold")
    for ax in (a, b):
        ax.tick_params(labelsize=7, colors=INK, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    res = base / cfg["paths"]["results"]
    out = base.parent / "artifacts" / "aligned"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(res / "site_selection_summary.csv", out / "site_selection_summary.csv")
    hosp = pd.read_csv(res / "los_practice_hospitals.csv")
    lps = pd.concat([pd.read_csv(res / "los_practice_summary.csv"), membership_los(hosp, res)], ignore_index=True)
    lps.to_csv(out / "los_practice_summary.csv", index=False)
    hosp[["hospitalid", "stays", "oe", "outside_limits", "angle_diabetes", "angle_hypertension", "angle_chf",
          "angle_af", "casemix_distance"]].to_csv(out / "los_practice_hospitals.csv", index=False)
    cohort(cfg, res).round(5).to_csv(out / "site_selection_cohort.csv", index=False)
    mt = mimic_tests(res)
    mt.round(5).to_csv(out / "mimic_stay_tests.csv", index=False)
    mimic_mortality(res).round(5).to_csv(out / "mimic_mortality.csv", index=False)
    sm = pd.read_csv(res / "site_selection_summary.csv")
    figure3(sm, mt, out / "figure3_site_selection.png")
    mb = pd.read_csv(res / "membership_eicu_summary.csv")
    mb.to_csv(out / "membership_eicu_summary.csv", index=False)
    ma = mimic_angles(res)
    ma.round(5).to_csv(out / "mimic_angles.csv", index=False)
    figure3_v2(sm, mb, ma, out / "figure3_site_selection_v2.png")
    ids = [int(x) for x in pd.read_csv(res / "w1_mimic.csv").set_index("arm").loc["nearest", "hospitals"].split(";")]
    figure4(hosp, lps, ids, out / "figure4_los_practice.png")
    print(pd.read_csv(out / "site_selection_cohort.csv").to_string(index=False))
    print(mt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
