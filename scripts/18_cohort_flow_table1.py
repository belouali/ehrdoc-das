#!/usr/bin/env python
"""Cohort flow diagram and Table 1.

Outputs (artifacts/general/):
  cohort_flow.csv                      -- attrition at each stage, per condition
  cohort_flow_<condition>.png          -- flow diagram for the primary condition
  table1_<condition>.csv               -- patient characteristics by DAS quartile (tableone)
  hospital_characteristics_by_quartile_<condition>.csv
                                       -- region / bed size / teaching / medication availability by DAS quartile

Prereqs: scripts 01-03.

Usage:
  python scripts/18_cohort_flow_table1.py --condition diabetes
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ehrdoc.data.cohort import load_base_cohort, attach_das, analytic_sample
from ehrdoc.utils.io import ensure_dir
from ehrdoc.venn.vectors import hospital_venn_counts, source_shares


def _dist_csv(processed: Path, condition: str) -> Path:
    cand = sorted(processed.glob(f"dist_{condition}_min*.csv"), key=lambda p: p.stat().st_mtime)
    if not cand:
        raise FileNotFoundError(f"No dist_{condition}_min*.csv in {processed}")
    return cand[-1]


def simple_table1(df: pd.DataFrame, group_col: str, continuous: list, categorical: list) -> pd.DataFrame:
    """Table 1: overall and by group. Continuous as mean (SD) or median [IQR];
    categorical as n (%). Deterministic and dependency-free."""
    groups = ["Overall"] + [str(g) for g in sorted(df[group_col].dropna().unique())]
    subsets = {"Overall": df}
    subsets.update({str(g): df[df[group_col] == g] for g in sorted(df[group_col].dropna().unique())})
    rows = [{"Variable": "n (ICU stays)", "Level": "", **{g: f"{len(s):,}" for g, s in subsets.items()}},
            {"Variable": "Hospitals", "Level": "", **{g: f"{s['hospitalid'].nunique()}" for g, s in subsets.items()}}]
    for col, label, how in continuous:
        row = {"Variable": label, "Level": ""}
        for g, s in subsets.items():
            v = s[col].dropna()
            if how == "median":
                row[g] = f"{v.median():.0f} [{v.quantile(.25):.0f}-{v.quantile(.75):.0f}]"
            else:
                row[g] = f"{v.mean():.1f} ({v.std():.1f})"
        rows.append(row)
    for col, label in categorical:
        levels = df[col].value_counts().index.tolist()
        for i, lvl in enumerate(levels):
            row = {"Variable": label if i == 0 else "", "Level": str(lvl)}
            for g, s in subsets.items():
                n = int((s[col] == lvl).sum())
                row[g] = f"{n:,} ({100 * n / max(len(s), 1):.1f}%)"
            rows.append(row)
    return pd.DataFrame(rows)[["Variable", "Level"] + groups]


def flow_figure(flow: list[dict], condition: str, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 0.95 * len(flow) + 1), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(0, len(flow) + 0.5); ax.axis("off")
    y = len(flow)
    prev = None
    for st in flow:
        txt = f"{st['stage']}\n{st['n_stays']:,} ICU stays"
        if pd.notna(st.get("n_patients")):
            txt += f" | {int(st['n_patients']):,} unique patients"
        txt += f" | {st['n_hospitals']} hospitals"
        ax.text(3.5, y, txt, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3fb", edgecolor="#2a78d6"))
        if prev is not None:
            ax.annotate("", xy=(3.5, y + 0.42), xytext=(3.5, y + 0.58),
                        arrowprops=dict(arrowstyle="->", color="#52514e"))
            dropped = prev["n_stays"] - st["n_stays"]
            dh = prev["n_hospitals"] - st["n_hospitals"]
            ax.text(7.2, y + 0.5, f"excluded: {dropped:,} stays, {dh} hospitals",
                    ha="left", va="center", fontsize=7.5, color="#52514e")
        prev = st
        y -= 1
    ax.set_title(f"Cohort flow ({condition})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--cohorts", default="configs/cohorts.yaml")
    ap.add_argument("--conditions", nargs="+", default=["diabetes", "hypertension", "chf", "af"])
    ap.add_argument("--condition", default="diabetes", help="Primary condition for the figure and Table 1")
    ap.add_argument("--ref-hospital", type=int, default=73)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    with open(args.paths) as f:
        cfg = yaml.safe_load(f)
    with open(args.cohorts) as f:
        thresholds = (yaml.safe_load(f) or {}).get("phenotype_thresholds", {})
    processed = Path(cfg["outputs"]["processed"])
    general = ensure_dir(Path(cfg["outputs"]["artifacts"]) / "general")
    e = cfg["eicu"]
    e.setdefault("apachePatientResult", "data/raw/apachePatientResult.csv.gz")

    cohort, base_flow = load_base_cohort(e, return_flow=True)
    all_rows = []
    for cond in args.conditions:
        k = int(thresholds.get(cond, 100) or 100)
        flow = [dict(condition=cond, **st) for st in base_flow]
        merged = attach_das(cohort, _dist_csv(processed, cond), args.ref_hospital)
        flow.append({"condition": cond, "stage": f"In hospitals with >= {k} stays with {cond} evidence (DAS computable)",
                     "n_stays": len(merged), "n_patients": merged["uniquepid"].nunique() if "uniquepid" in merged else np.nan,
                     "n_hospitals": merged["hospitalid"].nunique()})
        cc, info = analytic_sample(merged, drop_no_variation=False)
        flow.append({"condition": cond, "stage": "Complete model covariates",
                     "n_stays": len(cc), "n_patients": cc["uniquepid"].nunique() if "uniquepid" in cc else np.nan,
                     "n_hospitals": cc["hospitalid"].nunique()})
        an, info = analytic_sample(merged, drop_no_variation=True)
        flow.append({"condition": cond, "stage": "Hospitals with outcome variation (analytic sample)",
                     "n_stays": len(an), "n_patients": an["uniquepid"].nunique() if "uniquepid" in an else np.nan,
                     "n_hospitals": an["hospitalid"].nunique()})
        all_rows.extend(flow)
        for st in flow:
            print(f"  [{cond}] {st['stage']}: {st['n_stays']:,} stays, {st['n_hospitals']} hospitals")

        if cond == args.condition:
            flow_figure(flow, cond, general / f"cohort_flow_{cond}.png")

            # ── Table 1 (patient level, by DAS quartile) ──
            an = an.copy()
            an["Angular_Separation_Quartile"] = pd.qcut(an["Angular_Separation"], 4,
                                                        labels=["q1", "q2", "q3", "q4"], duplicates="drop")
            an["mortality"] = an["actualhospitalmortality"].map({0: "Survived", 1: "Died"})
            t1 = simple_table1(
                an, group_col="Angular_Separation_Quartile",
                continuous=[("agenum", "Age, years, mean (SD)", "mean"),
                            ("apachescore", "APACHE IVa score, median [IQR]", "median")],
                categorical=[("gender", "Sex"), ("ethnicity", "Race/ethnicity"),
                             ("unitstaytype", "Unit stay type"), ("unitadmitsource", "Admission source"),
                             ("mortality", "Hospital mortality")])
            t1.to_csv(general / f"table1_{cond}.csv", index=False)
            print(f"  Wrote {general / f'table1_{cond}.csv'}")

            # ── Hospital characteristics by DAS quartile ──
            flags = pd.read_parquet(processed / f"flags_{cond}.parquet")
            counts = hospital_venn_counts(flags, min_patients_per_hospital=k)
            shares = source_shares(counts)
            hosp = an.groupby("hospitalid").agg(
                n_stays=("patientunitstayid", "size"), DAS=("Angular_Separation", "first"),
                quartile=("Angular_Separation_Quartile", "first"), region=("region", "first"),
                beds=("numbedscategory", "first"), teaching=("teachingstatus", "first"),
                mortality=("actualhospitalmortality", "mean")).reset_index()
            hosp = hosp.merge(shares, on="hospitalid", how="left")
            hosp["no_medication_records"] = hosp["share_Med"] == 0
            rows = []
            for q, g in hosp.groupby("quartile"):
                row = {"DAS_quartile": q, "n_hospitals": len(g), "DAS_range": f"{g['DAS'].min():.1f}-{g['DAS'].max():.1f}",
                       "median_stays": int(g["n_stays"].median()), "mean_mortality": round(g["mortality"].mean(), 3),
                       "no_medication_records": int(g["no_medication_records"].sum()),
                       "teaching": int((g["teaching"].astype(str).str.lower().isin(["true", "t", "1", "yes"])).sum())}
                for col, prefix in (("region", "region"), ("beds", "beds")):
                    for lvl, n in g[col].value_counts().items():
                        row[f"{prefix}: {lvl}"] = int(n)
                rows.append(row)
            pd.DataFrame(rows).fillna(0).to_csv(general / f"hospital_characteristics_by_quartile_{cond}.csv", index=False)
            print(f"  Wrote {general / f'hospital_characteristics_by_quartile_{cond}.csv'}")

    pd.DataFrame(all_rows).to_csv(general / "cohort_flow.csv", index=False)
    print(f"Wrote {general / 'cohort_flow.csv'}")


if __name__ == "__main__":
    main()
