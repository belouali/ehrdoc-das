#!/usr/bin/env python
"""Table 1: hospitals and patients by condition (columns = conditions).

Rows: hospitals with a documentation vector, hospitals with no medication
records, hospital descriptors (region, bed size, teaching), DAS summary
(pairwise and from the reference hospital), and the mortality-model analytic
sample (stays, patients, age, sex, APACHE IVa, race/ethnicity, transfers,
hospital mortality).

Writes artifacts/general/table1_by_condition.csv

Usage:
  python scripts/23_table1_by_condition.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ehrdoc.data.cohort import load_base_cohort, attach_das, analytic_sample
from ehrdoc.utils.io import ensure_dir
from ehrdoc.venn.vectors import hospital_venn_counts, zero_source_hospitals

COND = ["diabetes", "hypertension", "chf", "af"]
NICE = {"diabetes": "Diabetes", "hypertension": "Hypertension", "chf": "CHF", "af": "AF"}


def pct(n, d):
    return f"{n:,} ({100 * n / d:.1f}%)" if d else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--cohorts", default="configs/cohorts.yaml")
    ap.add_argument("--ref-hospital", type=int, default=73)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.paths))
    thresholds = (yaml.safe_load(open(args.cohorts)) or {}).get("phenotype_thresholds", {})
    processed = Path(cfg["outputs"]["processed"]); art = Path(cfg["outputs"]["artifacts"])
    e = cfg["eicu"]; e.setdefault("apachePatientResult", "data/raw/apachePatientResult.csv.gz")

    cohort = load_base_cohort(e)
    rows = {}  # row label -> {condition: value}
    order = []

    def put(label, cond, val):
        if label not in rows:
            rows[label] = {}; order.append(label)
        rows[label][cond] = val

    for c in COND:
        k = int(thresholds.get(c, 100) or 100)
        flags = pd.read_parquet(processed / f"flags_{c}.parquet")
        counts = hospital_venn_counts(flags, min_patients_per_hospital=k)
        hids = counts["hospitalid"].astype(int).tolist()
        zs = zero_source_hospitals(counts)
        n_h = len(hids)
        put("Hospitals with a documentation vector (≥100 stays with evidence)", c, f"{n_h}")
        put("  Stays with condition evidence, median [IQR] per hospital", c,
            f"{counts.drop(columns='hospitalid').sum(axis=1).median():.0f} [{counts.drop(columns='hospitalid').sum(axis=1).quantile(.25):.0f}-{counts.drop(columns='hospitalid').sum(axis=1).quantile(.75):.0f}]")
        put("  Hospitals with no medication records", c, pct(len(zs["Med"]), n_h))

        # hospital descriptors (one row per hospital)
        hc = flags.drop_duplicates("hospitalid").set_index("hospitalid").loc[hids]
        for reg in ["Midwest", "Northeast", "South", "West"]:
            put(f"  Region: {reg}", c, pct(int((hc["region"] == reg).sum()), n_h))
        put("  Region: not recorded", c, pct(int(hc["region"].isna().sum()), n_h))
        for b in ["<100", "100 - 249", "250 - 499", ">= 500"]:
            put(f"  Beds: {b.replace('>= ', '≥')}", c, pct(int((hc["numbedscategory"] == b).sum()), n_h))
        put("  Beds: not recorded", c, pct(int(hc["numbedscategory"].isna().sum()), n_h))
        put("  Teaching hospital", c, pct(int(hc["teachingstatus"].astype(str).str.lower().isin(["t", "true", "1", "yes"]).sum()), n_h))

        # DAS summaries
        npz = sorted(processed.glob(f"dist_{c}_min*.npz"), key=lambda p: p.stat().st_mtime)[-1]
        z = np.load(npz, allow_pickle=True); D = z["D"]; ids = [int(i) for i in z["ids"]]
        iu = np.triu_indices_from(D, 1); pw = D[iu]
        put("Pairwise DAS, degrees, median [IQR]", c, f"{np.median(pw):.1f} [{np.percentile(pw, 25):.1f}-{np.percentile(pw, 75):.1f}]")
        put("  Range", c, f"{pw.min():.1f}-{pw.max():.1f}")
        if args.ref_hospital in ids:
            r = D[ids.index(args.ref_hospital)]; r = np.delete(r, ids.index(args.ref_hospital))
            put(f"DAS from reference hospital {args.ref_hospital}, median [IQR]", c, f"{np.median(r):.1f} [{np.percentile(r, 25):.1f}-{np.percentile(r, 75):.1f}]")

        # analytic sample
        dist_csv = sorted(processed.glob(f"dist_{c}_min*.csv"), key=lambda p: p.stat().st_mtime)[-1]
        m = attach_das(cohort, dist_csv, args.ref_hospital)
        an, info = analytic_sample(m, drop_no_variation=True)
        n = len(an)
        put("Mortality-model sample: ICU stays", c, f"{n:,}")
        put("  Hospitals", c, f"{info['hospitals_analytic']}")
        put("  Unique patients", c, f"{an['uniquepid'].nunique():,}" if "uniquepid" in an else "")
        put("  Age, years, mean (SD)", c, f"{an['agenum'].mean():.1f} ({an['agenum'].std():.1f})")
        put("  Male", c, pct(int((an["gender"] == "Male").sum()), n))
        put("  APACHE IVa score, median [IQR]", c, f"{an['apachescore'].median():.0f} [{an['apachescore'].quantile(.25):.0f}-{an['apachescore'].quantile(.75):.0f}]")
        put("  Race/ethnicity: Caucasian", c, pct(int((an["ethnicity"] == "Caucasian").sum()), n))
        put("  Race/ethnicity: African American", c, pct(int((an["ethnicity"] == "African American").sum()), n))
        put("  Race/ethnicity: Hispanic", c, pct(int((an["ethnicity"] == "Hispanic").sum()), n))
        put("  Race/ethnicity: Asian", c, pct(int((an["ethnicity"] == "Asian").sum()), n))
        put("  Race/ethnicity: other or unknown", c, pct(int((~an["ethnicity"].isin(["Caucasian", "African American", "Hispanic", "Asian"])).sum()), n))
        put("  Unit stay type: transfer", c, pct(int((an["unitstaytype"] == "transfer").sum()), n))
        put("  Unit stay type: readmission", c, pct(int((an["unitstaytype"] == "readmit").sum()), n))
        put("  Admission source: emergency department", c, pct(int((an["unitadmitsource"] == "Emergency Department").sum()), n))
        put("  Admission source: operating room", c, pct(int((an["unitadmitsource"] == "Operating Room").sum()), n))
        put("  Hospital mortality", c, pct(int(an["actualhospitalmortality"].sum()), n))
        print(f"{c}: {n_h} hospitals, {n:,} stays")

    out = pd.DataFrame([{"Characteristic": lab, **{NICE[c]: rows[lab].get(c, "") for c in COND}} for lab in order])
    p = ensure_dir(art / "general") / "table1_by_condition.csv"
    out.to_csv(p, index=False); print("wrote", p)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
