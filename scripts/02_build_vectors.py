#!/usr/bin/env python
"""Build hospital 7-lobe counts + normalized vectors for each condition.

Inputs (from 01_build_flags.py):
  data/processed/flags_<condition>.parquet

Outputs:
  data/processed/counts_<condition>_min<k>.parquet
  data/processed/vectors_<condition>_min<k>.parquet

Usage:
  python scripts/02_build_vectors.py --conditions diabetes hypertension --min-per-hospital 500
  python scripts/02_build_vectors.py --conditions all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ehrdoc.venn.vectors import hospital_venn_counts, hospital_venn_vectors
from ehrdoc.utils.io import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--cohorts", default="configs/cohorts.yaml")
    ap.add_argument("--conditions", nargs="+", default=["all"])
    ap.add_argument("--min-per-hospital", type=int, default=None,
                    help="Override per-condition threshold (else use configs/cohorts.yaml phenotype_thresholds)")
    args = ap.parse_args()

    with open(args.paths, "r") as f:
        paths_cfg = yaml.safe_load(f)
    processed = ensure_dir(paths_cfg["outputs"]["processed"])

    with open(args.cohorts, "r") as f:
        cohort_cfg = yaml.safe_load(f)
    thresholds = cohort_cfg.get("phenotype_thresholds", {})

    # infer available conditions from flags_* files
    available = sorted([p.name.replace("flags_", "").replace(".parquet", "")
                        for p in processed.glob("flags_*.parquet")])
    conditions = available if args.conditions == ["all"] else args.conditions

    for cond in conditions:
        flag_path = processed / f"flags_{cond}.parquet"
        if not flag_path.exists():
            raise FileNotFoundError(f"Missing {flag_path}. Run scripts/01_build_flags.py first.")

        k = args.min_per_hospital
        if k is None:
            k = int(thresholds.get(cond, 0) or 0)
        k = None if k == 0 else k

        flags = pd.read_parquet(flag_path)
        counts = hospital_venn_counts(flags, min_patients_per_hospital=k)
        vectors = hospital_venn_vectors(counts)

        suffix = f"_min{int(k)}" if k is not None else ""
        out_counts = processed / f"counts_{cond}{suffix}.parquet"
        out_vectors = processed / f"vectors_{cond}{suffix}.parquet"
        counts.to_parquet(out_counts, index=False)
        vectors.to_parquet(out_vectors, index=False)
        print(f"Wrote {out_counts} (n_hospitals={counts['hospitalid'].nunique():,})")
        print(f"Wrote {out_vectors} (n_hospitals={vectors['hospitalid'].nunique():,})")


if __name__ == "__main__":
    main()
