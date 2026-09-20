#!/usr/bin/env python
"""Compute hospital×hospital distance matrices for each condition.

Inputs (from 02_build_vectors.py):
  data/processed/vectors_<condition>_min<k>.parquet

Outputs:
  data/processed/dist_<condition>_min<k>.npz   (keys: D, ids)
  data/processed/dist_<condition>_min<k>.csv   (long-form table)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ehrdoc.distances.matrices import distance_matrix
from ehrdoc.utils.io import ensure_dir


def _to_long(D: np.ndarray, ids: list[int]) -> pd.DataFrame:
    rows = []
    n = len(ids)
    for i in range(n):
        for j in range(n):
            rows.append((ids[i], ids[j], float(D[i, j])))
    return pd.DataFrame(rows, columns=["hospital_1", "hospital_2", "angular_separation_deg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--conditions", nargs="+", default=["all"])
    ap.add_argument("--suffix", default=None,
                    help="Vector suffix like 'min500'. If omitted, uses the most recent vectors_* file per condition.")
    args = ap.parse_args()

    with open(args.paths, "r") as f:
        paths_cfg = yaml.safe_load(f)
    processed = ensure_dir(paths_cfg["outputs"]["processed"])

    # infer conditions from vectors files
    vec_files = sorted(processed.glob("vectors_*.parquet"))
    if not vec_files:
        raise FileNotFoundError("No vectors_*.parquet found. Run scripts/02_build_vectors.py first.")

    available = sorted({p.name.split("vectors_")[1].split(".parquet")[0].split("_min")[0] for p in vec_files})
    conditions = available if args.conditions == ["all"] else args.conditions

    for cond in conditions:
        # choose file
        if args.suffix:
            vec_path = processed / f"vectors_{cond}_{args.suffix}.parquet"
            if not vec_path.exists():
                vec_path = processed / f"vectors_{cond}_min{args.suffix.replace('min','')}.parquet"
        else:
            # pick the most recently modified vectors file for this condition
            cand = sorted(processed.glob(f"vectors_{cond}*.parquet"),
                          key=lambda p: p.stat().st_mtime)
            vec_path = cand[-1]  # most recent

        vectors = pd.read_parquet(vec_path)
        D, ids = distance_matrix(vectors)

        stem = vec_path.name.replace("vectors_", "dist_").replace(".parquet", "")
        out_npz = processed / f"{stem}.npz"
        out_csv = processed / f"{stem}.csv"
        np.savez_compressed(out_npz, D=D, ids=np.array(ids))
        _to_long(D, ids).to_csv(out_csv, index=False)
        print(f"Wrote {out_npz} (n_hospitals={len(ids):,})")
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
