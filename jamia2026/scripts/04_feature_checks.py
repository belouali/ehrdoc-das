#!/usr/bin/env python
"""Stage 3 checks: are the P0 predictors comparable, and does P0 work in eICU?

  1. Coverage and distribution of every predictor in eICU and MIMIC-IV
     (MIMIC-IV reference and evaluation subsets separately; no MIMIC-IV outcome is read).
  2. eICU hospitals with little or no coverage of each measurement.
  3. eICU internal performance of P0 with hospital-grouped 5-fold cross-validation,
     over a small grid of L2 penalties, with the APACHE IVa prediction on the same
     stays as a benchmark. The chosen penalty is written to results/p0_locked.yaml.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      python scripts/04_feature_checks.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_model import LABS, VITALS, feature_columns, make_pipeline, metrics  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0]


def describe(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    d = df[cols]
    return pd.DataFrame({"dataset": label, "feature": cols,
                         "coverage": d.notna().mean().to_numpy(),
                         "median": d.median().to_numpy(),
                         "q1": d.quantile(0.25).to_numpy(), "q3": d.quantile(0.75).to_numpy(),
                         "mean": d.mean().to_numpy(), "sd": d.std().to_numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    cols = feature_columns()

    e = pd.read_parquet(rowlevel / "eicu_features.parquet")
    m = pd.read_parquet(rowlevel / "mimic_features.parquet")
    split = pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]]
    m = m.merge(split, on="stay_id")
    assert "death" not in m.columns and "hospital_expire_flag" not in m.columns

    # 1. coverage and distributions
    desc = pd.concat([describe(e, cols, "eICU"),
                      describe(m[m.reference], cols, "MIMIC-IV reference"),
                      describe(m[~m.reference], cols, "MIMIC-IV evaluation")])
    wide = desc.pivot(index="feature", columns="dataset", values=["coverage", "median"])
    wide.columns = [f"{a}|{b}" for a, b in wide.columns]
    smd = ((e[cols].mean() - m.loc[~m.reference, cols].mean())
           / np.sqrt((e[cols].var() + m.loc[~m.reference, cols].var()) / 2)).rename("smd_eicu_vs_eval")
    desc.round(4).to_csv(res / "p0_feature_distributions.csv", index=False)
    print(wide.round(2).join(smd.round(2)).loc[cols].to_string())

    # 2. hospital-level coverage in eICU
    measures = {v: f"{v}_median" for v in VITALS} | {"gcs": "gcs_min"} | {l: f"{l}_min" for l in LABS}
    hosp = e.groupby("hospitalid")[[c for c in measures.values()]].apply(lambda g: g.notna().mean())
    hosp.columns = list(measures)
    hcov = pd.DataFrame({"measure": hosp.columns,
                         "hospitals_zero": (hosp == 0).sum().to_numpy(),
                         "hospitals_below_50pct": (hosp < 0.5).sum().to_numpy(),
                         "median_hospital_coverage": hosp.median().to_numpy()})
    hcov.round(3).to_csv(res / "p0_hospital_coverage_eicu.csv", index=False)
    print(hcov.round(3).to_string(index=False))

    # 3. eICU internal performance, hospital-grouped CV
    d = e[e.death.notna()].reset_index(drop=True)
    E = cfg["paths"]["eicu"]
    apache = duckdb.execute(f"""SELECT patientunitstayid, predictedhospitalmortality::DOUBLE apache
        FROM '{E}/apachePatientResult.parquet' WHERE apacheversion = 'IVa'""").df()
    apache = apache[apache.apache > 0]
    d = d.merge(apache, on="patientunitstayid", how="left")
    print(f"eICU modelling cohort: {len(d):,} stays, {int(d.death.sum()):,} deaths "
          f"({d.death.mean():.1%}), {d.hospitalid.nunique()} hospitals")
    folds = list(GroupKFold(n_splits=5).split(d, d.death, d.hospitalid))
    rows, oof = [], {}
    for C in C_GRID:
        p = np.zeros(len(d))
        for tr, te in folds:
            pipe = make_pipeline(C).fit(d.loc[tr, cols], d.loc[tr, "death"])
            p[te] = pipe.predict_proba(d.loc[te, cols])[:, 1]
        oof[C] = p
        rows.append(dict(model=f"P0 C={C}", **metrics(d.death, p)))
    has_ap = d.apache.notna().to_numpy()
    best = max(C_GRID, key=lambda c: rows[C_GRID.index(c)]["auroc"] - 1e-9 * c)
    rows.append(dict(model="APACHE IVa (stays with a score)", **metrics(d.death[has_ap], d.apache[has_ap])))
    rows.append(dict(model=f"P0 C={best} (same stays)", **metrics(d.death[has_ap], oof[best][has_ap])))
    perf = pd.DataFrame(rows)
    perf.round(4).to_csv(res / "p0_internal_eicu.csv", index=False)
    print(perf.round(4).to_string(index=False))

    # per-hospital AUROC spread of the chosen model (hospitals with >= 20 deaths)
    per_h = [roc_auc_score(g.death, oof[best][g.index]) for _, g in d.groupby("hospitalid") if g.death.sum() >= 20]
    print(f"Per-hospital AUROC (C={best}, {len(per_h)} hospitals with >= 20 deaths): "
          f"median {np.median(per_h):.3f}, IQR {np.percentile(per_h, 25):.3f}-{np.percentile(per_h, 75):.3f}")

    lock = dict(features=cols, C=best, chosen_by="highest hospital-grouped 5-fold CV AUROC in eICU; ties go to stronger penalty",
                c_grid=C_GRID, eicu_cv_auroc=round(float(perf.loc[C_GRID.index(best), "auroc"]), 4))
    yaml.safe_dump(lock, open(res / "p0_locked.yaml", "w"), sort_keys=False)
    print(f"Locked C = {best}")


if __name__ == "__main__":
    main()
