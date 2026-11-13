#!/usr/bin/env python
"""Why does the CHF angle pick good training hospitals for ICU stay > 7 days? (descriptive)

Exploratory, no test. Requested by the user 2026-11-06 after W1, W1_pooled, W1_proxy and
W1_diabetes. Every MIMIC-IV quantity here was already read in those runs.

  practice   per eICU pool hospital: observed rate of stay > 7 days (O) and expected rate
             (E) from a hospital-grouped 5-fold out-of-fold model (M1 inputs + Charlson +
             12 flags, as script 21); O/E and the share of hospitals outside 99.8% binomial
             limits. MIMIC-IV O/E uses the model fitted on every eICU stay.
  sets       stay-weighted O/E, raw rate and Northeast share for CHF nearest 10, diabetes
             nearest 10, case-mix nearest 10 and the mean of 500 random sets of 10.
  angles     Spearman correlation, across pool hospitals, of each condition's DHM angle to
             MIMIC-IV with |log(O/E) - log(O/E of MIMIC-IV)| and with each CHF lobe share.

Outputs (aggregate): results/los_practice_hospitals.csv (hospital-level), results/los_practice_summary.csv

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with scipy python scripts/27_los_practice.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binom, spearmanr
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import C_M1, design_columns  # noqa: E402
from p0_model import make_pipeline  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
K, N_RANDOM, MIN_POOL = 10, 500, 100
SEED = 20261115


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    rng = np.random.default_rng(SEED)
    cols = design_columns() + ["charlson"] + FLAGS

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    los = duckdb.execute(f"SELECT patientunitstayid, unitdischargeoffset / 1440.0 AS los_d FROM '{E}/patient.parquet'").df()
    e = e.merge(los, on="patientunitstayid", how="left").reset_index(drop=True)
    X, y = e[cols].to_numpy(float), (e.los_d > 7).to_numpy(int)
    hid = e.hospitalid.to_numpy()
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, hid):
        p[te] = make_pipeline(C_M1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    h = pd.DataFrame({"hospitalid": hid, "y": y, "p": p, "los_d": e.los_d}).groupby("hospitalid").agg(
        stays=("y", "size"), observed=("y", "mean"), expected=("p", "mean"), median_los_d=("los_d", "median"))
    h = h[h.stays >= MIN_POOL]
    h["oe"] = h.observed / h.expected
    lo = binom.ppf(0.001, h.stays, h.expected) / h.stays
    hi = binom.ppf(0.999, h.stays, h.expected) / h.stays
    h["outside_limits"] = (h.observed < lo) | (h.observed > hi)

    # MIMIC-IV O/E under the all-eICU model
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    m = m[~m.reference].reset_index(drop=True)
    m = m.merge(pd.read_parquet(rowlevel / "mimic_m2_features.parquet")[["stay_id", "charlson"]], on="stay_id", how="left")
    dm = [f for f in FLAGS if not f.endswith("_H")]
    hf = [f for f in FLAGS if f.endswith("_H")]
    m = m.merge(pd.read_parquet(rowlevel / "mimic_flags.parquet")[["stay_id"] + dm], on="stay_id", how="left")
    m = m.merge(pd.read_parquet(rowlevel / "mimic_history_flags.parquet")[["stay_id"] + hf], on="stay_id", how="left")
    m[FLAGS] = m[FLAGS].fillna(0).astype(float)
    m = m.merge(duckdb.execute(f"SELECT stay_id, los FROM '{M}/icu/icustays.parquet'").df(), on="stay_id", how="left")
    pm = make_pipeline(C_M1).fit(X, y).predict_proba(m[cols].to_numpy(float))[:, 1]
    mim = dict(observed=(m.los > 7).mean(), expected=pm.mean(), median_los_d=m.los.median())
    mim["oe"] = mim["observed"] / mim["expected"]

    # hospital type, angles, CHF lobe shares
    ht = duckdb.execute(f"SELECT * FROM '{E}/hospital.parquet'").df().set_index("hospitalid")
    h["northeast"] = ht.reindex(h.index).region.eq("Northeast")
    das = pd.read_csv(res / "das_to_mimic.csv")
    das = das[(das.construct == "DHM") & (das.reference == "ref_2011_2019")]
    for c in CONDITIONS:
        h[f"angle_{c}"] = das[das.condition == c].set_index("hospitalid").das.reindex(h.index)
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    labels = s02.lobe_labels(("D", "H", "M"))
    cnt = s02.hospital_counts(flags, s02.lobe_codes(flags, "chf", ("D", "H", "M")), 7)
    sh = cnt.div(cnt.sum(axis=1), axis=0)
    sh.columns = [f"chf_{l}" for l in labels]
    h = h.join(sh)
    h["chf_any_D"] = h[[c for c in sh.columns if "D" in c.split("_", 1)[1]]].sum(axis=1)
    h["dist_logoe"] = (np.log(h.oe) - np.log(mim["oe"])).abs()
    cm = pd.read_csv(res / "casemix_distance.csv").set_index("hospitalid").casemix_distance
    h["casemix_distance"] = cm.reindex(h.index)

    def setstats(name, hs):
        s = h.loc[hs]
        return dict(item=name, hospitals=len(s), observed=(s.observed * s.stays).sum() / s.stays.sum(),
                    oe=(s.observed * s.stays).sum() / (s.expected * s.stays).sum(),
                    median_los_d=s.median_los_d.median(), northeast=int(s.northeast.sum()),
                    chf_any_D=s.chf_any_D.mean())

    rows = [dict(item="mimic_iv", **mim),
            dict(item="eicu_pool", hospitals=len(h), observed=(h.observed * h.stays).sum() / h.stays.sum(),
                 oe_sd=h.oe.std(), oe_p10=h.oe.quantile(0.1), oe_p90=h.oe.quantile(0.9),
                 share_outside_limits=h.outside_limits.mean(), median_los_d=h.median_los_d.median(),
                 chf_any_D=h.chf_any_D.mean())]
    for c in ("chf", "diabetes"):
        pool = h[h[f"angle_{c}"].notna()]
        rows.append(setstats(f"{c}_nearest10", pool[f"angle_{c}"].sort_values(kind="stable").index[:K]))
        r = [setstats("r", rng.choice(pool.index, K, replace=False)) for _ in range(N_RANDOM)]
        rows.append(dict(item=f"{c}_random10_mean", **pd.DataFrame(r).drop(columns="item").mean().to_dict()))
    pool = h[h.angle_chf.notna() & h.casemix_distance.notna()]
    rows.append(setstats("casemix_nearest10", pool.casemix_distance.sort_values(kind="stable").index[:K]))
    for c in CONDITIONS + ["casemix"]:
        col = f"angle_{c}" if c != "casemix" else "casemix_distance"
        ok = h[col].notna()
        rho, pv = spearmanr(h.loc[ok, col], h.loc[ok, "dist_logoe"])
        rows.append(dict(item=f"spearman_{col}_vs_oe_distance", hospitals=int(ok.sum()), rho=rho, p=pv))
    ok = h.angle_chf.notna()
    for c in list(sh.columns) + ["chf_any_D"]:
        rho, pv = spearmanr(h.loc[ok, "angle_chf"], h.loc[ok, c])
        rows.append(dict(item=f"spearman_angle_chf_vs_{c}", hospitals=int(ok.sum()), rho=rho, p=pv))
        rho, pv = spearmanr(h.loc[ok, c], h.loc[ok, "oe"])
        rows.append(dict(item=f"spearman_{c}_vs_oe", hospitals=int(ok.sum()), rho=rho, p=pv))
    out = pd.DataFrame(rows)
    out.to_csv(res / "los_practice_summary.csv", index=False, float_format="%.4f")
    h.drop(columns=["northeast"]).round(5).to_csv(res / "los_practice_hospitals.csv")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
