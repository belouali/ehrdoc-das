#!/usr/bin/env python
"""Reverse direction: the paper's model trained on MIMIC-IV, scored at each eICU hospital.

Question: does the MIMIC-IV model lose more at eICU hospitals whose documentation is
farther from MIMIC-IV's? One large source, 80-150 targets, so the test is a correlation
across hospitals rather than one nearest-versus-random comparison.
Settings: configs/reverse_settings.yaml (locked before the first run).

Outputs (aggregate): results/reverse_hospitals.csv (one row per eICU hospital),
results/reverse_summary.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with statsmodels python scripts/13_reverse_transport.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_model import C_M1, design_columns  # noqa: E402
from p0_model import make_pipeline, metrics  # noqa: E402

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
CONSTRUCTS = ["DHM", "DHAM", "DAM"]
SAMPLES = {"primary": (300, 20), "sensitivity": (100, 10)}
N_BOOT = 2000


def leave_out(X, y, train, test):
    return make_pipeline(C_M1).fit(X[train], y[train]).predict_proba(X[test])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--settings", default="configs/reverse_settings.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    settings = yaml.safe_load(open(args.settings))
    assert settings["status"] == "locked", "settings must be locked before outcomes are read"
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    rng = np.random.default_rng(settings["seed"])
    cols = design_columns()

    # MIMIC-IV model, all cohort stays
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet")
    out = duckdb.execute(f"SELECT hadm_id, hospital_expire_flag death FROM '{M}/hosp/admissions.parquet'").df()
    m = m.merge(out, on="hadm_id", how="left")
    assert m.death.notna().all()
    mimic_model = make_pipeline(C_M1).fit(m[cols].to_numpy(float), m.death.to_numpy(int))
    print(f"MIMIC-IV training: {len(m):,} stays, {int(m.death.sum()):,} deaths", flush=True)

    # eICU hospitals
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    X, y, hid = e[cols].to_numpy(float), e.death.to_numpy(int), e.hospitalid.to_numpy()
    g = e.groupby("hospitalid").death.agg(["size", "sum"])
    lo_s, lo_d = SAMPLES["sensitivity"]
    H = np.array(sorted(g[(g["size"] >= lo_s) & (g["sum"] >= lo_d)].index))
    p_mimic = mimic_model.predict_proba(X)[:, 1]
    p_lo = Parallel(n_jobs=args.jobs)(delayed(leave_out)(X, y, np.flatnonzero(hid != h), np.flatnonzero(hid == h)) for h in H)

    rows = []
    for h, pl in zip(H, p_lo):
        r = hid == h
        mm, ml = metrics(y[r], p_mimic[r]), metrics(y[r], pl)
        rows.append(dict(hospitalid=h, stays=int(r.sum()), deaths=int(y[r].sum()),
                         auroc_mimic=mm["auroc"], auroc_eicu=ml["auroc"], loss_auroc=ml["auroc"] - mm["auroc"],
                         brier_mimic=mm["brier"], brier_eicu=ml["brier"], loss_brier=mm["brier"] - ml["brier"],
                         citl_mimic=mm["cal_intercept"], abs_citl_mimic=abs(mm["cal_intercept"])))
    hosp = pd.DataFrame(rows)
    info = duckdb.execute(f"SELECT hospitalid, region, teachingstatus FROM '{E}/hospital.parquet'").df()
    hosp = hosp.merge(info, on="hospitalid", how="left")
    hosp["northeast"] = (hosp.region == "Northeast").astype(float)
    hosp["teaching"] = hosp.teachingstatus.astype(str).str.lower().isin(["true", "t", "1"]).astype(float)
    cm = pd.read_csv(res / "casemix_distance.csv")[["hospitalid", "casemix_distance"]]
    hosp = hosp.merge(cm, on="hospitalid", how="left")
    das = pd.read_csv(res / "das_to_mimic.csv")
    das = das[das.reference == "ref_2011_2019"]
    for c in CONSTRUCTS:
        for cond in CONDITIONS:
            s = das[(das.construct == c) & (das.condition == cond)].set_index("hospitalid").das
            hosp[f"das_{c}_{cond}"] = hosp.hospitalid.map(s)
    hosp.round(5).to_csv(res / "reverse_hospitals.csv", index=False)

    def spearman_ci(a, b):
        rho = pd.Series(a).rank().corr(pd.Series(b).rank())
        n = len(a)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            bs.append(pd.Series(a[i]).rank().corr(pd.Series(b[i]).rank()))
        return rho, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

    summ = []
    for sample, (ms, md) in SAMPLES.items():
        hs = hosp[(hosp.stays >= ms) & (hosp.deaths >= md)]
        for loss in ["loss_auroc", "loss_brier", "abs_citl_mimic"]:
            rows_cmp = [("casemix", "casemix_distance")] + [(f"{c}_{cond}", f"das_{c}_{cond}") for c in CONSTRUCTS for cond in CONDITIONS]
            for name, col in rows_cmp:
                d = hs.dropna(subset=[col, "casemix_distance", loss])
                rho, rlo, rhi = spearman_ci(d[col].to_numpy(), d[loss].to_numpy())
                row = dict(sample=sample, loss=loss, distance=name, n=len(d), mean_loss=d[loss].mean(),
                           spearman=rho, spearman_lo=rlo, spearman_hi=rhi)
                if name != "casemix":
                    z = lambda v: (v - v.mean()) / v.std()
                    Xa = pd.DataFrame(dict(das=z(d[col]), casemix=z(d.casemix_distance), log_stays=np.log(d.stays),
                                           northeast=d.northeast, teaching=d.teaching))
                    fit = sm.OLS(d[loss], sm.add_constant(Xa)).fit(cov_type="HC3")
                    ci = fit.conf_int()
                    row |= dict(adj_das_per_sd=fit.params["das"], adj_das_lo=ci.loc["das", 0], adj_das_hi=ci.loc["das", 1],
                                adj_das_p=fit.pvalues["das"], adj_casemix_per_sd=fit.params["casemix"],
                                adj_casemix_p=fit.pvalues["casemix"], adj_northeast=fit.params["northeast"],
                                adj_northeast_p=fit.pvalues["northeast"])
                summ.append(row)
    summ = pd.DataFrame(summ)
    summ.round(5).to_csv(res / "reverse_summary.csv", index=False)
    pd.set_option("display.width", 250)
    show = summ[(summ["sample"] == "primary") & (summ.loss == "loss_auroc")]
    print(f"primary targets: {int(((hosp.stays >= 300) & (hosp.deaths >= 20)).sum())}, "
          f"mean AUROC MIMIC model {hosp.auroc_mimic.mean():.4f}, leave-out eICU {hosp.auroc_eicu.mean():.4f}")
    print(show.drop(columns=["sample", "loss"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
