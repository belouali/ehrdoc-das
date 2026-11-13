#!/usr/bin/env python
"""ICU readmission added to the documentation-and-outcome mixed models (author's request 2026-11-12).

Same models, code and inputs as 40_das_variation.py (A3) and 43_partA_effects.py, for one more outcome: ICU
readmission (a later ICU stay of type readmit in the same hospital stay), among stays alive at ICU discharge, with
the first-day patient covariates. Both samples (all hospitals with DAS, without no-medication hospitals); no survivor
analysis (the cohort is already alive at ICU discharge). Rows are appended to results/partA_outcome_models.csv,
partA_intercept_mantel.csv, partA_mor.csv and partA_nomed_or.csv (earlier icu_readmit rows replaced).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib \
      --with statsmodels --with scipy python scripts/51_partA_readmission.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("p43", HERE / "43_partA_effects.py")
p43 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p43)
p40 = p43.p40
TASK, POP = "icu_readmit", "all stays"


def main():
    p40.s38.TASKS = p40.s38.TASKS + ["icu_readmit", "disch_facility"]  # builds Y and ELIG for readmission
    cfg, _, res, rowlevel = p40.s38.setup("configs/external.yaml")
    st = yaml.safe_load(open(Path("configs/screen_settings.yaml")))["S_das_variation"]
    nodes, perms = 9, 9999
    rng = np.random.default_rng([int(st["seed"]), 51])
    split = pd.read_csv(res / "iecv_split.csv")
    e, X, XM, Y, rows_of, V, no_med = p40.s38.load_eicu(cfg, rowlevel)
    assert p40.s38.DAS_MIN[0] == 0, "DAS must use the 100-record minimum of the documentation analyses"
    elig = p40.s38.ELIG[TASK]
    hosp_141 = np.sort(split.hospitalid.to_numpy())
    hosp_all = hosp_141[[any(h in V[c].index for c in p40.CONDITIONS) for h in hosp_141]]
    assert len(hosp_all) == int(pd.read_csv(res / "partA_hospitals.csv").with_das.iloc[0])
    ht_all = p40.hospital_table(cfg, e, hosp_all)
    mats_all = p40.pairwise(V, hosp_all)
    om, mant, mor, orr = [], [], [], []
    for sample in ["all", "with_medication"]:
        keep = np.array([h not in no_med for h in hosp_all]) if sample == "with_medication" else np.ones(len(hosp_all), bool)
        hosp = hosp_all[keep]
        ht = ht_all.loc[hosp]
        D_mean = p40.sub(mats_all["mean"], keep)
        positions = p40.pcoa(D_mean)["X"][:, :p40.N_AXES]
        hosp_X = {"das": positions, "characteristics": p40.design_matrix(ht[p40.CATS + p40.MIX])}
        hosp_X["both"] = np.column_stack([hosp_X["das"], hosp_X["characteristics"]])
        nm = np.array([1.0 if h in no_med else 0.0 for h in hosp])
        hix = {h: i for i, h in enumerate(hosp)}
        msk = e.hospitalid.isin(hosp).to_numpy() & elig
        ee, y = e[msk], Y[TASK][msk]
        hi = ee.hospitalid.map(hix).to_numpy()
        Xp = p40.patient_design(ee)
        base, th0, cov0 = p43.fit_with_cov(Xp, y, hi, nodes)
        fits = {"base": (base, th0, cov0)}
        for name, H in hosp_X.items():
            fits[f"+ {name}"] = p43.fit_with_cov(np.column_stack([Xp, H[hi]]), y, hi, nodes, b0=np.r_[base.beta, np.zeros(H.shape[1])])
        row = dict(sample=sample, task=TASK, population=POP, stays=int(msk.sum()), events=int(y.sum()), hospitals=len(np.unique(hi)))
        for name, (m, th, cov) in fits.items():
            k = 0 if name == "base" else hosp_X[name[2:]].shape[1]
            chi = np.nan if name == "base" else 2 * (m.loglik - base.loglik)
            om.append(dict(row, model=name, added=k, sd=np.sqrt(m.s2), var=m.s2, share_explained=0.0 if name == "base" else 1 - m.s2 / base.s2,
                           loglik=m.loglik, lrt_chi2=chi, lrt_df=np.nan if name == "base" else k,
                           lrt_p=np.nan if name == "base" else stats.chi2.sf(chi, k), converged=m.converged, max_grad=m.max_grad))
            sd, se = np.sqrt(m.s2), np.sqrt(cov[-1, -1])
            lo, hi_ = np.exp(th[-1] - p43.Z * se), np.exp(th[-1] + p43.Z * se)
            mor.append(dict(sample=sample, population=POP, task=TASK, model=name, hospitals=len(hosp), sd=sd, sd_lo=lo, sd_hi=hi_,
                            mor=np.exp(p43.K * sd), mor_lo=np.exp(p43.K * lo), mor_hi=np.exp(p43.K * hi_), converged=m.converged))
        u = np.empty(len(hosp)); u[base.groups] = base.blup
        r, p = p40.mantel(D_mean, np.abs(u[:, None] - u[None, :]), perms, rng)
        mant.append(dict(sample=sample, task=TASK, population=POP, hospitals=len(hosp), spearman=r, p=p))
        if sample == "all":
            for adj, extra in [("unadjusted", None), ("adjusted for characteristics and case mix", hosp_X["characteristics"])]:
                Hx = nm[:, None] if extra is None else np.column_stack([nm, extra])
                m, th, cov = p43.fit_with_cov(np.column_stack([Xp, Hx[hi]]), y, hi, nodes, b0=np.r_[base.beta, np.zeros(Hx.shape[1])])
                j = Xp.shape[1]
                b, se = th[j], np.sqrt(cov[j, j])
                orr.append(dict(task=TASK, model=adj, hospitals=len(hosp), no_med_hospitals=int(nm.sum()), beta=b, se=se,
                                odds_ratio=np.exp(b), lo=np.exp(b - p43.Z * se), hi=np.exp(b + p43.Z * se),
                                p=2 * stats.norm.sf(abs(b / se)), sd=np.sqrt(m.s2), converged=m.converged))
        print(pd.DataFrame(om[-4:])[["sample", "model", "stays", "events", "sd", "share_explained", "lrt_p", "converged"]].round(4).to_string(index=False), flush=True)
    for name, rows in [("outcome_models", om), ("intercept_mantel", mant), ("mor", mor), ("nomed_or", orr)]:
        f = res / f"partA_{name}.csv"
        old = pd.read_csv(f)
        old = old[old.task != TASK]
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(f, index=False, float_format="%.6f")
    print(pd.DataFrame(orr).round(3).to_string(index=False))
    print(pd.DataFrame(mant).round(3).to_string(index=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()
