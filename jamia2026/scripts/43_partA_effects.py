#!/usr/bin/env python
"""Effect sizes for the documentation-and-outcome mixed models (S_das_variation; added on the author's request 2026-11-09).

1. Median odds ratio, MOR = exp(0.6745 * sqrt(2) * sigma), for every model of scripts/40 (base, + das,
   + characteristics, + both) in each sample (all stays, survivors, without no-medication hospitals). The models
   are refitted with the same code and inputs; the refitted SDs must reproduce results/partA_outcome_models.csv.
   95% CI from the observed information: numerical Hessian of the marginal log-likelihood (central differences
   of the analytic gradient) at the optimum, Wald interval on log(sigma), transformed to the MOR scale.
2. Documentation odds ratio: a hospital-level indicator for no medication records (1) against medication records
   (0) added to the base model, alone and with hospital characteristics and case mix; OR with Wald 95% CI from
   the same Hessian. Same hospitals as the base model (those with a DAS measure).

Outputs (aggregate): results/partA_mor.csv, results/partA_nomed_or.csv
Usage (from mimic_external/): bash scripts/partA.sh is for 40; run this with the same environment.
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
spec = importlib.util.spec_from_file_location("p40", HERE / "40_das_variation.py")
p40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p40)
Z = stats.norm.ppf(0.975)
K = 0.6745 * np.sqrt(2)


def hessian(model, theta, h=1e-4):
    k = len(theta)
    H = np.empty((k, k))
    for j in range(k):
        e = np.zeros(k); e[j] = h
        H[:, j] = (model._eval(theta + e)[1] - model._eval(theta - e)[1]) / (2 * h)
    return (H + H.T) / 2


def fit_with_cov(X, y, g, nodes, b0=None):
    m = p40.RandomInterceptLogit(X, y, g, nodes).fit(b0=b0)
    theta = np.r_[m.beta, 0.5 * np.log(m.s2)]
    cov = np.linalg.inv(-hessian(m, theta))
    return m, theta, cov


def main(quick=False):
    cfg, _, res, rowlevel = p40.s38.setup("configs/external.yaml")
    st = yaml.safe_load(open(Path("configs/screen_settings.yaml")))["S_das_variation"]
    assert st["status"] == "locked"
    nodes = 9
    split = pd.read_csv(res / "iecv_split.csv")
    e, X, XM, Y, rows_of, V, no_med = p40.s38.load_eicu(cfg, rowlevel)
    hosp_141 = np.sort(split.hospitalid.to_numpy())
    hosp_all = hosp_141[[any(h in V[c].index for c in p40.CONDITIONS) for h in hosp_141]]
    ht_all = p40.hospital_table(cfg, e, hosp_all)
    mats_all = p40.pairwise(V, hosp_all)
    stored = pd.read_csv(res / "partA_outcome_models.csv")
    mor, orr = [], []
    for sample in ["all", "with_medication"]:
        keep = np.array([h not in no_med for h in hosp_all]) if sample == "with_medication" else np.ones(len(hosp_all), bool)
        hosp = hosp_all[keep]
        ht = ht_all.loc[hosp]
        positions = p40.pcoa(p40.sub(mats_all["mean"], keep))["X"][:, :p40.N_AXES]
        hosp_X = {"das": positions, "characteristics": p40.design_matrix(ht[p40.CATS + p40.MIX])}
        hosp_X["both"] = np.column_stack([hosp_X["das"], hosp_X["characteristics"]])
        nm = np.array([1.0 if h in no_med else 0.0 for h in hosp])
        hix = {h: i for i, h in enumerate(hosp)}
        keep_stays = e.hospitalid.isin(hosp).to_numpy()
        tasks = [(t, "all stays") for t in p40.TASKS] + ([(t, "survivors") for t in p40.STAY_TASKS] if sample == "all" else [])
        if quick:
            tasks = tasks[:1]
        for task, pop in tasks:
            msk = keep_stays & (e.death.to_numpy() == 0 if pop == "survivors" else True)
            ee = e[msk]
            y = Y[task][msk]
            hi = ee.hospitalid.map(hix).to_numpy()
            Xp = p40.patient_design(ee)
            base, th0, cov0 = fit_with_cov(Xp, y, hi, nodes)
            fits = {"base": (base, th0, cov0)}
            for name, H in hosp_X.items():
                fits[f"+ {name}"] = fit_with_cov(np.column_stack([Xp, H[hi]]), y, hi, nodes, b0=np.r_[base.beta, np.zeros(H.shape[1])])
            for name, (m, th, cov) in fits.items():
                ref = stored[(stored["sample"] == sample) & (stored.population == pop) & (stored.task == task) & (stored.model == name)].sd.iloc[0]
                sd = np.sqrt(m.s2)
                assert abs(sd - ref) < 2e-3, f"refit SD {sd:.4f} differs from stored {ref:.4f} ({sample}, {pop}, {task}, {name})"
                se = np.sqrt(cov[-1, -1])
                lo, hi_ = np.exp(th[-1] - Z * se), np.exp(th[-1] + Z * se)
                mor.append(dict(sample=sample, population=pop, task=task, model=name, hospitals=len(hosp), sd=sd,
                                sd_lo=lo, sd_hi=hi_, mor=np.exp(K * sd), mor_lo=np.exp(K * lo), mor_hi=np.exp(K * hi_),
                                converged=m.converged))
            if pop == "all stays" and sample == "all":
                for adj, extra in [("unadjusted", None), ("adjusted for characteristics and case mix", hosp_X["characteristics"])]:
                    Hx = nm[:, None] if extra is None else np.column_stack([nm, extra])
                    m, th, cov = fit_with_cov(np.column_stack([Xp, Hx[hi]]), y, hi, nodes, b0=np.r_[base.beta, np.zeros(Hx.shape[1])])
                    j = Xp.shape[1]  # the indicator is the first added column
                    b, se = th[j], np.sqrt(cov[j, j])
                    orr.append(dict(task=task, model=adj, hospitals=len(hosp), no_med_hospitals=int(nm.sum()),
                                    beta=b, se=se, odds_ratio=np.exp(b), lo=np.exp(b - Z * se), hi=np.exp(b + Z * se),
                                    p=2 * stats.norm.sf(abs(b / se)), sd=np.sqrt(m.s2), converged=m.converged))
            print(pd.DataFrame(mor[-4:]).round(3).to_string(index=False), flush=True)
            if orr and orr[-1]["task"] == task:
                print(pd.DataFrame(orr[-2:]).round(3).to_string(index=False), flush=True)
    out = res / "_test" if quick else res
    out.mkdir(exist_ok=True)
    pd.DataFrame(mor).to_csv(out / "partA_mor.csv", index=False, float_format="%.6f")
    pd.DataFrame(orr).to_csv(out / "partA_nomed_or.csv", index=False, float_format="%.6f")
    print("done", flush=True)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
