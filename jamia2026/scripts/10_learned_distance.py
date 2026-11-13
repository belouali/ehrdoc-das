#!/usr/bin/env python
"""Learned transport distance from eICU hospital pairs (paper's model, M1).

Everything is fitted and validated within eICU; MIMIC-IV outcomes are not read.

1. Transport outcomes. Hospitals with >= MIN_STAYS stays and >= MIN_DEATHS deaths.
   For every source s, fit M1 on s alone (mild L2 penalty, C = SINGLE_C, because
   single-hospital fits of the unpenalized model are unstable) and score every
   other hospital t. For every target t, fit M1 on all other eICU hospitals
   (the paper's unpenalized fit) and score t. Transport loss
   L(s, t) = AUROC(all others -> t) - AUROC(s -> t).
2. Pair features (non-negative dissimilarities, each scaled to unit SD):
   documentation  DAS between s and t for DHM and DAM, four conditions
   case mix       squared standardized differences of the 19 script-07 features
   structure      different region, different teaching status, bed-size gap
   plus log(source stays) as an unconstrained covariate (bigger sources transport
   better whatever their similarity; it is not part of the distance).
3. Weights by bounded least squares (distance weights >= 0) for three nested
   distances: case mix + structure, documentation, all.
4. Validation with target-grouped 5-fold CV (pairs touching a held-out target are
   excluded from fitting): within-target Spearman correlation between predicted
   distance and actual loss, and a selection test. For each held-out target, the 10
   nearest sources under each distance (and DAS-DHM nearest per condition, and 20
   random sets) train a pooled M1, scored on the target.
5. Final weights on all pairs, 200 bootstraps over hospitals, then the distance from
   every eICU hospital to the MIMIC-IV reference profile (no outcomes), written as a
   ranking file for stage 4 (results/learned_distance_to_mimic.csv).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib --with scipy \
      python scripts/10_learned_distance.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.optimize import lsq_linear
from sklearn.metrics import roc_auc_score
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
s07 = _load("s07", "07_casemix_nearest.py")

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
CONSTRUCTS = {"DHM": ("D", "H", "M"), "DAM": ("D", "A", "M")}
MIN_STAYS, MIN_DEATHS = 300, 20
SINGLE_C = 1.0
K = 10
N_RANDOM = 20
N_BOOT = 200
SEED = 20261108
BEDS = {"<100": 0, "100 - 249": 1, "250 - 499": 2, ">= 500": 3}


def fit_single(X, y, rows_s, rows_all, C):
    pipe = make_pipeline(C).fit(X[rows_s], y[rows_s])
    return pipe.predict_proba(X[rows_all])[:, 1]


def fit_pool(X, y, rows_train, rows_test, C):
    return make_pipeline(C).fit(X[rows_train], y[rows_train]).predict_proba(X[rows_test])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E, M = cfg["paths"]["eicu"], cfg["paths"]["mimic"]
    rng = np.random.default_rng(SEED)

    # ---- data: M1 design matrix, eICU only
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    cols = design_columns()
    X, y = e[cols].to_numpy(float), e.death.to_numpy(int)
    hid = e.hospitalid.to_numpy()
    g = e.groupby("hospitalid").death.agg(["size", "sum"])
    H = np.array(sorted(g[(g["size"] >= MIN_STAYS) & (g["sum"] >= MIN_DEATHS)].index))
    rows_of = {h: np.flatnonzero(hid == h) for h in np.unique(hid)}
    rows_H = np.concatenate([rows_of[h] for h in H])
    print(f"{len(H)} hospitals, {len(rows_H):,} stays, {len(H) * (len(H) - 1):,} ordered pairs")

    # ---- 1. transport outcomes
    preds = Parallel(n_jobs=args.jobs)(delayed(fit_single)(X, y, rows_of[s], rows_H, SINGLE_C) for s in H)
    # position of each hospital's rows inside rows_H (rows_H concatenates rows_of[h] in H order)
    offsets = np.cumsum([0] + [len(rows_of[h]) for h in H])
    pos = {h: np.arange(offsets[i], offsets[i + 1]) for i, h in enumerate(H)}
    auc = np.full((len(H), len(H)), np.nan)
    for i, p in enumerate(preds):
        for j, t in enumerate(H):
            if i != j:
                auc[i, j] = roc_auc_score(y[rows_of[t]], p[pos[t]])
    all_rows = np.arange(len(y))
    ref = Parallel(n_jobs=args.jobs)(delayed(fit_pool)(X, y, all_rows[hid != t], rows_of[t], C_M1) for t in H)
    auc_ref = np.array([roc_auc_score(y[rows_of[t]], p) for t, p in zip(H, ref)])
    loss = auc_ref[None, :] - auc

    # ---- 2. hospital profiles and pair features
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    das_pairs = {}
    for cname, srcs in CONSTRUCTS.items():
        for cond in CONDITIONS:
            codes = s02.lobe_codes(flags, cond, srcs)
            cnt = s02.hospital_counts(flags, codes, 2 ** len(srcs) - 1)
            cnt = cnt[cnt.sum(axis=1) >= cfg["das"]["min_records_per_hospital"]]
            V = cnt.div(cnt.sum(axis=1), axis=0)
            Vh = V.reindex(H).to_numpy(float)
            ok = ~np.isnan(Vh).any(axis=1)
            P = np.full((len(H), len(H)), np.nan)
            P[np.ix_(ok, ok)] = s02.pairwise_angles(Vh[ok])
            das_pairs[f"das_{cname}_{cond}"] = (P, V)

    feat = pd.read_parquet(rowlevel / "eicu_features.parquet")
    feat = feat[feat.death.notna()]
    eu = duckdb.execute(f"SELECT patientunitstayid, unittype FROM '{E}/patient.parquet'").df()
    feat = s07.add_units(feat.reset_index(drop=True), feat[["patientunitstayid"]].merge(eu, how="left").unittype, s07.EICU_UNITS)
    hmeans = feat.groupby("hospitalid")[s07.CASEMIX].mean()
    mu, sd = hmeans.mean(), hmeans.std()
    Z = ((hmeans - mu) / sd)
    hosp = duckdb.execute(f"SELECT hospitalid, region, teachingstatus, numbedscategory FROM '{E}/hospital.parquet'").df().set_index("hospitalid")
    beds = hosp.numbedscategory.map(BEDS)

    def pair_frame(src_ids, tgt_ids, zt=None, das_t=None, tgt_struct=None):
        """Feature rows for every (source, target) combination given."""
        rows = []
        for s in src_ids:
            for t in tgt_ids:
                if s == t:
                    continue
                r = dict(source=s, target=t)
                zs = Z.loc[s] if s in Z.index else pd.Series(np.nan, index=s07.CASEMIX)
                ztt = zt if zt is not None else (Z.loc[t] if t in Z.index else pd.Series(np.nan, index=s07.CASEMIX))
                for c in s07.CASEMIX:
                    r[f"cm_{c}"] = (zs[c] - ztt[c]) ** 2
                st = tgt_struct if tgt_struct is not None else (hosp.region.get(t), hosp.teachingstatus.get(t), beds.get(t))
                r["diff_region"] = float(hosp.region.get(s) != st[0]) if pd.notna(hosp.region.get(s)) and pd.notna(st[0]) else np.nan
                r["diff_teaching"] = float(bool(hosp.teachingstatus.get(s)) != bool(st[1]))
                r["diff_beds"] = abs(beds.get(s) - st[2]) if pd.notna(beds.get(s)) and pd.notna(st[2]) else np.nan
                r["log_source_stays"] = np.log(len(rows_of.get(s, [])) + 1)
                rows.append(r)
        return pd.DataFrame(rows)

    pairs = pair_frame(H, H)
    idx = {h: i for i, h in enumerate(H)}
    pi, pj = pairs.source.map(idx).to_numpy(), pairs.target.map(idx).to_numpy()
    for name, (P, _) in das_pairs.items():
        pairs[name] = P[pi, pj]
    pairs["loss"] = loss[pi, pj]
    pairs = pairs[pairs.loss.notna()].reset_index(drop=True)

    DOC = list(das_pairs)
    CASE = [f"cm_{c}" for c in s07.CASEMIX] + ["diff_region", "diff_teaching", "diff_beds"]
    SETS = {"casemix": CASE, "documentation": DOC, "all": CASE + DOC}
    fill = pairs[CASE + DOC].median()
    scale = pairs[CASE + DOC].fillna(fill).std().replace(0, 1)

    def design_pairs(df, feats):
        A = (df[feats].fillna(fill[feats]) / scale[feats]).to_numpy(float)
        return np.column_stack([A, df.log_source_stays.to_numpy(float), np.ones(len(df))])

    def fit_weights(df, feats):
        A = design_pairs(df, feats)
        lb = np.r_[np.zeros(len(feats)), -np.inf, -np.inf]
        return lsq_linear(A, df.loss.to_numpy(float), bounds=(lb, np.full(A.shape[1], np.inf))).x

    def distance(df, feats, w):
        return design_pairs(df, feats)[:, :len(feats)] @ w[:len(feats)]

    # ---- 4. target-grouped CV
    gkf = GroupKFold(n_splits=5)
    cv_rows, sel_tasks = [], []
    targets_all = pairs.target.to_numpy()
    for fold, (_, te_idx) in enumerate(gkf.split(pairs, groups=targets_all)):
        held = set(pairs.target.iloc[te_idx])
        train = pairs[~pairs.target.isin(held) & ~pairs.source.isin(held)]
        test = pairs[pairs.target.isin(held)]
        for sname, feats in SETS.items():
            w = fit_weights(train, feats)
            d = distance(test, feats, w)
            tt = test.assign(d=d)
            for t, gtt in tt.groupby("target"):
                rho = gtt.d.rank().corr(gtt.loss.rank())
                cv_rows.append(dict(fold=fold, distance=sname, target=t, spearman=rho))
                near = gtt.nsmallest(K, "d").source.tolist()
                sel_tasks.append((t, f"learned_{sname}", near))
        for t in held:
            cand = [h for h in H if h != t]
            for name in [n for n in DOC if n.startswith("das_DHM")]:
                P = das_pairs[name][0][:, idx[t]]
                cd = pd.Series({h: P[idx[h]] for h in cand}).dropna().sort_values()
                if len(cd) >= K:   # targets not eligible for this condition have no DAS ranking
                    sel_tasks.append((t, name.replace("das_", "dasnearest_"), cd.index[:K].tolist()))
            for r in range(N_RANDOM):
                sel_tasks.append((t, f"random_{r}", rng.choice(cand, size=K, replace=False).tolist()))
    cv = pd.DataFrame(cv_rows)
    sel_auc = Parallel(n_jobs=args.jobs, batch_size=8)(
        delayed(fit_pool)(X, y, np.concatenate([rows_of[h] for h in hs]), rows_of[t], C_M1) for t, _, hs in sel_tasks)
    sel = pd.DataFrame([dict(target=t, strategy=s, auroc=roc_auc_score(y[rows_of[t]], p),
                             n_train=sum(len(rows_of[h]) for h in hs))
                        for (t, s, hs), p in zip(sel_tasks, sel_auc)])
    sel["family"] = np.where(sel.strategy.str.startswith("random"), "random", sel.strategy)
    per_t = sel.groupby(["target", "family"]).agg(auroc=("auroc", "mean"), n_train=("n_train", "mean")).reset_index()
    wide = per_t.pivot(index="target", columns="family", values="auroc")
    comp = []
    for fam in [c for c in wide.columns if c != "random"]:
        d = (wide[fam] - wide["random"]).dropna()
        boots = [d.sample(len(d), replace=True, random_state=int(b)).mean() for b in range(1000)]
        comp.append(dict(strategy=fam, targets=len(d), mean_auroc=wide[fam].mean(), random_mean=wide["random"].mean(),
                         diff_vs_random=d.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                         share_targets_better=np.mean(d > 0),
                         n_train_mean=per_t[per_t.family == fam].n_train.mean()))
    comp = pd.DataFrame(comp).sort_values("diff_vs_random", ascending=False)
    cvs = cv.groupby("distance").spearman.agg(["mean", "median", "count"]).reset_index()

    # ---- 5. final weights, bootstrap over hospitals, distance to MIMIC-IV
    wt_rows = []
    finals = {}
    for sname, feats in SETS.items():
        w = fit_weights(pairs, feats)
        finals[sname] = w
        bw = []
        for b in range(N_BOOT):
            hs = rng.choice(H, size=len(H), replace=True)
            mult = pd.Series(hs).value_counts()
            bp = pairs[pairs.source.isin(mult.index) & pairs.target.isin(mult.index)]
            rep = (bp.source.map(mult) * bp.target.map(mult)).to_numpy()
            bp = bp.loc[bp.index.repeat(rep)]
            bw.append(fit_weights(bp, feats))
        bw = np.array(bw)
        for j, f in enumerate(feats + ["log_source_stays", "intercept"]):
            wt_rows.append(dict(distance=sname, feature=f, weight=w[j], lo=np.percentile(bw[:, j], 2.5),
                                hi=np.percentile(bw[:, j], 97.5), share_boot_positive=np.mean(bw[:, j] > 1e-9)))
    weights = pd.DataFrame(wt_rows)
    doc_share = []
    for sname in ["all"]:
        w = weights[(weights.distance == sname) & weights.feature.isin(CASE + DOC)]
        doc_share.append(dict(distance=sname, documentation_weight_share=w[w.feature.isin(DOC)].weight.sum() / w.weight.sum()))

    # MIMIC-IV reference profile (no outcomes)
    m = pd.read_parquet(rowlevel / "mimic_features.parquet").merge(
        pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]], on="stay_id")
    mu_unit = duckdb.execute(f"SELECT stay_id, first_careunit FROM '{M}/icu/icustays.parquet'").df()
    m = s07.add_units(m, m[["stay_id"]].merge(mu_unit, how="left").first_careunit, s07.MIMIC_UNITS)
    zt = ((m[m.reference][s07.CASEMIX].mean() - mu) / sd)
    das_m = pd.read_csv(res / "das_to_mimic.csv")
    das_m = das_m[das_m.reference == "ref_2011_2019"]
    all_h = sorted(set(das_m[das_m.construct == "DHM"].hospitalid))
    mp = pair_frame(all_h, ["MIMIC"], zt=zt, tgt_struct=("Northeast", True, 3))
    for cname in CONSTRUCTS:
        for cond in CONDITIONS:
            dd = das_m[(das_m.construct == cname) & (das_m.condition == cond)].set_index("hospitalid").das
            mp[f"das_{cname}_{cond}"] = mp.source.map(dd)
    mp["learned_all"] = distance(mp, SETS["all"], finals["all"])
    mp["learned_casemix"] = distance(mp, SETS["casemix"], finals["casemix"])
    rank_rows = []
    for cond in CONDITIONS:
        pool = das_m[(das_m.construct == "DHM") & (das_m.condition == cond)][["hospitalid", "zero_source"]]
        r = pool.merge(mp[["source", "learned_all"]].rename(columns={"source": "hospitalid", "learned_all": "das"}), on="hospitalid")
        r["condition"] = cond
        rank_rows.append(r[["condition", "hospitalid", "das", "zero_source"]])
    pd.concat(rank_rows).round(6).to_csv(res / "learned_distance_to_mimic.csv", index=False)

    # outputs
    pd.DataFrame(dict(hospital=H, auroc_all_others=auc_ref)).round(4).to_csv(res / "learned_targets.csv", index=False)
    cvs.round(4).to_csv(res / "learned_cv_spearman.csv", index=False)
    comp.round(5).to_csv(res / "learned_selection_eicu.csv", index=False)
    weights.round(5).to_csv(res / "learned_weights.csv", index=False)
    yaml.safe_dump(dict(min_stays=MIN_STAYS, min_deaths=MIN_DEATHS, single_c=SINGLE_C, hospitals=int(len(H)),
                        pairs=int(len(pairs)), features=SETS["all"],
                        weights={f: float(w) for f, w in zip(SETS["all"] + ["log_source_stays", "intercept"], finals["all"])},
                        fill=fill.round(6).to_dict(), scale=scale.round(6).to_dict()),
                   open(res / "learned_distance_locked.yaml", "w"), sort_keys=False)
    pd.set_option("display.width", 220)
    print(f"Single-hospital transport: median AUROC {np.nanmedian(auc):.3f}; median loss {np.nanmedian(loss):.3f}")
    print(cvs.round(3).to_string(index=False))
    print(comp.round(4).to_string(index=False))
    print(pd.DataFrame(doc_share).round(3).to_string(index=False))
    w = weights[(weights.distance == "all") & (weights.weight > 1e-9)].sort_values("weight", ascending=False)
    print(w.round(4).to_string(index=False))
    top = mp.nsmallest(10, "learned_all")[["source", "learned_all"]]
    print("Learned-nearest 10 to MIMIC-IV:", top.source.tolist())


if __name__ == "__main__":
    main()
