#!/usr/bin/env python
"""B4 shared hospital habit (predictions P9-P12), as locked in configs/screen_settings.yaml.

Hospital documentation effects (random-intercept modes) for obesity and four lab-defined
conditions, from the P5/P7 models of script 15; their correlation, a shared factor, its
correlates, and its link to DAS. Documentation only, no outcomes. Aggregates only in results/.

Usage (from mimic_external/), after script 15 has built the row-level table:
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scipy \
      --with statsmodels python scripts/16_shared_habit.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s15", HERE / "15_positive_controls.py")
s15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s15)
glmm = s15.glmm
assert s15.SET["B4_shared_habit"]["status"].startswith("locked")
CONDS = ["obesity", "aki", "hyponatremia", "anemia", "thrombocytopenia"]
PUBLISHED = ["diabetes", "hypertension", "chf", "af"]
MIN_STAYS, B, B12, PERM = 30, 1000, 2000, 1000
rng = np.random.default_rng(s15.SEED + 4)


def frames(e):
    """Model frames identical to script 15 (P5 obesity, P7 per lab-defined condition)."""
    out = {}
    ob = e[e.bmi >= 30].dropna(subset=["female"]).copy()
    ob["sev"] = np.select([ob.bmi < 35, ob.bmi < 40], [0, 1], 2)
    ob["y"] = ob.obesity_doc.astype(int)
    out["obesity"] = ob
    e2 = e.dropna(subset=["female"]).copy()
    for cond, d in s15.ACUTE.items():
        if cond == "aki":
            st = s15.stage_aki(e2[~e2.renal_excl])
            dd = e2.loc[st.index].assign(sev=st)
        else:
            col = {"sodium": "na_min", "hgb": "hgb_min", "platelets": "plt_min"}[d["lab"]]
            dd = e2.assign(sev=s15.stage_lab(e2[col], d["cuts"]))
        dd = dd[dd.sev >= 1].copy()
        dd["y"] = dd[f"{cond}_doc"].astype(int)
        out[cond] = dd
    for d in out.values():
        d["age10"] = d.age_n // 1 / 10
    return out


def hospital_modes(d, xcols):
    keys = xcols + ["hospitalid"]
    a = d.groupby(keys, observed=True).y.agg(["sum", "count"]).reset_index()
    codes, uniq = pd.factorize(a.hospitalid)
    X = np.c_[np.ones(len(a)), a[xcols].to_numpy(float)]
    r = glmm.fit(X, a["sum"].to_numpy(float), a["count"].to_numpy(float), codes, se=False)
    return pd.Series(r["u"], index=uniq), r["sd"]


def pc1(Z):
    C = np.corrcoef(Z, rowvar=False)
    w, v = np.linalg.eigh(C)
    load = v[:, -1] * np.sign(v[:, -1].sum())
    return w[-1] / w.sum(), load


def das_published(stays):
    f = pd.read_parquet(s15.ROW / "eicu_flags.parquet")
    f = f[f.patientunitstayid.isin(stays)]
    out = {}
    for cond in PUBLISHED:
        code = np.zeros(len(f), int)
        for s in ["D", "H", "M"]:
            code = code * 2 + f[f"{cond}_{s}"].to_numpy(int)
        lobe = np.where(code > 0, 7 - code, -1)
        cnt = pd.crosstab(f.hospitalid[lobe >= 0], lobe[lobe >= 0]).reindex(columns=range(7), fill_value=0)
        cnt = cnt[cnt.sum(axis=1) >= s15.CFG["das"]["min_records_per_hospital"]]
        ref, v = cnt.sum(axis=0).to_numpy(float), cnt.to_numpy(float)
        cos = (v @ ref) / (np.linalg.norm(v, axis=1) * np.linalg.norm(ref))
        out[cond] = pd.Series(np.degrees(np.arccos(np.clip(cos, -1, 1))), index=cnt.index)
    return pd.DataFrame(out).median(axis=1)


def main():
    e = pd.read_parquet(s15.ROW / "eicu_positive_controls.parquet")
    lpd = e.groupby("hospitalid").apply(lambda g: g.n_lab.sum() / g.icu_days.sum(), include_groups=False)
    e["labs_pd_z"] = e.hospitalid.map((lpd - lpd.mean()) / lpd.std())
    fr = frames(e)
    elig = set.intersection(*[set(c[c >= MIN_STAYS].index) for c in (d.hospitalid.value_counts() for d in fr.values())])
    print(f"eligible hospitals: {len(elig)}")
    rows, load_rows = [], []
    eff = {}
    for version, extra in [("primary", []), ("adjusted_labs_per_day", ["labs_pd_z"])]:
        U = {}
        for cond, d in fr.items():
            u, sd = hospital_modes(d, ["sev", "age10", "female"] + extra)
            U[cond] = u
        U = pd.DataFrame(U).loc[sorted(elig)]
        eff[version] = U
        Z = ((U - U.mean()) / U.std()).to_numpy()
        # P9 pairwise Spearman, hospital bootstrap on the fixed modes
        iu = np.triu_indices(len(CONDS), 1)
        med = np.median(stats.spearmanr(Z).statistic[iu])
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(Z), len(Z))
            bs.append(np.median(stats.spearmanr(Z[i]).statistic[iu]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p9 = s15.boot_p(bs, 0)
        rows.append(dict(prediction="P9", version=version, test="median pairwise Spearman of hospital effects",
                         estimate=med, lo=lo, hi=hi, p=p9, criterion_met=(med > 0) and (lo > 0), in_holm=version == "primary",
                         n_hospitals=len(Z)))
        # P10 PC1 share vs per-condition permutation
        share, load = pc1(Z)
        null = []
        for _ in range(PERM):
            Zp = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
            null.append(pc1(Zp)[0])
        p10 = (1 + np.sum(np.array(null) >= share)) / (1 + PERM)
        rows.append(dict(prediction="P10", version=version, test="PC1 variance share vs permutation null",
                         estimate=share, lo=np.percentile(null, 2.5), hi=np.percentile(null, 97.5), p=p10,
                         criterion_met=p10 < 0.05, in_holm=version == "primary", null_median=np.median(null), n_hospitals=len(Z)))
        for c, l in zip(CONDS, load):
            load_rows.append(dict(version=version, condition=c, pc1_loading=l))
        score = pd.Series(Z @ load, index=U.index)
        # P11 descriptive correlates
        hosp = pd.read_parquet(f"{s15.E}/hospital.parquet").set_index("hospitalid")
        size = e.groupby("hospitalid").size()
        for name, x in [("labs per patient-day", lpd), ("cohort stays", size)]:
            r = stats.spearmanr(score, x.reindex(score.index), nan_policy="omit").statistic
            rows.append(dict(prediction="P11", version=version, test=f"Spearman(PC1, {name})", estimate=r, in_holm=False))
        for col in ["numbedscategory", "teachingstatus", "region"]:
            g = pd.DataFrame(dict(s=score, v=hosp[col].reindex(score.index).astype(str))).groupby("v").s
            for lev, sub in g:
                rows.append(dict(prediction="P11", version=version, test=f"median PC1, {col} = {lev}",
                                 estimate=sub.median(), n_hospitals=len(sub), in_holm=False))
        # P12 |PC1| vs median published-condition DAS
        das = das_published(set(e.patientunitstayid))
        j = score.index.intersection(das.index)
        x, y = np.abs(score[j].to_numpy()), das[j].to_numpy()
        rho = stats.spearmanr(x, y).statistic
        bs = []
        for _ in range(B12):
            i = rng.integers(0, len(x), len(x))
            bs.append(stats.spearmanr(x[i], y[i]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(prediction="P12", version=version, test="Spearman(|PC1|, median DHM angle to pooled eICU)",
                         estimate=rho, lo=lo, hi=hi, p=s15.boot_p(bs, 0), criterion_met=(rho > 0) and (lo > 0),
                         in_holm=version == "primary", n_hospitals=len(j)))
        # signed PC1 vs DAS, reported
        rows.append(dict(prediction="P12", version=version, test="Spearman(signed PC1, median DHM angle), reported",
                         estimate=stats.spearmanr(score[j], y).statistic, in_holm=False, n_hospitals=len(j)))
    out = pd.DataFrame(rows)
    h = out[out.in_holm]
    order = np.argsort(h.p.to_numpy())
    adj = np.minimum(1, np.maximum.accumulate((len(h) - np.arange(len(h))) * h.p.to_numpy()[order]))
    out.loc[h.index[order], "p_holm"] = adj
    out["passed"] = np.where(out.in_holm, out.criterion_met.astype(bool) & (out.p_holm < 0.05), np.nan)
    out.to_csv(s15.RES / "shared_habit.csv", index=False)
    lt = pd.DataFrame(load_rows)
    corr = eff["primary"].corr(method="spearman").round(3)
    lt.to_csv(s15.RES / "shared_habit_loadings.csv", index=False)
    corr.to_csv(s15.RES / "shared_habit_correlations.csv")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string())
    print(lt.round(3).to_string())
    print(corr.to_string())


if __name__ == "__main__":
    main()
