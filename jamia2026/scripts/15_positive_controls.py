#!/usr/bin/env python
"""Positive controls B5-B8 (predictions P1-P8), as locked in configs/screen_settings.yaml.

B5  race-based eGFR and CKD documentation (MIMIC-IV 3.1 primary, eICU secondary)
B6  obesity documentation against measured BMI (eICU)
B7  lab-defined acute conditions by severity stage (eICU)
B8  labs per patient-day as a hospital-level adjustment (eICU)

Documentation only, no outcomes. Row-level tables go to the external drive. results/ gets
aggregates only, with counts below 10 suppressed.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scipy \
      --with statsmodels python scripts/15_positive_controls.py [--boot 1000] [--rebuild]
"""

from __future__ import annotations

import argparse
import sys
import multiprocessing as mp
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glmm  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE / "configs/external.yaml"))
SET = yaml.safe_load(open(BASE / "configs/screen_settings.yaml"))
assert SET["positive_controls_status"] == "locked"
E = CFG["paths"]["eicu"]
M31 = "/Volumes/Crucial X9/research_data/mimic-iv/3.1-parquet"
ROW = Path(CFG["paths"]["rowlevel"])
RES = BASE / CFG["paths"]["results"]
SEED = 20261106
PL = CFG["features"]["plausible"]

CKD_MEDS = ["sevelamer", "renvela", "renagel", "calcium acetate", "phoslo", "lanthanum", "fosrenol", "calcitriol",
            "paricalcitol", "zemplar", "doxercalciferol", "hectorol", "cinacalcet", "sensipar", "epoetin", "epogen",
            "procrit", "darbepoetin", "aranesp", "ferric citrate", "auryxia", "velphoro"]
ACUTE = {  # documentation (diagnosis table) and worst-value staging
    "aki": dict(dx=["acute renal failure", "acute kidney injury"]),
    "hyponatremia": dict(dx=["hyponatremia"], lab="sodium", cuts=[(130, 135), (125, 130), (-np.inf, 125)]),
    "anemia": dict(dx=["anemia"], lab="hgb", cuts=[(10, 11), (8, 10), (-np.inf, 8)]),
    "thrombocytopenia": dict(dx=["thrombocytopenia"], lab="platelets", cuts=[(100, 150), (50, 100), (-np.inf, 50)]),
}


def like_any(col, terms):
    return "(" + " OR ".join(f"lower({col}) LIKE '%{t.lower()}%'" for t in terms) + ")"


def egfr_2021(scr, age, female):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1.0)


# ----------------------------------------------------------------------------- data

def build_eicu(con) -> pd.DataFrame:
    con.execute(f"""
        CREATE OR REPLACE TABLE ecoh AS
        WITH p AS (
          SELECT patientunitstayid, patienthealthsystemstayid, uniquepid, hospitalid, gender, ethnicity,
                 CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_n,
                 admissionheight h, admissionweight w, unitdischargeoffset, hospitaldischargeyear, unitvisitnumber
          FROM '{E}/patient.parquet'),
        q AS (SELECT * FROM p WHERE age_n >= {CFG['cohort']['min_age']} AND unitdischargeoffset > {CFG['cohort']['landmark_hours'] * 60})
        SELECT * EXCLUDE (rn) FROM (
          SELECT *, row_number() OVER (PARTITION BY uniquepid
                   ORDER BY hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber) rn FROM q)
        WHERE rn = 1""")
    labmap = {"creatinine": "creatinine", "sodium": "sodium", "hgb": "Hgb", "platelets": "platelets x 1000"}
    case = " ".join(f"WHEN labname = '{v}' THEN '{k}'" for k, v in labmap.items())
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE el AS
        SELECT l.patientunitstayid, CASE {case} END AS lab, labresult v, labresultoffset t
        FROM '{E}/lab.parquet' l JOIN ecoh c USING (patientunitstayid)
        WHERE labname IN ({", ".join(f"'{v}'" for v in labmap.values())})
          AND labresultoffset BETWEEN -360 AND c.unitdischargeoffset AND labresult IS NOT NULL""")
    plaus = " OR ".join(f"(lab = '{k}' AND v BETWEEN {PL[k][0]} AND {PL[k][1]})" for k in labmap)
    labs = con.execute(f"""
        SELECT patientunitstayid,
          min(v) FILTER (lab = 'creatinine') cr_min, max(v) FILTER (lab = 'creatinine') cr_max,
          arg_min(v, t) FILTER (lab = 'creatinine') cr_first, count(*) FILTER (lab = 'creatinine') cr_n,
          min(v) FILTER (lab = 'sodium') na_min, min(v) FILTER (lab = 'hgb') hgb_min,
          min(v) FILTER (lab = 'platelets') plt_min
        FROM el WHERE {plaus} GROUP BY 1""").df()
    nlab = con.execute(f"""SELECT patientunitstayid, count(*) n_lab FROM '{E}/lab.parquet' l JOIN ecoh c USING (patientunitstayid)
                           WHERE labresultoffset BETWEEN 0 AND c.unitdischargeoffset GROUP BY 1""").df()

    def ids(sql):
        return set(con.execute(sql).df().iloc[:, 0])

    dx = f"SELECT DISTINCT patientunitstayid FROM '{E}/diagnosis.parquet' JOIN ecoh USING (patientunitstayid) WHERE "
    ph = f"SELECT DISTINCT patientunitstayid FROM '{E}/pastHistory.parquet' JOIN ecoh USING (patientunitstayid) WHERE "
    flags = {
        "ckd_D": ids(dx + like_any("diagnosisstring", ["chronic kidney disease", "chronic renal insufficiency", "ESRD"])),
        "ckd_H": ids(ph + "(pasthistorypath LIKE '%/Renal Insufficiency/%' OR pasthistorypath LIKE '%/Renal Failure/%')"),
        "ckd_M": ids(f"SELECT DISTINCT patientunitstayid FROM '{E}/medication.parquet' JOIN ecoh USING (patientunitstayid) WHERE "
                     + like_any("drugname", CKD_MEDS)),
        "ckd_A": ids(f"SELECT DISTINCT patientunitstayid FROM '{E}/admissionDrug.parquet' JOIN ecoh USING (patientunitstayid) WHERE "
                     + like_any("drugname", CKD_MEDS)),
        "renal_excl": ids(dx + like_any("diagnosisstring", ["ESRD", "kidney transplant", "hemodialysis"]))
        | ids(ph + like_any("pasthistorypath", ["hemodialysis", "peritoneal dialysis", "Renal Transplant"])),
        "obesity_doc": ids(dx + like_any("diagnosisstring", ["|obesity"])),
    }
    for c, d in ACUTE.items():
        flags[f"{c}_doc"] = ids(dx + like_any("diagnosisstring", d["dx"]))
    coh = con.execute("SELECT * FROM ecoh").df().merge(labs, how="left").merge(nlab, how="left")
    for k, s in flags.items():
        coh[k] = coh.patientunitstayid.isin(s)
    coh["female"] = (coh.gender == "Female").astype(float).where(coh.gender.isin(["Female", "Male"]))
    coh["black"] = (coh.ethnicity == "African American").astype(float).where(~coh.ethnicity.isin([None, "Other/Unknown"]) & coh.ethnicity.notna())
    ok = coh.h.between(120, 220) & coh.w.between(30, 300)
    coh["bmi"] = (coh.w / (coh.h / 100) ** 2).where(ok)
    coh["icu_days"] = coh.unitdischargeoffset / 1440
    coh["n_lab"] = coh.n_lab.fillna(0)
    print(f"eICU cohort {len(coh):,} stays, {coh.hospitalid.nunique()} hospitals")
    return coh


def build_mimic(con) -> pd.DataFrame:
    con.execute(f"""
        CREATE OR REPLACE TABLE mcoh AS
        WITH s AS (
          SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.los, a.admittime, a.race, p.gender,
                 p.anchor_age + year(a.admittime) - p.anchor_year AS age,
                 CAST(left(p.anchor_year_group, 4) AS INTEGER) + year(a.admittime) - p.anchor_year AS yr_lo
          FROM '{M31}/icu/icustays.parquet' i JOIN '{M31}/hosp/admissions.parquet' a USING (subject_id, hadm_id)
          JOIN '{M31}/hosp/patients.parquet' p USING (subject_id))
        SELECT * EXCLUDE (rn) FROM (
          SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY intime) rn FROM s
          WHERE age >= {CFG['cohort']['min_age']} AND los >= {CFG['cohort']['landmark_hours'] / 24})
        WHERE rn = 1""")
    prior = con.execute(f"""
        WITH l AS (
          SELECT l.subject_id, l.hadm_id lh, l.charttime, l.valuenum, c.hadm_id, c.admittime
          FROM '{M31}/hosp/labevents.parquet' l JOIN mcoh c USING (subject_id)
          WHERE l.itemid = 50912 AND l.valuenum BETWEEN {PL['creatinine'][0]} AND {PL['creatinine'][1]}
            AND l.charttime < c.admittime AND l.charttime >= c.admittime - INTERVAL 365 DAY),
        ed AS (SELECT e.subject_id, e.hadm_id eh, e.intime, e.outtime FROM '{M31}/ed/edstays.parquet' e)
        SELECT hadm_id, arg_max(valuenum, charttime) cr_prior FROM l
        WHERE (lh IS NOT NULL AND lh <> hadm_id)
           OR EXISTS (SELECT 1 FROM ed WHERE ed.subject_id = l.subject_id AND l.charttime BETWEEN ed.intime AND ed.outtime
                      AND (ed.eh IS NULL OR ed.eh <> l.hadm_id))
        GROUP BY 1""").df()
    dx = con.execute(f"""
        SELECT d.hadm_id,
          max(CASE WHEN (icd_version = 9 AND (regexp_matches(icd_code, '^585[1-59]') OR regexp_matches(icd_code, '^40[34]')))
                     OR (icd_version = 10 AND (regexp_matches(icd_code, '^N18[1-59]') OR regexp_matches(icd_code, '^I1[23]')))
              THEN 1 ELSE 0 END) ckd_doc,
          max(CASE WHEN icd_code IN ('5856', 'N186', 'V4511', 'Z992', 'V420', 'Z940') THEN 1 ELSE 0 END) renal_excl
        FROM '{M31}/hosp/diagnoses_icd.parquet' d JOIN mcoh USING (hadm_id) GROUP BY 1""").df()
    coh = con.execute("SELECT * FROM mcoh").df().merge(prior, how="left").merge(dx, how="left")
    coh[["ckd_doc", "renal_excl"]] = coh[["ckd_doc", "renal_excl"]].fillna(0).astype(bool)
    unknown = coh.race.isin(["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]) | coh.race.isna()
    coh["black"] = coh.race.str.startswith("BLACK").astype(float).where(~unknown)
    coh["female"] = (coh.gender == "F").astype(float)
    coh["period"] = np.select([coh.yr_lo.between(2011, 2017), coh.yr_lo >= 2020], ["2011-2019", "2020-2022"], "other")
    print(f"MIMIC-IV 3.1 cohort {len(coh):,} stays, prior creatinine for {coh.cr_prior.notna().sum():,}")
    return coh


# ----------------------------------------------------------------------------- models

def cells(df, xcols, ycol, gcol=None):
    keys = xcols + ([gcol] if gcol else [])
    a = df.groupby(keys, observed=True)[ycol].agg(["sum", "count"]).reset_index()
    X = np.c_[np.ones(len(a)), a[xcols].to_numpy(float)]
    g = pd.factorize(a[gcol])[0] if gcol else None
    return X, a["sum"].to_numpy(float), a["count"].to_numpy(float), g, a


def logit(df, xcols, ycol):
    X, y, n, _, _ = cells(df, xcols, ycol)
    r = sm.GLM(np.c_[y, n - y], X, family=sm.families.Binomial()).fit()
    return dict(zip(["const"] + xcols, r.params)), dict(zip(["const"] + xcols, r.bse)), r.cov_params()


def mixed(df, xcols, ycol, gcol="hospitalid", se=True):
    X, y, n, g, _ = cells(df, xcols, ycol, gcol)
    r = glmm.fit(X, y, n, g, se=se)
    names = ["const"] + xcols
    return dict(beta=dict(zip(names, r["beta"])), se=dict(zip(names, r.get("se_beta", [np.nan] * len(names)))), sd=r["sd"])


def wald(b, se, null=0.0):
    z = (b - null) / se
    return b, b - 1.96 * se, b + 1.96 * se, 2 * stats.norm.sf(abs(z))


def boot_p(est_b, null):
    est_b = np.asarray(est_b)
    return max(min(1.0, 2 * min((est_b <= null).mean(), (est_b >= null).mean())), 1 / len(est_b))


_W = {}


def _boot_sd(args):
    """One hospital bootstrap draw: SD for each (subset, xcols) model on the same resampled hospitals."""
    i, specs = args
    rng = np.random.default_rng(SEED + i)
    hosp = _W["hosp"]
    pick = rng.choice(hosp, len(hosp), replace=True)
    out = []
    for key, xcols, ycol in specs:
        by = _W[key]
        parts = [by[h].assign(hospitalid=j) for j, h in enumerate(pick) if h in by]
        out.append(mixed(pd.concat(parts), xcols, ycol, se=False)["sd"] if parts else np.nan)
    return out


def bootstrap_sds(frames: dict, specs: list, B: int, procs: int = 8) -> np.ndarray:
    _W.clear()
    _W.update({k: dict(tuple(f.groupby("hospitalid"))) for k, f in frames.items()})
    _W["hosp"] = np.unique(np.concatenate([f.hospitalid.unique() for f in frames.values()]))
    with mp.get_context("fork").Pool(procs) as pool:
        return np.array(pool.map(_boot_sd, [(i, specs) for i in range(B)]))


# ----------------------------------------------------------------------------- analyses

def stage_aki(d):
    ratio, rise = d.cr_max / d.cr_first, d.cr_max - d.cr_first
    st = np.select([(ratio >= 3) | ((d.cr_max >= 4) & (rise >= 0.3)), ratio >= 2, (ratio >= 1.5) | (rise >= 0.3)], [3, 2, 1], 0)
    return pd.Series(np.where((d.cr_n >= 2), st, np.nan), index=d.index)


def stage_lab(v, cuts):
    st = np.full(len(v), np.nan)
    st[v.notna().to_numpy()] = 0
    for s, (lo, hi) in enumerate(cuts, start=1):
        st[(v >= lo).to_numpy() & (v < hi).to_numpy()] = s
    return st


def p1_p2_frames(d, cr_col, ckd_col):
    d = d[~d.renal_excl & d[cr_col].notna() & d.black.notna() & d.female.notna()].copy()
    d["egfr"] = egfr_2021(d[cr_col].to_numpy(), d.age_n.to_numpy() if "age_n" in d else d.age.to_numpy(), d.female.to_numpy() == 1)
    d["age10"] = (d.age_n if "age_n" in d else d.age) // 1 / 10
    d["egfr10"] = (d.egfr / 10).round(1)
    d["y"] = d[ckd_col].astype(int)
    band = d[(d.egfr >= 45) & (d.egfr < 60)]
    both = d[((d.egfr >= 45) & (d.egfr < 60)) | (d.egfr < 30)].copy()
    both["flip"] = (both.egfr >= 45).astype(int)
    both["black_flip"] = both.black * both.flip
    return band, both


def run(args):
    con = duckdb.connect()
    ep, mp = ROW / "eicu_positive_controls.parquet", ROW / "mimic31_positive_controls.parquet"
    if args.rebuild or not ep.exists():
        build_eicu(con).to_parquet(ep)
    if args.rebuild or not mp.exists():
        build_mimic(con).to_parquet(mp)
    e, m = pd.read_parquet(ep), pd.read_parquet(mp)
    e["age10"] = e.age_n / 10
    lpd = e.groupby("hospitalid").apply(lambda g: g.n_lab.sum() / g.icu_days.sum(), include_groups=False)
    e["labs_pd_z"] = e.hospitalid.map((lpd - lpd.mean()) / lpd.std())
    rows, desc = [], []

    def add(pred, data, label, est, lo, hi, p, passed, in_holm, **kw):
        rows.append(dict(prediction=pred, data=data, test=label, estimate=est, lo=lo, hi=hi, p=p,
                         criterion_met=passed, in_holm=in_holm, **kw))

    # B5: P1, P2 (MIMIC-IV primary, eICU secondary), P3
    for data, d, cr, ckd, primary in [("MIMIC-IV 3.1", m, "cr_prior", "ckd_doc", True),
                                      ("eICU", e.assign(ckd_doc=e.ckd_D | e.ckd_H), "cr_min", "ckd_doc", False)]:
        band, both = p1_p2_frames(d, cr, ckd)
        xc = ["black", "age10", "female", "egfr10"]
        if primary:
            b, s, _ = logit(band, xc, "y")
        else:
            r = mixed(band, xc, "y"); b, s = r["beta"], r["se"]
        est, lo, hi, p = wald(b["black"], s["black"])
        add("P1", data, "OR Black vs non-Black, eGFR 45-60", np.exp(est), np.exp(lo), np.exp(hi), p, np.exp(hi) < 1, primary,
            n=len(band), n_black=int(band.black.sum()))
        xc2 = ["black", "flip", "black_flip", "age10", "female", "egfr10"]
        if primary:
            b, s, _ = logit(both, xc2, "y")
        else:
            r = mixed(both, xc2, "y"); b, s = r["beta"], r["se"]
        ie, il, ih, ip = wald(b["black_flip"], s["black_flip"])
        ce, cl, chh, cp = wald(b["black"], s["black"])
        add("P2", data, "interaction Black x flip band (log OR)", ie, il, ih, ip, (ih < 0) and (np.exp(cl) <= 1 <= np.exp(chh)), primary,
            control_band_or=np.exp(ce), control_band_or_lo=np.exp(cl), control_band_or_hi=np.exp(chh), n=len(both))
        for grp, dd in [("band 45-60", band), ("band <30", both[both.flip == 0])]:
            for bl, ddd in dd.groupby("black"):
                desc.append(dict(block="B5", data=data, group=grp, level="Black" if bl else "non-Black",
                                 n=len(ddd), documented=ddd.y.mean()))
    for per in ["2011-2019", "2020-2022"]:
        band, _ = p1_p2_frames(m[m.period == per], "cr_prior", "ckd_doc")
        b, s, _ = logit(band, ["black", "age10", "female", "egfr10"], "y")
        est, lo, hi, p = wald(b["black"], s["black"])
        add("P3", "MIMIC-IV 3.1", f"OR Black vs non-Black, eGFR 45-60, {per} (exploratory)", np.exp(est), np.exp(lo), np.exp(hi), p,
            np.nan, False, n=len(band), n_black=int(band.black.sum()))

    # B5: P4, hospital CKD DAS vs Black share (eICU hospitals >= 100 stays), no race adjustment
    big = e.groupby("hospitalid").filter(lambda g: len(g) >= 100)
    rng = np.random.default_rng(SEED)
    for cname, srcs in [("DHM", ["D", "H", "M"]), ("DHA", ["D", "H", "A"])]:
        code = np.zeros(len(big), int)
        for s in srcs:
            code = code * 2 + big[f"ckd_{s}"].to_numpy(int)
        lobe = np.where(code > 0, 7 - code, -1)
        cnt = pd.crosstab(big.hospitalid[lobe >= 0], lobe[lobe >= 0]).reindex(columns=range(7), fill_value=0)
        ref = cnt.sum(axis=0).to_numpy(float)
        v = cnt.to_numpy(float)
        cos = (v @ ref) / (np.linalg.norm(v, axis=1) * np.linalg.norm(ref))
        das = pd.Series(np.degrees(np.arccos(np.clip(cos, -1, 1))), index=cnt.index)
        share = big.groupby("hospitalid").black.mean().reindex(das.index)
        keep = share.notna()
        x, y = das[keep].to_numpy(), share[keep].to_numpy()
        rho = stats.spearmanr(x, y).statistic
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            bs.append(stats.spearmanr(x[i], y[i]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        add("P4", "eICU", f"Spearman(CKD DAS {cname} to pooled eICU, Black share)", rho, lo, hi, boot_p(bs, 0),
            (rho > 0) and (lo > 0), cname == "DHM", n=int(keep.sum()))

    B = args.boot
    # B6: P5, P6 (obesity among BMI >= 30)
    ob = e[e.bmi >= 30].dropna(subset=["female"]).copy()
    ob["cls"] = np.select([ob.bmi < 35, ob.bmi < 40], [0, 1], 2)
    ob["age10"] = ob.age_n // 1 / 10
    ob["y"] = ob.obesity_doc.astype(int)
    for c, dd in ob.groupby("cls"):
        desc.append(dict(block="B6", data="eICU", group="obesity", level=["30-34.9", "35-39.9", ">=40"][c], n=len(dd), documented=dd.y.mean()))
    r = mixed(ob, ["cls", "age10", "female"], "y")
    est, lo, hi, p = wald(r["beta"]["cls"], r["se"]["cls"])
    add("P5", "eICU", "trend OR per BMI class", np.exp(est), np.exp(lo), np.exp(hi), p, np.exp(lo) > 1, True, n=len(ob))
    r_adj = mixed(ob, ["cls", "age10", "female", "labs_pd_z"], "y", se=False)
    bs = bootstrap_sds({"ob": ob}, [("ob", ["cls", "age10", "female"], "y"), ("ob", ["cls", "age10", "female", "labs_pd_z"], "y")], B)
    lo, hi = np.nanpercentile(bs[:, 0], [2.5, 97.5])
    add("P6", "eICU", "hospital SD (logit), same BMI class", r["sd"], lo, hi, boot_p(bs[:, 0], 0.3), (r["sd"] >= 0.5) and (lo > 0.3), True,
        n=len(ob), n_hospitals=ob.hospitalid.nunique())
    lo, hi = np.nanpercentile(bs[:, 1], [2.5, 97.5])
    add("P6", "eICU", "hospital SD, adjusted for labs per patient-day (B8)", r_adj["sd"], lo, hi, boot_p(bs[:, 1], 0.3), np.nan, False)

    # B7: P7, P8 per condition
    e2 = e.dropna(subset=["female"]).copy()
    e2["age10"] = e2.age_n // 1 / 10
    for cond, d in ACUTE.items():
        if cond == "aki":
            st = stage_aki(e2[~e2.renal_excl])
            dd = e2.loc[st.index].assign(stage=st)
        else:
            dd = e2.assign(stage=stage_lab(e2[{"sodium": "na_min", "hgb": "hgb_min", "platelets": "plt_min"}[d["lab"]]], d["cuts"]))
        dd = dd[dd.stage >= 1].copy()
        dd["y"] = dd[f"{cond}_doc"].astype(int)
        for s, x in dd.groupby("stage"):
            desc.append(dict(block="B7", data="eICU", group=cond, level=f"stage {int(s)}", n=len(x), documented=x.y.mean()))
        r = mixed(dd, ["stage", "age10", "female"], "y")
        est, lo, hi, p = wald(r["beta"]["stage"], r["se"]["stage"])
        add("P7", "eICU", f"{cond}: trend OR per stage", np.exp(est), np.exp(lo), np.exp(hi), p, np.exp(lo) > 1, True,
            n=len(dd), top_stage_documented=dd.loc[dd.stage == 3, "y"].mean())
        mild, sev = dd[dd.stage == 1], dd[dd.stage == 3]
        sds = [mixed(x, cols, "y", se=False)["sd"] for x in (mild, sev)
               for cols in (["age10", "female"], ["age10", "female", "labs_pd_z"])]
        bs = bootstrap_sds({"mild": mild, "sev": sev},
                           [("mild", ["age10", "female"], "y"), ("sev", ["age10", "female"], "y"),
                            ("mild", ["age10", "female", "labs_pd_z"], "y"), ("sev", ["age10", "female", "labs_pd_z"], "y")], B)
        for adj, (a, b_), cols in [(False, (0, 1), None), (True, (2, 3), None)]:
            ratio = (sds[0] / sds[2]) if not adj else (sds[1] / sds[3])
            rb = bs[:, a] / bs[:, b_]
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            add("P8", "eICU", f"{cond}: SD mild / SD severe" + (", adjusted for labs per patient-day (B8)" if adj else ""),
                ratio, lo, hi, boot_p(rb[np.isfinite(rb)], 1), (lo > 1) if not adj else np.nan, not adj,
                sd_mild=sds[0] if not adj else sds[1], sd_severe=sds[2] if not adj else sds[3], n_mild=len(mild), n_severe=len(sev))
        print(f"  done {cond}", flush=True)

    out = pd.DataFrame(rows)
    h = out[out.in_holm]
    order = np.argsort(h.p.to_numpy())
    adj = np.minimum(1, np.maximum.accumulate((len(h) - np.arange(len(h))) * h.p.to_numpy()[order]))
    out.loc[h.index[order], "p_holm"] = adj
    out["passed"] = np.where(out.in_holm, out.criterion_met.astype(bool) & (out.p_holm < 0.05), np.nan)
    out.to_csv(RES / "positive_controls.csv", index=False)
    dsc = pd.DataFrame(desc)
    dsc["documented"] = dsc.documented.where(dsc.n >= 10)
    dsc["n"] = dsc.n.astype(object).where(dsc.n >= 10, "<10")
    dsc.to_csv(RES / "positive_controls_descriptive.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["prediction", "data", "test", "estimate", "lo", "hi", "p", "p_holm", "criterion_met", "passed"]].round(4).to_string())
    print(dsc.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true")
    run(ap.parse_args())
