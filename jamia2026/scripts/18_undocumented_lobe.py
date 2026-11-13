#!/usr/bin/env python
"""B9 undocumented lobe (predictions P13-P18), as locked in configs/screen_settings.yaml.

Part 1: diabetes DHM-U = published DHM lobes + U (glucose anchor positive, no D/H/M record).
Part 2: anchored completeness profile over obesity and four lab-defined conditions (eICU).
Noise floor as in script 02: split each hospital's stays into random halves 20 times; the share of
between-hospital half angles above the 95th percentile of within-hospital half angles.
Documentation only, no outcomes. Aggregates only in results/.

Usage (from mimic_external/), after scripts 02, 15 and 17:
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scipy \
      --with statsmodels python scripts/18_undocumented_lobe.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s16", HERE / "16_shared_habit.py")
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)
s15, glmm = s16.s15, s16.glmm
LOCK = s15.SET["B9_undocumented_lobe"]
assert LOCK["status"].startswith("locked")
E, ROW, RES, CFG = s15.E, s15.ROW, s15.RES, s15.CFG
M = CFG["paths"]["mimic"]
GLU_ITEMS = LOCK["part1_diabetes"]["mimic_itemids"]
STEROIDS = ["hydrocortisone", "methylprednisolone", "prednisone", "prednisolone", "dexamethasone", "solu-medrol",
            "solu-cortef", "decadron"]
NOT_SYSTEMIC = ["inhal", "neb", "topical", "ophth", "otic", "nasal", "rectal", "intra-artic", "IH", "TP", "PR"]
PUBLISHED = ["diabetes", "hypertension", "chf", "af"]
SRCS = ("D", "H", "M")
N_SPLITS, B13, B14, MIN_N = 20, 1000, 2000, 100
rng = np.random.default_rng(s15.SEED + 9)


def like_any(col, terms):
    return "(" + " OR ".join(f"lower({col}) LIKE '%{t.lower()}%'" for t in terms) + ")"


def codes(df, cond, srcs, anchor=None):
    """Lobe index: 2^n - 1 published lobes in script 02 order, then U = 2^n - 1; -1 = not in population."""
    code = np.zeros(len(df), int)
    for s in srcs:
        code = code * 2 + df[f"{cond}_{s}"].to_numpy(bool)
    out = np.where(code > 0, 2 ** len(srcs) - 1 - code, -1)
    if anchor is not None:
        out = np.where((out < 0) & np.asarray(anchor, bool), 2 ** len(srcs) - 1, out)
    return out


def unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, n, out=np.zeros_like(A), where=n > 0)


def vec(hidx, lob, H, L):
    A = np.zeros((H, L))
    np.add.at(A, (hidx, lob), 1)
    return A


def split_angles(comps, H, Uni):
    """comps: list of (stay position, hospital index, lobe, n lobes). Returns splits x H x H half-angle matrices."""
    X = []
    for s in range(N_SPLITS):
        PA, PB = [], []
        for pos, hidx, lob, L in comps:
            half = Uni[s, pos] < 0.5
            PA.append(unit(vec(hidx[half], lob[half], H, L)))
            PB.append(unit(vec(hidx[~half], lob[~half], H, L)))
        A, B = unit(np.hstack(PA)), unit(np.hstack(PB))
        X.append(np.degrees(np.arccos(np.clip(A @ B.T, -1, 1))))
    return np.array(X)


def noise_share(X, idx):
    sub = X[:, idx][:, :, idx]
    k = len(idx)
    within = sub[:, np.arange(k), np.arange(k)].ravel()
    diff = idx[:, None] != idx[None, :]
    between = sub[:, diff].ravel()
    return np.mean(between > np.percentile(within, 95))


def comp(df, lob, hosp_index, L):
    keep = (lob >= 0) & df.hospitalid.isin(hosp_index).to_numpy()
    return (np.flatnonzero(keep), pd.Index(hosp_index).get_indexer(df.hospitalid.to_numpy()[keep]), lob[keep], L)


def profile(comps, H):
    return unit(np.hstack([unit(vec(h, l, H, L)) for _, h, l, L in comps]))


def angle_to(V, ref):
    return np.degrees(np.arccos(np.clip(unit(V) @ (ref / np.linalg.norm(ref)), -1, 1)))


def boot_diff(f, n, B):
    out = []
    for _ in range(B):
        out.append(f(rng.integers(0, n, n)))
    return np.array(out)


# ----------------------------------------------------------------------------- data

def eicu_part1(con):
    f = pd.read_parquet(ROW / "eicu_flags.parquet")
    con.register("ef", f[["patientunitstayid"]])
    g = con.sql(f"""
        SELECT l.patientunitstayid, count(*) n_glu,
          count(*) FILTER (labresult >= 200) n200,
          max(labresultoffset) FILTER (labresult >= 200) - min(labresultoffset) FILTER (labresult >= 200) span200
        FROM '{E}/lab.parquet' l JOIN ef USING (patientunitstayid) JOIN '{E}/patient.parquet' p USING (patientunitstayid)
        WHERE labname IN ('glucose', 'bedside glucose') AND labresult BETWEEN 1 AND 3000
          AND labresultoffset BETWEEN -360 AND p.unitdischargeoffset
        GROUP BY 1""").df()
    st = con.sql(f"""
        SELECT DISTINCT m.patientunitstayid FROM '{E}/medication.parquet' m JOIN ef USING (patientunitstayid)
        JOIN '{E}/patient.parquet' p USING (patientunitstayid)
        WHERE {like_any('drugname', STEROIDS)} AND NOT {like_any("coalesce(routeadmin, '')", NOT_SYSTEMIC[:8])}
          AND drugstartoffset BETWEEN -360 AND p.unitdischargeoffset""").df()
    p = con.sql(f"""SELECT patientunitstayid, unitdischargeoffset / 1440.0 icu_days, gender,
                     CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END age_n
                     FROM '{E}/patient.parquet'""").df()
    f = f.merge(g, how="left").merge(p, how="left")
    f[["n_glu", "n200"]] = f[["n_glu", "n200"]].fillna(0)
    f["steroid"] = f.patientunitstayid.isin(st.patientunitstayid)
    return f


def mimic_part1(con):
    f = pd.read_parquet(ROW / "mimic_flags.parquet")
    f = f.merge(pd.read_parquet(ROW / "mimic_history_flags.parquet")[["stay_id"] + [f"{c}_H" for c in PUBLISHED]])
    f = f.merge(pd.read_parquet(ROW / "mimic_split.parquet")[["stay_id", "reference"]])
    con.register("mf", f[["stay_id", "hadm_id"]])
    g = con.sql(f"""
        SELECT i.stay_id, count(*) n_glu, count(*) FILTER (valuenum >= 200) n200,
          epoch(max(charttime) FILTER (valuenum >= 200) - min(charttime) FILTER (valuenum >= 200)) / 60 span200
        FROM '{M}/icu/icustays.parquet' i JOIN mf USING (stay_id)
        JOIN '{M}/hosp/labevents.parquet' l ON l.subject_id = i.subject_id
         AND l.charttime BETWEEN i.intime - INTERVAL 6 HOUR AND i.outtime
        WHERE itemid IN ({", ".join(map(str, GLU_ITEMS))}) AND valuenum BETWEEN 1 AND 3000
        GROUP BY 1""").df()
    st = con.sql(f"""
        SELECT DISTINCT i.stay_id FROM '{M}/icu/icustays.parquet' i JOIN mf USING (stay_id)
        JOIN '{M}/hosp/prescriptions.parquet' r ON r.hadm_id = i.hadm_id
         AND r.starttime BETWEEN i.intime - INTERVAL 6 HOUR AND i.outtime
        WHERE {like_any('drug', STEROIDS)} AND coalesce(route, '') NOT IN ({", ".join(f"'{r}'" for r in NOT_SYSTEMIC[8:])})
          AND NOT {like_any("coalesce(route, '')", NOT_SYSTEMIC[:8])}""").df()
    f = f.merge(g, how="left")
    f[["n_glu", "n200"]] = f[["n_glu", "n200"]].fillna(0)
    f["steroid"] = f.stay_id.isin(st.stay_id)
    years = CFG["split"]["reference_year_groups"]
    return f[f.reference & f.anchor_year_group.isin(years)]


ANCHORS = {
    "primary (>= 2 values >= 200)": lambda d: d.n200 >= 2,
    "one value >= 200": lambda d: d.n200 >= 1,
    "two values >= 24 h apart": lambda d: (d.n200 >= 2) & (d.span200 >= 1440),
    "primary, steroid stays not anchored": lambda d: (d.n200 >= 2) & ~d.steroid,
}


# ----------------------------------------------------------------------------- analyses

def part1(e, m):
    rows, desc, hosp = [], [], {}
    Uni = rng.random((N_SPLITS, len(e)))
    L7, L8 = 7, 8
    dhm = codes(e, "diabetes", SRCS)
    cnt_d = pd.Series(dhm >= 0).groupby(e.hospitalid.to_numpy()).sum()
    Xd = None
    for aname, af in ANCHORS.items():
        anchor = af(e).to_numpy()
        lu = codes(e, "diabetes", SRCS, anchor)
        cnt_u = pd.Series(lu >= 0).groupby(e.hospitalid.to_numpy()).sum()
        H_U = np.array(sorted(cnt_u[cnt_u >= MIN_N].index))
        H_D = np.array([h for h in H_U if cnt_d.get(h, 0) >= MIN_N])
        Xu = split_angles([comp(e, lu, H_D, L8)], len(H_D), Uni)
        if Xd is None:
            Xd = split_angles([comp(e, dhm, H_D, L7)], len(H_D), Uni)
        idx = np.arange(len(H_D))
        su, sd = noise_share(Xu, idx), noise_share(Xd, idx)
        primary = aname.startswith("primary (")
        bs = boot_diff(lambda i: noise_share(Xu, i) - noise_share(Xd, i), len(idx), B13 if primary else 200)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(prediction="P13", anchor=aname, test="noise-floor share, DHM-U minus DHM (paired hospitals)",
                         estimate=su - sd, lo=lo, hi=hi, p=s15.boot_p(bs, 0) if primary else np.nan,
                         criterion_met=(su - sd > 0) and lo > 0, in_holm=primary, value_U=su, value_DHM=sd,
                         n_hospitals=len(H_D)))
        if primary:
            Xu_full = split_angles([comp(e, lu, H_U, L8)], len(H_U), Uni)
            rows.append(dict(prediction="P13", anchor=aname, test="noise-floor share, DHM-U, all DHM-U hospitals",
                             estimate=noise_share(Xu_full, np.arange(len(H_U))), in_holm=False, n_hospitals=len(H_U)))
            # angles to pooled eICU
            ref_u = np.bincount(lu[lu >= 0], minlength=L8).astype(float)
            ref_d = np.bincount(dhm[dhm >= 0], minlength=L7).astype(float)
            cu, hu, lob, _ = comp(e, lu, H_U, L8)
            Vu = vec(hu, lob, len(H_U), L8)
            ang_u = pd.Series(angle_to(Vu, ref_u), index=H_U)
            cd_, hd, lobd, _ = comp(e, dhm, H_D, L7)
            ang_d = pd.Series(angle_to(vec(hd, lobd, len(H_D), L7), ref_d), index=H_D)
            # P14 documentation effect among anchor-positive stays
            a = e[anchor].assign(y=lambda d: (d.diabetes_D | d.diabetes_H | d.diabetes_M).astype(int))
            a = a.assign(age10=a.age_n // 10, female=(a.gender == "Female").astype(int),
                         ln200=np.log(a.n200.clip(upper=20)))
            a = a.dropna(subset=["age10"])
            eff, sd_re = s16.hospital_modes(a, ["age10", "female", "ln200"])
            eff = eff.abs().reindex(H_D)
            ok = eff.notna().to_numpy()
            x, yu, yd = eff.to_numpy()[ok], ang_u.reindex(H_D).to_numpy()[ok], ang_d.to_numpy()[ok]
            r_u, r_d = stats.spearmanr(x, yu).statistic, stats.spearmanr(x, yd).statistic
            bs = boot_diff(lambda i: stats.spearmanr(x[i], yu[i]).statistic - stats.spearmanr(x[i], yd[i]).statistic,
                           len(x), B14)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(prediction="P14", anchor=aname,
                             test="Spearman(angle, |diabetes documentation effect|), DHM-U minus DHM",
                             estimate=r_u - r_d, lo=lo, hi=hi, p=s15.boot_p(bs, 0), criterion_met=(r_u - r_d > 0) and lo > 0,
                             in_holm=True, value_U=r_u, value_DHM=r_d, n_hospitals=int(ok.sum()), re_sd=sd_re))
            # P15 glucose testing
            gpd = e.groupby("hospitalid").apply(lambda d: d.n_glu.sum() / d.icu_days.sum(), include_groups=False)
            r15 = stats.spearmanr(ang_u, gpd.reindex(H_U)).statistic
            rows.append(dict(prediction="P15", anchor=aname, test="Spearman(DHM-U angle, glucose values per patient-day)",
                             estimate=r15, criterion_met=abs(r15) < 0.3, in_holm=False, n_hospitals=len(H_U),
                             value_DHM=stats.spearmanr(ang_d, gpd.reindex(H_D)).statistic))
            hosp["part1"] = pd.DataFrame(dict(angle_U=ang_u, angle_DHM=ang_d, abs_effect=eff, glu_per_day=gpd.reindex(H_U)))
        # descriptive: MIMIC-IV percentile among eICU hospitals, pooled-eICU reference
        mu = codes(m, "diabetes", SRCS, af(m).to_numpy())
        md = codes(m, "diabetes", SRCS)
        ref_u = np.bincount(lu[lu >= 0], minlength=L8).astype(float)
        ref_d = np.bincount(dhm[dhm >= 0], minlength=L7).astype(float)
        _, hu, lob, _ = comp(e, lu, H_D, L8)
        _, hd, lobd, _ = comp(e, dhm, H_D, L7)
        eu, ed = angle_to(vec(hu, lob, len(H_D), L8), ref_u), angle_to(vec(hd, lobd, len(H_D), L7), ref_d)
        m_u = angle_to(np.bincount(mu[mu >= 0], minlength=L8)[None].astype(float), ref_u)[0]
        m_d = angle_to(np.bincount(md[md >= 0], minlength=L7)[None].astype(float), ref_d)[0]
        rows.append(dict(prediction="MIMIC", anchor=aname, test="MIMIC-IV percentile among eICU hospitals (angle to pooled eICU)",
                         value_U=100 * np.mean(eu < m_u), value_DHM=100 * np.mean(ed < m_d), in_holm=False,
                         n_hospitals=len(H_D)))
        for name, d, lobes in [("eICU", e, lu), ("MIMIC-IV 2.2 reference", m, mu)]:
            a = af(d).to_numpy()
            desc.append(dict(anchor=aname, data=name, stays=len(d), anchor_pos=int(a.sum()),
                             anchor_pos_undocumented_pct=100 * np.mean(lobes[a] == 7),
                             U_share_of_vector_pct=100 * np.mean(lobes[lobes >= 0] == 7),
                             steroid_pct=100 * d.steroid.mean()))
    return rows, desc, hosp


def part2(e_pc, e_flags):
    rows = []
    hab = pd.read_parquet(ROW / "eicu_habit_hospital_values.parquet")
    H = np.array(sorted(hab.index))
    con = duckdb.connect()
    con.register("ids", e_pc[["patientunitstayid"]])
    anem_h = set(con.sql(f"""SELECT DISTINCT patientunitstayid FROM '{E}/pastHistory.parquet' JOIN ids USING (patientunitstayid)
                             WHERE {like_any('pasthistorypath', ['anemia'])}""").df().patientunitstayid)
    d = e_pc.dropna(subset=["female"]).copy()
    comps, pooled = [], []
    pos_of = pd.Series(np.arange(len(e_pc)), index=e_pc.patientunitstayid)
    for cond in s16.CONDS:
        if cond == "obesity":
            x = d.copy()
            anchor = x.bmi >= 30
        elif cond == "aki":
            x = d[~d.renal_excl].copy()
            anchor = s15.stage_aki(x).reindex(x.index).fillna(0) >= 1
        else:
            col = {"hyponatremia": "na_min", "anemia": "hgb_min", "thrombocytopenia": "plt_min"}[cond]
            x = d.copy()
            anchor = s15.stage_lab(x[col], s15.ACUTE[cond]["cuts"]) >= 1
        x[f"{cond}_D"] = x[f"{cond}_doc"]
        x[f"{cond}_H"] = x.patientunitstayid.isin(anem_h) if cond == "anemia" else False
        lob = codes(x, cond, ("D", "H"), np.asarray(anchor, bool))
        keep = (lob >= 0) & x.hospitalid.isin(H).to_numpy()
        comps.append((pos_of[x.patientunitstayid.to_numpy()[keep]].to_numpy(),
                      pd.Index(H).get_indexer(x.hospitalid.to_numpy()[keep]), lob[keep], 4))
        pooled.append(np.bincount(lob[lob >= 0], minlength=4).astype(float))
        rows.append(dict(prediction="P16-desc", test=f"{cond}: lobe shares D+H / D only / H only / U (pooled eICU)",
                         value_U=" / ".join(f"{100 * v:.1f}" for v in pooled[-1] / pooled[-1].sum()), in_holm=False))
    Uni = rng.random((N_SPLITS, len(e_pc)))
    Xp = split_angles(comps, len(H), Uni)
    # published DHM, same hospitals that have >= 100 DHM records for every condition
    f = e_flags.set_index("patientunitstayid").reindex(e_pc.patientunitstayid).reset_index()
    ok = set(H)
    dcodes = {}
    for cond in PUBLISHED:
        c = codes(f, cond, SRCS)
        dcodes[cond] = c
        n = pd.Series(c >= 0).groupby(f.hospitalid.to_numpy()).sum()
        ok &= set(n[n >= MIN_N].index)
    Hc = np.array(sorted(ok))
    sel = pd.Index(H).get_indexer(Hc)
    Xd = {}
    for cond in PUBLISHED:
        Xd[cond] = split_angles([comp(f, dcodes[cond], Hc, 7)], len(Hc), Uni)

    def diff(i):
        return noise_share(Xp, sel[i]) - np.median([noise_share(Xd[c], i) for c in PUBLISHED])

    sp_all = noise_share(Xp, np.arange(len(H)))
    sp = noise_share(Xp, sel)
    sd = {c: noise_share(Xd[c], np.arange(len(Hc))) for c in PUBLISHED}
    bs = boot_diff(diff, len(Hc), B13)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    est = sp - np.median(list(sd.values()))
    rows.append(dict(prediction="P16", test="noise-floor share, profile minus median of 4 published DHM (common hospitals)",
                     estimate=est, lo=lo, hi=hi, p=s15.boot_p(bs, 0), criterion_met=(sp_all >= 0.5) and (est > 0) and lo > 0,
                     in_holm=True, value_U=sp, value_DHM=np.median(list(sd.values())), n_hospitals=len(Hc),
                     value_U_all_hospitals=sp_all, n_hospitals_all=len(H),
                     dhm_by_condition=" / ".join(f"{c} {v:.3f}" for c, v in sd.items())))
    # angle to pooled profile
    V = profile(comps, len(H))
    ref = np.concatenate([p / np.linalg.norm(p) for p in pooled])
    ang = pd.Series(angle_to(V, ref), index=H)
    lpd = e_pc.groupby("hospitalid").apply(lambda g: g.n_lab.sum() / g.icu_days.sum(), include_groups=False)
    r17 = stats.spearmanr(ang, lpd.reindex(H)).statistic
    rows.append(dict(prediction="P17", test="Spearman(profile angle, labs per patient-day)", estimate=r17,
                     criterion_met=abs(r17) < 0.3, in_holm=False, n_hospitals=len(H)))
    rows.append(dict(prediction="P18", test="Spearman(profile angle, |PC1|) (partly circular)",
                     estimate=stats.spearmanr(ang, hab.pc1.abs().reindex(H)).statistic, in_holm=False, n_hospitals=len(H)))
    rows.append(dict(prediction="P18", test="Spearman(profile angle, signed PC1), reported",
                     estimate=stats.spearmanr(ang, hab.pc1.reindex(H)).statistic, in_holm=False, n_hospitals=len(H)))
    return rows, pd.DataFrame(dict(profile_angle=ang, labs_per_day=lpd.reindex(H)))


def main():
    con = duckdb.connect()
    e = eicu_part1(con)
    m = mimic_part1(con)
    print(f"eICU {len(e):,} stays, MIMIC-IV reference {len(m):,} stays")
    # sanity: published DHM diabetes noise share against script 02 (different random halves)
    r1, desc, hosp = part1(e, m)
    e_pc = pd.read_parquet(ROW / "eicu_positive_controls.parquet")
    r2, hosp2 = part2(e_pc, pd.read_parquet(ROW / "eicu_flags.parquet"))
    out = pd.DataFrame(r1 + r2)
    h = out[out.in_holm.astype(bool)]
    order = np.argsort(h.p.to_numpy())
    adj = np.minimum(1, np.maximum.accumulate((len(h) - np.arange(len(h))) * h.p.to_numpy()[order]))
    out.loc[h.index[order], "p_holm"] = adj
    out["passed"] = np.where(out.in_holm.astype(bool), out.criterion_met.astype(bool) & (out.p_holm < 0.05), np.nan)
    out.to_csv(RES / "undocumented_lobe.csv", index=False)
    pd.DataFrame(desc).to_csv(RES / "undocumented_lobe_descriptive.csv", index=False)
    hosp["part1"].to_parquet(ROW / "eicu_undocumented_lobe_part1.parquet")
    hosp2.to_parquet(ROW / "eicu_undocumented_lobe_part2.parquet")
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(out.drop(columns=[c for c in ["dhm_by_condition"] if c in out]).round(4).to_string())
    print(out.dhm_by_condition.dropna().to_string())
    print(pd.DataFrame(desc).round(2).to_string())


if __name__ == "__main__":
    main()
