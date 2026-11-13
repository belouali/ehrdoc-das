#!/usr/bin/env python
"""S_das_variation (configs/screen_settings.yaml, locked 2026-11-09 before computing): Part A of the paper.

Documentation variation across the 141 hospitals of S_site_selection_iecv, all together, no split.
  DAS     pairwise angles (degrees) between DHM documentation vectors per condition; mean DAS of a pair is the mean
          over the conditions both hospitals have; hospital positions = principal coordinates; MIMIC-IV added as an
          out-of-sample point on the mean-DAS map (Gower's formula)
  A1      spread of between-hospital DAS against a split-half noise floor (200 random halvings per hospital);
          Mantel tests (Spearman) between condition matrices
  A2      PERMANOVA of region, bed size, teaching status and case mix (hospital mean APACHE IV score, share admitted
          from the ED, share transferred from another hospital), each alone and all together; PERMDISP per
          categorical variable
  A3      per outcome, logistic mixed model (M1 patient covariates + hospital random intercept, adaptive
          Gauss-Hermite quadrature); + DAS position (3 axes), + characteristics and case mix, + both; share of
          between-hospital variance explained, likelihood-ratio tests; Mantel of mean DAS against intercept differences
  Main measure mean DAS; condition measures secondary (A1, A2). Sensitivity: without no-medication hospitals
  (A1-A3); stay outcomes among survivors (A3).

Outputs (aggregate, hospital level at most): results/partA_*.csv

Usage (from mimic_external/):  bash scripts/partA.sh [--quick] [--selftest]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from scipy import optimize, special, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s38 = _load("s38", "38_iecv.py")
CONDITIONS, TASKS = s38.CONDITIONS, s38.TASKS
STAY_TASKS = [t for t in TASKS if t != "mortality"]
CATS = ["region", "bed_size", "teaching"]
MIX = ["mean_apache", "share_ed", "share_transfer_in"]
N_AXES = 3


# ---------------------------------------------------------------- distance tools
def pairwise(V, hosp):
    """Per-condition angle matrices on hosp (NaN where a hospital lacks the condition) and the mean-DAS matrix."""
    n = len(hosp)
    mats = {}
    for c in CONDITIONS:
        m = np.full((n, n), np.nan)
        have = np.array([h in V[c].index for h in hosp])
        idx = np.flatnonzero(have)
        m[np.ix_(idx, idx)] = s02.pairwise_angles(V[c].loc[hosp[have]].to_numpy(float))
        np.fill_diagonal(m, 0.0)
        mats[c] = m
    stack = np.stack([mats[c] for c in CONDITIONS])
    cnt = np.isfinite(stack).sum(0)
    assert (cnt > 0).all(), "a hospital pair shares no condition"
    mats["mean"] = np.nanmean(stack, axis=0)
    return mats


def sub(m, keep):
    return m[np.ix_(keep, keep)]


def upper(m):
    return m[np.triu_indices(len(m), 1)]


def pcoa(D):
    n = len(D)
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = vals > 1e-9 * vals[0]
    return dict(B=B, vals=vals, vecs=vecs, X=vecs[:, pos] * np.sqrt(vals[pos]),
                share=vals[pos] / vals[pos].sum())


def add_point(p, d):
    """Coordinates of a new point from its distances d to the n fitted points (Gower 1968)."""
    b = 0.5 * (np.diag(p["B"]) - d ** 2)
    k = p["X"].shape[1]
    return (p["vecs"][:, :k].T @ b) / np.sqrt(p["vals"][:k])


def mantel(A, B, perms, rng):
    """Spearman Mantel statistic and one-sided permutation p (rows and columns of B permuted together)."""
    n = len(A)
    iu = np.triu_indices(n, 1)
    ra = stats.rankdata(A[iu])
    RB = np.zeros((n, n))
    RB[iu] = stats.rankdata(B[iu])
    RB = RB + RB.T
    ra = (ra - ra.mean()) / ra.std()

    def r(M):
        x = M[iu]
        return float(((x - x.mean()) / x.std() * ra).mean())

    obs = r(RB)
    hits = sum(r(RB[np.ix_(pp, pp)]) >= obs for pp in (rng.permutation(n) for _ in range(perms)))
    return obs, (hits + 1) / (perms + 1)


def design_matrix(df):
    """Numeric design (no intercept) for categorical and continuous hospital variables."""
    parts = []
    for c in df.columns:
        if c in CATS:
            d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
            parts.append(d)
        else:
            x = df[c].astype(float)
            parts.append(((x - x.mean()) / x.std()).to_frame(c))
    return pd.concat(parts, axis=1).to_numpy(float)


def permanova(D, X, perms, rng):
    n = len(D)
    J = np.eye(n) - 1 / n
    G = -0.5 * J @ (D ** 2) @ J
    X1 = np.column_stack([np.ones(n), X])
    p = np.linalg.matrix_rank(X1) - 1
    H = X1 @ np.linalg.pinv(X1)
    sst = np.trace(G)

    def f(Gm):
        ssm = (H * Gm).sum()
        return (ssm / p) / ((sst - ssm) / (n - p - 1)), ssm / sst

    F, R2 = f(G)
    hits = 0
    for _ in range(perms):
        pp = rng.permutation(n)
        hits += f(G[np.ix_(pp, pp)])[0] >= F
    return dict(n=n, df=p, R2=R2, F=F, p=(hits + 1) / (perms + 1))


def permdisp(D, groups, perms, rng):
    X = pcoa(D)["X"]
    g = pd.Series(groups).astype(str).to_numpy()
    levels = np.unique(g)

    def F(lab):
        z = np.empty(len(lab))
        for l in levels:
            m = lab == l
            z[m] = np.linalg.norm(X[m] - X[m].mean(0), axis=1)
        return stats.f_oneway(*[z[lab == l] for l in levels]).statistic, z

    f0, z = F(g)
    hits = sum(F(rng.permutation(g))[0] >= f0 for _ in range(perms))
    return dict(n=len(g), groups=len(levels), F=f0, p=(hits + 1) / (perms + 1),
                **{f"disp_{l}": float(z[g == l].mean()) for l in levels})


# ---------------------------------------------------------------- random-intercept logistic model (AGHQ)
class RandomInterceptLogit:
    """y_ij ~ Bernoulli(expit(x_ij b + u_j)), u_j ~ N(0, s2); marginal likelihood by adaptive Gauss-Hermite."""

    def __init__(self, X, y, g, nodes=9):
        self.X, self.y = X, y.astype(float)
        self.g, self.groups = pd.factorize(g)
        self.G = self.g.max() + 1
        self.z, self.w = np.polynomial.hermite.hermgauss(nodes)

    def _modes(self, eta, s2):
        g, G, y = self.g, self.G, self.y
        u = np.zeros(G)
        for _ in range(50):
            p = special.expit(eta + u[g])
            g1 = np.bincount(g, y - p, G) - u / s2
            g2 = -np.bincount(g, p * (1 - p), G) - 1 / s2
            step = g1 / g2
            u -= step
            if np.abs(step).max() < 1e-10:
                break
        return u, np.sqrt(-1 / g2)

    def _eval(self, theta):
        b, ls = theta[:-1], theta[-1]
        s2 = np.exp(2 * ls)
        eta = self.X @ b
        u, s = self._modes(eta, s2)
        U = u[:, None] + np.sqrt(2) * s[:, None] * self.z[None, :]
        E = eta[:, None] + U[self.g]
        ll = self.y[:, None] * E - np.logaddexp(0, E)
        L = np.stack([np.bincount(self.g, ll[:, k], self.G) for k in range(len(self.z))], axis=1)
        logterm = L - U ** 2 / (2 * s2) + self.z ** 2 + np.log(self.w)
        lse = special.logsumexp(logterm, axis=1)
        loglik = (lse + np.log(np.sqrt(2) * s) - 0.5 * np.log(2 * np.pi * s2)).sum()
        W = np.exp(logterm - lse[:, None])
        R = ((self.y[:, None] - special.expit(E)) * W[self.g]).sum(1)
        grad = np.r_[self.X.T @ R, (W * (U ** 2 / s2 - 1)).sum()]
        return loglik, grad, (W * U).sum(1)

    def fit(self, b0=None):
        p = self.X.shape[1]
        th = np.r_[np.zeros(p) if b0 is None else b0, np.log(0.4)]
        res = optimize.minimize(lambda t: tuple(-v for v in self._eval(t)[:2]), th, jac=True, method="L-BFGS-B",
                                options=dict(maxiter=5000, gtol=1e-5, ftol=1e-15, maxcor=30))
        ll, grad, blup = self._eval(res.x)
        self.loglik, self.beta, self.s2, self.blup = ll, res.x[:-1], np.exp(2 * res.x[-1]), blup
        self.max_grad = float(np.abs(grad).max())
        self.converged = self.max_grad < 0.05
        return self


def selftest():
    rng = np.random.default_rng(1)
    G, n_per, s = 141, 300, 0.4
    g = np.repeat(np.arange(G), rng.integers(100, 2 * n_per, G))
    n = len(g)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([-2.0, 0.8, -0.5])
    u = rng.normal(0, s, G)
    y = rng.random(n) < special.expit(X @ beta + u[g])
    m = RandomInterceptLogit(X, y, g).fit()
    print(f"selftest: beta {np.round(m.beta, 3)} (true {beta}), sd {np.sqrt(m.s2):.3f} (true {s}), "
          f"BLUP corr {np.corrcoef(m.blup, u)[0, 1]:.3f}, converged {m.converged}")
    m1 = RandomInterceptLogit(X, y, g, nodes=1).fit()
    m15 = RandomInterceptLogit(X, y, g, nodes=15).fit()
    print(f"loglik nodes 1/9/15: {m1.loglik:.3f} / {m.loglik:.3f} / {m15.loglik:.3f}")
    assert abs(np.sqrt(m.s2) - s) < 0.08 and np.abs(m.beta - beta).max() < 0.08 and m.converged
    assert abs(m.loglik - m15.loglik) < 0.05


# ---------------------------------------------------------------- data
def patient_design(e):
    cols = [c for c in s38.design_columns() if c in e.columns]
    num = ["apachescore", "agenum"]
    X = e[cols].astype(float).copy()
    for c in num:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    drop = []
    for prefix in sorted({c.split("=")[0] for c in cols if "=" in c}):
        grp = [c for c in cols if c.startswith(prefix + "=")]
        kept = [c for c in grp if X[c].sum() > 0]
        drop += [c for c in grp if c not in kept] + [max(kept, key=lambda c: X[c].sum())]  # largest = reference
    X = X.drop(columns=drop)
    return np.column_stack([np.ones(len(X)), X.to_numpy(float)])


def hospital_table(cfg, e, hosp):
    E = cfg["paths"]["eicu"]
    h = duckdb.execute(f"SELECT hospitalid, region, numbedscategory, teachingstatus FROM '{E}/hospital.parquet'").df()
    h = h.set_index("hospitalid").reindex(hosp)
    out = pd.DataFrame(index=hosp)
    out["region"] = h.region.fillna("unknown").astype(str)
    out["bed_size"] = h.numbedscategory.fillna("unknown").astype(str)
    out["teaching"] = h.teachingstatus.map({True: "teaching", False: "non-teaching"}).fillna("unknown")
    ee = e[e.hospitalid.isin(hosp)]
    grp = ee.groupby("hospitalid")
    out["mean_apache"] = grp.apachescore.mean()
    out["share_ed"] = grp["unitadmitsource=Emergency Department"].mean()
    out["share_transfer_in"] = grp["unitadmitsource=Other Hospital"].mean()
    return out


def noise_floor(flags, V, hosp, splits, rng):
    """Split-half DAS within each hospital: per condition and for the hospital's mean over its conditions."""
    rows, mean_draws = [], {h: np.zeros(splits) for h in hosp}
    n_cond = {h: 0 for h in hosp}
    fl = flags[flags.hospitalid.isin(hosp)]
    for c in CONDITIONS:
        codes = s02.lobe_codes(fl, c, ("D", "H", "M"))
        keep = codes >= 0
        by_h = pd.Series(codes[keep]).groupby(fl.hospitalid.to_numpy()[keep]).apply(np.asarray)
        for h in hosp:
            if h not in V[c].index:
                continue
            x = by_h[h]
            half = len(x) // 2
            d = np.empty(splits)
            for s in range(splits):
                p = rng.permutation(x)
                a, b = np.bincount(p[:half], minlength=7), np.bincount(p[half:2 * half], minlength=7)
                d[s] = s02.angles(a[None, :].astype(float), b.astype(float))[0]
            rows.append(dict(hospitalid=h, condition=c, records=len(x), split_half_mean=d.mean(),
                             split_half_q95=np.percentile(d, 95)))
            mean_draws[h] += d
            n_cond[h] += 1
    for h in hosp:
        d = mean_draws[h] / n_cond[h]
        rows.append(dict(hospitalid=h, condition="mean", records=np.nan, split_half_mean=d.mean(),
                         split_half_q95=np.percentile(d, 95)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analyses
def part_a1_a2(sample, mats, hosp, ht, nf, perms, rng):
    spread, mant, perm, disp = [], [], [], []
    for meas in CONDITIONS + ["mean"]:
        m = mats[meas]
        have = np.isfinite(m).sum(1) > 1  # the hospital has this measure
        D = sub(m, have)
        hs = hosp[have]
        q = nf[nf.condition == meas].set_index("hospitalid").reindex(hs)
        thr = np.maximum.outer(q.split_half_q95.to_numpy(), q.split_half_q95.to_numpy())
        between = upper(D)
        spread.append(dict(sample=sample, measure=meas, hospitals=len(hs), pairs=len(between),
                           das_median=np.median(between), das_q25=np.percentile(between, 25),
                           das_q75=np.percentile(between, 75), das_mean=between.mean(),
                           noise_mean=q.split_half_mean.mean(), excess_over_noise=between.mean() - q.split_half_mean.mean(),
                           share_pairs_beyond_noise=(between > upper(thr)).mean()))
        role = "main" if meas == "mean" else "secondary"
        for var in CATS + MIX + ["all"]:
            cols = CATS + MIX if var == "all" else [var]
            d = ht.loc[hs, cols]
            ok = ~(d[[c for c in cols if c in CATS]] == "unknown").any(axis=1).to_numpy()
            ok &= d.notna().all(axis=1).to_numpy()
            r = permanova(sub(D, ok), design_matrix(d[ok]), perms, rng)
            perm.append(dict(sample=sample, measure=meas, role=role, variable=var, **r))
            if var in CATS:
                disp.append(dict(sample=sample, measure=meas, role=role, variable=var,
                                 **permdisp(sub(D, ok), d[ok][var].to_numpy(), perms, rng)))
    for i, a in enumerate(CONDITIONS):
        for b in CONDITIONS[i + 1:]:
            both = (np.isfinite(mats[a]).sum(1) > 1) & (np.isfinite(mats[b]).sum(1) > 1)
            r, p = mantel(sub(mats[a], both), sub(mats[b], both), perms, rng)
            mant.append(dict(sample=sample, pair=f"{a} vs {b}", hospitals=int(both.sum()), spearman=r, p=p))
    return spread, mant, perm, disp


def part_a3(sample, e, Y, keep_stays, hosp, positions, ht, D_mean, perms, rng, nodes):
    out, mant = [], []
    hix = {h: i for i, h in enumerate(hosp)}
    tasks = [(t, "all stays") for t in TASKS] + ([(t, "survivors") for t in STAY_TASKS] if sample == "all" else [])
    hosp_X = {"das": positions,
              "characteristics": design_matrix(ht[CATS + MIX])}
    hosp_X["both"] = np.column_stack([hosp_X["das"], hosp_X["characteristics"]])
    for task, pop in tasks:
        m = keep_stays & (e.death.to_numpy() == 0 if pop == "survivors" else True)
        ee = e[m]
        y = Y[task][m]
        hi = ee.hospitalid.map(hix).to_numpy()
        Xp = patient_design(ee)
        base = RandomInterceptLogit(Xp, y, hi, nodes).fit()
        assert len(base.groups) == len(hosp)
        row = dict(sample=sample, task=task, population=pop, stays=int(m.sum()), events=int(y.sum()),
                   hospitals=len(np.unique(hi)), model="base", added=0, sd=np.sqrt(base.s2), var=base.s2,
                   share_explained=0.0, loglik=base.loglik, lrt_chi2=np.nan, lrt_df=np.nan, lrt_p=np.nan,
                   converged=base.converged, max_grad=base.max_grad)
        out.append(row)
        for name, H in hosp_X.items():
            X = np.column_stack([Xp, H[hi]])
            mm = RandomInterceptLogit(X, y, hi, nodes).fit(b0=np.r_[base.beta, np.zeros(H.shape[1])])
            chi = 2 * (mm.loglik - base.loglik)
            out.append(dict(row, model=f"+ {name}", added=H.shape[1], sd=np.sqrt(mm.s2), var=mm.s2,
                            share_explained=1 - mm.s2 / base.s2, loglik=mm.loglik, lrt_chi2=chi, lrt_df=H.shape[1],
                            lrt_p=stats.chi2.sf(chi, H.shape[1]), converged=mm.converged,
                            max_grad=mm.max_grad))
        # base-model intercepts in hospital order
        u = np.empty(len(hosp))
        u[base.groups] = base.blup
        r, p = mantel(D_mean, np.abs(u[:, None] - u[None, :]), perms, rng)
        mant.append(dict(sample=sample, task=task, population=pop, hospitals=len(hosp), spearman=r, p=p))
        print(pd.DataFrame(out[-4:])[["task", "population", "model", "sd", "share_explained", "lrt_p", "converged"]]
              .round(4).to_string(index=False), flush=True)
    return out, mant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--quick", action="store_true", help="few permutations and splits; writes results/_test")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    cfg, _, res, rowlevel = s38.setup(args.config)
    import yaml
    st = yaml.safe_load(open(Path(args.config).resolve().parent / "screen_settings.yaml"))["S_das_variation"]
    assert st["status"] == "locked", "S_das_variation must be locked"
    perms, splits, nodes = (199, 20, 5) if args.quick else (9999, 200, 9)
    outdir = res / "_test" if args.quick else res
    outdir.mkdir(exist_ok=True)
    seed = int(st["seed"])
    rng = np.random.default_rng([seed, 40])

    split = pd.read_csv(res / "iecv_split.csv")
    e, X, XM, Y, rows_of, V, no_med = s38.load_eicu(cfg, rowlevel)
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    hosp_141 = np.sort(split.hospitalid.to_numpy())
    assert set(no_med & set(hosp_141)) == set(split[split.no_medication].hospitalid)
    # a hospital needs >= 100 records of at least one condition to have a DAS (as in Part B)
    has_das = np.array([any(h in V[c].index for c in CONDITIONS) for h in hosp_141])
    hosp_all = hosp_141[has_das]
    pd.DataFrame([dict(eligible=len(hosp_141), with_das=len(hosp_all), without_das=int((~has_das).sum()),
                       without_das_stays=int(sum(len(rows_of[h]) for h in hosp_141[~has_das])),
                       without_das_no_medication=int(sum(h in no_med for h in hosp_141[~has_das])),
                       **{f"with_{c}": int(sum(h in V[c].index for h in hosp_all)) for c in CONDITIONS})]) \
        .to_csv(outdir / "partA_hospitals.csv", index=False)
    ht_all = hospital_table(cfg, e, hosp_all)
    nf = noise_floor(flags, V, hosp_all, splits, rng)
    nf.to_csv(outdir / "partA_noise_floor_hospitals.csv", index=False, float_format="%.4f")
    mats_all = pairwise(V, hosp_all)
    mimic = pd.read_csv(res / "iecv_mimic_relatedness.csv").set_index("hospitalid")

    spread, mant, perm, disp, a3, a3m, pos_rows, axes = [], [], [], [], [], [], [], []
    for sample in ["all", "with_medication"]:
        keep = np.array([h not in no_med for h in hosp_all]) if sample == "with_medication" else np.ones(len(hosp_all), bool)
        hosp = hosp_all[keep]
        mats = {k: sub(v, keep) for k, v in mats_all.items()}
        ht = ht_all.loc[hosp]
        p = pcoa(mats["mean"])
        positions = p["X"][:, :N_AXES]
        axes.append(dict(sample=sample, hospitals=len(hosp), **{f"axis{i + 1}_share": p["share"][i] for i in range(5)}))
        for i, h in enumerate(hosp):
            pos_rows.append(dict(sample=sample, unit=str(h), no_medication=h in no_med,
                                 **{f"axis{k + 1}": positions[i, k] for k in range(N_AXES)}))
        mpt = add_point(p, mimic.loc[hosp, "das_mean"].to_numpy(float))
        pos_rows.append(dict(sample=sample, unit="MIMIC-IV", no_medication=False,
                             **{f"axis{k + 1}": mpt[k] for k in range(N_AXES)}))
        s1, m1, p1, d1 = part_a1_a2(sample, mats, hosp, ht, nf, perms, rng)
        spread += s1; mant += m1; perm += p1; disp += d1
        print(pd.DataFrame(s1).round(3).to_string(index=False), flush=True)
        keep_stays = e.hospitalid.isin(hosp).to_numpy()
        o, om = part_a3(sample, e, Y, keep_stays, hosp, positions, ht, mats["mean"], perms, rng, nodes)
        a3 += o; a3m += om
        for name, rows in [("spread", spread), ("mantel_conditions", mant), ("permanova", perm),
                           ("permdisp", disp), ("outcome_models", a3), ("intercept_mantel", a3m),
                           ("positions", pos_rows), ("pcoa_axes", axes)]:
            pd.DataFrame(rows).to_csv(outdir / f"partA_{name}.csv", index=False, float_format="%.6f")
    print("done", flush=True)


if __name__ == "__main__":
    main()
