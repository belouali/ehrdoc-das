from __future__ import annotations
import numpy as np
import pandas as pd


def _ss_within(D2: np.ndarray, idx: np.ndarray, n_groups: int) -> float:
    s = 0.0
    for k in range(n_groups):
        m = idx == k
        nk = int(m.sum())
        if nk < 2:
            continue
        sub = D2[np.ix_(m, m)]
        s += sub[np.triu_indices(nk, k=1)].sum() / nk
    return s


def permanova(
    D: np.ndarray,
    groups,
    n_perm: int = 999,
    seed: int = 0,
) -> dict:
    """One-way PERMANOVA (Anderson 2001) on a distance matrix.

    Tests whether hospitals in different groups (e.g. regions) are farther
    apart than hospitals in the same group, using the full pairwise distance
    matrix rather than a per-hospital summary. Returns the pseudo-F, the
    permutation p-value, and R^2 = SS_between / SS_total (share of distance
    variation explained by the grouping).

    Hospitals with a missing group label are dropped. Groups with a single
    member contribute to SS_total but not SS_within.
    """
    groups = pd.Series(list(groups))
    mask = groups.notna().to_numpy()
    D = np.asarray(D, dtype=float)[np.ix_(mask, mask)]
    g = groups[mask].astype(str).to_numpy()
    n = len(g)
    labels, idx = np.unique(g, return_inverse=True)
    a = len(labels)
    out = {"n_hospitals": n, "n_groups": a, "pseudo_F": np.nan, "p_value": np.nan, "R2": np.nan}
    if a < 2 or n <= a:
        return out

    D2 = D ** 2
    ss_total = D2[np.triu_indices(n, k=1)].sum() / n
    ss_within = _ss_within(D2, idx, a)
    ss_between = ss_total - ss_within
    if ss_within <= 0:
        return out
    F = (ss_between / (a - 1)) / (ss_within / (n - a))
    R2 = ss_between / ss_total

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        pidx = rng.permutation(idx)
        ssw = _ss_within(D2, pidx, a)
        if ssw <= 0:
            continue
        Fp = ((ss_total - ssw) / (a - 1)) / (ssw / (n - a))
        if Fp >= F:
            count += 1
    out.update({"pseudo_F": float(F), "p_value": (count + 1) / (n_perm + 1), "R2": float(R2)})
    return out


# ── Multivariable PERMANOVA (marginal tests) and PERMDISP ──────────────────────

def _gower_center(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    A = -0.5 * (D ** 2)
    J = np.eye(n) - np.ones((n, n)) / n
    return J @ A @ J


def _design(df: pd.DataFrame, terms: list[str]) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Intercept + treatment-coded dummies for each categorical term.

    Returns the design matrix and the column indices belonging to each term.
    """
    cols = [np.ones((len(df), 1))]
    idx: dict[str, list[int]] = {}
    c = 1
    for t in terms:
        d = pd.get_dummies(df[t].astype(str), drop_first=True).to_numpy(dtype=float)
        idx[t] = list(range(c, c + d.shape[1]))
        cols.append(d)
        c += d.shape[1]
    return np.hstack(cols), idx


def _hat(X: np.ndarray) -> tuple[np.ndarray, int]:
    pinv = np.linalg.pinv(X.T @ X)
    H = X @ pinv @ X.T
    return H, int(np.linalg.matrix_rank(X))


def permanova_multivariable(D: np.ndarray, covariates: pd.DataFrame, terms: list[str],
                            n_perm: int = 999, seed: int = 0) -> pd.DataFrame:
    """Multivariable PERMANOVA (McArdle & Anderson 2001) with marginal term tests.

    Each term is tested conditional on all other terms (Type II / "by margin"):
    SS_term = SS(full model) - SS(model without the term). The null distribution
    is obtained by permuting hospital labels of the Gower-centered matrix, and
    the same permutation is applied to every term. Rows with a missing
    covariate are dropped. R2 is the marginal share of total squared-distance
    variation; R2_joint is the share explained by the full model.
    """
    df = covariates.reset_index(drop=True).copy()
    mask = df[terms].notna().all(axis=1).to_numpy()
    df = df[mask].reset_index(drop=True)
    G = _gower_center(np.asarray(D, dtype=float)[np.ix_(mask, mask)])
    n = G.shape[0]
    X, idx = _design(df, terms)
    H_full, r_full = _hat(X)
    ss_total = np.trace(G)
    ss_model = np.sum(H_full * G)
    ss_res = ss_total - ss_model
    df_res = n - r_full
    reduced = {}
    for t in terms:
        keep = [j for j in range(X.shape[1]) if j not in idx[t]]
        H_r, r_r = _hat(X[:, keep])
        reduced[t] = (H_r, r_full - r_r)

    def stats(Gp):
        ssm = np.sum(H_full * Gp); ssr = np.trace(Gp) - ssm
        out = {"__joint__": (ssm, (ssm / (r_full - 1)) / (ssr / df_res) if ssr > 0 else np.nan)}
        for t in terms:
            H_r, df_t = reduced[t]
            ss_t = ssm - np.sum(H_r * Gp)
            out[t] = (ss_t, (ss_t / df_t) / (ssr / df_res) if df_t > 0 and ssr > 0 else np.nan)
        return out

    obs = stats(G)
    rng = np.random.default_rng(seed)
    counts = {t: 0 for t in terms + ["__joint__"]}
    for _ in range(n_perm):
        p = rng.permutation(n)
        Gp = G[np.ix_(p, p)]
        per = stats(Gp)
        for t in terms + ["__joint__"]:
            if np.isfinite(per[t][1]) and per[t][1] >= obs[t][1]:
                counts[t] += 1
    rows = []
    for t in terms:
        ss_t, F = obs[t]
        rows.append({"term": t, "df": reduced[t][1], "SS": ss_t, "pseudo_F": F, "R2_marginal": ss_t / ss_total,
                     "p_value": (counts[t] + 1) / (n_perm + 1)})
    rows.append({"term": "Model (joint)", "df": r_full - 1, "SS": ss_model, "pseudo_F": (ss_model / (r_full - 1)) / (ss_res / df_res),
                 "R2_marginal": ss_model / ss_total, "p_value": (counts["__joint__"] + 1) / (n_perm + 1)})
    rows.append({"term": "Residual", "df": df_res, "SS": ss_res, "pseudo_F": np.nan, "R2_marginal": ss_res / ss_total, "p_value": np.nan})
    out = pd.DataFrame(rows)
    out["n_hospitals"] = n
    out["n_perm"] = n_perm
    return out


def permdisp(D: np.ndarray, groups, n_perm: int = 999, seed: int = 0) -> dict:
    """PERMDISP (Anderson 2006): homogeneity of multivariate dispersion.

    Hospitals are embedded by principal coordinates (keeping the imaginary
    axes from negative eigenvalues), each hospital's distance to its group
    centroid is computed as in Anderson (2006), and a one-way ANOVA F on those
    distances is compared with a permutation distribution obtained by shuffling
    the distances across groups (as in vegan's permutest.betadisper).
    """
    groups = pd.Series(list(groups))
    mask = groups.notna().to_numpy()
    D = np.asarray(D, dtype=float)[np.ix_(mask, mask)]
    g = groups[mask].astype(str).to_numpy()
    n = len(g)
    labels, idx = np.unique(g, return_inverse=True)
    a = len(labels)
    out = {"n_hospitals": n, "n_groups": a, "F": np.nan, "p_value": np.nan, "group_mean_dispersion": {}}
    if a < 2 or n <= a:
        return out
    G = _gower_center(D)
    w, U = np.linalg.eigh(G)
    pos = w > 1e-10; neg = w < -1e-10
    Xp = U[:, pos] * np.sqrt(w[pos]); Xn = U[:, neg] * np.sqrt(-w[neg])
    z = np.zeros(n)
    for k in range(a):
        m = idx == k
        cp = Xp[m].mean(axis=0); cn = Xn[m].mean(axis=0) if Xn.shape[1] else np.zeros(0)
        d2 = ((Xp[m] - cp) ** 2).sum(axis=1) - (((Xn[m] - cn) ** 2).sum(axis=1) if Xn.shape[1] else 0.0)
        z[m] = np.sqrt(np.clip(d2, 0, None))

    def fstat(zz):
        gm = zz.mean(); ssb = 0.0; ssw = 0.0
        for k in range(a):
            m = idx == k; nk = m.sum()
            if nk == 0:
                continue
            mk = zz[m].mean(); ssb += nk * (mk - gm) ** 2; ssw += ((zz[m] - mk) ** 2).sum()
        return (ssb / (a - 1)) / (ssw / (n - a)) if ssw > 0 else np.nan

    F = fstat(z)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        Fp = fstat(z[rng.permutation(n)])
        if np.isfinite(Fp) and Fp >= F:
            count += 1
    out.update({"F": float(F), "p_value": (count + 1) / (n_perm + 1),
                "group_mean_dispersion": {labels[k]: float(z[idx == k].mean()) for k in range(a)},
                "group_n": {labels[k]: int((idx == k).sum()) for k in range(a)}})
    return out
