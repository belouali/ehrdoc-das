from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata

from .matrices import flatten_upper_triangle


def _upper(D: np.ndarray) -> np.ndarray:
    return D[np.triu_indices_from(D, k=1)]


def mantel_test(
    D_a: np.ndarray,
    D_b: np.ndarray,
    n_perm: int = 1999,
    seed: int = 0,
    method: str = "spearman",
) -> tuple[float, float]:
    """Mantel permutation test between two aligned distance matrices.

    Hospital pairs are not independent observations (each hospital appears in
    n-1 pairs), so the naive p-value from a correlation over flattened pairs is
    anticonservative. The Mantel test permutes hospital labels of one matrix
    (rows and columns together) and recomputes the correlation.

    Returns (observed_correlation, permutation_p). The p-value is one-sided
    (H1: positive concordance) and uses the (k+1)/(n_perm+1) convention.
    """
    if D_a.shape != D_b.shape or D_a.shape[0] != D_a.shape[1]:
        raise ValueError("Distance matrices must be square and aligned")
    n = D_a.shape[0]
    iu = np.triu_indices(n, k=1)
    va = D_a[iu]
    if method == "spearman":
        va_r = rankdata(va)
        def corr(vb):
            return np.corrcoef(va_r, rankdata(vb))[0, 1]
    else:
        def corr(vb):
            return np.corrcoef(va, vb)[0, 1]

    obs = corr(D_b[iu])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        vb = D_b[np.ix_(p, p)][iu]
        if corr(vb) >= obs:
            count += 1
    return float(obs), (count + 1) / (n_perm + 1)


def distance_matrix_concordance(
    D_a: np.ndarray, ids_a: list,
    D_b: np.ndarray, ids_b: list,
    method: str = "spearman",
    n_perm: int = 1999,
    seed: int = 0,
) -> dict:
    """Concordance between two distance matrices on their common hospitals.

    Returns the Spearman rho over the flattened upper triangles, the naive
    pair-level p-value (kept for reference only) and the Mantel permutation
    p-value, which is the one to report.
    """
    common = sorted(set(ids_a).intersection(ids_b))
    if len(common) < 3:
        return {"n_hospitals": len(common), "n_pairs": 0, "rho": np.nan, "p": np.nan, "mantel_p": np.nan}

    idx_a = [ids_a.index(i) for i in common]
    idx_b = [ids_b.index(i) for i in common]
    Da = D_a[np.ix_(idx_a, idx_a)]
    Db = D_b[np.ix_(idx_b, idx_b)]

    va = flatten_upper_triangle(Da)
    vb = flatten_upper_triangle(Db)
    if method == "spearman":
        rho, p = spearmanr(va, vb)
    else:
        rho, p = pearsonr(va, vb)

    mantel_p = np.nan
    if n_perm and n_perm > 0:
        _, mantel_p = mantel_test(Da, Db, n_perm=n_perm, seed=seed, method=method)

    return {"n_hospitals": len(common), "n_pairs": len(va), "rho": float(rho),
            "p": float(p), "mantel_p": float(mantel_p)}
