"""Random-intercept logistic regression by adaptive Gauss-Hermite quadrature, on aggregated binomial cells.

fit(X, y, n, g) -> dict(beta, sd, se_beta, loglik). X is cells x p (with intercept), y successes,
n trials, g hospital index 0..H-1. Hospitals are independent, so a hospital bootstrap is a
resample of g with new indices (see resample()).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

GH_Z, GH_W = np.polynomial.hermite.hermgauss(9)


def _modes(eta, y, n, g, H, sigma, u0, iters=100):
    u = u0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(eta + u[g])))
        grad = np.bincount(g, y - n * p, H) - u / sigma**2
        hess = -np.bincount(g, n * p * (1 - p), H) - 1 / sigma**2
        step = np.clip(grad / hess, -2, 2)
        u -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    p = 1 / (1 + np.exp(-(eta + u[g])))
    hess = -np.bincount(g, n * p * (1 - p), H) - 1 / sigma**2
    return u, hess


def _negll(theta, X, y, n, g, H, state):
    beta, sigma = theta[:-1], np.exp(theta[-1])
    eta = X @ beta
    u, hess = _modes(eta, y, n, g, H, sigma, state["u"])
    state["u"] = u
    s = 1 / np.sqrt(-hess)
    terms = []
    for z, w in zip(GH_Z, GH_W):
        uk = u + np.sqrt(2) * s * z
        e = eta + uk[g]
        f = np.bincount(g, y * e - n * np.logaddexp(0, e), H) - uk**2 / (2 * sigma**2)
        terms.append(np.log(w) + f + z**2)
    terms = np.array(terms)
    m = terms.max(0)
    lh = m + np.log(np.exp(terms - m).sum(0)) + np.log(np.sqrt(2) * s) - 0.5 * np.log(2 * np.pi * sigma**2)
    return -lh.sum()


def _start(X, y, n):
    beta = np.zeros(X.shape[1])
    for _ in range(30):  # IRLS for the fixed-effects logistic start
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = n * p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-8 * np.eye(X.shape[1]), X.T @ (y - n * p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def fit(X, y, n, g, se=True, start=None):
    X, y, n = np.asarray(X, float), np.asarray(y, float), np.asarray(n, float)
    g = np.asarray(g, int)
    H = g.max() + 1
    theta0 = np.r_[_start(X, y, n), np.log(0.5)] if start is None else start
    state = {"u": np.zeros(H)}
    f = lambda t: _negll(t, X, y, n, g, H, state)  # noqa: E731
    res = minimize(f, theta0, jac=lambda t: _grad(f, t), method="L-BFGS-B",
                   bounds=[(None, None)] * X.shape[1] + [(np.log(1e-3), np.log(10))])
    out = dict(beta=res.x[:-1], sd=float(np.exp(res.x[-1])), loglik=-res.fun, theta=res.x, converged=res.success)
    out["u"] = _modes(X @ res.x[:-1], y, n, g, H, out["sd"], state["u"])[0]  # hospital modes, index = g
    if se:
        out["se_beta"] = np.sqrt(np.diag(np.linalg.inv(_hessian(f, res.x))))[:-1]
    return out


def _grad(f, x, h=1e-5):
    """Central differences; the inner modes are solved to 1e-11, so the objective is smooth at this step."""
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


def _hessian(f, x, h=1e-3):
    k = len(x)
    Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            Hm[i, j] = Hm[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return Hm


def resample(g, rng):
    """Hospital bootstrap: returns (row index, new hospital index)."""
    H = g.max() + 1
    pick = rng.integers(0, H, H)
    by = [np.flatnonzero(g == h) for h in range(H)]
    rows = [by[h] for h in pick]
    return np.concatenate(rows), np.repeat(np.arange(H), [len(r) for r in rows])
