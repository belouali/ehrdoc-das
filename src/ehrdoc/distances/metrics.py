from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean, cityblock
from scipy.spatial import distance
from scipy.linalg import inv
from scipy.stats import entropy

def angular_separation(p: np.ndarray, q: np.ndarray) -> float:
    """Cosine-based angular separation in degrees."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    cos_sim = 1 - cosine(p, q)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))

def euclidean_distance(p, q) -> float:
    return float(euclidean(p, q))

def manhattan_distance(p, q) -> float:
    return float(cityblock(p, q))

def spearman_correlation(p, q) -> float:
    # return distance-like value (1 - rho)
    rho, _ = spearmanr(p, q)
    return float(1 - rho)

def pearson_correlation(p, q) -> float:
    r, _ = pearsonr(p, q)
    return float(1 - r)

def kl_divergence(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))

def jensen_shannon_divergence(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5*(p+q)
    return float(0.5*entropy(p, m) + 0.5*entropy(q, m))

def mahalanobis_distance(p, q, VI=None) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if VI is None:
        # simple diagonal covariance as fallback
        cov = np.diag(np.maximum(p, 1e-6))
        VI = inv(cov)
    return float(distance.mahalanobis(p, q, VI))
