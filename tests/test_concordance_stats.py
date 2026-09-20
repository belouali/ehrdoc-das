import numpy as np

from ehrdoc.distances.concordance import mantel_test, distance_matrix_concordance
from ehrdoc.distances.permanova import permanova


def _random_dist(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return D


def test_mantel_identical_matrices_is_significant():
    D = _random_dist(30, 1)
    rho, p = mantel_test(D, D, n_perm=199)
    assert rho > 0.999
    assert p < 0.05


def test_mantel_unrelated_matrices_not_significant():
    Da = _random_dist(30, 1)
    Db = _random_dist(30, 2)
    _, p = mantel_test(Da, Db, n_perm=199)
    assert p > 0.05


def test_concordance_uses_common_ids_and_returns_mantel_p():
    D = _random_dist(20, 3)
    ids_a = list(range(20))
    ids_b = list(range(5, 25))
    out = distance_matrix_concordance(D, ids_a, D, ids_b, n_perm=99)
    assert out["n_hospitals"] == 15
    assert "mantel_p" in out


def test_permanova_detects_group_structure():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(5, 1, (20, 2))])
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    groups = ["a"] * 20 + ["b"] * 20
    out = permanova(D, groups, n_perm=199)
    assert out["p_value"] < 0.05
    assert out["R2"] > 0.5


def test_permanova_null():
    D = _random_dist(40, 5)
    rng = np.random.default_rng(1)
    groups = rng.choice(["a", "b", "c"], size=40)
    out = permanova(D, groups, n_perm=199)
    assert 0 <= out["R2"] <= 1
    assert out["p_value"] > 0.01
