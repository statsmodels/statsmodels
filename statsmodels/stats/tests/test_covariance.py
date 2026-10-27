import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import linalg

from statsmodels.stats.covariance import (
    corr_normal_scores,
    corr_quadrant,
    corr_rank,
    transform_corr_normal,
)


def test_transform_corr_normal():

    # numbers from footnote of Table 1 Boudt, Cornelissen, Croux 2012
    vp, vgr, vs, vk, vmcd = np.array(
        [[0.92, 0.92, 1.02, 1.01, 1.45], [0.13, 0.13, 0.16, 0.15, 0.20]]
    ).T

    rho = np.array([0.2, 0.8])
    for method in ["pearson", "gauss_rank"]:
        r, v = transform_corr_normal(rho, method, return_var=True)
        assert_equal(r, rho)
        assert_allclose(v, vp, atol=0.05)

    method = "kendal"
    r, v = transform_corr_normal(rho, method, return_var=True)
    # hardcoded transformation from BCC 2012 (round trip test)
    assert_allclose(2 / np.pi * np.arcsin(r), rho, atol=0.05)
    assert_allclose(v, vk, atol=0.05)

    method = "quadrant"
    rho_ = 0.8  # Croux, Dehon 2010
    r, v = transform_corr_normal(rho_, method, return_var=True)
    # hardcoded transformation from BCC 2012 (round trip test)
    assert_allclose(2 / np.pi * np.arcsin(r), rho_, atol=0.05)
    assert_allclose(v, 0.58, atol=0.05)

    method = "spearman"
    r, v = transform_corr_normal(rho, method, return_var=True)
    # hardcoded transformation from BCC 2012 (round trip test)
    assert_allclose(6 / np.pi * np.arcsin(r / 2), rho, atol=0.05)
    assert_allclose(v, vs, atol=0.05)


def test_corr_qu_ns_REGRESSION():
    # regression tests, numbers from results
    nobs, k_vars = 100, 3
    mean = np.zeros(k_vars)
    cov = linalg.toeplitz(1.0 / np.arange(1, k_vars + 1))

    rs = np.random.RandomState(187649)
    x = rs.multivariate_normal(mean, cov, size=nobs)
    x = np.round(x, 3)

    res_cns = np.array(
        [
            [1.0, 0.39765225, 0.27222425],
            [0.39765225, 1.0, 0.38073085],
            [0.27222425, 0.38073085, 1.00000000],
        ]
    )
    cns = corr_normal_scores(x)
    assert_allclose(cns, res_cns, atol=1e-4)

    res_cnq = np.array([[1.0, 0.28, 0.12], [0.28, 1.0, 0.28], [0.12, 0.28, 1.00]])
    cnq = corr_quadrant(x)
    assert_allclose(cnq, res_cnq, atol=1e-4)


@pytest.mark.parametrize("func", [corr_rank, corr_normal_scores, corr_quadrant])
def test_corr_one_variable(func):
    # The correlation matrix of a single variable is a 1x1 matrix. corr_rank
    # and corr_normal_scores returned a scalar, the shape of np.corrcoef.
    rng = np.random.default_rng(8446)
    x = rng.standard_normal((40, 1))
    corr = func(x)
    assert corr.shape == (1, 1)
    assert_allclose(corr, [[1.0]])
    # two variables are unchanged
    x2 = np.column_stack([x, x + rng.standard_normal((40, 1))])
    corr2 = func(x2)
    assert corr2.shape == (2, 2)
    assert_allclose(np.diag(corr2), 1.0)
    assert 0 < corr2[0, 1] < 1
