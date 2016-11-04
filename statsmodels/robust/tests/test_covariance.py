
import numpy as np
from scipy import linalg
from numpy.testing import assert_allclose, assert_equal, assert_

from statsmodels import robust
import statsmodels.robust.covariance as robcov


def test_mahalanobis():
    x = np.random.randn(10, 3)

    d1 = (x**2).sum(1)
    d0 = robcov.mahalanobis(x, np.eye(3))
    assert_allclose(d0, d1, rtol=1e-10)
    d2 = robcov.mahalanobis(x, cov_inv=np.eye(3))
    assert_allclose(d2, d1, rtol=1e-10)

    d3 = robcov.mahalanobis(x, 2*np.eye(3))
    assert_allclose(d3, 0.5 * d1, rtol=1e-10)
    d4 = robcov.mahalanobis(x, cov_inv=2*np.eye(3))
    assert_allclose(d4, 2 * d1, rtol=1e-10)

def test_outliers_gy():
    # regression test and basic properties
    # no test for tie warnings
    seed = 567812 #123
    np.random.seed(seed)

    nobs = 1000
    x = np.random.randn(nobs)
    d = x**2
    d2 = d.copy()
    n_outl = 10
    d2[:n_outl] += 10
    res = robcov._outlier_gy(d2, distr=None, k_endog=1, trim_prob=0.975)
    # next is regression test
    res1 = [0.017865444296085831, 8.4163674239050081, 17.0, 42.0,
            5.0238861873148881]
    assert_allclose(res, res1, rtol=1e-13)
    reject_thr = (d2 > res[1]).sum()
    reject_float = nobs * res[0]
    assert_equal(reject_thr, res[2])
    assert_equal(int(reject_float), res[2])
    # tests for fixed cutoff at 0.975
    assert_equal((d2 > res[4]).sum(), res[3])
    assert_allclose(res[3], nobs * 0.025  + n_outl, rtol=0.5)
    # + n_outl because not under Null

    x3 = x[:-1].reshape(-1, 3)
    # standardize, otherwise the sample wouldn't be close enough to distr
    x3 = (x3 - x3.mean(0)) / x3.std(0)
    d3 = (x3**2).sum(1)
    nobs = len(d3)
    n_outl = 0

    res = robcov._outlier_gy(d3, distr=None, k_endog=3, trim_prob=0.975)
    # next is regression test
    res1 = [0.0085980695527445583, 12.605802816238732, 2.0, 9.0,
            9.3484036044961485]
    assert_allclose(res, res1, rtol=1e-13)
    reject_thr = (d3 > res[1]).sum()
    reject_float = nobs * res[0]
    assert_equal(reject_thr, res[2])
    assert_equal(int(reject_float), res[2])
    # tests for fixed cutoff at 0.975
    assert_equal((d3 > res[4]).sum(), res[3])
    assert_allclose(res[3], nobs * 0.025  + n_outl, rtol=0.5)
    # fixed cutoff at 0.975, + n_outl because not under Null


def test_robcov_SMOKE():
    # currently only smoke test
    nobs, k_vars = 100, 3

    mean = np.zeros(k_vars)
    cov = linalg.toeplitz(1. / np.arange(1, k_vars+1))

    np.random.seed(187649)
    x = np.random.multivariate_normal(mean, cov, size=nobs)
    n_outliers = 1
    x[0,:2] = 50

    xtx = x.T.dot(x)
    cov_emp = np.cov(x.T)
    cov_clean = np.cov(x[n_outliers:].T)

    # GK, OGK
    robcov.cov_gk1(x[:, 0], x[:,2])
    robcov.cov_gk(x)
    robcov.cov_ogk(x)

    # Tyler
    robcov.cov_tyler(x)
    robcov.cov_tyler_regularized(x, shrinkage_factor=0.1)

    x2 = np.array([x, x])
    x2_ = np.rollaxis(x2, 1)
    robcov.cov_tyler_pairs_regularized(x2_,
                                start_cov=np.diag(robust.mad(x)**2),
                                shrinkage_factor=0.1,
                                nobs=x.shape[0], k_vars=x.shape[1])

    # others, M-, ...

    # estimation for multivariate t
    r = robcov._cov_iter(x, robcov.weights_mvt, weights_args=(3, k_vars))
    # trimmed sample covariance
    r = robcov._cov_iter(x, robcov.weights_quantile, weights_args=(0.50, ),
                         rescale="med")

    # We use 0.75 quantile for truncation to get better efficiency
    # at q=0.5, cov is pretty noisy at nobs=100 and passes at rtol=1
    res_li = robcov._cov_starting(x, is_standardized=False, quantile=0.75)
    for i, res in enumerate(res_li):
        # note: basic cov are not properly scaled
        # check only those with _cov_iter rescaling
        if hasattr(res, 'cov'):
            # inconsistent returns, redundant for now b/c no arrays
            c = getattr(res, 'cov', res)
            # rough comparison with DGP cov
            assert_allclose(c, cov, rtol=0.5)
            # check average scaling
            assert_allclose(np.diag(c).sum(), np.diag(cov).sum(), rtol=0.25)
            c1, m1 = robcov._reweight(x, res.mean, res.cov)
            assert_allclose(c1, cov, rtol=0.5)
            assert_allclose(c1, cov_clean, rtol=0.25) # oracle, w/o outliers
