
import os

import numpy as np
from scipy import linalg
from numpy.testing import assert_allclose, assert_equal
import pandas as pd

from statsmodels import robust
import statsmodels.robust.covariance as robcov

from .results import results_cov as res_cov


cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'hbk.csv'
file_path = os.path.join(cur_dir, 'results', file_name)

dta_hbk = pd.read_csv(file_path)


def test_mahalanobis():
    np.random.seed(987676453)
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
    seed = 567812  # 123
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
    assert_allclose(res[3], nobs * 0.025 + n_outl, rtol=0.5)
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
    assert_allclose(res[3], nobs * 0.025 + n_outl, rtol=0.5)
    # fixed cutoff at 0.975, + n_outl because not under Null


class TestOGKMad():

    @classmethod
    def setup_class(cls):

        cls.res1 = robcov.cov_ogk(dta_hbk, rescale=False, ddof=0, reweight=0.9)
        cls.res2 = res_cov.results_ogk_mad

    def test(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.cov, res2.cov, rtol=1e-13)
        assert_allclose(res1.mean, res2.center, rtol=1e-13)
        assert_allclose(res1.cov_raw, res2.cov_raw, rtol=1e-11)
        assert_allclose(res1.loc_raw, res2.center_raw, rtol=1e-10)


class TestOGKTau(TestOGKMad):

    @classmethod
    def setup_class(cls):

        def sfunc(x):
            return robcov.scale_tau(x, normalize=False, ddof=0)[1]

        cls.res1 = robcov.cov_ogk(dta_hbk,
                                  scale_func=sfunc,
                                  rescale=False, ddof=0, reweight=0.9)
        cls.res2 = res_cov.results_ogk_tau

    def test(self):
        # not inherited because of weak agreement with R
        # I did not find options to improve agreement
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.cov, res2.cov, atol=0.05, rtol=1e-13)
        assert_allclose(res1.mean, res2.center, atol=0.03, rtol=1e-13)
        # cov raw differs in scaling, no idea why
        # note rrcov uses C code for this case with hardoced tau scale
        # our results are "better", i.e. correct outliers same as dgp
        # rrcov has one extra outlier
        fact = 1.1356801031633883
        assert_allclose(res1.cov_raw, res2.cov_raw * fact, rtol=1e-11)
        assert_allclose(res1.loc_raw, res2.center_raw, rtol=0.2, atol=0.1)


def test_tyler():

    # > library(ICSNP)
    # > resty = tyler.shape(hbk, location = ccogk$center, eps=1e-13,
    # print.it=TRUE)
    # [1] "convergence was reached after 55 iterations"

    res2 = np.array([
        [1.277856643343122, 0.298374848328023, 0.732491311584908,
         0.232045093295329],
        [0.298374848328023, 1.743589223324287, 1.220675037619406,
         0.212549156887607],
        [0.732491311584907, 1.220675037619407, 2.417486791841682,
         0.295767635758891],
        [0.232045093295329, 0.212549156887607, 0.295767635758891,
         0.409157014373402]
        ])

    # center is from an OGK version
    center = np.array(
        [1.5583333333333333, 1.8033333333333335, 1.6599999999999999,
         -0.0866666666666667]
        )
    k_vars = len(center)

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center, normalize="trace")
    assert_allclose(np.trace(res1.cov), k_vars, rtol=1e-13)
    cov_det = res1.cov / np.linalg.det(res1.cov)**(1. / k_vars)
    assert_allclose(cov_det, res2, rtol=1e-11)
    assert res1.n_iter == 55

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center, normalize="det")
    assert_allclose(np.linalg.det(res1.cov), 1, rtol=1e-13)
    assert_allclose(res1.cov, res2, rtol=1e-11)
    assert res1.n_iter == 55

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center, normalize="normal")
    cov_det = res1.cov / np.linalg.det(res1.cov)**(1. / k_vars)
    assert_allclose(cov_det, res2, rtol=1e-11)
    assert res1.n_iter == 55

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center)
    cov_det = res1.cov / np.linalg.det(res1.cov)**(1. / k_vars)
    assert_allclose(cov_det, res2, rtol=1e-11)
    assert res1.n_iter == 55


def test_robcov_SMOKE():
    # currently only smoke test or very loose comparisons to dgp
    nobs, k_vars = 100, 3

    mean = np.zeros(k_vars)
    cov = linalg.toeplitz(1. / np.arange(1, k_vars+1))

    np.random.seed(187649)
    x = np.random.multivariate_normal(mean, cov, size=nobs)
    n_outliers = 1
    x[0, :2] = 50

    # xtx = x.T.dot(x)
    # cov_emp = np.cov(x.T)
    cov_clean = np.cov(x[n_outliers:].T)

    # GK, OGK
    robcov.cov_gk1(x[:, 0], x[:, 2])
    robcov.cov_gk(x)
    robcov.cov_ogk(x)

    # Tyler
    robcov.cov_tyler(x)
    robcov.cov_tyler_regularized(x, shrinkage_factor=0.1)

    x2 = np.array([x, x])
    x2_ = np.rollaxis(x2, 1)
    robcov.cov_tyler_pairs_regularized(
        x2_,
        start_cov=np.diag(robust.mad(x)**2),
        shrinkage_factor=0.1,
        nobs=x.shape[0],
        k_vars=x.shape[1],
        )

    # others, M-, ...

    # estimation for multivariate t
    r = robcov._cov_iter(x, robcov.weights_mvt, weights_args=(3, k_vars))
    # rough comparison with DGP cov
    assert_allclose(r.cov, cov, rtol=0.5)

    # trimmed sample covariance
    r = robcov._cov_iter(x, robcov.weights_quantile, weights_args=(0.50, ),
                         rescale="med")
    # rough comparison with DGP cov
    assert_allclose(r.cov, cov, rtol=0.5)

    # We use 0.75 quantile for truncation to get better efficiency
    # at q=0.5, cov is pretty noisy at nobs=100 and passes at rtol=1
    res_li = robcov._cov_starting(x, is_standardized=False, quantile=0.75)
    for _, res in enumerate(res_li):
        # note: basic cov are not properly scaled
        # check only those with _cov_iter rescaling, `n_iter`
        # include also ogk
        # need more generic detection of appropriate cov
        if hasattr(res, 'n_iter') or hasattr(res, 'cov_ogk_raw'):
            # inconsistent returns, redundant for now b/c no arrays
            c = getattr(res, 'cov', res)
            # rough comparison with DGP cov
            assert_allclose(c, cov, rtol=0.5)
            # check average scaling
            assert_allclose(np.diag(c).sum(), np.diag(cov).sum(), rtol=0.25)
            c1, m1 = robcov._reweight(x, res.mean, res.cov)
            assert_allclose(c1, cov, rtol=0.5)
            assert_allclose(c1, cov_clean, rtol=0.25)  # oracle, w/o outliers
            assert_allclose(m1, mean, rtol=0.5, atol=0.2)
