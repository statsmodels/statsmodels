"""
Created on Sat Apr 16 15:02:13 2011
@author: Josef Perktold
"""

from statsmodels.compat.scipy import SP_LT_116

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
from scipy.special import gamma as sps_gamma

from statsmodels.sandbox.distributions.multivariate import (
    bghfactor,
    chi2_pdf,
    chi_logpdf,
    chi_pdf,
    multivariate_t_rvs,
    mvstdnormcdf,
    mvstdtprob,
)
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal


class Test_MVN_MVT_prob:
    # test for block integratal, cdf, of multivariate t and normal
    # comparison results from R

    @classmethod
    def setup_class(cls):
        cls.corr_equal = np.asarray([[1.0, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
        cls.a = -1 * np.ones(3)
        cls.b = 3 * np.ones(3)
        cls.df = 4

        corr2 = cls.corr_equal.copy()
        corr2[2, 1] = -0.5
        cls.corr2 = corr2

    @pytest.mark.skipif(not SP_LT_116, reason="mvndst is not available in scipy 1.16+")
    def test_mvn_mvt_1(self):
        a, b = self.a, self.b
        df = self.df
        corr_equal = self.corr_equal
        # result from R, mvtnorm with option
        # algorithm = GenzBretz(maxpts = 100000, abseps = 0.000001, releps = 0)
        #     or higher
        probmvt_R = 0.60414  # report, ed error approx. 7.5e-06
        probmvn_R = 0.673970  # reported error approx. 6.4e-07
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr_equal, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr_equal, abseps=1e-5), 4)

        mvn_high = mvstdnormcdf(a, b, corr_equal, abseps=1e-8, maxpts=10000000)
        assert_almost_equal(probmvn_R, mvn_high, 5)
        # this still barely fails sometimes at 6 why?? error is -7.2627419411830374e-007
        # >>> 0.67396999999999996 - 0.67397072627419408
        # -7.2627419411830374e-007
        # >>> assert_almost_equal(0.67396999999999996, 0.67397072627419408, 6)
        # Fail

    @pytest.mark.skipif(not SP_LT_116, reason="mvndst is not available in scipy 1.16+")
    def test_mvn_mvt_2(self):
        a, b = self.a, self.b
        df = self.df
        corr2 = self.corr2

        probmvn_R = 0.6472497  # reported error approx. 7.7e-08
        probmvt_R = 0.5881863  # highest reported error up to approx. 1.99e-06
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr2, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr2, abseps=1e-5), 4)

    @pytest.mark.skipif(not SP_LT_116, reason="mvndst is not available in scipy 1.16+")
    def test_mvn_mvt_3(self):
        a, b = self.a, self.b
        df = self.df
        corr2 = self.corr2

        a2 = a.copy()
        a2[:] = -np.inf
        # using higher precision in R, error approx. 6.866163e-07
        probmvn_R = 0.9961141
        # using higher precision in R, error approx. 1.6e-07
        probmvt_R = 0.9522146
        quadkwds = {"epsabs": 1e-08}
        probmvt = mvstdtprob(a2, b, corr2, df, quadkwds=quadkwds)
        assert_allclose(probmvt_R, probmvt, atol=5e-4)
        probmvn = mvstdnormcdf(a2, b, corr2, maxpts=100000, abseps=1e-5)
        assert_allclose(probmvn_R, probmvn, atol=1e-4)

    @pytest.mark.skipif(not SP_LT_116, reason="mvndst is not available in scipy 1.16+")
    def test_mvn_mvt_4(self):
        a = self.a
        df = self.df
        corr2 = self.corr2

        # from 0 to inf
        # print '0 inf'
        a2 = a.copy()
        a2[:] = -np.inf
        probmvn_R = 0.1666667  # error approx. 6.1e-08
        probmvt_R = 0.1666667  # error approx. 8.2e-08
        assert_almost_equal(probmvt_R, mvstdtprob(np.zeros(3), -a2, corr2, df), 4)
        assert_almost_equal(
            probmvn_R,
            mvstdnormcdf(np.zeros(3), -a2, corr2, maxpts=100000, abseps=1e-5),
            4,
        )

    @pytest.mark.skipif(not SP_LT_116, reason="mvndst is not available in scipy 1.16+")
    def test_mvn_mvt_5(self):
        df = self.df
        corr2 = self.corr2

        # unequal integration bounds
        # print "ue"
        a3 = np.array([0.5, -0.5, 0.5])
        probmvn_R = 0.06910487  # using higher precision in R, error approx. 3.5e-08
        probmvt_R = 0.05797867  # using higher precision in R, error approx. 5.8e-08
        assert_almost_equal(mvstdtprob(a3, a3 + 1, corr2, df), probmvt_R, 4)
        assert_almost_equal(
            probmvn_R, mvstdnormcdf(a3, a3 + 1, corr2, maxpts=100000, abseps=1e-5), 4
        )


class TestMVDistributions:
    # this is not well organized

    @classmethod
    def setup_class(cls):
        mu3 = [-1, 0.0, 2.0]
        cov3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])
        cls.mu3 = mu3
        cls.cov3 = cov3

        mvn3 = MVNormal(mu3, cov3)
        mvn3c = MVNormal(np.array([0, 0, 0]), cov3)
        cls.mvn3 = mvn3
        cls.mvn3c = mvn3c

    def test_mvn_pdf(self):
        cov3 = self.cov3
        mvn3 = self.mvn3

        r_val = [-7.667977543898155, -6.917977543898155, -5.167977543898155]
        assert_allclose(mvn3.logpdf(cov3), r_val, rtol=1e-13)

        r_val = [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
        assert_allclose(mvn3.pdf(cov3), r_val, rtol=1e-13)

        mvn3b = MVNormal(np.array([0, 0, 0]), cov3)
        r_val = [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]
        assert_allclose(mvn3b.pdf(cov3), r_val, rtol=1e-13)

    def test_mvt_pdf(self):
        rs = np.random.RandomState(38909894)
        cov3 = self.cov3
        mu3 = self.mu3

        mvt = MVT((0, 0), 1, 5)
        assert_almost_equal(
            mvt.logpdf(np.array([0.0, 0.0])), -1.837877066409345, decimal=15
        )
        assert_almost_equal(
            mvt.pdf(np.array([0.0, 0.0])), 0.1591549430918953, decimal=15
        )

        mvt.logpdf(np.array([1.0, 1.0])) - (-3.01552989458359)

        mvt1 = MVT((0, 0), 1, 1)
        mvt1.logpdf(np.array([1.0, 1.0])) - (-3.48579549941151)  # decimal=16

        rvs = mvt.rvs(100000, rng=rs)
        assert_almost_equal(np.cov(rvs, rowvar=False), mvt.cov, decimal=1)

        mvt31 = MVT(mu3, cov3, 1)
        assert_almost_equal(
            mvt31.pdf(cov3),
            [0.0007276818698165781, 0.0009980625182293658, 0.0027661422056214652],
            decimal=17,
        )

        mvt = MVT(mu3, cov3, 3)
        assert_almost_equal(
            mvt.pdf(cov3),
            [0.000863777424247410, 0.001277510788307594, 0.004156314279452241],
            decimal=17,
        )


def test_chi2_pdf_matches_scipy_with_workaround_extra_arg():
    # the only way to actually call this function today
    result = chi2_pdf(2.0, 3)
    assert_allclose(result, stats.chi2.pdf(2.0, 3), rtol=1e-10)


def test_chi_pdf_matches_scipy():
    x, df = 1.5, 4
    assert_allclose(chi_pdf(x, df), stats.chi.pdf(x, df), rtol=1e-10)


def test_chi_logpdf_matches_scipy():
    x, df = 1.5, 4
    assert_allclose(chi_logpdf(x, df), stats.chi.logpdf(x, df), rtol=1e-10)


def test_chi_pdf_is_exp_of_chi_logpdf():
    x, df = 2.3, 7
    assert_allclose(chi_pdf(x, df), np.exp(chi_logpdf(x, df)), rtol=1e-10)


def test_bghfactor_matches_direct_formula():
    for df in [2, 4, 6.5, 10]:
        expected = np.power(2.0, 1 - df * 0.5) / sps_gamma(df * 0.5)
        assert_allclose(bghfactor(df), expected, rtol=1e-10)


class TestMultivariateTRvs:
    def test_shape(self):
        rvs = multivariate_t_rvs([0.0, 0.0], np.eye(2), df=5, n=37, rng=0)
        assert rvs.shape == (37, 2)

    def test_reproducible_with_same_rng_seed(self):
        S = np.array([[1.0, 0.5], [0.5, 1.0]])
        r1 = multivariate_t_rvs([10.0, 20.0], S, df=6, n=10, rng=42)
        r2 = multivariate_t_rvs([10.0, 20.0], S, df=6, n=10, rng=42)
        assert_allclose(r1, r2)

    def test_moments_match_theoretical_by_monte_carlo(self):
        mean = np.array([10.0, 20.0])
        S = np.array([[1.0, 0.5], [0.5, 1.0]])
        df = 6
        rvs = multivariate_t_rvs(mean, S, df=df, n=300000, rng=0)
        assert_allclose(rvs.mean(0), mean, atol=0.05)
        expected_cov = S * df / (df - 2)
        assert_allclose(np.cov(rvs, rowvar=False), expected_cov, atol=0.05)

    def test_df_inf_reduces_to_multivariate_normal(self):
        S = np.array([[1.0, 0.5], [0.5, 1.0]])
        rvs = multivariate_t_rvs([0.0, 0.0], S, df=np.inf, n=300000, rng=0)
        assert_allclose(rvs.mean(0), [0.0, 0.0], atol=0.02)
        assert_allclose(np.cov(rvs, rowvar=False), S, atol=0.02)
