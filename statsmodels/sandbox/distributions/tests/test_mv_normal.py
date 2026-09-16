"""
Tests corresponding to sandbox.distributions.mv_normal
"""
from statsmodels.compat.scipy import SP_LT_116

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from scipy import stats

from statsmodels.sandbox.distributions.mv_normal import (
    MVT,
    BivariateNormal,
    MVNormal,
    MVNormal0,
    bivariate_normal,
    expect_mc,
)

MEAN3 = np.array([-1, 0.0, 2.0])
COV3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])


def test_bivariate_normal_function_matches_scipy():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    mu = [0.0, 0.0]
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    result = bivariate_normal(x, mu, cov)
    expected = stats.multivariate_normal.pdf(x, mean=mu, cov=cov)
    assert_allclose(result, expected, rtol=1e-10)


def test_bivariate_normal_class_construction():
    BivariateNormal([0.0, 0.0], np.eye(2))


class TestMVNormal:
    def setup_method(self):
        self.mvn = MVNormal(MEAN3, COV3)

    def test_logpdf_matches_scipy(self):
        pt = np.array([0.0, 1.0, 1.5])
        result = self.mvn.logpdf(pt)
        expected = stats.multivariate_normal.logpdf(pt, mean=MEAN3, cov=COV3)
        assert_allclose(result, expected, rtol=1e-10)

    def test_logpdf_2d_input_matches_scipy_rowwise(self):
        pts = np.array([[0.0, 1.0, 1.5], [-1.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
        result = self.mvn.logpdf(pts)
        expected = stats.multivariate_normal.logpdf(pts, mean=MEAN3, cov=COV3)
        assert_allclose(result, expected, rtol=1e-10)

    def test_pdf_is_exp_logpdf(self):
        pt = np.array([0.0, 1.0, 1.5])
        assert_allclose(self.mvn.pdf(pt), np.exp(self.mvn.logpdf(pt)))

    def test_cov_property_equals_sigma(self):
        assert_array_almost_equal(self.mvn.cov, COV3)

    def test_std_and_corr_consistent_with_cov(self):
        std = self.mvn.std
        corr = self.mvn.corr
        assert_array_almost_equal(std, np.sqrt(np.diag(COV3)))
        reconstructed = corr * np.outer(std, std)
        assert_array_almost_equal(reconstructed, COV3)

    def test_rvs_moments_match_by_monte_carlo(self):
        rvs = self.mvn.rvs(size=200000)
        assert_allclose(rvs.mean(0), MEAN3, atol=0.02)
        assert_allclose(np.cov(rvs, rowvar=False), COV3, atol=0.05)

    @pytest.mark.skipif(
        not SP_LT_116,
        reason=(
            "MVNormal.cdf() delegates to sandbox.distributions.extras."
            "mvnormcdf -> mvstdnormcdf -> mvndst, a compiled scipy.stats "
            "Fortran routine that was removed in SciPy >= 1.16.0 (see "
            "sandbox.distributions.tests.test_multivariate for the same "
            "issue against mvstdtprob/mvstdnormcdf directly)."
        ),
    )
    def test_cdf_matches_scipy(self):
        pt = np.array([0.0, 1.0, 1.5])
        result = self.mvn.cdf(pt)
        expected = stats.multivariate_normal.cdf(pt, mean=MEAN3, cov=COV3)
        assert_allclose(result, expected, atol=1e-4)

    def test_whiten_standardize_relationship(self):
        x = np.array([0.5, 1.0, 2.5])
        assert_array_almost_equal(
            self.mvn.standardize(x), self.mvn.whiten(x - MEAN3)
        )

    def test_normalized_has_correlation_as_sigma_and_zero_mean(self):
        normalized = self.mvn.normalized()
        assert_array_almost_equal(normalized.sigma, self.mvn.corr)
        assert_array_almost_equal(normalized.mean, np.zeros(3))

    def test_normalized_not_demeaned(self):
        normalized = self.mvn.normalized(demeaned=False)
        assert_array_almost_equal(normalized.mean, MEAN3 / self.mvn.std_sigma)

    def test_normalized2_sigma_matches_normalized(self):
        assert_array_almost_equal(
            self.mvn.normalized().sigma, self.mvn.normalized2().sigma
        )

    @pytest.mark.xfail(
        reason=(
            "BUG: normalized2(demeaned=True) computes shift=-self.mean and "
            "passes it to affine_transformed(shift, scale), which computes "
            "mean_new = scale @ mean + shift. That only cancels to zero "
            "when scale is the identity; here scale=diag(1/std_sigma), so "
            "the result is mean/std_sigma - mean, not zero. normalized() "
            "(a separate, non-affine implementation) gets this right by "
            "directly setting mean_new = zeros. The same shift-vs-scale "
            "bug affects .standardized(), see "
            "test_standardized_mean_is_zero below."
        ),
        raises=AssertionError,
        strict=True,
    )
    def test_normalized2_mean_matches_normalized(self):
        assert_array_almost_equal(
            self.mvn.normalized().mean, self.mvn.normalized2().mean
        )

    def test_marginal(self):
        marg = self.mvn.marginal(np.array([0, 2]))
        assert_array_almost_equal(marg.mean, MEAN3[[0, 2]])
        assert_array_almost_equal(marg.sigma, COV3[np.ix_([0, 2], [0, 2])])

    def test_conditional_matches_textbook_formula(self):
        keep = np.array([0])
        given = np.array([1, 2])
        values = np.array([1.0, 1.5])
        cond = self.mvn.conditional(keep, values)

        sigmakk = COV3[np.ix_(keep, keep)]
        sigmagg = COV3[np.ix_(given, given)]
        sigmakg = COV3[np.ix_(keep, given)]
        expected_mean = MEAN3[keep] + sigmakg @ np.linalg.solve(
            sigmagg, values - MEAN3[given]
        )
        expected_sigma = sigmakk - sigmakg @ np.linalg.solve(sigmagg, sigmakg.T)
        assert_array_almost_equal(cond.mean, expected_mean)
        assert_array_almost_equal(cond.sigma, expected_sigma)

    def test_affine_transformed(self):
        shift = np.array([1.0, 1.0, 1.0])
        scale = np.eye(3) * 2
        transformed = self.mvn.affine_transformed(shift, scale)
        assert_array_almost_equal(transformed.mean, shift + scale @ MEAN3)
        assert_array_almost_equal(transformed.sigma, scale @ COV3 @ scale.T)

    def test_standardized_has_identity_sigma(self):
        standardized = self.mvn.standardized()
        assert_array_almost_equal(standardized.sigma, np.eye(3), decimal=8)

    @pytest.mark.xfail(
        reason=(
            "BUG: standardized() calls "
            "affine_transformed(-self.mean, self.cholsigmainv), but "
            "affine_transformed computes mean_new = B @ mean + shift. "
            "With B=cholsigmainv and shift=-mean (unscaled), the result is "
            "cholsigmainv @ mean - mean, not zero. The instance *method* "
            ".standardize(x) is implemented correctly (it does whiten(x - "
            "mean), see test_whiten_standardize_relationship and "
            "test_manual_standardize_of_samples_has_zero_mean); only the "
            "*factory* .standardized(), which builds a new zero-mean "
            "MVNormal via the generic affine-transform path, has the bug. "
            "The correct shift would be -B @ self.mean."
        ),
        raises=AssertionError,
        strict=True,
    )
    def test_standardized_mean_is_zero(self):
        standardized = self.mvn.standardized()
        assert_array_almost_equal(standardized.mean, np.zeros(3))

    def test_manual_standardize_of_samples_has_zero_mean(self):
        # the .standardize(x) instance method (as opposed to the buggy
        # .standardized() factory above) correctly produces zero-mean
        # output when applied to samples from the distribution
        rng = np.random.default_rng(0)
        rvs = rng.multivariate_normal(MEAN3, COV3, size=200000)
        whitened = self.mvn.standardize(rvs)
        assert_allclose(whitened.mean(0), np.zeros(3), atol=0.02)
        assert_allclose(np.cov(whitened, rowvar=False), np.eye(3), atol=0.05)


class TestMVNormalConstruction:
    def test_scalar_sigma_is_iid(self):
        mvn = MVNormal([0.0, 0.0], 2.0)
        assert_array_almost_equal(mvn.cov, 2.0 * np.eye(2))

    def test_1d_sigma_is_diagonal(self):
        mvn = MVNormal([0.0, 0.0, 0.0], np.array([1.0, 2.0, 3.0]))
        assert_array_almost_equal(mvn.cov, np.diag([1.0, 2.0, 3.0]))

    def test_invalid_sigma_shape_raises(self):
        with pytest.raises(ValueError, match="sigma has invalid shape"):
            MVNormal([0.0, 0.0], np.ones((3, 2)))


class TestMVT:
    def setup_method(self):
        self.df = 5
        self.mvt = MVT(MEAN3, COV3, self.df)

    def test_logpdf_matches_scipy_multivariate_t(self):
        pt = np.array([0.0, 1.0, 1.5])
        result = self.mvt.logpdf(pt)
        expected = stats.multivariate_t.logpdf(
            pt, loc=MEAN3, shape=COV3, df=self.df
        )
        assert_allclose(result, expected, rtol=1e-10)

    def test_cov_formula_for_df_greater_than_2(self):
        expected = self.df / (self.df - 2.0) * COV3
        assert_array_almost_equal(self.mvt.cov, expected)

    def test_cov_is_nan_for_df_leq_2(self):
        mvt2 = MVT(MEAN3, COV3, df=2)
        assert np.all(np.isnan(mvt2.cov))
        mvt1 = MVT(MEAN3, COV3, df=1)
        assert np.all(np.isnan(mvt1.cov))

    def test_rvs_covariance_matches_by_monte_carlo(self):
        rvs = self.mvt.rvs(size=200000)
        assert_allclose(np.cov(rvs, rowvar=False), self.mvt.cov, atol=0.1)

    def test_affine_transformed(self):
        shift = np.array([1.0, 1.0, 1.0])
        scale = np.eye(3) * 2
        transformed = self.mvt.affine_transformed(shift, scale)
        assert_array_almost_equal(transformed.mean, shift + scale @ MEAN3)
        assert_array_almost_equal(transformed.sigma, scale @ COV3 @ scale.T)
        assert transformed.df == self.df

    def test_marginal_preserves_df(self):
        marg = self.mvt.marginal(np.array([0, 2]))
        assert marg.df == self.df
        assert_array_almost_equal(marg.mean, MEAN3[[0, 2]])


class TestMVNormal0:
    # legacy reference implementation kept alongside MVNormal/MVElliptical
    def test_logpdf_matches_mvnormal(self):
        legacy = MVNormal0(MEAN3, COV3)
        current = MVNormal(MEAN3, COV3)
        pt = np.array([0.0, 1.0, 1.5])
        assert_allclose(legacy.logpdf(pt), current.logpdf(pt), rtol=1e-10)

    @pytest.mark.singleton_randomstate
    def test_rvs_shape(self):
        # MVNormal0.rvs() calls np.random.multivariate_normal directly
        # (the legacy singleton RandomState API, unlike MVNormal.rvs()
        # which threads an explicit rng/Generator), so this legitimately
        # mutates global numpy random state.
        legacy = MVNormal0(MEAN3, COV3)
        rvs = legacy.rvs(size=5)
        assert rvs.shape == (5, 3)


def test_expect_mc_estimates_normal_orthant_probability():
    # expected value of an indicator function via Monte Carlo should be
    # close to the true probability computed from the cdf
    mvn = MVNormal([0.0, 0.0], np.eye(2))
    a = np.array([0.0, 0.0])
    result = expect_mc(mvn, lambda x: (x < a).all(-1), size=200000)
    expected = stats.multivariate_normal.cdf(a, mean=[0.0, 0.0], cov=np.eye(2))
    assert_allclose(result, expected, atol=0.01)
