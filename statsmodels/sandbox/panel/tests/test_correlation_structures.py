"""
Tests corresponding to sandbox.panel.correlation_structures
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
import pytest

from statsmodels.sandbox.panel.correlation_structures import (
    ARCovariance,
    corr2cov,
    corr_ar,
    corr_arma,
    corr_equi,
    whiten_ar,
    yule_walker_acov,
)


def test_corr_equi():
    corr = corr_equi(3, 0.5)
    expected = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    assert_array_equal(corr, expected)
    assert_array_equal(np.diag(corr), np.ones(3))


def test_corr_ar_pads_short_ar():
    # ar shorter than k_vars is zero-padded before building the Toeplitz
    # matrix, giving the documented (k_vars, k_vars) shape
    corr = corr_ar(5, [1, 0.5])
    assert corr.shape == (5, 5)
    expected = np.array(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.5, 1.0],
        ]
    )
    assert_array_almost_equal(corr, expected)


@pytest.mark.xfail(
    reason=(
        "BUG: corr_ar only pads `ar` up to k_vars when it is *shorter* "
        "than k_vars; when `ar` is already >= k_vars long it is passed to "
        "toeplitz() unmodified/untruncated, so the returned matrix has "
        "shape (len(ar), len(ar)) instead of the documented (k_vars, "
        "k_vars)."
    ),
    raises=AssertionError,
    strict=True,
)
def test_corr_ar_truncates_long_ar():
    k_vars = 3
    corr = corr_ar(k_vars, [1, 0.5, 0.3, 0.1, 0.05])
    assert corr.shape == (k_vars, k_vars)


def test_corr_arma():
    corr = corr_arma(4, [1, 0.5], [1, 0.2])
    assert corr.shape == (4, 4)
    assert_array_equal(np.diag(corr), np.ones(4))
    # symmetric, as any correlation matrix from a Toeplitz construction
    assert_array_almost_equal(corr, corr.T)


def test_corr2cov_roundtrip_with_cov2corr():
    from statsmodels.stats.moment_helpers import cov2corr

    corr = corr_equi(3, 0.4)
    std = np.array([1.0, 2.0, 3.0])
    cov = corr2cov(corr, std)
    assert_array_almost_equal(np.diag(cov), std**2)
    # corr2cov and cov2corr (the real, non-sandbox helper) are inverses
    assert_array_almost_equal(cov2corr(cov), corr)


def test_corr2cov_scalar_std():
    corr = corr_equi(3, 0.4)
    cov_scalar = corr2cov(corr, 2.0)
    cov_array = corr2cov(corr, np.array([2.0, 2.0, 2.0]))
    assert_array_almost_equal(cov_scalar, cov_array)


def test_whiten_ar_matches_manual_ar1_transform():
    x = np.arange(10.0)
    rho = 0.5
    result = whiten_ar(x, np.array([rho]), order=1)
    expected = x[1:] - rho * x[:-1]
    assert_array_almost_equal(result, expected)
    assert_array_equal(result.shape, (9,))


def test_whiten_ar_order_zero_is_identity():
    x = np.arange(5.0)
    result = whiten_ar(x, np.array([]), order=0)
    assert_array_almost_equal(result, x)


def test_whiten_ar_2d():
    x = np.column_stack([np.arange(10.0), np.arange(10.0) * 2])
    rho = 0.5
    result = whiten_ar(x, np.array([rho]), order=1)
    expected = x[1:] - rho * x[:-1]
    assert_array_almost_equal(result, expected)


def test_yule_walker_acov_default_method_is_broken():
    # BUG: yule_walker_acov's default `method="unbiased"` is not a value
    # accepted by the underlying statsmodels.regression.linear_model
    # .yule_walker (which only accepts 'adjusted' or 'mle'), so calling
    # yule_walker_acov with its own default arguments always raises.
    acov = np.array([1.0, 0.5, 0.3])
    with pytest.raises(ValueError, match="method must be one of"):
        yule_walker_acov(acov, order=1)


def test_yule_walker_acov_with_valid_method():
    acov = np.array([1.0, 0.5, 0.3])
    rho, sigma = yule_walker_acov(acov, order=1, method="adjusted")
    assert_allclose(rho, [0.72761194], rtol=1e-6)
    assert sigma > 0


def test_arcovariance_corr_from_ar():
    cov = ARCovariance(ar=np.array([1, -0.5]))
    result = cov.corr()
    expected = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert_array_almost_equal(result, expected)


def test_arcovariance_corr_from_ar_coefs():
    cov = ARCovariance(ar_coefs=np.array([0.5]))
    result = cov.corr()
    expected = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert_array_almost_equal(result, expected)


def test_arcovariance_ar_and_ar_coefs_constructions_are_equivalent():
    # ar_coefs = -ar[1:], so these two constructions describe the same process
    from_ar = ARCovariance(ar=np.array([1, -0.5]))
    from_coefs = ARCovariance(ar_coefs=np.array([0.5]))
    assert_array_almost_equal(from_ar.corr(), from_coefs.corr())


@pytest.mark.xfail(
    reason=(
        "BUG: ARCovariance(ar_coefs=...) stores the coefficients as "
        "self.arcoefs (no underscore), but .whiten() reads self.ar_coefs "
        "(with underscore) -- a typo that means .whiten() always raises "
        "AttributeError when the instance was built via the ar_coefs= "
        "constructor path."
    ),
    raises=AttributeError,
    strict=True,
)
def test_arcovariance_whiten_after_ar_coefs_construction():
    cov = ARCovariance(ar_coefs=np.array([0.5]))
    cov.whiten(np.arange(10.0))


@pytest.mark.xfail(
    reason=(
        "BUG: ARCovariance.whiten() reads self.order, but no constructor "
        "path (ar= or ar_coefs=) ever sets self.order, so .whiten() always "
        "raises AttributeError regardless of how the instance was built."
    ),
    raises=AttributeError,
    strict=True,
)
def test_arcovariance_whiten_after_ar_construction():
    cov = ARCovariance(ar=np.array([1, -0.5]))
    cov.whiten(np.arange(10.0))


@pytest.mark.xfail(
    reason=(
        "BUG: ARCovariance.__init__ accepts a `sigma` parameter but never "
        "stores it as self.sigma, so .cov() always raises AttributeError. "
        "Separately, .cov() calls cov2corr(self.corr(...), self.sigma) -- "
        "cov2corr converts covariance to correlation, the opposite of what "
        "a method named .cov() should be doing; it likely meant to call "
        "this module's own corr2cov(corr, std) instead."
    ),
    raises=AttributeError,
    strict=True,
)
def test_arcovariance_cov():
    cov = ARCovariance(ar=np.array([1, -0.5]), sigma=2.0)
    cov.cov()


@pytest.mark.xfail(
    reason=(
        "BUG: ARCovariance.fit() calls yule_walker_acov(cov, order=order, "
        "**kwds) without a method= kwarg, hitting the same broken default "
        "as test_yule_walker_acov_default_method_is_broken, so .fit() "
        "always raises unless the caller passes method= explicitly."
    ),
    raises=ValueError,
    strict=True,
)
def test_arcovariance_fit_default_method():
    acov = np.array([1.0, 0.5, 0.3])
    ARCovariance.fit(acov, order=1)


def test_arcovariance_fit_with_explicit_method():
    acov = np.array([1.0, 0.5, 0.3])
    fitted = ARCovariance.fit(acov, order=1, method="adjusted")
    assert_allclose(fitted.arcoefs, [0.72761194], rtol=1e-6)
