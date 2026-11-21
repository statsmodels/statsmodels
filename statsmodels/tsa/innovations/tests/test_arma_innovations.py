"""
Tests for ARMA innovations algorithm wrapper
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests._contract import series_forms


@pytest.mark.parametrize("ar_params,ma_params,sigma2", [
    (np.array([]), np.array([]), 1),
    (np.array([0.]), np.array([0.]), 1),
    (np.array([0.9]), np.array([]), 1),
    (np.array([]), np.array([0.9]), 1),
    (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123),
    (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
])
def test_innovations_algo_filter_kalman_filter(ar_params, ma_params, sigma2):
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process
    rg = np.random.RandomState(42)
    endog = rg.normal(size=100)

    # Innovations algorithm approach
    llf = arma_innovations.arma_loglike(endog, ar_params, ma_params, sigma2)
    llf_obs = arma_innovations.arma_loglikeobs(endog, ar_params, ma_params,
                                               sigma2)
    score = arma_innovations.arma_score(endog, ar_params, ma_params, sigma2)
    score_obs = arma_innovations.arma_scoreobs(endog, ar_params, ma_params,
                                               sigma2)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    params = np.r_[ar_params, ma_params, sigma2]

    # Test that the two approaches are the same
    assert_allclose(llf, mod.loglike(params))
    assert_allclose(llf_obs, mod.loglikeobs(params))
    # Note: the tolerance on the two gets worse as more nobs are added
    assert_allclose(score, mod.score(params), atol=1e-5)
    assert_allclose(score_obs, mod.score_obs(params), atol=1e-5)


@pytest.mark.parametrize("ar_params", [(1.9, -0.8), (1.0), (2.0, -1.0)])
def test_innovations_nonstationary(ar_params):
    rg = np.random.RandomState(42)
    endog = rg.normal(size=100)
    with pytest.raises(ValueError, match="The model's autoregressive"):
        arma_innovations.arma_innovations(endog, ar_params=ar_params)


AR = np.array([0.5, -0.2])
MA = np.array([0.3])
SIGMA2 = 1.7
PARAMS = np.r_[AR, MA, SIGMA2]


@pytest.fixture(scope="module")
def arma_data():
    # Integer-valued so that the int dtype form holds the same values
    rng = np.random.default_rng([20261120, 7])
    return np.round(2 * rng.standard_normal(80))


def _kalman_filter(endog, order=(2, 0, 1), params=PARAMS):
    # The Kalman filter with the stationary initialization gives the exact
    # one-step prediction errors and variances, as the innovations algorithm
    # does, so the two agree by construction
    mod = SARIMAX(endog, order=order)
    return mod.filter(params).filter_results


@pytest.mark.parametrize("normalize", [False, True])
def test_innovations_match_kalman_filter(arma_data, normalize):
    u, v = arma_innovations.arma_innovations(
        arma_data, AR, MA, SIGMA2, normalize=normalize
    )
    kf = _kalman_filter(arma_data)
    # innovations_mse is the prediction error variance divided by sigma2
    assert_allclose(SIGMA2 * v, kf.forecasts_error_cov[0, 0], rtol=1e-9)
    if normalize:
        # Dividing by the square root of innovations_mse leaves variance sigma2
        expected = kf.standardized_forecasts_error[0] * np.sqrt(SIGMA2)
    else:
        expected = kf.forecasts_error[0]
    # The prediction errors cross zero, so atol is set from their scale (~1)
    assert_allclose(u, expected, rtol=1e-8, atol=1e-9)


def test_innovations_columns(arma_data):
    # Each column of a 2-d endog is filtered separately by the same process
    endog = np.column_stack([arma_data, arma_data[::-1] + 1])
    u, v = arma_innovations.arma_innovations(
        endog, AR, MA, SIGMA2, normalize=True
    )
    assert u.shape == endog.shape
    for i in range(2):
        kf = _kalman_filter(endog[:, i])
        assert_allclose(SIGMA2 * v, kf.forecasts_error_cov[0, 0], rtol=1e-9)
        expected = kf.standardized_forecasts_error[0] * np.sqrt(SIGMA2)
        # The prediction errors cross zero, so atol is set from their scale
        assert_allclose(u[:, i], expected, rtol=1e-8, atol=1e-9)


def test_innovations_explicit_prefix(arma_data):
    kf = _kalman_filter(arma_data)
    u, v = arma_innovations.arma_innovations(
        arma_data, AR, MA, SIGMA2, prefix="d"
    )
    assert u.dtype == np.float64
    # The prediction errors cross zero, so atol is set from their scale
    assert_allclose(u, kf.forecasts_error[0], rtol=1e-8, atol=1e-9)
    # The complex routines, with real values held in complex arrays
    u, v = arma_innovations.arma_innovations(
        arma_data.astype(complex),
        AR.astype(complex),
        MA.astype(complex),
        SIGMA2,
        prefix="z",
    )
    assert u.dtype == np.complex128
    assert_allclose(u.real, kf.forecasts_error[0], rtol=1e-8, atol=1e-9)
    # Some platforms (Linux, wasm) leave imaginary rounding noise of about
    # 1e-16 relative to the real parts, so atol is set from their scale
    assert_allclose(u.imag, 0.0, atol=1e-12 * np.abs(u.real).max())
    assert_allclose(SIGMA2 * v.real, kf.forecasts_error_cov[0, 0], rtol=1e-9)


@pytest.mark.parametrize("prefix,dtype,rtol", [
    ("s", np.float32, 1e-5),
    ("d", np.float64, 1e-9),
    ("z", np.complex128, 1e-9),
])
def test_loglikeobs_explicit_prefix(arma_data, prefix, dtype, rtol):
    # The inputs are cast to the type of the prefix. rtol allows for the
    # rounding of single precision.
    expected = SARIMAX(arma_data, order=(2, 0, 1)).loglikeobs(PARAMS)
    llf_obs = arma_innovations.arma_loglikeobs(
        arma_data, AR, MA, SIGMA2, prefix=prefix
    )
    assert llf_obs.dtype == dtype
    assert_allclose(llf_obs.real, expected, rtol=rtol)
    # Some platforms (Linux, wasm) leave imaginary rounding noise of about
    # 1e-16 relative to the real parts, so atol is set from their scale
    assert_allclose(llf_obs.imag, 0.0, atol=1e-12 * np.abs(llf_obs.real).max())


def test_innovations_nonfinite_params(arma_data):
    # Non-finite parameters give non-finite innovation variances, which are
    # rejected rather than returned
    with pytest.raises(ValueError, match="innovations algorithm cannot be used"):
        arma_innovations.arma_innovations(arma_data, AR, [np.nan], SIGMA2)


def _central_difference(func, params, step=1e-6):
    grad = []
    for i in range(params.shape[0]):
        delta = np.zeros_like(params)
        delta[i] = step
        grad.append((func(params + delta) - func(params - delta)) / (2 * step))
    return np.stack(grad, axis=-1)


def test_score_numerical_derivative(arma_data):
    # The parameters are far from the MLE, so that no element of the score
    # is close to zero
    def loglike(params):
        return arma_innovations.arma_loglike(
            arma_data, params[:2], params[2:3], params[3]
        )

    def loglikeobs(params):
        return arma_innovations.arma_loglikeobs(
            arma_data, params[:2], params[2:3], params[3]
        )

    score = arma_innovations.arma_score(arma_data, AR, MA, SIGMA2)
    assert score.shape == (4,)
    assert_allclose(score, _central_difference(loglike, PARAMS), rtol=1e-6)
    scoreobs = arma_innovations.arma_scoreobs(arma_data, AR, MA, SIGMA2)
    expected = _central_difference(loglikeobs, PARAMS)
    assert scoreobs.shape == expected.shape == (arma_data.shape[0], 4)
    # The scores of single observations cross zero, so atol is set from the
    # largest of them
    atol = 1e-6 * np.abs(expected).max()
    assert_allclose(scoreobs, expected, rtol=1e-6, atol=atol)
    # The score is the sum of the scores of the observations
    assert_allclose(scoreobs.sum(0), score, rtol=1e-10)


@pytest.mark.parametrize("form", ["list", "tuple", "series", "int"])
def test_endog_forms(arma_data, form):
    if form == "int":
        x = arma_data.astype(np.int64)
    else:
        x = series_forms(arma_data)[form]
    mod = SARIMAX(arma_data, order=(2, 0, 1))
    llf = arma_innovations.arma_loglike(x, AR, MA, SIGMA2)
    assert_allclose(llf, mod.loglike(PARAMS), rtol=1e-8)
    llf_obs = arma_innovations.arma_loglikeobs(x, AR, MA, SIGMA2)
    assert isinstance(llf_obs, np.ndarray)
    assert_allclose(llf_obs, mod.loglikeobs(PARAMS), rtol=1e-8)
    u, v = arma_innovations.arma_innovations(x, AR, MA, SIGMA2)
    assert isinstance(u, np.ndarray)
    assert u.shape == v.shape == arma_data.shape
    # The prediction errors cross zero, so atol is set from their scale
    expected = _kalman_filter(arma_data).forecasts_error[0]
    assert_allclose(u, expected, rtol=1e-8, atol=1e-9)
    score = arma_innovations.arma_score(x, AR, MA, SIGMA2)
    expected = arma_innovations.arma_score(arma_data, AR, MA, SIGMA2)
    assert_allclose(score, expected, rtol=1e-12)
    scoreobs = arma_innovations.arma_scoreobs(x, AR, MA, SIGMA2)
    expected = arma_innovations.arma_scoreobs(arma_data, AR, MA, SIGMA2)
    assert_allclose(scoreobs, expected, rtol=1e-12)


# The likelihood and score functions are left out: they raise ValueError for
# 2-d endog
@pytest.mark.parametrize("form", ["column", "dataframe"])
def test_innovations_column_forms(arma_data, form):
    x = series_forms(arma_data)[form]
    u, v = arma_innovations.arma_innovations(x, AR, MA, SIGMA2)
    # A 2-d endog keeps its shape
    assert u.shape == (arma_data.shape[0], 1)
    # The prediction errors cross zero, so atol is set from their scale
    expected = _kalman_filter(arma_data).forecasts_error[0]
    assert_allclose(u[:, 0], expected, rtol=1e-8, atol=1e-9)


# A Series of parameters is left out: with pandas copy-on-write its values
# are read-only, which the compiled routines reject
@pytest.mark.parametrize("form", [list, tuple, np.asarray])
def test_params_forms(arma_data, form):
    ar, ma = form(AR.tolist()), form(MA.tolist())
    mod = SARIMAX(arma_data, order=(2, 0, 1))
    llf = arma_innovations.arma_loglike(arma_data, ar, ma, SIGMA2)
    assert_allclose(llf, mod.loglike(PARAMS), rtol=1e-8)
    llf_obs = arma_innovations.arma_loglikeobs(arma_data, ar, ma, SIGMA2)
    assert_allclose(llf_obs, mod.loglikeobs(PARAMS), rtol=1e-8)
    u, _ = arma_innovations.arma_innovations(arma_data, ar, ma, SIGMA2)
    # The prediction errors cross zero, so atol is set from their scale
    expected = _kalman_filter(arma_data).forecasts_error[0]
    assert_allclose(u, expected, rtol=1e-8, atol=1e-9)
    score = arma_innovations.arma_score(arma_data, ar, ma, SIGMA2)
    expected = arma_innovations.arma_score(arma_data, AR, MA, SIGMA2)
    assert_allclose(score, expected, rtol=1e-12)
    scoreobs = arma_innovations.arma_scoreobs(arma_data, ar, ma, SIGMA2)
    expected = arma_innovations.arma_scoreobs(arma_data, AR, MA, SIGMA2)
    assert_allclose(scoreobs, expected, rtol=1e-12)


def test_scalar_params(arma_data):
    # A scalar is a single coefficient (arma_score and arma_scoreobs need a
    # sequence)
    params = np.array([0.5, 0.3, SIGMA2])
    mod = SARIMAX(arma_data, order=(1, 0, 1))
    llf = arma_innovations.arma_loglike(arma_data, 0.5, 0.3, SIGMA2)
    assert_allclose(llf, mod.loglike(params), rtol=1e-8)
    llf_obs = arma_innovations.arma_loglikeobs(arma_data, 0.5, 0.3, SIGMA2)
    assert_allclose(llf_obs, mod.loglikeobs(params), rtol=1e-8)
    u, _ = arma_innovations.arma_innovations(arma_data, 0.5, 0.3, SIGMA2)
    expected = _kalman_filter(arma_data, (1, 0, 1), params).forecasts_error[0]
    # The prediction errors cross zero, so atol is set from their scale
    assert_allclose(u, expected, rtol=1e-8, atol=1e-9)


def test_white_noise_defaults(arma_data):
    # Without parameters the process is white noise with variance sigma2:
    # the innovations are the data and innovations_mse is one
    u, v = arma_innovations.arma_innovations(arma_data, sigma2=SIGMA2)
    assert_allclose(u, arma_data, rtol=1e-14)
    assert_allclose(v, np.ones_like(arma_data), rtol=1e-14)
    expected = stats.norm.logpdf(arma_data, scale=np.sqrt(SIGMA2))
    llf_obs = arma_innovations.arma_loglikeobs(arma_data, sigma2=SIGMA2)
    assert_allclose(llf_obs, expected, rtol=1e-12)
    llf = arma_innovations.arma_loglike(arma_data, sigma2=SIGMA2)
    assert_allclose(llf, expected.sum(), rtol=1e-12)
    # d loglike / d sigma2 = sum(x ** 2 / sigma2 - 1) / (2 sigma2). The score
    # is a complex-step derivative: the imaginary rounding noise that some
    # platforms (Linux, wasm) leave, about 1e-16 per observation, is divided
    # by the step of about 1e-8, so it is accurate to about 1e-8 there.
    score = arma_innovations.arma_score(arma_data, sigma2=SIGMA2)
    expected = np.sum(arma_data**2 / SIGMA2 - 1) / (2 * SIGMA2)
    assert_allclose(score, [expected], rtol=1e-7)
