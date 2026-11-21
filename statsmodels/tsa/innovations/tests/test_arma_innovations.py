"""
Tests for ARMA innovations algorithm wrapper
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX


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


FUNCTIONS = [
    "arma_innovations",
    "arma_loglike",
    "arma_loglikeobs",
    "arma_score",
    "arma_scoreobs",
]
# The Kalman filter matches the innovations algorithm to about 1e-9 here. The
# scores are complex-step derivatives of the two likelihoods, which agree to
# about 1e-7 for each observation.
RTOL = {
    "arma_innovations": 1e-6,
    "arma_loglike": 1e-7,
    "arma_loglikeobs": 1e-7,
    "arma_score": 1e-6,
    "arma_scoreobs": 1e-5,
}


def _kalman_filter_reference(func, endog, ar_params, ma_params, sigma2):
    # The Kalman filter of the same ARMA model, with the stationary
    # initialization, gives the exact likelihood. Its one-step prediction
    # errors are the innovations, and its prediction error variances are
    # sigma2 times the mean square errors of the innovations.
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    params = np.r_[ar_params, ma_params, sigma2]
    if func == "arma_innovations":
        res = mod.filter(params)
        return res.forecasts_error[0], res.forecasts_error_cov[0, 0] / sigma2
    method = {
        "arma_loglike": "loglike",
        "arma_loglikeobs": "loglikeobs",
        "arma_score": "score",
        "arma_scoreobs": "score_obs",
    }[func]
    return getattr(mod, method)(params)


def _read_only(values):
    arr = np.array(values, dtype=float)
    arr.setflags(write=False)
    return arr


@pytest.mark.parametrize("func", FUNCTIONS)
@pytest.mark.parametrize(
    "form", [_read_only, pd.Series], ids=["read-only", "Series"]
)
def test_read_only_inputs(func, form):
    # The Cython routines need writable buffers, so read-only parameter arrays
    # raised "buffer source array is read-only". The array of a pandas Series
    # is read-only under copy-on-write (pandas 3).
    rng = np.random.default_rng(20261121)
    endog = rng.standard_normal(60)
    ar_params, ma_params, sigma2 = [0.5], [0.3], 1.7
    res = getattr(arma_innovations, func)(
        form(endog), form(ar_params), form(ma_params), sigma2=sigma2
    )
    expected = _kalman_filter_reference(func, endog, ar_params, ma_params, sigma2)
    assert_allclose(res, expected, rtol=RTOL[func])


# arma_innovations filters each column of a 2-d endog, the others need one
# series
@pytest.mark.parametrize("func", FUNCTIONS[1:])
@pytest.mark.parametrize(
    "form", [lambda y: y[:, None], pd.DataFrame], ids=["column", "DataFrame"]
)
def test_single_column_endog(func, form):
    # A single column raised "Buffer has wrong number of dimensions"
    rng = np.random.default_rng(20261121)
    endog = rng.standard_normal(60)
    ar_params, ma_params, sigma2 = [0.5], [0.3], 1.7
    res = getattr(arma_innovations, func)(
        form(endog), ar_params, ma_params, sigma2=sigma2
    )
    expected = _kalman_filter_reference(func, endog, ar_params, ma_params, sigma2)
    assert_allclose(res, expected, rtol=RTOL[func])


@pytest.mark.parametrize("func", FUNCTIONS[1:])
def test_multiple_column_endog_raises(func):
    endog = np.zeros((10, 2))
    with pytest.raises(ValueError, match="endog is required to have ndim 1"):
        getattr(arma_innovations, func)(endog, [0.5], [0.3])


def test_single_column_endog_keeps_dtype():
    # endog is converted without changing its dtype, so that float32 data and
    # parameters are still evaluated in single precision
    rng = np.random.default_rng(20261121)
    endog = rng.standard_normal(60).astype(np.float32)
    res = arma_innovations.arma_loglikeobs(
        endog[:, None], np.float32([0.5]), np.float32([0.3]), np.float32(1.7)
    )
    assert res.dtype == np.float32
    expected = _kalman_filter_reference(
        "arma_loglikeobs", endog.astype(float), [0.5], [0.3], 1.7
    )
    # Single precision
    assert_allclose(res, expected, rtol=1e-4)


@pytest.mark.parametrize("func", FUNCTIONS)
def test_scalar_params(func):
    # The score functions called len() on the parameters, so scalars raised a
    # TypeError, although the other functions accept them
    rng = np.random.default_rng(20261121)
    endog = rng.standard_normal(60)
    res = getattr(arma_innovations, func)(endog, 0.5, 0.3, sigma2=1.7)
    expected = _kalman_filter_reference(func, endog, [0.5], [0.3], 1.7)
    assert_allclose(res, expected, rtol=RTOL[func])


# Single precision (eps 1.2e-7), with rounding errors that accumulate over the
# recursions
RTOL_SINGLE = 1e-4


@pytest.mark.parametrize(
    "prefix, dtype, rtol",
    [
        ("s", np.float32, RTOL_SINGLE),
        ("d", np.float64, RTOL["arma_innovations"]),
        ("c", np.complex64, RTOL_SINGLE),
        ("z", np.complex128, RTOL["arma_innovations"]),
    ],
    ids=["s", "d", "c", "z"],
)
def test_innovations_prefix(prefix, dtype, rtol):
    # The lag polynomials were built before the parameters were converted to
    # the dtype of the prefix, so a prefix other than the one found from the
    # input raised "Buffer dtype mismatch"
    rng = np.random.default_rng(20261121)
    endog = rng.standard_normal(60)
    u, v = arma_innovations.arma_innovations(
        endog, [0.5], [0.3], sigma2=1.7, prefix=prefix
    )
    assert u.dtype == dtype
    assert v.dtype == dtype
    expected = _kalman_filter_reference(
        "arma_innovations", endog, [0.5], [0.3], 1.7
    )
    assert_allclose((u, v), expected, rtol=rtol)


@pytest.mark.parametrize(
    "sigma2, dtype, rtol",
    [
        (1, np.float64, RTOL["arma_innovations"]),
        (np.float32(1), np.float32, RTOL_SINGLE),
    ],
    ids=["int-sigma2", "float32-sigma2"],
)
def test_innovations_single_precision_input(sigma2, dtype, rtol):
    # float32 parameters gave float32 lag polynomials, which raised "Buffer
    # dtype mismatch" when the prefix found from all of the input was another
    # one. A Python int sigma2 gives double precision.
    rng = np.random.default_rng(20261121)
    endog = rng.standard_normal(60).astype(np.float32)
    u, v = arma_innovations.arma_innovations(
        endog, np.float32([0.5]), np.float32([0.3]), sigma2=sigma2
    )
    assert u.dtype == dtype
    assert v.dtype == dtype
    expected = _kalman_filter_reference(
        "arma_innovations", endog.astype(float), [0.5], [0.3], 1.0
    )
    assert_allclose((u, v), expected, rtol=rtol)
