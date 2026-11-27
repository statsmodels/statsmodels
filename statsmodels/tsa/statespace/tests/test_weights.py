"""
Tests for computation of weight functions in state space models

Author: Chad Fulton
License: Simplified-BSD
"""

from statsmodels.compat.platform import PLATFORM_WIN

import contextlib
import copy
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tsa.statespace import sarimax, tools, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS

dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range(start="1959Q1", end="2009Q3", freq="Q")


def expect_diagonalized(expected=True):
    # The univariate filter diagonalizes the non-diagonal obs_cov of TVSS
    if expected:
        return pytest.warns(
            OutputWarning, match="The univariate filtering approach diagonalized"
        )
    return contextlib.nullcontext()


@pytest.mark.parametrize("use_exog", [False, True])
@pytest.mark.parametrize("trend", ["n", "c", "t"])
@pytest.mark.parametrize("concentrate_scale", [False, True])
@pytest.mark.parametrize("measurement_error", [False, True])
def test_smoothed_state_obs_weights_sarimax(
    use_exog, trend, concentrate_scale, measurement_error
):
    endog = np.array([[0.2, np.nan, 1.2, -0.3, -1.5]]).T
    exog = np.array([2, 5.3, -1, 3.4, 0.0]) if use_exog else None

    trend_params = [0.1]
    ar_params = [0.5]
    exog_params = [1.4]
    meas_err_params = [1.2]
    cov_params = [0.8]

    params = []
    if trend in ["c", "t"]:
        params += trend_params
    if use_exog:
        params += exog_params
    params += ar_params
    if measurement_error:
        params += meas_err_params
    if not concentrate_scale:
        params += cov_params

    # Fit the models
    mod = sarimax.SARIMAX(
        endog,
        order=(1, 0, 0),
        trend=trend,
        exog=exog if use_exog else None,
        concentrate_scale=concentrate_scale,
        measurement_error=measurement_error,
    )
    prior_mean = np.array([-0.4])
    prior_cov = np.eye(1) * 1.2
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)

    # Compute the desiried weights
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog

    desired = np.zeros((n, n, m, p)) * np.nan
    # Here we manually compute the weights by adjusting one observation at a
    # time
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] += 1.0
                tmp_mod = sarimax.SARIMAX(
                    y,
                    order=(1, 0, 0),
                    trend=trend,
                    exog=exog if use_exog else None,
                    concentrate_scale=concentrate_scale,
                    measurement_error=measurement_error,
                )
                tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
                tmp_res = tmp_mod.smooth(params)

                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    # Here we manually compute the weights by adjusting one state intercept
    # at a time
    for j in range(n):
        for ell in range(m):
            tmp_mod = sarimax.SARIMAX(
                endog,
                order=(1, 0, 0),
                trend=trend,
                exog=exog if use_exog else None,
                concentrate_scale=concentrate_scale,
                measurement_error=measurement_error,
            )
            tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
            tmp_mod.update(params)
            if tmp_mod["state_intercept"].ndim == 1:
                si = tmp_mod["state_intercept"]
                tmp_mod["state_intercept"] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod["state_intercept", :, :] = si
            tmp_mod["state_intercept", ell, j] += 1.0
            tmp_res = tmp_mod.ssm.smooth()

            desired_state_intercept_weights[:, j, :, ell] = (
                tmp_res.smoothed_state.T - res.smoothed_state.T
            )

    desired_prior_weights = np.zeros((n, m, m)) * np.nan
    # Here we manually compute the weights by adjusting one prior element at
    # a time
    for i in range(m):
        a = prior_mean.copy()
        a[i] += 1
        tmp_mod = sarimax.SARIMAX(
            endog,
            order=(1, 0, 0),
            trend=trend,
            exog=exog if use_exog else None,
            concentrate_scale=concentrate_scale,
            measurement_error=measurement_error,
        )
        tmp_mod.ssm.initialize_known(a, prior_cov)
        tmp_res = tmp_mod.smooth(params)

        desired_prior_weights[:, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    mod.ssm.initialize_known(prior_mean, prior_cov)
    actual, actual_state_intercept_weights, actual_prior_weights = (
        tools.compute_smoothed_state_weights(res)
    )

    assert_allclose(actual, desired, atol=1e-8)
    assert_allclose(
        actual_state_intercept_weights, desired_state_intercept_weights, atol=1e-12
    )
    assert_allclose(actual_prior_weights, desired_prior_weights, atol=1e-12)


@pytest.mark.parametrize("use_exog", [False, True])
@pytest.mark.parametrize("trend", ["n", "c", "t"])
def test_smoothed_state_obs_weights_varmax(use_exog, trend):
    endog = np.zeros((5, 2))
    endog[0, 0] = np.nan
    endog[1, :] = np.nan
    endog[2, 1] = np.nan
    exog = np.array([2, 5.3, -1, 3.4, 0.0]) if use_exog else None

    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1.0, 2.0]
    cov_params = [1.0, 0.0, 1.0]

    params = []
    if trend in ["c", "t"]:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params

    # Fit the model
    mod = varmax.VARMAX(
        endog, order=(1, 0), trend=trend, exog=exog if use_exog else None
    )
    prior_mean = np.array([-0.4, 0.9])
    prior_cov = np.array([[1.4, 0.3], [0.3, 2.6]])
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)

    # Compute the desiried weights
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog

    desired = np.zeros((n, n, m, p)) * np.nan
    # Here we manually compute the weights by adjusting one observation at a
    # time
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = varmax.VARMAX(
                    y, order=(1, 0), trend=trend, exog=exog if use_exog else None
                )
                tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
                tmp_res = tmp_mod.smooth(params)

                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    # Here we manually compute the weights by adjusting one state intercept
    # at a time
    for j in range(n):
        for ell in range(m):
            tmp_mod = varmax.VARMAX(
                endog, order=(1, 0), trend=trend, exog=exog if use_exog else None
            )
            tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
            tmp_mod.update(params)
            if tmp_mod["state_intercept"].ndim == 1:
                si = tmp_mod["state_intercept"]
                tmp_mod["state_intercept"] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod["state_intercept", :, :] = si[:, None]
            tmp_mod["state_intercept", ell, j] += 1.0
            tmp_res = tmp_mod.ssm.smooth()

            desired_state_intercept_weights[:, j, :, ell] = (
                tmp_res.smoothed_state.T - res.smoothed_state.T
            )

    desired_prior_weights = np.zeros((n, m, m)) * np.nan
    for i in range(m):
        a = prior_mean.copy()
        a[i] += 1
        tmp_mod = varmax.VARMAX(
            endog, order=(1, 0), trend=trend, exog=exog if use_exog else None
        )
        tmp_mod.ssm.initialize_known(a, prior_cov)
        tmp_res = tmp_mod.smooth(params)

        desired_prior_weights[:, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    mod.ssm.initialize_known(prior_mean, prior_cov)
    actual, actual_state_intercept_weights, actual_prior_weights = (
        tools.compute_smoothed_state_weights(res)
    )

    assert_allclose(actual, desired, atol=1e-8)
    assert_allclose(
        actual_state_intercept_weights,
        desired_state_intercept_weights,
        atol=1e-8,
        rtol=1e-6,
    )
    assert_allclose(actual_prior_weights, desired_prior_weights, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("diffuse", [0, 1, 4])
@pytest.mark.parametrize("univariate", [False, True])
def test_smoothed_state_obs_weights_TVSS(univariate, diffuse):
    endog = np.zeros((10, 3))
    # One simple way to introduce more diffuse periods is to have fully missing
    # observations at the beginning
    if diffuse == 4:
        endog[:3] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)

    prior_mean = np.array([1.2, 0.8])
    prior_cov = np.eye(2)
    if not diffuse:
        mod.ssm.initialize_known(prior_mean, prior_cov)
    if univariate:
        mod.ssm.filter_univariate = True
    # The exact diffuse initialization always uses the univariate filter
    uses_univariate = univariate or diffuse > 0
    with expect_diagonalized(uses_univariate):
        res = mod.smooth([])

    # Compute the desiried weights
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog

    desired = np.zeros((n, n, m, p)) * np.nan
    # Here we manually compute the weights by adjusting one observation at a
    # time
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = mod.clone(y)
                if not diffuse:
                    tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
                if univariate:
                    tmp_mod.ssm.filter_univariate = True
                with expect_diagonalized(uses_univariate):
                    tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    # Here we manually compute the weights by adjusting one state intercept
    # at a time
    for j in range(n):
        for ell in range(m):
            tmp_mod = mod.clone(endog)
            if not diffuse:
                tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
            if univariate:
                tmp_mod.ssm.filter_univariate = True
            if tmp_mod["state_intercept"].ndim == 1:
                si = tmp_mod["state_intercept"]
                tmp_mod["state_intercept"] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod["state_intercept", :, :] = si[:, None]
            tmp_mod["state_intercept", ell, j] += 1.0
            with expect_diagonalized(uses_univariate):
                tmp_res = tmp_mod.ssm.smooth()

            desired_state_intercept_weights[:, j, :, ell] = (
                tmp_res.smoothed_state.T - res.smoothed_state.T
            )

    desired_prior_weights = np.zeros((n, m, m)) * np.nan
    if not diffuse:
        for i in range(m):
            a = prior_mean.copy()
            a[i] += 1
            tmp_mod = mod.clone(endog)
            tmp_mod.ssm.initialize_known(a, prior_cov)
            tmp_res = tmp_mod.smooth([])

            desired_prior_weights[:, :, i] = (
                tmp_res.smoothed_state.T - res.smoothed_state.T
            )

    if not diffuse:
        mod.ssm.initialize_known(prior_mean, prior_cov)
    actual, actual_state_intercept_weights, actual_prior_weights = (
        tools.compute_smoothed_state_weights(res)
    )

    d = res.nobs_diffuse
    _diffuse = (diffuse,)
    if PLATFORM_WIN:
        # Occasional failures on some windows when diffuse is 4, get 3
        _diffuse += (diffuse - 1,) if diffuse else ()
    assert d in _diffuse
    if diffuse:
        assert_allclose(actual[:d], np.nan, atol=1e-12)
        assert_allclose(actual[:, :d], np.nan, atol=1e-12)
        assert_allclose(actual_state_intercept_weights[:d], np.nan)
        assert_allclose(actual_state_intercept_weights[:, :d], np.nan)
        assert_allclose(actual_prior_weights, np.nan)
    else:
        # Test that the weights are the same
        assert_allclose(
            actual_prior_weights, desired_prior_weights, atol=1e-8, rtol=1e-6
        )

        # In the non-diffuse case, we can actually use the weights along with
        # the prior and observations to compute the smoothed state directly,
        # and then compare that to what was returned by the usual Kalman
        # smoothing routines
        # Note that TVSS sets the state intercept to zeros, so this does not
        # test that, although those weights are tested separately, see above
        # and below.
        contribution_prior = np.nansum(
            actual_prior_weights * prior_mean[None, None, :], axis=2
        )
        contribution_endog = np.nansum(
            actual * (endog - mod["obs_intercept"].T)[None, :, None, :], axis=(1, 3)
        )
        computed_smoothed_state = contribution_prior + contribution_endog
        assert_allclose(computed_smoothed_state, res.smoothed_state.T)
    assert_allclose(actual[d:, d:], desired[d:, d:], atol=1e-12)
    assert_allclose(
        actual_state_intercept_weights[d:, d:],
        desired_state_intercept_weights[d:, d:],
        atol=1e-12,
    )


@pytest.mark.parametrize("singular", ["both", 0, 1])
@pytest.mark.parametrize("periods", [1, 2])
def test_smoothed_state_obs_weights_univariate_singular(singular, periods):
    # Tests for the univariate case when the forecast error covariance matrix
    # is singular (so the multivariate approach cannot be used, and the use of
    # pinv in computing the weights becomes actually operative)
    endog = np.zeros((10, 2))
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2) * 0)
    if singular == "both":
        mod["obs_cov", ..., :periods] = 0
    else:
        mod["obs_cov", 0, 1, :periods] = 0
        mod["obs_cov", 1, 0, :periods] = 0
        mod["obs_cov", singular, singular, :periods] = 0
    mod["state_cov", :, :, :periods] = 0
    mod.ssm.filter_univariate = True
    with expect_diagonalized():
        res = mod.smooth([])

    # Make sure we actually have singular covariance matrices in the periods
    # specified
    for i in range(periods):
        eigvals = np.linalg.eigvalsh(res.forecasts_error_cov[..., i])
        assert_equal(np.min(eigvals), 0)

    # Compute the desiried weights
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog

    desired = np.zeros((n, n, m, p)) * np.nan
    # Here we manually compute the weights by adjusting one observation at a
    # time
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = mod.clone(y)
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2) * 0)
                tmp_mod.ssm.filter_univariate = True
                with expect_diagonalized():
                    tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    actual, _, _ = tools.compute_smoothed_state_weights(res)

    assert_allclose(actual, desired, atol=1e-12)


def test_smoothed_state_obs_weights_collapsed():
    # Tests for the collapsed case
    endog = np.zeros((20, 6))
    endog[2, :] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod["obs_intercept"] = np.zeros((6, 1))
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    mod.ssm.filter_collapsed = True
    res = mod.smooth([])

    # Compute the desiried weights
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog

    desired = np.zeros((n, n, m, p)) * np.nan
    # Here we manually compute the weights by adjusting one observation at a
    # time
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = mod.clone(y)
                tmp_mod["obs_intercept"] = np.zeros((6, 1))
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                mod.ssm.filter_collapsed = True
                tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    # Here we manually compute the weights by adjusting one state intercept
    # at a time
    for j in range(n):
        for ell in range(m):
            tmp_mod = mod.clone(endog)
            tmp_mod["obs_intercept"] = np.zeros((6, 1))
            tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
            mod.ssm.filter_collapsed = True

            if tmp_mod["state_intercept"].ndim == 1:
                si = tmp_mod["state_intercept"]
                tmp_mod["state_intercept"] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod["state_intercept", :, :] = si[:, None]
            tmp_mod["state_intercept", ell, j] += 1.0
            tmp_res = tmp_mod.ssm.smooth()

            desired_state_intercept_weights[:, j, :, ell] = (
                tmp_res.smoothed_state.T - res.smoothed_state.T
            )

    actual, actual_state_intercept_weights, _ = tools.compute_smoothed_state_weights(
        res
    )

    assert_allclose(actual, desired, atol=1e-8)
    assert_allclose(
        actual_state_intercept_weights, desired_state_intercept_weights, atol=1e-10
    )


@pytest.mark.parametrize("compute_j", [np.arange(10), [0, 1, 2], [5, 0, 9], 8])
@pytest.mark.parametrize("compute_t", [np.arange(10), [3, 2, 2], [0, 2, 5], 5])
def test_compute_t_compute_j(compute_j, compute_t):
    # Tests for the collapsed case
    endog = np.zeros((10, 6))
    endog[2, :] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod["obs_intercept"] = np.zeros((6, 1))
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    mod.ssm.filter_collapsed = True
    res = mod.smooth([])

    # Compute the desiried weights
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog

    desired = np.zeros((n, n, m, p)) * np.nan
    # Here we manually compute the weights by adjusting one observation at a
    # time
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = mod.clone(y)
                tmp_mod["obs_intercept"] = np.zeros((6, 1))
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                mod.ssm.filter_collapsed = True
                tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T

    actual, _, _ = tools.compute_smoothed_state_weights(
        res, compute_t=compute_t, compute_j=compute_j
    )

    compute_t = np.unique(np.atleast_1d(compute_t))
    compute_t.sort()
    compute_j = np.unique(np.atleast_1d(compute_j))
    compute_j.sort()
    for t in np.arange(10):
        if t not in compute_t:
            desired[t, :] = np.nan
    for j in np.arange(10):
        if j not in compute_j:
            desired[:, j] = np.nan

    # Subset to the actual required compute_t and compute_j
    ix = np.ix_(compute_t, compute_j)
    desired = desired[ix]

    assert_allclose(actual, desired, atol=2e-5)


def test_resmooth():
    # Tests that resmooth works as it ought to, to reset the filter
    endog = [0.1, -0.3, -0.1, 0.5, 0.02]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), measurement_error=True)

    res1 = mod.smooth([0.5, 2.0, 1.0])
    weights1_original, _, _ = tools.compute_smoothed_state_weights(res1, resmooth=False)
    res2 = mod.smooth([0.2, 1.0, 1.2])
    weights2_original, _, _ = tools.compute_smoothed_state_weights(res2, resmooth=False)

    weights1_no_resmooth, _, _ = tools.compute_smoothed_state_weights(
        res1, resmooth=False
    )
    weights1_resmooth, _, _ = tools.compute_smoothed_state_weights(res1, resmooth=True)

    weights2_no_resmooth, _, _ = tools.compute_smoothed_state_weights(
        res2, resmooth=False
    )
    weights2_resmooth, _, _ = tools.compute_smoothed_state_weights(res2, resmooth=True)

    weights1_default, _, _ = tools.compute_smoothed_state_weights(res1)
    weights2_default, _, _ = tools.compute_smoothed_state_weights(res2)

    assert_allclose(weights1_no_resmooth, weights2_original)
    assert_allclose(weights1_resmooth, weights1_original)
    assert_allclose(weights1_default, weights1_original)

    assert_allclose(weights2_no_resmooth, weights1_original)
    assert_allclose(weights2_resmooth, weights2_original)
    assert_allclose(weights2_default, weights2_original)


@pytest.mark.parametrize("copy_method", ["pickle", "deepcopy"])
@pytest.mark.parametrize("resmooth", [None, False])
def test_compute_smoothed_state_weights_copied_results(copy_method, resmooth):
    # The compiled Kalman smoother pickled its smoothed state autocovariance
    # in place of the innovations transition matrices L_t, so the weights of
    # copied (pickled or deep-copied) results, which by default reuse the
    # compiled smoother when its smoothed state is unchanged, were wrong. They
    # equal those of the original results and reproduce the smoothed state.
    endog = np.cumsum(np.random.default_rng(6042).standard_normal(20)) + 10
    mod = sarimax.SARIMAX(endog, order=(1, 0, 1), trend="c")
    res = mod.smooth([1.0, 0.5, 0.2, 1.0])
    if copy_method == "pickle":
        copied = pickle.loads(pickle.dumps(res))
    else:
        copied = copy.deepcopy(res)

    actual = tools.compute_smoothed_state_weights(copied, resmooth=resmooth)
    desired = tools.compute_smoothed_state_weights(res, resmooth=True)
    for actual_weights, desired_weights in zip(actual, desired, strict=True):
        assert_allclose(actual_weights, desired_weights, rtol=1e-10)

    # The smoothed state is the weighted sum of the observations, the state
    # intercepts and the initial state mean
    weights, state_intercept_weights, prior_weights = actual
    # (time-invariant: the same intercept in every period)
    state_intercept = res.filter_results.state_intercept[:, 0]
    smoothed_state = (
        np.einsum("tjmp,jp->tm", weights, endog[:, None])
        + np.einsum("tjml,l->tm", state_intercept_weights, state_intercept)
        + prior_weights @ res.filter_results.initial_state
    )
    assert_allclose(smoothed_state, res.smoothed_state.T, rtol=1e-10)


@pytest.mark.parametrize("resmooth", [None, True])
def test_weights_filter_results(resmooth):
    # The results of filter() can be used, also if the model was never
    # smoothed (it raised AttributeError): the weights then reproduce the
    # smoothed state from the observations and the prior mean
    endog = np.array([0.1, -0.3, np.nan, 0.5, 0.02, -0.4, 0.3])
    params = [0.5, 0.4, 1.2]
    kwargs = {
        "order": (1, 0, 0),
        "measurement_error": True,
        "initialization": "known",
        "initial_state": [0.3],
        "initial_state_cov": [[1.5]],
    }
    res = sarimax.SARIMAX(endog, **kwargs).filter(params)
    assert res.smoothed_state is None
    weights, _, prior_weights = tools.compute_smoothed_state_weights(
        res, resmooth=resmooth
    )

    smoothed = sarimax.SARIMAX(endog, **kwargs).smooth(params)
    # weights: t, j, m, p; missing observations have NaN weights
    actual = np.nansum(weights[..., 0, 0] * endog[None, :], axis=1)
    actual += prior_weights[:, 0, 0] * 0.3
    assert_allclose(actual, smoothed.smoothed_state[0], rtol=1e-12)
