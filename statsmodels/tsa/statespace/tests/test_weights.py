"""
Tests for computation of weight functions in state space models

Author: Chad Fulton
License: Simplified-BSD
"""

import pytest

import numpy as np
import pandas as pd

from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_

from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS


dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range(start='1959Q1', end='2009Q3', freq='Q')


@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
@pytest.mark.parametrize('concentrate_scale', [False, True])
@pytest.mark.parametrize('measurement_error', [False, True])
def test_smoothed_state_obs_weights_sarimax(use_exog, trend,
                                            concentrate_scale,
                                            measurement_error):
    endog = np.array([[0.2, np.nan, 1.2, -0.3, -1.5]]).T
    exog = np.array([2, 5.3, -1, 3.4, 0.]) if use_exog else None

    trend_params = [0.1]
    ar_params = [0.5]
    exog_params = [1.4]
    meas_err_params = [1.2]
    cov_params = [0.8]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    if use_exog:
        params += exog_params
    params += ar_params
    if measurement_error:
        params += meas_err_params
    if not concentrate_scale:
        params += cov_params

    # Fit the models
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend=trend,
                          exog=exog if use_exog else None,
                          concentrate_scale=concentrate_scale,
                          measurement_error=measurement_error)
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
                tmp_mod = sarimax.SARIMAX(y, order=(1, 0, 0), trend=trend,
                                          exog=exog if use_exog else None,
                                          concentrate_scale=concentrate_scale,
                                          measurement_error=measurement_error)
                tmp_res = tmp_mod.smooth(params)

                desired[:, j, :, i] = (tmp_res.smoothed_state.T
                                       - res.smoothed_state.T)

    actual = tools.compute_obs_weights_smoothed_state(res)

    assert_allclose(actual[:, :, 0, 0], desired[:, :, 0, 0], atol=1e-12)


@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_smoothed_state_obs_weights_varmax(use_exog, trend):
    endog = np.zeros((5, 2))
    endog[0, 0] = np.nan
    endog[1, :] = np.nan
    endog[2, 1] = np.nan
    exog = np.array([2, 5.3, -1, 3.4, 0.]) if use_exog else None

    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1., 2.]
    cov_params = [1., 0., 1.]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params

    # Fit the model
    mod = varmax.VARMAX(endog, order=(1, 0), trend=trend,
                        exog=exog if use_exog else None)
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
                tmp_mod = varmax.VARMAX(y, order=(1, 0), trend=trend,
                                        exog=exog if use_exog else None)
                tmp_res = tmp_mod.smooth(params)

                desired[:, j, :, i] = (tmp_res.smoothed_state.T
                                       - res.smoothed_state.T)

    actual = tools.compute_obs_weights_smoothed_state(res)

    assert_allclose(actual, desired, atol=1e-12)


@pytest.mark.parametrize('diffuse', [0, 1, 4])
@pytest.mark.parametrize('univariate', [False, True])
def test_smoothed_state_obs_weights_TVSS(univariate, diffuse,
                                         reset_randomstate):
    endog = np.zeros((10, 3))
    # One simple way to introduce more diffuse periods is to have fully missing
    # observations at the beginning
    if diffuse == 4:
        endog[:3] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    if not diffuse:
        mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    if univariate:
        mod.ssm.filter_univariate = True
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
                    tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                if univariate:
                    tmp_mod.ssm.filter_univariate = True
                tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = (tmp_res.smoothed_state.T
                                       - res.smoothed_state.T)

    actual = tools.compute_obs_weights_smoothed_state(res)

    d = res.nobs_diffuse
    assert_equal(d, diffuse)
    if diffuse:
        assert_allclose(actual[:d], np.nan, atol=1e-12)
        assert_allclose(actual[:, :d], np.nan, atol=1e-12)
    assert_allclose(actual[d:, d:], desired[d:, d:], atol=1e-12)


@pytest.mark.parametrize('singular', ['both', 0, 1])
@pytest.mark.parametrize('periods', [1, 2])
def test_smoothed_state_obs_weights_univariate_singular(singular, periods,
                                                        reset_randomstate):
    # Tests for the univariate case when the forecast error covariance matrix
    # is singular (so the multivariate approach cannot be used, and the use of
    # pinv in computing the weights becomes actually operative)
    endog = np.zeros((10, 2))
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2) * 0)
    if singular == 'both':
        mod['obs_cov', ..., :periods] = 0
    else:
        mod['obs_cov', 0, 1, :periods] = 0
        mod['obs_cov', 1, 0, :periods] = 0
        mod['obs_cov', singular, singular, :periods] = 0
    mod['state_cov', :, :, :periods] = 0
    mod.ssm.filter_univariate = True
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
                tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = (tmp_res.smoothed_state.T
                                       - res.smoothed_state.T)

    actual = tools.compute_obs_weights_smoothed_state(res)

    assert_allclose(actual, desired, atol=1e-12)


def test_smoothed_state_obs_weights_collapsed(reset_randomstate):
    # Tests for the collapsed case
    endog = np.zeros((20, 6))
    endog[2, :] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod['obs_intercept'] = np.zeros((6, 1))
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
                tmp_mod['obs_intercept'] = np.zeros((6, 1))
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                mod.ssm.filter_collapsed = True
                tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = (tmp_res.smoothed_state.T
                                       - res.smoothed_state.T)

    actual = tools.compute_obs_weights_smoothed_state(res)

    assert_allclose(actual, desired, atol=1e-12)

@pytest.mark.parametrize('compute_j', [np.arange(10), [0, 1, 2], [5, 0, 9], 8])
@pytest.mark.parametrize('compute_t', [np.arange(10), [3, 2, 2], [0, 2, 5], 5])
def test_compute_t_compute_j(compute_j, compute_t, reset_randomstate):
    # Tests for the collapsed case
    endog = np.zeros((10, 6))
    endog[2, :] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod['obs_intercept'] = np.zeros((6, 1))
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
                tmp_mod['obs_intercept'] = np.zeros((6, 1))
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                mod.ssm.filter_collapsed = True
                tmp_res = tmp_mod.smooth([])

                desired[:, j, :, i] = (tmp_res.smoothed_state.T
                                       - res.smoothed_state.T)

    actual = tools.compute_obs_weights_smoothed_state(
        res, compute_t=compute_t, compute_j=compute_j)

    compute_t = np.atleast_1d(compute_t)
    compute_j = np.atleast_1d(compute_j)
    for t in np.arange(10):
        if t not in compute_t:
            desired[t, :] = np.nan
    for j in np.arange(10):
        if j not in compute_j:
            desired[:, j] = np.nan

    assert_allclose(actual, desired, atol=1e-12)


def test_resmooth(reset_randomstate):
    # Tests that resmooth works as it ought to, to reset the filter
    endog = [0.1, -0.3, -0.1, 0.5, 0.02]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    print(mod.param_names)
    res1 = mod.smooth([0.5, 2.0, 1.0])
    weights1_original = tools.compute_obs_weights_smoothed_state(
        res1, resmooth=False)
    res2 = mod.smooth([0.2, 1.0, 1.2])
    weights2_original = tools.compute_obs_weights_smoothed_state(
        res2, resmooth=False)

    weights1_no_resmooth = tools.compute_obs_weights_smoothed_state(
        res1, resmooth=False)
    weights1_resmooth = tools.compute_obs_weights_smoothed_state(
        res1, resmooth=True)

    weights2_no_resmooth = tools.compute_obs_weights_smoothed_state(
        res2, resmooth=False)
    weights2_resmooth = tools.compute_obs_weights_smoothed_state(
        res2, resmooth=True)

    weights1_default = tools.compute_obs_weights_smoothed_state(res1)
    weights2_default = tools.compute_obs_weights_smoothed_state(res2)

    assert_allclose(weights1_no_resmooth, weights2_original)
    assert_allclose(weights1_resmooth, weights1_original)
    assert_allclose(weights1_default, weights1_original)

    assert_allclose(weights2_no_resmooth, weights1_original)
    assert_allclose(weights2_resmooth, weights2_original)
    assert_allclose(weights2_default, weights2_original)
