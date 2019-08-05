"""
Tests for memory conservation in state space models

Author: Chad Fulton
License: BSD-3
"""


import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_equal, assert_allclose, assert_

from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
    sarimax, varmax, dynamic_factor)
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_NO_GAIN, MEMORY_CONSERVE)

dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')


@pytest.mark.parametrize("concentrate", [True, False])
@pytest.mark.parametrize("univariate", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
@pytest.mark.parametrize("timing_init_filtered", [True, False])
def test_memory_no_likelihood(concentrate, univariate, diffuse,
                              timing_init_filtered):
    # Basic test that covers a variety of special filtering cases with a
    # simple univariate model
    endog = dta['infl'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0),
                          concentrate_scale=concentrate)
    if timing_init_filtered:
        mod.timing_init_filtered = True
    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True

    params = [0.85]
    if not concentrate:
        params.append(7.)
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)

    # Check that we really did conserve memory in the second case
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)

    # Check that the loglikelihood computations are identical
    assert_allclose(res1.llf, res2.llf)


@pytest.mark.parametrize("concentrate", [True, False])
@pytest.mark.parametrize("univariate", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
@pytest.mark.parametrize("timing_init_filtered", [True, False])
def test_memory_no_likelihood_extras(concentrate, univariate, diffuse,
                                     timing_init_filtered):
    # Test that adds extra features (missing data, exog variables) to the
    # variety of special filtering cases in a univariate model
    endog = dta['infl'].iloc[:20].copy()
    endog[0] = np.nan
    endog[4:6] = np.nan
    exog = dta['realint'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), exog=exog,
                          concentrate_scale=concentrate)
    if timing_init_filtered:
        mod.timing_init_filtered = True
    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True

    params = [1.2, 0.85]
    if not concentrate:
        params.append(7.)
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)

    # Check that we really did conserve memory in the second case
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)

    # Check that the loglikelihood computations are identical
    assert_allclose(res1.llf, res2.llf)


@pytest.mark.parametrize("univariate", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
def test_memory_no_likelihood_multivariate(univariate, diffuse):
    # Test with multivariate data, and also missing values, exog
    endog = dta[['infl', 'realint']].iloc[:20].copy()
    endog.iloc[0, 0] = np.nan
    endog.iloc[4:6, :] = np.nan
    exog = np.log(dta['realgdp'].iloc[:20])
    mod = varmax.VARMAX(endog, order=(1, 0), exog=exog, trend='c')

    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True

    params = [1.4, 1.3, 0.1, 0.01, 0.02, 0.3, -0.001, 0.001, 1.0, -0.1, 0.6]
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)

    # Check that we really did conserve memory in the second case
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)

    # Check that the loglikelihood computations are identical
    assert_allclose(res1.llf, res2.llf)


@pytest.mark.parametrize("univariate", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
@pytest.mark.parametrize("collapsed", [True, False])
def test_memory_no_likelihood_multivariate_extra(univariate, diffuse,
                                                 collapsed):
    # Test with multivariate data, missing values, and collapsed approach
    endog = dta[['infl', 'realint']].iloc[:20].copy()
    endog.iloc[0, 0] = np.nan
    endog.iloc[4:6, :] = np.nan
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)

    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True
    if collapsed:
        mod.ssm.filter_collapsed = True

    params = [4, -4.5, 0.8, 0.9, -0.5]
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)

    # Check that we really did conserve memory in the second case
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)

    # Check that the loglikelihood computations are identical
    assert_allclose(res1.llf, res2.llf)


def test_fit():
    # Test that fitting works regardless of the level of memory conservation
    # used
    endog = dta['infl'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)

    res = mod.fit(disp=False)

    options_smooth = [
        'memory_no_forecast',  'memory_no_filtered', 'memory_no_likelihood',
        'memory_no_std_forecast']
    for option in options_smooth:
        mod.ssm.set_conserve_memory(0)
        setattr(mod.ssm, option, True)
        res2 = mod.fit(res.params, disp=False)

        # General check that smoothing results are available
        assert_allclose(res2.smoothed_state, res.smoothed_state, atol=1e-10)

        # Specific checks for each type
        if option == 'memory_no_forecast':
            assert_(res2.forecasts is None)
            assert_(res2.forecasts_error is None)
            assert_(res2.forecasts_error_cov is None)
        else:
            assert_allclose(res2.forecasts, res.forecasts)
            assert_allclose(res2.forecasts_error, res.forecasts_error)
            assert_allclose(res2.forecasts_error_cov, res.forecasts_error_cov)

        if option == 'memory_no_filtered':
            assert_(res2.filtered_state is None)
            assert_(res2.filtered_state_cov is None)
        else:
            assert_allclose(res2.filtered_state, res.filtered_state)
            assert_allclose(res2.filtered_state_cov, res.filtered_state_cov)

        assert_allclose(res2.llf, res.llf)
        if option == 'memory_no_likelihood':
            assert_(res2.llf_obs is None)
        else:
            assert_allclose(res2.llf_obs, res.llf_obs)

        if option == 'memory_no_std_forecast':
            assert_(res2.standardized_forecasts_error is None)
        else:
            assert_allclose(res2.standardized_forecasts_error,
                            res.standardized_forecasts_error)

    options_filter_only = [
        'memory_no_predicted', 'memory_no_gain', 'memory_no_smoothing',
        'memory_conserve']
    for option in options_filter_only[2:]:
        mod.ssm.set_conserve_memory(0)
        setattr(mod.ssm, option, True)
        res2 = mod.fit(res.params, disp=False)

        # General check that smoothing results are not available
        assert_(res2.smoothed_state is None)

        # Specific checks for each type
        if option in ['memory_no_predicted', 'memory_conserve']:
            assert_(res2.predicted_state is None)
            assert_(res2.predicted_state_cov is None)
        else:
            assert_allclose(res2.predicted_state, res.predicted_state)
            assert_allclose(res2.predicted_state_cov, res.predicted_state_cov)

        if option in ['memory_no_gain', 'memory_conserve']:
            assert_(res2.filter_results._kalman_gain is None)
        else:
            assert_allclose(res2.filter_results.kalman_gain,
                            res.filter_results.kalman_gain)


def test_low_memory_filter():
    endog = dta['infl'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod.ssm.set_conserve_memory(MEMORY_NO_GAIN)

    res = mod.filter([0.5], low_memory=True)
    assert_equal(res.filter_results.conserve_memory, MEMORY_CONSERVE)
    assert_(res.llf_obs is None)
    assert_equal(mod.ssm.conserve_memory, MEMORY_NO_GAIN)


def test_low_memory_fit():
    endog = dta['infl'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod.ssm.set_conserve_memory(MEMORY_NO_GAIN)

    res = mod.fit(low_memory=True, disp=False)
    assert_equal(res.filter_results.conserve_memory, MEMORY_CONSERVE)
    assert_(res.llf_obs is None)
    assert_equal(mod.ssm.conserve_memory, MEMORY_NO_GAIN)
