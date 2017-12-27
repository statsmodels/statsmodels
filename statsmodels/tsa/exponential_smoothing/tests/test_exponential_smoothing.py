"""
Tests for exponential smoothing models

Author: Chad Fulton
License: BSD-3
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import warnings
from statsmodels.tsa.api import ExponentialSmoothing
from numpy.testing import (assert_equal, assert_almost_equal, assert_raises,
                           assert_allclose)

current_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(current_path, 'results')
fitted_path = os.path.join(results_path, 'fitted.csv')
results_fitted = pd.read_csv(fitted_path)

# R, fpp: oildata <- window(oil,start=1996,end=2007)
oildata = pd.Series([
    446.6565229, 454.4733065, 455.6629740, 423.6322388, 456.2713279,
    440.5880501, 425.3325201, 485.1494479, 506.0481621, 526.7919833,
    514.2688890, 494.2110193],
    index=pd.PeriodIndex(start='1996', end='2007', freq='A'))

# R, fpp: air <- window(ausair,start=1990,end=2004)
air = pd.Series([
    17.553400, 21.860100, 23.886600, 26.929300, 26.888500,
    28.831400, 30.075100, 30.953500, 30.185700, 31.579700,
    32.577569, 33.477398, 39.021581, 41.386432, 41.596552],
    index=pd.PeriodIndex(start='1990', end='2004', freq='A'))

# R, fpp: aust <- window(austourists,start=2005)
aust = pd.Series([
    41.727458, 24.041850, 32.328103, 37.328708, 46.213153,
    29.346326, 36.482910, 42.977719, 48.901525, 31.180221,
    37.717881, 40.420211, 51.206863, 31.887228, 40.978263,
    43.772491, 55.558567, 33.850915, 42.076383, 45.642292,
    59.766780, 35.191877, 44.319737, 47.913736],
    index=pd.PeriodIndex(start='2005Q1', end='2010Q4', freq='Q'))


def test_ses():
    # Test simple exponential smoothing (FPP: 7.1)

    # Fixed coefficients
    mod = ExponentialSmoothing(oildata)
    mod.initialize_simple()

    res1 = mod.filter([0.2])
    forecast1 = res1.forecast(17)

    res2 = mod.filter([0.6])
    forecast2 = res2.forecast(17)

    mod.initialize_estimated()
    res3 = mod.filter([0.8920537799707837, 447.4808338900134572])
    forecast3 = res3.forecast(17)

    # Test fitted values
    nobs = len(oildata)
    assert_allclose(res1.fittedvalues, results_fitted['oil_fit1'][:nobs])
    assert_allclose(res2.fittedvalues, results_fitted['oil_fit2'][:nobs])
    assert_allclose(res3.fittedvalues, results_fitted['oil_fit3'][:nobs])

    # Test forecasts
    assert_allclose(forecast1, results_fitted['oil_fit1'][nobs:])
    assert_allclose(forecast2, results_fitted['oil_fit2'][nobs:])
    assert_allclose(forecast3, results_fitted['oil_fit3'][nobs:])

    # Test that our fitted coefficients are close
    res4 = mod.fit(disp=0)
    assert_allclose(res3.sse, res4.sse)


def test_holt():
    # Test Holt's linear trend method (FPP: 7.2)

    # Fixed coefficients
    mod = ExponentialSmoothing(air, trend='add')
    mod.initialize_simple()

    res1 = mod.filter([0.8, 0.2])
    forecast1 = res1.forecast(14)

    # Note: not sure what "exponential=TRUE" means in the FPP example, so
    # we don't use air_fit2

    mod = ExponentialSmoothing(air, trend='add', damped_trend=True)
    params = [0.8, 0.2, 0.8502192719485773, 13.9648223911758613,
              4.5054521189765335]
    # Correct the parameters, since FPP uses beta = 0.2 = alpha beta^*
    # whereas we use beta^*
    params[1] = params[1] / params[0]
    res3 = mod.filter(params)
    forecast3 = res3.forecast(14)

    # Test fitted values
    nobs = len(air)
    assert_allclose(res1.fittedvalues, results_fitted['air_fit1'][:nobs])
    # assert_allclose(res2.fittedvalues, results_fitted['air_fit2'][:nobs])
    assert_allclose(res3.fittedvalues, results_fitted['air_fit3'][:nobs])

    # Test forecasts
    assert_allclose(forecast1, results_fitted['air_fit1'][nobs:])
    # assert_allclose(forecast2, results_fitted['air_fit2'][nobs:])
    assert_allclose(forecast3, results_fitted['air_fit3'][nobs:])

    # Test that our fitted coefficients are at least as good
    res4 = mod.fit(disp=0)
    assert_equal(res4.sse <= res3.sse, True)


def test_holt_winters():
    # Test Holt-Winters seasonal method (FPP: 7.5)

    # Additive seasonal
    mod = ExponentialSmoothing(aust, trend='add', seasonal='add',
                               seasonal_periods=4)
    params = [0.3348574920655552067, 0.0001004364187189504,
              0.6646924189795750948, 31.5959653237241901991,
              0.6372054007993471769, 1.3338300720957152468,
              -1.7975502536799012887, -9.2926619452692182932]
    # Correct the parameters, since FPP returns the most recent three,
    # whereas we expect the first three
    params[-3:] = [-np.sum(params[-3:]), params[-1], params[-2]]
    # Correct the parameters, since FPP uses beta = 0.2 = alpha beta^*
    # whereas we use beta^*
    params[1] = params[1] / params[0]

    res1 = mod.filter(params)
    forecast1 = res1.forecast(5)

    # Test fitted values, forecasts
    nobs = len(aust)
    assert_allclose(res1.fittedvalues, results_fitted['aust_fit1'][:nobs])
    assert_allclose(forecast1, results_fitted['aust_fit1'][nobs:])

    # res1_fit = mod.fit(disp=0, maxiter=1000)
    # assert_allclose(res1_fit.sse, 55.415665248542453014)

    # Multiplicative seasonal
    mod = ExponentialSmoothing(aust, trend='add', seasonal='mul',
                               seasonal_periods=4)
    params = [4.403205660911278e-01, 5.108847435980930e-02,
              1.701284241111066e-04, 3.231170100330772e+01,
              9.711447080686224e-01, 1.026014323491403e+00,
              9.477819993836413e-01, 7.657808219931680e-01]
    # Correct the parameters, since FPP returns the most recent three,
    # whereas we expect the first three
    params[-3:] = [mod.seasonal_periods - np.sum(params[-3:]), params[-1], params[-2]]
    # Correct the parameters, since FPP uses beta = 0.2 = alpha beta^*
    # whereas we use beta^*
    params[1] = params[1] / params[0]

    res2 = mod.filter(params)
    forecast2 = res2.forecast(5)

    assert_allclose(res2.fittedvalues, results_fitted['aust_fit2'][:nobs])
    assert_allclose(forecast2, results_fitted['aust_fit2'][nobs:], atol=1e-5)
    # Note: for some reason, the forecast test fails at full precision. There
    # appear to be small errors that accumulate as the number of forecasts
    # increases. However, it appears that these accumulate in FPP, since the
    # answer we get corresponds to the exact answer, as we test below:
    desired = (res2.level[-1] + 5 * res2.slope[-1]) * res2.seasonal[-4]
    assert_allclose(forecast2[-1], desired)

    # res2_fit = mod.fit(disp=0, maxiter=1000)
    # assert_allclose(res2_fit.sse, 0.028339250333730307135)
