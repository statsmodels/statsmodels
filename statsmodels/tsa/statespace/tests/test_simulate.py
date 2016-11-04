"""
Tests for simulation of time series

Author: Chad Fulton
License: Simplified-BSD
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os
from scipy.signal import lfilter

from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.tools import compatibility_mode
from numpy.testing import (assert_allclose, assert_almost_equal, assert_equal,
                           assert_raises)
from nose.exc import SkipTest


def test_arma_lfilter():
    # Tests of an ARMA model simulation against scipy.signal.lfilter
    # Note: the first elements of the generated SARIMAX datasets are based on
    # the initial state, so we don't include them in the comparisons
    np.random.seed(10239)
    nobs = 100
    eps = np.random.normal(size=nobs)

    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0])
    desired = lfilter([1], [1, -0.5], eps)
    assert_allclose(actual[0, 1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0])
    desired = lfilter([1, 0.5], [1], eps)
    assert_allclose(actual[0, 1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.], nobs + 1, state_shocks=np.r_[eps, 0])
    desired = lfilter([1, 0.2], [1, -0.5], eps)
    assert_allclose(actual[0, 1:], desired)


def test_arma_direct():
    # Tests of an ARMA model simulation against direct construction
    # This is useful for e.g. trend components
    # Note: the first elements of the generated SARIMAX datasets are based on
    # the initial state, so we don't include them in the comparisons
    np.random.seed(10239)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=nobs)

    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0])
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + eps[i]
    assert_allclose(actual[0, 1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0])
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * eps[i - 1] + eps[i]
    assert_allclose(actual[0, 1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.], nobs + 1, state_shocks=np.r_[eps, 0])
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[0, 1:], desired)

    # ARMA(1, 1) + intercept
    mod = sarimax.SARIMAX([0], order=(1, 0, 1), trend='c')
    actual = mod.simulate([1.3, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0])
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    assert_allclose(actual[0, 1:], desired)

    # ARMA(1, 1) + intercept + time trend
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0])
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    assert_allclose(actual[0, 1:], desired)

    # ARMA(1, 1) + intercept + time trend
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    # Note: the model is regression with SARIMAX errors, so the exog is
    # introduced into the observation equation rather than the ARMA part
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), exog=np.r_[0, exog],
                          order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, -0.5, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0])
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    desired = desired - 0.5 * exog
    assert_allclose(actual[0, 1:], desired)
