"""
Tests for simulation of time series

Author: Chad Fulton
License: Simplified-BSD
"""

from __future__ import division, absolute_import, print_function

import warnings
import numpy as np
import pandas as pd
import os
from scipy.signal import lfilter

from statsmodels.tsa.statespace import (sarimax, structural, varmax,
                                        dynamic_factor)
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
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = lfilter([1], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = lfilter([1, 0.5], [1], eps)
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = lfilter([1, 0.2], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)


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
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept
    mod = sarimax.SARIMAX([0], order=(1, 0, 1), trend='c')
    actual = mod.simulate([1.3, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept + time trend
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept + time trend + exog
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    # Note: the model is regression with SARIMAX errors, so the exog is
    # introduced into the observation equation rather than the ARMA part
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), exog=np.r_[0, exog],
                          order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, -0.5, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    desired = desired - 0.5 * exog
    assert_allclose(actual[1:], desired)


def test_structural():
    # Clear warnings
    structural.__warningregistry__ = {}

    np.random.seed(38947)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=nobs)

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # AR(1)
    mod1 = structural.UnobservedComponents([0], autoregressive=1)
    mod2 = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod1.simulate([1, 0.5], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # ARX(1)
    mod1 = structural.UnobservedComponents(np.zeros(nobs), exog=exog,
                                           autoregressive=1)
    mod2 = sarimax.SARIMAX(np.zeros(nobs), exog=exog, order=(1, 0, 0))
    actual = mod1.simulate([1, 0.5, 0.2], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod2.k_states))
    desired = mod2.simulate([0.2, 0.5, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # Irregular
    mod = structural.UnobservedComponents([0], 'irregular')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps)

    # Fixed intercept
    # (in practice this is a deterministic constant, because an irregular
    #  component must be added)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = structural.UnobservedComponents([0], 'fixed intercept')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=[10])
    assert_allclose(actual, 10 + eps)

    # Deterministic constant
    mod = structural.UnobservedComponents([0], 'deterministic constant')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=[10])
    assert_allclose(actual, 10 + eps)

    # Local level
    mod = structural.UnobservedComponents([0], 'local level')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2,
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps + eps3)

    # Random walk
    mod = structural.UnobservedComponents([0], 'random walk')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2,
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps + eps3)

    # Fixed slope
    # (in practice this is a deterministic trend, because an irregular
    #  component must be added)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = structural.UnobservedComponents([0], 'fixed slope')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    assert_allclose(actual, eps + np.arange(100))

    # Deterministic trend
    mod = structural.UnobservedComponents([0], 'deterministic trend')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    assert_allclose(actual, eps + np.arange(100))

    # Local linear deterministic trend
    mod = structural.UnobservedComponents(
        [0], 'local linear deterministic trend')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    # Random walk with drift
    mod = structural.UnobservedComponents([0], 'random walk with drift')
    actual = mod.simulate([1.], nobs, state_shocks=eps2,
                          initial_state=[0, 1])
    desired = np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    # Local linear trend
    mod = structural.UnobservedComponents([0], 'local linear trend')
    actual = mod.simulate([1., 1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=np.c_[eps2, eps1], initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1., 1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=np.c_[eps1, eps2], initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Smooth trend
    mod = structural.UnobservedComponents([0], 'smooth trend')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps1, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Random trend
    mod = structural.UnobservedComponents([0], 'random trend')
    actual = mod.simulate([1., 1.], nobs,
                          state_shocks=eps1, initial_state=[0, 1])
    desired = np.r_[np.arange(100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1., 1.], nobs,
                          state_shocks=eps2, initial_state=[0, 1])
    desired = np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Seasonal (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2,
                                          stochastic_seasonal=False)
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=[10])
    desired = eps + np.tile([10, -10], 50)
    assert_allclose(actual, desired)

    # Seasonal (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2)
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[10])
    desired = eps + np.r_[np.tile([10, -10], 25), np.tile([11, -11], 25)]
    assert_allclose(actual, desired)

    # Cycle (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True)
    actual = mod.simulate([1., 1.2], nobs, measurement_shocks=eps,
                          initial_state=[1, 0])
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = eps
    states = [1, 0]
    for i in range(nobs):
        desired[i] += states[0]
        states = np.dot(T, states)
    assert_allclose(actual, desired)

    # Cycle (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True,
                                          stochastic_cycle=True)
    actual = mod.simulate([1., 1., 1.2], nobs, measurement_shocks=eps,
                          state_shocks=np.c_[eps2, eps2], initial_state=[1, 0])
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = eps
    states = [1, 0]
    for i in range(nobs):
        desired[i] += states[0]
        states = np.dot(T, states) + eps2[i]
    assert_allclose(actual, desired)


def test_varmax():
    # Clear warnings
    varmax.__warningregistry__ = {}

    np.random.seed(371934)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=(nobs, 1))

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # VAR(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 0), trend='nc')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VMA(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(0, 2), trend='nc')
    mod2 = sarimax.SARIMAX([0], order=(0, 0, 2))
    actual = mod1.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VARMA(2, 2) - single series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='nc')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2))
    actual = mod1.simulate([0.5, 0.2, 0.1, -0.2, 1], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 0.1, -0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VARMA(2, 2) + trend - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='c')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod1.simulate([10, 0.5, 0.2, 0.1, -0.2, 1], nobs,
                           state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([10, 0.5, 0.2, 0.1, -0.2, 1], nobs,
                            state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VAR(1)
    transition = np.array([[0.5,  0.1],
                           [-0.1, 0.2]])

    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend='nc')
    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1.], nobs,
                          state_shocks=np.c_[eps1, eps1],
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, 0)

    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1.], nobs,
                          state_shocks=np.c_[eps1, eps1], initial_state=[1, 1])
    desired = np.zeros((nobs, 2))
    state = np.r_[1, 1]
    for i in range(nobs):
        desired[i] = state
        state = np.dot(transition, state)
    assert_allclose(actual, desired)

    # VAR(1) + measurement error
    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend='nc',
                        measurement_error=True)
    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1., 1., 1.], nobs,
                          measurement_shocks=np.c_[eps, eps],
                          state_shocks=np.c_[eps1, eps1],
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, np.c_[eps, eps])

    # VARX(1)
    mod = varmax.VARMAX(np.zeros((nobs, 2)), order=(1, 0), trend='nc',
                        exog=exog)
    actual = mod.simulate(np.r_[transition.ravel(), 5, -2, 1., 0, 1.], nobs,
                          state_shocks=np.c_[eps1, eps1], initial_state=[1, 1])
    desired = np.zeros((nobs, 2))
    state = np.r_[1, 1]
    for i in range(nobs):
        desired[i] = state
        state = exog[i] * [5, -2] + np.dot(transition, state)
    assert_allclose(actual, desired)

    # VMA(1)
    # TODO: This is just a smoke test
    mod = varmax.VARMAX(
        np.random.normal(size=(nobs, 2)), order=(0, 1), trend='nc')
    mod.simulate(mod.start_params, nobs)

    # VARMA(2, 2) + trend + exog
    # TODO: This is just a smoke test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = varmax.VARMAX(
            np.random.normal(size=(nobs, 2)), order=(2, 2), trend='c',
            exog=exog)
    mod.simulate(mod.start_params, nobs)


def test_dynamic_factor():
    np.random.seed(93739)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=(nobs, 1))

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # DFM: 2 series, AR(2) factor
    mod1 = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=2)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate([-0.9, 0.8, 1., 1., 0.5, 0.2], nobs,
                           measurement_shocks=np.c_[eps1, eps1],
                           state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM: 2 series, AR(2) factor, exog
    mod1 = dynamic_factor.DynamicFactor(np.zeros((nobs, 2)), k_factors=1,
                                        factor_order=2, exog=exog)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate([-0.9, 0.8, 5, -2, 1., 1., 0.5, 0.2], nobs,
                           measurement_shocks=np.c_[eps1, eps1],
                           state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual[:, 0], -0.9 * desired + 5 * exog[:, 0])
    assert_allclose(actual[:, 1], 0.8 * desired - 2 * exog[:, 0])

    # DFM, 3 series, VAR(2) factor, exog, error VAR
    # TODO: This is just a smoke test
    mod = dynamic_factor.DynamicFactor(np.random.normal(size=(nobs, 3)),
                                       k_factors=2, factor_order=2, exog=exog,
                                       error_order=2, error_var=True)
    mod.simulate(mod.start_params, nobs)
