"""
Tests for simulation of time series

Author: Chad Fulton
License: Simplified-BSD
"""

from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.python import PYTHON_IMPL_WASM

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter

import statsmodels
from statsmodels.tools.sm_exceptions import (
    EstimationWarning,
    SpecificationWarning,
)
from statsmodels.tsa.statespace import (
    dynamic_factor,
    mlemodel,
    sarimax,
    structural,
    varmax,
)

from .test_impulse_responses import TVSS


def test_arma_lfilter():
    # Tests of an ARMA model simulation against scipy.signal.lfilter
    # Note: the first elements of the generated SARIMAX datasets are based on
    # the initial state, so we do not include them in the comparisons
    rs = np.random.RandomState(10239)
    nobs = 100
    eps = rs.normal(size=nobs)

    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate(
        [0.5, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = lfilter([1], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate(
        [0.5, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = lfilter([1, 0.5], [1], eps)
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate(
        [0.5, 0.2, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = lfilter([1, 0.2], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)


def test_arma_direct():
    # Tests of an ARMA model simulation against direct construction
    # This is useful for e.g., trend components
    # Note: the first elements of the generated SARIMAX datasets are based on
    # the initial state, so we do not include them in the comparisons
    rs = np.random.RandomState(10239)
    nobs = 100
    eps = rs.normal(size=nobs)
    exog = rs.normal(size=nobs)

    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate(
        [0.5, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate(
        [0.5, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate(
        [0.5, 0.2, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept
    mod = sarimax.SARIMAX([0], order=(1, 0, 1), trend="c")
    actual = mod.simulate(
        [1.3, 0.5, 0.2, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = trend + 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept + time trend
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), order=(1, 0, 1), trend="ct")
    actual = mod.simulate(
        [1.3, 0.2, 0.5, 0.2, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = trend + 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept + time trend + exog
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    # Note: the model is regression with SARIMAX errors, so the exog is
    # introduced into the observation equation rather than the ARMA part
    mod = sarimax.SARIMAX(
        np.zeros(nobs + 1), exog=np.r_[0, exog], order=(1, 0, 1), trend="ct"
    )
    actual = mod.simulate(
        [1.3, 0.2, -0.5, 0.5, 0.2, 1.0],
        nobs + 1,
        state_shocks=np.r_[eps, 0],
        initial_state=np.zeros(mod.k_states),
    )
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = trend + 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    desired = desired - 0.5 * exog
    assert_allclose(actual[1:], desired)


def test_structural():
    rs = np.random.RandomState(38947)
    nobs = 100
    eps = rs.normal(size=nobs)
    exog = rs.normal(size=nobs)

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # AR(1)
    mod1 = structural.UnobservedComponents([0], autoregressive=1)
    mod2 = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod1.simulate(
        [1, 0.5], nobs, state_shocks=eps, initial_state=np.zeros(mod1.k_states)
    )
    desired = mod2.simulate(
        [0.5, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    assert_allclose(actual, desired)

    # ARX(1)
    mod1 = structural.UnobservedComponents(np.zeros(nobs), exog=exog, autoregressive=1)
    mod2 = sarimax.SARIMAX(np.zeros(nobs), exog=exog, order=(1, 0, 0))
    actual = mod1.simulate(
        [1, 0.5, 0.2], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    desired = mod2.simulate(
        [0.2, 0.5, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    assert_allclose(actual, desired)

    # Irregular
    mod = structural.UnobservedComponents([0], "irregular")
    actual = mod.simulate(
        [1.0], nobs, measurement_shocks=eps, initial_state=np.zeros(mod.k_states)
    )
    assert_allclose(actual, eps)

    # Fixed intercept
    # (in practice this is a deterministic constant, because an irregular
    #  component must be added)
    warning = SpecificationWarning
    match = "irregular component added"
    with pytest.warns(warning, match=match):
        mod = structural.UnobservedComponents([0], "fixed intercept")
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=[10])
    assert_allclose(actual, 10 + eps)

    # Deterministic constant
    mod = structural.UnobservedComponents([0], "deterministic constant")
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=[10])
    assert_allclose(actual, 10 + eps)

    # Local level
    mod = structural.UnobservedComponents([0], "local level")
    actual = mod.simulate(
        [1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=eps2,
        initial_state=np.zeros(mod.k_states),
    )
    assert_allclose(actual, eps + eps3)

    # Random walk
    mod = structural.UnobservedComponents([0], "random walk")
    actual = mod.simulate(
        [1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=eps2,
        initial_state=np.zeros(mod.k_states),
    )
    assert_allclose(actual, eps + eps3)

    # Fixed slope
    # (in practice this is a deterministic trend, because an irregular
    #  component must be added)
    warning = SpecificationWarning
    match = "irregular component added"
    with pytest.warns(warning, match=match):
        mod = structural.UnobservedComponents([0], "fixed slope")
    actual = mod.simulate(
        [1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=eps2,
        initial_state=[0, 1],
    )
    assert_allclose(actual, eps + np.arange(100))

    # Deterministic trend
    mod = structural.UnobservedComponents([0], "deterministic trend")
    actual = mod.simulate(
        [1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[0, 1]
    )
    assert_allclose(actual, eps + np.arange(100))

    # Local linear deterministic trend
    mod = structural.UnobservedComponents([0], "local linear deterministic trend")
    actual = mod.simulate(
        [1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=eps2,
        initial_state=[0, 1],
    )
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    # Random walk with drift
    mod = structural.UnobservedComponents([0], "random walk with drift")
    actual = mod.simulate([1.0], nobs, state_shocks=eps2, initial_state=[0, 1])
    desired = np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    # Local linear trend
    mod = structural.UnobservedComponents([0], "local linear trend")
    actual = mod.simulate(
        [1.0, 1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=np.c_[eps2, eps1],
        initial_state=[0, 1],
    )
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    actual = mod.simulate(
        [1.0, 1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=np.c_[eps1, eps2],
        initial_state=[0, 1],
    )
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Smooth trend
    mod = structural.UnobservedComponents([0], "smooth trend")
    actual = mod.simulate(
        [1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=eps1,
        initial_state=[0, 1],
    )
    desired = eps + np.r_[np.arange(100)]
    assert_allclose(actual, desired)

    actual = mod.simulate(
        [1.0, 1.0],
        nobs,
        measurement_shocks=eps,
        state_shocks=eps2,
        initial_state=[0, 1],
    )
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Random trend
    mod = structural.UnobservedComponents([0], "random trend")
    actual = mod.simulate([1.0, 1.0], nobs, state_shocks=eps1, initial_state=[0, 1])
    desired = np.r_[np.arange(100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1.0, 1.0], nobs, state_shocks=eps2, initial_state=[0, 1])
    desired = np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Seasonal (deterministic)
    mod = structural.UnobservedComponents(
        [0], "irregular", seasonal=2, stochastic_seasonal=False
    )
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=[10])
    desired = eps + np.tile([10, -10], 50)
    assert_allclose(actual, desired)

    # Seasonal (stochastic)
    mod = structural.UnobservedComponents([0], "irregular", seasonal=2)
    actual = mod.simulate(
        [1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[10]
    )
    desired = eps + np.r_[np.tile([10, -10], 25), np.tile([11, -11], 25)]
    assert_allclose(actual, desired)

    # Cycle (deterministic)
    mod = structural.UnobservedComponents([0], "irregular", cycle=True)
    actual = mod.simulate(
        [1.0, 1.2], nobs, measurement_shocks=eps, initial_state=[1, 0]
    )
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
    mod = structural.UnobservedComponents(
        [0], "irregular", cycle=True, stochastic_cycle=True
    )
    actual = mod.simulate(
        [1.0, 1.0, 1.2],
        nobs,
        measurement_shocks=eps,
        state_shocks=np.c_[eps2, eps2],
        initial_state=[1, 0],
    )
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
    rs = np.random.RandomState(371934)
    nobs = 100
    eps = rs.normal(size=nobs)
    exog = rs.normal(size=(nobs, 1))

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # VAR(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 0), trend="n")
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate(
        [0.5, 0.2, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod1.k_states)
    )
    desired = mod2.simulate(
        [0.5, 0.2, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    assert_allclose(actual, desired)

    # VMA(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(0, 2), trend="n")
    mod2 = sarimax.SARIMAX([0], order=(0, 0, 2))
    actual = mod1.simulate(
        [0.5, 0.2, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod1.k_states)
    )
    desired = mod2.simulate(
        [0.5, 0.2, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    assert_allclose(actual, desired)

    # VARMA(2, 2) - single series
    warning = EstimationWarning
    match = r"VARMA\(p,q\) models is not"
    with pytest.warns(warning, match=match):
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend="n")
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2))
    actual = mod1.simulate(
        [0.5, 0.2, 0.1, -0.2, 1],
        nobs,
        state_shocks=eps,
        initial_state=np.zeros(mod1.k_states),
    )
    desired = mod2.simulate(
        [0.5, 0.2, 0.1, -0.2, 1],
        nobs,
        state_shocks=eps,
        initial_state=np.zeros(mod2.k_states),
    )
    assert_allclose(actual, desired)

    # VARMA(2, 2) + trend - single series
    warning = EstimationWarning
    match = r"VARMA\(p,q\) models is not"
    with pytest.warns(warning, match=match):
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend="c")
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2), trend="c")
    actual = mod1.simulate(
        [10, 0.5, 0.2, 0.1, -0.2, 1],
        nobs,
        state_shocks=eps,
        initial_state=np.zeros(mod1.k_states),
    )
    desired = mod2.simulate(
        [10, 0.5, 0.2, 0.1, -0.2, 1],
        nobs,
        state_shocks=eps,
        initial_state=np.zeros(mod2.k_states),
    )
    assert_allclose(actual, desired)

    # VAR(1)
    transition = np.array([[0.5, 0.1], [-0.1, 0.2]])

    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend="n")
    actual = mod.simulate(
        np.r_[transition.ravel(), 1.0, 0, 1.0],
        nobs,
        state_shocks=np.c_[eps1, eps1],
        initial_state=np.zeros(mod.k_states),
    )
    assert_allclose(actual, 0)

    actual = mod.simulate(
        np.r_[transition.ravel(), 1.0, 0, 1.0],
        nobs,
        state_shocks=np.c_[eps1, eps1],
        initial_state=[1, 1],
    )
    desired = np.zeros((nobs, 2))
    state = np.r_[1, 1]
    for i in range(nobs):
        desired[i] = state
        state = np.dot(transition, state)
    assert_allclose(actual, desired)

    # VAR(1) + measurement error
    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend="n", measurement_error=True)
    actual = mod.simulate(
        np.r_[transition.ravel(), 1.0, 0, 1.0, 1.0, 1.0],
        nobs,
        measurement_shocks=np.c_[eps, eps],
        state_shocks=np.c_[eps1, eps1],
        initial_state=np.zeros(mod.k_states),
    )
    assert_allclose(actual, np.c_[eps, eps])

    # VARX(1)
    mod = varmax.VARMAX(np.zeros((nobs, 2)), order=(1, 0), trend="n", exog=exog)
    actual = mod.simulate(
        np.r_[transition.ravel(), 5, -2, 1.0, 0, 1.0],
        nobs,
        state_shocks=np.c_[eps1, eps1],
        initial_state=[1, 1],
    )
    desired = np.zeros((nobs, 2))
    state = np.r_[1, 1]
    for i in range(nobs):
        desired[i] = state
        if i < nobs - 1:
            state = exog[i + 1] * [5, -2] + np.dot(transition, state)
    assert_allclose(actual, desired)

    # VMA(1)
    # TODO: This is just a smoke test
    mod = varmax.VARMAX(rs.normal(size=(nobs, 2)), order=(0, 1), trend="n")
    mod.simulate(mod.start_params, nobs)

    # VARMA(2, 2) + trend + exog
    # TODO: This is just a smoke test
    warning = EstimationWarning
    match = r"VARMA\(p,q\) models is not"
    with pytest.warns(warning, match=match):
        mod = varmax.VARMAX(
            rs.normal(size=(nobs, 2)), order=(2, 2), trend="c", exog=exog
        )
    mod.simulate(mod.start_params, nobs)


def test_dynamic_factor():
    rs = np.random.RandomState(93739)
    nobs = 100
    eps = rs.normal(size=nobs)
    exog = rs.normal(size=(nobs, 1))

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # DFM: 2 series, AR(2) factor
    mod1 = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=2)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate(
        [-0.9, 0.8, 1.0, 1.0, 0.5, 0.2],
        nobs,
        measurement_shocks=np.c_[eps1, eps1],
        state_shocks=eps,
        initial_state=np.zeros(mod1.k_states),
    )
    desired = mod2.simulate(
        [0.5, 0.2, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM: 2 series, AR(2) factor, exog
    mod1 = dynamic_factor.DynamicFactor(
        np.zeros((nobs, 2)), k_factors=1, factor_order=2, exog=exog
    )
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate(
        [-0.9, 0.8, 5, -2, 1.0, 1.0, 0.5, 0.2],
        nobs,
        measurement_shocks=np.c_[eps1, eps1],
        state_shocks=eps,
        initial_state=np.zeros(mod1.k_states),
    )
    desired = mod2.simulate(
        [0.5, 0.2, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states)
    )
    assert_allclose(actual[:, 0], -0.9 * desired + 5 * exog[:, 0])
    assert_allclose(actual[:, 1], 0.8 * desired - 2 * exog[:, 0])

    # DFM, 3 series, VAR(2) factor, exog, error VAR
    # TODO: This is just a smoke test
    mod = dynamic_factor.DynamicFactor(
        rs.normal(size=(nobs, 3)),
        k_factors=2,
        factor_order=2,
        exog=exog,
        error_order=2,
        error_var=True,
    )
    mod.simulate(mod.start_params, nobs)


def test_known_initialization():
    # Need to test that "known" initialization is taken into account in
    # time series simulation
    rs = np.random.RandomState(38947)
    nobs = 100
    eps = rs.normal(size=nobs)

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # SARIMAX
    # (test that when state shocks are shut down, the initial state
    # geometrically declines according to the AR parameter)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    mod.ssm.initialize_known([100], [[0]])
    actual = mod.simulate([0.5, 1.0], nobs, state_shocks=eps1)
    assert_allclose(actual, 100 * 0.5 ** np.arange(nobs))

    # Unobserved components
    # (test that the initial level shifts the entire path)
    mod = structural.UnobservedComponents([0], "local level")
    mod.ssm.initialize_known([100], [[0]])
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2)
    assert_allclose(actual, 100 + eps + eps3)

    # VARMAX
    # (here just test that with an independent VAR we have each initial state
    # geometrically declining at the appropriate rate)
    transition = np.diag([0.5, 0.2])
    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend="n")
    mod.initialize_known([100, 50], np.diag([0, 0]))
    actual = mod.simulate(
        np.r_[transition.ravel(), 1.0, 0, 1.0],
        nobs,
        measurement_shocks=np.c_[eps1, eps1],
        state_shocks=np.c_[eps1, eps1],
    )

    assert_allclose(
        actual, np.c_[100 * 0.5 ** np.arange(nobs), 50 * 0.2 ** np.arange(nobs)]
    )

    # Dynamic factor
    # (test that the initial state declines geometrically and then loads
    # correctly onto the series)
    mod = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=1)
    mod.initialize_known([100], [[0]])

    actual = mod.simulate(
        [0.8, 0.2, 1.0, 1.0, 0.5],
        nobs,
        measurement_shocks=np.c_[eps1, eps1],
        state_shocks=eps1,
    )
    tmp = 100 * 0.5 ** np.arange(nobs)
    assert_allclose(actual, np.c_[0.8 * tmp, 0.2 * tmp])


def test_sequential_simulate():
    # Test that we can perform simulation, change the system matrices, and then
    # perform simulation again (i.e., check that everything updates correctly
    # in the simulation smoother).
    n_simulations = 100
    mod = sarimax.SARIMAX([1], order=(0, 0, 0), trend="c")

    actual = mod.simulate([1, 0], n_simulations)
    assert_allclose(actual, np.ones(n_simulations))

    actual = mod.simulate([10, 0], n_simulations)
    assert_allclose(actual, np.ones(n_simulations) * 10)


def test_sarimax_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant SARIMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 5)

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    desired = 10 * 0.5 ** np.arange(1, nsimulations + 1)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_simple_differencing_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant SARIMAX model
    # in which simple differencing is used.
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter([0.5, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 5)

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    desired = 10 * 0.5 ** np.arange(1, nsimulations + 1)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_time_invariant_shocks():
    # Test simulating values from the end of a time-invariant SARIMAX model,
    # with nonzero shocks
    rs = np.random.RandomState(28312391)
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = (
        lfilter([1], [1, -0.5], np.r_[initial_state, state_shocks])[:-1]
        + measurement_shocks
    )
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_sarimax_simple_differencing_end_time_invariant_shocks():
    # Test simulating values from the end of a time-invariant SARIMAX model
    # in which simple differencing is used.
    # In this test, we suppress randomness by setting the shocks to zeros
    rs = np.random.RandomState(28312392)
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter([0.5, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = (
        lfilter([1], [1, -0.5], np.r_[initial_state, state_shocks])[:-1]
        + measurement_shocks
    )
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_sarimax_time_varying_trend_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog, trend="t")
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 12)

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    desired = lfilter([1], [1, -0.2], np.r_[12, np.arange(11, 20)])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_simple_differencing_time_varying_trend_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # in which simple differencing is used.
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), trend="t", simple_differencing=True)
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 12)

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    desired = lfilter([1], [1, -0.2], np.r_[12, np.arange(11, 20)])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_time_varying_trend_shocks():
    # Test simulating values from the end of a time-varying SARIMAX model,
    # with nonzero shocks
    rs = np.random.RandomState(28312393)
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog, trend="t")
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    x = np.r_[initial_state, state_shocks + np.arange(11, 21)]
    desired = lfilter([1], [1, -0.2], x)[:-1] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_sarimax_simple_differencing_time_varying_trend_shocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # in which simple differencing is used.
    # with nonzero shocks
    rs = np.random.RandomState(28312394)
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), trend="t", simple_differencing=True)
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]
    assert_allclose(initial_state, 12)

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    x = np.r_[initial_state, state_shocks + np.arange(11, 21)]
    desired = lfilter([1], [1, -0.2], x)[:-1] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_sarimax_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    # Note that `exog` here has basically the same effect as measurement shocks
    endog = np.arange(1, 11)
    exog = np.arange(1, 21) ** 2
    mod = sarimax.SARIMAX(endog, exog=exog[:10])
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    desired = lfilter([1], [1, -0.2], np.r_[initial_state, [0] * 9]) + exog[10:]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_sarimax_simple_differencing_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # with simple differencing
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    exog = np.cumsum(np.arange(0, 21) ** 2)
    mod = sarimax.SARIMAX(
        endog, order=(1, 1, 0), exog=exog[:11], simple_differencing=True
    )
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(
        nsimulations,
        exog=exog[11:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    desired = (
        lfilter([1], [1, -0.2], np.r_[initial_state, [0] * 9]) + np.diff(exog)[10:]
    )
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[11:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[11:]))


def test_sarimax_time_varying_exog_shocks():
    # Test simulating values from the end of a time-varying SARIMAX model,
    # with nonzero shocks
    rs = np.random.RandomState(28312396)
    endog = np.arange(1, 11)
    exog = np.arange(1, 21) ** 2
    mod = sarimax.SARIMAX(endog, exog=exog[:10])
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10

    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    x = np.r_[initial_state, state_shocks[:-1]]
    desired = lfilter([1], [1, -0.2], x) + exog[10:] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_sarimax_simple_differencing_time_varying_exog_shocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # Note that `exog` here has basically the same effect as measurement shocks
    rs = np.random.RandomState(28312397)
    endog = np.cumsum(np.arange(0, 11))
    exog = np.cumsum(np.arange(0, 21) ** 2)
    mod = sarimax.SARIMAX(
        endog, order=(1, 1, 0), exog=exog[:11], simple_differencing=True
    )
    res = mod.filter([1.0, 0.2, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        exog=exog[11:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Compute the desired simulated values directly
    x = np.r_[initial_state, state_shocks[:-1]]
    desired = lfilter([1], [1, -0.2], x) + np.diff(exog)[10:] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[11:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_unobserved_components_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant
    # UnobservedComponents model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    mod = structural.UnobservedComponents(endog, "llevel")
    res = mod.filter([1.0, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # The mean of the simulated local level values is just the last value
    desired = initial_state[0]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_unobserved_components_end_time_invariant_shocks():
    # Test simulating values from the end of a time-invariant
    # UnobservedComponents model, with nonzero shocks
    rs = np.random.RandomState(28312399)
    endog = np.arange(1, 11)
    mod = structural.UnobservedComponents(endog, "llevel")
    res = mod.filter([1.0, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = (
        initial_state + np.cumsum(np.r_[0, state_shocks[:-1]]) + measurement_shocks
    )
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_unobserved_components_end_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying
    # UnobservedComponents model with exog
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    exog = np.arange(1, 21) ** 2
    mod = structural.UnobservedComponents(endog, "llevel", exog=exog[:10])
    res = mod.filter([1.0, 1.0, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # The mean of the simulated local level values is just the last value
    desired = initial_state[0] + exog[10:]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_unobserved_components_end_time_varying_exog_shocks():
    # Test simulating values from the end of a time-varying
    # UnobservedComponents model with exog
    rs = np.random.RandomState(2831231)
    endog = np.arange(1, 11)
    exog = np.arange(1, 21) ** 2
    mod = structural.UnobservedComponents(endog, "llevel", exog=exog[:10])
    res = mod.filter([1.0, 1.0, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=nsimulations)
    state_shocks = rs.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = (
        initial_state
        + np.cumsum(np.r_[0, state_shocks[:-1]])
        + measurement_shocks
        + exog[10:]
    )
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_varmax_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant VARMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend="n")
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[:, -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = (initial_state[:, None] * 2 ** np.arange(10)).T
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_varmax_end_time_invariant_shocks():
    # Test simulating values from the end of a time-invariant VARMAX model,
    # with nonzero shocks
    rs = np.random.RandomState(2831233)
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend="n")
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=(nsimulations, mod.k_endog))
    state_shocks = rs.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[:, -1]

    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_varmax_end_time_varying_trend_noshocks():
    # Test simulating values from the end of a time-varying VARMAX model
    # with a trend
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend="ct")
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1, 1, 1.0, 1.0, 1.0, 0.5, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))

    # Need to set the final predicted state given the new trend
    with res._set_final_predicted_state(exog=None, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    tmp_trend = 1 + np.arange(11, 21)
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + tmp_trend[i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_varmax_end_time_varying_trend_shocks():
    # Test simulating values from the end of a time-varying VARMAX model
    # with a trend
    rs = np.random.RandomState(2831235)
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend="ct")
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1, 1, 1.0, 1.0, 1.0, 0.5, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=(nsimulations, mod.k_endog))
    state_shocks = rs.normal(size=(nsimulations, mod.k_states))

    # Need to set the final predicted state given the new trend
    with res._set_final_predicted_state(exog=None, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    tmp_trend = 1 + np.arange(11, 21)
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + tmp_trend[i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_varmax_end_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying VARMAX model
    # with exog
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    exog = np.arange(1, 21) ** 2
    mod = varmax.VARMAX(endog, trend="n", exog=exog[:10])
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))

    # Need to set the final predicted state given the new exog
    tmp_exog = mod._validate_out_of_sample_exog(exog[10:], out_of_sample=10)
    with res._set_final_predicted_state(exog=tmp_exog, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + exog[10 + i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_varmax_end_time_varying_exog_shocks():
    # Test simulating values from the end of a time-varying VARMAX model
    # with exog
    rs = np.random.RandomState(2831237)
    endog = np.arange(1, 23).reshape(11, 2)
    exog = np.arange(1, 21) ** 2
    mod = varmax.VARMAX(endog[:10], trend="n", exog=exog[:10])
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])

    mod2 = varmax.VARMAX(endog, trend="n", exog=exog[:11])
    res2 = mod2.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=(nsimulations, mod.k_endog))
    state_shocks = rs.normal(size=(nsimulations, mod.k_states))

    # Need to set the final predicted state given the new exog
    tmp_exog = mod._validate_out_of_sample_exog(exog[10:], out_of_sample=10)
    with res._set_final_predicted_state(exog=tmp_exog, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )
    actual2 = res2.simulate(
        nsimulations,
        exog=exog[11:],
        anchor=-1,
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=res2.predicted_state[:, -2],
    )

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + exog[10 + i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)
    assert_allclose(actual2, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_dynamic_factor_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant dynamic factor
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    # The factor is a random walk, so it is not initialized as stationary
    mod = dynamic_factor.DynamicFactor(
        endog, k_factors=1, factor_order=1, initialization="diffuse"
    )
    mod.ssm.filter_univariate = True
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulation
    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_dynamic_factor_end_time_invariant_shocks():
    # Test simulating values from the end of a time-invariant dynamic factor
    rs = np.random.RandomState(2831239)
    endog = np.arange(1, 21).reshape(10, 2)
    # The factor is a random walk, so it is not initialized as stationary
    mod = dynamic_factor.DynamicFactor(
        endog, k_factors=1, factor_order=1, initialization="diffuse"
    )
    mod.ssm.filter_univariate = True
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=(nsimulations, mod.k_endog))
    state_shocks = rs.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulation
    actual = res.simulate(
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_dynamic_factor_end_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying dynamic factor
    # model with exogenous inputs
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    exog = np.arange(1, 21) ** 2
    # The factor is a random walk, so it is not initialized as stationary
    mod = dynamic_factor.DynamicFactor(
        endog,
        k_factors=1,
        factor_order=1,
        exog=exog[:10],
        initialization="diffuse",
    )
    mod.ssm.filter_univariate = True
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulation
    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks + exog[10:, None]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_dynamic_factor_end_time_varying_exog_shocks():
    # Test simulating values from the end of a time-varying dynamic factor
    # model with exogenous inputs
    rs = np.random.RandomState(283121)
    endog = np.arange(1, 23).reshape(11, 2)
    exog = np.arange(1, 21) ** 2
    # The factor is a random walk, so it is not initialized as stationary
    mod = dynamic_factor.DynamicFactor(
        endog[:10],
        k_factors=1,
        factor_order=1,
        exog=exog[:10],
        initialization="diffuse",
    )
    mod.ssm.filter_univariate = True
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    mod2 = dynamic_factor.DynamicFactor(
        endog,
        k_factors=1,
        factor_order=1,
        exog=exog[:11],
        initialization="diffuse",
    )
    mod2.ssm.filter_univariate = True
    res2 = mod2.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    nsimulations = 10
    measurement_shocks = rs.normal(size=(nsimulations, mod.k_endog))
    state_shocks = rs.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulations
    actual = res.simulate(
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )
    actual2 = res2.simulate(
        nsimulations,
        exog=exog[11:],
        anchor=-1,
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks + exog[10:, None]
    assert_allclose(actual, desired)
    assert_allclose(actual2, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params,
        nsimulations,
        exog=exog[10:],
        anchor="end",
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state,
    )

    assert_allclose(mod_actual, desired)


def test_pandas_univariate_rangeindex():
    # Simulate will also have RangeIndex
    endog = pd.Series(np.zeros(2))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros(2), initial_state=np.zeros(1))
    desired = pd.Series([0, 0])
    assert_allclose(actual, desired)

    # Alternative anchor changes the index
    actual = res.simulate(
        2, anchor=2, state_shocks=np.zeros(2), initial_state=np.zeros(1)
    )
    ix = pd.RangeIndex(2, 4)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_univariate_rangeindex_repetitions():
    # Simulate will also have RangeIndex
    endog = pd.Series(np.zeros(2))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(
        2, state_shocks=np.zeros(2), initial_state=np.zeros(1), repetitions=2
    )
    columns = pd.MultiIndex.from_product([["y"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(
        2, anchor=2, state_shocks=np.zeros(2), initial_state=np.zeros(1), repetitions=2
    )
    ix = pd.RangeIndex(2, 4)
    columns = pd.MultiIndex.from_product([["y"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_univariate_dateindex():
    # Simulation will maintain have date index
    ix = pd.date_range(start="2000", periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros(2), initial_state=np.zeros(1))
    ix = pd.date_range(start="2000-01", periods=2, freq=MONTH_END)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

    # Alternative anchor changes the index
    actual = res.simulate(
        2, anchor=2, state_shocks=np.zeros(2), initial_state=np.zeros(1)
    )
    ix = pd.date_range(start="2000-03", periods=2, freq=MONTH_END)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)


def test_pandas_univariate_dateindex_repetitions():
    # Simulation will maintain have date index
    ix = pd.date_range(start="2000", periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(
        2, state_shocks=np.zeros(2), initial_state=np.zeros(1), repetitions=2
    )
    ix = pd.date_range(start="2000-01", periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([["y"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(
        2, anchor=2, state_shocks=np.zeros(2), initial_state=np.zeros(1), repetitions=2
    )
    ix = pd.date_range(start="2000-03", periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([["y"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_multivariate_rangeindex():
    # Simulate will also have RangeIndex
    endog = pd.DataFrame(np.zeros((2, 2)))
    mod = varmax.VARMAX(endog, trend="n")
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2))
    desired = pd.DataFrame(np.zeros((2, 2)))
    assert_allclose(actual, desired)

    # Alternative anchor changes the index
    actual = res.simulate(
        2, anchor=2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2)
    )
    ix = pd.RangeIndex(2, 4)
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_multivariate_rangeindex_repetitions():
    # Simulate will also have RangeIndex
    endog = pd.DataFrame(np.zeros((2, 2)), columns=["y1", "y2"])
    mod = varmax.VARMAX(endog, trend="n")
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(
        2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2), repetitions=2
    )
    columns = pd.MultiIndex.from_product([["y1", "y2"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(
        2,
        anchor=2,
        state_shocks=np.zeros((2, 2)),
        initial_state=np.zeros(2),
        repetitions=2,
    )
    ix = pd.RangeIndex(2, 4)
    columns = pd.MultiIndex.from_product([["y1", "y2"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_multivariate_dateindex():
    # Simulate will also have RangeIndex
    ix = pd.date_range(start="2000", periods=2, freq=MONTH_END)
    endog = pd.DataFrame(np.zeros((2, 2)), index=ix)
    mod = varmax.VARMAX(endog, trend="n")
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2))
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix)
    assert_allclose(actual, desired)

    # Alternative anchor changes the index
    actual = res.simulate(
        2, anchor=2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2)
    )
    ix = pd.date_range(start="2000-03", periods=2, freq=MONTH_END)
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_multivariate_dateindex_repetitions():
    # Simulate will also have RangeIndex
    ix = pd.date_range(start="2000", periods=2, freq=MONTH_END)
    endog = pd.DataFrame(np.zeros((2, 2)), columns=["y1", "y2"], index=ix)
    mod = varmax.VARMAX(endog, trend="n")
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(
        2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2), repetitions=2
    )
    columns = pd.MultiIndex.from_product([["y1", "y2"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), columns=columns, index=ix)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(
        2,
        anchor=2,
        state_shocks=np.zeros((2, 2)),
        initial_state=np.zeros(2),
        repetitions=2,
    )
    ix = pd.date_range(start="2000-03", periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([["y1", "y2"], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_anchor():
    # Test that anchor with dates works
    ix = pd.date_range(start="2000", periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])

    desired = res.simulate(
        2, anchor=1, state_shocks=np.zeros(2), initial_state=np.zeros(1)
    )

    # Anchor to date
    actual = res.simulate(
        2, anchor=ix[1], state_shocks=np.zeros(2), initial_state=np.zeros(1)
    )
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

    # Anchor to negative index
    actual = res.simulate(
        2, anchor=-1, state_shocks=np.zeros(2), initial_state=np.zeros(1)
    )
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


@pytest.mark.smoke
def test_time_varying():
    mod = TVSS(np.zeros((10, 2)))
    mod.simulate([], 10)


def test_time_varying_obs_cov():
    mod = TVSS(np.zeros((10, 2)))
    mod["obs_cov"] = np.zeros((mod.k_endog, mod.k_endog, mod.nobs))
    mod["obs_cov", ..., 9] = np.eye(mod.k_endog)
    mod["state_intercept", :] = 0
    mod["state_cov"] = mod["state_cov", :, :, 0] * 0
    mod["selection"] = mod["selection", :, :, 0]
    assert_equal(mod["state_cov"].shape, (mod.ssm.k_posdef, mod.ssm.k_posdef))
    assert_equal(mod["selection"].shape, (mod.k_states, mod.ssm.k_posdef))

    sim = mod.simulate([], 10, initial_state=np.zeros(mod.k_states))
    assert_allclose(sim[:9], mod["obs_intercept", :, :9].T)


def test_time_varying_state_cov():
    mod = TVSS(np.zeros((10, 2)))
    mod["obs_cov"] = mod["obs_cov", :, :, 0] * 0
    mod["selection"] = mod["selection", :, :, 0]
    mod["state_intercept", :] = 0
    mod["state_cov"] = np.zeros((mod.ssm.k_posdef, mod.ssm.k_posdef, mod.nobs))
    mod["state_cov", ..., -1] = np.eye(mod.ssm.k_posdef)
    assert_equal(mod["obs_cov"].shape, (mod.k_endog, mod.k_endog))
    assert_equal(mod["selection"].shape, (mod.k_states, mod.ssm.k_posdef))
    sim = mod.simulate([], 10)
    assert_allclose(sim, mod["obs_intercept"].T)


@pytest.mark.smoke
def test_time_varying_selection():
    mod = TVSS(np.zeros((10, 2)))
    mod["obs_cov"] = mod["obs_cov", :, :, 0]
    mod["state_cov"] = mod["state_cov", :, :, 0]
    assert_equal(mod["obs_cov"].shape, (mod.k_endog, mod.k_endog))
    assert_equal(mod["state_cov"].shape, (mod.ssm.k_posdef, mod.ssm.k_posdef))
    mod.simulate([], 10)


SYSTEM_MATRICES = [
    "design",
    "obs_intercept",
    "obs_cov",
    "transition",
    "state_intercept",
    "selection",
    "state_cov",
]


def _simulate_recursion(ssm, nsimulations, eps, eta, initial_state):
    # y_t = d_t + Z_t a_t + eps_t and a_{t+1} = c_t + T_t a_t + R_t eta_t,
    # starting from a_1 = initial_state; a time-invariant matrix has a single
    # period
    def matrix_at(name, t):
        matrix = getattr(ssm, name)
        return matrix[..., t if matrix.shape[-1] > 1 else 0]

    obs = np.zeros((nsimulations, ssm.k_endog))
    states = np.zeros((nsimulations, ssm.k_states))
    state = np.asarray(initial_state, dtype=float)
    for t in range(nsimulations):
        states[t] = state
        obs[t] = matrix_at("obs_intercept", t) + matrix_at("design", t) @ state
        obs[t] += eps[t]
        state = (
            matrix_at("state_intercept", t)
            + matrix_at("transition", t) @ state
            + matrix_at("selection", t) @ eta[t]
        )
    return obs, states


@pytest.mark.parametrize("missing", [False, True])
def test_simulate_fewer_periods_time_varying(missing):
    # A simulation shorter than a time-varying model raised "Invalid
    # time-varying dimension": the simulated model was built from the
    # full-length matrices. It uses the first nsimulations periods, as a model
    # built on those periods does. With missing data, the second simulated
    # model (built for simulation smoothing) raised the same error.
    nobs, nsimulations = 10, 5
    endog = np.zeros((nobs, 2))
    if missing:
        endog[2, 0] = np.nan
    mod = TVSS(endog)
    rng = np.random.default_rng(3461)
    # TVSS has a time-invariant state intercept
    mod["state_intercept"] = rng.standard_normal((mod.k_states, nobs))
    eps = rng.standard_normal((nsimulations, mod.k_endog))
    eta = rng.standard_normal((nsimulations, mod.ssm.k_posdef))
    initial_state = rng.standard_normal(mod.k_states)
    shocks = dict(
        measurement_shocks=eps, state_shocks=eta, initial_state=initial_state
    )

    actual = mod.ssm.simulate(nsimulations, **shocks)

    first = mlemodel.MLEModel(
        endog[:nsimulations],
        k_states=mod.k_states,
        k_posdef=mod.ssm.k_posdef,
        initialization="diffuse",
    )
    for name in SYSTEM_MATRICES:
        first.ssm[name] = getattr(mod.ssm, name)[..., :nsimulations]
    desired = first.ssm.simulate(nsimulations, **shocks)
    recursion = _simulate_recursion(mod.ssm, nsimulations, eps, eta, initial_state)
    for i in range(2):
        assert_allclose(actual[i], desired[i], rtol=1e-12)
        assert_allclose(actual[i], recursion[i], rtol=1e-10)


def _local_level_ar(endog):
    mod = mlemodel.MLEModel(endog, k_states=1, k_posdef=1)
    mod["design"] = [[1.0]]
    mod["obs_intercept"] = [0.3]
    mod["obs_cov"] = [[1.0]]
    mod["transition"] = [[0.5]]
    mod["state_intercept"] = [0.2]
    mod["selection"] = [[1.0]]
    mod["state_cov"] = [[1.0]]
    mod.ssm.initialize_known([0.0], [[1.0]])
    return mod


@pytest.mark.parametrize("missing", [False, True])
def test_simulate_more_periods_than_model(missing):
    # A time-invariant model can simulate more periods than it has. With
    # missing data, the simulator copied nsimulations periods of the model's
    # observations (reading past their end) into a second simulated model
    # that a simulation does not use. The simulation equals that of the
    # model extended to nsimulations periods.
    nobs, nsimulations = 10, 25
    endog = np.zeros(nobs)
    if missing:
        endog[3] = np.nan
    mod = _local_level_ar(endog)
    rng = np.random.default_rng(7310)
    eps = rng.standard_normal((nsimulations, 1))
    eta = rng.standard_normal((nsimulations, 1))
    initial_state = rng.standard_normal(1)
    shocks = dict(
        measurement_shocks=eps, state_shocks=eta, initial_state=initial_state
    )

    actual = mod.ssm.simulate(nsimulations, **shocks)
    extended = mod.ssm.extend(np.zeros((nsimulations - nobs, 1)))
    assert_equal(extended.nobs, nsimulations)
    desired = extended.simulate(nsimulations, **shocks)
    recursion = _simulate_recursion(mod.ssm, nsimulations, eps, eta, initial_state)
    for i in range(2):
        assert_allclose(actual[i], desired[i], rtol=1e-12)
        assert_allclose(actual[i], recursion[i], rtol=1e-10)


SIMULATE_LONG_MISSING = """
import numpy as np
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother

endog = np.arange(10.0)
endog[3] = np.nan
ssm = SimulationSmoother(k_endog=1, k_states=1, k_posdef=1)
ssm.bind(endog)
ssm["design"] = [[1.0]]
ssm["obs_cov"] = [[1.0]]
ssm["transition"] = [[0.5]]
ssm["selection"] = [[1.0]]
ssm["state_cov"] = [[1.0]]
ssm.initialize_known([0.0], [[1.0]])
obs, states = ssm.simulate(10**6, rng=0)
assert obs.shape == (10**6, 1)
print("simulated")
"""


@pytest.mark.skipif(PYTHON_IMPL_WASM, reason="Emscripten does not support processes")
def test_simulate_much_longer_than_model_missing(tmp_path):
    # With missing data, a simulation much longer than the model read far past
    # the end of the model's observations and crashed the interpreter (an
    # access violation on Windows), so this runs in a subprocess. The
    # subprocess imports the statsmodels that runs the tests.
    env = os.environ.copy()
    root = str(Path(statsmodels.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    res = subprocess.run(
        [sys.executable, "-c", SIMULATE_LONG_MISSING],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "simulated" in res.stdout
