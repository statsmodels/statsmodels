"""
Tests for impulse responses of time series

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
from numpy.testing import (assert_allclose, assert_almost_equal, assert_equal)


def test_sarimax():
    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    phi = 0.5
    actual = mod.impulse_responses([phi, 1], steps=10)
    desired = np.r_[[phi**i for i in range(11)]]
    assert_allclose(actual, desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    theta = 0.5
    actual = mod.impulse_responses([theta, 1], steps=10)
    desired = np.r_[1, theta, [0]*9]
    assert_allclose(actual, desired)

    # ARMA(2, 2) + constant
    # Stata:
    # webuse lutkepohl2
    # arima dln_inc, arima(2, 0, 2)
    # irf create irf1, set(irf1) step(10)
    # irf table irf
    params = [.01928228, -.03656216, .7588994,
              .27070341, -.72928328, .01122177**0.5]
    mod = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod.impulse_responses(params, steps=10)
    desired = [1, .234141, .021055, .17692, .00951, .133917, .002321, .101544,
               -.001951, .077133, -.004301]
    assert_allclose(actual, desired, atol=1e-6)

    # SARIMAX(1,1,1)x(1,0,1,4) + constant + exog
    # Stata:
    # webuse lutkepohl2
    # gen exog = _n^2
    # arima inc exog, arima(1,1,1) sarima(1,0,1,4)
    # irf create irf2, set(irf2) step(10)
    # irf table irf
    params = [.12853289, 12.207156, .86384742, -.71463236,
              .81878967, -.9533955, 14.043884**0.5]
    exog = np.arange(1, 92)**2
    mod = sarimax.SARIMAX(np.zeros(91), order=(1, 1, 1),
                          seasonal_order=(1, 0, 1, 4), trend='c', exog=exog,
                          simple_differencing=True)
    actual = mod.impulse_responses(params, steps=10)
    desired = [1, .149215, .128899, .111349, -.038417, .063007, .054429,
               .047018, -.069598, .018641, .016103]
    assert_allclose(actual, desired, atol=1e-6)


def test_structural():
    steps = 10

    # AR(1)
    mod = structural.UnobservedComponents([0], autoregressive=1)
    phi = 0.5
    actual = mod.impulse_responses([1, phi], steps)
    desired = np.r_[[phi**i for i in range(steps + 1)]]
    assert_allclose(actual, desired)

    # ARX(1)
    # This is adequately tested in test_simulate.py, since in the time-varying
    # case `impulse_responses` just calls `simulate`

    # Irregular
    mod = structural.UnobservedComponents([0], 'irregular')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Fixed intercept
    # (in practice this is a deterministic constant, because an irregular
    #  component must be added)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = structural.UnobservedComponents([0], 'fixed intercept')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Deterministic constant
    mod = structural.UnobservedComponents([0], 'deterministic constant')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Local level
    mod = structural.UnobservedComponents([0], 'local level')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, 1)

    # Random walk
    mod = structural.UnobservedComponents([0], 'random walk')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 1)

    # Fixed slope
    # (in practice this is a deterministic trend, because an irregular
    #  component must be added)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = structural.UnobservedComponents([0], 'fixed slope')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Deterministic trend
    mod = structural.UnobservedComponents([0], 'deterministic trend')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Local linear deterministic trend
    mod = structural.UnobservedComponents(
        [0], 'local linear deterministic trend')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, 1)

    # Random walk with drift
    mod = structural.UnobservedComponents([0], 'random walk with drift')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 1)

    # Local linear trend
    mod = structural.UnobservedComponents([0], 'local linear trend')
    # - shock the level
    actual = mod.impulse_responses([1., 1., 1.], steps)
    assert_allclose(actual, 1)
    # - shock the trend
    actual = mod.impulse_responses([1., 1., 1.], steps, impulse=1)
    assert_allclose(actual, np.arange(steps + 1))

    # Smooth trend
    mod = structural.UnobservedComponents([0], 'smooth trend')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, np.arange(steps + 1))

    # Random trend
    mod = structural.UnobservedComponents([0], 'random trend')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, np.arange(steps + 1))

    # Seasonal (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2,
                                          stochastic_seasonal=False)
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Seasonal (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2)
    actual = mod.impulse_responses([1., 1.], steps)
    desired = np.r_[1, np.tile([-1, 1], steps // 2)]
    assert_allclose(actual, desired)

    # Cycle (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True)
    actual = mod.impulse_responses([1., 1.2], steps)
    assert_allclose(actual, 0)

    # Cycle (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True,
                                          stochastic_cycle=True)
    actual = mod.impulse_responses([1., 1., 1.2], steps=10)
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = np.zeros(steps + 1)
    states = [1, 0]
    for i in range(steps + 1):
        desired[i] += states[0]
        states = np.dot(T, states)
    assert_allclose(actual, desired)


def test_varmax():
    steps = 10

    # Clear warnings
    varmax.__warningregistry__ = {}

    # VAR(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 0), trend='nc')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([0.5, 0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual, desired)

    # VMA(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(0, 2), trend='nc')
    mod2 = sarimax.SARIMAX([0], order=(0, 0, 2))
    actual = mod1.impulse_responses([0.5, 0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual, desired)

    # VARMA(2, 2) - single series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='nc')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2))
    actual = mod1.impulse_responses([0.5, 0.2, 0.1, -0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 0.1, -0.2, 1], steps)
    assert_allclose(actual, desired)

    # VARMA(2, 2) + trend - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='c')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod1.impulse_responses([10, 0.5, 0.2, 0.1, -0.2, 1], steps)
    desired = mod2.impulse_responses([10, 0.5, 0.2, 0.1, -0.2, 1], steps)
    assert_allclose(actual, desired)

    # VAR(2) + constant
    # Stata:
    # webuse lutkepohl2
    # var dln_inv dln_inc, lags(1/2)
    # irf create irf3, set(irf3) step(10)
    # irf table irf
    # irf table oirf
    params = [-.00122728, .01503679,
              -.22741923, .71030531, -.11596357, .51494891,
              .05974659, .02094608, .05635125, .08332519,
              .04297918, .00159473, .01096298]
    irf_00 = [1, -.227419, -.021806, .093362, -.001875, -.00906, .009605,
              .001323, -.001041, .000769, .00032]
    irf_01 = [0, .059747, .044015, -.008218, .007845, .004629, .000104,
              .000451, .000638, .000063, .000042]
    irf_10 = [0, .710305, .36829, -.065697, .084398, .043038, .000533,
              .005755, .006051, .000548, .000526]
    irf_11 = [1, .020946, .126202, .066419, .028735, .007477, .009878,
              .003287, .001266, .000986, .0005]
    oirf_00 = [0.042979, -0.008642, -0.00035, 0.003908, 0.000054, -0.000321,
               0.000414, 0.000066, -0.000035, 0.000034, 0.000015]
    oirf_01 = [0.001595, 0.002601, 0.002093, -0.000247, 0.000383, 0.000211,
               0.00002, 0.000025, 0.000029, 4.30E-06, 2.60E-06]
    oirf_10 = [0, 0.007787, 0.004037, -0.00072, 0.000925, 0.000472, 5.80E-06,
               0.000063, 0.000066, 6.00E-06, 5.80E-06]
    oirf_11 = [0.010963, 0.00023, 0.001384, 0.000728, 0.000315, 0.000082,
               0.000108, 0.000036, 0.000014, 0.000011, 5.50E-06]

    mod = varmax.VARMAX([[0, 0]], order=(2, 0), trend='c')

    # IRFs
    actual = mod.impulse_responses(params, steps, impulse=0)
    assert_allclose(actual, np.c_[irf_00, irf_01], atol=1e-6)

    actual = mod.impulse_responses(params, steps, impulse=1)
    assert_allclose(actual, np.c_[irf_10, irf_11], atol=1e-6)

    # Orthogonalized IRFs
    actual = mod.impulse_responses(params, steps, impulse=0,
                                   orthogonalized=True)
    assert_allclose(actual, np.c_[oirf_00, oirf_01], atol=1e-6)

    actual = mod.impulse_responses(params, steps, impulse=1,
                                   orthogonalized=True)
    assert_allclose(actual, np.c_[oirf_10, oirf_11], atol=1e-6)

    # VARMA(2, 2) + trend + exog
    # TODO: This is just a smoke test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = varmax.VARMAX(
            np.random.normal(size=(steps, 2)), order=(2, 2), trend='c',
            exog=np.ones(steps), enforce_stationarity=False,
            enforce_invertibility=False)
    mod.impulse_responses(mod.start_params, steps)


def test_dynamic_factor():
    steps = 10
    exog = np.random.normal(size=steps)

    # DFM: 2 series, AR(2) factor
    mod1 = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=2)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([-0.9, 0.8, 1., 1., 0.5, 0.2], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM: 2 series, AR(2) factor, exog
    mod1 = dynamic_factor.DynamicFactor(np.zeros((steps, 2)), k_factors=1,
                                        factor_order=2, exog=exog)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses(
        [-0.9, 0.8, 5, -2, 1., 1., 0.5, 0.2], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM, 3 series, VAR(2) factor, exog, error VAR
    # TODO: This is just a smoke test
    mod = dynamic_factor.DynamicFactor(np.random.normal(size=(steps, 3)),
                                       k_factors=2, factor_order=2, exog=exog,
                                       error_order=2, error_var=True,
                                       enforce_stationarity=False)
    mod.impulse_responses(mod.start_params, steps)
