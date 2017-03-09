"""
Tests for initialization

Author: Chad Fulton
License: Simplified-BSD
"""

from __future__ import division, absolute_import, print_function

import warnings
import numpy as np
import pandas as pd
import os
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import lfilter

from statsmodels.tsa.statespace import (sarimax, structural, varmax,
                                        dynamic_factor)
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.tools import compatibility_mode
from numpy.testing import (assert_allclose, assert_almost_equal, assert_equal,
                           assert_raises)
from nose.exc import SkipTest

if compatibility_mode:
    raise SkipTest


def test_global_known():
    # Test for global known initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    # Known, mean
    init = Initialization(mod.k_states, 'known', constant=[1.5])
    a, Pinf, Pstar = init()
    assert_equal(a, [1.5])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.diag([0]))

    # Known, covariance
    init = Initialization(mod.k_states, 'known', stationary_cov=np.diag([1]))
    a, Pinf, Pstar = init()
    assert_equal(a, [0])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.diag([1]))

    # Known, both
    init = Initialization(mod.k_states, 'known', constant=[1.5],
                          stationary_cov=np.diag([1]))
    a, Pinf, Pstar = init()
    assert_equal(a, [1.5])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.diag([1]))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    # Known, mean
    init = Initialization(mod.k_states, 'known', constant=[1.5, -0.2])
    a, Pinf, Pstar = init()
    assert_equal(a, [1.5, -0.2])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_equal(Pstar, np.diag([0, 0]))

    # Known, covariance
    init = Initialization(mod.k_states, 'known',
                          stationary_cov=np.diag([1, 4.2]))
    a, Pinf, Pstar = init()
    assert_equal(a, [0, 0])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_equal(Pstar, np.diag([1, 4.2]))

    # Known, both
    init = Initialization(mod.k_states, 'known', constant=[1.5, -0.2],
                          stationary_cov=np.diag([1, 4.2]))
    a, Pinf, Pstar = init()
    assert_equal(a, [1.5, -0.2])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_equal(Pstar, np.diag([1, 4.2]))


def test_global_diffuse():
    # Test for global diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    init = Initialization(mod.k_states, 'diffuse')
    a, Pinf, Pstar = init()
    assert_equal(a, [0])
    assert_equal(Pinf, np.eye(1))
    assert_equal(Pstar, np.diag([0]))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    init = Initialization(mod.k_states, 'diffuse')
    a, Pinf, Pstar = init()
    assert_equal(a, [0, 0])
    assert_equal(Pinf, np.eye(2))
    assert_equal(Pstar, np.diag([0, 0]))


def test_global_approximate_diffuse():
    # Test for global approximate diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    init = Initialization(mod.k_states, 'approximate_diffuse')
    a, Pinf, Pstar = init()
    assert_equal(a, [0])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.eye(1) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse', constant=[1.2])
    a, Pinf, Pstar = init()
    assert_equal(a, [1.2])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.eye(1) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse',
                          approximate_diffuse_variance=1e10)
    a, Pinf, Pstar = init()
    assert_equal(a, [0])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.eye(1) * 1e10)

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    init = Initialization(mod.k_states, 'approximate_diffuse')
    a, Pinf, Pstar = init()
    assert_equal(a, [0, 0])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_equal(Pstar, np.eye(2) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse',
                          constant=[1.2, -0.2])
    a, Pinf, Pstar = init()
    assert_equal(a, [1.2, -0.2])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_equal(Pstar, np.eye(2) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse',
                          approximate_diffuse_variance=1e10)
    a, Pinf, Pstar = init()
    assert_equal(a, [0, 0])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_equal(Pstar, np.eye(2) * 1e10)


def test_global_stationary():
    # Test for global approximate diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend='c')

    # no intercept
    intercept = 0
    phi = 0.5
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    assert_equal(a, [0])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.eye(1) * sigma2 / (1 - phi**2))

    # intercept
    intercept = 1.2
    phi = 0.5
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    assert_equal(a, [intercept / (1 - phi)])
    assert_equal(Pinf, np.diag([0]))
    assert_equal(Pstar, np.eye(1) * sigma2 / (1 - phi**2))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0), trend='c')

    # no intercept
    intercept = 0
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    assert_equal(a, [0, 0])
    assert_equal(Pinf, np.diag([0, 0]))
    T = np.array([[0.5, 1],
                  [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov = solve_discrete_lyapunov(T, Q)
    assert_allclose(Pstar, desired_cov)

    # intercept
    intercept = 1.2
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    desired_intercept = np.linalg.inv(np.eye(2) - T).dot([intercept, 0])
    assert_allclose(a, desired_intercept)
    assert_equal(Pinf, np.diag([0, 0]))
    assert_allclose(Pstar, desired_cov)


def test_mixed_basic():
    # Performs a number of tests for setting different initialization for
    # different blocks

    # - 2-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[phi, sigma2])

    # known has constant
    init = Initialization(mod.k_states)
    init.set(0, 'known', constant=[1.2])

    # > known has constant
    init.set(1, 'known', constant=[-0.2])
    a, Pinf, Pstar = init()
    assert_allclose(a, [1.2, -0.2])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_allclose(Pstar, np.diag([0, 0]))

    # > diffuse
    init.unset(1)
    init.set(1, 'diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [1.2, 0])
    assert_equal(Pinf, np.diag([0, 1]))
    assert_allclose(Pstar, np.diag([0, 0]))

    # > approximate diffuse
    init.unset(1)
    init.set(1, 'approximate_diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [1.2, 0])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_allclose(Pstar, np.diag([0, 1e6]))

    # > stationary
    init.unset(1)
    init.set(1, 'stationary')
    a, Pinf, Pstar = init(model=mod)

    assert_allclose(a, [1.2, 0])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_allclose(Pstar, np.diag([0, 0]))

    # known has cov
    init = Initialization(mod.k_states)
    init.set(0, 'known', stationary_cov=np.diag([1]))
    init.set(1, 'diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [0, 0])
    assert_equal(Pinf, np.diag([0, 1]))
    assert_allclose(Pstar, np.diag([1, 0]))

    # known has both
    init = Initialization(mod.k_states)
    init.set(0, 'known', constant=[1.2], stationary_cov=np.diag([1]))
    init.set(1, 'diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [1.2, 0])
    assert_equal(Pinf, np.diag([0, 1]))
    assert_allclose(Pstar, np.diag([1, 0]))

    # - 3-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(3, 0, 0))

    # known has constant
    init = Initialization(mod.k_states)
    init.set((0, 2), 'known', constant=[1.2, -0.2])
    init.set(2, 'diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [1.2, -0.2, 0])
    assert_equal(Pinf, np.diag([0, 0, 1]))
    assert_allclose(Pstar, np.diag([0, 0, 0]))

    # known has cov
    init = Initialization(mod.k_states)
    init.set((0, 2), 'known', stationary_cov=np.diag([1, 4.2]))
    init.set(2, 'diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [0, 0, 0])
    assert_equal(Pinf, np.diag([0, 0, 1]))
    assert_allclose(Pstar, np.diag([1, 4.2, 0]))

    # known has both
    init = Initialization(mod.k_states)
    init.set((0, 2), 'known', constant=[1.2, -0.2],
             stationary_cov=np.diag([1, 4.2]))
    init.set(2, 'diffuse')
    a, Pinf, Pstar = init()

    assert_allclose(a, [1.2, -0.2, 0])
    assert_equal(Pinf, np.diag([0, 0, 1]))
    assert_allclose(Pstar, np.diag([1, 4.2, 0]))


def test_mixed_stationary():
    # More specific tests when one or more blocks are initialized as stationary
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 1, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[phi, sigma2])

    init = Initialization(mod.k_states)
    init.set(0, 'diffuse')
    init.set((1, 3), 'stationary')
    a, Pinf, Pstar = init(model=mod)

    assert_allclose(a, [0, 0, 0])
    assert_equal(Pinf, np.diag([1, 0, 0]))
    desired_cov = np.zeros((3, 3))
    T = np.array([[0.5, 1],
                  [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov[1:, 1:] = solve_discrete_lyapunov(T, Q)
    assert_allclose(Pstar, desired_cov)

    init.clear()
    init.set(0, 'diffuse')
    init.set(1, 'stationary')
    init.set(2, 'approximate_diffuse')
    a, Pinf, Pstar = init(model=mod)

    assert_allclose(a, [0, 0, 0])
    assert_equal(Pinf, np.diag([1, 0, 0]))
    T = np.array([[0.5]])
    Q = np.diag([sigma2])
    desired_cov = np.diag([0, solve_discrete_lyapunov(T, Q), 1e6])
    assert_allclose(Pstar, desired_cov)

    init.clear()
    init.set(0, 'diffuse')
    init.set(1, 'stationary')
    init.set(2, 'stationary')
    a, Pinf, Pstar = init(model=mod)

    desired_cov[2, 2] = 0
    assert_allclose(Pstar, desired_cov)

    # Test with a VAR model
    endog = np.zeros((10, 2))
    mod = varmax.VARMAX(endog, order=(1, 0), )
    intercept = [1.5, -0.1]
    transition = np.array([[0.5, -0.2],
                           [0.1, 0.8]])
    cov = np.array([[1.2, -0.4],
                    [-0.4, 0.4]])
    tril = np.tril_indices(2)
    params = np.r_[intercept, transition.ravel(),
                   np.linalg.cholesky(cov)[tril]]
    mod.update(params)

    # > stationary, global
    init = Initialization(mod.k_states, 'stationary')
    a, Pinf, Pstar = init(model=mod)

    desired_intercept = np.linalg.solve(np.eye(2) - transition, intercept)
    desired_cov = solve_discrete_lyapunov(transition, cov)
    assert_allclose(a, desired_intercept)
    assert_equal(Pinf, np.diag([0, 0]))
    assert_allclose(Pstar, desired_cov)

    # > diffuse, global
    init.set(None, 'diffuse')
    a, Pinf, Pstar = init()
    assert_allclose(a, [0, 0])
    assert_equal(Pinf, np.eye(2))
    assert_allclose(Pstar, np.diag([0, 0]))

    # > stationary, individually
    init.unset(None)
    init.set(0, 'stationary')
    init.set(1, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    assert_allclose(a, [intercept[0] / (1 - transition[0, 0]),
                        intercept[1] / (1 - transition[1, 1])])
    assert_equal(Pinf, np.diag([0, 0]))
    assert_allclose(Pstar, np.diag([cov[0, 0] / (1 - transition[0, 0]**2),
                                    cov[1, 1] / (1 - transition[1, 1]**2)]))


def test_invalid():
    # Invalid initializations (also tests for some invalid calls to set)
    assert_raises(ValueError, Initialization, 5, '')
    assert_raises(ValueError, Initialization, 5, 'stationary', constant=[1, 2])
    assert_raises(ValueError, Initialization, 5, 'stationary',
                  stationary_cov=[1, 2])
    assert_raises(ValueError, Initialization, 5, 'stationary',
                  approximate_diffuse_variance=1e10)
    assert_raises(ValueError, Initialization, 5, 'known')
    assert_raises(ValueError, Initialization, 5, 'known', constant=[1])
    assert_raises(ValueError, Initialization, 5, 'known', stationary_cov=[0])

    # Invalid set() / unset() calls
    init = Initialization(5)
    assert_raises(ValueError, init.set, -1, 'diffuse')
    assert_raises(ValueError, init.unset, -1)
    assert_raises(ValueError, init.set, 5, 'diffuse')
    assert_raises(ValueError, init.unset, 5)
    assert_raises(ValueError, init.set, 'x', 'diffuse')
    assert_raises(ValueError, init.unset, 'x')
    assert_raises(ValueError, init.set, (1, 2, 3), 'diffuse')
    assert_raises(ValueError, init.unset, (1, 2, 3))
    init.set(None, 'diffuse')
    assert_raises(ValueError, init.set, 1, 'diffuse')
    init.clear()
    init.set(1, 'diffuse')
    assert_raises(ValueError, init.set, None, 'stationary')

    init.clear()
    assert_raises(ValueError, init.unset, 1)

    # Invalid __call__
    init = Initialization(2)
    assert_raises(ValueError, init)
    init = Initialization(2, 'stationary')
    assert_raises(ValueError, init)
