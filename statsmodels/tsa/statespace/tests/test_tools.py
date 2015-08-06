"""
Tests for tools

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace import tools
# from .results import results_sarimax
from numpy.testing import (
    assert_allclose, assert_equal, assert_array_equal, assert_almost_equal,
    assert_raises
)

class TestCompanionMatrix(object):

    cases = [
        (2, np.array([[0,1],[0,0]])),
        ([1,-1,-2], np.array([[1,1],
                              [2,0]])),
        ([1,-1,-2,-3], np.array([[1,1,0],
                                 [2,0,1],
                                 [3,0,0]])),
        ([1,-np.array([[1,2],[3,4]]),-np.array([[5,6],[7,8]])],
         np.array([[1,2,5,6],
                   [3,4,7,8],
                   [1,0,0,0],
                   [0,1,0,0]]).T)
    ]

    def test_cases(self):
        for polynomial, result in self.cases:
            assert_equal(tools.companion_matrix(polynomial), result)

class TestDiff(object):

    x = np.arange(10)
    cases = [
        # diff = 1
        ([1,2,3], 1, None, 1, [1, 1]),
        # diff = 2
        (x, 2, None, 1, [0]*8),
        # diff = 1, seasonal_diff=1, k_seasons=4
        (x, 1, 1, 4, [0]*5),
        (x**2, 1, 1, 4, [8]*5),
        (x**3, 1, 1, 4, [60, 84, 108, 132, 156]),
        # diff = 1, seasonal_diff=2, k_seasons=2
        (x, 1, 2, 2, [0]*5),
        (x**2, 1, 2, 2, [0]*5),
        (x**3, 1, 2, 2, [24]*5),
        (x**4, 1, 2, 2, [240, 336, 432, 528, 624]),
    ]

    def test_cases(self):
        # Basic cases
        for series, diff, seasonal_diff, k_seasons, result in self.cases:
            
            # Test numpy array
            x = tools.diff(series, diff, seasonal_diff, k_seasons)
            assert_almost_equal(x, result)

            # Test as Pandas Series
            series = pd.Series(series)

            # Rewrite to test as n-dimensional array
            series = np.c_[series, series]
            result = np.c_[result, result]

            # Test Numpy array
            x = tools.diff(series, diff, seasonal_diff, k_seasons)
            assert_almost_equal(x, result)

            # Test as Pandas Dataframe
            series = pd.DataFrame(series)
            x = tools.diff(series, diff, seasonal_diff, k_seasons)
            assert_almost_equal(x, result)

class TestIsInvertible(object):

    cases = [
        ([1, -0.5], True),
        ([1, 1-1e-9], True),
        ([1, 1], False),
        ([1, 0.9,0.1], True),
        (np.array([1,0.9,0.1]), True),
        (pd.Series([1,0.9,0.1]), True)
    ]

    def test_cases(self):
        for polynomial, invertible in self.cases:
            assert_equal(tools.is_invertible(polynomial), invertible)

class TestConstrainStationaryUnivariate(object):

    cases = [
        (np.array([2.]), -2./((1+2.**2)**0.5))
    ]

    def test_cases(self):
        for unconstrained, constrained in self.cases:
            result = tools.constrain_stationary_univariate(unconstrained)
            assert_equal(result, constrained)

class TestValidateMatrixShape(object):
    # name, shape, nrows, ncols, nobs
    valid = [
        ('TEST', (5,2), 5, 2, None),
        ('TEST', (5,2), 5, 2, 10),
        ('TEST', (5,2,10), 5, 2, 10),
    ]
    invalid = [
        ('TEST', (5,), 5, None, None),
        ('TEST', (5,1,1,1), 5, 1, None),
        ('TEST', (5,2), 10, 2, None),
        ('TEST', (5,2), 5, 1, None),
        ('TEST', (5,2,10), 5, 2, None),
        ('TEST', (5,2,10), 5, 2, 5),
    ]

    def test_valid_cases(self):
        for args in self.valid:
            # Just testing that no exception is raised
            tools.validate_matrix_shape(*args)

    def test_invalid_cases(self):
        for args in self.invalid:
            assert_raises(
                ValueError, tools.validate_matrix_shape, *args
            )

class TestValidateVectorShape(object):
    # name, shape, nrows, ncols, nobs
    valid = [
        ('TEST', (5,), 5, None),
        ('TEST', (5,), 5, 10),
        ('TEST', (5,10), 5, 10),
    ]
    invalid = [
        ('TEST', (5,2,10), 5, 10),
        ('TEST', (5,), 10, None),
        ('TEST', (5,10), 5, None),
        ('TEST', (5,10), 5, 5),
    ]

    def test_valid_cases(self):
        for args in self.valid:
            # Just testing that no exception is raised
            tools.validate_vector_shape(*args)

    def test_invalid_cases(self):
        for args in self.invalid:
            assert_raises(
                ValueError, tools.validate_vector_shape, *args
            )

def test_multivariate_acovf():
    _acovf = tools._compute_multivariate_acovf_from_coefficients

    # Test for a VAR(1) process. From Lutkepohl (2007), pages 27-28.
    # See (2.1.14) for Phi_1, (2.1.33) for Sigma_u, and (2.1.34) for Gamma_0
    Sigma_u = np.array([[2.25, 0,   0],
                        [0,    1.0, 0.5],
                        [0,    0.5, 0.74]])
    Phi_1 = np.array([[0.5, 0,   0],
                      [0.1, 0.1, 0.3],
                      [0,   0.2, 0.3]])
    Gamma_0 = np.array([[3.0,   0.161, 0.019],
                        [0.161, 1.172, 0.674],
                        [0.019, 0.674, 0.954]])
    # assert_allclose(_acovf([Phi_1], Sigma_u)[0], Gamma_0, atol=1e-3)

    # Test for a VAR(2) process. From Lutkepohl (2007), pages 28-29
    # See (2.1.40) for Phi_1, Phi_2, (2.1.14) for Sigma_u, and (2.1.42) for
    # Gamma_0, Gamma_1
    Sigma_u = np.diag([0.09, 0.04])
    Phi_1 = np.array([[0.5, 0.1],
                      [0.4, 0.5]])
    Phi_2 = np.array([[0,    0],
                      [0.25, 0]])
    Gamma_0 = np.array([[0.131, 0.066],
                        [0.066, 0.181]])
    Gamma_1 = np.array([[0.072, 0.051],
                        [0.104, 0.143]])
    Gamma_2 = np.array([[0.046, 0.040],
                        [0.113, 0.108]])
    Gamma_3 = np.array([[0.035, 0.031],
                        [0.093, 0.083]])

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=0),
        [Gamma_0], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=1),
        [Gamma_0, Gamma_1], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u),
        [Gamma_0, Gamma_1], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=2),
        [Gamma_0, Gamma_1, Gamma_2], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=3),
        [Gamma_0, Gamma_1, Gamma_2, Gamma_3], atol=1e-3)

class TestConstrainStationaryMultivariate(object):

    cases = [
        (np.array([[2.]]), np.array([[2./((1+2.**2)**0.5)]]))
    ]

    def test_cases(self):
        for unconstrained, constrained in self.cases:
            result = tools.constrain_stationary_multivariate(
                unconstrained, np.eye(unconstrained.shape[0]))
            assert_allclose(result[0], constrained)
