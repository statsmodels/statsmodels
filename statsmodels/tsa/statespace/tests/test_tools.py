"""
Tests for tools

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd

from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.api import acovf
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
        # diff = 1, seasonal_diff=1, seasonal_periods=4
        (x, 1, 1, 4, [0]*5),
        (x**2, 1, 1, 4, [8]*5),
        (x**3, 1, 1, 4, [60, 84, 108, 132, 156]),
        # diff = 1, seasonal_diff=2, seasonal_periods=2
        (x, 1, 2, 2, [0]*5),
        (x**2, 1, 2, 2, [0]*5),
        (x**3, 1, 2, 2, [24]*5),
        (x**4, 1, 2, 2, [240, 336, 432, 528, 624]),
    ]

    def test_cases(self):
        # Basic cases
        for series, diff, seasonal_diff, seasonal_periods, result in self.cases:
            
            # Test numpy array
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)

            # Test as Pandas Series
            series = pd.Series(series)

            # Rewrite to test as n-dimensional array
            series = np.c_[series, series]
            result = np.c_[result, result]

            # Test Numpy array
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)

            # Test as Pandas Dataframe
            series = pd.DataFrame(series)
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)

class TestSolveDiscreteLyapunov(object):

    def solve_dicrete_lyapunov_direct(self, a, q, complex_step=False):
        # This is the discrete Lyapunov solver as "real function of real
        # variables":  the difference between this and the usual, complex,
        # version is that in the Kronecker product the second argument is
        # *not* conjugated here.
        if not complex_step:
            lhs = np.kron(a, a.conj())
            lhs = np.eye(lhs.shape[0]) - lhs
            x = np.linalg.solve(lhs, q.flatten())
        else:
            lhs = np.kron(a, a)
            lhs = np.eye(lhs.shape[0]) - lhs
            x = np.linalg.solve(lhs, q.flatten())

        return np.reshape(x, q.shape)

    def test_univariate(self):
        # Real case
        a = np.array([[0.5]])
        q = np.array([[10.]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a complex
        # function)
        a = np.array([[0.5+1j]])
        q = np.array([[10.]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a real
        # function)
        a = np.array([[0.5+1j]])
        q = np.array([[10.]])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)

    def test_multivariate(self):
        # Real case
        a = tools.companion_matrix([1, -0.4, 0.5])
        q = np.diag([10., 5.])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a complex
        # function)
        a = tools.companion_matrix([1, -0.4+0.1j, 0.5])
        q = np.diag([10., 5.])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=False)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=False)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a real
        # function)
        a = tools.companion_matrix([1, -0.4+0.1j, 0.5])
        q = np.diag([10., 5.])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)

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

class TestUnconstrainStationaryUnivariate(object):

    cases = [
        (np.array([-2./((1+2.**2)**0.5)]), np.array([2.]))
    ]

    def test_cases(self):
        for constrained, unconstrained in self.cases:
            result = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(result, unconstrained)

class TestStationaryUnivariate(object):
    # Test that the constraint and unconstraint functions are inverses

    constrained_cases = [
        np.array([0]), np.array([0.1]), np.array([-0.5]), np.array([0.999])]
    unconstrained_cases = [
        np.array([10.]), np.array([-40.42]), np.array([0.123])]

    def test_cases(self):
        for constrained in self.constrained_cases:
            unconstrained = tools.unconstrain_stationary_univariate(constrained)
            reconstrained = tools.constrain_stationary_univariate(unconstrained)
            assert_allclose(reconstrained, constrained)

        for unconstrained in self.unconstrained_cases:
            constrained = tools.constrain_stationary_univariate(unconstrained)
            reunconstrained = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(reunconstrained, unconstrained)

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
    assert_allclose(_acovf([Phi_1], Sigma_u)[0], Gamma_0, atol=1e-3)

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

    # Test sample acovf in the univariate case against sm.tsa.acovf
    x = np.arange(20)*1.0
    assert_allclose(
        np.squeeze(tools._compute_multivariate_sample_acovf(x, maxlag=4)),
        acovf(x)[:5])


def test_multivariate_pacf():
    # Test sample acovf in the univariate case against sm.tsa.acovf
    np.random.seed(1234)
    x = np.arange(10000)
    y = np.random.normal(size=10000)
    # Note: could make this test more precise with higher nobs, but no need to
    assert_allclose(
        tools._compute_multivariate_sample_pacf(np.c_[x, y], maxlag=1)[0],
        np.diag([1, 0]), atol=1e-2)

class TestConstrainStationaryMultivariate(object):

    cases = [
        # This is the same test as the univariate case above, except notice
        # the sign difference; this is an array input / output
        (np.array([[2.]]), np.eye(1), np.array([[2./((1+2.**2)**0.5)]])),
        # Same as above, but now a list input / output
        ([np.array([[2.]])], np.eye(1), [np.array([[2./((1+2.**2)**0.5)]])])
    ]

    eigval_cases = [
        [np.array([[0]])],
        [np.array([[100]]), np.array([[50]])],
        [np.array([[30, 1], [-23, 15]]), np.array([[10, .3], [.5, -30]])],
    ]

    def test_cases(self):
        # Test against known results
        for unconstrained, error_variance, constrained in self.cases:
            result = tools.constrain_stationary_multivariate(
                unconstrained, error_variance)
            assert_allclose(result[0], constrained)

        # Test that the constrained results correspond to companion matrices
        # with eigenvalues less than 1 in modulus
        for unconstrained in self.eigval_cases:
            if type(unconstrained) == list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            companion = tools.companion_matrix(
                [1] + [-constrained[i] for i in range(len(constrained))]
            ).T
            assert_equal(np.max(np.abs(np.linalg.eigvals(companion))) < 1, True)
            


class TestUnconstrainStationaryMultivariate(object):

    cases = [
        # This is the same test as the univariate case above, except notice
        # the sign difference; this is an array input / output
        (np.array([[2./((1+2.**2)**0.5)]]), np.eye(1), np.array([[2.]])),
        # Same as above, but now a list input / output
        ([np.array([[2./((1+2.**2)**0.5)]])], np.eye(1), [np.array([[2.]])])
    ]

    def test_cases(self):
        for constrained, error_variance, unconstrained in self.cases:
            result = tools.unconstrain_stationary_multivariate(
                constrained, error_variance)
            assert_allclose(result[0], unconstrained)

class TestStationaryMultivariate(object):
    # Test that the constraint and unconstraint functions are inverses

    constrained_cases = [
        np.array([[0]]), np.array([[0.1]]), np.array([[-0.5]]), np.array([[0.999]]),
        [np.array([[0]])],
        np.array([[0.8, -0.2]]),
        [np.array([[0.8]]), np.array([[-0.2]])],
        [np.array([[0.3, 0.01], [-0.23, 0.15]]), np.array([[0.1, 0.03], [0.05, -0.3]])],
        np.array([[0.3, 0.01, 0.1, 0.03], [-0.23, 0.15, 0.05, -0.3]])
    ]
    unconstrained_cases = [
        np.array([[0]]), np.array([[-40.42]]), np.array([[0.123]]),
        [np.array([[0]])],
        np.array([[100, 50]]),
        [np.array([[100]]), np.array([[50]])],
        [np.array([[30, 1], [-23, 15]]), np.array([[10, .3], [.5, -30]])],
        np.array([[30, 1, 10, .3], [-23, 15, .5, -30]])
    ]

    def test_cases(self):
        for constrained in self.constrained_cases:
            if type(constrained) == list:
                cov = np.eye(constrained[0].shape[0])
            else:
                cov = np.eye(constrained.shape[0])
            unconstrained, _ = tools.unconstrain_stationary_multivariate(constrained, cov)
            reconstrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            assert_allclose(reconstrained, constrained)

        for unconstrained in self.unconstrained_cases:
            if type(unconstrained) == list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            reunconstrained, _ = tools.unconstrain_stationary_multivariate(constrained, cov)
            # Note: low tolerance comes from last example in unconstrained_cases,
            # but is not a real problem
            assert_allclose(reunconstrained, unconstrained, atol=1e-4)
