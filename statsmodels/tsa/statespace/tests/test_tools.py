"""
Tests for tools

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_equal,
)
import pandas as pd
import pytest
from scipy.linalg import solve_discrete_lyapunov

from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf


class TestCompanionMatrix:

    cases = [
        (2, np.array([[0, 1], [0, 0]])),
        ([1, -1, -2], np.array([[1, 1], [2, 0]])),
        ([1, -1, -2, -3], np.array([[1, 1, 0], [2, 0, 1], [3, 0, 0]])),
        # GH 5570
        (np.int64(2), np.array([[0, 1], [0, 0]])),
    ]

    def test_cases(self):
        for polynomial, result in self.cases:
            assert_equal(tools.companion_matrix(polynomial), result)

    def test_special_case(self):
        polynomial = [1, -np.array([[1, 2], [3, 4]]), -np.array([[5, 6], [7, 8]])]
        result = np.array([[1, 2, 5, 6], [3, 4, 7, 8], [1, 0, 0, 0], [0, 1, 0, 0]]).T
        assert_equal(tools.companion_matrix(polynomial), result)

    def test_c0_identity_input_not_mutated_and_tuple_accepted(self):
        # Passing C_0 = 1 (documented shorthand for an identity matrix) must not
        # replace the scalar in the caller's list, and must also work when the
        # polynomial is passed as a tuple.
        c1 = -np.array([[1.0, 2.0], [3.0, 4.0]])
        c2 = -np.array([[5.0, 6.0], [7.0, 8.0]])

        poly_list = [1, c1, c2]
        result_list = tools.companion_matrix(poly_list)
        assert isinstance(poly_list[0], int)
        assert poly_list[0] == 1

        result_tuple = tools.companion_matrix((1, c1, c2))
        assert_equal(result_tuple, result_list)


class TestDiff:

    x = np.arange(10)
    cases = [
        # diff = 1
        ([1, 2, 3], 1, None, 1, [1, 1]),
        # diff = 2
        (x, 2, None, 1, [0] * 8),
        # diff = 1, seasonal_diff=1, seasonal_periods=4
        (x, 1, 1, 4, [0] * 5),
        (x**2, 1, 1, 4, [8] * 5),
        (x**3, 1, 1, 4, [60, 84, 108, 132, 156]),
        # diff = 1, seasonal_diff=2, seasonal_periods=2
        (x, 1, 2, 2, [0] * 5),
        (x**2, 1, 2, 2, [0] * 5),
        (x**3, 1, 2, 2, [24] * 5),
        (x**4, 1, 2, 2, [240, 336, 432, 528, 624]),
    ]

    # TODO: use pytest.mark.parametrize?
    def test_cases(self):
        # Basic cases
        for series, diff, seas_diff, seasonal_periods, result in self.cases:
            seasonal_diff = seas_diff

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

            # Test as Pandas DataFrame
            series = pd.DataFrame(series)
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)


class TestSolveDiscreteLyapunov:

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
        q = np.array([[10.0]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a complex
        # function)
        a = np.array([[0.5 + 1j]])
        q = np.array([[10.0]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a real
        # function)
        a = np.array([[0.5 + 1j]])
        q = np.array([[10.0]])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)

    def test_multivariate(self):
        # Real case
        a = tools.companion_matrix([1, -0.4, 0.5])
        q = np.diag([10.0, 5.0])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a complex
        # function)
        a = tools.companion_matrix([1, -0.4 + 0.1j, 0.5])
        q = np.diag([10.0, 5.0])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=False)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=False)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a real
        # function)
        a = tools.companion_matrix([1, -0.4 + 0.1j, 0.5])
        q = np.diag([10.0, 5.0])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)


class TestConcat:

    x = np.arange(10)

    valid = [
        (((1, 2, 3), (4,)), (1, 2, 3, 4)),
        (((1, 2, 3), [4]), (1, 2, 3, 4)),
        (([1, 2, 3], np.r_[4]), (1, 2, 3, 4)),
        ((np.r_[1, 2, 3], pd.Series([4])), 0, True, (1, 2, 3, 4)),
        ((pd.Series([1, 2, 3]), pd.Series([4])), 0, True, (1, 2, 3, 4)),
        ((np.c_[x[:2], x[:2]], np.c_[x[2:3], x[2:3]]), np.c_[x[:3], x[:3]]),
        ((np.c_[x[:2], x[:2]].T, np.c_[x[2:3], x[2:3]].T), 1, np.c_[x[:3], x[:3]].T),
        (
            (pd.DataFrame(np.c_[x[:2], x[:2]]), np.c_[x[2:3], x[2:3]]),
            0,
            True,
            np.c_[x[:3], x[:3]],
        ),
    ]

    invalid = [
        (((1, 2, 3), pd.Series([4])), ValueError),
        (((1, 2, 3), np.array([[1, 2]])), ValueError),
    ]

    def test_valid(self):
        for args in self.valid:
            assert_array_equal(tools.concat(*args[:-1]), args[-1])

    def test_invalid(self):
        for args in self.invalid:
            with pytest.raises(args[-1]):
                tools.concat(*args[:-1])


class TestIsInvertible:

    cases = [
        ([1, -0.5], True),
        ([1, 1 - 1e-9], True),
        ([1, 1], False),
        ([1, 0.9, 0.1], True),
        (np.array([1, 0.9, 0.1]), True),
        (pd.Series([1, 0.9, 0.1]), True),
    ]

    def test_cases(self):
        for polynomial, invertible in self.cases:
            assert_equal(tools.is_invertible(polynomial), invertible)


class TestConstrainStationaryUnivariate:

    cases = [(np.array([2.0]), -2.0 / ((1 + 2.0**2) ** 0.5))]

    def test_cases(self):
        for unconstrained, constrained in self.cases:
            result = tools.constrain_stationary_univariate(unconstrained)
            assert_equal(result, constrained)


class TestUnconstrainStationaryUnivariate:

    cases = [(np.array([-2.0 / ((1 + 2.0**2) ** 0.5)]), np.array([2.0]))]

    def test_cases(self):
        for constrained, unconstrained in self.cases:
            result = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(result, unconstrained)


class TestStationaryUnivariate:
    # Test that the constraint and unconstrained functions are inverses

    constrained_cases = [
        np.array([0]),
        np.array([0.1]),
        np.array([-0.5]),
        np.array([0.999]),
    ]
    unconstrained_cases = [np.array([10.0]), np.array([-40.42]), np.array([0.123])]

    def test_cases(self):
        for constrained in self.constrained_cases:
            unconstrained = tools.unconstrain_stationary_univariate(constrained)
            reconstrained = tools.constrain_stationary_univariate(unconstrained)
            assert_allclose(reconstrained, constrained)

        for unconstrained in self.unconstrained_cases:
            constrained = tools.constrain_stationary_univariate(unconstrained)
            reunconstrained = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(reunconstrained, unconstrained)


class TestValidateMatrixShape:
    # name, shape, nrows, ncols, nobs
    valid = [
        ("TEST", (5, 2), 5, 2, None),
        ("TEST", (5, 2), 5, 2, 10),
        ("TEST", (5, 2, 10), 5, 2, 10),
    ]
    invalid = [
        ("TEST", (5,), 5, None, None),
        ("TEST", (5, 1, 1, 1), 5, 1, None),
        ("TEST", (5, 2), 10, 2, None),
        ("TEST", (5, 2), 5, 1, None),
        ("TEST", (5, 2, 10), 5, 2, None),
        ("TEST", (5, 2, 10), 5, 2, 5),
    ]

    def test_valid_cases(self):
        for args in self.valid:
            # Just testing that no exception is raised
            tools.validate_matrix_shape(*args)

    def test_invalid_cases(self):
        for args in self.invalid:
            with pytest.raises(ValueError):
                tools.validate_matrix_shape(*args)


class TestValidateVectorShape:
    # name, shape, nrows, ncols, nobs
    valid = [
        ("TEST", (5,), 5, None),
        ("TEST", (5,), 5, 10),
        ("TEST", (5, 10), 5, 10),
    ]
    invalid = [
        ("TEST", (5, 2, 10), 5, 10),
        ("TEST", (5,), 10, None),
        ("TEST", (5, 10), 5, None),
        ("TEST", (5, 10), 5, 5),
    ]

    def test_valid_cases(self):
        for args in self.valid:
            # Just testing that no exception is raised
            tools.validate_vector_shape(*args)

    def test_invalid_cases(self):
        for args in self.invalid:
            with pytest.raises(ValueError):
                tools.validate_vector_shape(*args)


def test_multivariate_acovf():
    _acovf = tools._compute_multivariate_acovf_from_coefficients

    # Test for a VAR(1) process. From Lutkepohl (2007), pages 27-28.
    # See (2.1.14) for Phi_1, (2.1.33) for Sigma_u, and (2.1.34) for Gamma_0
    Sigma_u = np.array([[2.25, 0, 0], [0, 1.0, 0.5], [0, 0.5, 0.74]])
    Phi_1 = np.array([[0.5, 0, 0], [0.1, 0.1, 0.3], [0, 0.2, 0.3]])
    Gamma_0 = np.array(
        [[3.0, 0.161, 0.019], [0.161, 1.172, 0.674], [0.019, 0.674, 0.954]]
    )
    assert_allclose(_acovf([Phi_1], Sigma_u)[0], Gamma_0, atol=1e-3)

    # Test for a VAR(2) process. From Lutkepohl (2007), pages 28-29
    # See (2.1.40) for Phi_1, Phi_2, (2.1.14) for Sigma_u, and (2.1.42) for
    # Gamma_0, Gamma_1
    Sigma_u = np.diag([0.09, 0.04])
    Phi_1 = np.array([[0.5, 0.1], [0.4, 0.5]])
    Phi_2 = np.array([[0, 0], [0.25, 0]])
    Gamma_0 = np.array([[0.131, 0.066], [0.066, 0.181]])
    Gamma_1 = np.array([[0.072, 0.051], [0.104, 0.143]])
    Gamma_2 = np.array([[0.046, 0.040], [0.113, 0.108]])
    Gamma_3 = np.array([[0.035, 0.031], [0.093, 0.083]])

    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u, maxlag=0), [Gamma_0], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=1), [Gamma_0, Gamma_1], atol=1e-3
    )

    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u), [Gamma_0, Gamma_1], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=2),
        [Gamma_0, Gamma_1, Gamma_2],
        atol=1e-3,
    )

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=3),
        [Gamma_0, Gamma_1, Gamma_2, Gamma_3],
        atol=1e-3,
    )

    # Test sample acovf in the univariate case against sm.tsa.acovf
    x = np.arange(20) * 1.0
    assert_allclose(
        np.squeeze(tools._compute_multivariate_sample_acovf(x, maxlag=4)),
        acovf(x, fft=False)[:5],
    )


def test_multivariate_pacf():
    # Test sample acovf in the univariate case against sm.tsa.acovf
    rs = np.random.RandomState(1234)
    x = np.arange(10000)
    y = rs.normal(size=10000)
    # Note: could make this test more precise with higher nobs, but no need to
    assert_allclose(
        tools._compute_multivariate_sample_pacf(np.c_[x, y], maxlag=1)[0],
        np.diag([1, 0]),
        atol=1e-2,
    )


class TestConstrainStationaryMultivariate:

    cases = [
        # This is the same test as the univariate case above, except notice
        # the sign difference; this is an array input / output
        (np.array([[2.0]]), np.eye(1), np.array([[2.0 / ((1 + 2.0**2) ** 0.5)]])),
        # Same as above, but now a list input / output
        ([np.array([[2.0]])], np.eye(1), [np.array([[2.0 / ((1 + 2.0**2) ** 0.5)]])]),
    ]

    eigval_cases = [
        [np.array([[0]])],
        [np.array([[100]]), np.array([[50]])],
        [np.array([[30, 1], [-23, 15]]), np.array([[10, 0.3], [0.5, -30]])],
    ]

    def test_cases(self):
        # Test against known results
        for unconstrained, error_variance, constrained in self.cases:
            result = tools.constrain_stationary_multivariate(
                unconstrained, error_variance
            )
            assert_allclose(result[0], constrained)

        # Test that the constrained results correspond to companion matrices
        # with eigenvalues less than 1 in modulus
        for unconstrained in self.eigval_cases:
            if type(unconstrained) is list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            companion = tools.companion_matrix(
                [1] + [-np.squeeze(constrained[i]) for i in range(len(constrained))]
            ).T
            assert_array_less(np.abs(np.linalg.eigvals(companion)), 1)


class TestUnconstrainStationaryMultivariate:

    cases = [
        # This is the same test as the univariate case above, except notice
        # the sign difference; this is an array input / output
        (np.array([[2.0 / ((1 + 2.0**2) ** 0.5)]]), np.eye(1), np.array([[2.0]])),
        # Same as above, but now a list input / output
        ([np.array([[2.0 / ((1 + 2.0**2) ** 0.5)]])], np.eye(1), [np.array([[2.0]])]),
    ]

    def test_cases(self):
        for constrained, error_variance, unconstrained in self.cases:
            result = tools.unconstrain_stationary_multivariate(
                constrained, error_variance
            )
            assert_allclose(result[0], unconstrained)


class TestStationaryMultivariate:
    # Test that the constraint and unconstrained functions are inverses

    constrained_cases = [
        np.array([[0]]),
        np.array([[0.1]]),
        np.array([[-0.5]]),
        np.array([[0.999]]),
        [np.array([[0]])],
        np.array([[0.8, -0.2]]),
        [np.array([[0.8]]), np.array([[-0.2]])],
        [np.array([[0.3, 0.01], [-0.23, 0.15]]), np.array([[0.1, 0.03], [0.05, -0.3]])],
        np.array([[0.3, 0.01, 0.1, 0.03], [-0.23, 0.15, 0.05, -0.3]]),
    ]
    unconstrained_cases = [
        np.array([[0]]),
        np.array([[-40.42]]),
        np.array([[0.123]]),
        [np.array([[0]])],
        np.array([[100, 50]]),
        [np.array([[100]]), np.array([[50]])],
        [np.array([[30, 1], [-23, 15]]), np.array([[10, 0.3], [0.5, -30]])],
        np.array([[30, 1, 10, 0.3], [-23, 15, 0.5, -30]]),
    ]

    def test_cases(self):
        for constrained in self.constrained_cases:
            if type(constrained) is list:
                cov = np.eye(constrained[0].shape[0])
            else:
                cov = np.eye(constrained.shape[0])
            unconstrained, _ = tools.unconstrain_stationary_multivariate(
                constrained, cov
            )
            reconstrained, _ = tools.constrain_stationary_multivariate(
                unconstrained, cov
            )
            assert_allclose(reconstrained, constrained)

        for unconstrained in self.unconstrained_cases:
            if type(unconstrained) is list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)
            reunconstrained, _ = tools.unconstrain_stationary_multivariate(
                constrained, cov
            )
            # Note: low tolerance comes from last example in
            # unconstrained_cases, but is not a real problem
            assert_allclose(reunconstrained, unconstrained, atol=1e-4)


def _assert_twins_agree(actual, expected, rtol):
    # The compiled routine returns the dtype of its input, while the Python
    # twin returns float64 for float32 input: compare in the wider dtype.
    # Rounding error in these Cholesky-based recursions is relative to the
    # largest entry of a matrix, not to each entry, so entries much smaller
    # than the largest get an absolute allowance of the same relative size.
    dtype = np.promote_types(actual.dtype, expected.dtype)
    actual = np.asarray(actual, dtype=dtype)
    expected = np.asarray(expected, dtype=dtype)
    assert_allclose(actual, expected, rtol=rtol, atol=rtol * np.abs(expected).max())


def _assert_stationary_var(coefficients, k_endog, order):
    # The coefficients of a stationary VAR give a companion matrix whose
    # eigenvalues are all inside the unit circle
    coefficients = np.real(coefficients).astype(np.float64)
    polynomial = [1] + [
        -np.squeeze(coefficients[:, i * k_endog : (i + 1) * k_endog])
        for i in range(order)
    ]
    eigenvalues = np.linalg.eigvals(tools.companion_matrix(polynomial))
    assert_array_less(np.abs(eigenvalues), 1)


# Input dtype, scale of the unconstrained draws and rtol for comparing
# constrain_stationary_multivariate_python with the compiled
# constrain_stationary_multivariate. Over 200 draws for each k_endog 1-4,
# order 1-3 and transform_variance, the largest difference relative to the
# largest entry was 2e-13 for standard normal draws in double precision
# (float64 or complex128). Scaled by 10 it was 3.4e-7: large unconstrained
# values give partial autocorrelations with singular values close to one, and
# the variance recursion becomes ill-conditioned. For float32 input, which the
# compiled routine computes in single precision, it was 3.1e-5. Float32 input
# is not scaled up because single precision cannot resolve singular values
# that close to one.
TWIN_CASES = [
    pytest.param(np.float64, 1.0, 1e-10, id="float64"),
    pytest.param(np.float64, 10.0, 1e-5, id="float64-large"),
    pytest.param(np.complex128, 1.0, 1e-10, id="complex128"),
    pytest.param(np.complex128, 10.0, 1e-5, id="complex128-large"),
    pytest.param(np.float32, 1.0, 1e-3, id="float32"),
    pytest.param(np.complex64, 1.0, 1e-3, id="complex64"),
]


@pytest.mark.parametrize("dtype, scale, rtol", TWIN_CASES)
@pytest.mark.parametrize("transform_variance", [True, False])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("k_endog", [1, 2, 3, 4])
def test_constrain_stationary_multivariate_python_twin(
    k_endog, order, transform_variance, dtype, scale, rtol
):
    # constrain_stationary_multivariate_python is the pure-Python twin of the
    # compiled constrain_stationary_multivariate: both must return the same
    # coefficients and variance, and coefficients of a stationary VAR.
    # Complex input has a zero imaginary part; these routines are not
    # complex-step safe (the Cholesky factorizations conjugate).
    rng = np.random.default_rng([k_endog, order, transform_variance, int(scale)])
    for _ in range(3):
        unconstrained = scale * rng.standard_normal((k_endog, k_endog * order))
        factor = rng.standard_normal((k_endog, k_endog))
        variance = factor @ factor.T + k_endog * np.eye(k_endog)
        unconstrained = unconstrained.astype(dtype)
        variance = variance.astype(dtype)

        expected = tools.constrain_stationary_multivariate(
            unconstrained, variance, transform_variance
        )
        actual = tools.constrain_stationary_multivariate_python(
            unconstrained, variance, transform_variance
        )
        # Coefficients, then variance
        _assert_twins_agree(actual[0], expected[0], rtol)
        _assert_twins_agree(actual[1], expected[1], rtol)
        _assert_stationary_var(actual[0], k_endog, order)
        _assert_stationary_var(expected[0], k_endog, order)


@pytest.mark.parametrize("transform_variance", [True, False])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("k_endog", [1, 2, 3, 4])
def test_constrain_stationary_multivariate_python_twin_list(
    k_endog, order, transform_variance
):
    # A list of coefficient matrices gives a list back, with the blocks of the
    # result for the horizontally concatenated matrices
    rng = np.random.default_rng([k_endog, order, transform_variance])
    unconstrained = rng.standard_normal((k_endog, k_endog * order))
    blocks = [
        unconstrained[:, i * k_endog : (i + 1) * k_endog] for i in range(order)
    ]
    variance = np.eye(k_endog)

    from_array, _ = tools.constrain_stationary_multivariate_python(
        unconstrained, variance, transform_variance
    )
    actual, actual_variance = tools.constrain_stationary_multivariate_python(
        blocks, variance, transform_variance
    )
    expected, expected_variance = tools.constrain_stationary_multivariate(
        blocks, variance, transform_variance
    )
    assert isinstance(actual, list)
    assert len(actual) == order
    for i in range(order):
        block = from_array[:, i * k_endog : (i + 1) * k_endog]
        assert_allclose(actual[i], block, rtol=1e-14)
        _assert_twins_agree(actual[i], expected[i], 1e-10)
    _assert_twins_agree(actual_variance, expected_variance, 1e-10)


def test_reorder_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 3

    missing = np.zeros((k_endog, nobs))
    given = np.zeros((k_endog, k_states, nobs))
    given[:, :, :] = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])[
        :, :, np.newaxis
    ]
    desired = given.copy()

    missing[0, 0] = 1
    given[:, :, 0] = np.array([[21, 22, 23], [31, 32, 33], [0, 0, 0]])
    desired[0, :, 0] = 0

    missing[:2, 1] = 1
    given[:, :, 1] = np.array([[31, 32, 33], [0, 0, 0], [0, 0, 0]])
    desired[:2, :, 1] = 0

    missing[0, 2] = 1
    missing[2, 2] = 1
    given[:, :, 2] = np.array([[21, 22, 23], [0, 0, 0], [0, 0, 0]])
    desired[0, :, 2] = 0
    desired[2, :, 2] = 0

    missing[1, 3] = 1
    given[:, :, 3] = np.array([[11, 12, 13], [31, 32, 33], [0, 0, 0]])
    desired[1, :, 3] = 0

    missing[2, 4] = 1
    given[:, :, 4] = np.array([[11, 12, 13], [21, 22, 23], [0, 0, 0]])
    desired[2, :, 4] = 0

    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing, True, False, False, inplace=True)

    assert_equal(actual, desired)


def test_reorder_matrix_cols():
    nobs = 5
    k_endog = 3
    k_states = 3

    missing = np.zeros((k_endog, nobs))
    given = np.zeros((k_endog, k_states, nobs))
    given[:, :, :] = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])[
        :, :, np.newaxis
    ]
    desired = given.copy()

    missing[0, 0] = 1
    given[:, :, :] = np.array([[12, 13, 0], [22, 23, 0], [32, 33, 0]])[:, :, np.newaxis]
    desired[:, 0, 0] = 0

    missing[:2, 1] = 1
    given[:, :, 1] = np.array([[13, 0, 0], [23, 0, 0], [33, 0, 0]])
    desired[:, :2, 1] = 0

    missing[0, 2] = 1
    missing[2, 2] = 1
    given[:, :, 2] = np.array([[12, 0, 0], [22, 0, 0], [32, 0, 0]])
    desired[:, 0, 2] = 0
    desired[:, 2, 2] = 0

    missing[1, 3] = 1
    given[:, :, 3] = np.array([[11, 13, 0], [21, 23, 0], [31, 33, 0]])
    desired[:, 1, 3] = 0

    missing[2, 4] = 1
    given[:, :, 4] = np.array([[11, 12, 0], [21, 22, 0], [31, 32, 0]])
    desired[:, 2, 4] = 0

    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing, False, True, False, inplace=True)

    assert_equal(actual[:, :, 4], desired[:, :, 4])


def test_reorder_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    given = np.zeros((k_endog, k_endog, nobs))
    given[:, :, :] = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])[
        :, :, np.newaxis
    ]
    desired = given.copy()

    given[:, :, 0] = np.array([[22, 23, 0], [32, 33, 0], [0, 0, 0]])
    desired[0, :, 0] = 0
    desired[:, 0, 0] = 0

    given[:, :, 1] = np.array([[33, 0, 0], [0, 0, 0], [0, 0, 0]])
    desired[:2, :, 1] = 0
    desired[:, :2, 1] = 0

    given[:, :, 2] = np.array([[22, 0, 0], [0, 0, 0], [0, 0, 0]])
    desired[0, :, 2] = 0
    desired[:, 0, 2] = 0
    desired[2, :, 2] = 0
    desired[:, 2, 2] = 0

    given[:, :, 3] = np.array([[11, 13, 0], [31, 33, 0], [0, 0, 0]])
    desired[1, :, 3] = 0
    desired[:, 1, 3] = 0

    given[:, :, 4] = np.array([[11, 12, 0], [21, 22, 0], [0, 0, 0]])
    desired[2, :, 4] = 0
    desired[:, 2, 4] = 0

    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing, True, True, False, inplace=True)

    assert_equal(actual, desired)


def test_reorder_diagonal_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    given = np.zeros((k_endog, k_endog, nobs))
    given[:, :, :] = np.array([[11, 00, 00], [00, 22, 00], [00, 00, 33]])[
        :, :, np.newaxis
    ]
    desired = given.copy()

    given[:, :, 0] = np.array([[22, 00, 0], [00, 33, 0], [0, 0, 0]])
    desired[0, :, 0] = 0
    desired[:, 0, 0] = 0

    given[:, :, 1] = np.array([[33, 0, 0], [0, 0, 0], [0, 0, 0]])
    desired[:2, :, 1] = 0
    desired[:, :2, 1] = 0

    given[:, :, 2] = np.array([[22, 0, 0], [0, 0, 0], [0, 0, 0]])
    desired[0, :, 2] = 0
    desired[:, 0, 2] = 0
    desired[2, :, 2] = 0
    desired[:, 2, 2] = 0

    given[:, :, 3] = np.array([[11, 00, 0], [00, 33, 0], [0, 0, 0]])
    desired[1, :, 3] = 0
    desired[:, 1, 3] = 0

    given[:, :, 4] = np.array([[11, 00, 0], [00, 22, 0], [0, 0, 0]])
    desired[2, :, 4] = 0
    desired[:, 2, 4] = 0

    actual = np.asfortranarray(given.copy())
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing, True, True, False, inplace=True)
    assert_equal(actual, desired)

    actual = np.asfortranarray(given.copy())
    tools.reorder_missing_matrix(actual, missing, True, True, True, inplace=True)
    assert_equal(actual, desired)


def test_reorder_vector():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    given = np.zeros((k_endog, nobs))
    given[:, :] = np.array([1, 2, 3])[:, np.newaxis]
    desired = given.copy()

    given[:, 0] = [2, 3, 0]
    desired[:, 0] = [0, 2, 3]
    given[:, 1] = [3, 0, 0]
    desired[:, 1] = [0, 0, 3]
    given[:, 2] = [2, 0, 0]
    desired[:, 2] = [0, 2, 0]
    given[:, 3] = [1, 3, 0]
    desired[:, 3] = [1, 0, 3]
    given[:, 4] = [1, 2, 0]
    desired[:, 4] = [1, 2, 0]

    actual = np.asfortranarray(given.copy())
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_vector(actual, missing, inplace=True)
    assert_equal(actual, desired)


def test_copy_missing_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 2

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, k_states, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :, t] = 1.0
    B = np.zeros((k_endog, k_states, nobs), order="F")

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, False, False, inplace=True)
    assert_equal(B, A)


def test_copy_missing_matrix_cols():
    nobs = 5
    k_endog = 3
    k_states = 2

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_states, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:, :n, t] = 1.0
    B = np.zeros((k_states, k_endog, nobs), order="F")

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, False, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_missing_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :n, t] = 1.0
    B = np.zeros((k_endog, k_endog, nobs), order="F")

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_missing_diagonal_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :n, t] = np.eye(n)
    B = np.zeros((k_endog, k_endog, nobs), order="F")

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, True, False, inplace=True)
    assert_equal(B, A)

    B = np.zeros((k_endog, k_endog, nobs), order="F")
    tools.copy_missing_matrix(A, B, missing, True, True, True, inplace=True)
    assert_equal(B, A)


def test_copy_missing_vector():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, t] = 1.0
    B = np.zeros((k_endog, nobs), order="F")

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_vector(A, B, missing, inplace=True)
    assert_equal(B, A)


def test_copy_index_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 2

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, k_states, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, :, t] = 1.0
    B = np.zeros((k_endog, k_states, nobs), order="F")

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, True, False, False, inplace=True)
    assert_equal(B, A)


def test_copy_index_matrix_cols():
    nobs = 5
    k_endog = 3
    k_states = 2

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_states, k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[:, i, t] = 1.0
    B = np.zeros((k_states, k_endog, nobs), order="F")

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, False, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_index_submatrix():
    nobs = 5
    k_endog = 3

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, :, t] = 1.0
                A[:, i, t] = 1.0
    B = np.zeros((k_endog, k_endog, nobs), order="F")

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, True, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_index_diagonal_submatrix():
    nobs = 5
    k_endog = 3

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, i, t] = 1.0
    B = np.zeros((k_endog, k_endog, nobs), order="F")

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, True, True, False, inplace=True)
    assert_equal(B, A)

    B = np.zeros((k_endog, k_endog, nobs), order="F")
    tools.copy_index_matrix(A, B, index, True, True, True, inplace=True)
    assert_equal(B, A)


def test_copy_index_vector():
    nobs = 5
    k_endog = 3

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, t] = 1.0
    B = np.zeros((k_endog, nobs), order="F")

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_vector(A, B, index, inplace=True)
    assert_equal(B, A)


@pytest.mark.parametrize("trend", [[0, 1, 1], (1, 0, 1), [1]])
def test_prepare_trend_spec_array(trend):
    # GH 9145, a polynomial trend given as an ndarray raised ValueError
    expected = tools.prepare_trend_spec(trend)
    actual = tools.prepare_trend_spec(np.array(trend))
    assert_equal(actual[0], expected[0])
    assert_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "trend",
    [
        np.array(["c", "t"]),
        ["c", "t"],
        np.array([b"c", b"t"]),
        [None, 1],
        [[1, 0], [0, 1]],
        [np.nan, 1],
        [np.inf, 1],
        np.array([1j, 1]),
    ],
    ids=["str array", "str list", "bytes", "None", "2d", "nan", "inf", "complex"],
)
def test_prepare_trend_spec_invalid_array(trend):
    # a trend that is not a one-dimensional array of numbers used to raise a
    # UFuncTypeError, or to be accepted and give a trend with the wrong terms
    with pytest.raises(ValueError, match="Valid trend inputs are"):
        tools.prepare_trend_spec(trend)


@pytest.mark.parametrize("dtype", [bool, np.int64, np.intc])
def test_missing_index_dtype(dtype):
    # GH 10770: masks that were not C int raised a buffer dtype error
    rng = np.random.default_rng(10770)
    a = rng.standard_normal((2, 2, 10))
    b = rng.standard_normal((2, 2, 10))
    mask = rng.uniform(size=(2, 10)) < 0.5
    expected = tools.reorder_missing_matrix(a, mask.astype(np.intc), True)
    actual = tools.reorder_missing_matrix(a, mask.astype(dtype), True)
    assert_equal(actual, expected)
    expected = tools.copy_missing_vector(a[0], b[0], mask.astype(np.intc))
    actual = tools.copy_missing_vector(a[0], b[0], mask.astype(dtype))
    assert_equal(actual, expected)
    expected = tools.copy_index_matrix(a, b, mask.astype(np.intc), True)
    actual = tools.copy_index_matrix(a, b, mask.astype(dtype), True)
    assert_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs", [{}, {"compatibility": None}, {"compatibility": False}]
)
def test_set_mode_off(kwargs):
    with pytest.warns(FutureWarning, match=r"tools\.set_mode is deprecated"):
        assert tools.set_mode(**kwargs) is None


@pytest.mark.parametrize("compatibility", [True, 1])
def test_set_mode_compatibility_raises(compatibility):
    with (
        pytest.warns(FutureWarning, match=r"tools\.set_mode is deprecated"),
        pytest.raises(NotImplementedError, match=r"statsmodels <= 0\.9"),
    ):
        tools.set_mode(compatibility)


def _mask_results(mask):
    rng = np.random.default_rng(10795)
    a = rng.standard_normal((2, 2, 10))
    b = rng.standard_normal((2, 2, 10))
    return [
        tools.reorder_missing_matrix(a, mask, True),
        tools.reorder_missing_vector(a[0], mask),
        tools.copy_missing_matrix(a, b, mask, True),
        tools.copy_missing_vector(a[0], b[0], mask),
        tools.copy_index_matrix(a, b, mask, True),
        tools.copy_index_vector(a[0], b[0], mask),
    ]


def test_missing_index_nan_mask():
    # NaN is nonzero, so it is True. Cast to int it is undefined, and the
    # Cython code used it as an index and crashed the interpreter.
    rng = np.random.default_rng(10795)
    mask = rng.uniform(size=(2, 10)) < 0.5
    nan_mask = np.where(mask, np.nan, 0.0)
    for actual, expected in zip(
        _mask_results(nan_mask), _mask_results(mask), strict=True
    ):
        assert_equal(actual, expected)


@pytest.mark.parametrize(
    "make_mask",
    [
        lambda m: m.astype(np.uint8),
        lambda m: m.astype(np.float64),
        lambda m: 2 * m.astype(np.int64),
        lambda m: np.asfortranarray(m.astype(np.int64)),
        lambda m: np.repeat(m, 2, axis=1)[:, ::2],
    ],
    ids=["uint8", "float01", "int64-two", "int64-fortran", "strided"],
)
def test_missing_index_values(make_mask):
    # any nonzero value of a mask is True, as for a bool mask
    mask = np.random.default_rng(10795).uniform(size=(2, 10)) < 0.5
    for actual, expected in zip(
        _mask_results(make_mask(mask)), _mask_results(mask), strict=True
    ):
        assert_equal(actual, expected)


COMPANION_C0 = np.array([[2.0, 0.5], [0.0, 1.0]])
COMPANION_C1 = np.array([[0.3, 0.1], [0.2, 0.4]])
COMPANION_C2 = np.array([[-0.1, 0.05], [0.0, 0.2]])


@pytest.mark.parametrize(
    "polynomial, desired",
    [
        # An ndarray leading matrix in a list or tuple (it raised "truth value
        # is ambiguous"), against nested lists
        (
            [COMPANION_C0, COMPANION_C1, COMPANION_C2],
            [COMPANION_C0.tolist(), COMPANION_C1, COMPANION_C2],
        ),
        ((COMPANION_C0, COMPANION_C1), [COMPANION_C0.tolist(), COMPANION_C1]),
        ([np.eye(2), COMPANION_C1], [1, COMPANION_C1]),
        # A 3-dimensional array (it raised a broadcasting error)
        (
            np.array([COMPANION_C0, COMPANION_C1, COMPANION_C2]),
            [COMPANION_C0.tolist(), COMPANION_C1, COMPANION_C2],
        ),
        (np.array([np.eye(2), COMPANION_C1]), [1, COMPANION_C1]),
        (np.array([[[2.0]], [[-0.6]], [[0.4]]]), [2.0, -0.6, 0.4]),
    ],
)
def test_companion_matrix_arrays(polynomial, desired):
    actual = tools.companion_matrix(polynomial)
    assert_allclose(actual, tools.companion_matrix(desired), rtol=1e-15)

    # The eigenvalues z of the companion matrix solve
    # det(C_0 z^p + C_1 z^(p-1) + ... + C_p) = 0
    coefficients = np.asarray(desired[1:], dtype=float)
    leading = desired[0]
    order = len(coefficients)
    for z in np.linalg.eigvals(actual):
        if np.ndim(coefficients) == 1:
            value = leading * z**order + np.polyval(coefficients, z)
            scale = abs(leading * z**order)
        else:
            c0 = np.eye(2) if np.ndim(leading) == 0 else np.asarray(leading)
            value = np.linalg.det(
                c0 * z**order
                + sum(c * z ** (order - 1 - i) for i, c in enumerate(coefficients))
            )
            scale = abs(np.linalg.det(c0 * z**order))
        # atol: the determinant is zero up to rounding
        assert_allclose(abs(value), 0, atol=1e-12 * max(scale, 1.0))
