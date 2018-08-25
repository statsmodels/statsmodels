from statsmodels.tools import linalg
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz


def test_stationary_solve_1d():
    b = np.random.uniform(size=10)
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)


def test_stationary_solve_2d():
    b = np.random.uniform(size=(10, 2))
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)


def test_scipy_equivalence():
    # This test was moved from the __main__ section of tools.linalg
    # Note on Windows32:
    #    linalg doesn't always produce the same results in each call
    import scipy.linalg
    a0 = np.random.randn(100, 10)
    b0 = a0.sum(1)[:, None] + np.random.randn(100, 3)

    result = linalg.pinv(a0)
    expected = scipy.linalg.pinv(a0)
    assert_allclose(result, expected)

    result = linalg.pinv2(a0)
    expected = scipy.linalg.pinv2(a0)
    assert_allclose(result, expected)

    result = linalg.lstsq(a0, b0)
    expected = scipy.linalg.lstsq(a0, b0)
    assert_allclose(result[0], expected[0])
    assert_allclose(result[1], expected[1])
    assert_allclose(result[2], expected[2])
    assert_allclose(result[3], expected[3])
