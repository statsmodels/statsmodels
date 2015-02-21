from statsmodels.tools import linalg
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz

def test_durbin():
    np.random.seed(3424)
    rhs = np.random.uniform(size=8)
    t = np.concatenate((np.r_[1], rhs[0:-1]))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, rhs)
    soln1 = linalg.durbin(rhs)
    assert_allclose(soln, soln1[-1])


def test_toeplitz_solve():
    b = np.random.uniform(size=10)
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.toeplitz_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)
