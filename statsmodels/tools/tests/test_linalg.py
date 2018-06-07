from statsmodels.tools import linalg
from statsmodels.tools.linalg import _smw_solver, _smw_logdet
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz
from scipy import sparse


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


def test_smw_solver():

    np.random.seed(23)

    def tester(p, q, r, s):

        d = q - r

        A = np.random.normal(size=(p, q))
        AtA = np.dot(A.T, A)

        B = np.zeros((q, q))
        B[0:r, 0:r] = np.random.normal(size=(r, r))
        di = np.random.uniform(size=d)
        B[r:q, r:q] = np.diag(1 / di)
        Qi = np.linalg.inv(B[0:r, 0:r])
        s = 0.5

        x = np.random.normal(size=p)
        y2 = np.linalg.solve(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)), x)

        f = _smw_solver(s, A, AtA, Qi, di)
        y1 = f(x)
        assert_allclose(y1, y2)

        f = _smw_solver(s, sparse.csr_matrix(A), sparse.csr_matrix(AtA), Qi,
                        di)
        y1 = f(x)
        assert_allclose(y1, y2)

    for p in (5, 10):
        for q in (4, 8):
            for r in (2, 3):
                for s in (0, 0.5):
                    tester(p, q, r, s)


def test_smw_logdet():

    np.random.seed(23)

    def tester(p, q, r, s):

        d = q - r
        A = np.random.normal(size=(p, q))
        AtA = np.dot(A.T, A)

        B = np.zeros((q, q))
        c = np.random.normal(size=(r, r))
        B[0:r, 0:r] = np.dot(c.T, c)
        di = np.random.uniform(size=d)
        B[r:q, r:q] = np.diag(1 / di)
        Qi = np.linalg.inv(B[0:r, 0:r])
        s = 0.5

        _, d2 = np.linalg.slogdet(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)))

        _, bd = np.linalg.slogdet(B)
        d1 = _smw_logdet(s, A, AtA, Qi, di, bd)
        assert_allclose(d1, d2)

    for p in (5, 10):
        for q in (4, 8):
            for r in (2, 3):
                for s in (0, 0.5):
                    tester(p, q, r, s)
