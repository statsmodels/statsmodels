"""
Linear Algebra solvers and other helpers
"""
from __future__ import print_function
from statsmodels.compat.python import range
import numpy as np
from scipy.linalg import pinv, pinv2, lstsq  # noqa:F421


def logdet_symm(m, check_symm=False):
    """
    Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : array-like
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
    from scipy import linalg
    if check_symm:
        if not np.all(m == m.T):  # would be nice to short-circuit check
            raise ValueError("m is not symmetric.")
    c, _ = linalg.cho_factor(m, lower=True)
    return 2*np.sum(np.log(c.diagonal()))


def stationary_solve(r, b):
    """
    Solve a linear system for a Toeplitz correlation matrix.

    A Toeplitz correlation matrix represents the covariance of a
    stationary series with unit variance.

    Parameters
    ----------
    r : array-like
        A vector describing the coefficient matrix.  r[0] is the first
        band next to the diagonal, r[1] is the second band, etc.
    b : array-like
        The right-hand side for which we are solving, i.e. we solve
        Tx = b and return b, where T is the Toeplitz coefficient matrix.

    Returns
    -------
    The solution to the linear system.
    """

    db = r[0:1]

    dim = b.ndim
    if b.ndim == 1:
        b = b[:, None]
    x = b[0:1, :]

    for j in range(1, len(b)):
        rf = r[0:j][::-1]
        a = (b[j, :] - np.dot(rf, x)) / (1 - np.dot(rf, db[::-1]))
        z = x - np.outer(db[::-1], a)
        x = np.concatenate((z, a[None, :]), axis=0)

        if j == len(b) - 1:
            break

        rn = r[j]
        a = (rn - np.dot(rf, db)) / (1 - np.dot(rf, db[::-1]))
        z = db - a*db[::-1]
        db = np.concatenate((z, np.r_[a]))

    if dim == 1:
        x = x[:, 0]

    return x
