'''local, adjusted version from scipy.linalg.basic.py


changes:
The only changes are that additional results are returned

'''
from __future__ import print_function
from statsmodels.compat.python import lmap, range
import numpy as np
from scipy.linalg import svd as decomp_svd
from scipy.linalg.lapack import get_lapack_funcs
from numpy import asarray, zeros, sum, conjugate, dot, transpose
import numpy
from numpy import asarray_chkfinite, single
from numpy.linalg import LinAlgError


### Linear Least Squares

def lstsq(a, b, cond=None, overwrite_a=0, overwrite_b=0):
    """Compute least-squares solution to equation :m:`a x = b`

    Compute a vector x such that the 2-norm :m:`|b - a x|` is minimised.

    Parameters
    ----------
    a : array, shape (M, N)
    b : array, shape (M,) or (M, K)
    cond : float
        Cutoff for 'small' singular values; used to determine effective
        rank of a. Singular values smaller than rcond*largest_singular_value
        are considered zero.
    overwrite_a : boolean
        Discard data in a (may enhance performance)
    overwrite_b : boolean
        Discard data in b (may enhance performance)

    Returns
    -------
    x : array, shape (N,) or (N, K) depending on shape of b
        Least-squares solution
    residues : array, shape () or (1,) or (K,)
        Sums of residues, squared 2-norm for each column in :m:`b - a x`
        If rank of matrix a is < N or > M this is an empty array.
        If b was 1-d, this is an (1,) shape array, otherwise the shape is (K,)
    rank : integer
        Effective rank of matrix a
    s : array, shape (min(M,N),)
        Singular values of a. The condition number of a is abs(s[0]/s[-1]).

    Raises LinAlgError if computation does not converge

    """
    a1, b1 = lmap(asarray_chkfinite, (a, b))
    if a1.ndim != 2:
        raise ValueError('expected matrix')
    m, n = a1.shape
    if b1.ndim == 2:
        nrhs = b1.shape[1]
    else:
        nrhs = 1
    if m != b1.shape[0]:
        raise ValueError('incompatible dimensions')
    gelss, = get_lapack_funcs(('gelss',), (a1, b1))
    if n > m:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        b2 = zeros((n, nrhs), dtype=gelss.dtype)
        if b1.ndim == 2:
            b2[:m, :] = b1
        else:
            b2[:m, 0] = b1
        b1 = b2
    overwrite_a = overwrite_a or (a1 is not a and not hasattr(a, '__array__'))
    overwrite_b = overwrite_b or (b1 is not b and not hasattr(b, '__array__'))

    if gelss.module_name[:7] == 'flapack':

        # get optimal work array
        work = gelss(a1, b1, lwork=-1)[4]
        lwork = work[0].real.astype(np.int)
        v, x, s, rank, work, info = gelss(
            a1, b1, cond=cond, lwork=lwork, overwrite_a=overwrite_a,
            overwrite_b=overwrite_b)

    else:
        raise NotImplementedError('calling gelss from %s' %
                                  gelss.module_name)
    if info > 0:
        raise LinAlgError("SVD did not converge in Linear Least Squares")
    if info < 0:
        raise ValueError('illegal value in %-th argument of '
                         'internal gelss' % -info)
    resids = asarray([], dtype=x.dtype)
    if n < m:
        x1 = x[:n]
        if rank == n:
            resids = sum(x[n:]**2, axis=0)
        x = x1
    return x, resids, rank, s


def pinv(a, cond=None, rcond=None):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using a least-squares
    solver.

    Parameters
    ----------
    a : array, shape (M, N)
        Matrix to be pseudo-inverted
    cond, rcond : float
        Cutoff for 'small' singular values in the least-squares solver.
        Singular values smaller than rcond*largest_singular_value are
        considered zero.

    Returns
    -------
    B : array, shape (N, M)

    Raises LinAlgError if computation does not converge

    Examples
    --------
    >>> from numpy import *
    >>> a = random.randn(9, 6)
    >>> B = linalg.pinv(a)
    >>> allclose(a, dot(a, dot(B, a)))
    True
    >>> allclose(B, dot(B, dot(a, B)))
    True

    """
    a = asarray_chkfinite(a)
    b = numpy.identity(a.shape[0], dtype=a.dtype)
    if rcond is not None:
        cond = rcond
    return lstsq(a, b, cond=cond)[0]


eps = numpy.finfo(float).eps
feps = numpy.finfo(single).eps

_array_precision = {'f': 0, 'd': 1, 'F': 0, 'D': 1}


def pinv2(a, cond=None, rcond=None):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.

    Parameters
    ----------
    a : array, shape (M, N)
        Matrix to be pseudo-inverted
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Singular values smaller than rcond*largest_singular_value are
        considered zero.

        If None or -1, suitable machine precision is used.

    Returns
    -------
    B : array, shape (N, M)

    Raises LinAlgError if SVD computation does not converge

    Examples
    --------
    >>> from numpy import *
    >>> a = random.randn(9, 6)
    >>> B = linalg.pinv2(a)
    >>> allclose(a, dot(a, dot(B, a)))
    True
    >>> allclose(B, dot(B, dot(a, B)))
    True

    """
    a = asarray_chkfinite(a)
    u, s, vh = decomp_svd(a)
    t = u.dtype.char
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        cond = {0: feps*1e3, 1: eps*1e6}[_array_precision[t]]
    m, n = a.shape
    cutoff = cond*numpy.maximum.reduce(s)
    psigma = zeros((m, n), t)
    for i in range(len(s)):
        if s[i] > cutoff:
            psigma[i, i] = 1.0/conjugate(s[i])
    # XXX: use lapack/blas routines for dot
    return transpose(conjugate(dot(dot(u, psigma), vh)))


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
    x = b[0:1,:]

    for j in range(1, len(b)):
        rf = r[0:j][::-1]
        a = (b[j,:] - np.dot(rf, x)) / (1 - np.dot(rf, db[::-1]))
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



if __name__ == '__main__':
    #for checking only,
    #Note on Windows32:
    #    linalg doesn't always produce the same results in each call
    a0 = np.random.randn(100,10)
    b0 = a0.sum(1)[:, None] + np.random.randn(100,3)
    lstsq(a0,b0)
    pinv(a0)
    pinv2(a0)
    x = pinv(a0)
    x2=scipy.linalg.pinv(a0)
    print(np.max(np.abs(x-x2)))
    x = pinv2(a0)
    x2 = scipy.linalg.pinv2(a0)
    print(np.max(np.abs(x-x2)))

