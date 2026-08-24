"""Utility functions used by statsmodels models"""

import numpy as np
import pandas as pd
import scipy.linalg

from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like


def asstr2(s):
    """
    Return s as a text string

    Parameters
    ----------
    s : str, bytes or object
        Value to convert to a text string. Bytes are decoded using the
        latin1 codec; any other type is converted with ``str``.

    Returns
    -------
    str
        `s` as a text string.
    """
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode("latin1")
    else:
        return str(s)


def drop_missing(y, x=None, axis=1):
    """
    Return views on the arrays y and x where missing observations are dropped

    Parameters
    ----------
    y : array_like
        Data with observations possibly containing NaN values.
    x : array_like, optional
        Additional data with observations possibly containing NaN
        values. If provided, an observation is dropped if it is
        missing in either `y` or `x`.
    axis : int, optional
        Axis along which to look for missing observations.  Default is 1, i.e.,
        observations in rows.

    Returns
    -------
    y : ndarray
        `y` with the rows (or columns) containing missing observations
        removed.
    x : ndarray
        `x` with the rows (or columns) containing missing observations
        removed. Only returned if `x` is not None.

    Notes
    -----
    If either y or x is 1d, it is reshaped to be 2d.

    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    # ``axis`` is reduced when looking for missing values, so the observations
    # that are kept must be selected along the other axis.
    keep_axis = 1 - axis
    if x is not None:
        x = np.array(x)
        if x.ndim == 1:
            x = x[:, None]
        keepidx = np.logical_and(~np.isnan(y).any(axis), ~np.isnan(x).any(axis))
        return (
            np.compress(keepidx, y, axis=keep_axis),
            np.compress(keepidx, x, axis=keep_axis),
        )
    else:
        keepidx = ~np.isnan(y).any(axis)
        return np.compress(keepidx, y, axis=keep_axis)


# TODO: add an axis argument to this for sysreg
def add_constant(data, prepend=True, has_constant="skip"):
    """
    Add a column of ones to an array

    Parameters
    ----------
    data : array_like
        A column-ordered design matrix.
    prepend : bool, optional
        If True (default), the constant is in the first column. If False, the
        constant is appended (last column).
    has_constant : {'raise', 'add', 'skip'}, optional
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if any column has a constant value. Using 'add' will add a
        column of 1s if a constant column is present.

    Returns
    -------
    array_like
        The original values with a constant (column of ones) as the first or
        last column. Returned value type depends on input type.

    Notes
    -----
    When the input is a pandas Series or DataFrame, the added column's name
    is 'const'.

    """
    if _is_using_pandas(data, None):
        from statsmodels.tsa.tsatools import add_trend

        return add_trend(data, trend="c", prepend=prepend, has_constant=has_constant)

    # Special case for NumPy
    x = np.asarray(data)
    ndim = x.ndim
    if ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError("Only implemented for 2-dimensional arrays")

    is_nonzero_const = np.ptp(x, axis=0) == 0
    is_nonzero_const &= np.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == "skip":
            return x
        elif has_constant == "raise":
            if ndim == 1:
                raise ValueError("data is constant.")
            else:
                columns = np.arange(x.shape[1])
                cols = ",".join([str(c) for c in columns[is_nonzero_const]])
                raise ValueError(f"Column(s) {cols} are constant.")

    x = [np.ones(x.shape[0]), x]
    x = x if prepend else x[::-1]
    return np.column_stack(x)


def isestimable(c, d):
    """
    True if (Q, P) contrast `c` is estimable for (N, P) design `d`

    From an Q x P contrast matrix `C` and an N x P design matrix `D`, checks if
    the contrast `C` is estimable by looking at the rank of ``vstack([C,D])``
    and verifying it is the same as the rank of `D`.

    Parameters
    ----------
    c : array_like
        A contrast matrix with shape (Q, P). If 1 dimensional assume shape is
        (1, P).
    d : array_like
        The design matrix, (N, P).

    Returns
    -------
    bool
        True if the contrast `c` is estimable on design `d`.

    Examples
    --------
    >>> d = np.array([[1, 1, 1, 0, 0, 0],
    ...               [0, 0, 0, 1, 1, 1],
    ...               [1, 1, 1, 1, 1, 1]]).T
    >>> isestimable([1, 0, 0], d)
    False
    >>> isestimable([1, -1, 0], d)
    True

    """
    c = array_like(c, "c", maxdim=2)
    d = array_like(d, "d", ndim=2)
    c = c[None, :] if c.ndim == 1 else c
    if c.shape[1] != d.shape[1]:
        raise ValueError(f"Contrast should have {d.shape[1]:d} columns")
    new = np.vstack([c, d])
    if np.linalg.matrix_rank(new) != np.linalg.matrix_rank(d):
        return False
    return True


def pinv_extended(x, rcond=1e-15):
    """
    Return the pinv of an array x as well as the singular values used in computation

    Parameters
    ----------
    x : array_like
        The array to invert, 2d.
    rcond : float, optional
        Singular values below ``rcond * max(singular values)`` are
        treated as zero.

    Returns
    -------
    ndarray
        The pseudo-inverse of `x`.
    ndarray
        The singular values of `x` used in computing the pseudo-inverse.

    Notes
    -----
    Code adapted from numpy.
    """
    x = np.asarray(x)
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1.0 / s[i]
        else:
            s[i] = 0.0
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))
    return res, s_orig


def recipr(x):
    """
    Reciprocal of an array with entries less than or equal to 0 set to 0

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.

    """
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=np.float64)
    nans = np.isnan(x.flat)
    pos = ~nans
    pos[pos] = pos[pos] & (x.flat[pos] > 0)
    out.flat[pos] = 1.0 / x.flat[pos]
    out.flat[nans] = np.nan
    return out


def recipr0(x):
    """
    Reciprocal of an array with entries equal to 0 set to 0

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.

    """
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=np.float64)
    nans = np.isnan(x.flat)
    non_zero = ~nans
    non_zero[non_zero] = non_zero[non_zero] & (x.flat[non_zero] != 0)
    out.flat[non_zero] = 1.0 / x.flat[non_zero]
    out.flat[nans] = np.nan
    return out


def clean0(matrix):
    """
    Erase columns of zeros: can save some time in pseudoinverse

    Parameters
    ----------
    matrix : ndarray
        The array to clean.

    Returns
    -------
    ndarray
        The cleaned array.

    """
    colsum = np.add.reduce(matrix**2, 0)
    val = [matrix[:, i] for i in np.flatnonzero(colsum)]
    return np.array(np.transpose(val))


def fullrank(x, r=None):
    """
    Return an array whose column span is the same as x

    Parameters
    ----------
    x : ndarray
        The array to adjust, 2d.
    r : int, optional
        The rank of x. If not provided, determined by `np.linalg.matrix_rank`.

    Returns
    -------
    ndarray
        The array adjusted to have full rank.

    Notes
    -----
    If the rank of x is known it can be specified as r -- no check
    is made to ensure that this really is the rank of x.

    """
    if r is None:
        r = np.linalg.matrix_rank(x)

    v, d, u = np.linalg.svd(x, full_matrices=False)
    order = np.argsort(d)
    order = order[::-1]
    value = [v[:, order[i]] for i in range(r)]
    return np.asarray(np.transpose(value)).astype(np.float64)


def unsqueeze(data, axis, oldshape):
    """
    Unsqueeze a collapsed array

    Parameters
    ----------
    data : ndarray
        The data to unsqueeze.
    axis : int
        The axis to unsqueeze.
    oldshape : tuple[int]
        The original shape before the squeeze or reduce operation.

    Returns
    -------
    ndarray
        The unsqueezed array.

    Examples
    --------
    >>> from numpy import mean
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)
    >>>

    """
    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)


def nan_dot(A, B):
    """
    Return np.dot(A, B), preserving NaNs from nonzero products

    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A : ndarray
        Left-hand input array.
    B : ndarray
        Right-hand input array.

    Returns
    -------
    ndarray
        Dot product using the NaN convention.

    """
    # Find out who should be nan due to nan * nonzero
    should_be_nan_1 = np.dot(np.isnan(A), (B != 0))
    should_be_nan_2 = np.dot((A != 0), np.isnan(B))
    should_be_nan = should_be_nan_1 + should_be_nan_2

    # Multiply after setting all nan to 0
    # This is what happens if there were no nan * nonzero conflicts
    C = np.dot(np.nan_to_num(A), np.nan_to_num(B))

    C[should_be_nan] = np.nan

    return C


def maybe_unwrap_results(results):
    """
    Get raw results back from wrapped results

    Can be used in plotting functions or other post-estimation type
    routines.

    Parameters
    ----------
    results : Results or ResultsWrapper
        A results instance, possibly wrapped in a ResultsWrapper.

    Returns
    -------
    Results
        The underlying (unwrapped) results instance, or `results`
        itself if it was not wrapped.
    """
    return getattr(results, "_results", results)


class Bunch(dict):
    """
    Returns a dict-like object with keys accessible via attribute lookup

    Parameters
    ----------
    *args
        Arguments passed to dict constructor, tuples (key, value).
    **kwargs
        Keyword argument passed to dict constructor, key=value.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_2d(x, ndarray=False):
    """
    Ensure that an input is 2-dimensional, converting or reshaping as needed

    Parameters
    ----------
    x : ndarray, Series, or DataFrame
        Input to verify dimensions, and to transform as necessary
    ndarray : bool, optional
        Flag indicating whether to always return a NumPy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : ndarray, DataFrame
        array or DataFrame with 2 dimensions.  One dimensional arrays are
        returned as nobs by 1.
    names : list of str or None
        list containing variables names when the input is a pandas datatype.
        Returns None if the input is an ndarray.

    Notes
    -----
    Accepts None for simplicity
    """
    is_pandas = _is_using_pandas(x, None)
    if x.ndim == 2:
        if is_pandas:
            return x, x.columns
        else:
            return x, None
    elif x.ndim > 2:
        raise ValueError("x must be 1 or 2-dimensional.")

    name = x.name if is_pandas else None
    if ndarray:
        return np.asarray(x)[:, None], name
    else:
        return pd.DataFrame(x), name


def matrix_rank(m, tol=None, method="qr"):
    """
    Matrix rank calculation using QR or SVD

    Parameters
    ----------
    m : array_like
        A 2-d array-like object to test
    tol : float, optional
        The tolerance to use when testing the matrix rank. If not provided
        an appropriate value is selected.
    method : {"ip", "qr", "svd"}, optional
        The method used. "ip" uses the inner-product of a normalized version
        of m and then computes the rank using NumPy's matrix_rank.
        "qr" uses a QR decomposition and is the default. "svd" defers to
        NumPy's matrix_rank.

    Returns
    -------
    int
        The rank of m.

    Notes
    -----
    When using a QR factorization, the factorization is column pivoted and
    the rank is determined by the number of elements on the leading diagonal
    of the R matrix that are above tol in absolute value.

    """
    m = array_like(m, "m", ndim=2)
    if method == "ip":
        m = m[:, np.any(m != 0, axis=0)]
        m = m / np.sqrt((m**2).sum(0))
        m = m.T @ m
        return np.linalg.matrix_rank(m, tol=tol, hermitian=True)
    elif method == "qr":
        # The pivoted factorization orders the diagonal of R by decreasing
        # magnitude, which makes the number of diagonal elements above tol a
        # rank estimate and keeps tol keyed to the largest of them. Without
        # pivoting the count depends on the order of the columns of m.
        r, _ = scipy.linalg.qr(m, mode="r", pivoting=True)
        abs_diag = np.abs(np.diag(r))
        if tol is None:
            tol = abs_diag[0] * m.shape[1] * np.finfo(float).eps
        return int((abs_diag > tol).sum())
    else:
        return np.linalg.matrix_rank(m, tol=tol)
