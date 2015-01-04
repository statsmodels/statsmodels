"""
Statespace Tools

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from statsmodels.tools.data import _is_using_pandas
from . import _statespace

try:
    from scipy.linalg.blas import find_best_blas_type
except ImportError:
    # Shim for SciPy 0.11, derived from tag=0.11 scipy.linalg.blas
    _type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z', 'G': 'z'}

    def find_best_blas_type(arrays):
        dtype, index = max(
            [(ar.dtype, i) for i, ar in enumerate(arrays)])
        prefix = _type_conv.get(dtype.char, 'd')
        return (prefix, dtype, None)


prefix_dtype_map = {
    's': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128
}
prefix_statespace_map = {
    's': _statespace.sStatespace, 'd': _statespace.dStatespace,
    'c': _statespace.cStatespace, 'z': _statespace.zStatespace
}
prefix_kalman_filter_map = {
    's': _statespace.sKalmanFilter, 'd': _statespace.dKalmanFilter,
    'c': _statespace.cKalmanFilter, 'z': _statespace.zKalmanFilter
}


def companion_matrix(n, values=None):
    r"""
    Create a companion matrix

    Returns a matrix of the form

    .. math::
        \begin{bmatrix}
            \phi_1 & 1      & 0 & \cdots & 0 \\
            \phi_2 & 0      & 1 &        & 0 \\
            \vdots &        &   & \ddots & 0 \\
                   &        &   &        & 1 \\
            \phi_n & 0      & 0 & \cdots & 0 \\
        \end{bmatrix}

    where some or all of the :math:`\phi_i` may be non-zero (if `values` is
    None, then all are equal to zero).

    Parameters
    ----------
    n : int
        The size of the companion matrix.
    values : array_like, optional.
        The values to use in the first column of the companion matrix. Default
        is zeros.

    Returns
    -------
    companion_matrix : array
    """
    matrix = np.zeros((n, n))
    idx = np.diag_indices(n-1)
    idx = (idx[0], idx[1]+1)
    matrix[idx] = 1
    if values is not None:
        matrix[:, 0] = values
    return matrix


def diff(series, diff=1, seasonal_diff=None, k_seasons=1):
    """
    Difference a series simply and / or seasonally along the zero-th axis.

    Given a series (denoted :math:`y_t`), performs the differencing operation

    .. math::

        \Delta^d \Delta_s^D y_t

    where :math:`d = diff`, :math:`s = k\_seasons`, :math:`D = seasonal\_diff`,
    and :math:`\Delta` is the lag operator.

    Parameters
    ----------
    series : array_like
        The series to be differenced.
    diff : int, optional
        The number of simple differences to perform. Default is 1.
    seasonal_diff : int or None, optional
        The number of seasonal differences to perform. Default is no seasonal
        differencing.
    k_seasons : int, optional
        The seasonal lag. Default is 1. Unused if there is no seasonal
        differencing.

    Returns
    -------
    differenced : array
        The differenced array.
    """
    pandas = _is_using_pandas(series, None)
    differenced = np.asanyarray(series) if not pandas else series

    # Seasonal differencing
    if seasonal_diff is not None:
        while seasonal_diff > 0:
            differenced = differenced[k_seasons:] - differenced[:-k_seasons]
            seasonal_diff -= 1

    # Simple differencing
    if not pandas:
        differenced = np.diff(differenced, diff, axis=0)
    else:
        differenced = differenced.diff(diff)[diff:]
    return differenced


def is_invertible(params):
    """
    Determine if a set of parameters (corresponding to the coefficients to the
    non-constant terms of a polynomial) represents an invertible lag
    polynomial. Requires all roots of the polynomial to lie outside the unit
    circle.

    Parameters
    ----------
    params : array_like
        Coefficients of the non-constant terms of a polynomial. For example,
        `params=[0.5]` corresponds to the polynomial :math:`1 + 0.5x` which
        has root :math:`-2`.

    Examples
    --------
    >>> dp.ssm.is_invertible([0.5])
    True
    >>> dp.ssm.is_invertible([1])
    False
    """
    return np.all(np.abs(np.roots(np.r_[1, params][::-1])) > 1)


def constrain_stationary_univariate(unconstrained):
    """
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation

    Parameters
    ----------
    unconstrained : array_like
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    Returns
    -------
    constrained : array_like
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    References
    ----------

    Monahan, John F. 1984.
    "A Note on Enforcing Stationarity in
    Autoregressive-moving Average Models."
    Biometrika 71 (2) (August 1): 403-404.
    """

    n = unconstrained.shape[0]
    y = np.zeros((n, n), dtype=unconstrained.dtype)
    r = unconstrained/((1+unconstrained**2)**0.5)
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k-1, i] + r[k] * y[k-1, k-i-1]
        y[k, k] = r[k]
    return -y[n-1, :]


def unconstrain_stationary_univariate(constrained):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : array_like
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    Returns
    -------
    unconstrained : array_like
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    References
    ----------

    Monahan, John F. 1984.
    "A Note on Enforcing Stationarity in
    Autoregressive-moving Average Models."
    Biometrika 71 (2) (August 1): 403-404.
    """
    n = constrained.shape[0]
    y = np.zeros((n, n), dtype=constrained.dtype)
    y[n-1:] = -constrained
    for k in range(n-1, 0, -1):
        for i in range(k):
            y[k-1, i] = (y[k, i] - y[k, k]*y[k, k-i-1]) / (1 - y[k, k]**2)
    r = y.diagonal()
    x = r / ((1 - r**2)**0.5)
    return x


def validate_matrix_shape(name, shape, nrows, ncols, nobs):
    """
    Validate the shape of a possibly time-varing matrix, or raise an exception

    Parameters
    ----------
    name : str
        The name of the matrix being validated (used in exception messages)
    shape : array_like
        The shape of the matrix to be validated. May be of size 2 or (if
        the matrix is time-varying) 3.
    nrows : int
        The expected number of rows.
    ncols : int
        The expected number of columns.
    nobs : int
        The number of observations (used to validate the last dimension of a
        time-varying matrix)

    Raises
    ------
    ValueError
        If the matrix is not of the desired shape.
    """
    ndim = len(shape)

    # Enforce dimension
    if ndim not in [2, 3]:
        raise ValueError('Invalid value for %s matrix. Requires a'
                         ' 2- or 3-dimensional array, got %d dimensions' %
                         (name, ndim))
    # Enforce the shape of the matrix
    if not shape[0] == nrows:
        raise ValueError('Invalid dimensions for %s matrix: requires %d'
                         ' rows, got %d' % (name, nrows, shape[0]))
    if not shape[1] == ncols:
        raise ValueError('Invalid dimensions for %s matrix: requires %d'
                         ' columns, got %d' % (name, ncols, shape[1]))

    # If we don't yet know `nobs`, don't allow time-varying arrays
    if nobs is None and not (ndim == 2 or shape[-1] == 1):
        raise ValueError('Time-varying state space matrices cannot be given'
                         ' unless `nobs` is specified (implicitly when a'
                         ' dataset is bound or else set explicity)')

    # Enforce time-varying array size
    if ndim == 3 and nobs is not None and not shape[-1] in [1, nobs]:
        raise ValueError('Invalid dimensions for time-varying %s'
                         ' matrix. Requires shape (*,*,%d), got %s' %
                         (name, nobs, str(shape)))


def validate_vector_shape(name, shape, nrows, nobs):
    """
    Validate the shape of a possibly time-varing vector, or raise an exception

    Parameters
    ----------
    name : str
        The name of the vector being validated (used in exception messages)
    shape : array_like
        The shape of the vector to be validated. May be of size 1 or (if
        the vector is time-varying) 2.
    nrows : int
        The expected number of rows (elements of the vector).
    nobs : int
        The number of observations (used to validate the last dimension of a
        time-varying vector)

    Raises
    ------
    ValueError
        If the vector is not of the desired shape.
    """
    ndim = len(shape)
    # Enforce dimension
    if ndim not in [1, 2]:
        raise ValueError('Invalid value for %s vector. Requires a'
                         ' 1- or 2-dimensional array, got %d dimensions' %
                         (name, ndim))
    # Enforce the shape of the vector
    if not shape[0] == nrows:
        raise ValueError('Invalid dimensions for %s vector: requires %d'
                         ' rows, got %d' % (name, nrows, shape[0]))

    # If we don't yet know `nobs`, don't allow time-varying arrays
    if nobs is None and not (ndim == 1 or shape[-1] == 1):
        raise ValueError('Time-varying state space vectors cannot be given'
                         ' unless `nobs` is specified (implicitly when a'
                         ' dataset is bound or else set explicity)')

    # Enforce time-varying array size
    if ndim == 2 and not shape[1] in [1, nobs]:
        raise ValueError('Invalid dimensions for time-varying %s'
                         ' vector. Requires shape (*,%d), got %s' %
                         (name, nobs, str(shape)))
