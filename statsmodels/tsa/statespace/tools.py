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
except ImportError:  # pragma: no cover
    # Shim for SciPy 0.11, derived from tag=0.11 scipy.linalg.blas
    _type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z', 'G': 'z'}

    def find_best_blas_type(arrays):
        dtype, index = max(
            [(ar.dtype, i) for i, ar in enumerate(arrays)])
        prefix = _type_conv.get(dtype.char, 'd')
        return prefix, dtype, None


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


def companion_matrix(polynomial):
    r"""
    Create a companion matrix

    Parameters
    ----------
    polynomial : array_like or list
        If an iterable, interpreted as the coefficients of the polynomial from
        which to form the companion matrix. Polynomial coefficients are in
        order of increasing degree, and may be either scalars (as in an AR(p)
        model) or coefficient matrices (as in a VAR(p) model). If an integer,
        it is interpereted as the size of a companion matrix of a scalar
        polynomial, where the polynomial coefficients are initialized to zeros.
        If a matrix polynomial is passed, :math:`C_0` may be set to the scalar
        value 1 to indicate an identity matrix (doing so will improve the speed
        of the companion matrix creation).

    Returns
    -------
    companion_matrix : array

    Notes
    -----

    Returns a matrix of the form

    .. math::
        \begin{bmatrix}
            \phi_1 & 1      & 0 & \cdots & 0 \\
            \phi_2 & 0      & 1 &        & 0 \\
            \vdots &        &   & \ddots & 0 \\
                   &        &   &        & 1 \\
            \phi_n & 0      & 0 & \cdots & 0 \\
        \end{bmatrix}

    where some or all of the :math:`\phi_i` may be non-zero (if `polynomial` is
    None, then all are equal to zero).

    If the coefficients provided are scalars :math:`(c_0, c_1, \dots, c_{n})`,
    then the companion matrix is an :math:`n \times n` matrix formed with the
    elements in the first column defined as
    :math:`\phi_i = -\frac{c_i}{c_0}, i \in 1, \dots, n`.

    If the coefficients provided are matrices :math:`(C_0, C_1, \dots, C_{n})`,
    each of shape :math:`(m, m)`, then the companion matrix is an
    :math:`nm \times nm` matrix formed with the elements in the first column
    defined as :math:`\phi_i = -C_0^{-1} C_i', i \in 1, \dots, n`.
    """
    identity_matrix = False
    if isinstance(polynomial, int):
        n = polynomial
        polynomial = None
    else:
        n = len(polynomial) - 1

        if n < 1:
            raise ValueError("Companion matrix polynomials must include at"
                             " least two terms.")

        if isinstance(polynomial, list) or isinstance(polynomial, tuple):
            m = len(polynomial[1])

            # Check if we just have a scalar polynomial
            if m == 1:
                polynomial = np.asanyarray(polynomial)
            # Check if 1 was passed as the first argument (indicating an
            # identity matrix)
            elif polynomial[0] == 1:
                polynomial[0] = np.eye(m)
                identity_matrix = True
        else:
            m = 1
            polynomial = np.asanyarray(polynomial)

    matrix = np.zeros((n * m, n * m))
    idx = np.diag_indices((n - 1) * m)
    idx = (idx[0], idx[1] + m)
    matrix[idx] = 1
    if polynomial is not None and n > 0:
        if m == 1:
            matrix[:, 0] = -polynomial[1:] / polynomial[0]
        elif identity_matrix:
            for i in range(n):
                matrix[i * m:(i + 1) * m, :m] = polynomial[i+1].T
        else:
            inv = np.linalg.inv(polynomial[0])
            for i in range(n):
                matrix[i * m:(i + 1) * m, :m] = np.dot(inv, polynomial[i+1]).T
    return matrix


def diff(series, k_diff=1, k_seasonal_diff=None, k_seasons=1):
    r"""
    Difference a series simply and/or seasonally along the zero-th axis.

    Given a series (denoted :math:`y_t`), performs the differencing operation

    .. math::

        \Delta^d \Delta_s^D y_t

    where :math:`d =` `diff`, :math:`s =` `k_seasons`,
    :math:`D =` `seasonal\_diff`, and :math:`\Delta` is the difference
    operator.

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
    if k_seasonal_diff is not None:
        while k_seasonal_diff > 0:
            if not pandas:
                differenced = (
                    differenced[k_seasons:] - differenced[:-k_seasons]
                )
            else:
                differenced = differenced.diff(k_seasons)[k_seasons:]
            k_seasonal_diff -= 1

    # Simple differencing
    if not pandas:
        differenced = np.diff(differenced, k_diff, axis=0)
    else:
        while k_diff > 0:
            differenced = differenced.diff()[1:]
            k_diff -= 1
    return differenced


def is_invertible(polynomial, threshold=1.):
    r"""
    Determine if a polynomial is invertible.

    Requires all roots of the polynomial lie inside the unit circle.

    Parameters
    ----------
    polynomial : array_like or tuple, list
        Coefficients of a polynomial, in order of increasing degree.
        For example, `polynomial=[1, -0.5]` corresponds to the polynomial
        :math:`1 - 0.5x` which has root :math:`2`. If it is a matrix
        polynomial (in which case the coefficients are coefficient matrices),
        a tuple or list of matrices should be passed.
    threshold : number
        Allowed threshold for `is_invertible` to return True. Default is 1.

    Notes
    -----

    If the coefficients provided are scalars :math:`(c_0, c_1, \dots, c_n)`,
    then the corresponding polynomial is :math:`c_0 + c_1 L + \dots + c_n L^n`.


    If the coefficients provided are matrices :math:`(C_0, C_1, \dots, C_n)`,
    then the corresponding polynomial is :math:`C_0 + C_1 L + \dots + C_n L^n`.

    There are three equivalent methods of determining if the polynomial
    represented by the coefficients is invertible:

    The first method factorizes the polynomial into:

    .. math::

        C(L) & = c_0 + c_1 L + \dots + c_n L^n \\
             & = constant (1 - \lambda_1 L)
                 (1 - \lambda_2 L) \dots (1 - \lambda_n L)

    In order for :math:`C(L)` to be invertible, it must be that each factor
    :math:`(1 - \lambda_i L)` is invertible; the condition is then that
    :math:`|\lambda_i| < 1`, where :math:`\lambda_i` is a root of the
    polynomial.

    The second method factorizes the polynomial into:

    .. math::

        C(L) & = c_0 + c_1 L + \dots + c_n L^n \\
             & = constant (L - \zeta_1) (L - \zeta_2) \dots (L - \zeta_3)

    The condition is now :math:`|\zeta_i| > 1`, where :math:`\zeta_i` is a root
    of the polynomial with reversed coefficients and
    :math:`\lambda_i = \frac{1}{\zeta_i}`.

    Finally, a companion matrix can be formed using the coefficients of the
    polynomial. Then the eigenvalues of that matrix give the roots of the
    polynomial. This last method is the one actually used.

    See Also
    --------
    companion_matrix
    """
    # First method:
    # np.all(np.abs(np.roots(np.r_[1, params])) < 1)
    # Second method:
    # np.all(np.abs(np.roots(np.r_[1, params][::-1])) > 1)
    # Final method:
    eigvals = np.linalg.eigvals(companion_matrix(polynomial))
    return np.all(np.abs(eigvals) < threshold)


def constrain_stationary_univariate(unconstrained):
    """
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation

    Parameters
    ----------
    unconstrained : array
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    Returns
    -------
    constrained : array
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
    r = unconstrained/((1 + unconstrained**2)**0.5)
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
        y[k, k] = r[k]
    return -y[n - 1, :]


def unconstrain_stationary_univariate(constrained):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : array
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    Returns
    -------
    unconstrained : array
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


def _constrain_sv_less_than_one(unconstrained, order=None, k_endog=None):
    """
    Transform arbitrary matrices to matrices with singular values less than
    one.

    Corresponds to Lemma 2.2 in Ansley and Kohn (1986). See
    `constrain_stationary_multivariate` for more details.
    """
    from scipy import linalg

    constrained = []  # P_s,  s = 1, ..., p
    if order is None:
        order = len(unconstrained)
    if k_endog is None:
        k_endog = unconstrained[0].shape[0]

    eye = np.eye(k_endog)
    for i in range(order):
        A = unconstrained[i]
        B, lower = linalg.cho_factor(eye + np.dot(A, A.T), lower=True)
        constrained.append(linalg.solve_triangular(B, A, lower=lower))
    return constrained


def _constrain_stationary_multivariate(sv_constrained, variance, order=None,
                                       k_endog=None):
    """
    Transform matrices with singular values less than one to matrices
    corresponding to a stationary (or invertible) process.

    Corresponds to Lemma 2.1 in Ansley and Kohn (1986). See
    `constrain_stationary_multivariate` for more details.
    """
    from scipy import linalg

    if order is None:
        order = len(sv_constrained)
    if k_endog is None:
        k_endog = sv_constrained[0].shape[0]

    forward_variances = [variance]   # \Sigma_s
    backward_variances = [variance]  # \Sigma_s^*,  s = 0, ..., p
    variances = [variance]           # \Gamma_s
    # \phi_{s,k}, s = 1, ..., p
    #             k = 1, ..., s+1
    constrained = []
    # \phi_{s,k}^*
    backwards = []

    variance_factor = linalg.cholesky(variance, lower=True)

    forward_factors = [variance_factor]
    backward_factors = [variance_factor]

    # We fill in the entries as follows:
    # [1,1]
    # [2,2], [2,1]
    # [3,3], [3,1], [3,2]
    # ...
    # [p,p], [p,1], ..., [p,p-1]
    # the last row, correctly ordered, is then used as the coefficients
    for s in range(order):  # s = 0, ..., p-1
        prev_constrained = constrained
        prev_backwards = backwards
        constrained = []
        backwards = []

        # Create the "last" (k = s+1) matrix
        # Note: this is for k = s+1. However, below we then have to fill
        # in for k = 1, ..., s in order.
        # P L^{-1} = x
        # x L = P
        # L' x' = P'
        constrained.append(linalg.solve_triangular(
            backward_factors[s], sv_constrained[s].T, lower=True, trans='T'))
        constrained[0] = np.dot(forward_factors[s], constrained[0].T)

        # P' L^{-1} = x
        # x L = P'
        # L' x' = P
        backwards.append(linalg.solve_triangular(
            forward_factors[s], sv_constrained[s], lower=True, trans='T'))
        backwards[0] = np.dot(backward_factors[s], backwards[0].T)

        # Update the variance
        # Note: if s >= 1, this will be further updated in the for loop
        # below
        # Also, this calculation will be re-used in the forward variance
        tmp = np.dot(constrained[0], backward_variances[s])
        variances.append(tmp.copy())

        # Create the remaining k = 1, ..., s matrices,
        # only has an effect if s >= 1
        for k in range(s):
            constrained.insert(k, prev_constrained[k] - np.dot(
                constrained[k], prev_backwards[s-(k+1)]))

            backwards.insert(k, prev_backwards[k] - np.dot(
                backwards[k], prev_constrained[s-(k+1)]))

            variances[s+1] += np.dot(prev_constrained[s-(k+1)],
                                     variances[k+1])

        # Create forward and backwards variances
        forward_variances.append(
            forward_variances[s] - np.dot(tmp, constrained[s].T)
        )
        backward_variances.append(
            backward_variances[s] -
            np.dot(
                np.dot(backwards[s], forward_variances[s]),
                backwards[s].T
            )
        )

        # Cholesky factors
        forward_factors.append(
            linalg.cholesky(forward_variances[s+1], lower=True)
        )
        backward_factors.append(
            linalg.cholesky(backward_variances[s+1], lower=True)
        )

    return constrained, forward_variances[-1]


def constrain_stationary_multivariate(unconstrained, variance,
                                      transform_variance=False):
    """
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation for a vector autoregression.

    Parameters
    ----------
    unconstrained : iterable
        Arbitrary matrices to be transformed to stationary coefficient matrices
        of the VAR.
    variance : array, 2-dim
        Variance matrix corresponding to the error term. This is used as
        input in the algorithm even if is not transformed by it (when
        `transform_variance` is False. The error term variance is required
        input when transformation is used either to force an autoregressive
        component to be stationary or to force  a moving average component to
        be invertible.
    transform_variance : boolean, optional
        Whether or not to transform the error variance term. This option is
        not typically used, and the default is False.

    Returns
    -------
    constrained : list
        A list of coefficient matrices which lead to a stationary VAR.

    Notes
    -----
    In the notation of [1]_, the arguments `(variance, unconstrained)` are
    written as :math:`(\Sigma, A_1, \dots, A_p)`, where :math:`p` is the order
    of the vector autoregression, and is here determined by the length of
    the `unconstrained` argument.

    There are two steps in the constraining algorithm.

    First, :math:`(A_1, \dots, A_p)` are transformed into
    :math:`(P_1, \dots, P_p)` via Lemma 2.2 of [1]_.

    Second, :math:`(\Sigma, P_1, \dots, P_p)` are transformed into
    :math:`(\Sigma, \phi_1, \dots, \phi_p)` via Lemmas 2.1 and 2.3 of [1]_.

    If `transform_variance=True`, then only Lemma 2.1 is applied in the second
    step.

    While this function can be used even in the univariate case, it is much
    slower, so in that case `constrain_stationary_univariate` is preferred.

    References
    ----------
    Ansley, Craig F., and Robert Kohn. 1986.
    "A Note on Reparameterizing a Vector Autoregressive Moving Average Model to
    Enforce Stationarity."
    Journal of Statistical Computation and Simulation 24 (2): 99-106.

    """
    from scipy import linalg

    order = len(unconstrained)
    k_endog = unconstrained[0].shape[0]

    # Step 1: convert from arbitrary matrices to those with singular values
    # less than one.
    sv_constrained = _constrain_sv_less_than_one(unconstrained, order, k_endog)

    # Step 2: convert matrices from our "partial autocorrelation matrix" space
    # (matrices with singular values less than one) to the space of stationary
    # coefficient matrices
    if transform_variance:
        input_variance = variance
    else:
        # Need to make the input variance large enough that the recursions
        # don't lead to zero-matrices due to roundoff error, which would case
        # exceptions from the Cholesky decompositions.
        # Note that this will still not always ensure positive definiteness,
        # and for k_endog, order large enough an exception may still be raised
        input_variance = np.eye(k_endog) * (order + k_endog)**10
    constrained, transformed_variance = (
        _constrain_stationary_multivariate(sv_constrained, input_variance,
                                           order, k_endog)
    )

    # Step 3: If we do not want to use the transformed variance, we need to
    # adjust the constrained matrices, as presented in Lemma 2.3, see Notes
    if not transform_variance:
        # Here, we need to construct T such that:
        # variance = T * initial_variance * T'
        # To do that, consider the Cholesky of variance (L) and
        # input_variance (M) to get:
        # L L' = T M M' T' = (TM) (TM)'
        # => L = T M
        # => L M^{-1} = T
        variance_factor = np.linalg.cholesky(variance)
        input_variance_factor = np.linalg.cholesky(input_variance)
        transform = np.dot(variance_factor,
                           np.linalg.inv(input_variance_factor))
        inv_transform = np.linalg.inv(transform)

        for i in range(order):
            constrained[i] = (
                np.dot(np.dot(transform, constrained[i]), inv_transform)
            )

    return constrained, variance


def unconstrain_stationary_multivariate(constrained):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : array
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    Returns
    -------
    unconstrained : array
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    References
    ----------
    Ansley, Craig F., and Robert Kohn. 1986.
    "A Note on Reparameterizing a Vector Autoregressive Moving Average Model to
    Enforce Stationarity."
    Journal of Statistical Computation and Simulation 24 (2): 99-106.

    """
    raise NotImplementedError


def validate_matrix_shape(name, shape, nrows, ncols, nobs):
    """
    Validate the shape of a possibly time-varying matrix, or raise an exception

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
        raise ValueError('Invalid dimensions for %s matrix: time-varying'
                         ' matrices cannot be given unless `nobs` is specified'
                         ' (implicitly when a dataset is bound or else set'
                         ' explicity)' % name)

    # Enforce time-varying array size
    if ndim == 3 and nobs is not None and not shape[-1] in [1, nobs]:
        raise ValueError('Invalid dimensions for time-varying %s'
                         ' matrix. Requires shape (*,*,%d), got %s' %
                         (name, nobs, str(shape)))


def validate_vector_shape(name, shape, nrows, nobs):
    """
    Validate the shape of a possibly time-varying vector, or raise an exception

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
        raise ValueError('Invalid dimensions for %s vector: time-varying'
                         ' vectors cannot be given unless `nobs` is specified'
                         ' (implicitly when a dataset is bound or else set'
                         ' explicity)' % name)

    # Enforce time-varying array size
    if ndim == 2 and not shape[1] in [1, nobs]:
        raise ValueError('Invalid dimensions for time-varying %s'
                         ' vector. Requires shape (*,%d), got %s' %
                         (name, nobs, str(shape)))
