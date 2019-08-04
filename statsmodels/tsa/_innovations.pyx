#cython: language_level=3, wraparound=False, cdivision=True, boundscheck=False
import numpy as np


from statsmodels.tools.validation import (array_like, PandasWrapper,
                                          int_like, float_like)


def innovations_algo(acov, nobs=None, rtol=None):
    """
    innovations_algo(acov, nobs=None, rtol=None)

    Innovations algorithm to convert autocovariances to MA parameters.

    Parameters
    ----------
    acov : array_like
        Array containing autocovariances including lag 0.
    nobs : int, optional
        Number of periods to run the algorithm.  If not provided, nobs is
        equal to the length of acovf.
    rtol : float, optional
        Tolerance used to check for convergence. Default value is 0 which will
        never prematurely end the algorithm. Checks after 10 iterations and
        stops if sigma2[i] - sigma2[i - 10] < rtol * sigma2[0]. When the
        stopping condition is met, the remaining values in theta and sigma2
        are forward filled using the value of the final iteration.

    Returns
    -------
    theta : ndarray
        Innovation coefficients of MA representation. Array is (nobs, q) where
        q is the largest index of a non-zero autocovariance. theta
        corresponds to the first q columns of the coefficient matrix in the
        common description of the innovation algorithm.
    sigma2 : ndarray
        The prediction error variance (nobs,).

    See Also
    --------
    innovations_filter : Filter a series using the innovations algorithm.

    References
    ----------
    .. [*] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> rgdpg = data.data['realgdp'].pct_change().dropna()
    >>> acov = sm.tsa.acovf(rgdpg)
    >>> nobs = activity.shape[0]
    >>> theta, sigma2  = innovations_algo(acov[:4], nobs=nobs)
    """
    cdef double[::1] v, _acov
    cdef double[:, ::1] theta
    cdef Py_ssize_t i, j, k, max_lag
    cdef double sub, _rtol

    acov = array_like(acov, 'acov', contiguous=True, ndim=1)
    nobs = int_like(nobs, 'nobs', optional=True)
    rtol = float_like(rtol, 'rtol', optional=True)
    _acov = acov

    rtol = 0.0 if rtol is None else rtol
    if not rtol >= 0:
        raise ValueError('rtol must be a non-negative float or None.')
    _rtol = rtol
    if nobs is not None and nobs < 1:
        raise ValueError('nobs must be a positive integer')
    n = acov.shape[0] if nobs is None else nobs
    max_lag = int(np.max(np.argwhere(acov != 0)))

    v = np.zeros(n + 1)
    v[0] = _acov[0]
    # Retain only the relevant columns of theta
    theta = np.zeros((n + 1, max_lag + 1))
    for i in range(1, n):
        for k in range(max(i - max_lag, 0), i):
            sub = 0
            for j in range(max(i - max_lag, 0), k):
                sub += theta[k, k - j] * theta[i, i - j] * v[j]
            theta[i, i - k] = 1. / v[k] * (_acov[i - k] - sub)
        v[i] = _acov[0]
        for j in range(max(i - max_lag, 0), i):
            v[i] -= theta[i, i - j] ** 2 * v[j]
        # Break if v has converged
        if i >= 10:
            if v[i - 10] - v[i] < v[0] * _rtol:
                # Forward fill all remaining values
                v[i + 1:] = v[i]
                theta[i + 1:] = theta[i]
                break

    return np.asarray(theta[:n, 1:]), np.asarray(v[:n])


def innovations_filter(endog, theta):
    """
    innovations_filter(endog, theta)

    Filter observations using the innovations algorithm.

    Parameters
    ----------
    endog : array_like
        The time series to filter (nobs,). Should be demeaned if not mean 0.
    theta : ndarray
        Innovation coefficients of MA representation. Array must be (nobs, q)
        where q order of the MA.

    Returns
    -------
    ndarray
        Array of filtered innovations.

    See Also
    --------
    innovations_algo : Convert autocovariances to MA parameters.

    References
    ----------
    .. [*] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> rgdpg = data.data['realgdp'].pct_change().dropna()
    >>> acov = sm.tsa.acovf(rgdpg)
    >>> nobs = activity.shape[0]
    >>> theta, sigma2  = innovations_algo(acov[:4], nobs=nobs)
    >>> resid = innovations_filter(rgdpg, theta)
    """
    cdef Py_ssize_t i, j, k, n_theta, nobs
    cdef double[::1] _endog, u
    cdef double[:, ::1] _theta
    cdef double hat

    pw = PandasWrapper(endog)
    endog = array_like(endog, 'endog', contiguous=True, ndim=1)
    theta = array_like(theta, 'theta', contiguous=True, ndim=2)
    nobs = endog.shape[0]
    n_theta, k = theta.shape
    if nobs != n_theta:
        raise ValueError('theta must be (nobs, q) where q is the moder order')
    _endog = endog
    _theta = theta
    u = np.empty(nobs)
    u[0] = _endog[0]
    for i in range(1, nobs):
        hat = 0.0
        for j in range(min(i, k)):
            hat += _theta[i, j] * u[i - j - 1]
        u[i] = _endog[i] - hat

    _u = np.asarray(u)
    return pw.wrap(_u)
