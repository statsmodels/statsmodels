"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions haven't been formally tested.
"""

from scipy import stats
import numpy as np

# TODO: these are pretty straightforward but they should be tested
# TODO: Critical Values
def durbin_watson(resids, axis=0):
    """
    Calculates the Durbin-Watson statistic

    Parameters
    -----------
    resids : array-like

    Returns
        --------
    Durbin Watson statistic.  This is defined as
    sum_(t=2)^(T)((e_t - e_(t-1))^(2))/sum_(t=1)^(T)e_t^(2)
    """
    resids=np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids ** 2.0, axis=axis) / np.sum(resids ** 2.0, axis=axis)
    return dw

def omni_normtest(resids, axis=0):
    """
    Omnibus test for normality

    Parameters
    -----------
    resid : array-like
    axis : int, optional
        Default is 0

    Returns
    -------
    Chi^2 score, two-tail probability
    """
    #TODO: change to exception in summary branch and catch in summary()
    #behavior changed between scipy 0.9 and 0.10
    resids = np.asarray(resids)
    n = resids.shape[axis]
    if n < 8:
        from warnings import warn
        warn("omni_normtest is not valid with less than 8 observations; %i samples"
            " were given." % int(n))
        return np.nan, np.nan

    return stats.normaltest(resids, axis=axis)

def jarque_bera(resids, axis=0):
    """
    Calculate residual skewness, kurtosis, and do the JB test for normality

    Parameters
    -----------
    resids : array-like

    Returns
    -------
    JB, JBpv, skew, kurtosis

    JB = n/6*(S^2 + (K-3)^2/4)

    JBpv is the Chi^2 two-tail probability value

    skew is the measure of skewness

    kurtosis is the measure of kurtosis

    """
    resids = np.asarray(resids)
    # Calculate residual skewness and kurtosis
    skew = stats.skew(resids, axis=axis)
    kurtosis = 3 + stats.kurtosis(resids, axis=axis)

    # Calculate the Jarque-Bera test for normality
    n = resids.shape[axis]
    JB = (n / 6.) * (skew**2 + (1 / 4.) * (kurtosis-3)**2)
    JBpv = stats.chi2.sf(JB,2)

    return JB, JBpv, skew, kurtosis


# def _scalar_slice(i, axis, ndim):
#     if ndim==1:
#         return i
#
#     s = [slice(None)] * ndim
#     s[axis] = i
#     return s


# def _weighted_quantile(a, axis, q):
#
#     n = a.shape[axis]
#     ndim = a.ndim
#
#     loc1 = int(q*(n-1) - 1)
#     loc2 = loc1 + 1
#
#     w = 1 - ((n-1) * q  - loc1)
#
#     s1 = _scalar_slice(loc1, axis, ndim)
#     s2 = _scalar_slice(loc2, axis, ndim)
#
#     return w * a[s1] + (1-w) * a[s2]


def robust_skewness(y, axis=0):
    """
    Calculates the four skewness measures in Kim & White

    Notes
    -----
    .. [1] Tae-Hwan Kim and Halbert White, "On more robust estimation of
    skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
    March 2004.
    """

    if axis is None:
        y = y.flat[:]
        axis = 0

    y = np.sort(y, axis)

    q1 = np.percentile(y,25.0,axis=axis)
    q2 = np.percentile(y,50.0,axis=axis)
    q3 = np.percentile(y,75.0,axis=axis)

    mu = y.mean(axis)
    shape = (y.size,)
    if axis is not None:
        shape = list(mu.shape)
        shape.insert(axis, 1)
        shape = tuple(shape)

    mu_b = np.reshape(mu, shape)
    q2_b = np.reshape(q2, shape)

    sigma = np.mean(((y - mu_b) ** 2.0), axis)

    sk1 =  stats.skew(y, axis=axis)
    sk2 = (q1 + q3 - 2.0 * q2) / (q3 - q1)
    sk3 = (mu - q2) / np.mean(abs(y - q2_b), axis=axis)
    sk4 = (mu - q2) / sigma

    return sk1, sk2, sk3, sk4


def _kr3(y, alpha, beta):

    l_alpha = np.mean(y[y<np.percentile(y,alpha)])
    u_alpha = np.mean(y[y>np.percentile(y,100.0-alpha)])

    l_beta = np.mean(y[y<np.percentile(y,beta)])
    u_beta = np.mean(y[y>np.percentile(y,100.0-beta)])

    return (u_alpha - l_alpha) / (u_beta - l_beta) - 2.5852205221971283


def robust_kurtosis(y, axis=0):
    """
    Calculates the four kurtosis measures in Kim & White

    Notes
    -----
    .. [1] Tae-Hwan Kim and Halbert White, "On more robust estimation of
    skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
    March 2004.
    """
    e1 = np.percentile(y, 12.5, axis=axis)
    e2 = np.percentile(y, 25.0, axis=axis)
    e3 = np.percentile(y, 37.5, axis=axis)
    e5 = np.percentile(y, 62.5, axis=axis)
    e6 = np.percentile(y, 75.0, axis=axis)
    e7 = np.percentile(y, 87.5, axis=axis)

    alpha, beta = 5.0, 50.0

    f1ma = np.percentile(y, 97.5, axis)
    fa = np.percentile(y, 2.5, axis)
    f1mb = np.percentile(y, 75.0, axis)
    fb = np.percentile(y, 25.0, axis)

    kr1 = stats.kurtosis(y,axis)
    kr2 = ((e7 - e5) + (e3 - e1)) / (e6 - e2) - 1.2330951154852172
    kr3 = np.squeeze(np.apply_along_axis(_kr3, axis, y, alpha, beta))
    kr4 = (f1ma - fa) / (f1mb - fb) - 2.9058469516701639
    return kr1, kr2, kr3, kr4


def _medcouple_1d(y):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array-like, 1-d

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [1] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
    distributions" Computational Statistics & Data Analysis, vol. 52,
    pp. 5186-5201, August 2008.
    """

    # Parameter changes the algorithm to the slower for large n

    y = np.squeeze(np.asarray(y))
    if len(y.shape) > 1:
        raise ValueError("y must be squeezable to a 1-d array")

    y = np.sort(y)

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n//2 - 1] + y[n//2 ])/2
    else:
        mf = y[(n-1)//2]

    lower = y[y<mf]
    upper = y[y>mf]
    upper = upper[:,None]
    diff = upper - lower
    sum = upper + lower
    return np.median((sum - 2.0 * mf) / diff)


def medcouple(y, axis=0):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array-like
    axis : int or None, optional
        Axis along which the medcouple statistic is computed.  If `None`, the
        entire array is used.

    Returns
    -------
    mc : ndarray
        The medcouple statistic with the same shape as `y`, with the specified
        axis removed.

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [1] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
    distributions" Computational Statistics & Data Analysis, vol. 52,
    pp. 5186-5201, August 2008.
    """
    if axis is None:
        return _medcouple_1d(y.flat[:])

    return np.apply_along_axis(_medcouple_1d, axis, y)