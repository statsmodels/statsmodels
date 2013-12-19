"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions haven't been formally tested.
"""

from scipy import stats
import numpy as np


# TODO: these are pretty straightforward but they should be tested
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
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
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
    axis : int, optional
        Default is 0

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
    jb = (n / 6.) * (skew**2 + (1 / 4.) * (kurtosis - 3)**2)
    jb_pv = stats.chi2.sf(jb, 2)

    return jb, jb_pv, skew, kurtosis


def robust_skewness(y, axis=0):
    """
    Calculates the four skewness measures in Kim & White

    Parameters
    ----------
    y : array-like
    axis : int or None, optional
        Axis along which the skewness measures are computed.  If `None`, the
        entire array is used.

    Returns
    -------
    sk1 : ndarray
          The standard skewness estimator.
    sk2 : ndarray
          Skewness estimator based on quartiles.
    sk3 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          absolute deviation.
    sk4 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          standard deviation.

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

    q1, q2, q3 = np.percentile(y, [25.0, 50.0, 75.0], axis=axis)

    mu = y.mean(axis)
    shape = (y.size,)
    if axis is not None:
        shape = list(mu.shape)
        shape.insert(axis, 1)
        shape = tuple(shape)

    mu_b = np.reshape(mu, shape)
    q2_b = np.reshape(q2, shape)

    sigma = np.mean(((y - mu_b)**2), axis)

    sk1 = stats.skew(y, axis=axis)
    sk2 = (q1 + q3 - 2.0 * q2) / (q3 - q1)
    sk3 = (mu - q2) / np.mean(abs(y - q2_b), axis=axis)
    sk4 = (mu - q2) / sigma

    return sk1, sk2, sk3, sk4


def _kr3(y, alpha=5.0, beta=50.0):
    """
    KR3 estimator from Kim & White

    Parameters
    ----------
    y : array-like, 1-d
    alpha : float, optional
            Lower cut-off for measuring expectation in tail.
    beta :  float, optional
            Lower cut-off for measuring expectation in center.

    Returns
    -------
    kr3 : float
          Robust kurtosis estimator based on

    Notes
    -----
    .. [1] Tae-Hwan Kim and Halbert White, "On more robust estimation of
    skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
    March 2004.
    """
    perc = (alpha, 100.0-alpha, beta, 100.0-beta)
    lower_alpha, upper_alpha, lower_beta, upper_beta = np.percentile(y, perc)
    l_alpha = np.mean(y[y < lower_alpha])
    u_alpha = np.mean(y[y > upper_alpha])

    l_beta = np.mean(y[y < lower_beta])
    u_beta = np.mean(y[y > upper_beta])

    return (u_alpha - l_alpha) / (u_beta - l_beta)


def robust_kurtosis(y, axis=0, excess=True):
    """
    Calculates the four kurtosis measures in Kim & White

    Parameters
    ----------
    y : array-like
    axis : int or None, optional
        Axis along which the kurtoses are computed.  If `None`, the
        entire array is used.
    excess : bool, optional
        If true (default), computed values are excess of those for a standard
        normal distribution.

    Returns
    -------
    kr1 : ndarray
          The standard kurtosis estimator.
    kr2 : ndarray
          Kurtosis estimator based on octiles.
    kr3 : ndarray
          Kurtosis estimators based on exceedence expectations.
    kr4 : ndarray
          Kurtosis measure based on the spread between high and low quantiles.

    Notes
    -----
    .. [1] Tae-Hwan Kim and Halbert White, "On more robust estimation of
    skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
    March 2004.
    """
    perc = (12.5, 25.0, 37.5, 62.5, 75.0, 87.5, 2.5, 97.5)
    e1, e2, e3, e5, e6, e7, fa, f1ma = np.percentile(y, perc, axis=axis)

    alpha, beta = 5.0, 50.0

    f1mb = e6
    fb = e2

    expected_value = np.zeros(4)
    if excess:
        ppf = stats.norm.ppf
        pdf = stats.norm.pdf
        q1, q2, q3, q5, q6, q7 = ppf(np.array((1.0, 2.0, 3.0, 5.0, 6.0, 7.0))/8)
        expected_value[0] = 3
        expected_value[1] = ((q7 - q5) +(q3 - q1)) / (q6 - q2)
        q50, q95= ppf(np.array((.50, .95)))
        expected_value[2] = (2 * pdf(q95)/.05) / (2 * pdf(q50)/.5)
        q025, q975 = ppf(np.array((.025, .975)))
        expected_value[3] = (q975 - q025) / (q6 - q2)

    kr1 = stats.kurtosis(y, axis, False) - expected_value[0]
    kr2 = ((e7 - e5) + (e3 - e1)) / (e6 - e2) - expected_value[1]
    if y.ndim == 1:
        kr3 = _kr3(y) - expected_value[2]
    else:
        kr3 = np.apply_along_axis(_kr3, axis, y, alpha, beta) - expected_value[2]
    kr4 = (f1ma - fa) / (f1mb - fb) - expected_value[3]
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
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    y = np.sort(y)

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]

    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    standardization[standardization==0] = np.inf
    spread = upper + lower
    return np.median(spread / standardization)


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