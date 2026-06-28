"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions have not been formally tested.
"""

import numpy as np
from scipy import stats

from statsmodels.tools.sm_exceptions import ValueWarning


def durbin_watson(resids, axis=0):
    r"""
    Calculates the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array_like
        Data for which to compute the Durbin-Watson statistic. Usually
        regression model residuals.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    dw : float, array_like
        The Durbin-Watson statistic.

    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation
    in the residuals.
    The Durbin-Watson test statistic is defined as:

    .. math::

       \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.
    """
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
    return dw


def omni_normtest(resids, axis=0):
    """
    Omnibus test for normality

    Parameters
    ----------
    resid : array_like
    axis : int, optional
        Default is 0

    Returns
    -------
    Chi^2 score, two-tail probability
    """
    # TODO: change to exception in summary branch and catch in summary()
    #   behavior changed between scipy 0.9 and 0.10
    resids = np.asarray(resids)
    n = resids.shape[axis]
    if n < 8:
        from warnings import warn
        warn("omni_normtest is not valid with less than 8 observations; %i "
             "samples were given." % int(n), ValueWarning, stacklevel=2)
        return np.nan, np.nan

    return stats.normaltest(resids, axis=axis)


def jarque_bera(resids, axis=0):
    r"""
    The Jarque-Bera test of normality.

    Parameters
    ----------
    resids : array_like
        Data to test for normality. Usually regression model residuals that
        are mean 0.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    JB : {float, ndarray}
        The Jarque-Bera test statistic.
    JBpv : {float, ndarray}
        The pvalue of the test statistic.
    skew : {float, ndarray}
        Estimated skewness of the data.
    kurtosis : {float, ndarray}
        Estimated kurtosis of the data.

    Notes
    -----
    Each output returned has 1 dimension fewer than data

    The Jarque-Bera test statistic tests the null that the data is normally
    distributed against an alternative that the data follow some other
    distribution. The test statistic is based on two moments of the data,
    the skewness, and the kurtosis, and has an asymptotic :math:`\chi^2_2`
    distribution.

    The test statistic is defined

    .. math:: JB = n(S^2/6+(K-3)^2/24)

    where n is the number of data points, S is the sample skewness, and K is
    the sample kurtosis of the data.
    """
    resids = np.atleast_1d(np.asarray(resids, dtype=float))
    if resids.size < 2:
        raise ValueError("resids must contain at least 2 elements")
    # Calculate residual skewness and kurtosis
    skew = stats.skew(resids, axis=axis)
    kurtosis = 3 + stats.kurtosis(resids, axis=axis)

    # Calculate the Jarque-Bera test for normality
    n = resids.shape[axis]
    jb = (n / 6.) * (skew ** 2 + (1 / 4.) * (kurtosis - 3) ** 2)
    jb_pv = stats.chi2.sf(jb, 2)

    return jb, jb_pv, skew, kurtosis


def robust_skewness(y, axis=0):
    """
    Calculates the four skewness measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
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
    The robust skewness measures are defined

    .. math::

        SK_{2}=\\frac{\\left(q_{.75}-q_{.5}\\right)
        -\\left(q_{.5}-q_{.25}\\right)}{q_{.75}-q_{.25}}

    .. math::

        SK_{3}=\\frac{\\mu-\\hat{q}_{0.5}}
        {\\hat{E}\\left[\\left|y-\\hat{\\mu}\\right|\\right]}

    .. math::

        SK_{4}=\\frac{\\mu-\\hat{q}_{0.5}}{\\hat{\\sigma}}

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """

    if axis is None:
        y = y.ravel()
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

    sigma = np.sqrt(np.mean(((y - mu_b)**2), axis))

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
    y : array_like, 1-d
        Data to compute use in the estimator.
    alpha : float, optional
        Lower cut-off for measuring expectation in tail.
    beta :  float, optional
        Lower cut-off for measuring expectation in center.

    Returns
    -------
    kr3 : float
        Robust kurtosis estimator based on standardized lower- and upper-tail
        expected values

    Notes
    -----
    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    perc = (alpha, 100.0 - alpha, beta, 100.0 - beta)
    lower_alpha, upper_alpha, lower_beta, upper_beta = np.percentile(y, perc)
    l_alpha = np.mean(y[y < lower_alpha])
    u_alpha = np.mean(y[y > upper_alpha])

    l_beta = np.mean(y[y < lower_beta])
    u_beta = np.mean(y[y > upper_beta])

    return (u_alpha - l_alpha) / (u_beta - l_beta)


def expected_robust_kurtosis(ab=(5.0, 50.0), dg=(2.5, 25.0)):
    """
    Calculates the expected value of the robust kurtosis measures in Kim and
    White assuming the data are normally distributed.

    Parameters
    ----------
    ab : iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure

    Returns
    -------
    ekr : ndarray, 4-element
        Contains the expected values of the 4 robust kurtosis measures

    Notes
    -----
    See `robust_kurtosis` for definitions of the robust kurtosis measures
    """

    alpha, beta = ab
    delta, gamma = dg
    expected_value = np.zeros(4)
    ppf = stats.norm.ppf
    pdf = stats.norm.pdf
    q1, q2, q3, q5, q6, q7 = ppf(np.array((1.0, 2.0, 3.0, 5.0, 6.0, 7.0)) / 8)
    expected_value[0] = 3

    expected_value[1] = ((q7 - q5) + (q3 - q1)) / (q6 - q2)

    q_alpha, q_beta = ppf(np.array((alpha / 100.0, beta / 100.0)))
    expected_value[2] = (2 * pdf(q_alpha) / alpha) / (2 * pdf(q_beta) / beta)

    q_delta, q_gamma = ppf(np.array((delta / 100.0, gamma / 100.0)))
    expected_value[3] = (-2.0 * q_delta) / (-2.0 * q_gamma)

    return expected_value


def robust_kurtosis(y, axis=0, ab=(5.0, 50.0), dg=(2.5, 25.0), excess=True):
    """
    Calculates the four kurtosis measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the kurtosis are computed.  If `None`, the
        entire array is used.
    a iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure
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
          Kurtosis estimators based on exceedance expectations.
    kr4 : ndarray
          Kurtosis measure based on the spread between high and low quantiles.

    Notes
    -----
    The robust kurtosis measures are defined

    .. math::

        KR_{2}=\\frac{\\left(\\hat{q}_{.875}-\\hat{q}_{.625}\\right)
        +\\left(\\hat{q}_{.375}-\\hat{q}_{.125}\\right)}
        {\\hat{q}_{.75}-\\hat{q}_{.25}}

    .. math::

        KR_{3}=\\frac{\\hat{E}\\left(y|y>\\hat{q}_{1-\\alpha}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\alpha}\\right)}
        {\\hat{E}\\left(y|y>\\hat{q}_{1-\\beta}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\beta}\\right)}

    .. math::

        KR_{4}=\\frac{\\hat{q}_{1-\\delta}-\\hat{q}_{\\delta}}
        {\\hat{q}_{1-\\gamma}-\\hat{q}_{\\gamma}}

    where :math:`\\hat{q}_{p}` is the estimated quantile at :math:`p`.

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    if (axis is None or
            (y.squeeze().ndim == 1 and y.ndim != 1)):
        y = y.ravel()
        axis = 0

    alpha, beta = ab
    delta, gamma = dg

    perc = (12.5, 25.0, 37.5, 62.5, 75.0, 87.5,
            delta, 100.0 - delta, gamma, 100.0 - gamma)
    e1, e2, e3, e5, e6, e7, fd, f1md, fg, f1mg = np.percentile(y, perc,
                                                               axis=axis)

    expected_value = (expected_robust_kurtosis(ab, dg)
                      if excess else np.zeros(4))

    kr1 = stats.kurtosis(y, axis, False) - expected_value[0]
    kr2 = ((e7 - e5) + (e3 - e1)) / (e6 - e2) - expected_value[1]
    if y.ndim == 1:
        kr3 = _kr3(y, alpha, beta)
    else:
        kr3 = np.apply_along_axis(_kr3, axis, y, alpha, beta)
    kr3 -= expected_value[2]
    kr4 = (f1md - fd) / (f1mg - fg) - expected_value[3]
    return kr1, kr2, kr3, kr4


def _medcouple_1d_legacy(y):
    """
    Calculates the medcouple robust measure of skew. Less efficient version of
    the algorithm which computes in O(N**2) time. Useful for validating the
    O(N log N) version and for applications requiring legacy behavior.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    This version of the algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
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
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    # GH5395
    num_ties = np.sum(lower == 0.0)
    if num_ties:

        # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
        # and 1 below the anti-diagonal
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)

        # Convert diagonal to anti-diagonal
        replacements = np.fliplr(replacements)

        # Always replace upper right block
        h[:num_ties, -num_ties:] = replacements

    return np.median(h)


def _wmedian(A, W):
    r"""
    Compute the weighted median of the values in A using the associated weights in W.

    Parameters
    ----------
    A : 1-d NumPy array of float
        The numeric values for which the weighted median is to be computed.
    W : 1-d NumPy array of int
        The corresponding non-negative integer weights for each value in A.

    Returns
    -------
    float
        The weighted median of A. If there are multiple medians due to tied weights,
        the lower median is returned.

    Notes
    -----
    This is a helper function for the O(N log N) medcouple algorithm.
    """

    # Validation: NaN protection
    if np.any(np.isnan(A)) or np.any(np.isnan(W)):
        raise ValueError("A and W may not contain NaN.")

    # Ensure 1-d arrays
    if A.ndim != 1 or W.ndim != 1:
        raise ValueError("A and W must be 1-dimensional arrays.")

    # Ensure same length
    if A.shape[0] != W.shape[0]:
        raise ValueError("A and W must have the same length.")

    # Sort A and W according to A
    idx = np.argsort(A)
    A_sorted = A[idx]
    W_sorted = W[idx]

    # Compute cumulative sum of weights
    w_cumsum = np.cumsum(W_sorted)
    wtot = w_cumsum[-1]

    # Find the smallest index i such that cumulative weight >= total weight / 2
    median_idx = np.searchsorted(w_cumsum, wtot / 2, side="left")

    return A_sorted[median_idx]


def _construct_A_W(L, R, Zplus, Zminus, n_plus, eps2):
    """
    Vectorized construction of A and W as NumPy arrays.

    Parameters
    ----------
    L : np.ndarray
        1-d array of left bounds.
    R : np.ndarray
        1-d array of right bounds.
    Zplus : np.ndarray
        1-d array of input values.
    Zminus :  np.ndarray
        1-d array of input values.
    n_plus : int
    eps2 : float

    Returns
    -------
    A : np.ndarray
        Array of kernel values.
    W : np.ndarray
        Corresponding weights.
    valid_i : np.ndarray
        Indices used for construction.
    """
    valid_i = np.where(L <= R)[0]
    L_valid = L[valid_i]
    R_valid = R[valid_i]
    mid_indices = (L_valid + R_valid) // 2

    A = np.empty_like(valid_i, dtype=float)
    for k in range(valid_i.size):
        A[k] = _h_kern(valid_i[k], mid_indices[k], Zplus, Zminus, n_plus, eps2)

    W = R_valid - L_valid + 1
    return A, W, valid_i


def _h_kern(index_plus, index_minus, Zplus, Zminus, n_plus, eps2):
    """
    H kernel function.

    Parameters
    ----------
    index_plus : int-like
        Index of Zplus.
    index_minus : int-like
        Index of Zminus.
    Zplus : np.ndarray
        1-d array of input values.
    Zminus :  np.ndarray
        1-d array of input values.
    n_plus : int
    eps2 : float

    Returns
    -------
    float
    """

    zp_i = Zplus[index_plus]
    zm_i = Zminus[index_minus]

    # tie breaker: np.sign functionally equivalent to signum
    if abs(zp_i - zm_i) <= 2 * eps2:
        return np.sign(n_plus - 1 - index_plus - index_minus)
    return (zp_i + zm_i) / (zp_i - zm_i)


def _finalize_h_kernel_sweep(L, R, Zplus, Zminus, n_plus, eps2):
    """
    Compute the final array A.

    Parameters
    ----------
    L : np.ndarray
        1-d array of left indices.
    R : np.ndarray
        1-d array of right indices.
    Zplus : np.ndarray
        1-d array of input values.
    Zminus :  np.ndarray
        1-d array of input values.
    n_plus : int
    eps2 : float

    Returns
    -------
    A : numpy.ndarray of float
        1-d array of sorted h_kern values in descending order.
    """

    # Determine total number of h_kern values.
    total_count = int(np.sum(R - L + 1))

    # Preallocate an array for the results.
    A = np.empty(total_count, dtype=np.float64)

    # Position to insert next block of values.
    pos = 0

    # Loop over each index_plus element.
    for i in range(L.shape[0]):
        left = L[i]
        right = R[i]

        # Loop over each corresponding index_minus.
        for j in range(left, right + 1):

            # Here both i and j are scalars.
            A[pos] = _h_kern(i, j, Zplus, Zminus, n_plus, eps2)
            pos += 1

    # Sort in descending order and return.
    return np.sort(A)[::-1]


def _medcouple_nlogn(X, eps1=2**-52, eps2=2**-1022):
    r"""
    Calculates the medcouple robust measure of skewness. Faster version of the
    algorithm which computes in O(N log N) time.

    Parameters
    ----------
    X : np.ndarray
        Input 1-d array of numeric values.

    Returns
    -------
    float
        The medcouple statistic.

    Notes
    -----

    NaNs are not automatically removed. If present in the input, the result
    will be NaN.

    .. [*] Guy Brys, Mia Hubert and Anja Struyf (2004) A Robust Measure
       of Skewness; JCGS 13 (4), 996-1017.
    """

    if np.any(np.isnan(X)):
        return np.nan

    n = X.shape[0]

    if n < 3:
        from warnings import warn
        msg = (
            "medcouple is undefined for input with less than 3 elements. "
            "Returning NaN."
        )
        warn(msg, ValueWarning)
        return np.nan

    if n < 10:
        from warnings import warn
        msg = (
            "Fast medcouple algorithm (use_fast=True) is not recommended "
            "for small datasets (N < 10). Results may be unstable. Consider "
            "using use_fast=False for accuracy."
        )
        warn(msg, UserWarning)

    Z = np.sort(X)[::-1]
    n2 = (n - 1) // 2
    Zmed = Z[n2] if n % 2 else (Z[n2] + Z[n2 + 1]) / 2

    if np.abs(Z[0] - Zmed) < eps1 * (eps1 + np.abs(Zmed)):
        return -1.0
    if np.abs(Z[-1] - Zmed) < eps1 * (eps1 + np.abs(Zmed)):
        return 1.0

    Z -= Zmed
    Zden = 2 * max(Z[0], -Z[-1])
    Z /= Zden
    Zmed /= Zden
    Zeps = eps1 * (eps1 + np.abs(Zmed))

    # Zplus, Zminus are 1-d np.ndarrays
    Zplus = Z[Z >= -Zeps]
    Zminus = Z[Z <= Zeps]

    # get lengths
    n_plus = Zplus.shape[0]
    n_minus = Zminus.shape[0]

    # construct L, R as numpy arrays
    L = np.zeros(n_plus, dtype=int)
    R = np.full(n_plus, n_minus - 1, dtype=int)

    Ltot = 0
    Rtot = n_minus * n_plus
    medc_idx = Rtot // 2

    while Rtot - Ltot > n_plus:

        # Construct A, W as NumPy arrays
        A, W, _ = _construct_A_W(L, R, Zplus, Zminus, n_plus, eps2)

        h_med = _wmedian(A, W)

        Am_eps = eps1 * (eps1 + np.abs(h_med))

        # Preallocate arrays P and Q of length n_plus.
        P = np.empty(n_plus, dtype=int)
        Q = np.empty(n_plus, dtype=int)

        # Construct P. Note: We traverse i in reversed order.
        j = 0
        for idx in range(n_plus):

            # i goes in reversed order; use reversed indices.
            i = n_plus - 1 - idx

            # Increase j until the condition is no longer met.
            while j < n_minus and \
                    _h_kern(i, j, Zplus, Zminus, n_plus, eps2) - h_med > Am_eps:
                j += 1

            # j-1 is our current value for that i.
            # Store it in P at the reversed index; we will fix the order later.
            P[idx] = j - 1

        # Reverse P to get the correct order.
        P = P[::-1]

        # Construct Q.
        j = n_minus - 1
        for i in range(n_plus):
            while j >= 0 and \
                    _h_kern(i, j, Zplus, Zminus, n_plus, eps2) - h_med < -Am_eps:
                j -= 1
            Q[i] = j + 1

        # Compute sumP and sumQ.
        sumP = np.sum(P) + n_plus
        sumQ = np.sum(Q)

        if medc_idx <= sumP - 1:
            R = P
            Rtot = sumP
        elif medc_idx > sumQ - 1:
            L = Q
            Ltot = sumQ
        else:
            return h_med

    A = _finalize_h_kernel_sweep(L, R, Zplus, Zminus, n_plus, eps2)
    return A[medc_idx - Ltot]


def _medcouple_1d(y, use_fast=True):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : np.ndarray
        1-d data to compute use in the estimator.
    use_fast : bool
        Whether to use the O(n log n) implementation. Defaults to True.

    Returns
    -------
    mc : float
        The medcouple statistic
    """
    y = np.squeeze(y)
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    if use_fast:
        return _medcouple_nlogn(y)
    else:
        return _medcouple_1d_legacy(y)


def medcouple(y, axis=0, use_fast=True):
    """
    Calculate the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : {int, None}
        Axis along which the medcouple statistic is computed.  If `None`, the
        entire array is used.
    use_fast : bool
        Whether to use the faster O(N log N) implementation. Default is True.
        To use the legacy O(N**2) version, set to False.

    Returns
    -------
    mc : float or ndarray
        The medcouple statistic.

    Notes
    -----
    The legacy algorithm (``use_fast=False``) uses an O(N**2) implementation
    which provides exact results and is reliable for all dataset sizes,
    including small inputs and cases with ties. However, it requires a O(N**2)
    memory allocations, and so may not work for very large arrays (N>10000).

    The fast algorithm (``use_fast=True``) implements an O(N log N)
    approximation which is optimized for large datasets. **It is not intended
    for small sample sizes (N < 10)** or datasets with a high proportion of
    ties, as it may yield numerically unstable or inaccurate results in these
    cases. For such inputs, prefer ``use_fast=False`` to ensure correctness.

    If NaNs are present in the input when use_fast=True, the result will be
    NaN. To preserve legacy behavior, a number may be returned when
    use_fast=False.

    If the size of ``y`` is less than 3 and ``use_fast=True``, the result will
    be NaN. To preserve legacy behavior, a value may be returned when
    ``use_fast=False``.

    Small numerical differences are possible based on the choice of algorithm.

    .. [*] Guy Brys, Mia Hubert and Anja Struyf (2004) A Robust Measure
       of Skewness; JCGS 13 (4), 996-1017.

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    y = np.asarray(y, dtype=np.double)  # GH 4243
    if axis is None:
        return _medcouple_1d(y.ravel(), use_fast=use_fast)

    return np.apply_along_axis(_medcouple_1d, axis, y, use_fast=use_fast)
