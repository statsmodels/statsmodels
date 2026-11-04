from statsmodels.compat.scipy import SP_LT_2

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from statsmodels.tools.validation import (
    PandasWrapper,
    array_like,
    bool_like,
    float_like,
)
from statsmodels.tsa.filters.filtertools import CycleTrendResult
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter


def hpfilter(x, lamb=1600, one_sided=False):
    """
    Hodrick-Prescott filter

    Parameters
    ----------
    x : array_like
        The time series to filter, 1-d.
    lamb : float, optional
        The Hodrick-Prescott smoothing parameter. A value of 1600 is
        suggested for quarterly data. Ravn and Uhlig suggest using a value
        of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly
        data.
    one_sided : bool, optional
        If True, use the one-sided HP filter where the trend at time t only
        depends on the observations up to and including t. The default is the
        usual two-sided filter. The first two observations are fit exactly,
        so the trend is equal to x and the cycle is 0, and the first
        non-trivial value of the cycle is at t=2. x must not be empty and must
        not contain NaN or inf values.

        .. versionadded:: 0.15.1

    Returns
    -------
    CycleTrendResult
        A result object with fields:

        cycle : ndarray or Series
            The estimated cycle in the data given lamb.
        trend : ndarray or Series
            The estimated trend in the data given lamb.

    Raises
    ------
    TypeError
        If one_sided is not a bool.
    ValueError
        If lamb is not positive. If one_sided is True and x is empty or
        contains NaN or inf.

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
        Baxter-King filter.
    statsmodels.tsa.filters.cf_filter.cffilter
        The Christiano Fitzgerald asymmetric, random walk filter.
    statsmodels.tsa.seasonal.seasonal_decompose
        Decompose a time series using moving averages.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.

    Notes
    -----
    The HP filter removes a smooth trend, `T`, from the data `x` by solving

    min sum((x[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)
     T   t

    Here we implemented the HP filter as a ridge-regression rule using
    scipy.sparse. In this sense, the solution can be written as

    T = inv(I + lamb*K'K)x

    where I is a nobs x nobs identity matrix, and K is a (nobs-2) x nobs matrix
    such that

    K[i,j] = 1 if i == j or i == j + 2
    K[i,j] = -2 if i == j + 1
    K[i,j] = 0 otherwise

    The one-sided filter uses the state space form of the HP filter

    x[t] = T[t] + c[t],                  c[t] ~ N(0, 1)
    T[t] = 2*T[t-1] - T[t-2] + e[t],     e[t] ~ N(0, 1/lamb)

    The trend is the filtered state of the Kalman filter with an exact diffuse
    initialization. The one-sided trend at time t is identical to the last
    value of the two-sided trend computed from x[:t+1] (Stock and Watson,
    1999). Because the initialization is exactly diffuse, the first
    observations differ slightly (by up to about 1e-3 for the real GDP series
    in the statsmodels tests, which has a level of about 3000) from
    implementations that approximate it with a large finite variance, for
    example ``hp1`` in the R package hpfilter, which uses a prior variance of
    1e5.

    See the notebook `Time Series Filters
    <../examples/notebooks/generated/tsa_filters.html>`__ for an overview.

    References
    ----------
    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An
        Empirical Investigation." *Carnegie Mellon University discussion
        paper no. 451*.
    Ravn, M.O and H. Uhlig. 2002. "On Adjusting the Hodrick-Prescott
        Filter for the Frequency of Observations." *The Review of Economics
        and Statistics*, 84(2), 371-80.
    Stock, J.H. and M.W. Watson. 1999. "Forecasting inflation." *Journal of
        Monetary Economics*, 44(2), 293-335.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.period_range('1959Q1', '2009Q3', freq='Q')
    >>> dta = dta.set_index(index)

    >>> cycle, trend = sm.tsa.filters.hpfilter(dta.realgdp, 1600)
    >>> gdp_decomp = dta[['realgdp']]
    >>> gdp_decomp["cycle"] = cycle
    >>> gdp_decomp["trend"] = trend

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax,
    ...                                                      fontsize=16)
    >>> plt.show()

    .. plot:: plots/hpf_plot.py
    """
    pw = PandasWrapper(x)
    x = array_like(x, "x", ndim=1)
    lamb = float_like(lamb, "lamb")
    if not lamb > 0:
        # the smoothing penalty must be positive: lamb = 0 makes the trend
        # identical to the series and negative values solve a non-convex
        # system, so both silently degenerate
        raise ValueError(f"lamb must be a positive number, got {lamb}")
    one_sided = bool_like(one_sided, "one_sided", strict=True)
    if one_sided:
        if x.shape[0] == 0:
            raise ValueError("x must not be empty when one_sided is True")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must not contain NaN or inf when one_sided is True")
        trend = _hpfilter_one_sided(x, lamb)
        cycle = x - trend
        return CycleTrendResult(
            pw.wrap(cycle, append="cycle"), pw.wrap(trend, append="trend")
        )
    nobs = len(x)
    offsets = np.array([0, 1, 2])
    data = np.repeat([[1.], [-2.], [1.]], nobs, axis=1)

    if SP_LT_2:
        eye = sparse.eye(nobs, nobs)
        K = sparse.dia_matrix((data, offsets), shape=(nobs - 2, nobs))
    else:
        eye = sparse.eye_array(nobs, nobs)
        K = sparse.dia_array((data, offsets), shape=(nobs - 2, nobs))

    use_umfpack = True
    trend = spsolve((eye+lamb*K.T.dot(K)).tocsc(), x, use_umfpack=use_umfpack)

    cycle = x - trend
    return CycleTrendResult(
        pw.wrap(cycle, append="cycle"), pw.wrap(trend, append="trend")
    )


def _hpfilter_one_sided(x, lamb):
    """
    One-sided HP trend from the Kalman filter of the state space form
    """
    kfilter = KalmanFilter(k_endog=1, k_states=2, k_posdef=1)
    # bind reshapes 1-d input in place with older NumPy, so pass a 2-d copy
    # to leave x unchanged
    kfilter.bind(np.array(x)[:, None])
    kfilter["design"] = np.array([[1.0, 0.0]])
    # The filtered state under an exactly diffuse prior only depends on the ratio
    # of the two variances. Scaling them as (lamb, 1) instead of (1, 1 / lamb)
    # gives the same trend but avoids the loss of precision of a tiny state
    # variance when lamb is large.
    kfilter["obs_cov"] = np.array([[lamb]])
    kfilter["transition"] = np.array([[2.0, -1.0], [1.0, 0.0]])
    kfilter["selection"] = np.array([[1.0], [0.0]])
    kfilter["state_cov"] = np.array([[1.0]])
    kfilter.initialize_diffuse()
    return np.array(kfilter.filter().filtered_state[0])
