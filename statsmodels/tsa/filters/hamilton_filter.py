"""
Hamilton (2018) filter — regression-based trend-cycle decomposition.

Reference
---------
Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
*Review of Economics and Statistics*, 100(5), 831-843.
"""

import numpy as np

from statsmodels.tools.validation import PandasWrapper, array_like


def hamilton_filter(x, h=8, p=4):
    r"""
    Hamilton (2018) regression-based trend-cycle decomposition.

    Decomposes a time series into trend and cycle components by projecting
    ``h``-period-ahead values on the ``p`` most recent lags and a constant:

    .. math::

        y_{t+h} = \alpha_0 + \alpha_1 y_t + \alpha_2 y_{t-1}
                  + \cdots + \alpha_p y_{t-p+1} + v_{t+h}

    The cycle at time ``t + h`` is the residual :math:`\hat{v}_{t+h}` from
    this OLS regression.  The trend is the corresponding fitted value.

    Parameters
    ----------
    x : array_like
        The time series to decompose, 1-d with at least ``p + h`` observations.
    h : int, optional
        Forecast horizon used in the projection.  Hamilton recommends:

        * **8** for quarterly data (two years ahead, default)
        * **24** for monthly data (two years ahead)
        * **2** for annual data

    p : int, optional
        Number of lagged values of ``x`` to include as regressors.
        Hamilton recommends **4** for quarterly data (one year of lags) and
        **12** for monthly data.

    Returns
    -------
    cycle : ndarray or Series
        Estimated cyclical component.  The first ``p + h - 1`` values are
        ``NaN`` because no regression can be formed for those periods.
    trend : ndarray or Series
        Estimated trend component.  The first ``p + h - 1`` values are
        likewise ``NaN``.

    See Also
    --------
    statsmodels.tsa.filters.hp_filter.hpfilter
        Hodrick-Prescott filter.
    statsmodels.tsa.filters.bk_filter.bkfilter
        Baxter-King bandpass filter.
    statsmodels.tsa.filters.cf_filter.cffilter
        Christiano-Fitzgerald asymmetric filter.

    Notes
    -----
    Unlike the HP filter, the Hamilton filter uses only lagged information and
    produces stationary residuals when the underlying series is I(1) or I(2).
    Hamilton (2018) shows that the HP filter introduces spurious cyclical
    dynamics; the regression-based filter avoids this by construction.

    The regression is estimated once on all available observations (i.e. this
    is *not* a rolling regression).  With ``h = 8`` and ``p = 4`` (the
    quarterly defaults), the first ``11`` observations of the output are
    ``NaN``.

    References
    ----------
    Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott
    Filter. *Review of Economics and Statistics*, 100(5), 831-843.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.period_range('1959Q1', '2009Q3', freq='Q')
    >>> dta.set_index(index, inplace=True)
    >>> cycle, trend = sm.tsa.filters.hamilton_filter(dta['realgdp'], h=8, p=4)
    """
    pw = PandasWrapper(x)
    x = array_like(x, "x", ndim=1)
    T = len(x)

    if h < 1:
        raise ValueError("h must be a positive integer.")
    if p < 1:
        raise ValueError("p must be a positive integer.")
    if T < p + h:
        raise ValueError(
            f"x must have at least p + h = {p + h} observations; "
            f"got {T}."
        )

    n_obs = T - p - h + 1

    # Dependent variable: y_{t+h} for t = p-1, p, ..., T-h-1
    Y = x[p + h - 1:]          # shape (n_obs,)

    # Regressors: [y_t, y_{t-1}, ..., y_{t-p+1}, 1]
    cols = [x[p - 1 - j : T - h - j] for j in range(p)]
    cols.append(np.ones(n_obs))
    X = np.column_stack(cols)   # shape (n_obs, p+1)

    # OLS estimate
    params, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    fitted = X @ params         # trend values at t+h
    resid = Y - fitted          # cycle values at t+h

    # Place results at the correct positions in output arrays
    trend = np.full(T, np.nan)
    cycle = np.full(T, np.nan)
    trend[p + h - 1:] = fitted
    cycle[p + h - 1:] = resid

    return pw.wrap(cycle, append="cycle"), pw.wrap(trend, append="trend")
