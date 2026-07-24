"""
Hamilton (2018) filter — regression-based trend-cycle decomposition.

References
----------
Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
*Review of Economics and Statistics*, 100(5), 831-843.
"""

import numpy as np

from statsmodels.tools.validation import PandasWrapper, array_like, int_like
from statsmodels.tsa.ar_model import AutoReg


def hamilton_filter(x, h=8, p=4):
    r"""
    Hamilton (2018) regression-based trend-cycle decomposition [Hamilton2018]_.

    Parameters
    ----------
    x : array_like
        The time series to decompose, 1-d with at least ``2 p + h`` observations.
    h : int, optional
        Forecast horizon used in the projection.  Hamilton recommends:

        * **8** for quarterly data (two years ahead, default)
        * **24** for monthly data (two years ahead)
        * **2** for annual data

    p : int, optional
        Number of lagged values of ``x`` to include as regressors.
        Hamilton recommends **4** for quarterly data (one year of lags),
        **12** for monthly data (one year of lags), and **1** for annual
        data (following the same one-year-of-lags rule).

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
    statsmodels.tsa.ar_model.AutoReg
        Autoregression estimation using OLS.

    Notes
    -----
    Decomposes a time series into trend and cycle components by projecting
    ``h``-period-ahead values on the ``p`` most recent lags and a constant:

    .. math::

        y_{t+h} = \alpha_0 + \alpha_1 y_t + \alpha_2 y_{t-1}
                  + \cdots + \alpha_p y_{t-p+1} + v_{t+h}

    The cycle at time ``t + h`` is the residual :math:`\hat{v}_{t+h}` from
    this OLS regression.  The trend is the corresponding fitted value.

    Unlike the HP filter, the Hamilton filter uses only lagged information and
    produces stationary residuals when the underlying series is I(1) or I(2).
    Hamilton (2018) shows that the HP filter introduces spurious cyclical
    dynamics; the regression-based filter avoids this by construction.

    The regression is estimated once on all available observations (i.e. this
    is *not* a rolling regression).  With ``h = 8`` and ``p = 4`` (the
    quarterly defaults), the first ``11`` observations of the output are
    ``NaN``.

    ``x`` must have at least ``2 * p + h`` observations so that the
    regression has at least ``p + 1`` usable observations (one per
    parameter, including the constant).  At exactly this minimum length,
    the regression is exactly determined and produces a degenerate,
    zero-residual fit -- the returned ``cycle`` is identically zero and
    ``trend`` equals ``x`` over the non-``NaN`` range.  Provide more
    observations than this bare minimum for a meaningful decomposition.

    References
    ----------
    .. [Hamilton2018] Hamilton, J. D. (2018). Why You Should Never Use the
       Hodrick-Prescott Filter. *Review of Economics and Statistics*, 100(5), 831-843.

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
    h = int_like(h, "h", strict=True, optional=False)
    p = int_like(p, "p", strict=True, optional=False)
    T = len(x)

    if h < 1:
        raise ValueError("h must be a positive integer.")
    if p < 1:
        raise ValueError("p must be a positive integer.")
    if T < (2 * p + h):
        raise ValueError(
            f"x must have at least 2p + h = {2 * p + h} observations; got {T}."
        )

    _lags = np.arange(h, h + p)
    mod = AutoReg(x, lags=_lags, trend="c")
    res = mod.fit(use_t=True)

    fitted = res.fittedvalues  # trend values at t+h
    resid = res.resid  # cycle values at t+h

    # Place results at the correct positions in output arrays
    trend = np.full(T, np.nan)
    cycle = np.full(T, np.nan)
    trend[p + h - 1 :] = fitted
    cycle[p + h - 1 :] = resid

    return pw.wrap(cycle, append="cycle"), pw.wrap(trend, append="trend")
