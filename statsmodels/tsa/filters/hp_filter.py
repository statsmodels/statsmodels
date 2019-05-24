from __future__ import absolute_import

from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
from ._utils import _maybe_get_pandas_wrapper


def hpfilter(X, lamb=1600):
    """
    Hodrick-Prescott filter

    Parameters
    ----------
    X : array-like
        The 1d ndarray timeseries to filter of length (nobs,) or (nobs,1)
    lamb : float
        The Hodrick-Prescott smoothing parameter. A value of 1600 is
        suggested for quarterly data. Ravn and Uhlig suggest using a value
        of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly
        data.

    Returns
    -------
    cycle : array
        The estimated cycle in the data given lamb.
    trend : array
        The estimated trend in the data given lamb.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.DatetimeIndex(start='1959Q1', end='2009Q4', freq='Q')
    >>> dta.set_index(index, inplace=True)

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

    Notes
    -----
    The HP filter removes a smooth trend, `T`, from the data `X`. by solving

    min sum((X[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)
     T   t

    Here we implemented the HP filter as a ridge-regression rule using
    scipy.sparse. In this sense, the solution can be written as

    T = inv(I - lamb*K'K)X

    where I is a nobs x nobs identity matrix, and K is a (nobs-2) x nobs matrix
    such that

    K[i,j] = 1 if i == j or i == j + 2
    K[i,j] = -2 if i == j + 1
    K[i,j] = 0 otherwise

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
    statsmodels.tsa.filters.cf_filter.cffilter
    statsmodels.tsa.seasonal.seasonal_decompose

    References
    ----------
    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An
        Empricial Investigation." `Carnegie Mellon University discussion
        paper no. 451`.
    Ravn, M.O and H. Uhlig. 2002. "Notes On Adjusted the Hodrick-Prescott
        Filter for the Frequency of Observations." `The Review of Economics and
        Statistics`, 84(2), 371-80.
    """
    _pandas_wrapper = _maybe_get_pandas_wrapper(X)
    X = np.asarray(X, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)
    I = sparse.eye(nobs, nobs)  # noqa:E741
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = sparse.dia_matrix((data, offsets), shape=(nobs-2,nobs))

    use_umfpack = True
    trend = spsolve(I+lamb*K.T.dot(K), X, use_umfpack=use_umfpack)

    cycle = X-trend
    if _pandas_wrapper is not None:
        return _pandas_wrapper(cycle), _pandas_wrapper(trend)
    return cycle, trend
