from statsmodels.compat.python import range

import numpy as np
from ._utils import _maybe_get_pandas_wrapper

# the data is sampled quarterly, so cut-off frequency of 18

# Wn is normalized cut-off freq
#Cutoff frequency is that frequency where the magnitude response of the filter
# is sqrt(1/2.). For butter, the normalized cutoff frequency Wn must be a
# number between  0 and 1, where 1 corresponds to the Nyquist frequency, p
# radians per sample.

#NOTE: uses a loop, could probably be sped-up for very large datasets
def cffilter(X, low=6, high=32, drift=True):
    """
    Christiano Fitzgerald asymmetric, random walk filter

    Parameters
    ----------
    X : array-like
        1 or 2d array to filter. If 2d, variables are assumed to be in columns.
    low : float
        Minimum period of oscillations. Features below low periodicity are
        filtered out. Default is 6 for quarterly data, giving a 1.5 year
        periodicity.
    high : float
        Maximum period of oscillations. Features above high periodicity are
        filtered out. Default is 32 for quarterly data, giving an 8 year
        periodicity.
    drift : bool
        Whether or not to remove a trend from the data. The trend is estimated
        as np.arange(nobs)*(X[-1] - X[0])/(len(X)-1)

    Returns
    -------
    cycle : array
        The features of `X` between periodicities given by low and high
    trend : array
        The trend in the data with the cycles removed.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.DatetimeIndex(start='1959Q1', end='2009Q4', freq='Q')
    >>> dta.set_index(index, inplace=True)

    >>> cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cf_cycles.plot(ax=ax, style=['r--', 'b-'])
    >>> plt.show()

    .. plot:: plots/cff_plot.py

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
    statsmodels.tsa.filters.hp_filter.hpfilter
    statsmodels.tsa.seasonal.seasonal_decompose

    """
    #TODO: cythonize/vectorize loop?, add ability for symmetric filter,
    #      and estimates of theta other than random walk.
    if low < 2:
        raise ValueError("low must be >= 2")
    _pandas_wrapper = _maybe_get_pandas_wrapper(X)
    X = np.asanyarray(X)
    if X.ndim == 1:
        X = X[:,None]
    nobs, nseries = X.shape
    a = 2*np.pi/high
    b = 2*np.pi/low

    if drift: # get drift adjusted series
        X = X - np.arange(nobs)[:,None]*(X[-1] - X[0])/(nobs-1)

    J = np.arange(1,nobs+1)
    Bj = (np.sin(b*J)-np.sin(a*J))/(np.pi*J)
    B0 = (b-a)/np.pi
    Bj = np.r_[B0,Bj][:,None]
    y = np.zeros((nobs,nseries))

    for i in range(nobs):

        B = -.5*Bj[0] -np.sum(Bj[1:-i-2])
        A = -Bj[0] - np.sum(Bj[1:-i-2]) - np.sum(Bj[1:i]) - B
        y[i] = Bj[0] * X[i] + np.dot(Bj[1:-i-2].T,X[i+1:-1]) + B*X[-1] + \
                np.dot(Bj[1:i].T, X[1:i][::-1]) + A*X[0]
    y = y.squeeze()

    cycle, trend = y, X.squeeze()-y

    if _pandas_wrapper is not None:
        return _pandas_wrapper(cycle), _pandas_wrapper(trend)

    return cycle, trend

if __name__ == "__main__":
    import statsmodels as sm
    dta = sm.datasets.macrodata.load().data[['infl','tbilrate']].view((float,2))[1:]
    cycle, trend = cffilter(dta, 6, 32, drift=True)
    dta = sm.datasets.macrodata.load().data['tbilrate'][1:]
    cycle2, trend2 = cffilter(dta, 6, 32, drift=True)

