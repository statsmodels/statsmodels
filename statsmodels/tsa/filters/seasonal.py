"""
Seasonal Decomposition by Moving Averages
"""
import numpy as np
from pandas.core.nanops import nanmean as pd_nanmean
from .utils import _maybe_get_pandas_wrapper_freq
from .filtertools import convolution_filter
from statsmodels.tsa.tsatools import freq_to_period
from statsmodels.tools.tools import Bunch


def seasonal_decompose(X, model="additive", filt=None, freq=None):
    """
    Parameters
    ----------
    X : array-like
        Time series
    model : str {"additive", "multiplicative"}
        Type of seasonal component. Abbreviations are accepted.
    filt : array-like
        The filter coefficients for filtering out the seasonal component.
        The default is a symmetric moving average.
    freq : int, optional
        Frequency of the series. Must be used if X is not a pandas
        object with a timeseries index.

    Returns
    -------
    results : obj
        A object with seasonal, trend, and resid attributes.

    Notes
    -----
    This is a naive decomposition. More sophisticated methods should
    be preferred.

    The additive model is Y[t] = T[t] + S[t] + e[t]

    The multiplicative model is Y[t] = T[t] * S[t] * e[t]

    The seasonal component is first removed by applying a convolution
    filter to the data. The average of this smoothed series for each
    period is the returned seasonal component.

    See Also
    --------
    statsmodels.tsa.filters.convolution_filter
    """
    if not np.all(np.isfinite(X)):
        raise ValueError("This function does not handle missing values")
    if model.startswith('m'):
        if np.any(X <= 0):
            raise ValueError("Multiplicative seasonality is not appropriate "
                             "for zero and negative values")

    _pandas_wrapper, pfreq = _maybe_get_pandas_wrapper_freq(X)
    if pfreq is not None:
        pfreq = freq_to_period(pfreq)
        if freq and pfreq != freq:
            raise ValueError("Inferred frequency of index and frequency "
                             "don't match. This function does not re-sample")
        else:
            freq = pfreq

    elif freq is None:
        raise ValueError("You must specify a freq or X must be a "
                         "pandas object with a timeseries index")

    X = np.asanyarray(X).squeeze()
    nobs = len(X)

    if filt is None:
        if freq % 2 == 0:  # split weights at ends
            filt = np.array([.5] + [1] * (freq - 1) + [.5]) / freq
        else:
            filt = np.repeat(1./freq, freq)
    drop_idx = freq // 2
    trend = convolution_filter(X, filt)

    # nan pad for conformability - convolve doesn't do it
    nan_pad = lambda y : np.r_[[np.nan] * drop_idx, y, [np.nan] * drop_idx]
    trend = nan_pad(trend)
    if model.startswith('m'):
        detrended = X/trend
    else:
        detrended = X - trend

    period_averages = np.array([pd_nanmean(detrended[i::freq])
                                for i in range(freq)])
    if model.startswith('m'):
        period_averages /= np.mean(period_averages)
    else:
        period_averages -= np.mean(period_averages)

    seasonal = np.tile(period_averages, nobs // freq + 1)[:nobs]

    if model.startswith('m'):
        resid = X / seasonal / trend
    else:
        resid = detrended - seasonal

    results = map(_pandas_wrapper, [seasonal, trend, resid])
    return Bunch(seasonal=results[0], trend=results[1], resid=results[2])


if __name__ == "__main__":
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                  530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                  232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    results = decompose(x)
