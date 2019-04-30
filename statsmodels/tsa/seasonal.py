"""
Seasonal Decomposition by Moving Averages
"""
from statsmodels.compat.python import lmap, range, iteritems
import numpy as np
from pandas.core.nanops import nanmean as pd_nanmean
from .filters._utils import (_maybe_get_pandas_wrapper_freq,
                             _maybe_get_pandas_wrapper)
from .filters.filtertools import convolution_filter
from statsmodels.tsa.tsatools import freq_to_period


def seasonal_mean(x, freq):
    """
    Return means for each period in x. freq is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    return np.array([pd_nanmean(x[i::freq], axis=0) for i in range(freq)])


def _extrapolate_trend(trend, npoints):
    """
    Replace nan values on trend's end-points with least-squares extrapolated
    values with regression considering npoints closest defined points.
    """
    front = next(i for i, vals in enumerate(trend)
                 if not np.any(np.isnan(vals)))
    back = trend.shape[0] - 1 - next(i for i, vals in enumerate(trend[::-1])
                                     if not np.any(np.isnan(vals)))
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)

    k, n = np.linalg.lstsq(
        np.c_[np.arange(front, front_last), np.ones(front_last - front)],
        trend[front:front_last], rcond=-1)[0]
    extra = (np.arange(0, front) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[:front] = extra

    k, n = np.linalg.lstsq(
        np.c_[np.arange(back_first, back), np.ones(back - back_first)],
        trend[back_first:back], rcond=-1)[0]
    extra = (np.arange(back + 1, trend.shape[0]) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[back + 1:] = extra

    return trend


def seasonal_decompose(x, model="additive", filt=None, freq=None, two_sided=True,
                       extrapolate_trend=0):
    """
    Seasonal decomposition using moving averages

    Parameters
    ----------
    x : array-like
        Time series. If 2d, individual series are in columns.
    model : str {"additive", "multiplicative"}
        Type of seasonal component. Abbreviations are accepted.
    filt : array-like
        The filter coefficients for filtering out the seasonal component.
        The concrete moving average method used in filtering is determined by two_sided.
    freq : int, optional
        Frequency of the series. Must be used if x is not a pandas object.
        Overrides default periodicity of x if x is a pandas
        object with a timeseries index.
    two_sided : bool
        The moving average method used in filtering.
        If True (default), a centered moving average is computed using the filt.
        If False, the filter coefficients are for past values only.
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends (or the single one
        if two_sided is False) considering this many (+1) closest points.
        If set to 'freq', use `freq` closest points. Setting this parameter
        results in no NaN values in trend or resid components.

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
    statsmodels.tsa.filters.bk_filter.bkfilter
    statsmodels.tsa.filters.cf_filter.xffilter
    statsmodels.tsa.filters.hp_filter.hpfilter
    statsmodels.tsa.filters.convolution_filter
    """
    if freq is None:
        _pandas_wrapper, pfreq = _maybe_get_pandas_wrapper_freq(x)
    else:
        _pandas_wrapper = _maybe_get_pandas_wrapper(x)
        pfreq = None
    x = np.asanyarray(x).squeeze()
    nobs = len(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("This function does not handle missing values")
    if model.startswith('m'):
        if np.any(x <= 0):
            raise ValueError("Multiplicative seasonality is not appropriate "
                             "for zero and negative values")

    if freq is None:
        if pfreq is not None:
            pfreq = freq_to_period(pfreq)
            freq = pfreq
        else:
            raise ValueError("You must specify a freq or x must be a "
                             "pandas object with a timeseries index with "
                             "a freq not set to None")

    if filt is None:
        if freq % 2 == 0:  # split weights at ends
            filt = np.array([.5] + [1] * (freq - 1) + [.5]) / freq
        else:
            filt = np.repeat(1./freq, freq)

    nsides = int(two_sided) + 1
    trend = convolution_filter(x, filt, nsides)

    if extrapolate_trend == 'freq':
        extrapolate_trend = freq - 1

    if extrapolate_trend > 0:
        trend = _extrapolate_trend(trend, extrapolate_trend + 1)

    if model.startswith('m'):
        detrended = x / trend
    else:
        detrended = x - trend

    period_averages = seasonal_mean(detrended, freq)

    if model.startswith('m'):
        period_averages /= np.mean(period_averages, axis=0)
    else:
        period_averages -= np.mean(period_averages, axis=0)

    seasonal = np.tile(period_averages.T, nobs // freq + 1).T[:nobs]

    if model.startswith('m'):
        resid = x / seasonal / trend
    else:
        resid = detrended - seasonal

    results = lmap(_pandas_wrapper, [seasonal, trend, resid, x])
    return DecomposeResult(seasonal=results[0], trend=results[1],
                           resid=results[2], observed=results[3])


class DecomposeResult(object):
    def __init__(self, **kwargs):
        for key, value in iteritems(kwargs):
            setattr(self, key, value)
        self.nobs = len(self.observed)

    def plot(self):
        from statsmodels.graphics.utils import _import_mpl
        plt = _import_mpl()
        fig, axes = plt.subplots(4, 1, sharex=True)
        if hasattr(self.observed, 'plot'):  # got pandas use it
            self.observed.plot(ax=axes[0], legend=False)
            axes[0].set_ylabel('Observed')
            self.trend.plot(ax=axes[1], legend=False)
            axes[1].set_ylabel('Trend')
            self.seasonal.plot(ax=axes[2], legend=False)
            axes[2].set_ylabel('Seasonal')
            self.resid.plot(ax=axes[3], legend=False)
            axes[3].set_ylabel('Residual')
        else:
            axes[0].plot(self.observed)
            axes[0].set_ylabel('Observed')
            axes[1].plot(self.trend)
            axes[1].set_ylabel('Trend')
            axes[2].plot(self.seasonal)
            axes[2].set_ylabel('Seasonal')
            axes[3].plot(self.resid)
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Time')
            axes[3].set_xlim(0, self.nobs)

        fig.tight_layout()
        return fig


if __name__ == "__main__":
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                  530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                  232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    results = seasonal_decompose(x, freq=4)

    from pandas import DataFrame, date_range
    data = DataFrame(x, date_range(start='1/1/1951',
                                   periods=len(x),
                                   freq='Q'))

    res = seasonal_decompose(data)
