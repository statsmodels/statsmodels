"""Correlation plot functions."""


import numpy as np

from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf, pacf

def plot_acf(x, ax=None, lags=None, alpha=.05, use_vlines=True, unbiased=False,
            fft=False, **kwargs):
    """Plot the autocorrelation function

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    x : array_like
        Array of time-series values
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : array_like, optional
        Array of lag values, used on horizontal axis.
        If not given, ``lags=np.arange(len(corr))`` is used.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula. If None, no confidence intervals are plotted.
    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    fft : bool, optional
        If True, computes the ACF via FFT.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    matplotlib.pyplot.xcorr
    matplotlib.pyplot.acorr
    mpl_examples/pylab_examples/xcorr_demo.py

    Notes
    -----
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    """
    fig, ax = utils.create_mpl_ax(ax)

    if lags is None:
        lags = np.arange(len(x))
        nlags = len(lags) - 1
    else:
        nlags = lags
        lags = np.arange(lags + 1) # +1 for zero lag

    acf_x, confint = acf(x, nlags=nlags, alpha=alpha, fft=fft,
                         unbiased=unbiased)

    if use_vlines:
        ax.vlines(lags, [0], acf_x, **kwargs)
        ax.axhline(**kwargs)

    # center the confidence interval TODO: do in acf?
    confint = confint - confint.mean(1)[:,None]
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    kwargs.setdefault('linestyle', 'None')
    ax.margins(.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.fill_between(lags, confint[:,0], confint[:,1], alpha=.25)
    ax.set_title("Autocorrelation")

    return fig

def plot_pacf(x, ax=None, lags=None, alpha=.05, method='ywm',
                use_vlines=True, **kwargs):
    """Plot the partial autocorrelation function

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    x : array_like
        Array of time-series values
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : array_like, optional
        Array of lag values, used on horizontal axis.
        If not given, ``lags=np.arange(len(corr))`` is used.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))
    method : 'ywunbiased' (default) or 'ywmle' or 'ols'
        specifies which method for the calculations to use:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction

    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    matplotlib.pyplot.xcorr
    matplotlib.pyplot.acorr
    mpl_examples/pylab_examples/xcorr_demo.py

    Notes
    -----
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    """
    fig, ax = utils.create_mpl_ax(ax)

    if lags is None:
        lags = np.arange(len(x))
        nlags = len(lags) - 1
    else:
        nlags = lags
        lags = np.arange(lags + 1) # +1 for zero lag

    acf_x, confint = pacf(x, nlags=nlags, alpha=alpha, method=method)

    if use_vlines:
        ax.vlines(lags, [0], acf_x, **kwargs)
        ax.axhline(**kwargs)

    # center the confidence interval TODO: do in acf?
    confint = confint - confint.mean(1)[:,None]
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    kwargs.setdefault('linestyle', 'None')
    ax.margins(.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.fill_between(lags, confint[:,0], confint[:,1], alpha=.25)
    ax.set_title("Partial Autocorrelation")

    return fig

def seasonal_plot(grouped_x, xticklabels, ylabel=None, ax=None):
    """
    Consider using one of month_plot or quarter_plot unless you need
    irregular plotting.

    Parameters
    ----------
    grouped_x : iterable of DataFrames
        Should be a GroupBy object (or similar pair of group_names and groups
        as DataFrames) with a DatetimeIndex or PeriodIndex
    """
    fig, ax = utils.create_mpl_ax(ax)
    start = 0
    ticks = []
    for season, df in grouped_x:
        df = df.copy() # or sort balks for series. may be better way
        df.sort()
        nobs = len(df)
        x_plot = np.arange(start, start + nobs)
        ticks.append(x_plot.mean())
        ax.plot(x_plot, df.values, 'k')
        ax.hlines(df.values.mean(), x_plot[0], x_plot[-1], colors='k')
        start += nobs

    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.margins(.1, .05)
    return fig


def month_plot(x, dates=None, ylabel=None, ax=None):
    """
    Seasonal plot of monthly data

    Parameters
    ----------
    x : array-like
        Seasonal data to plot. If dates is None, x must be a pandas object
        with a PeriodIndex or DatetimeIndex with a monthly frequency.
    dates : array-like, optional
        If `x` is not a pandas object, then dates must be supplied.
    ylabel : str, optional
        The label for the y-axis. Will attempt to use the `name` attribute
        of the Series.
    ax : matplotlib.axes, optional
        Existing axes instance.

    Returns
    -------
    matplotlib.Figure

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd

    >>> dta = sm.datasets.elnino.load_pandas().data
    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    >>> dta = dta.set_index('YEAR').T.unstack()
    >>> dates = map(lambda x : pd.datetools.parse('1 '+' '.join(x)),
    ...                                        dta.index.values)

    >>> dta.index = pd.DatetimeIndex(dates, freq='M')
    >>> fig = sm.graphics.tsa.month_plot(dta)

    .. plot:: plots/graphics_month_plot.py
    """
    from pandas import DataFrame

    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq="M")
    else:
        from pandas import Series, PeriodIndex
        x = Series(x, index=PeriodIndex(dates, freq="M"))

    xticklabels = ['j','f','m','a','m','j','j','a','s','o','n','d']
    return seasonal_plot(x.groupby(lambda y : y.month), xticklabels,
                         ylabel=ylabel, ax=ax)

def quarter_plot(x, dates=None, ylabel=None, ax=None):
    """
    Seasonal plot of quarterly data

    Parameters
    ----------
    x : array-like
        Seasonal data to plot. If dates is None, x must be a pandas object
        with a PeriodIndex or DatetimeIndex with a monthly frequency.
    dates : array-like, optional
        If `x` is not a pandas object, then dates must be supplied.
    ylabel : str, optional
        The label for the y-axis. Will attempt to use the `name` attribute
        of the Series.
    ax : matplotlib.axes, optional
        Existing axes instance.

    Returns
    -------
    matplotlib.Figure
    """
    from pandas import DataFrame

    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq="Q")
    else:
        from pandas import Series, PeriodIndex
        x = Series(x, index=PeriodIndex(dates, freq="Q"))

    xticklabels = ['q1', 'q2', 'q3', 'q4']
    return seasonal_plot(x.groupby(lambda y : y.quarter), xticklabels,
                         ylabel=ylabel, ax=ax)


if __name__ == "__main__":
    import pandas as pd

    #R code to run to load that dataset in this directory
    #data(co2)
    #library(zoo)
    #write.csv(as.data.frame(list(date=as.Date(co2), co2=coredata(co2))), "co2.csv", row.names=FALSE)
    co2 = pd.read_csv("co2.csv", index_col=0, parse_dates=True)
    month_plot(co2.co2)

    #will work when dates are sorted
    #co2 = sm.datasets.get_rdataset("co2", cache=True)

    x = pd.Series(np.arange(20),
                  index=pd.PeriodIndex(start='1/1/1990', periods=20, freq='Q'))
    quarter_plot(x)

