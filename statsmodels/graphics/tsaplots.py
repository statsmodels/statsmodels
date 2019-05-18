"""Correlation plot functions."""


import numpy as np

from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf, pacf


def _prepare_data_corr_plot(x, lags, zero):
    zero = bool(zero)
    irregular = False if zero else True
    if lags is None:
        # GH 4663 - use a sensible default value
        nobs = x.shape[0]
        lim = min(int(np.ceil(10 * np.log10(nobs))), nobs - 1)
        lags = np.arange(not zero, lim + 1)
    elif np.isscalar(lags):
        lags = np.arange(not zero, int(lags) + 1)  # +1 for zero lag
    else:
        irregular = True
        lags = np.asanyarray(lags).astype(np.int)
    nlags = lags.max(0)

    return lags, nlags, irregular


def _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines,
               vlines_kwargs, **kwargs):
    if irregular:
        acf_x = acf_x[lags]
        if confint is not None:
            confint = confint[lags]

    if use_vlines:
        ax.vlines(lags, [0], acf_x, **vlines_kwargs)
        ax.axhline(**kwargs)

    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    if 'ls' not in kwargs:
        # gh-2369
        kwargs.setdefault('linestyle', 'None')
    ax.margins(.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.set_title(title)

    if confint is not None:
        if lags[0] == 0:
            lags = lags[1:]
            confint = confint[1:]
            acf_x = acf_x[1:]
        lags = lags.astype(np.float)
        lags[0] -= 0.5
        lags[-1] += 0.5
        ax.fill_between(lags, confint[:, 0] - acf_x,
                        confint[:, 1] - acf_x, alpha=.25)


def plot_acf(x, ax=None, lags=None, alpha=.05, use_vlines=True, unbiased=False,
             fft=False, title='Autocorrelation', zero=True,
             vlines_kwargs=None, **kwargs):
    """Plot the autocorrelation function

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    x : array_like
        Array of time-series values
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : int or array_like, optional
        int or Array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
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
    title : str, optional
        Title to place on plot.  Default is 'Autocorrelation'
    zero : bool, optional
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
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

    Notes
    -----
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    kwargs is used to pass matplotlib optional arguments to both the line
    tracing the autocorrelations and for the horizontal line at 0. These
    options must be valid for a Line2D object.

    vlines_kwargs is used to pass additional optional arguments to the
    vertical lines connecting each autocorrelation to the axis.  These options
    must be valid for a LineCollection object.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    >>> dta = sm.datasets.sunspots.load_pandas().data
    >>> dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    >>> del dta["YEAR"]
    >>> sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
    >>> plt.show()

    .. plot:: plots/graphics_tsa_plot_acf.py
    """
    fig, ax = utils.create_mpl_ax(ax)

    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs

    confint = None
    # acf has different return type based on alpha
    if alpha is None:
        acf_x = acf(x, nlags=nlags, alpha=alpha, fft=fft,
                    unbiased=unbiased)
    else:
        acf_x, confint = acf(x, nlags=nlags, alpha=alpha, fft=fft,
                             unbiased=unbiased)

    _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines,
               vlines_kwargs, **kwargs)

    return fig


def plot_pacf(x, ax=None, lags=None, alpha=.05, method='ywunbiased',
              use_vlines=True, title='Partial Autocorrelation', zero=True,
              vlines_kwargs=None, **kwargs):
    """
    Plot the partial autocorrelation function

    Parameters
    ----------
    x : array_like
        Array of time-series values
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : int or array_like, optional
        int or Array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))
    method : {'ywunbiased', 'ywmle', 'ols'}
        Specifies which method for the calculations to use:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf. Default.
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction

    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    title : str, optional
        Title to place on plot.  Default is 'Partial Autocorrelation'
    zero : bool, optional
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
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

    Notes
    -----
    Plots lags on the horizontal and the correlations on vertical axis.
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    kwargs is used to pass matplotlib optional arguments to both the line
    tracing the autocorrelations and for the horizontal line at 0. These
    options must be valid for a Line2D object.

    vlines_kwargs is used to pass additional optional arguments to the
    vertical lines connecting each autocorrelation to the axis.  These options
    must be valid for a LineCollection object.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    >>> dta = sm.datasets.sunspots.load_pandas().data
    >>> dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    >>> del dta["YEAR"]
    >>> sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
    >>> plt.show()

    .. plot:: plots/graphics_tsa_plot_pacf.py
    """
    fig, ax = utils.create_mpl_ax(ax)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)

    confint = None
    if alpha is None:
        acf_x = pacf(x, nlags=nlags, alpha=alpha, method=method)
    else:
        acf_x, confint = pacf(x, nlags=nlags, alpha=alpha, method=method)

    _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines,
               vlines_kwargs, **kwargs)

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
    xticklabels : list of str
        List of season labels, one for each group.
    ylabel : str
        Lable for y axis
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    """
    fig, ax = utils.create_mpl_ax(ax)
    start = 0
    ticks = []
    for season, df in grouped_x:
        df = df.copy()  # or sort balks for series. may be better way
        df.sort_index()
        nobs = len(df)
        x_plot = np.arange(start, start + nobs)
        ticks.append(x_plot.mean())
        ax.plot(x_plot, df.values, 'k')
        ax.hlines(df.values.mean(), x_plot[0], x_plot[-1], colors='r',
                  linewidth=3)
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
    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',
    ...                                 dta.index.values)))
    >>> dta.index = pd.DatetimeIndex(dates, freq='MS')
    >>> fig = sm.graphics.tsa.month_plot(dta)

    .. plot:: plots/graphics_tsa_month_plot.py
    """

    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq="M")
    else:
        from pandas import Series, PeriodIndex
        x = Series(x, index=PeriodIndex(dates, freq="M"))

    xticklabels = ['j', 'f', 'm', 'a', 'm', 'j', 'j', 'a', 's', 'o', 'n', 'd']
    return seasonal_plot(x.groupby(lambda y: y.month), xticklabels,
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

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd

    >>> dta = sm.datasets.elnino.load_pandas().data
    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    >>> dta = dta.set_index('YEAR').T.unstack()
    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',
    ...                                 dta.index.values)))
    >>> dta.index = dates.to_period('Q')
    >>> fig = sm.graphics.tsa.quarter_plot(dta)

    .. plot:: plots/graphics_tsa_quarter_plot.py
    """

    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq="Q")
    else:
        from pandas import Series, PeriodIndex
        x = Series(x, index=PeriodIndex(dates, freq="Q"))

    xticklabels = ['q1', 'q2', 'q3', 'q4']
    return seasonal_plot(x.groupby(lambda y: y.quarter), xticklabels,
                         ylabel=ylabel, ax=ax)
