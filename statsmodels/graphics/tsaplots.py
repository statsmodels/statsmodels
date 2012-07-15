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
        specifies which method for the calculations to use,
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
