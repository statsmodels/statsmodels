"""Correlation plot functions."""


import numpy as np

from . import utils


#copied/moved from sandbox/tsa/example_arma.py
def plotacf(corr, ax=None, lags=None, use_vlines=True, **kwargs):
    """ Plot the auto or cross correlation.

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    corr : array_like
        Array of correlation values, used on the vertical axis.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : array_like, optional
        Array of lag values, used on horizontal axis.
        If not given, ``lags=np.arange(len(corr))`` is used.
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
    :func:`~matplotlib.pyplot.xcorr`
    :func:`~matplotlib.pyplot.acorr`
    mpl_examples/pylab_examples/xcorr_demo.py

    Notes
    -----
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    """
    fig, ax = utils.create_mpl_ax(ax)

    corr = np.asarray(corr)
    if lags is None:
        lags = np.arange(len(corr))
    else:
        if len(lags) != len(corr):
            raise ValueError('lags and corr must be of equal length')

    if use_vlines:
        ax.vlines(lags, [0], corr, **kwargs)
        ax.axhline(**kwargs)

    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('linestyle', 'None')
    ax.plot(lags, corr, **kwargs)

    return fig

