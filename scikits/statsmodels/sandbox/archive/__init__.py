

import numpy as np


#copied/moved from sandbox/tsa/example_arma.py
def plotacf(ax, corr, lags=None, usevlines=True, **kwargs):
    """
    Plot the auto or cross correlation.
    lags on horizontal and correlations on vertical axis

    Note: adjusted from matplotlib's pltxcorr

    Parameters
    ----------
    ax : matplotlib axis or plt
        ax can be matplotlib.pyplot or an axis of a figure
    lags : array or None
        array of lags used on horizontal axis,
        if None, then np.arange(len(corr)) is used
    corr : array
        array of values used on vertical axis
    usevlines : boolean
        If true, then vertical lines and markers are plotted. If false,
        only 'o' markers are plotted
    **kwargs : optional parameters for plot and axhline
        these are directly passed on to the matplotlib functions

    Returns
    -------
    a : matplotlib.pyplot.plot
        contains markers
    b : matplotlib.collections.LineCollection
        returned only if vlines is true, contains vlines
    c : instance of matplotlib.lines.Line2D
        returned only if vlines is true, contains axhline ???

    Data are plotted as ``plot(lags, c, **kwargs)``

    The default *linestyle* is *None* and the default *marker* is
    'o', though these can be overridden with keyword args.

    If *usevlines* is *True*:

       :func:`~matplotlib.pyplot.vlines`
       rather than :func:`~matplotlib.pyplot.plot` is used to draw
       vertical lines from the origin to the xcorr.  Otherwise the
       plotstyle is determined by the kwargs, which are
       :class:`~matplotlib.lines.Line2D` properties.

    See Also
    --------

    :func:`~matplotlib.pyplot.xcorr`
    :func:`~matplotlib.pyplot.acorr`
    mpl_examples/pylab_examples/xcorr_demo.py

    """

    if lags is None:
        lags = np.arange(len(corr))
    else:
        if len(lags) != len(corr):
            raise ValueError('lags and corr must be equal length')

    if usevlines:
        b = ax.vlines(lags, [0], corr, **kwargs)
        c = ax.axhline(**kwargs)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        a = ax.plot(lags, corr, **kwargs)
    else:
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        a, = ax.plot(lags, corr, **kwargs)
        b = c = None
    return a, b, c

