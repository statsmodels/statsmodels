'''correlation plots

Author: Josef Perktold
License: BSD-3

example for usage with different options in
statsmodels\sandbox\examples\thirdparty\ex_ratereturn.py

'''
import numpy as np

from . import utils


def plot_corr(dcorr, xnames=None, ynames=None, title=None, normcolor=False,
              ax=None):
    """Plot correlation of many variables in a tight color grid.

    This creates a new figure

    Parameters
    ----------
    dcorr : ndarray
        correlation matrix
    xnames : None or list of strings
        labels for x axis. If None, then the matplotlib defaults are used. If
        it is an empty list, [], then not ticks and labels are added.
    ynames : None or list of strings
        labels for y axis. If None, then the matplotlib defaults are used. If
        it is an empty list, [], then not ticks and labels are added.
    title : None or string
        title for figure. If None, then default is added. If title='', then no
        title is added
    normcolor : bool
        If false (default), then the color coding range corresponds to the
        lowest and highest correlation (automatic choice by matplotlib).
        If true, then the color range is normalized to (-1, 1). If this is a
        tuple of two numbers, then they define the range for the color bar.
    ax : Matplotlib AxesSubplot instance, optional
        If `ax` is None, then a figure is created. If an axis instance is
        given, then only the main plot but not the colorbar is created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    """
    if ax is None:
        create_colorbar = True
    else:
        create_colorbar = False

    fig, ax = utils.create_mpl_ax(ax)
    from matplotlib import cm
    from matplotlib.artist import setp

    nvars = dcorr.shape[0]

    if (ynames is None) and (not xnames is None):
        ynames = xnames
    if title is None:
        title = 'Correlation Matrix'
    if isinstance(normcolor, tuple):
        vmin, vmax = normcolor
    elif normcolor:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = None, None

    axim = ax.imshow(dcorr, cmap=cm.jet, interpolation='nearest',
                     extent=(0,30,0,30), vmin=vmin, vmax=vmax)
    if ynames:
        ax.set_yticks(np.arange(nvars)+0.5)
        ax.set_yticklabels(ynames[::-1], minor=True, fontsize='small',
                           horizontalalignment='right')
    elif ynames == []:
        ax.set_yticks([])

    if xnames:
        ax.set_xticks(np.arange(nvars)+0.5)
        ax.set_xticklabels(xnames, minor=True, fontsize='small', rotation=45,
                           horizontalalignment='right')
        #some keywords don't work in previous line ?
        #TODO: check if this is redundant
        setp(ax.get_xticklabels(), fontsize='small', rotation=45,
             horizontalalignment='right')
    elif xnames == []:
        ax.set_xticks([])

    if not title == '':
        ax.set_title(title)

    if create_colorbar:
        fig.colorbar(axim)

    return fig


def plot_corr_grid(dcorrs, titles=None, ncols=2, normcolor=False, xnames=None,
                   ynames=None, fig=None):
    """Create a grid of correlation plots.

    Parameters
    ----------
    dcorrs : list, iterable of ndarrays
        list of correlation matrices
    titles : None or iterable of strings
        list of titles for the subplots
    ncols : int
        number of columns in the subplot grid. Layout is designed for two or
        three columns.
    normcolor : bool or tuple
        If false (default), then the color coding range corresponds to the
        lowest and highest correlation (automatic choice by matplotlib).
        If true, then the color range is normalized to (-1, 1). If this is a
        tuple of two numbers, then they define the range for the color bar.
    xnames : None or list of strings
        labels for x axis. If None, then the matplotlib defaults are used. If
        it is an empty list, [], then not ticks and labels are added.
    ynames : None or list of strings
        labels for y axis. If None, then the matplotlib defaults are used. If
        it is an empty list, [], then not ticks and labels are added.
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Notes
    -----
    Possible extension for options, suppress labels except first column and
    last row.

    """
    fig = utils.create_mpl_fig(fig)

    if not titles:
        titles = [None]*len(dcorrs)

    nrows = int(np.ceil(len(dcorrs) / float(ncols)))

    for i, c in enumerate(dcorrs):
        ax = fig.add_subplot(nrows, ncols, i+1)
        plot_corr(c, xnames=xnames, ynames=ynames, title=titles[i],
                  normcolor=normcolor, ax=ax)

    images = [i for ax in fig.axes for i in ax.images ]
    fig.subplots_adjust(bottom=0.1, left=0.09, right=0.9, top=0.9)
    if ncols <=2:
        cax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
    else:
        cax = fig.add_axes([0.92, 0.1, 0.025, 0.8])

    fig.colorbar(images[0], cax=cax)

    return fig
