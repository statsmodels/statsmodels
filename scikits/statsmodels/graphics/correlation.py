
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print "plots not available without matplotlib"


def plot_corr(dcorr, xnames=None, ynames=None, title=None, normcolor=False,
              axis=None):
    '''plot correlation of many variables in a tight color grid

    This creates a new figure

    Parameters
    ----------
    dcorr : ndarray
        correlation matrix
    xnames : None or list of strings
        labels for x axis
    ynames : None or list of strings
        labels for y axis
    title : None or string
        title for figure. If None, then default is added. If title='', then no
        title is added
    normcolor : bool
        If false (default), then the color coding range corresponds to the
        lowest and highest correlation. If true, then the color range is
        normalized to (-1, 1).
    ax: None or axis instance
        If ax is None, then a figure is created. If an axis instance is given,
        then the only the main plot but not the colorbar is created.

    Returns
    -------
    fig_or_ax : matplotlib figure or axis instance


    '''

    nvars = dcorr.shape[0]
    #dcorr[range(nvars), range(nvars)] = np.nan

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

    if axis is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axis

    axim = ax.imshow(dcorr, cmap=plt.cm.jet, interpolation='nearest',
                     extent=(0,30,0,30), vmin=vmin, vmax=vmax)
    if ynames:
        ax.set_yticks(np.arange(nvars)+0.5)
        ax.set_yticklabels(ynames[::-1], minor=True, fontsize='small',
                           horizontalalignment='right')
    if xnames:
        ax.set_xticks(np.arange(nvars)+0.5)
        ax.set_xticklabels(xnames, minor=True, fontsize='small',rotation=45,
                           horizontalalignment='right')
        #some keywords don't work in previous line ?
        #TODO: check if this is redundant
        plt.setp( ax.get_xticklabels(), fontsize='small', rotation=45,
                 horizontalalignment='right')

    if not title == '':
        ax.set_title(title)

    if axis is None:
        fig.colorbar(axim)
        return fig
    else:
        return ax

def plot_corr_grid(dcorrs, titles=None, ncols=2, normcolor=False):
    '''greate a grid of correlation plots

    dcorrs: list, iterable
        list of correlation matrices
    titles : None or iterable of strings
        list of titles for the subplots
    '''
    if not titles:
        titles = [None]*len(dcorrs)
    nrows = int(np.ceil(len(dcorrs) / float(ncols)))

    fig = plt.figure()
    for i, c in enumerate(dcorrs):
        ax = fig.add_subplot(nrows, ncols, i+1)
        plot_corr(c, xnames=None, title=titles[i],
              normcolor=normcolor, axis=ax)

    #images = [c for ax in fig.axes for c in ax.get_children() if isinstance(c, mpl.image.AxesImage)]
    images = [i for ax in fig.axes for i in ax.images ]
    fig.subplots_adjust(bottom=0.1, left=0.09, right=0.9, top=0.9)
    if ncols <=2:
        cax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
    else:
        cax = fig.add_axes([0.92, 0.1, 0.025, 0.8])
    fig.colorbar(images[0], cax=cax)
    return fig
