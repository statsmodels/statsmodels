from statsmodels.compat.python import lrange, range
import numpy as np
import statsmodels.tsa.vector_ar.util as util
from statsmodels.graphics import utils as graphics_utils

#TODO: we should change sm.graphics.utils to also use subplots
def _create_mpl_subplots(ax=None, nrows=1, ncols=1, **kwargs):
    """
    Creates matplotlib figure and axes if it doesn't exist. Uses
    `matplotlib.pyplot.subplots` kwargs are passed to it.
    """
    if ax is None:
        plt = graphics_utils._import_mpl()
        fig, ax = plt.subplots(nrows, ncols, **kwargs)
    else:
        fig = ax.figure

    return fig, ax

class MPLConfigurator(object):

    def __init__(self):
        self._inverse_actions = []

    def revert(self):
        for action in self._inverse_actions:
            action()

    def set_fontsize(self, size):
        import matplotlib as mpl
        old_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = size

        def revert():
            mpl.rcParams['font.size'] = old_size

        self._inverse_actions.append(revert)

#-------------------------------------------------------------------------------
# Plotting functions

def plot_timeseries(Y, names=None, index=None, ax=None, figsize=(10,10)):
    """
    Plot multiple time series

    Parameters
    ----------
    Y : ndarray
        An array of timeseries with variables in columns
    names : list, optional
        Names of each series
    index : array-like
        An index for the rows of the series to be plotting as the x axis.
    ax : matplotlib.axes, optional
        An existing matplotlib.axes instance
    figsize : tuple
        The figure size if ax=None

    Returns
    -------
    fig : matplotlib.figure
        The matplotlib figure that contains the axes
    """
    neqs = Y.shape[1]
    fig, axes = _create_mpl_subplots(ax, neqs, 1, figsize=figsize)

    rows, cols = neqs, 1

    for j in range(neqs):
        ax = axes[j]
        ts = Y[:, j]

        if index is not None:
            ax.plot(index, ts)
        else:
            ax.plot(ts)

        if names is not None:
            ax.set_title(names[j])

    return fig

#TODO: Docs
def plot_var_forc(prior, forc, err_upper, err_lower, index=None, names=None,
                  plot_stderr=True):
    """
    Parameters
    ----------
    prior
    forc
    err_upper
    err_lower
    index
    names
    plot_stderr : bool
    """
    n, k = prior.shape
    rows, cols = k, 1
    fig, axes = _create_mpl_subplots(None, rows, 1, figsize=(10,10))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = axes[j]

        p1 = ax.plot(prange, prior[:, j], 'k', label='Observed')
        p2 = ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], 'k--',
                        label='Forecast')

        if plot_stderr:
            p3 = ax.plot(rng_err, err_upper[:, j], 'k-.',
                            label='Forc 2 STD err')
            ax.plot(rng_err, err_lower[:, j], 'k-.')

        if names is not None:
            ax.set_title(names[j])

        ax.legend(loc='upper right')

#TODO: Docs
def plot_with_error(y, error, x=None, axes=None, value_fmt='k',
                    error_fmt='k--', alpha=0.05, stderr_type = 'asym'):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y :
    error : array or None
    x
    axes : matplotlib.axes, optional
    value_fmt
    error_fmt
    alpha
    stderr_type

    Returns
    -------
    fig : matplotlib.figure
        The `matplotlib.figure` instance that contains the axes.
    """
    fig, axes = _create_mpl_subplots(axes, 1, 1)

    x = x if x is not None else lrange(len(y))
    plot_action = lambda y, fmt: axes.plot(x, y, fmt)
    plot_action(y, value_fmt)

    #changed this
    if error is not None:
        if stderr_type == 'asym':
            q = util.norm_signif_level(alpha)
            plot_action(y - q * error, error_fmt)
            plot_action(y + q * error, error_fmt)
        if stderr_type in ('mc','sz1','sz2','sz3'):
            plot_action(error[0], error_fmt)
            plot_action(error[1], error_fmt)

    return fig

def plot_full_acorr(acorr, fontsize=8, linewidth=8, err_bound=None,
                    names=None, figsize=(10,10), ax=None, **kwargs):
    """
    Plots the autocorrelations given by acorr.

    Parameters
    ----------
    acorr : ndarray
        The autocorrelation. Should be of shape (nlags, neqs, neqs).
    fontsize : int
        The fontsize
    linewidth : int
        The linewidth
    err_bound : array or None
        Error bounds for the autocorrelation.
    names : list or None
        The endogenous
    kwargs : kwargs
        Passed to `matplotlib.pyplot.vlines`
    """
    config = MPLConfigurator()
    config.set_fontsize(fontsize)

    neqs = acorr.shape[1]
    fig, axes = _create_mpl_subplots(ax, neqs, neqs, figsize=figsize,
                                     squeeze=False, sharex=True, sharey=True)

    for i in range(neqs):
        if names is not None:
            axes[i, 0].set_ylabel(names[i])
            axes[0, i].set_xlabel(names[i])
        for j in range(neqs):
            ax = axes[i, j]
            acorr_plot(acorr[:, i, j], linewidth=linewidth, ax=ax, **kwargs)

            if err_bound is not None:
                ax.axhline(err_bound, color='k', linestyle='--')
                ax.axhline(-err_bound, color='k', linestyle='--')

    adjust_subplots(fig)
    config.revert()

    return fig

def acorr_plot(acorr, linewidth=8, xlabel=None, ax=None, **kwargs):
    """
    Plot the autocorrelation.

    Parameters
    ----------
    acorr : ndarray
        Shape (nlags x neqs x neqs) array of autocorrelations.
    linewidth : int
        The linewidth
    xlabel : list, optional
        If None uses the lag number for the xticks starting at zero
    ax : matplotlib.axes, optional
        An existing matplotlib.axes instance.
    kwargs : kwargs
        The keyword arguments are passed on to `matplotlib.pyplot.vlines`
    """
    import matplotlib.pyplot as plt


    fig, ax = graphics_utils.create_mpl_ax(ax)

    if ax is None:
        ax = plt.gca()

    if xlabel is None:
        xlabel = np.arange(len(acorr))

    ax.vlines(xlabel, [0], acorr, lw=linewidth, **kwargs)

    ax.axhline(0, color='k')
    ax.set_ylim([-1, 1])

    # hack?
    ax.set_xlim([-1, xlabel[-1] + 1])
    return fig

def plot_acorr_with_error():
    pass

def adjust_subplots(fig, **kwds):
    passed_kwds = dict(bottom=0.05, top=0.925,
                       left=0.05, right=0.95,
                       hspace=0.2)
    passed_kwds.update(kwds)
    fig.subplots_adjust(**passed_kwds)

#-------------------------------------------------------------------------------
# Multiple impulse response (cum_effects, etc.) cplots

#TODO: Docs
def irf_grid_plot(values, stderr, impcol, rescol, names, title,
                  signif=0.05, hlines=None, subplot_params=None,
                  plot_params=None, figsize=(10,10), stderr_type='asym'):
    """
    Reusable function to make flexible grid plots of impulse responses and
    comulative effects

    values : ndarray
        Values to plot of shape (`nobs` + 1) x `neqs` x `neqs`
    stderr : ndarray
        Standard errors of shape (`nobs` x `neqs` x `neqs`)
    hlines : ndarray
        Shape (`neqs` x `neqs`)

    Return
    ------
    fig : `matplotlib.figure`
        The `matplotlib.figure` instance that contains the axes.
    """

    if subplot_params is None:
        subplot_params = {}
    if plot_params is None:
        plot_params = {}

    nrows, ncols, to_plot = _get_irf_plot_config(names, impcol, rescol)

    fig, axes = _create_mpl_subplots(None, nrows, ncols, sharex=True,
                                     squeeze=False, figsize=figsize)

    # fill out space
    adjust_subplots(fig)

    fig.suptitle(title, fontsize=14)

    subtitle_temp = r'%s$\rightarrow$%s'

    k = len(names)

    rng = lrange(len(values))
    for (j, i, ai, aj) in to_plot:
        ax = axes[ai][aj]

        # HACK?
        if stderr is not None:
            if stderr_type == 'asym':
                sig = np.sqrt(stderr[:, j * k + i, j * k + i])
                plot_with_error(values[:, i, j], sig, x=rng, axes=ax,
                        alpha=signif, value_fmt='b', stderr_type=stderr_type)
            if stderr_type in ('mc','sz1','sz2','sz3'):
                errs = stderr[0][:, i, j], stderr[1][:, i, j]
                plot_with_error(values[:, i, j], errs, x=rng, axes=ax,
                        alpha=signif, value_fmt='b', stderr_type=stderr_type)
        else:
            plot_with_error(values[:, i, j], None, x=rng, axes=ax,
                            value_fmt='b')

        ax.axhline(0, color='k')

        if hlines is not None:
            ax.axhline(hlines[i,j], color='k')

        sz = subplot_params.get('fontsize', 12)
        ax.set_title(subtitle_temp % (names[j], names[i]), fontsize=sz)

    return fig

def _get_irf_plot_config(names, impcol, rescol):
    nrows = ncols = k = len(names)
    if impcol is not None and rescol is not None:
        # plot one impulse-response pair
        nrows = ncols = 1
        j = util.get_index(names, impcol)
        i = util.get_index(names, rescol)
        to_plot = [(j, i, 0, 0)]
    elif impcol is not None:
        # plot impacts of impulse in one variable
        ncols = 1
        j = util.get_index(names, impcol)
        to_plot = [(j, i, i, 0) for i in range(k)]
    elif rescol is not None:
        # plot only things having impact on particular variable
        ncols = 1
        i = util.get_index(names, rescol)
        to_plot = [(j, i, j, 0) for j in range(k)]
    else:
        # plot everything
        to_plot = [(j, i, i, j) for i in range(k) for j in range(k)]

    return nrows, ncols, to_plot

#-------------------------------------------------------------------------------
# Forecast error variance decomposition


