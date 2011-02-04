import numpy as np
import scikits.statsmodels.tsa.var.util as util

import matplotlib.pyplot as plt
import matplotlib as mpl

class MPLConfigurator(object):

    def __init__(self):
        self._inverse_actions = []

    def revert(self):
        for action in self._inverse_actions:
            action()

    def set_fontsize(self, size):
        old_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = size

        def revert():
            mpl.rcParams['font.size'] = old_size

        self._inverse_actions.append(revert)

#-------------------------------------------------------------------------------
# Plotting functions

def plot_mts(Y, names=None, index=None):
    """
    Plot multiple time series
    """

    k = Y.shape[1]
    rows, cols = k, 1

    plt.figure(figsize=(10, 10))

    for j in range(k):
        ts = Y[:, j]

        ax = plt.subplot(rows, cols, j+1)
        if index is not None:
            ax.plot(index, ts)
        else:
            ax.plot(ts)

        if names is not None:
            ax.set_title(names[j])

def plot_var_forc(prior, forc, err_upper, err_lower,
                  index=None, names=None):
    n, k = prior.shape
    rows, cols = k, 1

    fig = plt.figure(figsize=(10, 10))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)

        p1 = ax.plot(prange, prior[:, j], 'k')
        p2 = ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], 'k--')
        p3 = ax.plot(rng_err, err_upper[:, j], 'k-.')
        ax.plot(rng_err, err_lower[:, j], 'k-.')

        if names is not None:
            ax.set_title(names[j])

    fig.legend((p1, p2, p3), ('Observed', 'Forecast', 'Forc 2 STD err'),
               'upper right')

def plot_with_error(y, error, x=None, axes=None, value_fmt='k',
                    error_fmt='k--', alpha=0.05):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y :
    error : array or None

    """
    if axes is None:
        axes = plt.gca()

    if x is not None:
        plot_action = lambda y, fmt: axes.plot(x, y, fmt)
    else:
        plot_action = lambda y, fmt: axes.plot(y, fmt)

    plot_action(y, value_fmt)

    if error is not None:
        q = util.norm_signif_level(alpha)
        plot_action(y - q * error, error_fmt)
        plot_action(y + q * error, error_fmt)

def plot_acorr(acf, fontsize=8, linewidth=8):
    """

    Parameters
    ----------



    """
    config = MPLConfigurator()
    config.set_fontsize(fontsize)

    lags, k, k = acf.shape
    acorr = util.acf_to_acorr(acf)
    plt.figure(figsize=(10, 10))
    xs = np.arange(lags)

    for i in range(k):
        for j in range(k):
            ax = plt.subplot(k, k, i * k + j + 1)
            ax.vlines(xs, [0], acorr[:, i, j], lw=linewidth)

            ax.axhline(0, color='k')
            ax.set_ylim([-1, 1])

            # hack?
            ax.set_xlim([-1, xs[-1] + 1])

    adjust_subplots()
    config.revert()

def plot_acorr_with_error():
    pass

def adjust_subplots(**kwds):
    passed_kwds = dict(bottom=0.05, top=0.925,
                       left=0.05, right=0.95,
                       hspace=0.2)
    passed_kwds.update(kwds)
    plt.subplots_adjust(**passed_kwds)

#-------------------------------------------------------------------------------
# Multiple impulse response (cum_effects, etc.) cplots

def irf_grid_plot(values, stderr, impcol, rescol, names, title,
                  signif=0.05, hlines=None, subplot_params=None,
                  plot_params=None, figsize=(10,10)):
    """
    Reusable function to make flexible grid plots of impulse responses and
    comulative effects

    values : (T + 1) x k x k
    stderr : T x k x k
    hlines : k x k
    """
    if subplot_params is None:
        subplot_params = {}
    if plot_params is None:
        plot_params = {}

    nrows, ncols, to_plot = _get_irf_plot_config(names, impcol, rescol)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             squeeze=False, figsize=figsize)

    # fill out space
    adjust_subplots()

    fig.suptitle(title, fontsize=14)

    subtitle_temp = r'%s$\rightarrow$%s'

    k = len(names)

    for (j, i, ai, aj) in to_plot:
        ax = axes[ai][aj]

        # hm, how hackish is this?
        if stderr is not None:
            sig = np.sqrt(stderr[:, j * k + i, j * k + i])
            plot_with_error(values[:, i, j], sig, axes=ax, alpha=signif,
                            value_fmt='b')
        else:
            plot_with_error(values[:, i, j], None, axes=ax, value_fmt='b')

        ax.axhline(0, color='k')

        if hlines is not None:
            ax.axhline(hlines[i,j], color='k')

        sz = subplot_params.get('fontsize', 12)
        ax.set_title(subtitle_temp % (names[j], names[i]), fontsize=sz)


def _get_irf_plot_config(names, impcol, rescol):
    def _get_index(name):
        try:
            result = names.index(name)
        except Exception:
            if not isinstance(name, int):
                raise
            result = name
        return result

    nrows = ncols = k = len(names)
    if impcol is not None and rescol is not None:
        # plot one impulse-response pair
        nrows = ncols = 1
        j = _get_index(impcol)
        i = _get_index(rescol)
        to_plot = [(j, i, 0, 0)]
    elif impcol is not None:
        # plot impacts of impulse in one variable
        ncols = 1
        j = _get_index(impcol)
        to_plot = [(j, i, i, 0) for i in range(k)]
    elif rescol is not None:
        # plot only things having impact on particular variable
        ncols = 1
        i = _get_index(rescol)
        to_plot = [(j, i, j, 0) for j in range(k)]
    else:
        # plot everything
        to_plot = [(j, i, i, j) for i in range(k) for j in range(k)]

    return nrows, ncols, to_plot

#-------------------------------------------------------------------------------
# Forecast error variance decomposition


