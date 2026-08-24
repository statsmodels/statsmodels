from statsmodels.compat.python import lrange

import numpy as np

from statsmodels.graphics.utils import _import_mpl
from statsmodels.tsa.vector_ar import util


class MPLConfigurator:

    def __init__(self):
        self._inverse_actions = []

    def revert(self):
        for action in self._inverse_actions:
            action()

    def set_fontsize(self, size):
        import matplotlib as mpl

        old_size = mpl.rcParams["font.size"]
        mpl.rcParams["font.size"] = size

        def revert():
            mpl.rcParams["font.size"] = old_size

        self._inverse_actions.append(revert)


#
# Plotting functions
#
def plot_mts(Y, names=None, index=None):
    """
    Plot multiple time series

    Parameters
    ----------
    Y : ndarray
        2-d array of the time series to plot, of shape (nobs, neqs).
    names : sequence of str, optional
        Titles to use for each subplot. If None, subplots are not
        given a title.
    index : array_like, optional
        Values to use for the x-axis of each subplot. If None, uses
        the default integer index.

    Returns
    -------
    Figure
        The figure containing the grid of time series plots.
    """
    plt = _import_mpl()

    k = Y.shape[1]
    rows, cols = k, 1

    fig = plt.figure(figsize=(10, 10))

    for j in range(k):
        ts = Y[:, j]

        ax = fig.add_subplot(rows, cols, j + 1)
        if index is not None:
            ax.plot(index, ts)
        else:
            ax.plot(ts)

        if names is not None:
            ax.set_title(names[j])

    return fig


def plot_var_forc(
    prior,
    forc,
    err_upper,
    err_lower,
    index=None,
    names=None,
    plot_stderr=True,
    legend_options=None,
):
    """
    Plot a forecast against the observed data, with optional error bands

    Parameters
    ----------
    prior : ndarray
        2-d array of the observed data prior to the forecast, of shape
        (nobs, neqs).
    forc : ndarray
        2-d array of forecast values, of shape (steps, neqs).
    err_upper : ndarray
        2-d array of upper confidence interval bounds for the forecast,
        of shape (steps, neqs).
    err_lower : ndarray
        2-d array of lower confidence interval bounds for the forecast,
        of shape (steps, neqs).
    index : array_like, optional
        Currently unused.
    names : sequence of str, optional
        Titles to use for each subplot. If None, subplots are not
        given a title.
    plot_stderr : bool, optional
        If True, plot the confidence interval around the forecast.
    legend_options : dict, optional
        Keyword arguments passed to ``ax.legend``. If None, uses
        ``{"loc": "upper right"}``.

    Returns
    -------
    Figure
        The figure containing the grid of forecast plots.
    """
    plt = _import_mpl()

    n, k = prior.shape
    rows, cols = k, 1

    fig = plt.figure(figsize=(10, 10))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = plt.subplot(rows, cols, j + 1)

        ax.plot(prange, prior[:, j], "k", label="Observed")
        ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], "k--", label="Forecast")

        if plot_stderr:
            ax.plot(rng_err, err_upper[:, j], "k-.", label="Forc 2 STD err")
            ax.plot(rng_err, err_lower[:, j], "k-.")

        if names is not None:
            ax.set_title(names[j])

        if legend_options is None:
            legend_options = {"loc": "upper right"}
        ax.legend(**legend_options)
    return fig


def plot_with_error(
    y,
    error,
    x=None,
    axes=None,
    value_fmt="k",
    error_fmt="k--",
    alpha=0.05,
    stderr_type="asym",
):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y : array_like
        The data to plot.
    error : array_like, tuple of array_like, or None
        The error used to plot error bars around `y`. If `stderr_type`
        is "asym", an array the same shape as `y` combined with `alpha`
        to compute the bands. If `stderr_type` is one of "mc", "sz1",
        "sz2", "sz3", a 2-tuple of arrays giving the lower and upper
        bands directly. If None, no error bars are plotted.
    x : array_like, optional
        The x-axis values to use. If None, uses a range the same length
        as `y`.
    axes : AxesSubplot, optional
        Matplotlib axes to plot on. If None, uses the current axes.
    value_fmt : str, optional
        Matplotlib format string used to plot `y`.
    error_fmt : str, optional
        Matplotlib format string used to plot the error bars.
    alpha : float, optional
        The significance level to use when `stderr_type` is "asym".
    stderr_type : {"asym", "mc", "sz1", "sz2", "sz3"}, optional
        The kind of error bars being plotted.
    """
    plt = _import_mpl()

    if axes is None:
        axes = plt.gca()

    x = x if x is not None else lrange(len(y))

    def plot_action(y, fmt):
        return axes.plot(x, y, fmt)

    plot_action(y, value_fmt)

    # changed this
    if error is not None:
        if stderr_type == "asym":
            q = util.norm_signif_level(alpha)
            plot_action(y - q * error, error_fmt)
            plot_action(y + q * error, error_fmt)
        if stderr_type in ("mc", "sz1", "sz2", "sz3"):
            plot_action(error[0], error_fmt)
            plot_action(error[1], error_fmt)


def plot_full_acorr(acorr, fontsize=8, linewidth=8, xlabel=None, err_bound=None):
    """
    Plot the autocorrelations of a multivariate time series in a grid

    Parameters
    ----------
    acorr : ndarray
        Array of autocorrelations, shape (nlags, k, k).
    fontsize : int, optional
        Font size used for the plot labels.
    linewidth : int, optional
        Width of the lines used in the autocorrelation plots.
    xlabel : array_like, optional
        Labels to use for the x-axis of each subplot. If None, uses a
        range the same length as `acorr`.
    err_bound : float, optional
        If provided, draws horizontal reference lines at +/- `err_bound`.

    Returns
    -------
    Figure
        The figure containing the grid of autocorrelation plots.
    """
    plt = _import_mpl()

    config = MPLConfigurator()
    config.set_fontsize(fontsize)

    k = acorr.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(10, 10), squeeze=False)

    for i in range(k):
        for j in range(k):
            ax = axes[i][j]
            acorr_plot(acorr[:, i, j], linewidth=linewidth, xlabel=xlabel, ax=ax)

            if err_bound is not None:
                ax.axhline(err_bound, color="k", linestyle="--")
                ax.axhline(-err_bound, color="k", linestyle="--")

    adjust_subplots()
    config.revert()

    return fig


def acorr_plot(acorr, linewidth=8, xlabel=None, ax=None):
    """
    Plot a single autocorrelation function as a stem plot

    Parameters
    ----------
    acorr : array_like
        1-d array of autocorrelations to plot.
    linewidth : int, optional
        Width of the vertical lines used in the plot.
    xlabel : array_like, optional
        Positions to use for the x-axis. If None, uses a range the
        same length as `acorr`.
    ax : AxesSubplot, optional
        Matplotlib axes to plot on. If None, uses the current axes.
    """
    plt = _import_mpl()

    if ax is None:
        ax = plt.gca()

    if xlabel is None:
        xlabel = np.arange(len(acorr))

    ax.vlines(xlabel, [0], acorr, lw=linewidth)

    ax.axhline(0, color="k")
    ax.set_ylim([-1, 1])

    # hack?
    ax.set_xlim([-1, xlabel[-1] + 1])


def plot_acorr_with_error():
    """Not implemented."""
    raise NotImplementedError


def adjust_subplots(**kwds):
    """
    Adjust subplot spacing using defaults suitable for the grid plots

    Parameters
    ----------
    **kwds
        Keyword arguments passed to ``matplotlib.pyplot.subplots_adjust``,
        overriding the defaults.
    """
    plt = _import_mpl()

    passed_kwds = {
        "bottom": 0.05,
        "top": 0.925,
        "left": 0.05,
        "right": 0.95,
        "hspace": 0.2,
    }
    passed_kwds.update(kwds)
    plt.subplots_adjust(**passed_kwds)


#
# Multiple impulse response (cum_effects, etc.) cplots
#


def irf_grid_plot(
    values,
    stderr,
    impcol,
    rescol,
    names,
    title,
    signif=0.05,
    hlines=None,
    subplot_params=None,
    plot_params=None,
    figsize=(10, 10),
    stderr_type="asym",
):
    """
    Reusable function to make flexible grid plots of impulse responses and
    cumulative effects

    Parameters
    ----------
    values : ndarray
        Array of values to plot, shape (T, k, k).
    stderr : ndarray, tuple of ndarray, or None
        Used to plot error bands around `values`. If `stderr_type` is
        "asym", an array of shape (T, k ** 2, k ** 2) giving the
        covariance matrix of the vectorized values at each period (as
        returned by :meth:`~statsmodels.tsa.vector_ar.irf.IRAnalysis.cov`).
        If `stderr_type` is one of "mc", "sz1", "sz2", "sz3", a 2-tuple
        of arrays of shape (T, k, k) giving the lower and upper error
        bands directly. If None, no error bands are plotted.
    impcol : int, str, or None
        Column of the impulse variable to plot. If None, plots impulses
        from all variables.
    rescol : int, str, or None
        Column of the response variable to plot. If None, plots responses
        of all variables.
    names : sequence of str
        Names of the variables in the system.
    title : str
        Title to use for the figure.
    signif : float, optional
        Significance level used when plotting error bands.
    hlines : ndarray, optional
        Array of horizontal reference lines to draw on each subplot,
        shape (k, k).
    subplot_params : dict, optional
        May contain a "fontsize" key controlling the subplot title font
        size.
    plot_params : dict, optional
        Additional keyword arguments (currently unused).
    figsize : tuple of float, optional
        The size of the figure to create.
    stderr_type : {"asym", "mc", "sz1", "sz2", "sz3"}, optional
        The kind of error bars being plotted.

    Returns
    -------
    Figure
        The figure containing the grid of impulse response plots.
    """
    plt = _import_mpl()

    if subplot_params is None:
        subplot_params = {}
    if plot_params is None:
        plot_params = {}

    nrows, ncols, to_plot = _get_irf_plot_config(names, impcol, rescol)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, squeeze=False, figsize=figsize
    )

    # fill out space
    adjust_subplots()

    fig.suptitle(title, fontsize=14)

    subtitle_temp = r"%s$\rightarrow$%s"

    k = len(names)

    rng = lrange(len(values))
    for j, i, ai, aj in to_plot:
        ax = axes[ai][aj]

        # HACK?
        if stderr is not None:
            if stderr_type == "asym":
                sig = np.sqrt(stderr[:, j * k + i, j * k + i])
                plot_with_error(
                    values[:, i, j],
                    sig,
                    x=rng,
                    axes=ax,
                    alpha=signif,
                    value_fmt="b",
                    stderr_type=stderr_type,
                )
            if stderr_type in ("mc", "sz1", "sz2", "sz3"):
                errs = stderr[0][:, i, j], stderr[1][:, i, j]
                plot_with_error(
                    values[:, i, j],
                    errs,
                    x=rng,
                    axes=ax,
                    alpha=signif,
                    value_fmt="b",
                    stderr_type=stderr_type,
                )
        else:
            plot_with_error(values[:, i, j], None, x=rng, axes=ax, value_fmt="b")

        ax.axhline(0, color="k")

        if hlines is not None:
            ax.axhline(hlines[i, j], color="k")

        sz = subplot_params.get("fontsize", 12)
        ax.set_title(subtitle_temp % (names[j], names[i]), fontsize=sz)

    return fig


def _get_irf_plot_config(names, impcol, rescol):
    """
    Determine the subplot grid layout for `irf_grid_plot`

    Parameters
    ----------
    names : sequence of str
        Names of the variables in the system.
    impcol : int, str, or None
        Column of the impulse variable to plot. If None, plots impulses
        from all variables.
    rescol : int, str, or None
        Column of the response variable to plot. If None, plots responses
        of all variables.

    Returns
    -------
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    to_plot : list of tuple
        Each tuple is ``(j, i, ai, aj)`` giving the impulse index `j`,
        response index `i`, and the subplot grid position (`ai`, `aj`)
        to plot it at.
    """
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


#
# Forecast error variance decomposition
