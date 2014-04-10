"""Module for functional boxplots."""

from statsmodels.compat.python import combinations, range
import numpy as np
from scipy import stats
from scipy.misc import factorial

from . import utils


__all__ = ['fboxplot', 'rainbowplot', 'banddepth']


def fboxplot(data, xdata=None, labels=None, depth=None, method='MBD',
             wfactor=1.5, ax=None, plot_opts={}):
    """Plot functional boxplot.

    A functional boxplot is the analog of a boxplot for functional data.
    Functional data is any type of data that varies over a continuum, i.e.
    curves, probabillity distributions, seasonal data, etc.

    The data is first ordered, the order statistic used here is `banddepth`.
    Plotted are then the median curve, the envelope of the 50% central region,
    the maximum non-outlying envelope and the outlier curves.

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    xdata : ndarray, optional
        The independent variable for the data.  If not given, it is assumed to
        be an array of integers 0..N with N the length of the vectors in
        `data`.
    labels : sequence of scalar or str, optional
        The labels or identifiers of the curves in `data`.  If given, outliers
        are labeled in the plot.
    depth : ndarray, optional
        A 1-D array of band depths for `data`, or equivalent order statistic.
        If not given, it will be calculated through `banddepth`.
    method : {'MBD', 'BD2'}, optional
        The method to use to calculate the band depth.  Default is 'MBD'.
    wfactor : float, optional
        Factor by which the central 50% region is multiplied to find the outer
        region (analog of "whiskers" of a classical boxplot).
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    plot_opts : dict, optional
        A dictionary with plotting options.  Any of the following can be
        provided, if not present in `plot_opts` the defaults will be used::

          - 'cmap_outliers', a Matplotlib LinearSegmentedColormap instance.
          - 'c_inner', valid MPL color. Color of the central 50% region
          - 'c_outer', valid MPL color. Color of the non-outlying region
          - 'c_median', valid MPL color. Color of the median.
          - 'lw_outliers', scalar.  Linewidth for drawing outlier curves.
          - 'lw_median', scalar.  Linewidth for drawing the median curve.
          - 'draw_nonout', bool.  If True, also draw non-outlying curves.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    depth : ndarray
        1-D array containing the calculated band depths of the curves.
    ix_depth : ndarray
        1-D array of indices needed to order curves (or `depth`) from most to
        least central curve.
    ix_outliers : ndarray
        1-D array of indices of outlying curves in `data`.

    See Also
    --------
    banddepth, rainbowplot

    Notes
    -----
    The median curve is the curve with the highest band depth.

    Outliers are defined as curves that fall outside the band created by
    multiplying the central region by `wfactor`.  Note that the range over
    which they fall outside this band doesn't matter, a single data point
    outside the band is enough.  If the data is noisy, smoothing may therefore
    be required.

    The non-outlying region is defined as the band made up of all the
    non-outlying curves.

    References
    ----------
    [1] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of Computational
        and Graphical Statistics, vol. 20, pp. 1-19, 2011.
    [2] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-25, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a functional boxplot.  We see that the years 1982-83 and 1997-98 are
    outliers; these are the years where El Nino (a climate pattern
    characterized by warming up of the sea surface and higher air pressures)
    occurred with unusual intensity.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.fboxplot(data.raw_data[:, 1:], wfactor=2.58,
    ...                            labels=data.raw_data[:, 0].astype(int),
    ...                            ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])

    >>> plt.show()

    .. plot:: plots/graphics_functional_fboxplot.py

    """
    fig, ax = utils.create_mpl_ax(ax)

    if plot_opts.get('cmap_outliers') is None:
        from matplotlib.cm import rainbow_r
        plot_opts['cmap_outliers'] = rainbow_r

    data = np.asarray(data)
    if xdata is None:
        xdata = np.arange(data.shape[1])

    # Calculate band depth if required.
    if depth is None:
        if method not in ['MBD', 'BD2']:
            raise ValueError("Unknown value for parameter `method`.")

        depth = banddepth(data, method=method)
    else:
        if depth.size != data.shape[0]:
            raise ValueError("Provided `depth` array is not of correct size.")

    # Inner area is 25%-75% region of band-depth ordered curves.
    ix_depth = np.argsort(depth)[::-1]
    median_curve = data[ix_depth[0], :]
    ix_IQR = data.shape[0] // 2
    lower = data[ix_depth[0:ix_IQR], :].min(axis=0)
    upper = data[ix_depth[0:ix_IQR], :].max(axis=0)

    # Determine region for outlier detection
    inner_median = np.median(data[ix_depth[0:ix_IQR], :], axis=0)
    lower_fence = inner_median - (inner_median - lower) * wfactor
    upper_fence = inner_median + (upper - inner_median) * wfactor

    # Find outliers.
    ix_outliers = []
    ix_nonout = []
    for ii in range(data.shape[0]):
        if np.any(data[ii, :] > upper_fence) or np.any(data[ii, :] < lower_fence):
            ix_outliers.append(ii)
        else:
            ix_nonout.append(ii)

    ix_outliers = np.asarray(ix_outliers)

    # Plot envelope of all non-outlying data
    lower_nonout = data[ix_nonout, :].min(axis=0)
    upper_nonout = data[ix_nonout, :].max(axis=0)
    ax.fill_between(xdata, lower_nonout, upper_nonout,
                    color=plot_opts.get('c_outer', (0.75,0.75,0.75)))

    # Plot central 50% region
    ax.fill_between(xdata, lower, upper,
                    color=plot_opts.get('c_inner', (0.5,0.5,0.5)))

    # Plot median curve
    ax.plot(xdata, median_curve, color=plot_opts.get('c_median', 'k'),
            lw=plot_opts.get('lw_median', 2))

    # Plot outliers
    cmap = plot_opts.get('cmap_outliers')
    for ii, ix in enumerate(ix_outliers):
        label = str(labels[ix]) if labels is not None else None
        ax.plot(xdata, data[ix, :],
                color=cmap(float(ii) / (len(ix_outliers)-1)), label=label,
                lw=plot_opts.get('lw_outliers', 1))

    if plot_opts.get('draw_nonout', False):
        for ix in ix_nonout:
            ax.plot(xdata, data[ix, :], 'k-', lw=0.5)

    if labels is not None:
        ax.legend()

    return fig, depth, ix_depth, ix_outliers


def rainbowplot(data, xdata=None, depth=None, method='MBD', ax=None,
                 cmap=None):
    """Create a rainbow plot for a set of curves.

    A rainbow plot contains line plots of all curves in the dataset, colored in
    order of functional depth.  The median curve is shown in black.

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    xdata : ndarray, optional
        The independent variable for the data.  If not given, it is assumed to
        be an array of integers 0..N with N the length of the vectors in
        `data`.
    depth : ndarray, optional
        A 1-D array of band depths for `data`, or equivalent order statistic.
        If not given, it will be calculated through `banddepth`.
    method : {'MBD', 'BD2'}, optional
        The method to use to calculate the band depth.  Default is 'MBD'.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    cmap : Matplotlib LinearSegmentedColormap instance, optional
        The colormap used to color curves with.  Default is a rainbow colormap,
        with red used for the most central and purple for the least central
        curves.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    banddepth, fboxplot

    References
    ----------
    [1] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-25, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a rainbow plot:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.rainbowplot(data.raw_data[:, 1:], ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])
    >>> plt.show()

    .. plot:: plots/graphics_functional_rainbowplot.py

    """
    fig, ax = utils.create_mpl_ax(ax)

    if cmap is None:
        from matplotlib.cm import rainbow_r
        cmap = rainbow_r

    data = np.asarray(data)
    if xdata is None:
        xdata = np.arange(data.shape[1])

    # Calculate band depth if required.
    if depth is None:
        if method not in ['MBD', 'BD2']:
            raise ValueError("Unknown value for parameter `method`.")

        depth = banddepth(data, method=method)
    else:
        if depth.size != data.shape[0]:
            raise ValueError("Provided `depth` array is not of correct size.")

    ix_depth = np.argsort(depth)[::-1]

    # Plot all curves, colored by depth
    num_curves = data.shape[0]
    for ii in range(num_curves):
        ax.plot(xdata, data[ix_depth[ii], :], c=cmap(ii / (num_curves - 1.)))

    # Plot the median curve
    median_curve = data[ix_depth[0], :]
    ax.plot(xdata, median_curve, 'k-', lw=2)

    return fig


def banddepth(data, method='MBD'):
    """Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves is the 50%
    central region.

    Parameters
    ----------
    data : ndarray
        The vectors of functions to create a functional boxplot from.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    method : {'MBD', 'BD2'}, optional
        Whether to use the original band depth (with J=2) of [1]_ or the
        modified band depth.  See Notes for details.

    Returns
    -------
    depth : ndarray
        Depth values for functional curves.

    Notes
    -----
    Functional band depth as an order statistic for functional data was
    proposed in [1]_ and applied to functional boxplots and bagplots in [2]_.

    The method 'BD2' checks for each curve whether it lies completely inside
    bands constructed from two curves.  All permutations of two curves in the
    set of curves are used, and the band depth is normalized to one.  Due to
    the complete curve having to fall within the band, this method yields a lot
    of ties.

    The method 'MBD' is similar to 'BD2', but checks the fraction of the curve
    falling within the bands.  It therefore generates very few ties.

    References
    ----------
    .. [1] S. Lopez-Pintado and J. Romo, "On the Concept of Depth for
           Functional Data", Journal of the American Statistical Association,
           vol.  104, pp. 718-734, 2009.
    .. [2] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of
           Computational and Graphical Statistics, vol. 20, pp. 1-19, 2011.

    """
    def _band2(x1, x2, curve):
        xb = np.vstack([x1, x2])
        if np.any(curve < xb.min(axis=0)) or np.any(curve > xb.max(axis=0)):
            res = 0
        else:
            res = 1

        return res

    def _band_mod(x1, x2, curve):
        xb = np.vstack([x1, x2])
        res = np.logical_and(curve >= xb.min(axis=0),
                             curve <= xb.max(axis=0))
        return np.sum(res) / float(res.size)

    if method == 'BD2':
        band = _band2
    elif method == 'MBD':
        band = _band_mod
    else:
        raise ValueError("Unknown input value for parameter `method`.")

    num = data.shape[0]
    ix = np.arange(num)
    depth = []
    for ii in range(num):
        res = 0
        for ix1, ix2 in combinations(ix, 2):
            res += band(data[ix1, :], data[ix2, :], data[ii, :])

        # Normalize by number of combinations to get band depth
        normfactor = factorial(num) / 2. / factorial(num - 2)
        depth.append(float(res) / normfactor)

    return np.asarray(depth)
