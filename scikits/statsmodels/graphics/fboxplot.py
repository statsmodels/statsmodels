"""Module for functional boxplots."""

import itertools

import numpy as np
from scipy import stats
from scipy.misc import factorial


__all__ = ['fboxplot', 'banddepth']


def fboxplot(data, xdata=None, labels=None, wfactor=1.5, ax=None):
    """Plot functional boxplot.

    A functional boxplot is the analog of a boxplot for functional data.
    Functional data is any type of data that varies over a continuum, i.e.
    curves, probabillity distributions, seasonal data, etc.

    The data is first ordered, the order statistic used here is `banddepth`.
    Plotted are then the "most central" curve, the envelope of the 50% central
    region, the maximum non-outlying envelope and the outlier curves.

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
    wfactor : float, optional
        Factor by which the central 50% region is multiplied to find the outer
        region (analog of "whiskers" of a classical boxplot).
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        The created figure.  If `ax` is not None, the returned value for `fig`
        is None.

    See Also
    --------
    banddepth

    Notes
    -----

    References
    ----------
    [1] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of Computational
        and Graphical Statistics, vol. 20, pp. 1-19, 2011.
    [2] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-25, 2010.

    Examples
    --------
    >>> data = sm.datasets.elnino.load()

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = fboxplot(data.raw_data[:, 1:], wfactor=2.58,
                   labels=data.raw_data[:, 0].astype(int), ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])

    >>> plt.show()

    """
    # TODO:
    # - add tests
    # - factor out part of the algorithm if useful by itself
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError("Matplotlib is not found.")

    data = np.asarray(data)

    if xdata is None:
        xdata = np.arange(data.shape[1])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    # Inner area is 25%-75% region of band-depth ordered curves.
    depth = banddepth(data, method='MBD')
    depth_ix = np.argsort(depth)[::-1]
    middle = data[depth_ix[0], :]
    ix_IQR = data.shape[0] // 2
    lower = data[depth_ix[0:ix_IQR], :].min(axis=0)
    upper = data[depth_ix[0:ix_IQR], :].max(axis=0)

    # Determine region for outlier detection
    inner_median = np.median(data[depth_ix[0:ix_IQR], :], axis=0)
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

    # Plot envelope of all non-outlying data
    lower_nonout = data[ix_nonout, :].min(axis=0)
    upper_nonout = data[ix_nonout, :].max(axis=0)
    ax.fill_between(xdata, lower_nonout, upper_nonout, color=(0.75,0.75,0.75))

    # Plot central 50% region
    ax.fill_between(xdata, lower, upper, color=(0.5,0.5,0.5))

    # Plot median curve
    ax.plot(xdata, middle, 'k-', lw=2)

    # Plot outliers
    for ii, ix in enumerate(ix_outliers):
        label = str(labels[ix]) if labels is not None else None
        ax.plot(xdata, data[ix, :],
                color=plt.cm.rainbow(float(ii) / (len(ix_outliers)-1)),
                label=label)

    if labels is not None:
        ax.legend()

    return fig, depth, depth_ix


def banddepth(data, method='MBD'):
    """Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves are the 50%
    central region.

    Parameters
    ----------
    data : ndarray
        The vectors of functions to create a functional boxplot from.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    method : {'BD2', 'MBD'}, optional
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
    [1] S. Lopez-Pintado and J. Romo, "On the Concept of Depth for Functional
        Data", Journal of the American Statistical Association, vol.  104, pp.
        718-734, 2009.
    [2] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of Computational
        and Graphical Statistics, vol. 20, pp. 1-19, 2011.

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
        res = np.logical_and(curve > xb.min(axis=0),
                             curve < xb.max(axis=0))
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
        for ix1, ix2 in itertools.combinations(ix, 2):
            res += band(data[ix1, :], data[ix2, :], data[ii, :])

        # Normalize by number of combinations to get band depth
        normfactor = factorial(num) / 2. / factorial(num - 2)
        depth.append(float(res) / normfactor)

    return np.asarray(depth)


def testfunc_harm(t):
    # Constant, 0 with p=0.9, 1 with p=1 - for creating outliers
    ci = int(np.random.random() > 0.9)
    a1i = np.random.random() * 0.05
    a2i = np.random.random() * 0.05
    b1i = (0.15 - 0.1) * np.random.random() + 0.1
    b2i = (0.15 - 0.1) * np.random.random() + 0.1

    func = (1 - ci) * (a1i * np.sin(t) + a2i * np.cos(t)) + \
           ci * (b1i * np.sin(t) + b2i * np.cos(t))

    return func


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Some basic test data, Model 6 from Sun and Genton.
    t = np.linspace(0, 2 * np.pi, 250)
    data = []
    for ii in range(40):
        data.append(testfunc_harm(t))

    # Create a plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, _, ix = fboxplot(data, wfactor=2, ax=ax)
    ax.set_xlabel(r'$t$')
    ax.text(100, 0.16, r'$(1-c_i)\{a_{1i}sin(t)+a_{2i}cos(t)\}$')
    ax.set_ylabel(r'$y(t)$')
    plt.show()
