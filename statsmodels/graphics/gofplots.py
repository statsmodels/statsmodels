import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from . import utils


__all__ = ['qqplot']


def qqplot(data, dist=stats.norm, distargs=(), a=0, loc=0, scale=1, fit=False,
                line=False, ax=None):
    """
    qqplot of the quantiles of x versus the quantiles/ppf of a distribution.

    Can take arguments specifying the parameters for dist or fit them
    automatically. (See fit under kwargs.)

    Parameters
    ----------
    data : array-like
        1d data array
    dist : A scipy.stats or statsmodels distribution
        Compare x against dist. The default
        is scipy.stats.distributions.norm (a standard normal).
    distargs : tuple
        A tuple of arguments passed to dist to specify it fully
        so dist.ppf may be called.
    loc : float
        Location parameter for dist
    a : float
        Offset for the plotting position of an expected order statistic, for
        example. The plotting positions are given by (i - a)/(nobs - 2*a + 1)
        for i in range(0,nobs+1)
    scale : float
        Scale parameter for dist
    fit : boolean
        If fit is false, loc, scale, and distargs are passed to the
        distribution. If fit is True then the parameters for dist
        are fit automatically using dist.fit. The quantiles are formed
        from the standardized data, after subtracting the fitted loc
        and dividing by the fitted scale.
    line : str {'45', 's', 'r', q'} or None
        Options for the reference line to which the data is compared.:

        - '45' - 45-degree line
        - 's' - standardized line, the expected order statistics are scaled
          by the standard deviation of the given sample and have the mean
          added to them
        - 'r' - A regression line is fit
        - 'q' - A line is fit through the quartiles.
        - None - by default no reference line is added to the plot.
        - If True a reference line is drawn on the graph. The default is to
          fit a line via OLS regression.

    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from matplotlib import pyplot as plt

    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> mod_fit = sm.OLS(data.endog, data.exog).fit()
    >>> res = mod_fit.resid
    >>> fig = sm.qqplot(res)
    >>> plt.show()

    qqplot against quantiles of t-distribution with 4 degrees of freedom:

    >>> import scipy.stats as stats
    >>> fig = sm.qqplot(res, stats.t, distargs=(4,))
    >>> plt.show()

    qqplot against same as above, but with mean 3 and std 10:

    >>> fig = sm.qqplot(res, stats.t, distargs=(4,), loc=3, scale=10)
    >>> plt.show()

    Automatically determine parameters for t distribution including the
    loc and scale:

    >>> fig = sm.qqplot(res, stats.t, fit=True, line='45')
    >>> plt.show()

    Notes
    -----
    Depends on matplotlib. If `fit` is True then the parameters are fit using
    the distribution's fit() method.

    """
    fig, ax = utils.create_mpl_ax(ax)

    if not hasattr(dist, 'ppf'):
        raise ValueError("distribution must have a ppf method")

    nobs = data.shape[0]

    if fit:
        fit_params = dist.fit(data)
        loc = fit_params[-2]
        scale = fit_params[-1]
        if len(fit_params)>2:
            dist = dist(*fit_params[:-2], loc = 0, scale = 1)
        else:
            dist = dist(loc=0, scale=1)
    elif distargs or loc != 0 or scale != 1:
        dist = dist(*distargs, **dict(loc=loc, scale=scale))


    try:
        theoretical_quantiles = dist.ppf(plotting_pos(nobs, a))
    except:
        raise ValueError('distribution requires more parameters')

    sample_quantiles = np.array(data, copy=True)
    sample_quantiles.sort()
    if fit:
        sample_quantiles -= loc
        sample_quantiles /= scale


    ax.set_xmargin(0.02)
    ax.plot(theoretical_quantiles, sample_quantiles, 'bo')
    if line:
        if line not in ['r','q','45','s']:
            msg = "%s option for line not understood" % line
            raise ValueError(msg)

        qqline(ax, line, theoretical_quantiles, sample_quantiles, dist)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    return fig


def qqline(ax, line, x=None, y=None, dist=None, fmt='r-'):
    """
    Plot a reference line for a qqplot.

    Parameters
    ----------
    ax : matplotlib axes instance
        The axes on which to plot the line
    line : str {'45','r','s','q'}
        Options for the reference line to which the data is compared.:

        - '45' - 45-degree line
        - 's'  - standardized line, the expected order statistics are scaled by          the standard deviation of the given sample and have the mean added
          to them
        - 'r'  - A regression line is fit
        - 'q'  - A line is fit through the quartiles.
        - None - By default no reference line is added to the plot.
    x : array
        X data for plot. Not needed if line is '45'.
    y : array
        Y data for plot. Not needed if line is '45'.
    dist : scipy.stats.distribution
        A scipy.stats distribution, needed if line is 'q'.

    Notes
    -----
    There is no return value. The line is plotted on the given `ax`.
    """
    if line == '45':
        end_pts = zip(ax.get_xlim(), ax.get_ylim())
        end_pts[0] = max(end_pts[0])
        end_pts[1] = min(end_pts[1])
        ax.plot(end_pts, end_pts, fmt)
        return # does this have any side effects?
    if x is None and y is None:
        raise ValueError("If line is not 45, x and y cannot be None.")
    elif line == 'r':
        # could use ax.lines[0].get_xdata(), get_ydata(),
        # but don't know axes are 'clean'
        y = OLS(y, add_constant(x)).fit().fittedvalues
        ax.plot(x,y,fmt)
    elif line == 's':
        m,b = y.std(), y.mean()
        ref_line = x*m + b
        ax.plot(x, ref_line, fmt)
    elif line == 'q':
        q25 = stats.scoreatpercentile(y, 25)
        q75 = stats.scoreatpercentile(y, 75)
        theoretical_quartiles = dist.ppf([.25,.75])
        m = (q75 - q25) / np.diff(theoretical_quartiles)
        b = q25 - m*theoretical_quartiles[0]
        ax.plot(x, m*x + b, fmt)


#about 10x faster than plotting_position in sandbox and mstats
def plotting_pos(nobs, a):
    """
    Generates sequence of plotting positions

    Parameters
    ----------
    nobs : int
        Number of probability points to plot
    a : float
        Offset for the plotting position of an expected order statistic, for
        example.

    Returns
    -------
    plotting_positions : array
        The plotting positions

    Notes
    -----
    The plotting positions are given by (i - a)/(nobs - 2*a + 1) for i in
    range(0,nobs+1)

    See also
    --------
    scipy.stats.mstats.plotting_positions
    """
    return (np.arange(1.,nobs+1) - a)/(nobs- 2*a + 1)
