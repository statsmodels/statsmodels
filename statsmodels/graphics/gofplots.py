import numpy as np
from scipy import stats
import matplotlib.scale as mscale

from statsmodels.compat.python import lzip, string_types
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tools.decorators import (resettable_cache,
                                          cache_readonly,
                                          cache_writable)

from . import utils
from ._probscale import ProbScale
mscale.register_scale(ProbScale)

__all__ = ['qqplot', 'qqplot_2samples', 'qqline', 'ProbPlot']

class ProbPlot(object):
    """
    Class for convenient construction of Q-Q, P-P, and probability plots.

    Can take arguments specifying the parameters for dist or fit them
    automatically. (See fit under kwargs.)

    Parameters
    ----------
    data : array-like
        1d data array
    dist : A scipy.stats or statsmodels distribution
        Compare x against dist. The default is
        scipy.stats.distributions.norm (a standard normal).
    distargs : tuple
        A tuple of arguments passed to dist to specify it fully
        so dist.ppf may be called.
    loc : float
        Location parameter for dist
    a, b : optional float (default = 0)
        Offset for the plotting position of an expected order
        statistic, for example. The plotting positions are given
        by (i - a)/(nobs - 2*a + 1) for i in range(0,nobs+1)
    scale : float
        Scale parameter for dist
    fit : boolean
        If fit is false, loc, scale, and distargs are passed to the
        distribution. If fit is True then the parameters for dist
        are fit automatically using dist.fit. The quantiles are formed
        from the standardized data, after subtracting the fitted loc
        and dividing by the fitted scale.

    See Also
    --------
    scipy.stats.probplot

    Notes
    -----
    1) Depends on matplotlib.
    2) If `fit` is True then the parameters are fit using the
        distribution's `fit()` method.
    3) The call signatures for the `qqplot`, `ppplot`, and `probplot`
        methods are similar, so examples 1 through 4 apply to all
        three methods.
    4) The three plotting methods are summarized below:
        ppplot : Probability-Probability plot
            Compares the sample and theoretical probabilities (percentiles).
        qqplot : Quantile-Quantile plot
            Compares the sample and theoretical quantiles
        probplot : Probability plot
            Same as a Q-Q plot, however probabilities are shown in the scale of
            the theoretical distribution (x-axis) and the y-axis contains
            unscaled quantiles of the sample data.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from matplotlib import pyplot as plt

    >>> # example 1
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> model = sm.OLS(data.endog, data.exog)
    >>> mod_fit = model.fit()
    >>> res = mod_fit.resid # residuals
    >>> probplot = sm.ProbPlot(res)
    >>> probplot.qqplot()
    >>> plt.show()

    qqplot of the residuals against quantiles of t-distribution with 4
    degrees of freedom:

    >>> # example 2
    >>> import scipy.stats as stats
    >>> probplot = sm.ProbPlot(res, stats.t, distargs=(4,))
    >>> fig = probplot.qqplot()
    >>> plt.show()

    qqplot against same as above, but with mean 3 and std 10:

    >>> # example 3
    >>> probplot = sm.ProbPlot(res, stats.t, distargs=(4,), loc=3, scale=10)
    >>> fig = probplot.qqplot()
    >>> plt.show()

    Automatically determine parameters for t distribution including the
    loc and scale:

    >>> # example 4
    >>> probplot = sm.ProbPlot(res, stats.t, fit=True)
    >>> fig = probplot.qqplot(line='45')
    >>> plt.show()

    A second `ProbPlot` object can be used to compare two seperate sample
    sets by using the `other` kwarg in the `qqplot` and `ppplot` methods.

    >>> # example 5
    >>> import numpy as np
    >>> x = np.random.normal(loc=8.25, scale=2.75, size=37)
    >>> y = np.random.normal(loc=8.75, scale=3.25, size=37)
    >>> pp_x = sm.ProbPlot(x, fit=True)
    >>> pp_y = sm.ProbPlot(y, fit=True)
    >>> fig = pp_x.qqplot(line='45', other=pp_y)
    >>> plt.show()

    The following plot displays some options, follow the link to see the
    code.

    .. plot:: plots/graphics_gofplots_qqplot.py
    """

    def __init__(self, data, dist=stats.norm, fit=False, a=0, b=0,
                 loc=0, scale=1, distargs=()):

        self.data = data
        self.a = a
        self.nobs = data.shape[0]
        self.distargs = distargs
        self.loc = loc
        self.scale = scale

        self.fit = fit

        if isinstance(dist, string_types):
            self._userdist = getattr(stats, dist)
        else:
            self._userdist = dist

        self._userdist_is_frozen = isinstance(self._userdist,
                                              stats.distributions.rv_frozen)

        self._dist = None
        self._cache = resettable_cache()

    def _get_dist(self):
        if self._userdist_is_frozen:
            dist = self._userdist
        else:
            if self.fit:
                dist = self._userdist(*self._userdist.fit(self.data))
            else:
                dist = self._userdist(
                    *self.distargs, loc=self.loc, scale=self.scale
                )

        _check_dist(dist)

        return dist

    @property
    def dist(self):
        self._dist = self._get_dist()
        self.distargs = self._dist.args
        self.loc = self._dist.kwds.get('loc', 0)
        self.scale = self._dist.kwds.get('scale', 1)
        return self._dist

    @cache_readonly
    def theoretical_percentiles(self):
        return plotting_pos(self.nobs, self.a)

    @cache_readonly
    def theoretical_quantiles(self):
        return self.dist.ppf(self.theoretical_percentiles)

    @cache_readonly
    def sorted_data(self):
        sorted_data = np.array(self.data, copy=True)
        sorted_data.sort()
        return sorted_data

    @cache_readonly
    def scaled_data(self):
        if self.loc == 0 and self.scale == 1:
            return (self.sorted_data-np.mean(self.data))/np.std(self.data)
        else:
            return self.sample_quantiles

    @cache_readonly
    def sample_quantiles(self):
        return self.sorted_data #(self.sorted_data-self.loc)/self.scale

    @cache_readonly
    def sample_percentiles(self):
        if self.fit or self._userdist_is_frozen:
            return self.dist.cdf(self.sorted_data)
        else:
            return self.dist.cdf(self.scaled_data)

    def ppplot(self, xlabel=None, ylabel=None, line=None, other=None,
               ax=None, plot_options={}):
        """
        P-P plot of the percentiles (probabilities) of x versus the
        probabilities (percetiles) of a distribution.

        Parameters
        ----------
        xlabel, ylabel : str or None, optional
            User-provided lables for the x-axis and y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : str {'45', 's', 'r', q'} or None, optional
            Options for the reference line to which the data is compared:

            - '45' - 45-degree line
            - 's' - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - 'r' - A regression line is fit
            - 'q' - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        other : `ProbPlot` instance, array-like, or None, optional
            If provided, the sample quantiles of this `ProbPlot` instance are
            plotted against the sample quantiles of the `other` `ProbPlot`
            instance. If an array-like object is provided, it will be turned
            into a `ProbPlot` instance using default parameters. If not provided
            (default), the theoretical quantiles are used.
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        plot_options : dict of additional matplotlib arguments to be passed to
            the `plot` command.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure. Otherwise the figure to which
            `ax` is connected.

        """
        if other is not None:
            check_other = isinstance(other, ProbPlot)
            if not check_other:
                other = ProbPlot(other)

            fig, ax = _do_plot(other.sample_percentiles,
                               self.sample_percentiles,
                               self.dist, ax=ax, line=line,
                               plot_options=plot_options)

            if xlabel is None:
                xlabel = 'Probabilities of 2nd Sample'
            if ylabel is None:
                ylabel = 'Probabilities of 1st Sample'

        else:
            fig, ax = _do_plot(self.theoretical_percentiles,
                               self.sample_percentiles,
                               self.dist, ax=ax, line=line,
                               plot_options=plot_options)
            if xlabel is None:
                xlabel = "Theoretical Probabilities"
            if ylabel is None:
                ylabel = "Sample Probabilities"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        return fig

    def qqplot(self, xlabel=None, ylabel=None, line=None, other=None,
               ax=None, plot_options={}):
        """
        Q-Q plot of the quantiles of x versus the quantiles/ppf of a
        distribution or the quantiles of another `ProbPlot` instance.

        Parameters
        ----------
        xlabel, ylabel : str or None, optional
            User-provided lables for the x-axis and y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : str {'45', 's', 'r', q'} or None, optional
            Options for the reference line to which the data is compared:

            - '45' - 45-degree line
            - 's' - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - 'r' - A regression line is fit
            - 'q' - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        other : `ProbPlot` instance, array-like, or None, optional
            If provided, the sample quantiles of this `ProbPlot` instance are
            plotted against the sample quantiles of the `other` `ProbPlot`
            instance. If an array-like object is provided, it will be turned
            into a `ProbPlot` instance using default parameters. If not
            provided (default), the theoretical quantiles are used.
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        plot_options : dict of additional matplotlib arguments to be passed to
            the `plot` command.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure. Otherwise the figure to which
            `ax` is connected.

        """
        if other is not None:
            check_other = isinstance(other, ProbPlot)
            if not check_other:
                other = ProbPlot(other)

            fig, ax = _do_plot(other.sample_quantiles,
                               self.sample_quantiles,
                               self.dist, ax=ax, line=line,
                               plot_options=plot_options)

            if xlabel is None:
                xlabel = 'Quantiles of 2nd Sample'
            if ylabel is None:
                ylabel = 'Quantiles of 1st Sample'

        else:
            fig, ax = _do_plot(self.theoretical_quantiles,
                               self.sample_quantiles,
                               self.dist, ax=ax, line=line,
                               plot_options=plot_options)
            if xlabel is None:
                xlabel = "Theoretical Quantiles"
            if ylabel is None:
                ylabel = "Sample Quantiles"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig

    def probplot(self, xlabel=None, ylabel=None, line=None,
                 exceed=False, ax=None, plot_options={}):
        """
        Probability plot of the unscaled quantiles of x versus the
        probabilities of a distibution (not to be confused with a P-P plot).

        The x-axis is scaled linearly with the quantiles, but the probabilities
        are used to label the axis.

        Parameters
        ----------
        xlabel, ylabel : str or None, optional
            User-provided lables for the x-axis and y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : str {'45', 's', 'r', q'} or None, optional
            Options for the reference line to which the data is compared:

            - '45' - 45-degree line
            - 's' - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - 'r' - A regression line is fit
            - 'q' - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        exceed : boolean, optional

             - If False (default) the raw sample quantiles are plotted against
               the theoretical quantiles, show the probability that a sample
               will not exceed a given value
             - If True, the theoretical quantiles are flipped such that the
               figure displays the probability that a sample will exceed a
               given value.
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        plot_options : dict of additional matplotlib arguments to be passed to
            the `plot` command.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure. Otherwise the figure to which
            `ax` is connected.

        """

        if ylabel is None:
            ylabel = "Sample Quantiles"

        if exceed:
            pcnts = self.theoretical_percentiles[::-1]
            if xlabel is None:
                xlabel = 'Probability of Exceedance (%)'
        else:
            pcnts = self.theoretical_percentiles
            if xlabel is None:
                xlabel = 'Non-exceedance Probability (%)'


        fig, ax = _do_plot(pcnts * 100, self.sorted_data, self.dist, ax=ax,
                           line=line, plot_options=plot_options)
        ax.set_xscale('prob', dist=self.dist)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig


def qqplot(data, dist=stats.norm, distargs=(), a=0, fit=False,
           line=False, ax=None, plot_options={}):
    """
    Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution.

    Can take arguments specifying the parameters for dist or fit them
    automatically. (See fit under Parameters.)

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
        Options for the reference line to which the data is compared:

        - '45' - 45-degree line
        - 's' - standardized line, the expected order statistics are scaled
          by the standard deviation of the given sample and have the mean
          added to them
        - 'r' - A regression line is fit
        - 'q' - A line is fit through the quartiles.
        - None - by default no reference line is added to the plot.

    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    plot_options : dict of additional matplotlib arguments to be passed to
        the `plot` command.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure. Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    scipy.stats.probplot

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from matplotlib import pyplot as plt
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> mod_fit = sm.OLS(data.endog, data.exog).fit()
    >>> res = mod_fit.resid # residuals
    >>> fig = sm.qqplot(res)
    >>> plt.show()

    qqplot of the residuals against quantiles of t-distribution with 4 degrees
    of freedom:

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

    The following plot displays some options, follow the link to see the code.

    .. plot:: plots/graphics_gofplots_qqplot.py

    Notes
    -----
    Depends on matplotlib. If `fit` is True then the parameters are fit using
    the distribution's fit() method.

    """
    probplot = ProbPlot(data, dist=dist, distargs=distargs, fit=fit, a=a)
    fig = probplot.qqplot(ax=ax, line=line, plot_options=plot_options)
    return fig


def qqplot_2samples(data1, data2, xlabel=None, ylabel=None, line=None,
                    ax=None, plot_options={}):
    """
    Q-Q Plot of two samples' quantiles.

    Can take either two `ProbPlot` instances or two array-like objects. In the
    case of the latter, both inputs will be converted to `ProbPlot` instances
    using only the default values - so use `ProbPlot` instances if
    finer-grained control of the quantile computations is required.

    Parameters
    ----------
    data1, data2 : array-like (1d) or `ProbPlot` instances
    xlabel, ylabel : str or None
        User-provided labels for the x-axis and y-axis. If None (default),
        other values are used.
    line : str {'45', 's', 'r', q'} or None
        Options for the reference line to which the data is compared:

        - '45' - 45-degree line
        - 's' - standardized line, the expected order statistics are scaled
          by the standard deviation of the given sample and have the mean
          added to them
        - 'r' - A regression line is fit
        - 'q' - A line is fit through the quartiles.
        - None - by default no reference line is added to the plot.

    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    plot_options : dict of additional matplotlib arguments to be passed to
        the `plot` command.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure. Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    scipy.stats.probplot

    Examples
    --------
    >>> x = np.random.normal(loc=8.5, scale=2.5, size=37)
    >>> y = np.random.normal(loc=8.0, scale=3.0, size=37)
    >>> pp_x = sm.ProbPlot(x)
    >>> pp_y = sm.ProbPlot(y)
    >>> qqplot_2samples(data1, data2, xlabel=None, ylabel=None, line=None, ax=None):

    Notes
    -----
    1) Depends on matplotlib.
    2) If `data1` and `data2` are not `ProbPlot` instances, instances will be
       created using the default parameters. Therefore, it is recommended to use
       `ProbPlot` instance if fine-grained control is needed in the computation
       of the quantiles.

    """
    check_data1 = isinstance(data1, ProbPlot)
    check_data2 = isinstance(data2, ProbPlot)

    if not check_data1 and not check_data2:
        data1 = ProbPlot(data1)
        data2 = ProbPlot(data2)

    fig = data1.qqplot(xlabel=xlabel, ylabel=ylabel, line=line, other=data2,
                       ax=ax, plot_options=plot_options)

    return fig


def qqline(ax, line, x=None, y=None, dist=None, fmt='r-', **lineoptions):
    """
    Plot a reference line for a qqplot.

    Parameters
    ----------
    ax : matplotlib axes instance
        The axes on which to plot the line
    line : str {'45','r','s','q'}
        Options for the reference line to which the data is compared.:

        - '45' - 45-degree line
        - 's'  - standardized line, the expected order statistics are scaled by
                 the standard deviation of the given sample and have the mean
                 added to them
        - 'r'  - A regression line is fit
        - 'q'  - A line is fit through the quartiles.
        - None - By default no reference line is added to the plot.

    x : array
        X data for plot. Not needed if line is '45'.
    y : array
        Y data for plot. Not needed if line is '45'.
    dist : scipy.stats.distribution
        A scipy.stats distribution, needed if line is 'q'.

    Returns
    -------
    Matplotlib line artist.

    Notes
    -----
    There is no return value. The line is plotted on the given `ax`.

    """

    valid_lines = ['45', 'q', 'r', 's']
    if line not in valid_lines:
        raise ValueError('`line` must be one of {0}'.format(valid_lines))

    if line == '45':
        end_pts = lzip(ax.get_xlim(), ax.get_ylim())
        end_pts[0] = min(end_pts[0])
        end_pts[1] = max(end_pts[1])
        lineartist, = ax.plot(end_pts, end_pts, fmt, **lineoptions)
        ax.set_xlim(end_pts)
        ax.set_ylim(end_pts)

    else:
        if x is None or y is None:
            raise ValueError("If line is not 45, x and y cannot be None.")
        else:
            x = np.array(x)
            y = np.array(y)

        if line == 'r':
            # could use ax.lines[0].get_xdata(), get_ydata(),
            # but don't know axes are 'clean'
            y = OLS(y, add_constant(x)).fit().fittedvalues
            lineartist, = ax.plot(x, y, fmt, **lineoptions)

        elif line == 's':
            m, b = np.std(y), np.mean(y)
            ref_line = x*m + b
            lineartist, = ax.plot(x, ref_line, fmt, **lineoptions)

        elif line == 'q':
            _check_dist(dist)
            q25 = stats.scoreatpercentile(y, 25)
            q75 = stats.scoreatpercentile(y, 75)
            theoretical_quartiles = dist.ppf([0.25, 0.75])
            m = (q75 - q25) / np.diff(theoretical_quartiles)
            b = q25 - m*theoretical_quartiles[0]
            lineartist, = ax.plot(x, m*x + b, fmt, **lineoptions)

    return lineartist


#about 10x faster than plotting_position in sandbox and mstats
def plotting_pos(nobs, a=0, b=0):
    """
    Generates sequence of plotting positions

    Parameters
    ----------
    nobs : int
        Number of probability points to plot
    a, b : optional float (defaults are 0)
        alpha and beta parameters for the plotting position of an expected
        order statistic, for example.

    Returns
    -------
    plotting_positions : array
        The plotting positions

    Notes
    -----
    The plotting positions are given by (i - a)/(nobs + 1 - a - b) for i in
    range(1, nobs+1)

    See also
    --------
    scipy.stats.mstats.plotting_positions for more info on alpha and beta

    """
    #mstats:(i-alpha)/(n+1-alpha-beta)
    return (np.arange(1., nobs+1) - a) / (nobs + 1 - a - b)


def _do_plot(x, y, dist=None, line=None, ax=None, step=False,
             plot_options={}):
    """
    Boiler plate plotting function for the `ppplot`, `qqplot`, and
    `probplot` methods of the `ProbPlot` class

    Parameteters
    ------------
    x, y : array-like
        Data to be plotted
    dist : scipy.stats.distribution
        A scipy.stats distribution, needed if `line` is 'q'.
    line : str {'45', 's', 'r', q'} or None
        Options for the reference line to which the data is compared.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    fmt : str, optional
        matplotlib-compatible formatting string for the data markers
    kwargs : keywords
        These are passed to matplotlib.plot

    Returns
    -------
    fig : Matplotlib Figure instance
    ax : Matplotlib AxesSubplot instance (see Parameters)

    """
    plot_style = {
        'marker': 'o',
        'markerfacecolor': 'blue',
        'linestyle': 'none'
    }

    plot_style.update(**plot_options)
    where = plot_style.pop('where', 'pre')

    fig, ax = utils.create_mpl_ax(ax)
    ax.set_xmargin(0.02)

    if step:
        ax.step(x, y, where=where, **plot_style)
    else:
        ax.plot(x, y, **plot_style)
    if line is not None:
        if line not in ['r', 'q', '45', 's']:
            msg = "'%s' option for line not understood" % line
            raise ValueError(msg)

        qqline(ax, line, x=x, y=y, dist=dist)

    return fig, ax


def _check_dist(dist):
    if not hasattr(dist, 'ppf') or not hasattr(dist, 'cdf'):
        raise ValueError("distribution must have ppf and cdf methods")
