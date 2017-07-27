from statsmodels.compat.python import lzip, string_types
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tools.decorators import (resettable_cache,
                                          cache_readonly,
                                          cache_writable)

from . import utils

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
    a : float
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

    def __init__(self, data, dist=stats.norm, fit=False,
                 distargs=(), a=0, loc=0, scale=1):

        self.data = data
        self.a = a
        self.nobs = data.shape[0]
        self.distargs = distargs
        self.fit = fit

        if isinstance(dist, string_types):
            dist = getattr(stats, dist)

        self.fit_params = dist.fit(data)
        if fit:
            self.loc = self.fit_params[-2]
            self.scale = self.fit_params[-1]
            if len(self.fit_params) > 2:
                self.dist = dist(*self.fit_params[:-2],
                                 **dict(loc = 0, scale = 1))
            else:
                self.dist = dist(loc=0, scale=1)
        elif distargs or loc == 0 or scale == 1:
            self.dist = dist(*distargs, **dict(loc=loc, scale=scale))
            self.loc = loc
            self.scale = scale
        else:
            self.dist = dist
            self.loc = loc
            self.scale = scale

        # propertes
        self._cache = resettable_cache()

    @cache_readonly
    def theoretical_percentiles(self):
        return plotting_pos(self.nobs, self.a)

    @cache_readonly
    def theoretical_quantiles(self):
        try:
            return self.dist.ppf(self.theoretical_percentiles)
        except TypeError:
            msg = '%s requires more parameters to ' \
                  'compute ppf'.format(self.dist.name,)
            raise TypeError(msg)
        except:
            msg = 'failed to compute the ppf of {0}'.format(self.dist.name,)
            raise

    @cache_readonly
    def sorted_data(self):
        sorted_data = np.array(self.data, copy=True)
        sorted_data.sort()
        return sorted_data

    @cache_readonly
    def sample_quantiles(self):
        if self.fit and self.loc != 0 and self.scale != 1:
            return (self.sorted_data-self.loc)/self.scale
        else:
            return self.sorted_data

    @cache_readonly
    def sample_percentiles(self):
        quantiles = \
            (self.sorted_data - self.fit_params[-2])/self.fit_params[-1]
        return self.dist.cdf(quantiles)

    def ppplot(self, xlabel=None, ylabel=None, line=None, other=None,
               ax=None, **plotkwargs):
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
        **plotkwargs : additional matplotlib arguments to be passed to the
            `plot` command.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        if other is not None:
            check_other = isinstance(other, ProbPlot)
            if not check_other:
                other = ProbPlot(other)

            fig, ax = _do_plot(other.sample_percentiles,
                               self.sample_percentiles,
                               self.dist, ax=ax, line=line,
                               **plotkwargs)

            if xlabel is None:
                xlabel = 'Probabilities of 2nd Sample'
            if ylabel is None:
                ylabel = 'Probabilities of 1st Sample'

        else:
            fig, ax = _do_plot(self.theoretical_percentiles,
                               self.sample_percentiles,
                               self.dist, ax=ax, line=line,
                               **plotkwargs)
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
               ax=None, **plotkwargs):
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
        **plotkwargs : additional matplotlib arguments to be passed to the
            `plot` command.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        if other is not None:
            check_other = isinstance(other, ProbPlot)
            if not check_other:
                other = ProbPlot(other)

            fig, ax = _do_plot(other.sample_quantiles,
                               self.sample_quantiles,
                               self.dist, ax=ax, line=line,
                               **plotkwargs)

            if xlabel is None:
                xlabel = 'Quantiles of 2nd Sample'
            if ylabel is None:
                ylabel = 'Quantiles of 1st Sample'

        else:
            fig, ax = _do_plot(self.theoretical_quantiles,
                               self.sample_quantiles,
                               self.dist, ax=ax, line=line,
                               **plotkwargs)
            if xlabel is None:
                xlabel = "Theoretical Quantiles"
            if ylabel is None:
                ylabel = "Sample Quantiles"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig

    def probplot(self, xlabel=None, ylabel=None, line=None,
                 exceed=False, ax=None, **plotkwargs):
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
        **plotkwargs : additional matplotlib arguments to be passed to the
            `plot` command.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        if exceed:
            fig, ax = _do_plot(self.theoretical_quantiles[::-1],
                               self.sorted_data,
                               self.dist, ax=ax, line=line,
                               **plotkwargs)
            if xlabel is None:
                xlabel = 'Probability of Exceedance (%)'

        else:
            fig, ax = _do_plot(self.theoretical_quantiles,
                               self.sorted_data,
                               self.dist, ax=ax, line=line,
                               **plotkwargs)
            if xlabel is None:
                xlabel = 'Non-exceedance Probability (%)'

        if ylabel is None:
            ylabel = "Sample Quantiles"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _fmt_probplot_axis(ax, self.dist, self.nobs)

        return fig

def qqplot(data, dist=stats.norm, distargs=(), a=0, loc=0, scale=1, fit=False,
           line=None, ax=None):
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

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
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
    probplot = ProbPlot(data, dist=dist, distargs=distargs,
                         fit=fit, a=a, loc=loc, scale=scale)
    fig = probplot.qqplot(ax=ax, line=line)
    return fig

def qqplot_2samples(data1, data2, xlabel=None, ylabel=None, line=None, ax=None):
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

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
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
    >>> qqplot_2samples(pp_x, pp_y)

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

    fig = data1.qqplot(xlabel=xlabel, ylabel=ylabel,
                       line=line, other=data2, ax=ax)

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

    Notes
    -----
    There is no return value. The line is plotted on the given `ax`.
    """
    if line == '45':
        end_pts = lzip(ax.get_xlim(), ax.get_ylim())
        end_pts[0] = min(end_pts[0])
        end_pts[1] = max(end_pts[1])
        ax.plot(end_pts, end_pts, fmt)
        ax.set_xlim(end_pts)
        ax.set_ylim(end_pts)
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
        _check_for_ppf(dist)
        q25 = stats.scoreatpercentile(y, 25)
        q75 = stats.scoreatpercentile(y, 75)
        theoretical_quartiles = dist.ppf([0.25, 0.75])
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

def _fmt_probplot_axis(ax, dist, nobs):
    """
    Formats a theoretical quantile axis to display the corresponding
    probabilities on the quantiles' scale.

    Parameteters
    ------------
    ax : Matplotlib AxesSubplot instance, optional
        The axis to be formatted
    nobs : scalar
        Numbero of observations in the sample
    dist : scipy.stats.distribution
        A scipy.stats distribution sufficiently specified to impletment its
        ppf() method.

    Returns
    -------
    There is no return value. This operates on `ax` in place
    """
    _check_for_ppf(dist)
    if nobs < 50:
        axis_probs = np.array([1,2,5,10,20,30,40,50,60,
                               70,80,90,95,98,99,])/100.0
    elif nobs < 500:
        axis_probs = np.array([0.1,0.2,0.5,1,2,5,10,20,30,40,50,60,70,
                               80,90,95,98,99,99.5,99.8,99.9])/100.0
    else:
        axis_probs = np.array([0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,
                               20,30,40,50,60,70,80,90,95,98,99,99.5,
                               99.8,99.9,99.95,99.98,99.99])/100.0
    axis_qntls = dist.ppf(axis_probs)
    ax.set_xticks(axis_qntls)
    ax.set_xticklabels(axis_probs*100, rotation=45,
                       rotation_mode='anchor',
                       horizontalalignment='right',
                       verticalalignment='center')
    ax.set_xlim([axis_qntls.min(), axis_qntls.max()])

def _do_plot(x, y, dist=None, line=False, ax=None, fmt='bo', **kwargs):
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
    fig, ax = utils.create_mpl_ax(ax)
    ax.set_xmargin(0.02)
    ax.plot(x, y, fmt, **kwargs)
    if line:
        if line not in ['r','q','45','s']:
            msg = "%s option for line not understood" % line
            raise ValueError(msg)

        qqline(ax, line, x=x, y=y, dist=dist)

    return fig, ax

def _check_for_ppf(dist):
    if not hasattr(dist, 'ppf'):
        raise ValueError("distribution must have a ppf method")
