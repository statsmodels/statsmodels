import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from . import utils


__all__ = ['qqplot', 'qqline', 'ProbPlot']


class ProbPlot(object):
    def __init__(self, data, dist=stats.norm, fit=False,
                 distargs=(), a=0, loc=0, scale=1):
        """
        Class for convenient construction of Q-Q, P-P, and probability plots.

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

        Plotting Methods
        ----------------
        All plotting methods listed below have the same call signatures which
        accept `line` and `ax` keyword arguments. See individual docstrings
        for more info.

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

        Sometimes Q-Q plots can be applied to two-sample problems to test the
        equality of the distributions between to sample sets. In that case,
        it's best to create a `ProbPlot` instance for each sample and then to
        plot each instance's `sample_quantiles` attributes:

        >>> import numpy as np
        >>> x = np.random.normal(loc=8.5, scale=2.5, size=37)
        >>> y = np.random.normal(loc=8.0, scale=3.0, size=37)
        >>> pp_x = sm.ProbPlot(x)
        >>> pp_y = sm.ProbPlot(y)
        >>> fig, ax = plt.subplots(nrows=1, ncols=1)
        >>> ax.plot(pp_x.sample_quantiles, pp_y.sample_quantiles, 'bo')
        >>> qqline(ax, line='45') # best-choice when evaluating equality
        >>> ax.set_xlabel('Quantiles of $x$')
        >>> ax.set_ylabel('Quantiles of $y$')

        The following plot displays some options, follow the link to see the
        code.

        .. plot:: plots/graphics_gofplots_qqplot.py

        Notes
        -----
        1) Depends on matplotlib.
        2) If `fit` is True then the parameters are fit using the
            distribution's `fit()` method.
        3) The call signatures for the `qqplot`, `ppplot`, and `probplot`
            methods are all identical, so examples 1 through 4 apply to all
            three methods.
        """

        if not hasattr(dist, 'ppf'):
            raise ValueError("distribution must have a ppf method")

        self.data = data
        self.a = a
        self.nobs = data.shape[0]
        self.distargs = distargs

        fit_params = dist.fit(data)
        if fit:
            self.loc = fit_params[-2]
            self.scale = fit_params[-1]
            if len(fit_params) > 2:
                self.dist = dist(*fit_params[:-2], **dict(loc = 0, scale = 1))
            else:
                self.dist = dist(loc=0, scale=1)
        elif distargs or loc == 0 or scale == 1:
            self.dist = dist(*distargs, **dict(loc=loc, scale=scale))
        else:
            self.dist = dist
            self.loc = loc
            self.scale = scale

        try:
            self.theoretical_percentiles = plotting_pos(self.nobs, self.a)
            self.theoretical_quantiles = self.dist.ppf(self.theoretical_percentiles)
        except:
            raise ValueError('distribution requires more parameters')

        self.sample_quantiles = np.array(data, copy=True)
        self.sample_quantiles.sort()
        self.raw_sample_quantiles = self.sample_quantiles.copy()
        fit_quantiles = (self.sample_quantiles - fit_params[-2])/fit_params[-1]
        self.sample_percentiles = self.dist.cdf(fit_quantiles)
        if fit and loc != 0 and scale != 1:
            self.sample_quantiles = fit_quantiles

    def ppplot(self, ax=None, line=False):
        """
        P-P plot of the percentiles (probabilities) of x versus the
        probabilities (percetiles) of a distribution.

        Parameters
        ----------
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new
            figure being created.
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

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        fig, ax = _do_plot(self.theoretical_percentiles,
                           self.sample_percentiles,
                           self.dist,
                           ax=ax, line=line)

        ax.set_ylabel("Sample Probabilities")
        ax.set_xlabel("Theoretical Probabilities")

        return fig

    def qqplot(self, ax=None, line=False):
        """
        Q-Q plot of the quantiles of x versus the quantiles/ppf of a
        distribution.

        Parameters
        ----------
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
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

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        fig, ax = _do_plot(self.theoretical_quantiles,
                           self.sample_quantiles,
                           self.dist,
                           ax=ax, line=line)

        ax.set_ylabel("Sample Quantiles")
        ax.set_xlabel("Theoretical Quantiles")

        return fig

    def probplot(self, ax=None, line=False):
        """
        Probability plot of the unscaled quantiles of x versus the
        probabilities of a distibution (not to be confused with a P-P plot).

        The x-axis is scaled linearly with the quantiles, but the probabilities
        are used to label the axis.

        Parameters
        ----------
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
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

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
        fig, ax = _do_plot(self.theoretical_quantiles,
                           self.raw_sample_quantiles,
                           self.dist,
                           ax=ax, line=line)

        ax.set_ylabel("Sample Quantiles")
        ax.set_xlabel('Non-exceedance Probability (%)')
        _fmt_probplot_axis(ax, self.dist, self.nobs)

        return fig

def _do_plot(x, y, dist, ax=None, line=False):
    """
    Boiler plate plotting function for the `ppplot`, `qqplot`, and
    `probplot` methods of the `ProbPlot` class
    """
    fig, ax = utils.create_mpl_ax(ax)
    ax.set_xmargin(0.02)
    ax.plot(x, y, 'bo')
    if line:
        if line not in ['r','q','45','s']:
            msg = "%s option for line not understood" % line
            raise ValueError(msg)

        qqline(ax, line, x, y, dist)
    return fig, ax

def qqplot(data_y, data, dist=stats.norm, distargs=(), a=0, loc=0, scale=1, fit=False,
           line=False, prob=False, ax=None):
    """
    qqplot of the quantiles of x versus the quantiles/ppf of a distribution.

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
        end_pts = zip(ax.get_xlim(), ax.get_ylim())
        end_pts[0] = max(end_pts[0])
        end_pts[1] = min(end_pts[1])
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

def _fmt_probplot_axis(ax, dist, nobs):
    """
    Formats a theoretical quantile axis to display the corresponding
    probabilities on the quantiles' scale.
    """
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
