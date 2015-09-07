import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils

def _calc_survfunc_right(time, status):
    """
    Calculate the survival function and its standard error for a single
    group.
    """

    time = np.asarray(time)
    status = np.asarray(status)

    # Convert the unique times to ranks (0, 1, 2, ...)
    time, rtime = np.unique(time, return_inverse=True)

    # Number of deaths at each unique time.
    d = np.bincount(rtime, weights=status)

    # Size of risk set just prior to each event time.
    n = np.bincount(rtime)
    n = np.cumsum(n[::-1])[::-1]

    # Only retain times where an event occured.
    ii = np.flatnonzero(d > 0)
    d = d[ii]
    n = n[ii]
    time = time[ii]

    # The survival function probabilities.
    sp = 1 - d / n.astype(np.float64)
    sp = np.log(sp)
    sp = np.cumsum(sp)
    sp = np.exp(sp)

    # Standard errors (Greenwood's formula).
    se = d / (n * (n - d)).astype(np.float64)
    se = np.cumsum(se)
    se = np.sqrt(se)
    se *= sp

    return sp, se, time, n, d



class survfunc_right(object):
    """
    Estimation and inference for a survival function.

    Only right censoring is supported.

    Parameters
    ----------
    time : array-like
        An array of times (censoring times or event times)
    status : array-like
        Status at the event time, status==1 is the 'event'
        (e.g. death, failure), meaning that the event
        occurs at the given value in `time`; status==0
        indicates that censoring has occured, meaning that
        the event occurs after the given value in `time`.
    title : string
        Optional title used for plots and summary output.

    Attributes
    ----------
    surv_prob : array-like
        The estimated value of the survivor function at each time
        point in `surv_times`.
    surv_prob_se : array-like
        The standard errors for the values in `surv_prob`.
    surv_times : array-like
        The points where the survival function changes.
    n_risk : array-like
        The number of subjects at risk just before each time value in
        `surv_times`.
    n_events : array-like
        The number of events (e.g. deaths) that occur at each point
        in `surv_times`.
    """

    def __init__(self, time, status, title=None):

        self.time = time
        self.status = status
        m = len(status)
        x = _calc_survfunc_right(time, status)
        self.surv_prob = x[0]
        self.surv_prob_se = x[1]
        self.surv_times = x[2]
        self.n_risk = x[3]
        self.n_events = x[4]
        self.title = "" if not title else title


    def plot(self, ax=None):
        """
        Plot the survival function.

        Examples
        --------
        Change the line color:

        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> ha, lb = ax.get_legend_handles_labels()
        >>> ha[0].set_color('purple')
        >>> ha[1].set_color('purple')

        Don't show the censoring points:

        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> ha, lb = ax.get_legend_handles_labels()
        >>> ha[1].set_visible(False)
        """

        return plot_survfunc(self, ax)


    def quantile(self, p):
        """
        Estimated quantile of a survival distribution.

        Parameters
        ----------
        p : float
            The probability point at which the quantile
            is determined.

        Returns the estimated quantile.
        """

        ii = np.flatnonzero(self.surv_prob < 1 - p)

        if len(ii) == 0:
            return np.nan

        return self.surv_times[ii[0]]


    def quantile_ci(self, p, alpha=0.05):
        """
        Returns a confidence interval for a survival quantile.

        Parameters
        ----------
        p : float
            The probability point for which a confidence interval is
            determined.
        alpha : float
            The confidence interval has nominal coverage probability
            1 - `alpha`.

        Returns
        -------
        lb : float
            The lower confidence limit.
        ub : float
            The upper confidence limit.

        Notes
        -----
        The confidence interval is obtained by inverting Z-tests.  The
        limits of the confidence interval will always be observed
        event times.
        """

        pc = 1 - p
        tr = norm.ppf(1 - alpha / 2)

        r = self.surv_prob - pc
        r /= self.surv_prob_se

        ii = np.flatnonzero(np.abs(r) <= tr)
        if len(ii) == 0:
            return np.nan, np.nan

        lb = self.surv_times[ii[0]]

        if ii[-1] == len(self.surv_times) - 1:
            ub = np.inf
        else:
            ub = self.surv_times[ii[-1] + 1]

        return lb, ub


def logrank(time1, status1, time2, status2):
    """
    Log-rank test for the equality of two survival distributions.

    Parameters:
    -----------
    time1 : array-like
        The event or censoring times for the first sample.
    status1 : array-like
        The censoring status variable for the first sample,
        status=1 indicates that the event occured, status=0
        indicates that the observation was censored.
    time2 : array-like
        The event or censoring times for the second sample.
    status2 : array-like
        The censoring status for the second sample, coded as
        status1.

    Returns:
    --------
    chisq : The chi-square (1 degree of freedom) distributed test
            statistic value
    pvalue : The p-value for the chi^2 test
    """

    # Get the unique times.
    utimes = np.union1d(time1, time2)

    status1 = status1.astype(np.bool)
    status2 = status2.astype(np.bool)

    # The positions of the observed event times in each group, in the
    # overall list of unique times.
    ix1 = np.searchsorted(utimes, time1[status1])
    ix2 = np.searchsorted(utimes, time2[status2])

    # Number of events observed at each time point, per group and
    # overall.
    obs1 = np.bincount(ix1, minlength=len(utimes))
    obs2 = np.bincount(ix2, minlength=len(utimes))
    obs = obs1 + obs2

    # Risk set size at each time point, per group and overall.
    nvec = []
    for time in time1, time2:
        ix = np.searchsorted(utimes, time)
        n = np.bincount(ix, minlength=len(utimes))
        n = np.cumsum(n)
        n = np.roll(n, 1)
        n[0] = 0
        n = len(time) - n
        nvec.append(n)
    n1, n2 = tuple(nvec)
    n = n1 + n2

    # The variance of event counts in the first group.
    r = n1 / n.astype(np.float64)
    var = obs * r * (1 - r) * (n - obs) / (n - 1)

    # The expected number of events in the first group.
    exp1 = obs * r

    # The Z-scale test statistic (compare to normal reference
    # distribution).
    ix = np.flatnonzero(n > 1)
    zstat = np.sum(obs1[ix] - exp1[ix]) / np.sqrt(np.sum(var[ix]))

    # The chi^2 test statistic and p-value.
    chisq = zstat**2
    pvalue = 1 - chi2.cdf(chisq, 1)

    return chisq, pvalue



def plot_survfunc(survfuncs, ax=None):
    """
    Plot one or more survivor functions.

    Arguments
    ---------
    survfuncs : object or array-like
        A single survfunc_right object, or a list or survfunc_right
        objects that are plotted together.

    Returns
    -------
    A figure instance on which the plot was drawn.

    Examples
    --------
    Add a legend:

    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> leg = fig.legend((ha[0], ha[2]), (lb[0], lb[2]), 'center right')

    Change the line colors:

    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> ha[0].set_color('purple')
    >>> ha[1].set_color('purple')
    >>> ha[2].set_color('orange')
    >>> ha[3].set_color('orange')
    """

    fig, ax = utils.create_mpl_ax(ax)

    # If we have only a single survival function to plot, put it into
    # a list.
    try:
        assert(type(survfuncs[0]) is survfunc_right)
    except:
        survfuncs = [survfuncs]

    for gx, sf in enumerate(survfuncs):

        # The estimated survival function does not include a point at
        # time 0, include it here for plotting.
        surv_times = np.concatenate(([0], sf.surv_times))
        surv_prob = np.concatenate(([1], sf.surv_prob))

        label = getattr(sf, "title", "Group %d" % (gx + 1))

        li, = ax.step(surv_times, surv_prob, '-', label=label, lw=2, where='post')

        # Plot the censored points.
        ii = np.flatnonzero(np.logical_not(sf.status))
        ti = sf.time[ii]
        jj = np.searchsorted(surv_times, ti) - 1
        sp = surv_prob[jj]
        ax.plot(ti, sp, '+', ms=12, color=li.get_color(),
                label=label + " points")

    ax.set_ylim(0, 1.01)

    return fig
