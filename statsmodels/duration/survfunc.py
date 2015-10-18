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


class SurvfuncRight(object):
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

        self.time = np.asarray(time)
        self.status = np.asarray(status)
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
        >>> li = ax.get_lines()
        >>> li[0].set_color('purple')
        >>> li[1].set_color('purple')

        Don't show the censoring points:

        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[1].set_visible(False)
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

        # SAS uses a strict inequality here.
        ii = np.flatnonzero(self.surv_prob < 1 - p)

        if len(ii) == 0:
            return np.nan

        return self.surv_times[ii[0]]


    def quantile_ci(self, p, alpha=0.05, method='cloglog'):
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
        method : string
            Function to use for g-transformation, must be ...

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

        References
        ----------
        The method is based on the approach used in SAS, documented here:

          http://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/viewer.htm#statug_lifetest_details03.htm
        """

        tr = norm.ppf(1 - alpha / 2)

        method = method.lower()
        if method == "cloglog":
            g = lambda x : np.log(-np.log(x))
            gprime = lambda x : -1 / (x * np.log(x))
        elif method == "linear":
            g = lambda x : x
            gprime = lambda x : 1
        elif method == "log":
            g = lambda x : np.log(x)
            gprime = lambda x : 1 / x
        elif method == "logit":
            g = lambda x : np.log(x / (1 - x))
            gprime = lambda x : 1 / (x * (1 - x))
        elif method == "asinsqrt":
            g = lambda x : np.arcsin(np.sqrt(x))
            gprime = lambda x : 1 / (2 * np.sqrt(x) * np.sqrt(1 - x))
        else:
            raise ValueError("unknown method")

        r = g(self.surv_prob) - g(1 - p)
        r /= (gprime(self.surv_prob) * self.surv_prob_se)

        ii = np.flatnonzero(np.abs(r) <= tr)
        if len(ii) == 0:
            return np.nan, np.nan

        lb = self.surv_times[ii[0]]

        if ii[-1] == len(self.surv_times) - 1:
            ub = np.inf
        else:
            ub = self.surv_times[ii[-1] + 1]

        return lb, ub


    def summary(self):
        """
        Return a summary of the estimated survival function.

        The summary is a datafram containing the unique event times,
        estimated survival function values, and related quantities.
        """

        df = pd.DataFrame(index=self.surv_times)
        df.index.name = "Time"
        df["Surv prob"] = self.surv_prob
        df["Surv prob SE"] = self.surv_prob_se
        df["num at risk"] = self.n_risk
        df["num events"] = self.n_events

        return df


    def simultaneous_cb(self, alpha=0.05, method="hw", transform="log"):
        """
        Returns a simultaneous confidence band for the survival function.

        Arguments
        ---------
        alpha : float
            `1 - alpha` is the desired simultaneous coverage
            probability for the confidence region.  Currently alpha
            must be set to 0.05, giving 95% simultaneous intervals.
        method : string
            The method used to produce the simultaneous confidence
            band.  Only the Hall-Wellner (hw) method is currently
            implemented.
        transform : string
            The used to produce the interval (note that the returned
            interval is on the survival probability scale regardless
            of which transform is used).  Only `log` and `arcsin` are
            implemented.

        Returns
        -------
        lcb : array-like
            The lower confidence limits corresponding to the points
            in `surv_times`.
        ucb : array-like
            The upper confidence limits corresponding to the points
            in `surv_times`.
        """

        method = method.lower()
        if method != "hw":
            raise ValueError("only the Hall-Wellner (hw) method is implemented")

        if alpha != 0.05:
            raise ValueError("alpha must be set to 0.05")

        transform = transform.lower()
        s2 = self.surv_prob_se**2 / self.surv_prob**2
        nn = self.n_risk
        if transform == "log":
            denom = np.sqrt(nn) * np.log(self.surv_prob)
            theta = 1.3581 * (1 + nn * s2) / denom
            theta = np.exp(theta)
            lcb = self.surv_prob**(1/theta)
            ucb = self.surv_prob**theta
        elif transform == "arcsin":
            k = 1.3581
            k *= (1 + nn * s2) / (2 * np.sqrt(nn))
            k *= np.sqrt(self.surv_prob / (1 - self.surv_prob))
            f = np.arcsin(np.sqrt(self.surv_prob))
            v = np.clip(f - k, 0, np.inf)
            lcb = np.sin(v)**2
            v = np.clip(f + k, -np.inf, np.pi/2)
            ucb = np.sin(v)**2
        else:
            raise ValueError("Unknown transform")

        return lcb, ucb



def survdiff(time, status, group, weight_type=None, strata=None, **kwargs):
    """
    Test for the equality of two survival distributions.

    Parameters:
    -----------
    time : array-like
        The event or censoring times.
    status : array-like
        The censoring status variable, status=1 indicates that the
        event occured, status=0 indicates that the observation was
        censored.
    group : array-like
        Indicators of the two groups
    weight_type : string
        The following weight types are implemented:
            None (default) : logrank test
            fh : Fleming-Harrington, weights by S^(fh_p),
                 requires exponent fh_p to be provided as keyword
                 argument; the weights are derived from S defined at
                 the previous event time, and the first weight is
                 always 1.
            gb : Gehan-Breslow, weights by the number at risk
            tw : Tarone-Ware, weights by the square root of the number
                 at risk
    strata : array-like
        Optional stratum indicators for a stratified test

    Returns:
    --------
    chisq : The chi-square (1 degree of freedom) distributed test
            statistic value
    pvalue : The p-value for the chi^2 test
    """

    # TODO: extend to handle more than two groups

    time = np.asarray(time)
    status = np.asarray(status)
    group = np.asarray(group)

    gr = np.unique(group)
    if len(gr) != 2:
        raise ValueError("logrank only supports two groups")

    if strata is None:
        obs, var = _survdiff(time, status, group, weight_type, gr,
                             **kwargs)
    else:
        strata = np.asarray(strata)
        stu = np.unique(strata)
        obs, var = 0., 0.
        for st in stu:
            # could be more efficient?
            ii = (strata == st)
            obs1, var1 = _survdiff(time[ii], status[ii], group[ii],
                                   weight_type, gr, **kwargs)
            obs += obs1
            var += var1

    zstat = obs / np.sqrt(var)

    # The chi^2 test statistic and p-value.
    chisq = zstat**2
    pvalue = 1 - chi2.cdf(chisq, 1)

    return chisq, pvalue


def _survdiff(time, status, group, weight_type, gr, **kwargs):
    # logrank test for one stratum

    ii = (group == gr[0])
    time1 = time[ii]
    status1 = status[ii]
    ii = (group == gr[1])
    time2 = time[ii]
    status2 = status[ii]

    # Get the unique times.
    utimes = np.unique(time)

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
    for time0 in time1, time2:
        ix = np.searchsorted(utimes, time0)
        n = np.bincount(ix, minlength=len(utimes))
        n = np.cumsum(n)
        n = np.roll(n, 1)
        n[0] = 0
        n = len(time0) - n
        nvec.append(n)
    n1, n2 = tuple(nvec)
    n = n1 + n2

    # The variance of event counts in the first group.
    r = n1 / n.astype(np.float64)
    var = obs * r * (1 - r) * (n - obs) / (n - 1)

    # The expected number of events in the first group.
    exp1 = obs * r

    weights = None
    if weight_type is not None:
        weight_type = weight_type.lower()
        if weight_type == "gb":
            weights = n
        elif weight_type == "tw":
            weights = np.sqrt(n)
        elif weight_type == "fh":
            if "fh_p" not in kwargs:
                raise ValueError("weight_type type 'fh' requires specification of fh_p")
            fh_p = kwargs["fh_p"]
            # Calculate the survivor function directly to avoid the
            # overhead of creating a SurvfuncRight object
            sp = 1 - obs / n.astype(np.float64)
            sp = np.log(sp)
            sp = np.cumsum(sp)
            sp = np.exp(sp)
            weights = sp**fh_p
            weights = np.roll(weights, 1)
            weights[0] = 1
        else:
            raise ValueError("weight_type not implemented")

    # The Z-scale test statistic (compare to normal reference
    # distribution).
    ix = np.flatnonzero(n > 1)
    if weights is None:
        obs = np.sum(obs1[ix] - exp1[ix])
        var = np.sum(var[ix])
    else:
        obs = np.dot(weights[ix], obs1[ix] - exp1[ix])
        var = np.dot(weights[ix]**2, var[ix])

    return obs, var



def plot_survfunc(survfuncs, ax=None):
    """
    Plot one or more survivor functions.

    Arguments
    ---------
    survfuncs : object or array-like
        A single SurvfuncRight object, or a list or SurvfuncRight
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
        assert(type(survfuncs[0]) is SurvfuncRight)
    except:
        survfuncs = [survfuncs]

    for gx, sf in enumerate(survfuncs):

        # The estimated survival function does not include a point at
        # time 0, include it here for plotting.
        surv_times = np.concatenate(([0], sf.surv_times))
        surv_prob = np.concatenate(([1], sf.surv_prob))

        # If the final times are censoring times they are not included
        # in the survival function so we add them here
        mxt = max(sf.time)
        if mxt > surv_times[-1]:
            surv_times = np.concatenate((surv_times, [mxt]))
            surv_prob = np.concatenate((surv_prob, [surv_prob[-1]]))

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
