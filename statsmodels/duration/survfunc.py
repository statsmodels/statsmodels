import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils


def _calc_survfunc_right(time, status, weights=None, entry=None, compress=True,
                         retall=True):
    """
    Calculate the survival function and its standard error for a single
    group.
    """

    # Convert the unique times to ranks (0, 1, 2, ...)
    if entry is None:
        utime, rtime = np.unique(time, return_inverse=True)
    else:
        tx = np.concatenate((time, entry))
        utime, rtime = np.unique(tx, return_inverse=True)
        rtime = rtime[0:len(time)]

    # Number of deaths at each unique time.
    ml = len(utime)
    if weights is None:
        d = np.bincount(rtime, weights=status, minlength=ml)
    else:
        d = np.bincount(rtime, weights=status*weights, minlength=ml)

    # Size of risk set just prior to each event time.
    if weights is None:
        n = np.bincount(rtime, minlength=ml)
    else:
        n = np.bincount(rtime, weights=weights, minlength=ml)
    if entry is not None:
        n = np.cumsum(n) - n
        rentry = np.searchsorted(utime, entry, side='left')
        if weights is None:
            n0 = np.bincount(rentry, minlength=ml)
        else:
            n0 = np.bincount(rentry, weights=weights, minlength=ml)
        n0 = np.cumsum(n0) - n0
        n = n0 - n
    else:
        n = np.cumsum(n[::-1])[::-1]

    # Only retain times where an event occured.
    if compress:
        ii = np.flatnonzero(d > 0)
        d = d[ii]
        n = n[ii]
        utime = utime[ii]

    # The survival function probabilities.
    sp = 1 - d / n.astype(np.float64)
    sp = np.log(sp)
    sp = np.cumsum(sp)
    sp = np.exp(sp)

    if not retall:
        return sp, utime, rtime, n, d

    # Standard errors
    if weights is None:
        # Greenwood's formula
        se = d / (n * (n - d)).astype(np.float64)
        se = np.cumsum(se)
        se = np.sqrt(se)
        se *= sp
    else:
        # Tsiatis' (1981) formula
        se = d / (n * n).astype(np.float64)
        se = np.cumsum(se)
        se = np.sqrt(se)

    return sp, se, utime, rtime, n, d


def _calc_incidence_right(time, status, weights=None):
    """
    Calculate the cumulative incidence function and its standard error.
    """

    # Calculate the all-cause survival function.
    status0 = (status >= 1).astype(np.float64)
    sp, utime, rtime, n, d = _calc_survfunc_right(time, status0, weights,
                                                  compress=False, retall=False)

    ngrp = status.max()

    # Number of cause-specific deaths at each unique time.
    d = []
    for k in range(ngrp):
        status0 = (status == k + 1).astype(np.float64)
        if weights is None:
            d0 = np.bincount(rtime, weights=status0, minlength=len(utime))
        else:
            d0 = np.bincount(rtime, weights=status0*weights,
                             minlength=len(utime))
        d.append(d0)

    # The cumulative incidence function probabilities.
    ip = []
    sp0 = np.r_[1, sp[:-1]] / n
    for k in range(ngrp):
        ip0 = np.cumsum(sp0 * d[k])
        ip.append(ip0)

    # The standard error of the cumulative incidence function.
    if weights is not None:
        return ip, None, utime
    se = []
    da = sum(d)
    for k in range(ngrp):

        ra = da / (n * (n - da))
        v = ip[k]**2 * np.cumsum(ra)
        v -= 2 * ip[k] * np.cumsum(ip[k] * ra)
        v += np.cumsum(ip[k]**2 * ra)

        ra = (n - d[k]) * d[k] / n
        v += np.cumsum(sp0**2 * ra)

        ra = sp0 * d[k] / n
        v -= 2 * ip[k] * np.cumsum(ra)
        v += 2 * np.cumsum(ip[k] * ra)

        se.append(np.sqrt(v))

    return ip, se, utime


def _checkargs(time, status, entry, freq_weights):

    if len(time) != len(status):
        raise ValueError("time and status must have the same length")

    if entry is not None and (len(entry) != len(time)):
        msg = "entry times and event times must have the same length"
        raise ValueError(msg)

    if entry is not None and np.any(entry >= time):
        msg = "Entry times must not occur on or after event times"
        raise ValueError(msg)

    if freq_weights is not None and (len(freq_weights) != len(time)):
        raise ValueError("weights, time and status must have the same length")


class CumIncidenceRight(object):
    """
    Estimation and inference for a cumulative incidence function.

    If J = 1, 2, ... indicates the event type, the cumulative
    incidence function for cause j is:

    I(t, j) = P(T <= t and J=j)

    Only right censoring is supported.  If frequency weights are provided,
    the point estimate is returned without a standard error.

    Parameters
    ----------
    time : array-like
        An array of times (censoring times or event times)
    status : array-like
        If status >= 1 indicates which event occured at time t.  If
        status = 0, the subject was censored at time t.
    title : string
        Optional title used for plots and summary output.
    freq_weights : array-like
        Optional frequency weights

    Attributes
    ----------
    times : array-like
        The distinct times at which the incidence rates are estimated
    cinc : list of arrays
        cinc[k-1] contains the estimated cumulative incidence rates
        for outcome k=1,2,...
    cinc_se : list of arrays
        The standard errors for the values in `cinc`.

    References
    ----------
    The Stata stcompet procedure:
        http://www.stata-journal.com/sjpdf.html?articlenum=st0059

    Dinse, G. E. and M. G. Larson. 1986. A note on semi-Markov models
    for partially censored data. Biometrika 73: 379-386.

    Marubini, E. and M. G. Valsecchi. 1995. Analysing Survival Data
    from Clinical Trials and Observational Studies. Chichester, UK:
    John Wiley & Sons.
    """

    def __init__(self, time, status, title=None, freq_weights=None):

        _checkargs(time, status, None, freq_weights)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)
        x = _calc_incidence_right(time, status, freq_weights)
        self.cinc = x[0]
        self.cinc_se = x[1]
        self.times = x[2]
        self.title = "" if not title else title


class SurvfuncRight(object):
    """
    Estimation and inference for a survival function.

    The survival function S(t) = P(T > t) is the probability that an
    event time T is greater than t.

    This class currently only supports right censoring.

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
    entry : array-like, optional An array of entry times for handling
        left truncation (the subject is not in the risk set on or
        before the entry time)
    title : string
        Optional title used for plots and summary output.
    freq_weights : array-like
        Optional frequency weights

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

    def __init__(self, time, status, entry=None, title=None,
                 freq_weights=None):

        _checkargs(time, status, entry, freq_weights)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)
        if entry is not None:
            entry = self.entry = np.asarray(entry)
        x = _calc_survfunc_right(time, status, weights=freq_weights,
                                 entry=entry)
        self.surv_prob = x[0]
        self.surv_prob_se = x[1]
        self.surv_times = x[2]
        self.n_risk = x[4]
        self.n_events = x[5]
        self.title = "" if not title else title

    def plot(self, ax=None):
        """
        Plot the survival function.

        Examples
        --------
        Change the line color:

        >>> import statsmodels.api as sm
        >>> data = sm.datasets.get_rdataset("flchain", "survival").data
        >>> df = data.loc[data.sex == "F", :]
        >>> sf = sm.SurvfuncRight(df["futime"], df["death"])
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
            g = lambda x: np.log(-np.log(x))
            gprime = lambda x: -1 / (x * np.log(x))
        elif method == "linear":
            g = lambda x: x
            gprime = lambda x: 1
        elif method == "log":
            g = lambda x: np.log(x)
            gprime = lambda x: 1 / x
        elif method == "logit":
            g = lambda x: np.log(x / (1 - x))
            gprime = lambda x: 1 / (x * (1 - x))
        elif method == "asinsqrt":
            g = lambda x: np.arcsin(np.sqrt(x))
            gprime = lambda x: 1 / (2 * np.sqrt(x) * np.sqrt(1 - x))
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

        Parameters
        ----------
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
            msg = "only the Hall-Wellner (hw) method is implemented"
            raise ValueError(msg)

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


def survdiff(time, status, group, weight_type=None, strata=None,
             entry=None, **kwargs):
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
    entry : array-like
        Entry times to handle left truncation. The subject is not in
        the risk set on or before the entry time.

    Returns
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
                             entry, **kwargs)
    else:
        strata = np.asarray(strata)
        stu = np.unique(strata)
        obs, var = 0., 0.
        for st in stu:
            # could be more efficient?
            ii = (strata == st)
            obs1, var1 = _survdiff(time[ii], status[ii], group[ii],
                                   weight_type, gr, entry, **kwargs)
            obs += obs1
            var += var1

    zstat = obs / np.sqrt(var)

    # The chi^2 test statistic and p-value.
    chisq = zstat**2
    pvalue = 1 - chi2.cdf(chisq, 1)

    return chisq, pvalue


def _survdiff(time, status, group, weight_type, gr, entry=None,
              **kwargs):
    # logrank test for one stratum

    # Get the unique times.
    if entry is None:
        utimes, rtimes = np.unique(time, return_inverse=True)
    else:
        utimes, rtimes = np.unique(np.concatenate((time, entry)),
                                   return_inverse=True)
        rtimes = rtimes[0:len(time)]

    # Split entry times by group if present (should use pandas groupby)
    tse = [(gr[0], None), (gr[1], None)]
    if entry is not None:
        for k in 0, 1:
            ii = (group == gr[k])
            entry1 = entry[ii]
            tse[k] = (gr[k], entry1)

    # Event count and risk set size at each time point, per group and overall.
    # TODO: should use Pandas groupby
    nrisk, obsv = [], []
    ml = len(utimes)
    for g, entry0 in tse:

        mk = (group == g)
        n = np.bincount(rtimes, weights=mk, minlength=ml)

        ob = np.bincount(rtimes, weights=status*mk, minlength=ml)
        obsv.append(ob)

        if entry is not None:
            n = np.cumsum(n) - n
            rentry = np.searchsorted(utimes, entry0, side='left')
            n0 = np.bincount(rentry, minlength=ml)
            n0 = np.cumsum(n0) - n0
            nr = n0 - n
        else:
            nr = np.cumsum(n[::-1])[::-1]

        nrisk.append(nr)

    obs = sum(obsv)
    nrisk_tot = sum(nrisk)

    # The variance of event counts in the first group.
    r = nrisk[0] / nrisk_tot
    var = obs * r * (1 - r) * (nrisk_tot - obs) / (nrisk_tot - 1)

    # The expected number of events in the first group.
    exp1 = obs * r

    weights = None
    if weight_type is not None:
        weight_type = weight_type.lower()
        if weight_type == "gb":
            weights = nrisk_tot
        elif weight_type == "tw":
            weights = np.sqrt(nrisk_tot)
        elif weight_type == "fh":
            if "fh_p" not in kwargs:
                msg = "weight_type type 'fh' requires specification of fh_p"
                raise ValueError(msg)
            fh_p = kwargs["fh_p"]
            # Calculate the survivor function directly to avoid the
            # overhead of creating a SurvfuncRight object
            sp = 1 - obs / nrisk_tot.astype(np.float64)
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
    ix = np.flatnonzero(nrisk_tot > 1)
    if weights is None:
        obs = np.sum(obsv[0][ix] - exp1[ix])
        var = np.sum(var[ix])
    else:
        obs = np.dot(weights[ix], obsv[0][ix] - exp1[ix])
        var = np.dot(weights[ix]**2, var[ix])

    return obs, var


def plot_survfunc(survfuncs, ax=None):
    """
    Plot one or more survivor functions.

    Parameters
    ----------
    survfuncs : object or array-like
        A single SurvfuncRight object, or a list or SurvfuncRight
        objects that are plotted together.

    Returns
    -------
    A figure instance on which the plot was drawn.

    Examples
    --------
    Add a legend:

    >>> import statsmodels.api as sm
    >>> from statsmodels.duration.survfunc import plot_survfunc
    >>> data = sm.datasets.get_rdataset("flchain", "survival").data
    >>> df = data.loc[data.sex == "F", :]
    >>> sf0 = sm.SurvfuncRight(df["futime"], df["death"])
    >>> sf1 = sm.SurvfuncRight(3.0 * df["futime"], df["death"])
    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> leg = fig.legend((ha[0], ha[1]), (lb[0], lb[1]), 'center right')

    Change the line colors:

    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> ha[0].set_color('purple')
    >>> ha[1].set_color('orange')
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

        li, = ax.step(surv_times, surv_prob, '-', label=label, lw=2,
                      where='post')

        # Plot the censored points.
        ii = np.flatnonzero(np.logical_not(sf.status))
        ti = sf.time[ii]
        jj = np.searchsorted(surv_times, ti) - 1
        sp = surv_prob[jj]
        ax.plot(ti, sp, '+', ms=12, color=li.get_color(),
                label=label + " points")

    ax.set_ylim(0, 1.01)

    return fig
