'''Test for ratio of Poisson intensities in two independent samples

Author: Josef Perktold
License: BSD-3

'''


import numpy as np
import warnings

from scipy import stats, optimize

from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2

# shorthand
norm = stats.norm


def test_poisson(count, nobs, value, method=None, alternative="two-sided",
                 dispersion=1):
    """ WIP  test for one sample poisson mean or rate

    See Also
    --------
    confint_poisson

    """

    n = nobs  # short hand
    rate = count / n

    if method is None:
        msg = "method needs to be specified, currently no default method"
        raise ValueError(msg)

    dist = "normal"

    if method == "wald":
        std = np.sqrt(dispersion * rate / n)
        statistic = (rate - value) / std

    elif method == "waldccv":
        # WCC in Barker 2002
        # add 0.5 event, not 0.5 event rate as in waldcc
        # std = np.sqrt((rate + 0.5 / n) / n)
        # statistic = (rate + 0.5 / n - value) / std
        std = np.sqrt((rate + 0.5 / n) / n)
        statistic = (rate - value) / std

    elif method == "score":
        std = np.sqrt(dispersion * value / n)
        statistic = (rate - value) / std
        pvalue = stats.norm.sf(statistic)

    elif method.startswith("exact-c") or method.startswith("midp-c"):
        pvalue = 2 * np.minimum(stats.poisson.cdf(count, n * value),
                                stats.poisson.sf(count - 1, n * value))
        statistic = None
        dist = "Poisson"

        if method.startswith("midp-c"):
            pvalue = pvalue - 0.5 * stats.poisson.pmf(count, n * value)

    elif method == "sqrt-a":
        # anscombe, based on Swift 2009 (with transformation to rate)
        std = 0.5
        statistic = (np.sqrt(count + 3 / 8) - np.sqrt(n * value + 3 / 8)) / std

    elif method == "sqrt-v":
        # vandenbroucke, based on Swift 2009 (with transformation to rate)
        std = 0.5
        crit = stats.norm.isf(0.025)
        statistic = (np.sqrt(count + (crit**2 + 2) / 12) -
                     # np.sqrt(n * value + (crit**2 + 2) / 12)) / std
                     np.sqrt(n * value)) / std

    else:
        raise ValueError("unknown method")

    if dispersion != 1 and dist != "normal":
        warnings.warn("Dispersion is ignored with method %s." % method)

    if dist == 'normal':
        statistic, pvalue = _zstat_generic2(statistic, 1, alternative)

    res = HolderTuple(
        statistic=statistic,
        pvalue=np.clip(pvalue, 0, 1),
        distribution=dist,
        method=method,
        alternative=alternative,
        rate=rate,
        nobs=n
        )
    return res


def confint_poisson(count, exposure, method=None, alpha=0.05):
    """Confidence interval for a Poisson mean or rate

    The function is vectorized for all methods except "midp-c", which uses
    an iterative method to invert the hypothesis test function.

    All current methods are central, that is the probability of each tail is
    smaller or equal to alpha / 2. The one-sided interval limits can be
    obtained by doubling alpha.

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
        This will likely change.
    method : str
        Method to use for confidence interval
        This is required, there is currently no default method
    alpha : float in (0, 1)
        Signifivance level, nominal coverage of the confidence interval is
        1 - alpha.

    Returns
    -------
    tuple (low, upp) : confidence limits.

    Notes
    -----
    Methods are mainly based on Barker (2002) [1]_ and Swift (2009) [3]_.

    Available methods are:

    - "exact-c" central confidence interval based on gamma distribution
    - "score" : based on score test, uses variance under null value
    - "wald" : based on wald test, uses variance base on estimated rate.
    - "waldcc" : based on wald test with 0.5 count added to variance
      computation. This does not use continuity correct for the center of the
      confidence interval.
    - "midp-c" : based on midp correction of central exact confidence interval.
      this uses numerical inversion of the test function. not vectorized.
    - "jeffreys" : based on Jeffreys' prior. computed using gamma distribution
    - "sqrt" : based on square root transformed counts
    - "sqrt-a" based on Anscombe square root transformation of counts + 3/8.
    - "sqrt-centcc" will likely be dropped. anscombe with continuity corrected
      center.
      (Similar to R survival cipoisson, but without the 3/8 right shift of
      the confidence interval).

    sqrt-cent is the same as sqrt-a, using a different computation, will be
    deleted.

    sqrt-v is a corrected square root method attributed to vandenbrouke, which
    might also be deleted.

    todo:
    - missing dispersion,
    - maybe split nobs and exposure (? needed in NB). Exposure could be used
      to standardize rate.
    - modified wald, switch method if count=0.

    See Also
    --------
    test_poisson

    References
    ----------
    .. [1] Barker, Lawrence. 2002. “A Comparison of Nine Confidence Intervals
       for a Poisson Parameter When the Expected Number of Events Is ≤ 5.”
       The American Statistician 56 (2): 85–89.
       https://doi.org/10.1198/000313002317572736.
    .. [2] Patil, VV, and HV Kulkarni. 2012. “Comparison of Confidence
       Intervals for the Poisson Mean: Some New Aspects.”
       REVSTAT–Statistical Journal 10(2): 211–27.
    .. [3] Swift, Michael Bruce. 2009. “Comparison of Confidence Intervals for
       a Poisson Mean – Further Considerations.” Communications in Statistics -
       Theory and Methods 38 (5): 748–59.
       https://doi.org/10.1080/03610920802255856.

    """
    n = exposure  # short hand
    rate = count / exposure
    alpha = alpha / 2  # two-sided

    if method is None:
        msg = "method needs to be specified, currently no default method"
        raise ValueError(msg)

    if method == "wald":
        whalf = stats.norm.isf(alpha) * np.sqrt(rate / n)
        ci = (rate - whalf, rate + whalf)

    elif method == "waldccv":
        # based on WCC in Barker 2002
        # add 0.5 event, not 0.5 event rate as in BARKER waldcc
        whalf = stats.norm.isf(alpha) * np.sqrt((rate + 0.5 / n) / n)
        ci = (rate - whalf, rate + whalf)

    elif method == "score":
        crit = stats.norm.isf(alpha)
        center = count + crit**2 / 2
        whalf = crit * np.sqrt((count + crit**2 / 4))
        ci = ((center - whalf) / n, (center + whalf) / n)

    elif method == "midp-c":
        # note local alpha above is for one tail
        ci = _invert_test_confint(count, n, alpha=2 * alpha, method="midp-c",
                                  method_start="exact-c")

    elif method == "sqrt":
        # drop, wrong n
        crit = stats.norm.isf(alpha)
        center = rate + crit**2 / (4 * n)
        whalf = crit * np.sqrt(rate / n)
        ci = (center - whalf, center + whalf)

    elif method == "sqrt-cent":
        crit = stats.norm.isf(alpha)
        center = count + crit**2 / 4
        whalf = crit * np.sqrt((count + 3 / 8))
        ci = ((center - whalf) / n, (center + whalf) / n)

    elif method == "sqrt-centcc":
        # drop with cc, does not match cipoisson in R survival
        crit = stats.norm.isf(alpha)
        # avoid sqrt of negative value if count=0
        center_low = np.sqrt(np.maximum(count + 3 / 8 - 0.5, 0))
        center_upp = np.sqrt(count + 3 / 8 + 0.5)
        whalf = crit / 2
        # above is for ci of count
        ci = (((np.maximum(center_low - whalf, 0))**2 - 3 / 8) / n,
              ((center_upp + whalf)**2 - 3 / 8) / n)

        # crit = stats.norm.isf(alpha)
        # center = count
        # whalf = crit * np.sqrt((count + 3 / 8 + 0.5))
        # ci = ((center - whalf - 0.5) / n, (center + whalf + 0.5) / n)

    elif method == "sqrt-a":
        # anscombe, based on Swift 2009 (with transformation to rate)
        crit = stats.norm.isf(alpha)
        center = np.sqrt(count + 3 / 8)
        whalf = crit / 2
        # above is for ci of count
        ci = (((np.maximum(center - whalf, 0))**2 - 3 / 8) / n,
              ((center + whalf)**2 - 3 / 8) / n)

    elif method == "sqrt-v":
        # vandenbroucke, based on Swift 2009 (with transformation to rate)
        crit = stats.norm.isf(alpha)
        center = np.sqrt(count + (crit**2 + 2) / 12)
        whalf = crit / 2
        # above is for ci of count
        ci = (np.maximum(center - whalf, 0))**2 / n, (center + whalf)**2 / n

    elif method in ["gamma", "exact-c"]:
        # garwood exact, gamma
        low = stats.gamma.ppf(alpha, count) / exposure
        upp = stats.gamma.isf(alpha, count+1) / exposure
        if np.isnan(low).any():
            # case with count = 0
            if np.size(low) == 1:
                low = 0.0
            else:
                low[np.isnan(low)] = 0.0

        ci = (low, upp)

    elif method.startswith("jeff"):
        # jeffreys, gamma
        countc = count + 0.5
        ci = (stats.gamma.ppf(alpha, countc) / exposure,
              stats.gamma.isf(alpha, countc) / exposure)

    else:
        raise ValueError("unknown method %s" % method)

    ci = (np.maximum(ci[0], 0), ci[1])
    return ci


def _invert_test_confint(count, nobs, alpha=0.05, method="midp-c",
                         method_start="exact-c"):

    def func(r):
        v = (test_poisson(count, nobs, value=r, method=method)[1] -
             alpha)**2
        return v

    ci = confint_poisson(count, nobs, method=method_start)
    low = optimize.fmin(func, ci[0], xtol=1e-8,  disp=False)
    upp = optimize.fmin(func, ci[1], xtol=1e-8, disp=False)
    assert np.size(low) == 1
    return low[0], upp[0]


def test_poisson_2indep(count1, exposure1, count2, exposure2, ratio_null=1,
                        method='score', alternative='two-sided',
                        etest_kwds=None):
    '''test for ratio of two sample Poisson intensities

    If the two Poisson rates are g1 and g2, then the Null hypothesis is

    - H0: g1 / g2 = ratio_null

    against one of the following alternatives

    - H1_2-sided: g1 / g2 != ratio_null
    - H1_larger: g1 / g2 > ratio_null
    - H1_smaller: g1 / g2 < ratio_null

    Parameters
    ----------
    count1 : int
        Number of events in first sample.
    exposure1 : float
        Total exposure (time * subjects) in first sample.
    count2 : int
        Number of events in second sample.
    exposure2 : float
        Total exposure (time * subjects) in second sample.
    ratio: float
        ratio of the two Poisson rates under the Null hypothesis. Default is 1.
    method : string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        Current Methods are based on Gu et. al 2008.
        Implemented are 'wald', 'score' and 'sqrt' based asymptotic normal
        distribution, and the exact conditional test 'exact-cond', and its
        mid-point version 'cond-midp'. method='etest' and method='etest-wald'
        provide pvalues from `etest_poisson_2indep` using score or wald
        statistic respectively.
        see Notes.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio of rates is not equal to ratio_null (default)
        - 'larger' :   H1: ratio of rates is larger than ratio_null
        - 'smaller' :  H1: ratio of rates is smaller than ratio_null
    etest_kwds: dictionary
        Additional parameters to be passed to the etest_poisson_2indep
        function, namely y_grid.

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    Notes
    -----
    - 'wald': method W1A, wald test, variance based on separate estimates
    - 'score': method W2A, score test, variance based on estimate under Null
    - 'wald-log': W3A
    - 'score-log' W4A
    - 'sqrt': W5A, based on variance stabilizing square root transformation
    - 'exact-cond': exact conditional test based on binomial distribution
    - 'cond-midp': midpoint-pvalue of exact conditional test
    - 'etest': etest with score test statistic
    - 'etest-wald': etest with wald test statistic

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

    See Also
    --------
    tost_poisson_2indep
    etest_poisson_2indep
    '''

    # shortcut names
    y1, n1, y2, n2 = count1, exposure1, count2, exposure2
    d = n2 / n1
    r = ratio_null
    r_d = r / d

    if method in ['score']:
        stat = (y1 - y2 * r_d) / np.sqrt((y1 + y2) * r_d)
        dist = 'normal'
    elif method in ['wald']:
        stat = (y1 - y2 * r_d) / np.sqrt(y1 + y2 * r_d**2)
        dist = 'normal'
    elif method in ['sqrt']:
        stat = 2 * (np.sqrt(y1 + 3 / 8.) - np.sqrt((y2 + 3 / 8.) * r_d))
        stat /= np.sqrt(1 + r_d)
        dist = 'normal'
    elif method in ['exact-cond', 'cond-midp']:
        from statsmodels.stats import proportion
        bp = r_d / (1 + r_d)
        y_total = y1 + y2
        stat = None
        # TODO: why y2 in here and not y1, check definition of H1 "larger"
        pvalue = proportion.binom_test(y1, y_total, prop=bp,
                                       alternative=alternative)
        if method in ['cond-midp']:
            # not inplace in case we still want binom pvalue
            pvalue = pvalue - 0.5 * stats.binom.pmf(y1, y_total, bp)

        dist = 'binomial'
    elif method.startswith('etest'):
        if method.endswith('wald'):
            method_etest = 'wald'
        else:
            method_etest = 'score'
        if etest_kwds is None:
            etest_kwds = {}

        stat, pvalue = etest_poisson_2indep(
            count1, exposure1, count2, exposure2, ratio_null=ratio_null,
            method=method_etest, alternative=alternative, **etest_kwds)

        dist = 'poisson'
    else:
        raise ValueError('method not recognized')

    if dist == 'normal':
        stat, pvalue = _zstat_generic2(stat, 1, alternative)

    rates = (y1 / n1, y2 / n2)
    ratio = rates[0] / rates[1]
    res = HolderTuple(statistic=stat,
                      pvalue=pvalue,
                      distribution=dist,
                      method=method,
                      alternative=alternative,
                      rates=rates,
                      ratio=ratio,
                      ratio_null=ratio_null)
    return res


def etest_poisson_2indep(count1, exposure1, count2, exposure2, ratio_null=1,
                         method='score', alternative='2-sided', ygrid=None,
                         y_grid=None):
    """E-test for ratio of two sample Poisson rates

    If the two Poisson rates are g1 and g2, then the Null hypothesis is

    - H0: g1 / g2 = ratio_null

    against one of the following alternatives

    - H1_2-sided: g1 / g2 != ratio_null
    - H1_larger: g1 / g2 > ratio_null
    - H1_smaller: g1 / g2 < ratio_null

    Parameters
    ----------
    count1 : int
        Number of events in first sample
    exposure1 : float
        Total exposure (time * subjects) in first sample
    count2 : int
        Number of events in first sample
    exposure2 : float
        Total exposure (time * subjects) in first sample
    ratio : float
        ratio of the two Poisson rates under the Null hypothesis. Default is 1.
    method : {"score", "wald"}
        Method for the test statistic that defines the rejection region.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

           'two-sided': H1: ratio of rates is not equal to ratio_null (default)
           'larger' :   H1: ratio of rates is larger than ratio_null
           'smaller' :  H1: ratio of rates is smaller than ratio_null

    y_grid : None or 1-D ndarray
        Grid values for counts of the Poisson distribution used for computing
        the pvalue. By default truncation is based on an upper tail Poisson
        quantiles.

    ygrid : None or 1-D ndarray
        Same as y_grid. Deprecated. If both y_grid and ygrid are provided,
        ygrid will be ignored.

    Returns
    -------
    stat_sample : float
        test statistic for the sample
    pvalue : float

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

    """
    y1, n1, y2, n2 = count1, exposure1, count2, exposure2
    d = n2 / n1
    r = ratio_null
    r_d = r / d

    eps = 1e-20  # avoid zero division in stat_func

    if method in ['score']:
        def stat_func(x1, x2):
            return (x1 - x2 * r_d) / np.sqrt((x1 + x2) * r_d + eps)
        # TODO: do I need these? return_results ?
        # rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
        # rate1_cmle = rate2_cmle * r
        # rate1 = rate1_cmle
        # rate2 = rate2_cmle
    elif method in ['wald']:
        def stat_func(x1, x2):
            return (x1 - x2 * r_d) / np.sqrt(x1 + x2 * r_d**2 + eps)
        # rate2_mle = y2 / n2
        # rate1_mle = y1 / n1
        # rate1 = rate1_mle
        # rate2 = rate2_mle
    else:
        raise ValueError('method not recognized')

    # The sampling distribution needs to be based on the null hypotheis
    # use constrained MLE from 'score' calculation
    rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
    rate1_cmle = rate2_cmle * r
    rate1 = rate1_cmle
    rate2 = rate2_cmle
    mean1 = n1 * rate1
    mean2 = n2 * rate2

    stat_sample = stat_func(y1, y2)

    if ygrid is not None:
        warnings.warn("ygrid is deprecated, use y_grid", DeprecationWarning)
    y_grid = y_grid if y_grid is not None else ygrid

    # The following uses a fixed truncation for evaluating the probabilities
    # It will currently only work for small counts, so that sf at truncation
    # point is small
    # We can make it depend on the amount of truncated sf.
    # Some numerical optimization or checks for large means need to be added.
    if y_grid is None:
        threshold = stats.poisson.isf(1e-13, max(mean1, mean2))
        threshold = max(threshold, 100)   # keep at least 100
        y_grid = np.arange(threshold + 1)
    else:
        y_grid = np.asarray(y_grid)
        if y_grid.ndim != 1:
            raise ValueError("y_grid needs to be None or 1-dimensional array")
    pdf1 = stats.poisson.pmf(y_grid, mean1)
    pdf2 = stats.poisson.pmf(y_grid, mean2)

    stat_space = stat_func(y_grid[:, None], y_grid[None, :])  # broadcasting
    eps = 1e-15   # correction for strict inequality check

    if alternative in ['two-sided', '2-sided', '2s']:
        mask = np.abs(stat_space) >= np.abs(stat_sample) - eps
    elif alternative in ['larger', 'l']:
        mask = stat_space >= stat_sample - eps
    elif alternative in ['smaller', 's']:
        mask = stat_space <= stat_sample + eps
    else:
        raise ValueError('invalid alternative')

    pvalue = ((pdf1[:, None] * pdf2[None, :])[mask]).sum()
    return stat_sample, pvalue


def tost_poisson_2indep(count1, exposure1, count2, exposure2, low, upp,
                        method='score'):
    '''Equivalence test based on two one-sided `test_proportions_2indep`

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis for equivalence testing are

    - H0: g1 / g2 <= low or upp <= g1 / g2
    - H1: low < g1 / g2 < upp

    where g1 and g2 are the Poisson rates.

    Parameters
    ----------
    count1 : int
        Number of events in first sample
    exposure1 : float
        Total exposure (time * subjects) in first sample
    count2 : int
        Number of events in second sample
    exposure2 : float
        Total exposure (time * subjects) in second sample
    low, upp :
        equivalence margin for the ratio of Poisson rates
    method: string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        Current Methods are based on Gu et. al 2008
        Implemented are 'wald', 'score' and 'sqrt' based asymptotic normal
        distribution, and the exact conditional test 'exact-cond', and its
        mid-point version 'cond-midp', see Notes

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    Notes
    -----
    - 'wald': method W1A, wald test, variance based on separate estimates
    - 'score': method W2A, score test, variance based on estimate under Null
    - 'wald-log': W3A  not implemented
    - 'score-log' W4A  not implemented
    - 'sqrt': W5A, based on variance stabilizing square root transformation
    - 'exact-cond': exact conditional test based on binomial distribution
    - 'cond-midp': midpoint-pvalue of exact conditional test

    The latter two are only verified for one-sided example.

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

    See Also
    --------
    test_poisson_2indep
    '''

    tt1 = test_poisson_2indep(count1, exposure1, count2, exposure2,
                              ratio_null=low, method=method,
                              alternative='larger')
    tt2 = test_poisson_2indep(count1, exposure1, count2, exposure2,
                              ratio_null=upp, method=method,
                              alternative='smaller')

    idx_max = 0 if tt1.pvalue < tt2.pvalue else 1
    res = HolderTuple(statistic=[tt1.statistic, tt2.statistic][idx_max],
                      pvalue=[tt1.pvalue, tt2.pvalue][idx_max],
                      method=method,
                      results_larger=tt1,
                      results_smaller=tt2,
                      title="Equivalence test for 2 independent Poisson rates"
                      )

    return res


def power_poisson_2indep(rate1, nobs1, rate2, nobs2, exposure, value=0,
                         alpha=0.025, dispersion=1, alternative="smaller"):
    """power for one-sided test of ratio of 2 independent poisson rates
    this is currently for superiority and non-inferiority testing

    signature not adjusted yet

    incomplete alternatives, var options missing
    """
    low = value  # alias from original equivalence test case
    nobs_ratio = nobs1 / nobs2
    v1 = dispersion / exposure * (1 / rate2 + 1 / (nobs_ratio * rate1))
    v0_low = v0_upp = v1

    crit = norm.isf(alpha)
    if alternative == "smaller":
        pow_ = norm.cdf((np.sqrt(nobs2) * (np.log(low) - np.log(rate1 / rate2))
                        - crit * np.sqrt(v0_low)) / np.sqrt(v1))
    elif alternative == "larger":
        pow_ = norm.sf((np.sqrt(nobs2) * (np.log(low) - np.log(rate1 / rate2))
                       + crit * np.sqrt(v0_low)) / np.sqrt(v1))
    return pow_


def power_equivalence_poisson_2indep(rate1, nobs1, rate2, nobs2, exposure,
                                     low, upp, alpha=0.025, dispersion=1):
    """power for equivalence test of ratio of 2 independent poisson rates

    WIP missing var option, redundant/unused nobs1
    """
    nobs_ratio = nobs1 / nobs2
    v1 = dispersion / exposure * (1 / rate2 + 1 / (nobs_ratio * rate1))
    v0_low = v0_upp = v1

    crit = norm.isf(alpha)
    pow_ = (
        norm.cdf((np.sqrt(nobs2) * (np.log(rate1 / rate2) - np.log(low))
                  - crit * np.sqrt(v0_low)) / np.sqrt(v1)) +
        norm.cdf((np.sqrt(nobs2) * (np.log(upp) - np.log(rate1 / rate2))
                  - crit * np.sqrt(v0_upp)) / np.sqrt(v1)) - 1
        )
    return pow_


def _std_2poisson_power(diff, rate2, nobs_ratio=1, alpha=0.05,
                        exposure=1, dispersion=1,
                        value=0):
    rate1 = rate2 + diff
    # v1 = dispersion / exposure * (1 / rate2 + 1 / (nobs_ratio * rate1))
    v1 = rate1 + nobs_ratio * rate2
    return None, np.sqrt(v1), np.sqrt(v1)


def power_ppoisson_diff_2indep(diff, rate2, nobs1, nobs_ratio=1, alpha=0.05,
                               value=0, alternative='two-sided',
                               return_results=True):
    """power for ztest that two independent poisson rates are equal

    Warning preliminary
    currently wald test type to replicate PASS chapter 436
    analogy to proportion test (copy paste docstring)
    TODO: nobs1 or nobs2 in signature

    This assumes that the variance is based on the ?????
    under the null and the non-pooled variance under the alternative

    Parameters
    ----------
    diff : float
        difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computing using p2 and diff
        p1 = p2 + diff
    nobs1 : float or int
        number of observations in sample 1
    ratio : float
        sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        currently only `value=0`, i.e. equality testing, is supported
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only the power is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        p_pooled
            pooled proportion, used for std_null
        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))
    """
    # TODO: avoid possible circular import, check if needed
    from statsmodels.stats.power import normal_power_het

    p_pooled, std_null, std_alt = _std_2poisson_power(diff, rate2,
                                                      nobs_ratio=nobs_ratio,
                                                      alpha=alpha, value=value)

    pow_ = normal_power_het(diff, nobs1, 0.05, std_null=std_null,
                            std_alternative=std_alt,
                            alternative=alternative)

    if return_results:
        res = HolderTuple(
            power=pow_,
            p_pooled=p_pooled,
            std_null=std_null,
            std_alt=std_alt,
            nobs1=nobs1,
            nobs2=nobs_ratio * nobs1,
            nobs_ratio=nobs_ratio,
            alpha=alpha,
            )
        return res
    else:
        return pow_
