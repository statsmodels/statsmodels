# -*- coding: utf-8 -*-
"""Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""
from statsmodels.compat.python import lzip, range
import numpy as np
from scipy import stats, optimize

from statsmodels.stats.base import AllPairsResults
#import statsmodels.stats.multitest as smt

def proportion_confint(count, nobs, alpha=0.05, method='normal'):
    '''confidence interval for a binomial proportion

    Parameters
    ----------
    count : int or array
        number of successes
    nobs : int
        total number of trials
    alpha : float in (0, 1)
        significance level, default 0.05
    method : string in ['normal']
        method to use for confidence interval,
        currently available methods :

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffrey` : Jeffrey's Bayesian Interval
         - `binom_test` : experimental, inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : float
        lower and upper confidence level with coverage (approximately) 1-alpha.
        Note: Beta has coverage
        coverage is only 1-alpha on average for some other methods.)

    Notes
    -----
    Beta, the Clopper-Pearson interval has coverage at least 1-alpha, but is
    in general conservative. Most of the other methods have average coverage
    equal to 1-alpha, but will have smaller coverage in some cases.

    Method "binom_test" directly inverts the binomial test in scipy.stats.
    which has discrete steps.

    TODO: binom_test intervals raise an exception in small samples if one
       interval bound is close to zero or one.

    References
    ----------
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001). "Interval
        Estimation for a Binomial Proportion",
        Statistical Science 16 (2): 101–133. doi:10.1214/ss/1009213286.
        TODO: Is this the correct one ?

    '''

    q_ = count * 1. / nobs
    alpha_2 = 0.5 * alpha

    if method == 'normal':
        std_ = np.sqrt(q_ * (1 - q_) / nobs)
        dist = stats.norm.isf(alpha / 2.) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist

    elif method == 'binom_test':
        # inverting the binomial test
        def func(qi):
            #return stats.binom_test(qi * nobs, nobs, p=q_) - alpha #/ 2.
            return stats.binom_test(q_ * nobs, nobs, p=qi) - alpha
        # Note: only approximate, step function at integer values of count
        # possible problems if bounds are too narrow
        # problem if we hit 0 or 1
        #    brentq fails ValueError: f(a) and f(b) must have different signs
        ci_low = optimize.brentq(func, q_ * 0.1, q_)
        #ci_low = stats.binom_test(qi_low * nobs, nobs, p=q_)
        #ci_low = np.floor(qi_low * nobs) / nobs
        ub = np.minimum(q_ + 2 * (q_ - ci_low), 1)
        ci_upp = optimize.brentq(func, q_, ub)
        #ci_upp = stats.binom_test(qi_upp * nobs, nobs, p=q_)
        #ci_upp = np.ceil(qi_upp * nobs) / nobs
        # TODO: check if we should round up or down, or interpolate

    elif method == 'beta':
        ci_low = stats.beta.ppf(alpha_2 , count, nobs - count + 1)
        ci_upp = stats.beta.isf(alpha_2, count + 1, nobs - count)

    elif method == 'agresti_coull':
        crit = stats.norm.isf(alpha / 2.)
        nobs_c = nobs + crit**2
        q_c = (count + crit**2 / 2.) / nobs_c
        std_c = np.sqrt(q_c * (1. - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist

    elif method == 'wilson':
        crit = stats.norm.isf(alpha / 2.)
        crit2 = crit**2
        denom = 1 + crit2 / nobs
        center = (q_ + crit2 / (2 * nobs)) / denom
        dist = crit * np.sqrt(q_ * (1. - q_) / nobs + crit2 / (4. * nobs**2))
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist

    elif method == 'jeffrey':
        ci_low, ci_upp = stats.beta.interval(1 - alpha,  count + 0.5,
                                             nobs - count + 0.5)

    else:
        raise NotImplementedError('method "%s" is not available' % method)
    return ci_low, ci_upp

def samplesize_confint_proportion(proportion, half_length, alpha=0.05,
                                  method='normal'):
    '''find sample size to get desired confidence interval length

    Parameters
    ----------
    proportion : float in (0, 1)
        proportion or quantile
    half_length : float in (0, 1)
        desired half length of the confidence interval
    alpha : float in (0, 1)
        significance level, default 0.05,
        coverage of the two-sided interval is (approximately) ``1 - alpha``
    method : string in ['normal']
        method to use for confidence interval,
        currently only normal approximation

    Returns
    -------
    n : float
        sample size to get the desired half length of the confidence interval

    Notes
    -----
    this is mainly to store the formula.
    possible application: number of replications in bootstrap samples

    '''
    q_ = proportion
    if method == 'normal':
        n = q_ * (1 - q_) / (half_length / stats.norm.isf(alpha / 2.))**2
    else:
        raise NotImplementedError('only "normal" is available')

    return n

def proportion_effectsize(prop1, prop2, method='normal'):
    '''effect size for a test comparing two proportions

    for use in power function

    Parameters
    ----------
    prop1, prop2: float or array_like

    Returns
    -------
    es : float or ndarray
        effect size for (transformed) prop1 - prop2

    Notes
    -----
    only method='normal' is implemented to match pwr.p2.test
    see http://www.statmethods.net/stats/power.html

    Effect size for `normal` is defined as ::

        2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))

    I think other conversions to normality can be used, but I need to check.

    Examples
    --------
    >>> smpr.proportion_effectsize(0.5, 0.4)
    0.20135792079033088
    >>> smpr.proportion_effectsize([0.3, 0.4, 0.5], 0.4)
    array([-0.21015893,  0.        ,  0.20135792])

    '''
    if method != 'normal':
        raise ValueError('only "normal" is implemented')

    es = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    return es

def std_prop(prop, nobs):
    '''standard error for the estimate of a proportion

    This is just ``np.sqrt(p * (1. - p) / nobs)``

    Parameters
    ----------
    prop : array_like
        proportion
    nobs : int, array_like
        number of observations

    Returns
    -------
    std : array_like
        standard error for a proportion of nobs independent observations
    '''
    return np.sqrt(prop * (1. - prop) / nobs)

def _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt,
                 alpha=0.05, discrete=True, dist='norm', nobs=None,
                 continuity=0, critval_continuity=0):
    '''Generic statistical power function for normal based equivalence test

    This includes options to adjust the normal approximation and can use
    the binomial to evaluate the probability of the rejection region

    see power_ztost_prob for a description of the options
    '''
    # TODO: refactor structure, separate norm and binom better
    if not isinstance(continuity, tuple):
        continuity = (continuity, continuity)
    crit = stats.norm.isf(alpha)
    k_low = mean_low + np.sqrt(var_low) * crit
    k_upp = mean_upp - np.sqrt(var_upp) * crit
    if discrete or dist == 'binom':
        k_low = np.ceil(k_low * nobs + 0.5 * critval_continuity)
        k_upp = np.trunc(k_upp * nobs - 0.5 * critval_continuity)
        if dist == 'norm':
            #need proportion
            k_low = (k_low) * 1. / nobs #-1 to match PASS
            k_upp = k_upp * 1. / nobs
#    else:
#        if dist == 'binom':
#            #need counts
#            k_low *= nobs
#            k_upp *= nobs
    #print mean_low, np.sqrt(var_low), crit, var_low
    #print mean_upp, np.sqrt(var_upp), crit, var_upp
    if np.any(k_low > k_upp):   #vectorize
        import warnings
        warnings.warn("no overlap, power is zero", UserWarning)
    std_alt = np.sqrt(var_alt)
    z_low = (k_low - mean_alt - continuity[0] * 0.5 / nobs) / std_alt
    z_upp = (k_upp - mean_alt + continuity[1] * 0.5 / nobs) / std_alt
    if dist == 'norm':
        power = stats.norm.cdf(z_upp) - stats.norm.cdf(z_low)
    elif dist == 'binom':
        power = (stats.binom.cdf(k_upp, nobs, mean_alt) -
                     stats.binom.cdf(k_low-1, nobs, mean_alt))
    return power, (k_low, k_upp, z_low, z_upp)


def binom_tost(count, nobs, low, upp):
    '''exact TOST test for one proportion using binomial distribution

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials.
    nobs : integer
        the number of trials or observations.
    low, upp : floats
        lower and upper limit of equivalence region

    Returns
    -------
    pvalue : float
        p-value of equivalence test
    pval_low, pval_upp : floats
        p-values of lower and upper one-sided tests

    '''
    # binom_test_stat only returns pval
    tt1 = binom_test(count, nobs, alternative='larger', prop=low)
    tt2 = binom_test(count, nobs, alternative='smaller', prop=upp)
    return np.maximum(tt1, tt2), tt1, tt2,


def binom_tost_reject_interval(low, upp, nobs, alpha=0.05):
    '''rejection region for binomial TOST

    The interval includes the end points,
    `reject` if and only if `r_low <= x <= r_upp`.

    The interval might be empty with `r_upp < r_low`.

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : integer
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : float
        lower and upper bound of rejection region

    '''
    x_low = stats.binom.isf(alpha, nobs, low) + 1
    x_upp = stats.binom.ppf(alpha, nobs, upp) - 1
    return x_low, x_upp

def binom_test_reject_interval(value, nobs, alpha=0.05, alternative='two-sided'):
    '''rejection region for binomial test for one sample proportion

    The interval includes the end points of the rejection region.

    Parameters
    ----------
    value : float
        proportion under the Null hypothesis
    nobs : integer
        the number of trials or observations.


    Returns
    -------
    x_low, x_upp : float
        lower and upper bound of rejection region


    '''
    if alternative in ['2s', 'two-sided']:
        alternative = '2s'  # normalize alternative name
        alpha = alpha / 2

    if alternative in ['2s', 'smaller']:
        x_low = stats.binom.ppf(alpha, nobs, value) - 1
    else:
        x_low = 0
    if alternative in ['2s', 'larger']:
        x_upp = stats.binom.isf(alpha, nobs, value) + 1
    else :
        x_upp = nobs

    return x_low, x_upp

def binom_test(count, nobs, prop=0.5, alternative='two-sided'):
    '''Perform a test that the probability of success is p.

    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials.
    nobs : integer
        the number of trials or observations.
    prop : float, optional
        The probability of success under the null hypothesis,
        `0 <= prop <= 1`. The default value is `prop = 0.5`
    alternative : string in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    p-value : float
        The p-value of the hypothesis test

    Notes
    -----
    This uses scipy.stats.binom_test for the two-sided alternative.

    '''

    if np.any(prop > 1.0) or np.any(prop < 0.0):
        raise ValueError("p must be in range [0,1]")
    if alternative in ['2s', 'two-sided']:
        pval = stats.binom_test(count, n=nobs, p=prop)
    elif alternative in ['l', 'larger']:
        pval = stats.binom.sf(count-1, nobs, prop)
    elif alternative in ['s', 'smaller']:
        pval = stats.binom.cdf(count, nobs, prop)
    else:
        raise ValueError('alternative not recognized\n'
                         'should be two-sided, larger or smaller')
    return pval


def power_binom_tost(low, upp, nobs, p_alt=None, alpha=0.05):
    if p_alt is None:
        p_alt = 0.5 * (low + upp)
    x_low, x_upp = binom_tost_reject_interval(low, upp, nobs, alpha=alpha)
    power = (stats.binom.cdf(x_upp, nobs, p_alt) -
                     stats.binom.cdf(x_low-1, nobs, p_alt))
    return power

def power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm',
                     variance_prop=None, discrete=True, continuity=0,
                     critval_continuity=0):
    '''Power of proportions equivalence test based on normal distribution

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        number of observations
    p_alt : float in (0,1)
        proportion under the alternative
    alpha : float in (0,1)
        significance level of the test
    dist : string in ['norm', 'binom']
        This defines the distribution to evalute the power of the test. The
        critical values of the TOST test are always based on the normal
        approximation, but the distribution for the power can be either the
        normal (default) or the binomial (exact) distribution.
    variance_prop : None or float in (0,1)
        If this is None, then the variances for the two one sided tests are
        based on the proportions equal to the equivalence limits.
        If variance_prop is given, then it is used to calculate the variance
        for the TOST statistics. If this is based on an sample, then the
        estimated proportion can be used.
    discrete : bool
        If true, then the critical values of the rejection region are converted
        to integers. If dist is "binom", this is automatically assumed.
        If discrete is false, then the TOST critical values are used as
        floating point numbers, and the power is calculated based on the
        rejection region that is not discretized.
    continuity : bool or float
        adjust the rejection region for the normal power probability. This has
        and effect only if ``dist='norm'``
    critval_continuity : bool or float
        If this is non-zero, then the critical values of the tost rejection
        region are adjusted before converting to integers. This affects both
        distributions, ``dist='norm'`` and ``dist='binom'``.

    Returns
    -------
    power : float
        statistical power of the equivalence test.
    (k_low, k_upp, z_low, z_upp) : tuple of floats
        critical limits in intermediate steps
        temporary return, will be changed

    Notes
    -----
    In small samples the power for the ``discrete`` version, has a sawtooth
    pattern as a function of the number of observations. As a consequence,
    small changes in the number of observations or in the normal approximation
    can have a large effect on the power.

    ``continuity`` and ``critval_continuity`` are added to match some results
    of PASS, and are mainly to investigate the sensitivity of the ztost power
    to small changes in the rejection region. From my interpretation of the
    equations in the SAS manual, both are zero in SAS.

    works vectorized

    **verification:**

    The ``dist='binom'`` results match PASS,
    The ``dist='norm'`` results look reasonable, but no benchmark is available.

    References
    ----------
    SAS Manual: Chapter 68: The Power Procedure, Computational Resources
    PASS Chapter 110: Equivalence Tests for One Proportion.

    '''
    mean_low = low
    var_low = std_prop(low, nobs)**2
    mean_upp = upp
    var_upp = std_prop(upp, nobs)**2
    mean_alt = p_alt
    var_alt = std_prop(p_alt, nobs)**2
    if variance_prop is not None:
        var_low = var_upp = std_prop(variance_prop, nobs)**2
    power = _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt,
                 alpha=alpha, discrete=discrete, dist=dist, nobs=nobs,
                 continuity=continuity, critval_continuity=critval_continuity)
    return np.maximum(power[0], 0), power[1:]


def _table_proportion(count, nobs):
    '''create a k by 2 contingency table for proportion

    helper function for proportions_chisquare

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials.
    nobs : integer
        the number of trials or observations.

    Returns
    -------
    table : ndarray
        (k, 2) contingency table

    Notes
    -----
    recent scipy has more elaborate contingency table functions

    '''
    table = np.column_stack((count, nobs - count))
    expected = table.sum(0) * table.sum(1)[:,None] * 1. / table.sum()
    n_rows = table.shape[0]
    return table, expected, n_rows


def proportions_ztest(count, nobs, value=None, alternative='two-sided',
                      prop_var=False):
    '''test for proportions based on normal (z) test

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : integer
        the number of trials or observations, with the same length as
        count.
    value : None or float or array_like
        This is the value of the null hypothesis equal to the proportion in the
        case of a one sample test. In the case of a two-sample test, the
        null hypothesis is that prop[0] - prop[1] = value, where prop is the
        proportion in the two samples
    alternative : string in ['two-sided', 'smaller', 'larger']
        The alternative hypothesis can be either two-sided or one of the one-
        sided tests, smaller means that the alternative hypothesis is
        ``prop < value` and larger means ``prop > value``, or the corresponding
        inequality for the two sample test.
    prop_var : False or float in (0, 1)
        If prop_var is false, then the variance of the proportion estimate is
        calculated based on the sample proportion. Alternatively, a proportion
        can be specified to calculate this variance. Common use case is to
        use the proportion under the Null hypothesis to specify the variance
        of the proportion estimate.
        TODO: change options similar to propotion_ztost ?

    Returns
    -------
    zstat : float
        test statistic for the z-test
    p-value : float
        p-value for the z-test


    Notes
    -----
    This uses a simple normal test for proportions. It should be the same as
    running the mean z-test on the data encoded 1 for event and 0 for no event,
    so that the sum corresponds to count.

    In the one and two sample cases with two-sided alternative, this test
    produces the same p-value as ``proportions_chisquare``, since the
    chisquare is the distribution of the square of a standard normal
    distribution.
    (TODO: verify that this really holds)

    TODO: add continuity correction or other improvements for small samples.

    '''
    prop = count * 1. / nobs
    k_sample = np.size(prop)
    if k_sample == 1:
        diff = prop - value
    elif k_sample == 2:
        diff = prop[0] - prop[1] - value
    else:
        msg = 'more than two samples are not implemented yet'
        raise NotImplementedError(msg)

    p_pooled = np.sum(count) * 1. / np.sum(nobs)

    nobs_fact = np.sum(1. / nobs)
    if prop_var:
        p_pooled = prop_var
    var_ = p_pooled * (1 - p_pooled) * nobs_fact
    std_diff = np.sqrt(var_)
    from statsmodels.stats.weightstats import _zstat_generic2
    return _zstat_generic2(diff, std_diff, alternative)

def proportions_ztost(count, nobs, low, upp, prop_var='sample'):
    '''Equivalence test based on normal distribution

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : integer
        the number of trials or observations, with the same length as
        count.
    low, upp : float
        equivalence interval low < prop1 - prop2 < upp
    prop_var : string or float in (0, 1)
        prop_var determines which proportion is used for the calculation
        of the standard deviation of the proportion estimate
        The available options for string are 'sample' (default), 'null' and
        'limits'. If prop_var is a float, then it is used directly.

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    checked only for 1 sample case

    '''
    if prop_var == 'limits':
        prop_var_low = low
        prop_var_upp = upp
    elif prop_var == 'sample':
        prop_var_low = prop_var_upp = False  #ztest uses sample
    elif prop_var == 'null':
        prop_var_low = prop_var_upp = 0.5 * (low + upp)
    elif np.isreal(prop_var):
        prop_var_low = prop_var_upp = prop_var

    tt1 = proportions_ztest(count, nobs, alternative='larger',
                            prop_var=prop_var_low, value=low)
    tt2 = proportions_ztest(count, nobs, alternative='smaller',
                            prop_var=prop_var_upp, value=upp)
    return np.maximum(tt1[1], tt2[1]), tt1, tt2,

def proportions_chisquare(count, nobs, value=None):
    '''test for proportions based on chisquare test

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : integer
        the number of trials or observations, with the same length as
        count.
    value : None or float or array_like

    Returns
    -------
    chi2stat : float
        test statistic for the chisquare test
    p-value : float
        p-value for the chisquare test
    (table, expected)
        table is a (k, 2) contingency table, ``expected`` is the corresponding
        table of counts that are expected under independence with given
        margins


    Notes
    -----
    Recent version of scipy.stats have a chisquare test for independence in
    contingency tables.

    This function provides a similar interface to chisquare tests as
    ``prop.test`` in R, however without the option for Yates continuity
    correction.

    count can be the count for the number of events for a single proportion,
    or the counts for several independent proportions. If value is given, then
    all proportions are jointly tested against this value. If value is not
    given and count and nobs are not scalar, then the null hypothesis is
    that all samples have the same proportion.

    '''
    nobs = np.atleast_1d(nobs)
    table, expected, n_rows = _table_proportion(count, nobs)
    if value is not None:
        expected = np.column_stack((nobs * value, nobs * (1 - value)))
        ddof = n_rows - 1
    else:
        ddof = n_rows

    #print table, expected
    chi2stat, pval = stats.chisquare(table.ravel(), expected.ravel(),
                                     ddof=ddof)
    return chi2stat, pval, (table, expected)




def proportions_chisquare_allpairs(count, nobs, multitest_method='hs'):
    '''chisquare test of proportions for all pairs of k samples

    Performs a chisquare test for proportions for all pairwise comparisons.
    The alternative is two-sided

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials.
    nobs : integer
        the number of trials or observations.
    prop : float, optional
        The probability of success under the null hypothesis,
        `0 <= prop <= 1`. The default value is `prop = 0.5`
    multitest_method : string
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.

    Notes
    -----
    Yates continuity correction is not available.
    '''
    #all_pairs = lmap(list, lzip(*np.triu_indices(4, 1)))
    all_pairs = lzip(*np.triu_indices(len(count), 1))
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)])[1]
               for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)

def proportions_chisquare_pairscontrol(count, nobs, value=None,
                               multitest_method='hs', alternative='two-sided'):
    '''chisquare test of proportions for pairs of k samples compared to control

    Performs a chisquare test for proportions for pairwise comparisons with a
    control (Dunnet's test). The control is assumed to be the first element
    of ``count`` and ``nobs``. The alternative is two-sided, larger or
    smaller.

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials.
    nobs : integer
        the number of trials or observations.
    prop : float, optional
        The probability of success under the null hypothesis,
        `0 <= prop <= 1`. The default value is `prop = 0.5`
    multitest_method : string
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.
    alternative : string in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.


    Notes
    -----
    Yates continuity correction is not available.

    ``value`` and ``alternative`` options are not yet implemented.

    '''
    if (value is not None) or (not alternative in ['two-sided', '2s']):
        raise NotImplementedError
    #all_pairs = lmap(list, lzip(*np.triu_indices(4, 1)))
    all_pairs = [(0, k) for k in range(1, len(count))]
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)],
                                   #alternative=alternative)[1]
                                   )[1]
               for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)
