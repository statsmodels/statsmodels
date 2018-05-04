# -*- coding: utf-8 -*-
"""Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""
from statsmodels.compat.python import lzip, range
import numpy as np
from scipy import stats, optimize
from sys import float_info

from statsmodels.stats.base import AllPairsResults
from statsmodels.tools.sm_exceptions import HypothesisTestWarning


def proportion_confint(count, nobs, alpha=0.05, method='normal'):
    '''confidence interval for a binomial proportion

    Parameters
    ----------
    count : int or array_array_like
        number of successes, can be pandas Series or DataFrame
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
         - `jeffreys` : Jeffreys Bayesian Interval
         - `binom_test` : experimental, inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : float, ndarray, or pandas Series or DataFrame
        lower and upper confidence level with coverage (approximately) 1-alpha.
        When a pandas object is returned, then the index is taken from the
        `count`.

    Notes
    -----
    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,
    but is in general conservative. Most of the other methods have average
    coverage equal to 1-alpha, but will have smaller coverage in some cases.

    The 'beta' and 'jeffreys' interval are central, they use alpha/2 in each
    tail, and alpha is not adjusted at the boundaries. In the extreme case
    when `count` is zero or equal to `nobs`, then the coverage will be only
    1 - alpha/2 in the case of 'beta'.

    The confidence intervals are clipped to be in the [0, 1] interval in the
    case of 'normal' and 'agresti_coull'.

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

    pd_index = getattr(count, 'index', None)
    if pd_index is not None and hasattr(pd_index, '__call__'):
        # this rules out lists, lists have an index method
        pd_index = None
    count = np.asarray(count)
    nobs = np.asarray(nobs)

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
            return stats.binom_test(q_ * nobs, nobs, p=qi) - alpha
        if count == 0:
            ci_low = 0
        else:
            ci_low = optimize.brentq(func, float_info.min, q_)
        if count == nobs:
            ci_upp = 1
        else:
            ci_upp = optimize.brentq(func, q_, 1. - float_info.epsilon)

    elif method == 'beta':
        ci_low = stats.beta.ppf(alpha_2, count, nobs - count + 1)
        ci_upp = stats.beta.isf(alpha_2, count + 1, nobs - count)

        if np.ndim(ci_low) > 0:
            ci_low[q_ == 0] = 0
            ci_upp[q_ == 1] = 1
        else:
            ci_low = ci_low if (q_ != 0) else 0
            ci_upp = ci_upp if (q_ != 1) else 1

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

    # method adjusted to be more forgiving of misspellings or incorrect option name
    elif method[:4] == 'jeff':
        ci_low, ci_upp = stats.beta.interval(1 - alpha, count + 0.5,
                                             nobs - count + 0.5)

    else:
        raise NotImplementedError('method "%s" is not available' % method)

    if method in ['normal', 'agresti_coull']:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if pd_index is not None and np.ndim(ci_low) > 0:
        import pandas as pd
        if np.ndim(ci_low) == 1:
            ci_low = pd.Series(ci_low, index=pd_index)
            ci_upp = pd.Series(ci_upp, index=pd_index)
        if np.ndim(ci_low) == 2:
            ci_low = pd.DataFrame(ci_low, index=pd_index)
            ci_upp = pd.DataFrame(ci_upp, index=pd_index)

    return ci_low, ci_upp


def multinomial_proportions_confint(counts, alpha=0.05, method='goodman'):
    '''Confidence intervals for multinomial proportions.

    Parameters
    ----------
    counts : array_like of int, 1-D
        Number of observations in each category.
    alpha : float in (0, 1), optional
        Significance level, defaults to 0.05.
    method : {'goodman', 'sison-glaz'}, optional
        Method to use to compute the confidence intervals; available methods
        are:

         - `goodman`: based on a chi-squared approximation, valid if all
           values in `counts` are greater or equal to 5 [2]_
         - `sison-glaz`: less conservative than `goodman`, but only valid if
           `counts` has 7 or more categories (``len(counts) >= 7``) [3]_

    Returns
    -------
    confint : ndarray, 2-D
        Array of [lower, upper] confidence levels for each category, such that
        overall coverage is (approximately) `1-alpha`.

    Raises
    ------
    ValueError
        If `alpha` is not in `(0, 1)` (bounds excluded), or if the values in
        `counts` are not all positive or null.
    NotImplementedError
        If `method` is not kown.
    Exception
        When ``method == 'sison-glaz'``, if for some reason `c` cannot be
        computed; this signals a bug and should be reported.

    Notes
    -----
    The `goodman` method [2]_ is based on approximating a statistic based on
    the multinomial as a chi-squared random variable. The usual recommendation
    is that this is valid if all the values in `counts` are greater than or
    equal to 5. There is no condition on the number of categories for this
    method.

    The `sison-glaz` method [3]_ approximates the multinomial probabilities,
    and evaluates that with a maximum-likelihood estimator. The first
    approximation is an Edgeworth expansion that converges when the number of
    categories goes to infinity, and the maximum-likelihood estimator converges
    when the number of observations (``sum(counts)``) goes to infinity. In
    their paper, Sison & Glaz demo their method with at least 7 categories, so
    ``len(counts) >= 7`` with all values in `counts` at or above 5 can be used
    as a rule of thumb for the validity of this method. This method is less
    conservative than the `goodman` method (i.e. it will yield confidence
    intervals closer to the desired significance level), but produces
    confidence intervals of uniform width over all categories (except when the
    intervals reach 0 or 1, in which case they are truncated), which makes it
    most useful when proportions are of similar magnitude.

    Aside from the original sources ([1]_, [2]_, and [3]_), the implementation
    uses the formulas (though not the code) presented in [4]_ and [5]_.

    References
    ----------
    .. [1] Levin, Bruce, "A representation for multinomial cumulative
           distribution functions," The Annals of Statistics, Vol. 9, No. 5,
           1981, pp. 1123-1126.

    .. [2] Goodman, L.A., "On simultaneous confidence intervals for multinomial
           proportions," Technometrics, Vol. 7, No. 2, 1965, pp. 247-254.

    .. [3] Sison, Cristina P., and Joseph Glaz, "Simultaneous Confidence
           Intervals and Sample Size Determination for Multinomial
           Proportions," Journal of the American Statistical Association,
           Vol. 90, No. 429, 1995, pp. 366-369.

    .. [4] May, Warren L., and William D. Johnson, "A SAS® macro for
           constructing simultaneous confidence intervals  for multinomial
           proportions," Computer methods and programs in Biomedicine, Vol. 53,
           No. 3, 1997, pp. 153-162.

    .. [5] May, Warren L., and William D. Johnson, "Constructing two-sided
           simultaneous confidence intervals for multinomial proportions for
           small counts in a large number of cells," Journal of Statistical
           Software, Vol. 5, No. 6, 2000, pp. 1-24.
    '''
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1), bounds excluded')
    counts = np.array(counts, dtype=np.float)
    if (counts < 0).any():
        raise ValueError('counts must be >= 0')

    n = counts.sum()
    k = len(counts)
    proportions = counts / n
    if method == 'goodman':
        chi2 = stats.chi2.ppf(1 - alpha / k, 1)
        delta = chi2 ** 2 + (4 * n * proportions * chi2 * (1 - proportions))
        region = ((2 * n * proportions + chi2 +
                   np.array([- np.sqrt(delta), np.sqrt(delta)])) /
                  (2 * (chi2 + n))).T
    elif method[:5] == 'sison':  # We accept any name starting with 'sison'
        # Define a few functions we'll use a lot.
        def poisson_interval(interval, p):
            """Compute P(b <= Z <= a) where Z ~ Poisson(p) and
            `interval = (b, a)`."""
            b, a = interval
            prob = stats.poisson.cdf(a, p) - stats.poisson.cdf(b - 1, p)
            if p == 0 and np.isnan(prob):
                # hack for older scipy <=0.16.1
                return int(b - 1 < 0)
            return prob

        def truncated_poisson_factorial_moment(interval, r, p):
            """Compute mu_r, the r-th factorial moment of a poisson random
            variable of parameter `p` truncated to `interval = (b, a)`."""
            b, a = interval
            return p ** r * (1 - ((poisson_interval((a - r + 1, a), p) -
                                   poisson_interval((b - r, b - 1), p)) /
                                  poisson_interval((b, a), p)))

        def edgeworth(intervals):
            """Compute the Edgeworth expansion term of Sison & Glaz's formula
            (1) (approximated probability for multinomial proportions in a
            given box)."""
            # Compute means and central moments of the truncated poisson
            # variables.
            mu_r1, mu_r2, mu_r3, mu_r4 = [
                np.array([truncated_poisson_factorial_moment(interval, r, p)
                          for (interval, p) in zip(intervals, counts)])
                for r in range(1, 5)
            ]
            mu = mu_r1
            mu2 = mu_r2 + mu - mu ** 2
            mu3 = mu_r3 + mu_r2 * (3 - 3 * mu) + mu - 3 * mu ** 2 + 2 * mu ** 3
            mu4 = (mu_r4 + mu_r3 * (6 - 4 * mu) +
                   mu_r2 * (7 - 12 * mu + 6 * mu ** 2) +
                   mu - 4 * mu ** 2 + 6 * mu ** 3 - 3 * mu ** 4)

            # Compute expansion factors, gamma_1 and gamma_2.
            g1 = mu3.sum() / mu2.sum() ** 1.5
            g2 = (mu4.sum() - 3 * (mu2 ** 2).sum()) / mu2.sum() ** 2

            # Compute the expansion itself.
            x = (n - mu.sum()) / np.sqrt(mu2.sum())
            phi = np.exp(- x ** 2 / 2) / np.sqrt(2 * np.pi)
            H3 = x ** 3 - 3 * x
            H4 = x ** 4 - 6 * x ** 2 + 3
            H6 = x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15
            f = phi * (1 + g1 * H3 / 6 + g2 * H4 / 24 + g1 ** 2 * H6 / 72)
            return f / np.sqrt(mu2.sum())


        def approximated_multinomial_interval(intervals):
            """Compute approximated probability for Multinomial(n, proportions)
            to be in `intervals` (Sison & Glaz's formula (1))."""
            return np.exp(
                np.sum(np.log([poisson_interval(interval, p)
                               for (interval, p) in zip(intervals, counts)])) +
                np.log(edgeworth(intervals)) -
                np.log(stats.poisson._pmf(n, n))
            )

        def nu(c):
            """Compute interval coverage for a given `c` (Sison & Glaz's
            formula (7))."""
            return approximated_multinomial_interval(
                [(np.maximum(count - c, 0), np.minimum(count + c, n))
                 for count in counts])

        # Find the value of `c` that will give us the confidence intervals
        # (solving nu(c) <= 1 - alpha < nu(c + 1).
        c = 1.0
        nuc = nu(c)
        nucp1 = nu(c + 1)
        while not (nuc <= (1 - alpha) < nucp1):
            if c > n:
                raise Exception("Couldn't find a value for `c` that "
                                "solves nu(c) <= 1 - alpha < nu(c + 1)")
            c += 1
            nuc = nucp1
            nucp1 = nu(c + 1)

        # Compute gamma and the corresponding confidence intervals.
        g = (1 - alpha - nuc) / (nucp1 - nuc)
        ci_lower = np.maximum(proportions - c / n, 0)
        ci_upper = np.minimum(proportions + (c + 2 * g) / n, 1)
        region = np.array([ci_lower, ci_upper]).T
    else:
        raise NotImplementedError('method "%s" is not available' % method)
    return region


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
    >>> import statsmodels.api as sm
    >>> sm.stats.proportion_effectsize(0.5, 0.4)
    0.20135792079033088
    >>> sm.stats.proportion_effectsize([0.3, 0.4, 0.5], 0.4)
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
        warnings.warn("no overlap, power is zero", HypothesisTestWarning)
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
    """
    Test for proportions based on normal (z) test

    Parameters
    ----------
    count : integer or array_like
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : integer or array-like
        the number of trials or observations, with the same length as
        count.
    value : float, array_like or None, optional
        This is the value of the null hypothesis equal to the proportion in the
        case of a one sample test. In the case of a two-sample test, the
        null hypothesis is that prop[0] - prop[1] = value, where prop is the
        proportion in the two samples. If not provided value = 0 and the null
        is prop[0] = prop[1]
    alternative : string in ['two-sided', 'smaller', 'larger']
        The alternative hypothesis can be either two-sided or one of the one-
        sided tests, smaller means that the alternative hypothesis is
        ``prop < value`` and larger means ``prop > value``. In the two sample
        test, smaller means that the alternative hypothesis is ``p1 < p2`` and
        larger means ``p1 > p2`` where ``p1`` is the proportion of the first
        sample and ``p2`` of the second one.
    prop_var : False or float in (0, 1)
        If prop_var is false, then the variance of the proportion estimate is
        calculated based on the sample proportion. Alternatively, a proportion
        can be specified to calculate this variance. Common use case is to
        use the proportion under the Null hypothesis to specify the variance
        of the proportion estimate.

    Returns
    -------
    zstat : float
        test statistic for the z-test
    p-value : float
        p-value for the z-test

    Examples
    --------
    >>> count = 5
    >>> nobs = 83
    >>> value = .05
    >>> stat, pval = proportions_ztest(count, nobs, value)
    >>> print('{0:0.3f}'.format(pval))
    0.695

    >>> import numpy as np
    >>> from statsmodels.stats.proportion import proportions_ztest
    >>> count = np.array([5, 12])
    >>> nobs = np.array([83, 99])
    >>> stat, pval = proportions_ztest(counts, nobs)
    >>> print('{0:0.3f}'.format(pval))
    0.159

    Notes
    -----
    This uses a simple normal test for proportions. It should be the same as
    running the mean z-test on the data encoded 1 for event and 0 for no event
    so that the sum corresponds to the count.

    In the one and two sample cases with two-sided alternative, this test
    produces the same p-value as ``proportions_chisquare``, since the
    chisquare is the distribution of the square of a standard normal
    distribution.
    """
    # TODO: verify that this really holds
    # TODO: add continuity correction or other improvements for small samples
    # TODO: change options similar to propotion_ztost ?

    count = np.asarray(count)
    nobs = np.asarray(nobs)

    if nobs.size == 1:
        nobs = nobs * np.ones_like(count)

    prop = count * 1. / nobs
    k_sample = np.size(prop)
    if value is None:
        if k_sample == 1:
            raise ValueError('value must be provided for a 1-sample test')
        value = 0
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
