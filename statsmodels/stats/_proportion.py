# -*- coding: utf-8 -*-
"""Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats, optimize

def confint_proportion(count, nobs, alpha=0.05, method='normal'):
    '''confidence interval for a binomial proportion

    Paremeters
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

    else:
        raise NotImplementedError('only "normal" is available')
    return ci_low, ci_upp

def samplesize_confint(proportion, half_length, alpha=0.05, method='normal'):
    '''find sample size to get desired confidence interval length

    Parameters
    ----------
    proportion : float in (0, 1)
        proportion or quantile
    half_length : float in (0, 1)
        desired half length of the confidence interval
    alpha : float in (0, 1)
        significance level, default 0.05,
        coverage of the interval is (approximately) ``1 - alpha``
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

count = 84
nobs = n_rep = 10000
low, upp = confint_proportion(count, nobs, alpha=0.05, method='normal')
print low, upp
print low * nobs, upp * nobs
ci_b = confint_proportion(count, nobs, alpha=0.05, method='binom_test')
print ci_b
print np.array(ci_b) * nobs
