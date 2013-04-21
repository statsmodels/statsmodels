# -*- coding: utf-8 -*-
"""Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats, optimize

import statsmodels.stats.multitest as smt

def confint_proportion(count, nobs, alpha=0.05, method='normal'):
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

def proportion_effectsize(prob1, prob2, method='normal'):
    '''effect size for a test comparing two proportions

    for use in power function

    Parameters
    ----------
    prob1, prob2: float or array_like

    Returns
    -------
    es : float or ndarray
        effect size

    Notes
    -----
    only method='normal' is implemented to match pwr.p2.test
    see http://www.statmethods.net/stats/power.html

    I think other conversions to normality can be used, but I need to check.

    '''
    if method != 'normal':
        raise ValueError('only "normal" is implemented')

    es = 2 * (np.arcsin(np.sqrt(prob2)) - np.arcsin(np.sqrt(prob1)))
    return es

def std_prop(p, nobs):
    '''standard error for the estimate of a proportion
    '''
    return np.sqrt(p * (1. - p) / nobs)

def _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt,
                 alpha=0.05, discrete=True, dist='norm', nobs=None):
    '''Generic statistical power function for normal based equivalence test
    '''
    crit = stats.norm.isf(alpha)
    k_low = mean_low + np.sqrt(var_low) * crit
    k_upp = mean_upp - np.sqrt(var_upp) * crit
    if discrete:
        k_low = np.ceil(k_low * nobs)
        k_upp = np.trunc(k_upp * nobs)
        if dist == 'norm':
            #need proportion
            k_low = k_low * 1. / nobs
            k_upp = k_upp * 1. / nobs
    else:
        if dist == 'binom':
            #need counts
            k_low *= nobs
            k_upp *= nobs
    #print mean_low, np.sqrt(var_low), crit, var_low
    #print mean_upp, np.sqrt(var_upp), crit, var_upp
    if np.any(k_low > k_upp):   #vectorize
        print "no overlap, power is zero, TODO"
    std_alt = np.sqrt(var_alt)
    z_low = (k_low - mean_alt) / std_alt
    z_upp = (k_upp - mean_alt) / std_alt
    if dist == 'norm':
        power = stats.norm.cdf(z_upp) - stats.norm.cdf(z_low)
    elif dist == 'binom':
        power = (stats.binom.cdf(k_upp, nobs, mean_alt) -
                     stats.binom.cdf(k_low-1, nobs, mean_alt))
    return power, (k_low, k_upp, z_low, z_upp)


def binom_tost(count, nobs, low, upp):
    '''exact tost for one proportion using binomial distribution
    '''
    # binom_test_stat only returns pval
    tt1 = binom_test_stat(count, nobs, alternative='larger', p=low)
    tt2 = binom_test_stat(count, nobs, alternative='smaller', p=upp)
    return np.maximum(tt1, tt2), tt1, tt2,


def binom_tost_reject_interval(low, upp, nobs, alpha=0.05):
    '''rejection region for binomial TOST

    The interval includes the end points,
    `reject` if and only if `r_low <= x <= r_upp`.

    The interval might be empty with `r_upp < r_low`.

    '''
    x_low = stats.binom.isf(alpha, nobs, low) + 1
    x_upp = stats.binom.ppf(alpha, nobs, upp) - 1
    return x_low, x_upp

def binom_test_reject_interval(value, nobs, alpha=0.05, alternative='2-sided'):
    '''rejection region for binomial test for one sample proportion

    The interval includes the end points of the rejection region.


    '''
    if alternative in ['2s', '2-sided']:
        alternative = '2s'
        alpha = alpha / 2

    if alternative in ['2s', 'larger']:
        x_low = stats.binom.ppf(alpha, nobs, value) - 1
    else:
        x_low = 0
    if alternative in ['2s', 'smaller']:
        x_upp = stats.binom.isf(alpha, nobs, value) + 1
    else :
        x_upp = nobs

    return x_low, x_upp

def binom_test_stat(n_success, nobs, p=0.5, alternative='2-sided'):
    '''Perform a test that the probability of success is p.

    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.

    Parameters
    ----------
    n_success : integer or array_like
        the number of successes in nobs trials.
    nobs : integer
        the number of trials or observations.
    p : float, optional
        The hypothesized probability of success.  0 <= p <= 1. The
        default value is p = 0.5

    Returns
    -------
    p-value : float
        The p-value of the hypothesis test

    Notes
    -----
    This uses scipy.stats.binom_test for the two sided alternative.
    '''
    if np.any(p > 1.0) or np.any(p < 0.0):
        raise ValueError("p must be in range [0,1]")
    if alternative in ['2s', '2-sided']:
        pval = stats.binom_test(n_success, n=nobs, p=p)
    elif alternative in ['l', 'larger']:
        pval = stats.binom.sf(n_success-1, nobs, p)
    elif alternative in ['s', 'smaller']:
        pval = stats.binom.cdf(n_success, nobs, p)
    else:
        raise ValueError('alternative not recognized\n'
                         'should be 2-sided, larger or smaller')
    return pval


def power_binom_tost(low, upp, nobs, p_alt=None, alpha=0.05):
    if p_alt is None:
        p_alt = 0.5 * (low + upp)
    x_low, x_upp = binom_tost_reject_interval(low, upp, nobs, alpha=alpha)
    power = (stats.binom.cdf(x_upp, nobs, p_alt) -
                     stats.binom.cdf(x_low-1, nobs, p_alt))
    return power

def power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, variance_prop=None,
                     discrete=True, dist='norm'):
    '''Statistical power of proportions test based on normal distribution

    Parameter
    ---------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        number of observations
    p_alt : float in (0,1)
        proportion under the alternative
    alpha : float in (0,1)
        significance level of the test
    variance_prop : None or float in (0,1)
        If this is None, then the variances for the two one sided tests are
        based on the proportions equal to the equivalence limits.
        If variance_prop is given, then it is used to calculate the variance
        for the TOST statistics. If this is based on an sample, then the
        estimated proportion can be used.

    Returns
    -------
    power : float
        statistical power of the equivalence test.
    (k_low, k_upp, z_low, z_upp) : tuple of floats
        critical limits in intermediate steps
        temporary return, will be changed

    Notes
    -----
    No benchmark case available yet for testing.
    Result is within 2 to 3 percent of an example of PASS.

    works vectorized (for n)
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
                 alpha=alpha, discrete=discrete, dist=dist, nobs=nobs)
    return power

def _table_proportion(n_success, nobs):
    '''create a k by 2 contingency table for proportion
    '''
    table = np.column_stack((n_success, nobs - n_success))
    expected = table.sum(0) * table.sum(1)[:,None] * 1. / table.sum()
    n_rows = table.shape[0]
    return table, expected, n_rows



def proportions_chisquare(n_success, nobs, value=None):
    '''test for proportions based on chisquare test

    Notes
    -----
    Recent version of scipy.stats have a chisquare test for independence in
    contingency tables.

    '''
    nobs = np.atleast_1d(nobs)
    table, expected, n_rows = _table_proportion(n_success, nobs)
    if value is not None:
        expected = np.column_stack((nobs * value, nobs * (1 - value)))
        ddof = n_rows - 1
    else:
        ddof = n_rows

    #print table, expected
    chi2stat, pval = stats.chisquare(table.ravel(), expected.ravel(),
                                     ddof=ddof)
    return chi2stat, pval, (table, expected)

class AllPairsResults(object):
    '''Results class for pairwise comparisons, based on p-values

    Parameter
    ---------
    pvals_raw : array_like, 1-D
        p-values from a pairwise comparison test
    all_pairs : list of tuples
        list of indices, one pair for each comparison
    multitest_method : string
        method that is used by default for p-value correction. This is used
        as default by the methods like if the multiple-testing method is not
        specified as argument.
    levels : None or list of strings
        optional names of the levels or groups
    n_levels : None or int
        If None, then the number of levels or groups is inferred from the
        other arguments. It can be explicitly specified, if it is not a
        standard all pairs comparison.

    Notes
    -----
    It should be possible to use this for other pairwise comparisons, for
    example all others compared to a control (Dunnet).


    '''


    def __init__(self, pvals_raw, all_pairs, multitest_method='hs',
                 levels=None, n_levels=None):
        self.pvals_raw = pvals_raw
        self.all_pairs = all_pairs
        if n_levels is None:
            # for all_pairs nobs*(nobs-1)/2
            #self.n_levels = (1. + np.sqrt(1 + 8 * len(all_pairs))) * 0.5
            self.n_levels = np.max(all_pairs) + 1
        else:
            self.n_levels = n_levels

        self.multitest_method = multitest_method
        self.levels = levels
        if levels is None:
            self.all_pairs_names = ['%r' % (pairs,) for pairs in all_pairs]
        else:
            self.all_pairs_names = ['%s-%s' % (levels[pairs[0]],
                                               levels[pairs[1]])
                                               for pairs in all_pairs]

    def pval_corrected(self, method=None):
        if method is None:
            method = self.multitest_method
        #TODO: breaks with method=None
        return smt.multipletests(self.pvals_raw, method=method)[1]

    def __str__(self):
        return self.summary()

    def pval_table(self):
        k = self.n_levels
        pvals_mat = np.zeros((k, k))
        # if we don't assume we have all pairs
        pvals_mat[zip(*self.all_pairs)] = self.pval_corrected()
        #pvals_mat[np.triu_indices(k, 1)] = self.pval_corrected()
        return pvals_mat

    def summary(self):
        maxlevel = max((len(ss) for ss in self.all_pairs_names))

        text = 'Corrected p-values using %s p-value correction\n\n' % \
                        smt.multitest_methods_names[self.multitest_method]
        text += 'Pairs' + (' ' * (maxlevel - 5 + 1)) + 'p-values\n'
        text += '\n'.join(('%s  %6.4g' % (pairs, pv) for (pairs, pv) in
                zip(self.all_pairs_names, self.pval_corrected())))
        return text



def proportions_chisquare_allpairs(n_success, nobs, value=None,
                                   multitest_method='hs'):
    '''chisquare test of proportions for all pairs of k samples

    Notes
    -----
    no Yates correction
    '''
    #all_pairs = map(list, zip(*np.triu_indices(4, 1)))
    all_pairs = zip(*np.triu_indices(4, 1))
    pvals = [proportions_chisquare(n_success[list(pair)], nobs[list(pair)])[1]
               for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)

def proportions_chisquare_pairscontrol(n_success, nobs, value=None,
                                   multitest_method='hs'):
    '''chisquare test of proportions for pairs of k samples compared to standard

    Notes
    -----
    no Yates correction
    '''
    #all_pairs = map(list, zip(*np.triu_indices(4, 1)))
    all_pairs = [(0, k) for k in range(1, len(n_success))]
    pvals = [proportions_chisquare(n_success[list(pair)], nobs[list(pair)])[1]
               for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)
