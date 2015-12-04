'''Test for ratio of Poisson intensities in two independent samples

Author: Josef Perktold
License: BSD-3

destination statsmodels

'''


from __future__ import division
import numpy as np
from scipy import stats


# copied from statsmodels.stats.weightstats
def _zstat_generic2(value, std_diff, alternative):
    '''generic (normal) z-test to save typing

    can be used as ztest based on summary statistics
    '''
    zstat = value / std_diff
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = stats.norm.sf(np.abs(zstat))*2
    elif alternative in ['larger', 'l']:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ['smaller', 's']:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError('invalid alternative')
    return zstat, pvalue


def poisson_twosample(count1, exposure1, count2, exposure2, ratio_null=1,
                      method='score', alternative='2-sided'):
    '''test for ratio of two sample Poisson intensities

    If the two Poisson rates are g1 and g2, then the Null hypothesis is

    H0: g1 / g2 = ratio_null

    against one of the following alternatives

    H1_2-sided: g1 / g2 != ratio_null
    H1_larger: g1 / g2 > ratio_null
    H1_smaller: g1 / g2 < ratio_null

    Parameters
    ----------
    count1: int
        Number of events in first sample
    exposure1: float
        Total exposure (time * subjects) in first sample
    count2: int
        Number of events in first sample
    exposure2: float
        Total exposure (time * subjects) in first sample
    ratio: float
        ratio of the two Poisson rates under the Null hypothesis. Default is 1.
    method: string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        Current Methods are based on Gu et. al 2008
        Implemented are 'wald', 'score' and 'sqrt' based asymptotic normal
        distribution, and the exact conditional test 'exact-cond', and its mid-point
        version 'cond-midp', see Notes
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

           'two-sided': H1: ratio of rates is not equal to ratio_null (default)
           'larger' :   H1: ratio of rates is larger than ratio_null
           'smaller' :  H1: ratio of rates is smaller than ratio_null

    Returns
    -------
    stat, pvalue two-sided

    not yet
    #results : Results instance
    #    The resulting test statistics and p-values are available as attributes.


    Notes
    -----
    'wald': method W1A, wald test, variance based on separate estimates
    'score': method W2A, score test, variance based on estimate under Null
    'wald-log': W3A
    'score-log' W4A
    'sqrt': W5A, based on variance stabilizing square root transformation
    'exact-cond': exact conditional test based on binomial distribution
    'cond-midp': midpoint-pvalue of exact conditional test

    The latter two are only verified for one-sided example.

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

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
        pvalue = proportion.binom_test(y1, y_total, prop=bp, alternative=alternative)
        if method in ['cond-midp']:
            # not inplace in case we still want binom pvalue
            pvalue = pvalue - 0.5 * stats.binom.pmf(y1, y_total, bp)

        dist = 'binomial'

    if dist == 'normal':
        return _zstat_generic2(stat, 1, alternative)
    else:
        return stat, pvalue


def etest_twopoisson(count1, exposure1, count2, exposure2, ratio_null=1,
                         method='score', alternative='2-sided'):
    """initial version of E-test for two sample Poisson comparison

    """
    y1, n1, y2, n2 = count1, exposure1, count2, exposure2
    d = n2 / n1
    r = ratio_null
    r_d = r / d

    if method in ['score']:
        stat_func = lambda x1, x2:  (x1 - x2 * r_d) / np.sqrt((x1 + x2) * r_d)
        rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
        rate1_cmle = rate2_cmle * r
        rate1 = rate1_cmle
        rate2 = rate2_cmle
    elif method in ['wald']:
        #stat_func = lambda x1, x2:  (x1 - x2 * r_d) / np.sqrt(x1 + x2 * r_d**2)
        def stat_func(x1, x2):
            return (x1 - x2 * r_d) / np.sqrt(x1 + x2 * r_d**2)
        rate2_mle = y2 / n2
        rate1_mle = y1 / n1
        rate1 = rate1_mle
        rate2 = rate2_mle

    # The sampling distribution needs to be based on the null hypotheis
    # use constrained MLE from 'score' calculation
    rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
    rate1_cmle = rate2_cmle * r
    rate1 = rate1_cmle
    rate2 = rate2_cmle
    mean1 = n1 * rate1
    mean2 = n2 * rate2

    stat_sample = stat_func(y1, y2)

    # The following uses a fixed truncation for evaluating the probabilities
    # It will currently only work for small counts, so that sf at truncation point
    # is small
    # We can make it depend on the amount of truncated sf.
    # Some numerical optimization or checks for large means need to be added.
    threshold = 100
    threshold = stats.poisson.isf(1e-13, max(mean1, mean2))
    threshold = max(threshold, 100)   # keep at least 100
    y_grid = np.arange(threshold + 1)
    pdf1 = stats.poisson.pmf(y_grid, mean1)
    pdf2 = stats.poisson.pmf(y_grid, mean2)

    stat_space = stat_func(y_grid[:, None], y_grid[None, :]) # use broadcasting
    eps = 1e-15   # correction for strict inequality check

    if alternative in ['two-sided', '2-sided', '2s']:
        mask = np.abs(stat_space) >= np.abs(stat_sample) - eps
    elif alternative in ['larger', 'l']:
        mask = stat_space >= stat_sample  - eps
    elif alternative in ['smaller', 's']:
        mask = stat_space <= stat_sample + eps
    else:
        raise ValueError('invalid alternative')


    print('debug')
    print(stat_sample, mask.sum(), mean1, mean2)
    pvalue = ((pdf1[:, None] * pdf2[None, :])[mask]).sum()
    return None, pvalue



