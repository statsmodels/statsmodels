# -*- coding: utf-8 -*-
"""

Created on Sun Jun 09 23:51:34 2013

Author: Josef Perktold
"""

import numpy as np
from scipy import stats

# the trimboth and trim_mean are taken from scipy.stats.stats
# and enhanced by axis
def trimboth(a, proportiontocut, axis=0):
    """
    Slices off a proportion of items from both ends of an array.

    Slices off the passed proportion of items from both ends of the passed
    array (i.e., with `proportiontocut` = 0.1, slices leftmost 10% **and**
    rightmost 10% of scores).  You must pre-sort the array if you want
    'proper' trimming.  Slices off less if proportion results in a
    non-integer slice index (i.e., conservatively slices off
    `proportiontocut`).

    Parameters
    ----------
    a : array_like
        Data to trim.
    proportiontocut : float or int
        Proportion of total data set to trim of each end.

    Returns
    -------
    out : ndarray
        Trimmed version of array `a`.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(20)
    >>> b = stats.trimboth(a, 0.1)
    >>> b.shape
    (16,)

    """
    a = np.asarray(a)
    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut >= uppercut):
        raise ValueError("Proportion too big.")

    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lowercut, uppercut)
    return a[sl]


def trim_mean(a, proportiontocut, axis=0):
    """
    Return mean of array after trimming distribution from both lower and upper
    tails.

    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of
    scores. Slices off LESS if proportion results in a non-integer slice
    index (i.e., conservatively slices off `proportiontocut` ).

    Parameters
    ----------
    a : array_like
        Input array
    proportiontocut : float
        Fraction to cut off of both tails of the distribution

    Returns
    -------
    trim_mean : ndarray
        Mean of trimmed array.

    """
    newa = trimboth(np.sort(a, axis), proportiontocut, axis=axis)
    return np.mean(newa, axis=axis)

class TrimmedMean(object):

    def __init__(self, data, fraction, is_sorted=False, axis=0):
        self.data = np.asarray(data)
        #TODO: add pandas handling, maybe not this is only internal
        self.fraction = fraction
        self.axis = axis
        self.nobs = nobs = self.data.shape[axis]
        self.lowercut = lowercut = int(fraction * nobs)
        self.uppercut = uppercut = nobs - lowercut
        if (lowercut >= uppercut):
            raise ValueError("Proportion too big.")
        self.nobs_reduced = nobs - 2 * lowercut

        self.sl = [slice(None)] * self.data.ndim
        self.sl[axis] = slice(self.lowercut, self.uppercut)
        if not is_sorted:
            self.data_sorted = np.sort(self.data, axis=axis)
        else:
            self.data_sorted = self.data

        self.lowerbound = self.data_sorted[lowercut]
        self.upperbound = self.data_sorted[uppercut - 1]

    @property
    def data_trimmed(self):
        # returns a view
        return self.data_sorted[self.sl]

    @property #cache
    def data_winsorized(self):
        return np.clip(self.data_sorted, self.lowerbound, self.upperbound)

    @property
    def mean_trimmed(self):
        return np.mean(self.data_sorted[self.sl], self.axis)

    @property
    def mean_winsorized(self):
        return np.mean(self.data_winsorized, self.axis)

    @property
    def var_winsorized(self):
        # hardcoded ddof = 1
        return np.var(self.data_winsorized - self.mean_winsorized,
                        ddof=1 + 2 * self.lowercut, axis=self.axis)

    @property
    def std_mean_trimmed(self):
        '''standard error of trimmed mean
        '''
        return np.sqrt(self.var_winsorized / self.nobs_reduced)

    @property
    def std_mean_winsorized(self):
        '''standard error of winsorized mean
        '''
        # formula from an old SAS manual page, simplified
        std_ = np.sqrt(tm.var_winsorized / (tm.nobs_reduced - 1) *
                       (tm.nobs - 1.) / tm.nobs)
        return std_

    def ttest_mean(self, value=0, transform='trimmed', alternative='two-sided'):
        '''One sample ttest for trimmed mean

        p-value is based on the approximate t-distribution of the test
        statistic. The approximation is valid if the underlying distribution
        is symmetric.
        '''
        import statsmodels.stats.weightstats as smws
        df = self.nobs_reduced - 1
        if transform == 'trimmed':
            mean_ = self.mean_trimmed
            std_ = self.std_mean_trimmed
        elif transform == 'winsorized':
            mean_ = self.mean_winsorized
            std_ = self.std_mean_winsorized
        else:
            raise ValueError("transform can only be 'trimmed' or 'winsorized'")

        res = smws._tstat_generic(mean_, 0, std_,
                                  df, alternative=alternative, diff=value)
        return res + (df,)


    def reset_fraction(self, frac):
        '''create a TrimmedMean instance with a new trimming fraction

        This reuses the sorted array from the current instance.
        '''
        tm = TrimmedMean(self.data_sorted, frac, is_sorted=True,
                         axis=self.axis)
        tm.data = self.data
        # TODO: this will not work if there is processing of meta-information
        #       in __init__,
        #       for example storing a pandas DataFrame or Series index
        return tm


def anova_bfm(*args):
    '''Brown-Forsythe Anova for comparison of means

    This currently returns both Brown-Forsythe, and Mehrotra adjusted pvalues.
    The Brown-Forsythe p-values are slightly liberal, i.e. reject too often.

    Parameters
    ----------
    args : sequence of arrays
        data for k independent samples

    Returns
    -------
    statistic : float
        test statistic for k-sample mean comparison which is approximately
        F-distributed.
    pval : float
        p-value Brown-Forsythe
    pval2 : float
       p-value as corrected by Mehrotra
    (df_denom, df_num, df_denom2) : tuple of floats
        degreeds of freedom for the F-distribution. `df_denom` is for
        Brown-Forsythe p-values, `df_denom2` is for Mehrotra p-values.
        `df_num` is the same numerator degrees of freedom for both p-values.

    References
    ----------
    Brown, Morton B., and Alan B. Forsythe. 1974. “The Small Sample Behavior of Some Statistics Which Test the Equality of Several Means.” Technometrics 16 (1) (February 1): 129–132. doi:10.2307/1267501.

    Mehrotra, Devan V. 1997. “Improving the Brown-forsythe Solution to the Generalized Behrens-fisher Problem.” Communications in Statistics - Simulation and Computation 26 (3): 1139–1145. doi:10.1080/03610919708813431.

    '''
    args = map(np.asarray, args)
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')

    nobs = np.array([len(x) for x in args], float)
    means = np.array([x.mean() for x in args])
    vars_ = np.array([x.var() for x in args])
    nobs_t = nobs.sum()
    mean_t = (nobs * means).sum() / nobs_t

    tmp = ((1. - nobs / nobs_t) * vars_).sum()
    #print 'tmp', tmp
    statistic = 1. * (nobs * (means - mean_t)**2).sum()
    statistic /= tmp

    df_denom = len(nobs) - 1
    df_num = tmp**2 / ((1. - nobs / nobs_t)**2 * vars_**2 / (nobs - 1)).sum()
    df_denom2 = tmp**2 / ((vars_**2).sum() +
                          (nobs / nobs_t * vars_).sum()**2 -
                           2 * (nobs / nobs_t * vars_**2).sum())


    pval = stats.f.sf(statistic, df_denom, df_num)
    pval2 = stats.f.sf(statistic, df_denom2, df_num)
    return statistic, pval, pval2, (df_denom, df_num, df_denom2)

def anova_welch(args, trim_frac=0):

    args = map(np.asarray, args)
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')

    n_groups = len(args)
    nobs = np.array([len(x) for x in args], float)
    if trim_frac == 0:
        means = np.array([x.mean() for x in args])
        vars_ = np.array([x.var(ddof=1) for x in args])
    else:
        tms = [TrimmedMean(x, trim_frac) for x in args]
        means = np.array([tm.mean_trimmed for tm in tms])
        vars_ = np.array([tm.var_winsorized for tm in tms])
        nobs_original = nobs # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    nobs_t = nobs.sum()
    #mean_t = (nobs * means).sum() / nobs_t
    weights = nobs / vars_
    weights_t = weights.sum()
    meanw_t = (weights * means).sum() / weights_t

    statistic = np.dot(weights, (means - meanw_t)**2) / (n_groups - 1.)
    tmp =  ((1 - weights / weights_t)**2 / (nobs - 1)).sum() / (n_groups**2 - 1)
    statistic /= 1 + 2 * (n_groups - 2) * tmp

    df_num = n_groups - 1.
    df_denom = 1. / (3. * tmp)

    pval = stats.f.sf(statistic, df_num, df_denom)
    return statistic, pval, (df_num, df_denom)


def scale_transform(data, center='median', transform='abs', frac=0.2, axis=0):
    '''transform data for variance comparison for Levene type tests

    Parameters
    ----------
    data : array_like
        observations for the data
    center : str in ['median', 'mean', 'trimmed']
        the statistic that is used as center for the data transformation
    transform : str in ['abs', 'square', 'exp', 'identity']
        the transform for the centered data
    frac : float in [0, 0.5)
        Fraction of observations that are trimmed on each side of the sorted
        observations. This is only used if center is `trimmed`.
    axis : int
        axis along which the data are transformed when centering.

    Returns
    -------
    res : ndarray
        transformed data in the same shape as the original data.

    '''
    data = np.asarray(data)

    if transform == 'abs':
        tfunc = np.abs
    elif transform == 'square':
        tfunc = lambda x : x * x
    elif transform == 'exp':
        tfunc = lambda x : np.exp(np.abs(x))
    elif transform == 'identity':
        tfunc = lambda x : x
    else:
        raise ValueError('transform should be abs, square or exp')

    if center == 'median':
        res = tfunc(x - np.expand_dims(np.median(x, axis=0), axis))
    elif center == 'mean':
        res = tfunc(x - np.expand_dims(np.mean(x, axis=0), axis))
    elif center == 'trimmed':
        center = trim_mean(x, frac, axis=0)
        res = tfunc(x - np.expand_dims(center, axis))
    else:
        raise ValueError('center should be median, mean or trimmed')

    return res

if __name__ == '__main__':
    examples = ['mc', 'anova', 'trimmed'][0]
    if 'mc' in examples:
        np.random.seed(19864256)
        nrep = 10000
        nobs = np.array([5,10,5,5]) * 3
        mm = (1, 1, 1, 2)
        ss = (0.8, 1, 1, 2)
        #ss = (1, 1, 1, 1)

        # run a Monte Carlo simulation to check size and power of tests
        res = np.zeros((nrep, 6))
        res_w = np.zeros((nrep, 4))
        for ii in range(nrep):
            #xx = [m + s * np.random.randn(n) for n, m, s in zip(nobs, mm, ss)]
            xx = [m + s * stats.t.rvs(3, size=n) for n, m, s in zip(nobs, mm, ss)]
            #xx = [m + s * stats.lognorm.rvs(1.5, size=n) - stats.lognorm.mean(1.5, scale=s) for n, m, s in zip(nobs, mm, ss)]
            #xxd = [np.abs(x - np.median(x)) for x in xx]
            xxd = [scale_transform(x, center='trimmed', transform='abs', frac=0.1)
                           for x in xx]
            #print bf_anova(*xx)[:2], bf_anova(*xxd)[:2]
            res[ii] = np.concatenate((anova_bfm(*xx)[:3], anova_bfm(*xxd)[:3]))
            res_w[ii] = np.concatenate((anova_welch(xx)[:2],
                                        anova_welch(xx, trim_frac=0.1)[:2]))

        #print res[:5]
        print nobs
        print mm
        print ss
        print '\nBF'
        print (res[:, [1, 2, 4, 5]] < 0.05).mean(0)
        print '\nWelch'
        print (res_w[:, [1, 3]] < 0.05).mean(0)
        print

    if 'anova' in examples:
        np.random.seed(19864256)
        nobs = np.array([5,10,5,5]) * 3
        mm = (1, 1, 1, 2)
        ss = (0.8, 1, 1, 2)

        xx = [m + s * np.random.randn(n) for n, m, s in zip(nobs, mm, ss)]
        print anova_bfm(*xx)
        print anova_welch(xx)

        npk_yield = np.array([
         49.5, 62.8, 46.8, 57, 59.8, 58.5, 55.5, 56, 62.8, 55.8, 69.5, 55, 62,
         48.8, 45.5, 44.2, 52, 51.5, 49.8, 48.8, 57.2, 59, 53.2, 56
        ])
        npk_block = np.array([
         1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6,
         6 ])
        xyield = [npk_yield[npk_block == idx] for idx in range(1,7)]
        print anova_bfm(*xyield)
        print anova_welch(xyield)
        idx_include = range(24)
        del idx_include[15]
        del idx_include[2]
        # unbalanced sample sizes
        npk_block_ub = npk_block[idx_include]
        npk_yield_ub = npk_yield[idx_include]
        xyield_ub = [npk_yield_ub[npk_block_ub == idx] for idx in range(1,7)]
        print anova_bfm(*xyield_ub)
        print anova_welch(xyield_ub)
        print anova_welch(xyield_ub, trim_frac=0.01)
        print anova_welch(xyield_ub, trim_frac=0.25)


    if 'trimmed' in examples:
        #x = np.random.permutation(np.arange(10))
        x = np.array([4, 9, 3, 1, 6, 5, 7, 10, 2, 8, 50])
        tm = TrimmedMean(x, 0.2)
        print vars(tm)
        print tm.data_winsorized
        print tm.data_trimmed
        print tm.mean_trimmed
        print tm.mean_winsorized
        print tm.var_winsorized
        tm2 = tm.reset_fraction(0.1)
        print tm2.data_winsorized
        print tm2.data_trimmed

        tm = tm.reset_fraction(0)
        import statsmodels.stats.weightstats as smws
        print smws._tstat_generic(tm.mean_trimmed, 0, tm.std_mean_trimmed,
                                  tm.nobs_reduced - 1,
                                  alternative='two-sided', diff=3)
        print smws.DescrStatsW(x).ttest_mean(3)
        print tm.ttest_mean(3, transform='winsorized')
