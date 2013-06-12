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


def anova_bfm(*args):
    '''Brown-Forsythe Anova for comparison of means

    This currently returns both Brown-Forsythe, and Mehrotra adjusted pvalues.
    The Brown-Forsythe p-values are slightly liberal, i.e. reject too often.

    References
    ----------
    Brown, Morton B., and Alan B. Forsythe. 1974. “The Small Sample Behavior of Some Statistics Which Test the Equality of Several Means.” Technometrics 16 (1) (February 1): 129–132. doi:10.2307/1267501.

    Mehrotra, Devan V. 1997. “Improving the Brown-forsythe Solution to the Generalized Behrens-fisher Problem.” Communications in Statistics - Simulation and Computation 26 (3): 1139–1145. doi:10.1080/03610919708813431.

    '''
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

def scale_transform(data, center='median', transform='abs', frac=0.2, axis=0):

    if transform == 'abs':
        tfunc = np.abs
    elif transform == 'square':
        tfunc = lambda x : x * x
    elif transform == 'exp':
        tfunc = lambda x : np.exp(np.abs(x))
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

np.random.seed(19864256)
nrep = 10000
nobs = np.array([5,10,5,5]) * 3
mm = (1, 1, 1, 1)
ss = (0.8, 1, 1, 2)
#ss = (1, 1, 1, 1)
res = np.zeros((nrep, 6))
for ii in range(nrep):
    #xx = [m + s * np.random.randn(n) for n, m, s in zip(nobs, mm, ss)]
    #xx = [m + s * stats.t.rvs(3, size=n) for n, m, s in zip(nobs, mm, ss)]
    xx = [m + s * stats.lognorm.rvs(1.5, size=n) - stats.lognorm.mean(1.5, scale=s) for n, m, s in zip(nobs, mm, ss)]
    #xxd = [np.abs(x - np.median(x)) for x in xx]
    xxd = [scale_transform(x, center='trimmed', transform='abs', frac=0.1)
                   for x in xx]
    #print bf_anova(*xx)[:2], bf_anova(*xxd)[:2]
    res[ii] = np.concatenate((anova_bfm(*xx)[:3], anova_bfm(*xxd)[:3]))

#print res[:5]
print nobs
print mm
print ss
print (res[:, [1, 2, 4, 5]] < 0.05).mean(0)
