# -*- coding: utf-8 -*-
"""Anova k-sample comparison without and with trimming

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
    axis : int or None
        Axis along which the observations are trimmed. The default is to trim
        along axis=0. If axis is None then the array will be flattened before
        trimming.

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
    if axis is None:
        a = a.ravel()
        axis = 0
    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut >= uppercut):
        raise ValueError("Proportion too big.")

    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lowercut, uppercut)
    return a[tuple(sl)]


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
    axis : int or None
        Axis along which the trimmed means are computed. The default is axis=0.
        If axis is None then the trimmed mean will be computed for the
        flattened array.

    Returns
    -------
    trim_mean : ndarray
        Mean of trimmed array.

    """
    newa = trimboth(np.sort(a, axis), proportiontocut, axis=axis)
    return np.mean(newa, axis=axis)


class TrimmedMean(object):
    '''class for trimmed and winsorized one sample statistics

    '''

    def __init__(self, data, fraction, is_sorted=False, axis=0):
        self.data = np.asarray(data)
        # TODO: add pandas handling, maybe not if this stays internal
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
        # numpy requires now tuple for indexing, not list
        self.sl = tuple(self.sl)
        if not is_sorted:
            self.data_sorted = np.sort(self.data, axis=axis)
        else:
            self.data_sorted = self.data

        self.lowerbound = self.data_sorted[lowercut]
        self.upperbound = self.data_sorted[uppercut - 1]

    @property
    def data_trimmed(self):
        # returns a view
        return self.data_sorted[tuple(self.sl)]

    @property  # cache
    def data_winsorized(self):
        return np.clip(self.data_sorted, self.lowerbound, self.upperbound)

    @property
    def mean_trimmed(self):
        return np.mean(self.data_sorted[tuple(self.sl)], self.axis)

    @property
    def mean_winsorized(self):
        return np.mean(self.data_winsorized, self.axis)

    @property
    def var_winsorized(self):
        # hardcoded ddof = 1
        return np.var(self.data_winsorized, ddof=1)
        return np.var(self.data_winsorized - self.mean_winsorized,
                      ddof=1 + 2 * self.lowercut, axis=self.axis)

    @property
    def std_mean_trimmed(self):
        '''standard error of trimmed mean
        '''
        se = np.sqrt(self.var_winsorized / self.nobs_reduced)
        # trimming creates correlation across trimmed observations
        # trimming is based on order statistics of the data
        # wilcox 2012, p.61
        se *= np.sqrt(self.nobs / self.nobs_reduced)
        return se

    @property
    def std_mean_winsorized(self):
        '''standard error of winsorized mean
        '''
        # the following matches Wilcox, WRS2
        std_ = np.sqrt(self.var_winsorized / self.nobs)
        std_ *= (self.nobs - 1) / (self.nobs_reduced - 1)
        # old version
        # tm = self
        # formula from an old SAS manual page, simplified
        # std_ = np.sqrt(tm.var_winsorized / (tm.nobs_reduced - 1) *
        #               (tm.nobs - 1.) / tm.nobs)
        return std_

    def ttest_mean(self, value=0, transform='trimmed',
                   alternative='two-sided'):
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


def anova_generic(means, vars_, nobs, use_var="unequal",
                  welch_correction=True):
    nobs_t = nobs.sum()
    n_groups = len(means)
    # mean_t = (nobs * means).sum() / nobs_t
    if use_var == "separate":
        weights = nobs / vars_
    else:
        weights = nobs

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    statistic = np.dot(weights, (means - meanw_t)**2) / (n_groups - 1.)

    if use_var == "unequal":
        use_satt = True
        tmp = ((1 - w_rel)**2 / (nobs - 1)).sum() / (n_groups**2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1. / (3. * tmp)

    else:
        use_satt = False
        # variance of group demeaned total sample, pooled var_resid
        tmp = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = 1. / (3. * tmp)

    df_num = n_groups - 1.
    if use_satt:  # Satterthwaite/Welch degrees of freedom
        df_denom = 1. / (3. * tmp)
    else:
        df_denom = nobs_t - n_groups

    pval = stats.f.sf(statistic, df_num, df_denom)
    return statistic, pval, df_num, df_denom


def anova_oneway(data, trim_frac=0):
    '''one-way anova assuming equal variances, unequal sample size

    another implementation
    '''
    args = list(map(np.asarray, data))
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')

    nobs = np.array([len(x) for x in args], float)
    n_groups = len(args)

    if trim_frac == 0:
        means = np.array([x.mean() for x in args])
        vars_ = np.array([x.var(ddof=1) for x in args])
    else:
        tms = [TrimmedMean(x, trim_frac) for x in args]
        means = np.array([tm.mean_trimmed for tm in tms])
        vars_ = np.array([tm.var_winsorized for tm in tms])
        nobs_original = nobs  # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    nobs_t = nobs.sum()
    mean_t = (nobs * means).sum() / nobs_t

    # variance of group demeaned total sample
    tmp = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
    # print 'tmp', tmp
    statistic = 1. * (nobs * (means - mean_t)**2).sum() / (n_groups - 1)
    statistic /= tmp

    df_num = n_groups - 1
    df_denom = nobs_t - n_groups

    pval = stats.f.sf(statistic, df_num, df_denom)
    return statistic, pval, (df_num, df_denom)


def anova_bfm(args, trim_frac=0):
    '''Brown-Forsythe Anova for comparison of means

    This currently returns both Brown-Forsythe and Mehrotra adjusted pvalues.
    The Brown-Forsythe p-values are slightly liberal, i.e. reject too often.
    It can be more powerful than the better sized then the test based on
    Mehrotra p-values.

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
        p-value as in Brown-Forsythe 1974
    pval2 : float
       p-value as corrected by Mehrotra 1997
    (df_denom, df_num, df_denom2) : tuple of floats
        degreeds of freedom for the F-distribution. `df_denom` is for
        Brown-Forsythe p-values, `df_denom2` is for Mehrotra p-values.
        `df_num` is the same numerator degrees of freedom for both p-values.

    References
    ----------
    Brown, Morton B., and Alan B. Forsythe. 1974. “The Small Sample Behavior
    of Some Statistics Which Test the Equality of Several Means.”
    Technometrics 16 (1) (February 1): 129–132. doi:10.2307/1267501.

    Mehrotra, Devan V. 1997. “Improving the Brown-Forsythe Solution to the
    Generalized Behrens-Fisher Problem.” Communications in Statistics -
    Simulation and Computation 26 (3): 1139–1145.
    doi:10.1080/03610919708813431.

    '''
    args = list(map(np.asarray, args))
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
        nobs_original = nobs  # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    nobs_t = nobs.sum()
    mean_t = (nobs * means).sum() / nobs_t

    tmp = ((1. - nobs / nobs_t) * vars_).sum()
    # print 'tmp', tmp
    statistic = 1. * (nobs * (means - mean_t)**2).sum()
    statistic /= tmp

    df_num = len(nobs) - 1
    df_denom = tmp**2 / ((1. - nobs / nobs_t)**2 * vars_**2 / (nobs - 1)).sum()
    df_num2 = tmp**2 / ((vars_**2).sum() +
                        (nobs / nobs_t * vars_).sum()**2 -
                        2 * (nobs / nobs_t * vars_**2).sum())

    pval = stats.f.sf(statistic, df_num, df_denom)
    pval2 = stats.f.sf(statistic, df_num2, df_denom)
    return statistic, pval, pval2, (df_num, df_denom, df_num2)


def anova_welch(args, trim_frac=0, welch_correction=True):
    '''Welch's one-way Anova for samples with heterogeneous variances

    Welch's anova is correctly sized (not liberal or conservative) in smaller
    samples if the distribution of the samples is not very far away from the
    normal distribution. The test can become liberal if the data is strongly
    skewed. Welch's Anova can also be correctly sized for discrete
    distributions with finite support, like Lickert scale data.
    The trimmed version is robust to many non-normal distributions, it stays
    correctly sized in many cases, and is more powerful in some cases with
    skewness or heavy tails.

    Parameters
    ----------
    args : tuple of array_like
        k independent samples, each array needs to be one dimensional and
        contain one independent sample
    trim_frac : float in [0, 0.5)
        optional trimming for Anova with trimmed mean and winsorized variances.
        With the default trim_frac equal to zero, the standard Welch's Anova
        is computed without trimming. I `trim_frac` is larger than zero, then
        then the largest and smallest observations in each sample are trimmed.
        The number of trimmed observations is this fraction of number of
        observations in the sample truncated to the next lower integer.
        `trim_frac` has to be smaller than 0.5, however, if the fraction is
        so large that there are not enough observations left over, then `nan`
        will be returned.

    Returns
    -------
    statistic : float
        test statistic that is approximately F distributed
    p_value : float
        p_value of the approximate F-test
    (df_num, df_denom) : tuple of floats
        degrees of freedom for the F distribution

    Notes
    -----
    This is a reference implementation. Welch's Anova produces the same
    result as `oneway.test` with `equal.var=FALSE` in R stats.
    The trimmed version has been verified by Monte Carlo simulations.

    This function will be replaced by one that is better integrated with
    related classes and functions in statsmodels.

    Trimming is currently based on the integer part of ``nobs * trim_frac``.
    The default might change to including fractional observations as in the
    original articles by Yuen.

    References
    ----------
    Welch

    Yuen

    '''

    args = list(map(np.asarray, args))
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')

    n_groups = len(args)

    # the next block can be replaced by different implementation
    #   for groupsstats
    nobs = np.array([len(x) for x in args], float)
    if trim_frac == 0:
        means = np.array([x.mean() for x in args])
        vars_ = np.array([x.var(ddof=1) for x in args])
    else:
        tms = [TrimmedMean(x, trim_frac) for x in args]
        means = np.array([tm.mean_trimmed for tm in tms])
        # vars_ = np.array([tm.var_winsorized for tm in tms])
        vars_ = np.array([tm.var_winsorized * (tm.nobs - 1) /
                          (tm.nobs_reduced - 1) for tm in tms])
        nobs_original = nobs  # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    nobs_t = nobs.sum()
    # mean_t = (nobs * means).sum() / nobs_t
    weights = nobs / vars_
    weights_t = weights.sum()
    meanw_t = (weights * means).sum() / weights_t

    statistic = np.dot(weights, (means - meanw_t)**2) / (n_groups - 1.)
    tmp = ((1 - weights / weights_t)**2 / (nobs - 1)).sum()
    tmp /= (n_groups**2 - 1)
    if welch_correction:
        statistic /= 1 + 2 * (n_groups - 2) * tmp

    df_num = n_groups - 1.
    df_denom = 1. / (3. * tmp)

    pval = stats.f.sf(statistic, df_num, df_denom)
    return statistic, pval, (df_num, df_denom)


def scale_transform(data, center='median', transform='abs', trim_frac=0.2,
                    axis=0):
    '''transform data for variance comparison for Levene type tests

    Parameters
    ----------
    data : array_like
        observations for the data
    center : str in ['median', 'mean', 'trimmed']
        the statistic that is used as center for the data transformation
    transform : str in ['abs', 'square', 'exp', 'identity']
        the transform for the centered data
    trim_frac : float in [0, 0.5)
        Fraction of observations that are trimmed on each side of the sorted
        observations. This is only used if center is `trimmed`.
    axis : int
        axis along which the data are transformed when centering.

    Returns
    -------
    res : ndarray
        transformed data in the same shape as the original data.

    '''
    x = np.asarray(data)  # x is shorthand from earlier code

    if transform == 'abs':
        tfunc = np.abs
    elif transform == 'square':
        tfunc = lambda x: x * x  #noqa
    elif transform == 'exp':
        tfunc = lambda x: np.exp(np.abs(x))  #noqa
    elif transform == 'identity':
        tfunc = lambda x: x  #noqa
    else:
        raise ValueError('transform should be abs, square or exp')

    if center == 'median':
        res = tfunc(x - np.expand_dims(np.median(x, axis=0), axis))
    elif center == 'mean':
        res = tfunc(x - np.expand_dims(np.mean(x, axis=0), axis))
    elif center == 'trimmed':
        center = trim_mean(x, trim_frac, axis=0)
        res = tfunc(x - np.expand_dims(center, axis))
    elif center == 'no':
        res = tfunc(x)
    else:
        raise ValueError('center should be median, mean or trimmed')

    return res


def anova_scale(data, method='bfm', center='median', transform='abs',
                trim_frac=0.2):
    # print(method, center, transform, trim_frac)
    data = map(np.asarray, data)
    # print [x.mean() for x in data]
    xxd = [scale_transform(x, center=center, transform=transform,
                           trim_frac=trim_frac) for x in data]
    print([x.mean() for x in xxd])
    print(method, method == 'bfm')
    if method == 'bfm':
        res = anova_bfm(xxd)
    elif method == 'levene':
        res = anova_oneway(xxd)
    elif method == 'welch':
        res = anova_welch(xxd)
    else:
        raise ValueError('method "%s" not supported' % method)

    return res, xxd
