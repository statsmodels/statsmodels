# -*- coding: utf-8 -*-
"""
Rank based methods for inferential statistics

Created on Sat Aug 15 10:18:53 2020

Author: Josef Perktold
License: BSD-3

"""


import numpy as np

from scipy import stats
from scipy.stats import rankdata

from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
    _zconfint_generic, _tconfint_generic, _zstat_generic, _tstat_generic)


class BrunnerMunzelResult(HolderTuple):
    """Results for rank comparison
    """

    def conf_int(self, alpha=0.05, value=None, alternative="two-sided"):

        p0 = value
        if p0 is None:
            p0 = 0
        diff = self.prob1 - p0
        std_diff = np.sqrt(self.var / self.nobs)

        if self.df is None:
            return _zconfint_generic(diff, std_diff, alpha, alternative)
        else:
            return _tconfint_generic(diff, std_diff, self.df, alpha,
                                     alternative)

    def test_prob_superior(self, value=0.5, alternative="two-sided"):
        """test for superiority probability

        H0: P(x1 > x2) + 0.5 * P(x1 = x2) = value
        """

        p0 = value  # alias
        # diff = self.prob1 - p0  # for reporting, not used in computation
        # TODO: use var_prob
        std_diff = np.sqrt(self.var / self.nobs)

        # TODO: return HolderTuple
        # corresponds to a one-sample test and either p0 or diff could be used
        if self.df is None:
            return _zstat_generic(self.prob1, p0, std_diff, alternative,
                                  diff=0)
        else:
            return _tstat_generic(self.prob1, p0, std_diff, self.df,
                                  alternative, diff=0)

    def tost_prob_superior(self, low, upp):
        '''test of stochastic (non-)equivalence of p = P(x1 > x2)

        null hypothesis:  p < low or p > upp
        alternative hypothesis:  low < p < upp

        where p is the probability that a random draw from the population of
        the first sample has a larger value than a random draw from the
        population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the probability p that distribution 1 is stochastically
        superior to distribution 2 is outside of the interval given by
        thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1, df1 : tuple
            test statistic, pvalue and degrees of freedom for lower threshold
            test
        t2, pv2, df2 : tuple
            test statistic, pvalue and degrees of freedom for upper threshold
            test

        '''

        t1, pv1 = self.test_prob_superior(low, alternative='larger')
        t2, pv2 = self.test_prob_superior(upp, alternative='smaller')
        df1 = df2 = None
        # TODO: return HolderTuple
        return np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2)


def brunnermunzel(x, y, alternative="two-sided", distribution="t",
                  nan_policy='propagate'):
    """
    Compute the Brunner-Munzel test on samples x and y.
    The Brunner-Munzel test is a nonparametric test of the null hypothesis that
    when values are taken one by one from each group, the probabilities of
    getting large values in both groups are equal.
    Unlike the Wilcoxon-Mann-Whitney's U test, this does not require the
    assumption of equivariance of two groups. Note that this does not assume
    the distributions are same. This test works on two independent samples,
    which may have different sizes.
    Parameters
    ----------
    x, y : array_like
        Array of samples, should be one-dimensional.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    distribution : {'t', 'normal'}, optional
        Defines how to get the p-value.
        The following options are available (default is 't'):
          * 't': get the p-value by t-distribution
          * 'normal': get the p-value by standard normal distribution.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.
    See Also
    --------
    mannwhitneyu : Mann-Whitney rank test on two samples.
    Notes
    -----
    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_).
    References
    ----------
    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal. Vol. 42(2000): 17-25.
    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the
           non-parametric Behrens-Fisher problem". Computational Statistics and
           Data Analysis. Vol. 51(2007): 5192-5204.
    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [1,2,1,1,1,1,1,1,1,1,2,4,1,1]
    >>> x2 = [3,3,4,3,1,2,3,1,1,5,4]
    >>> w, p_value = stats.brunnermunzel(x1, x2)
    >>> w
    3.1374674823029505
    >>> p_value
    0.0057862086661515377
    """
    x = np.asarray(x)
    y = np.asarray(y)

#    # check both x and y
#    cnx, npx = _contains_nan(x, nan_policy)
#    cny, npy = _contains_nan(y, nan_policy)
#    contains_nan = cnx or cny
#    if npx == "omit" or npy == "omit":
#        nan_policy = "omit"

#    if contains_nan and nan_policy == "propagate":
#        return BrunnerMunzelResult(np.nan, np.nan)
#    elif contains_nan and nan_policy == "omit":
#        x = ma.masked_invalid(x)
#        y = ma.masked_invalid(y)
#        return mstats_basic.brunnermunzel(x, y, alternative, distribution)

    nx = len(x)
    ny = len(y)
    nobs = nx + ny
    if nx == 0 or ny == 0:
        return BrunnerMunzelResult(np.nan, np.nan)
    rankc = rankdata(np.concatenate((x, y)))
    rankcx = rankc[0:nx]
    rankcy = rankc[nx:nx+ny]
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    rankx = rankdata(x)
    ranky = rankdata(y)
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)

    if distribution == "t":
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        df = df_numer / df_denom
        p = stats.t.cdf(wbfn, df)
    elif distribution == "normal":
        p = stats.norm.cdf(wbfn)
        df = None
    else:
        raise ValueError(
            "distribution should be 't' or 'normal'")

    if alternative == "greater":
        pass
    elif alternative == "less":
        p = 1 - p
    elif alternative == "two-sided":
        p = 2 * np.min([p, 1-p])
    else:
        raise ValueError(
            "alternative should be 'less', 'greater' or 'two-sided'")

    # other info
    nobs1, nobs2 = nx, ny   # rename
    mean1 = rankcx_mean
    mean2 = rankcy_mean

    var1 = Sx / (nobs - nx)**2
    var2 = Sy / (nobs - ny)**2
    var_prob = (var1 / nobs1 + var2 / nobs2)
    var = nobs * (var1 / nobs1 + var2 / nobs2)
    prob1 = (mean1 - (nobs1 + 1) / 2) / nobs2
    prob2 = (mean2 - (nobs2 + 1) / 2) / nobs1

    return BrunnerMunzelResult(statistic=wbfn, pvalue=p, x1=Sx, s2=Sy,
                               var1=var1, var2=var2, var=var,
                               var_prob=var_prob,
                               nobs1=nx, nobs2=ny, nobs=nobs,
                               mean1=mean1, mean2=mean2,
                               prob1=prob1, prob2=prob2,
                               somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1,
                               df=df
                               )
