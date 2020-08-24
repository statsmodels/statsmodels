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


def rankdata_2samp(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    nobs1 = len(x1)
    nobs2 = len(x2)
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError("one sample has zero length")

    x_combined = np.concatenate((x1, x2))
    if x_combined.ndim > 1:
        rank = np.apply_along_axis(rankdata, 0, x_combined)
    else:
        rank = rankdata(x_combined)  # no axis in older scipy
    rank1 = rank[:nobs1]
    rank2 = rank[nobs1:]
    if x_combined.ndim > 1:
        ranki1 = np.apply_along_axis(rankdata, 0, x1)
        ranki2 = np.apply_along_axis(rankdata, 0, x2)
    else:
        ranki1 = rankdata(x1)
        ranki2 = rankdata(x2)
    return rank1, rank2, ranki1, ranki2


class RankCompareResult(HolderTuple):
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
            stat, pv = _zstat_generic(self.prob1, p0, std_diff, alternative,
                                      diff=0)
            distr = "normal"
        else:
            stat, pv = _tstat_generic(self.prob1, p0, std_diff, self.df,
                                      alternative, diff=0)
            distr = "t"

        res = HolderTuple(statistic=stat,
                          pvalue=pv,
                          df=self.df,
                          distribution=distr
                          )
        return res

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

        t1 = self.test_prob_superior(low, alternative='larger')
        t2 = self.test_prob_superior(upp, alternative='smaller')

        # idx_max = 0 if t1.pvalue < t2.pvalue else 1
        idx_max = np.asarray(t1.pvalue < t2.pvalue, int)
        title = "Equivalence test for Prob(x1 > x2) + 0.5 Prob(x1 = x2) "
        res = HolderTuple(statistic=np.choose(idx_max,
                                              [t1.statistic, t2.statistic]),
                          # pvalue=[t1.pvalue, t2.pvalue][idx_max], # python
                          # use np.choose for vectorized selection
                          pvalue=np.choose(idx_max, [t1.pvalue, t2.pvalue]),
                          results_larger=t1,
                          results_smaller=t2,
                          title=title
                          )
        return res

    def summary(self, alpha=0.05, xname=None):

        yname = "None"
        effect = np.atleast_1d(self.prob1)
        pvalues = np.atleast_1d(self.pvalue)
        ci = np.atleast_2d(self.conf_int(alpha))
        if ci.shape[0] > 1:
            ci = ci.T
        use_t = (self.df is not None)
        sd = np.atleast_1d(np.sqrt(self.var_prob))
        statistic = np.atleast_1d(self.statistic)
        if xname is None:
            xname = ['c%d' % ii for ii in range(len(effect))]

        xname2 = ['prob(x1>x2) %s' % ii for ii in xname]

        title = "Probability sample 1 is stochastically larger"
        from statsmodels.iolib.summary import summary_params

        summ = summary_params((self, effect, sd, statistic,
                               pvalues, ci),
                              yname=yname, xname=xname2, use_t=use_t,
                              title=title, alpha=alpha)
        return summ


def rank_compare_2indep(x1, x2, use_t=True):
    """
    Statistics and tests for the probability that x1 has larger values than x2.

    p is the probability that a random draw from the population of
    the first sample has a larger value than a random draw from the
    population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

    This is a measure underlying Wilcoxon-Mann-Whitney's U test,
    Fligner-Policello test and Brunner-Munzel test, and
    Inference is based on the asymptotic distribution of the Brunner-Munzel
    test.

    The Null hypothesis for stochastic equality is p = 0.5, which corresponds
    to the Brunner-Munzel test.

    Parameters
    ----------
    x1, x2 : array_like
        Array of samples, should be one-dimensional.
    use_t : poolean
        If use_t is true, the t distribution with Welch-Satterthwaite type
        degrees of freedom is used for p-value and confidence interval.
        If use_t is false, then the normal distribution is used.

    Returns
    -------
    res : RankCompareResult

        statistic : float
            The Brunner-Munzer W statistic.
        pvalue : float
            p-value assuming an t distribution. One-sided or
            two-sided, depending on the choice of `alternative` and `use_t`.


    See Also
    --------
    scipy.stats.brunnermunzel : Brunner-Munzel test for stochastic equality
    scipy.stats.mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    Wilcoxon-Mann-Whitney assumes equal variance or equal distribution under
    the Null hypothesis. Fligner-Policello test allows for unequal variances
    but assumes continuous distribution, i.e. no ties.
    Brunner-Munzel extend the test to allow for unequal variance and discrete
    or ordered categorical random variables.

    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_).

    This measure has been introduced in the literature under many different
    names relying on a variety of assumptions.
    In psychology, ... introduced it as Common Language effect size for the
    continuous, normal distribution case, ... extended it to the nonparameteric
    continuous distribution case as in Fligner-Policello.

    Note: Brunner-Munzel define the probability for x1 to be stochastically
    smaller than x2, while here we use stochastically larger.

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
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    nobs1 = len(x1)
    nobs2 = len(x2)
    nobs = nobs1 + nobs2
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError("one sample has zero length")

    rank1, rank2, ranki1, ranki2 = rankdata_2samp(x1, x2)

    meanr1 = np.mean(rank1, axis=0)
    meanr2 = np.mean(rank2, axis=0)
    meanri1 = np.mean(ranki1, axis=0)
    meanri2 = np.mean(ranki2, axis=0)

    S1 = np.sum(np.power(rank1 - ranki1 - meanr1 + meanri1, 2.0), axis=0)
    S1 /= nobs1 - 1
    S2 = np.sum(np.power(rank2 - ranki2 - meanr2 + meanri2, 2.0), axis=0)
    S2 /= nobs2 - 1

    wbfn = nobs1 * nobs2 * (meanr1 - meanr2)
    wbfn /= (nobs1 + nobs2) * np.sqrt(nobs1 * S1 + nobs2 * S2)

    # Here we only use alternative == "two-sided"
    if use_t:
        df_numer = np.power(nobs1 * S1 + nobs2 * S2, 2.0)
        df_denom = np.power(nobs1 * S1, 2.0) / (nobs1 - 1)
        df_denom += np.power(nobs2 * S2, 2.0) / (nobs2 - 1)
        df = df_numer / df_denom
        pvalue = 2 * stats.t.sf(np.abs(wbfn), df)
    else:
        pvalue = 2 * stats.norm.sf(np.abs(wbfn))
        df = None

    # other info
    var1 = S1 / (nobs - nobs1)**2
    var2 = S2 / (nobs - nobs2)**2
    var_prob = (var1 / nobs1 + var2 / nobs2)
    var = nobs * (var1 / nobs1 + var2 / nobs2)
    prob1 = (meanr1 - (nobs1 + 1) / 2) / nobs2
    prob2 = (meanr2 - (nobs2 + 1) / 2) / nobs1

    return RankCompareResult(statistic=wbfn, pvalue=pvalue, s1=S1, s2=S2,
                             var1=var1, var2=var2, var=var,
                             var_prob=var_prob,
                             nobs1=nobs1, nobs2=nobs2, nobs=nobs,
                             mean1=meanr1, mean2=meanr2,
                             prob1=prob1, prob2=prob2,
                             somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1,
                             df=df
                             )
