# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:28:26 2015

Author: Josef Perktold
License: BSD-3

"""

from __future__ import division

import numpy as np
from scipy import stats


class ExactTwoProportion(object):
    """Tests for equality of two independent proportions

    This class provides methods for unconditional exact and for approximate or
    asymptotic tests for the difference between two proportions from
    independent samples.


    Warning: this is designed for small to moderate sample sizes. It creates
    arrays with size equal to the cardinality of the underlying sample space.


    There is some ambiguity in how the test statistic is defined in corner
    cases when all observations have the same outcome, i.e. points in the
    sample space where proportions are either zero or one.
    Currently the p-value when (p1, p2) is either (0, 0) or (1, 1) is one, i.e.
    it is never in the rejection region. This implies that the pvalue is
    always conservative if the both true probabilities approach zero or one.


    TODO: API is undecided
    - what goes in __init__, what in method arguments? e.g. `alternative`
    - return Results class or test class holds results?

    """

    # use a class to store attributes that don't always change
    # instead of nested functions or full arguments

    def __init__(self, count1, nobs1, count2, nobs2, alternative='2-sided'):
        # use yo1, yo2 as local alias for observed counts of successes
        self.count1 = yo1 = count1
        self.count2 = yo2 = count2
        self.count = count1 + count2
        self.nobs1 = n1 = nobs1
        self.nobs2 = n2 = nobs2
        self.nobs = nobs1 + nobs2
        self.alternative = alternative
        self.prob_pooled = (yo1 + yo2) / (n1 + n2)
        self.prop1 = yo1 / n1
        self.prop2 = yo2 / n2
        self.statistic_base, self.pvalue_base = self.chisquare_proportion_indep(yo1, yo2,
                                                 alternative=alternative)


    def statistic(self, y1, y2, prob_var=None, alternative='2-sided'):
        n1, n2 = self.nobs1, self.nobs2
        # n1, n2 from outer scope
        p1 = y1 / n1
        p2 = y2 / n2
        if prob_var is None:
            #use pooled probability for variance, i.e. under H0 p1 = p2
            prob_var = (y1 + y2) / (n1 + n2)


        v = prob_var * (1 - prob_var) * (1. / n1 + 1. / n2)
        v = np.atleast_1d(v)
        v[v == 0] = 1  #TODO: check corner cases
        # Todo what's the stat and pvalue if p1 = 0 and p2 = 1, or reversed?
        # This is well defined since we use pooled variance
        diff = p2 - p1
        if alternative == '2-sided':
            stat = diff**2 / v  # chisquare for two sided
            return stat
        else:
            return diff / np.sqrt(v)


    def chisquare_proportion_indep(self, y1, y2, prob_var=None, alternative='2-sided'):
        stat = self.statistic(y1, y2, prob_var=prob_var, alternative=alternative)
        if alternative == '2-sided':
            pvalue = stats.chi2.sf(stat, 1)
        else:
            pvalue = stats.norm.sf(stat)
        return stat, pvalue


    def test_asymptotic(self):
        return self.statistic_base, self.pvalue_base


    def pvalue_exactdist_mle(self):
        """pvalue based on exact distribution with MLE nuisance parameter

        this is not an exact test.
        The size of the test is close to the nominal value, but not guaranteed
        to be below it, i.e. test can be liberal in some cases and over reject.
        """
        n1, n2 = self.nobs1, self.nobs2
        alternative = self.alternative  # TODO check if it should be argument
        pvo = self.pvalue_base  # TODO check, define attribute rejection set
        # TODO: DRY separate out common code
        ys1 = np.arange(n1 + 1)
        ys2 = np.arange(n2 + 1)
        st, pv =  self.chisquare_proportion_indep(ys1[None, :], ys2[:,None],
                                     prob_var=self.prob_pooled,
                                     alternative=alternative)

        prob1 = stats.binom.pmf(np.arange(n1 + 1), n1, self.prob_pooled)
        prob2 = stats.binom.pmf(np.arange(n2 + 1), n1, self.prob_pooled)

        prob = prob1[None, :] * prob2[:, None]
        pvemp = prob[pv < pvo].sum()

        return pvemp


    def pvalue_exact_sup(self, grid=None):
        """pvalue based on max of pvalues over nuisance parameter

        This is an exact pvalue and maintains and guarantees the size.

        `grid` argument is currently not used.
        `grid` can be None, array, int or tuple
        If None, then the default method is used, currently a grid.
        If int, then the integer specifies the number of grid points.
        If grid is a tuple, then either the values are integers for linspace or
        ('bb', alpha, n_points) for Berger, Boos limitats for the maximum.

        TODO: replace pure grid search by coarse grid plus local optimize
        check diagnostic plot to see whether there are spikes
        """
        n1, n2 = self.nobs1, self.nobs2  # local alias
        alternative = self.alternative  # TODO check if it should be argument
        use_bb = False
        bb_correct = 0
        res = []
        if grid is None:
            grid = np.linspace(0.001, 0.999, 1001)
        elif isinstance(grid, tuple):
            if grid[0] == 'bb':
                use_bb = True
                y = self.count1 + self.count2
                nobs = self.nobs1 + self.nobs2
                alpha = grid[1]
                bb_correct = alpha
                if len(grid) > 2:
                    n_points = grid[2]
                else:
                    n_points = 1001
                import statsmodels.stats.proportion as smprop
                # use Clopper-Pearson interval
                ci = smprop.proportion_confint(y, nobs, method = 'beta',
                                               alpha=alpha)
                grid = np.linspace(ci[0], ci[1], n_points)

        grid = np.concatenate(([self.prob_pooled], grid))
        #TODO: store rejection region and use for ys1, ys2
        ys1 = np.arange(n1 + 1)
        ys2 = np.arange(n2 + 1)
        pvo = self.pvalue_base  # TODO check, define attribute rejection set

        for prob0 in grid:
            #print(prob,)
            st, pv =  self.chisquare_proportion_indep(ys1[None, :], ys2[:,None],
                                             #prob_var=prob0,  # trying out, doesn't work well
                                             alternative=alternative)

            prob1 = stats.binom.pmf(ys1, n1, prob0)
            prob2 = stats.binom.pmf(ys2, n2, prob0)

            prob = prob1[None, :] * prob2[:, None]
            #TODO: store rejection region
            reject_mask = pv <= pvo  # TODO move outside of loop
            pvemp = prob[reject_mask].sum()
            #print(pvemp)
            res.append([prob0, pvemp])

        res =np.array(res)
        pvm_ind = res[:,1].argmax(0)

        return res[pvm_ind, 1] + bb_correct, res[pvm_ind, 0], pvm_ind, res
