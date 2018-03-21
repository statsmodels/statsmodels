# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 20:20:16 2011

Author: Josef Perktold
License: BSD-3


TODO:
check orientation, size and alpha should be increasing for interp1d,
but what is alpha? can be either sf or cdf probability
change it to use one consistent notation

check: instead of bound checking I could use the fill-value of the interpolators


"""
from __future__ import print_function
from statsmodels.compat.python import range
import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly


class TableDist(object):
    '''Distribution, critical values and p-values from tables

    currently only 1 extra parameter, e.g. sample size

    Parameters
    ----------
    alpha : array_like, 1d
        probabiliy in the table, could be either sf (right tail) or cdf (left
        tail)
    size : array_like, 1d
        second paramater in the table
    crit_table : array_like, 2d
        array with critical values for sample size in rows and probability in
        columns

    Notes
    -----
    size and alpha should be increasing


    '''

    def __init__(self, alpha, size, crit_table):
        self.alpha = np.asarray(alpha)
        self.size = np.asarray(size)
        self.crit_table = np.asarray(crit_table)

        self.n_alpha = len(alpha)
        self.signcrit = np.sign(np.diff(self.crit_table, 1).mean())
        if self.signcrit > 0: #increasing
            self.critv_bounds = self.crit_table[:,[0,1]]
        else:
            self.critv_bounds = self.crit_table[:,[1,0]]

    @cache_readonly
    def polyn(self):
        polyn = [interp1d(self.size, self.crit_table[:,i])
                               for i in range(self.n_alpha)]
        return polyn

    @cache_readonly
    def poly2d(self):
        #check for monotonicity ?
        #fix this, interp needs increasing
        poly2d = interp2d(self.size, self.alpha, self.crit_table)
        return poly2d

    @cache_readonly
    def polyrbf(self):
        xs, xa = np.meshgrid(self.size.astype(float), self.alpha)
        polyrbf = Rbf(xs.ravel(), xa.ravel(), self.crit_table.T.ravel(),function='linear')
        return polyrbf

    def _critvals(self, n):
        '''rows of the table, linearly interpolated for given sample size

        Parameters
        ----------
        n : float
            sample size, second parameter of the table

        Returns
        -------
        critv : ndarray, 1d
            critical values (ppf) corresponding to a row of the table

        Notes
        -----
        This is used in two step interpolation, or if we want to know the
        critical values for all alphas for any sample size that we can obtain
        through interpolation

        '''
        return np.array([p(n) for p in self.polyn])

    def prob(self, x, n):
        '''find pvalues by interpolation, eiter cdf(x) or sf(x)

        returns extrem probabilities, 0.001 and 0.2, for out of range

        Parameters
        ----------
        x : array_like
            observed value, assumed to follow the distribution in the table
        n : float
            sample size, second parameter of the table

        Returns
        -------
        prob : arraylike
            This is the probability for each value of x, the p-value in
            underlying distribution is for a statistical test.

        '''
        critv = self._critvals(n)
        alpha = self.alpha
#        if self.signcrit == 1:
#            if x < critv[0]:  #generalize: ? np.sign(x - critvals[0]) == self.signcrit:
#                return alpha[0]
#            elif x > critv[-1]:
#                return alpha[-1]
#        elif self.signcrit == -1:
#            if x > critv[0]:
#                return alpha[0]
#            elif x < critv[-1]:
#                return alpha[-1]

        if self.signcrit < 1:
            #reverse if critv is decreasing
            critv, alpha = critv[::-1], alpha[::-1]

        #now critv is increasing
        if np.size(x) == 1:
            if x < critv[0]:
                return alpha[0]
            elif x > critv[-1]:
                return alpha[-1]
            return interp1d(critv, alpha)(x)[()]
        else:
            #vectorized
            cond_low = (x < critv[0])
            cond_high = (x > critv[-1])
            cond_interior = ~np.logical_or(cond_low, cond_high)

            probs = np.nan * np.ones(x.shape) #mistake if nan left
            probs[cond_low] = alpha[0]
            probs[cond_low] = alpha[-1]
            probs[cond_interior] = interp1d(critv, alpha)(x[cond_interior])

            return probs


    def crit2(self, prob, n):
        '''returns interpolated quantiles, similar to ppf or isf

        this can be either cdf or sf depending on the table, twosided?

        this doesn't work, no more knots warning

        '''
        return self.poly2d(n, prob)


    def crit(self, prob, n):
        '''returns interpolated quantiles, similar to ppf or isf

        use two sequential 1d interpolation, first by n then by prob

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob

        '''
        prob = np.asarray(prob)
        alpha = self.alpha
        critv = self._critvals(n)

        #vectorized
        cond_ilow = (prob > alpha[0])
        cond_ihigh = (prob < alpha[-1])
        cond_interior = np.logical_or(cond_ilow, cond_ihigh)

        #scalar
        if prob.size == 1:
            if cond_interior:
                return interp1d(alpha, critv)(prob)
            else:
                return np.nan

        #vectorized
        quantile = np.nan * np.ones(prob.shape) #nans for outside
        quantile[cond_interior] = interp1d(alpha, critv)(prob[cond_interior])
        return quantile

    def crit3(self, prob, n):
        '''returns interpolated quantiles, similar to ppf or isf

        uses Rbf to interpolate critical values as function of `prob` and `n`

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob, returns nan for arguments
            that are outside of the table bounds

        '''
        prob = np.asarray(prob)
        alpha = self.alpha

        #vectorized
        cond_ilow = (prob > alpha[0])
        cond_ihigh = (prob < alpha[-1])
        cond_interior = np.logical_or(cond_ilow, cond_ihigh)

        #scalar
        if prob.size == 1:
            if cond_interior:
                return self.polyrbf(n, prob)
            else:
                return np.nan

        #vectorized
        quantile = np.nan * np.ones(prob.shape) #nans for outside

        quantile[cond_interior] = self.polyrbf(n, prob[cond_interior])
        return quantile
