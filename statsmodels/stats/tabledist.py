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



if __name__ == '__main__':

    '''
    example Lilliefors test for normality
    An Analytic Approximation to the Distribution of Lilliefors's Test Statistic for Normality
    Author(s): Gerard E. Dallal and Leland WilkinsonSource: The American Statistician, Vol. 40, No. 4 (Nov., 1986), pp. 294-296Published by: American Statistical AssociationStable URL: http://www.jstor.org/stable/2684607 .
    '''

    #for this test alpha is sf probability, i.e. right tail probability

    alpha = np.array([ 0.2  ,  0.15 ,  0.1  ,  0.05 ,  0.01 ,  0.001])[::-1]
    size = np.array([ 4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
                     16,  17,  18,  19,  20,  25,  30,  40, 100, 400, 900], float)

    #critical values, rows are by sample size, columns are by alpha
    crit_lf = np.array(   [[303, 321, 346, 376, 413, 433],
                           [289, 303, 319, 343, 397, 439],
                           [269, 281, 297, 323, 371, 424],
                           [252, 264, 280, 304, 351, 402],
                           [239, 250, 265, 288, 333, 384],
                           [227, 238, 252, 274, 317, 365],
                           [217, 228, 241, 262, 304, 352],
                           [208, 218, 231, 251, 291, 338],
                           [200, 210, 222, 242, 281, 325],
                           [193, 202, 215, 234, 271, 314],
                           [187, 196, 208, 226, 262, 305],
                           [181, 190, 201, 219, 254, 296],
                           [176, 184, 195, 213, 247, 287],
                           [171, 179, 190, 207, 240, 279],
                           [167, 175, 185, 202, 234, 273],
                           [163, 170, 181, 197, 228, 266],
                           [159, 166, 176, 192, 223, 260],
                           [143, 150, 159, 173, 201, 236],
                           [131, 138, 146, 159, 185, 217],
                           [115, 120, 128, 139, 162, 189],
                           [ 74,  77,  82,  89, 104, 122],
                           [ 37,  39,  41,  45,  52,  61],
                           [ 25,  26,  28,  30,  35,  42]])[:,::-1] / 1000.


    lf = TableDist(alpha, size, crit_lf)
    print(lf.prob(0.166, 20), 'should be:', 0.15)
    print('')
    print(lf.crit2(0.15, 20), 'should be:', 0.166, 'interp2d bad')
    print(lf.crit(0.15, 20), 'should be:', 0.166, 'two 1d')
    print(lf.crit3(0.15, 20), 'should be:', 0.166, 'Rbf')
    print('')
    print(lf.crit2(0.17, 20), 'should be in:', (.159, .166), 'interp2d bad')
    print(lf.crit(0.17, 20), 'should be in:', (.159, .166), 'two 1d')
    print(lf.crit3(0.17, 20), 'should be in:', (.159, .166), 'Rbf')
    print('')
    print(lf.crit2(0.19, 20), 'should be in:', (.159, .166), 'interp2d bad')
    print(lf.crit(0.19, 20), 'should be in:', (.159, .166), 'two 1d')
    print(lf.crit3(0.19, 20), 'should be in:', (.159, .166), 'Rbf')
    print('')
    print(lf.crit2(0.199, 20), 'should be in:', (.159, .166), 'interp2d bad')
    print(lf.crit(0.199, 20), 'should be in:', (.159, .166), 'two 1d')
    print(lf.crit3(0.199, 20), 'should be in:', (.159, .166), 'Rbf')
    #testing
    print(np.max(np.abs(np.array([lf.prob(c, size[i]) for i in range(len(size)) for c in crit_lf[i]]).reshape(-1,lf.n_alpha) - lf.alpha)))
    #1.6653345369377348e-16
    print(np.max(np.abs(np.array([lf.crit(c, size[i]) for i in range(len(size)) for c in lf.alpha]).reshape(-1,lf.n_alpha) - crit_lf)))
    #6.9388939039072284e-18)
    print(np.max(np.abs(np.array([lf.crit3(c, size[i]) for i in range(len(size)) for c in lf.alpha]).reshape(-1,lf.n_alpha) - crit_lf)))
    #4.0615705243496336e-12)
    print((np.array([lf.crit3(c, size[i]) for i in range(len(size)) for c in lf.alpha[:-1]*1.1]).reshape(-1,lf.n_alpha-1) < crit_lf[:,:-1]).all())
    print((np.array([lf.crit3(c, size[i]) for i in range(len(size)) for c in lf.alpha[:-1]*1.1]).reshape(-1,lf.n_alpha-1) > crit_lf[:,1:]).all())
    print((np.array([lf.prob(c*0.9, size[i]) for i in range(len(size)) for c in crit_lf[i,:-1]]).reshape(-1,lf.n_alpha-1) > lf.alpha[:-1]).all())
    print((np.array([lf.prob(c*1.1, size[i]) for i in range(len(size)) for c in crit_lf[i,1:]]).reshape(-1,lf.n_alpha-1) < lf.alpha[1:]).all())
    #start at size_idx=2 because of non-monotonicity of lf_crit
    print((np.array([lf.prob(c, size[i]*0.9) for i in range(2,len(size)) for c in crit_lf[i,:-1]]).reshape(-1,lf.n_alpha-1) > lf.alpha[:-1]).all())
