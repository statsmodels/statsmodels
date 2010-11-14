# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 08:21:41 2010

Author: josef-pktd
License: BSD (3-clause)
"""


import numpy as np
from numpy.testing import assert_array_almost_equal
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tools import pca
from scikits.statsmodels.sandbox.tools.cross_val import LeaveOneOut

#converting example Principal Component Regression to a class
#from sandbox/example_pca_regression.py


class FactorModelUnivariate(object):
    '''

    Todo:
    check treatment of const, make it optional ?
        add hasconst (0 or 1), needed when selecting nfact+hasconst
    options are arguments in calc_factors, should be more public instead
    cross-validation is slow for large number of observations
    '''
    def __init__(self, endog, exog):
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)


    def calc_factors(self, x=None, keepdim=0, addconst=True):
        if x is None:
            x = self.exog
        else:
            x = np.asarray(x)
        xred, fact, evals, evecs  = pca(x, keepdim=keepdim, normalize=1)
        self.exog_reduced = xred
        self.factors = fact
        self.factors_wconst = sm.add_constant(fact, prepend=True)
        self.evals = evals
        self.evecs = evecs

    def fit_fixed_nfact(self, nfact):
        if not hasattr(self, 'factors_wconst'):
            self.calc_factors()
        return sm.OLS(self.endog, self.factors_wconst[:,:nfact+1]).fit()

    def fit_find_nfact(self, maxfact=None):
        #print 'OLS on Factors'
        if not hasattr(self, 'factors_wconst'):
            self.calc_factors()

        if maxfact is None:
            maxfact = self.factors.shape[1]

        #temporary safety
        maxfact = min(maxfact, 3)

        y0 = self.endog
        results = []
        #xred, fact, eva, eve  = pca(x0, keepdim=0, normalize=1)
        for k in range(0, maxfact+1): #x0.shape[1]+1):
            #xred, fact, eva, eve  = pca(x0, keepdim=k, normalize=1)
            # this is faster and same result
            fact_wconst = self.factors_wconst[:,:k+1]
            res = sm.OLS(y0, fact_wconst).fit()
        ##    print 'k =', k
        ##    print res.params
        ##    print 'aic:  ', res.aic
        ##    print 'bic:  ', res.bic
        ##    print 'llf:  ', res.llf
        ##    print 'R2    ', res.rsquared
        ##    print 'R2 adj', res.rsquared_adj
            prederr2 = 0.
            for inidx, outidx in LeaveOneOut(len(y0)):
                resl1o = sm.OLS(y0[inidx], fact_wconst[inidx,:]).fit()
                #print data.endog[outidx], res.model.predict(data.exog[outidx,:]),
                prederr2 += (y0[outidx] - resl1o.model.predict(fact_wconst[outidx,:]))**2.
            results.append([k, res.aic, res.bic, res.rsquared_adj, prederr2])

        self.results_find_nfact = np.array(results)

    def summary_find_nfact(self):
        if not hasattr(self, 'results_find_nfact'):
            self.fit_find_nfact()


        results = self.results_find_nfact
        sumstr = ''
        sumstr += '\n' + 'Best result for k, by AIC, BIC, R2_adj, L1O'
        best = np.r_[(np.argmin(results[:,1:3],0), np.argmax(results[:,3],0),
                     np.argmin(results[:,-1],0))]
        sumstr += '\n' + ' '*19 + '%5d %4d %6d %5d' % tuple(best)

        from scikits.statsmodels.iolib.table import (SimpleTable, default_txt_fmt,
                                default_latex_fmt, default_html_fmt)

        headers = 'k, AIC, BIC, R2_adj, L1O'.split(', ')
        numformat = ['%6d'] + ['%10.3f']*4 #'%10.4f'
        txt_fmt1 = dict(data_fmts = numformat)
        tabl = SimpleTable(results, headers, None, txt_fmt=txt_fmt1)

        sumstr += '\n' + "PCA regression on simulated data,"
        sumstr += '\n' + "DGP: 2 factors and 4 explanatory variables"
        sumstr += '\n' + tabl.__str__()
        sumstr += '\n' + "Notes: k is number of components of PCA,"
        sumstr += '\n' + "       constant is added additionally"
        sumstr += '\n' + "       k=0 means regression on constant only"
        sumstr += '\n' + "       L1O: sum of squared prediction errors for leave-one-out"
        return sumstr


if __name__ == '__main__':

    examples = [1]
    if 1 in examples:
        nobs = 500
        f0 = np.c_[np.random.normal(size=(nobs,2)), np.ones((nobs,1))]
        f2xcoef = np.c_[np.repeat(np.eye(2),2,0),np.arange(4)[::-1]].T
        f2xcoef = np.array([[ 1.,  1.,  0.,  0.],
                            [ 0.,  0.,  1.,  1.],
                            [ 3.,  2.,  1.,  0.]])
        f2xcoef = np.array([[ 0.1,  3.,  1.,    0.],
                            [ 0.,  0.,  1.5,   0.1],
                            [ 3.,  2.,  1.,    0.]])
        x0 = np.dot(f0, f2xcoef)
        x0 += 0.1*np.random.normal(size=x0.shape)
        ytrue = np.dot(f0,[1., 1., 1.])
        y0 = ytrue + 0.1*np.random.normal(size=ytrue.shape)

        mod = FactorModelUnivariate(y0, x0)
        print mod.summary_find_nfact()



