# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:34:59 2010

Author: josef-pktd
"""

import numpy as np
from scipy import stats, factorial
import scikits.statsmodels as sm
from scikits.statsmodels.model import GenericLikelihoodModel

def maxabs(arr1, arr2):
    return np.max(np.abs(arr1 - arr2))

def maxabsrel(arr1, arr2):
    return np.max(np.abs(arr2 / arr1 - 1))

class NonlinearDeltaCov(object):
    '''Asymptotic covariance by Deltamethod

    the function is designed for 2d array, with rows equal to
    the number of equations and columns equal to the number
    of parameters. 1d params work by chance ?

    fun: R^{m*k) -> R^{m}  where m is number of equations and k is
    the number of parameters.

    equations follow Greene

    '''
    def __init__(self, fun, params, cov_params):
        self.fun = fun
        self.params = params
        self.cov_params = cov_params

    def grad(self, params=None, **kwds):
        if params is None:
            params = self.params
        kwds.setdefault('epsilon', 1e-4)
        from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime1
        return approx_fprime1(params, self.fun, **kwds)

    def cov(self):
        g = self.grad()
        covar = np.dot(np.dot(g, self.cov_params), g.T)
        return covar

    def expected(self):
        # rename: misnomer, this is the MLE of the fun
        return self.fun(self.params)

    def wald(self, value):
        m = self.expected()
        v = self.cov()
        df = np.size(m)
        diff = m - value
        lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
        return lmstat, stats.chi2.sf(lmstat, df)




class MyPoisson(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    '''

    # copied from discretemod.Poisson
    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        --------
        .. math :: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        return np.exp(XB) -  endog*XB + np.log(factorial(endog))


if __name__ == '__main__':

    #Example:
    np.random.seed(98765678)
    nobs = 1000
    rvs = np.random.randn(nobs,6)
    data_exog = rvs
    data_exog = sm.add_constant(data_exog)
    xbeta = 1 + 0.1*rvs.sum(1)
    data_endog = np.random.poisson(np.exp(xbeta))
    #print data_endog

    modp = MyPoisson(data_endog, data_exog)
    resp = modp.fit()
    print resp.params
    print resp.bse

    from scikits.statsmodels.discretemod import Poisson
    resdp = Poisson(data_endog, data_exog).fit()
    print '\ncompare with discretemod'
    print 'compare params'
    print resdp.params - resp.params
    print 'compare bse'
    print resdp.bse - resp.bse

    gmlp = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
    resgp = gmlp.fit()
    ''' this creates a warning, bug bse is double defined ???
    c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\decorators.py:105: CacheWriteWarning: The attribute 'bse' cannot be overwritten
      warnings.warn(errmsg, CacheWriteWarning)
    '''
    print '\ncompare with GLM'
    print 'compare params'
    print resgp.params - resp.params
    print 'compare bse'
    print resgp.bse - resp.bse

    lam = np.exp(np.dot(data_exog, resp.params))
    '''mean of Poisson distribution'''
    predmean = stats.poisson.stats(lam,moments='m')
    print np.max(np.abs(predmean - lam))

    fun = lambda params: np.exp(np.dot(data_exog.mean(0), params))

    lamcov = NonlinearDeltaCov(fun, resp.params, resdp.cov_params())
    print lamcov.cov().shape
    print lamcov.cov()

    print 'analytical'
    xm = data_exog.mean(0)
    print np.dot(np.dot(xm, resdp.cov_params()), xm.T) * \
            np.exp(2*np.dot(data_exog.mean(0), resp.params))

    ''' cov of linear transformation of params
    >>> np.dot(np.dot(xm, resdp.cov_params()), xm.T)
    0.00038904130127582825
    >>> resp.cov_params(xm)
    0.00038902428119179394
    >>> np.dot(np.dot(xm, resp.cov_params()), xm.T)
    0.00038902428119179394
    '''

    print lamcov.wald(1.)
    print lamcov.wald(2.)
    print lamcov.wald(2.6)

    do_bootstrap = False
    if do_bootstrap:
        m,s,r = resp.bootstrap(method='newton')
        print m
        print s
        print resp.bse


    print '\ncomparison maxabs, masabsrel'
    print 'discr params', maxabs(resdp.params, resp.params), maxabsrel(resdp.params, resp.params)
    print 'discr bse   ', maxabs(resdp.bse, resp.bse), maxabsrel(resdp.bse, resp.bse)
    print 'discr bsejac', maxabs(resdp.bse, resp.bsejac), maxabsrel(resdp.bse, resp.bsejac)
    print 'discr bsejhj', maxabs(resdp.bse, resp.bsejhj), maxabsrel(resdp.bse, resp.bsejhj)
    print
    print 'glm params  ', maxabs(resdp.params, resp.params), maxabsrel(resdp.params, resp.params)
    print 'glm bse     ', maxabs(resdp.bse, resp.bse), maxabsrel(resdp.bse, resp.bse)

