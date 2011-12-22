# -*- coding: utf-8 -*-
"""

Created on Tue Dec 20 20:24:20 2011

Author: Josef Perktold
License: BSD-3

"""


import numpy as np
import scikits.statsmodels.base.model as base
from scikits.statsmodels.regression.linear_model import OLS, GLS, WLS, RegressionResults


def atleast_2dcols(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]
    return x


class GLSHet2(GLS):
    '''WLS with heteroscedasticity that depends on explanatory variables

    note: mixing GLS sigma and weights for heteroscedasticity might not make
    sense

    I think rewriting following the pattern of GLSAR is better
    stopping criteria: improve in GLSAR also, e.g. change in rho

    '''


    def __init__(self, endog, exog, exog_var, sigma=None):
        self.exog_var = atleast_2dcols(exog_var)
        super(self.__class__, self).__init__(endog, exog, sigma=sigma)


    def fit(self, lambd=1.):
        #maybe iterate
        #preliminary estimate
        res_gls = GLS(self.endog, self.exog, sigma=self.sigma).fit()
        res_resid = OLS(res_gls.resid**2, self.exog_var).fit()
        #or  log-link
        #res_resid = OLS(np.log(res_gls.resid**2), self.exog_var).fit()
        #here I could use whiten and current instance instead of delegating
        #but this is easier
        #see pattern of GLSAR, calls self.initialize and self.fit
        res_wls = WLS(self.endog, self.exog, weights=1./res_resid.fittedvalues).fit()

        res_wls._results.results_residual_regression = res_resid
        return res_wls


class GLSHet(WLS):
    """
    A regression model with an estimated heteroscedasticity.



    Notes
    -----
    GLSHet is considered to be experimental.

    subclassing WLS for now

    if weights is given then it is used as the weight array for the first
    estimation

    TODO
    check namings and convention for link function

    """
    def __init__(self, endog, exog, exog_var=None, weights=None, link=None):
        self.exog_var = atleast_2dcols(exog_var)
        if weights is None:
            weights = np.ones(endog.shape)
        if link is not None:
            self.link = link
            self.invlink = None   #will raise exception for now
        else:
            self.link = lambda x: x  #no transformation
            self.linkinv = lambda x: x

        super(self.__class__, self).__init__(endog, exog, weights=weights)

    def iterative_fit(self, maxiter=3):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have heteroscedastic errors sigma_i = Z*gamma.
        The variance is estimated by OLS regression of the residuals on Z.

        Parameters
        ----------
        maxiter : integer, optional
            the number of iterations


        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maciter>2: iteratively estimate and update the weights

        TODO:
        the maxiter counting looks wrong,
        3 is the first that estimates with an updated weight matrix ???
        silly mistake initialize instead of calling initialize()
        """

        res_resid = None
        for i in range(maxiter):
            #pinv_wexog is cached
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog

            self.initialize()
            results = self.fit()
            if not i == maxiter-1:  #skip for last iteration
                res_resid = OLS(self.link(results.resid**2), self.exog_var).fit()
                self.weights = 1./self.linkinv(res_resid.fittedvalues)
                self.weights /= self.weights.max()  #not required
                print 'in iter', i, self.weights.var()
        #why not another call to self.initialize - not done in GLSAR
#        if hasattr(self, 'pinv_wexog'):
#            del self.pinv_wexog
#        print 'initialize last time'
#        self.initialize()
#        results = self.fit() #final estimate
        #note results is the wrapper, results._results is the results instance
        results._results.results_residual_regression = res_resid
        return results



if __name__ == '__main__':
    pass
