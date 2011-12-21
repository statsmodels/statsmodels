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


class GLSAR(GLS):
    """
    A regression model with an AR(p) covariance structure.

    The linear autoregressive process of order p--AR(p)--is defined as:
        TODO

    Examples
    --------
    >>> import scikits.statsmodels.api as sm
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> Y = [1,3,4,5,8,10,9]
    >>> model = sm.GLSAR(Y, X, rho=2)
    >>> for i in range(6):
    ...    results = model.fit()
    ...    print "AR coefficients:", model.rho
    ...    rho, sigma = sm.regression.yule_walker(results.resid,
    ...                 order=model.order)
    ...    model = sm.GLSAR(Y, X, rho)
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.620642   -0.88654567]
    AR coefficients: [-0.61887622 -0.88137957]
    AR coefficients: [-0.61894058 -0.88152761]
    AR coefficients: [-0.61893842 -0.88152263]
    >>> results.params
    array([ 1.58747943, -0.56145497])
    >>> results.tvalues
    array([ 30.796394  ,  -2.66543144])
    >>> print results.t_test([0,1])
    <T test: effect=-0.56145497223945595, sd=0.21064318655324663, t=-2.6654314408481032, p=0.022296117189135045, df_denom=5>
    >>> import numpy as np
    >>> print(results.f_test(np.identity(2)))
    <F test: F=2762.4281271616205, p=2.4583312696e-08, df_denom=5, df_num=2>

    Or, equivalently

    >>> model2 = sm.GLSAR(Y, X, rho=2)
    >>> res = model2.iterative_fit(maxiter=6)
    >>> model2.rho
    array([-0.61893842, -0.88152263])

    Notes
    -----
    GLSAR is considered to be experimental.
    """
    def __init__(self, endog, exog=None, rho=1):
        if isinstance(rho, np.int):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0,1]:
                raise ValueError("AR parameters must be a scalar or a vector")
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        if exog is None:
            #JP this looks wrong, should be a regression on constant
            #results for rho estimate now identical to yule-walker on y
            #super(AR, self).__init__(endog, add_constant(endog))
            super(GLSAR, self).__init__(endog, np.ones((endog.shape[0],1)))
        else:
            super(GLSAR, self).__init__(endog, exog)

    def iterative_fit(self, maxiter=3):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated simultaneously.

        Parameters
        ----------
        maxiter : integer, optional
            the number of iterations
        """
#TODO: update this after going through example.
        for i in range(maxiter-1):
            self.initialize()
            results = self.fit()
            self.rho, _ = yule_walker(results.resid,
                                      order=self.order, df=None)
        #why not another call to self.initialize
        results = self.fit() #final estimate
        return results # add missing return

    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure.

        Parameters
        ----------
        X : array-like
            The data to be whitened

        Returns
        -------
        TODO
        """
#TODO: notation for AR process
        X = np.asarray(X, np.float64)
        _X = X.copy()
        #dimension handling is not DRY
        # I think previous code worked for 2d because of single index rows in np
        if X.ndim == 1:
            for i in range(self.order):
                _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
            return _X[self.order:]
        elif X.ndim == 2:
            for i in range(self.order):
                _X[(i+1):,:] = _X[(i+1):,:] - self.rho[i] * X[0:-(i+1),:]
                return _X[self.order:,:]


if __name__ == '__main__':
    pass
