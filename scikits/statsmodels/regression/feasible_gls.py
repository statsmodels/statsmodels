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




if __name__ == '__main__':
    pass
