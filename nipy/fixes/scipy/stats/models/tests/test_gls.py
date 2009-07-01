'''
Test functions for models.regression.GLS

'''

#TODO: Should this be put in test_regression?

# Heteroskedasticity is known up to a multiplicative constant.

import models
import numpy as np
from numpy.testing import *
from models.functions import add_constant

class TestRegression(TestCase):

    def test_gls_serial(self): # TEST FOR AR1 Errors
        from models.datasets.longley.data import load
        # From R Script
        R_params = (6.738948e-02, -4.742739e-01, 9.489888e+04)
        R_bse = (1.086675e-02, 1.557265e-01, 1.415760e+04 )
        data = load()
        exog = add_constant(np.column_stack((data.exog[:,1],data.exog[:,4])))
        tmp_results = models.regression.OLS(data.endog, exog).fit()
        rho = np.corrcoef(tmp_results.resid[1:],tmp_results.resid[:-1])[0][1] # by assumption
        rows = np.arange(1,17).reshape(16,1)*np.ones((16,16))
        cols = np.arange(1,17)*np.ones((16,16))
        sigma = rho**np.fabs((rows-cols))
        GLS_results = models.regression.GLS(data.endog, exog, sigma=sigma).fit()
        assert_almost_equal(GLS_results.params, R_params, 1) # intercept only to one decimal...
                                                             # how to handle these case?
#Changing the bit about using whitened design and response for resids fixed this...
#Changing regression line 58-ish about the residuals fixes this.
#Models should be take care of their own residuals.  Since GLM is a transformation, it should right
#it's own?
        assert_almost_equal(GLS_results.bse, R_bse, 2)

        # compare to the standard errors computed by hand like in R...
        # the residuals are also wrong, why do we whiten the outcome variable to compute
        # residuals, this is surely a mistake
 #       res = data.endog - GLS_results.predict
 #       sigma_hat = np.sqrt(np.sum(res**2)/GLS_results.df_resid)
 #       xpxi = np.linalg.inv(np.dot(np.dot(exog.T,np.linalg.inv(sigma)),exog))
 #       GOOD_bse = np.sqrt(np.diag(xpxi))*sigma_hat
 #       assert_almost_equal(GOOD_bse, R_bse, 2) # two decimals because of the intercept

#    def test_gls_var(self):
#        X1 = np.random.uniform(0,10,300)
#        X2 = np.random.uniform(0,100,300)
#        beta = np.array((3.5, 4.5, 1.25))
#        design = np.column_stack((np.ones((300,1)),X1,X2))
#        Y = np.dot(design,beta) + np.random.normal(0, 1.25*(X2),300)

    def test_gls_scalar(self):
        '''
        If no argument for sigma is given to GLS, then GLS equals OLS
        '''
        from models.datasets.longley.data import load
        data = load()
        data.exog = add_constant(data.exog)
        ols_res = models.regression.OLS(data.endog, data.exog).fit()
        gls_res = models.regression.GLS(data.endog, data.exog).fit()
        assert_equal(ols_res.params,gls_res.params)
        assert_equal(ols_res.bse,gls_res.bse)

if __name__=="__main__":
    run_module_suite()

