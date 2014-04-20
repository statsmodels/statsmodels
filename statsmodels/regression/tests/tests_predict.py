# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 17:12:53 2014

author: Josef Perktold

"""

import numpy as np
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std



def test_predict_se():
    import statsmodels.api as sm

    # generate dataset
    nsample = 50
    x1 = np.linspace(0, 20, nsample)
    X = np.c_[x1, (x1-5)**2, np.ones(nsample)]
    np.random.seed(0)#9876789) #9876543)
    beta = [0.5, -0.01, 5.]
    y_true2 = np.dot(X, beta)
    w = np.ones(nsample)
    w[nsample*6/10:] = 3
    sig = 0.5
    y2 = y_true2 + sig*w* np.random.normal(size=nsample)
    X2 = X[:,[0,2]]

    # estimate OLS, WLS, (OLS not used in these tests)
    res2 = sm.OLS(y2, X2).fit()
    res3 = sm.WLS(y2, X2, 1./w).fit()

    #direct calculation
    covb = res3.cov_params()
    predvar = res3.mse_resid*w + (X2 * np.dot(covb,X2.T).T).sum(1)
    predstd = np.sqrt(predvar)


    prstd, iv_l, iv_u = wls_prediction_std(res3)
    np.testing.assert_almost_equal(predstd, prstd, 15)

    # testing shapes of exog
    prstd, iv_l, iv_u = wls_prediction_std(res3, X2[-1:,:], weights=3.)
    np.testing.assert_equal( prstd[-1], prstd)
    prstd, iv_l, iv_u = wls_prediction_std(res3, X2[-1,:], weights=3.)
    np.testing.assert_equal( prstd[-1], prstd)
    #use wrong size for exog
    #prstd, iv_l, iv_u = wls_prediction_std(res3, X2[-1,0], weights=3.)
    np.testing.assert_raises(ValueError, wls_prediction_std, res3, X2[-1,0], weights=3.)

