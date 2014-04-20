# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 17:12:53 2014

author: Josef Perktold

"""

import numpy as np
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def test_predict_se():
    # this test doesn't use reference values
    # checks conistency across options, and compares to direct calculation

    # generate dataset
    nsample = 50
    x1 = np.linspace(0, 20, nsample)
    x = np.c_[x1, (x1 - 5)**2, np.ones(nsample)]
    np.random.seed(0)#9876789) #9876543)
    beta = [0.5, -0.01, 5.]
    y_true2 = np.dot(x, beta)
    w = np.ones(nsample)
    w[nsample * 6. / 10:] = 3
    sig = 0.5
    y2 = y_true2 + sig * w * np.random.normal(size=nsample)
    x2 = x[:,[0,2]]

    # estimate OLS
    res2 = OLS(y2, x2).fit()

    #direct calculation
    covb = res2.cov_params()
    predvar = res2.mse_resid + (x2 * np.dot(covb, x2.T).T).sum(1)
    predstd = np.sqrt(predvar)

    prstd, iv_l, iv_u = wls_prediction_std(res2)
    np.testing.assert_almost_equal(prstd, predstd, 15)

    #stats.t.isf(0.05/2., 50 - 2)
    q = 2.0106347546964458
    ci_half = q * predstd
    np.testing.assert_allclose(iv_u, res2.fittedvalues + ci_half, rtol=1e-12)
    np.testing.assert_allclose(iv_l, res2.fittedvalues - ci_half, rtol=1e-12)

    prstd, iv_l, iv_u = wls_prediction_std(res2, x2[:3,:])
    np.testing.assert_equal(prstd, prstd[:3])
    np.testing.assert_allclose(iv_u, res2.fittedvalues[:3] + ci_half[:3],
                               rtol=1e-12)
    np.testing.assert_allclose(iv_l, res2.fittedvalues[:3] - ci_half[:3],
                               rtol=1e-12)


    # check WLS
    res3 = WLS(y2, x2, 1. / w).fit()

    #direct calculation
    covb = res3.cov_params()
    predvar = res3.mse_resid * w + (x2 * np.dot(covb, x2.T).T).sum(1)
    predstd = np.sqrt(predvar)

    prstd, iv_l, iv_u = wls_prediction_std(res3)
    np.testing.assert_almost_equal(prstd, predstd, 15)

    #stats.t.isf(0.05/2., 50 - 2)
    q = 2.0106347546964458
    ci_half = q * predstd
    np.testing.assert_allclose(iv_u, res3.fittedvalues + ci_half, rtol=1e-12)
    np.testing.assert_allclose(iv_l, res3.fittedvalues - ci_half, rtol=1e-12)

    # testing shapes of exog
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1:,:], weights=3.)
    np.testing.assert_equal(prstd, prstd[-1])
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1,:], weights=3.)
    np.testing.assert_equal(prstd, prstd[-1])

    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-2:,:], weights=3.)
    np.testing.assert_equal(prstd, prstd[-2:])

    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-2:,:], weights=[3, 3])
    np.testing.assert_equal(prstd, prstd[-2:])

    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[:3,:])
    np.testing.assert_equal(prstd, prstd[:3])
    np.testing.assert_allclose(iv_u, res3.fittedvalues[:3] + ci_half[:3],
                               rtol=1e-12)
    np.testing.assert_allclose(iv_l, res3.fittedvalues[:3] - ci_half[:3],
                               rtol=1e-12)


    #use wrong size for exog
    #prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1,0], weights=3.)
    np.testing.assert_raises(ValueError, wls_prediction_std, res3, x2[-1,0],
                             weights=3.)

    # check some weight values
    sew1 = wls_prediction_std(res3, x2[-3:,:])[0]**2
    for wv in np.linspace(0.5, 3, 5):

        sew = wls_prediction_std(res3, x2[-3:,:], weights=1. / wv)[0]**2
        np.testing.assert_allclose(sew, sew1 + res3.scale * (wv - 1))
