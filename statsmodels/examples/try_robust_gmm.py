# -*- coding: utf-8 -*-
"""

Created on Wed Mar 12 11:28:20 2014

Author: Josef Perktold
"""

import numpy as np

#import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.base.model import GenericLikelihoodModel

import statsmodels.robust.norms as rnorms
import statsmodels.robust.scale as rscale
import statsmodels.robust.robust_linear_model as rlm

import statsmodels.miscmodels.robust_genericmle as robnl


norm = rnorms.HuberT()

#stats.norm.expect(lambda x, *args: rnorms.TukeyBiweight(1).rho(x))
scale_bias = 0.10903642477287441
meef_scale = lambda x: rnorms.TukeyBiweight(1).rho(x)-scale_bias + 0.16666



examples = ['ex1', 'exp'][:]

if 'ex1' in examples:
    seed = 976537
    np.random.seed(seed)
    nobs, k_vars = 500, 5
    n_outliers = 1
    sig_e = 0.5
    beta = np.ones(k_vars)
    beta[-2:] *= 0.25
    exog = add_constant(np.random.uniform(0, 1, size=(nobs, k_vars - 1)))
    y_true = np.dot(exog, beta)
    endog = y_true + sig_e * np.random.randn(nobs)
    endog[-n_outliers:] += 100

    start_params = np.array([ 1.16675244,  1.24129011,  0.75537835,
                             0.30819684,  0.1113282, -2 ])
    mod = robnl._RobustGMM(endog, exog, instrument=exog)
    res = mod.fit(start_params=start_params, maxiter=1,
                  inv_weights=np.eye(len(start_params)))
    #mod.data.xnames.append('scale')
    mod.exog_names.append('scale')
    print(res.summary())


if 'exp' in examples:
    # example from try_robust_genericmle.py
    func_m = lambda beta, x: np.exp(beta[0] + np.dot(x, beta[1:]))
    func_m = lambda beta, x: np.exp(np.dot(x, beta))

    nobs2 = 100
    sig_e = 2
    seed = np.random.randint(999999)
    #seed = 456753  # just typed
    #seed = 868149  # very good fit for GMM
    #seed = 456621  # very bad fit for GMM
    print('seed', seed)
    np.random.seed(seed)
    x = np.random.uniform(0, 10, size=nobs2)
    x.sort()
    exog = np.column_stack((np.ones(x.shape[0]), x))
    beta_m = [-5, 0.8]
    y_true = func_m(beta_m, exog)
    y = y_true + sig_e * np.random.randn(nobs2)
    y = y_true  + (1 + 0.2 * y_true) * np.random.randn(nobs2)
    # GMM doesn't look very stable, can get easily pulled towards outliers
    # I think the problem is that the weight matrix is not robust
    y[-20:-5:3] += 20  #15
    endog = y

    from scipy.optimize import leastsq
    res_cf = leastsq(lambda p: y - func_m(p,exog), x0=[0.5,0.5])
    fitted_ls = func_m(res_cf[0], exog)

    start_params = [-5, 0.75, 1]
    mod = robnl._ExpRobustGMM(endog, exog, instrument=exog)
    res = mod.fit(start_params=start_params, maxiter=2,
                  inv_weights=np.eye(len(start_params)), optim_method='nm')
    res = mod.fit(start_params=res.params, maxiter=1, inv_weights=np.eye(len(start_params)))
    mod.data.xnames.append('scale')
    print(res.summary())
    fittedvalues = res.predict()

    import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot()
    plt.plot(x, y, 'o')
    plt.plot(x, y_true, '-', color='b', lw=2, alpha = 0.75, label='true')
    plt.plot(x, fittedvalues, '-', lw=2, alpha = 0.75, label='fit_robust')
    plt.plot(x, fitted_ls, '-', lw=2, alpha = 0.75, label='fit_leastsq')
    plt.legend(loc='upper left')
    plt.title('Robust Nonlinear M-Estimation - outliers')
    #plt.title('Robust Nonlinear M-Estimation - no outliers')
    plt.show()
