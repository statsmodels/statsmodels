# -*- coding: utf-8 -*-
"""

Created on Tue Jan 08 19:03:20 2013

Author: Josef Perktold
"""

from __future__ import print_function


if __name__ == '__main__':

    import numpy as np

    from statsmodels.regression.linear_model import OLS
    #from statsmodels.nonparametric.api import KernelReg
    import statsmodels.sandbox.nonparametric.kernel_extras as smke

    seed = np.random.randint(999999)
    #seed = 661176
    print(seed)
    np.random.seed(seed)

    sig_e = 0.5 #0.1
    nobs, k_vars = 200, 1
    x = np.random.uniform(-2, 2, size=(nobs, k_vars))
    x.sort()

    order = 3
    exog = x**np.arange(order + 1)
    beta = np.array([1, 1, 0.1, 0.0])[:order+1] # 1. / np.arange(1, order + 2)
    y_true = np.dot(exog, beta)
    y = y_true + sig_e * np.random.normal(size=nobs)
    endog = y

    print('DGP')
    print('nobs=%d, beta=%r, sig_e=%3.1f' % (nobs, beta, sig_e))

    mod_ols = OLS(endog, exog[:,:2])
    res_ols = mod_ols.fit()
    #'cv_ls'[1000, 0.5][0.01, 0.45]
    tst = smke.TestFForm(endog, exog[:,:2], bw=[0.01, 0.45], var_type='cc',
                         fform=lambda x,p: mod_ols.predict(p,x),
                         estimator=lambda y,x: OLS(y,x).fit().params,
                         nboot=1000)

    print('bw', tst.bw)
    print('tst.test_stat', tst.test_stat)
    print(tst.sig)
    print('tst.boots_results mean, min, max', (tst.boots_results.mean(),
                                               tst.boots_results.min(),
                                               tst.boots_results.max()))
    print('lower tail bootstrap p-value', (tst.boots_results < tst.test_stat).mean())
    print('upper tail bootstrap p-value', (tst.boots_results >= tst.test_stat).mean())
    from scipy import stats
    print('aymp.normal p-value (2-sided)', stats.norm.sf(np.abs(tst.test_stat))*2)
    print('aymp.normal p-value (upper)', stats.norm.sf(tst.test_stat))

    do_plot=True
    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, res_ols.fittedvalues)
        plt.title('OLS fit')

        plt.figure()
        plt.hist(tst.boots_results.ravel(), bins=20)
        plt.title('bootstrap histogram or test statistic')
        plt.show()
