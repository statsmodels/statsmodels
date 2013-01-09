# -*- coding: utf-8 -*-
"""Example TestFForm with Li Wang DGP1

Created on Tue Jan 08 19:03:20 2013

Author: Josef Perktold


trying to replicate some examples in
Li, Q., and Suojin Wang. 1998. "A Simple Consistent Bootstrap Test for a
    Parametric Regression Function."
    Journal of Econometrics 87 (1) (November): 145-165.
    doi:10.1016/S0304-4076(98)00011-6.

currently DGP1

in a few runs the test_statistic is around -1.8, while Li/Wang report a
mean=-0.764 in the simulations.
bootstrap distribution is also wider
maybe a scaling factor 0.5 missing in test statistic?

"""



if __name__ == '__main__':

    import numpy as np

    from statsmodels.regression.linear_model import OLS
    #from statsmodels.nonparametric.api import KernelReg
    import statsmodels.sandbox.nonparametric.kernel_extras as smke

    seed = np.random.randint(999999)
    #seed = 661176
    print seed
    np.random.seed(seed)

    sig_e = 0.1 #0.5 #0.1
    nobs, k_vars = 100, 1
    x = np.random.uniform(0, 1, size=(nobs, k_vars))
    x.sort(0)

    order = 2
    exog = x**np.arange(1, order + 1)
    beta = np.array([2, -1.])[:order+1-1] # 1. / np.arange(1, order + 2)
    y_true = np.dot(exog, beta)
    y = y_true + sig_e * np.random.normal(size=nobs)
    endog = y

    mod_ols = OLS(endog, exog[:,:2])
    res_ols = mod_ols.fit()
    #'cv_ls'[1000, 0.5]
    bw_lw = [1./np.sqrt(12.) * nobs**(-0.2)]*2  #(-1. / 5.)
    tst = smke.TestFForm(endog, exog[:,:2], bw=bw_lw, var_type='cc',
                         fform=lambda x,p: mod_ols.predict(p,x),
                         estimator=lambda y,x: OLS(y,x).fit().params,
                         nboot=399)

    print 'bw', tst.bw
    print 'tst.test_stat', tst.test_stat
    print tst.sig
    print 'tst.boots_results min, max', tst.boots_results.min(), tst.boots_results.max()
    print 'lower tail bootstrap p-value', (tst.boots_results < tst.test_stat).mean()
    from scipy import stats
    print 'aymp.normal p-value (2-sided)', stats.norm.sf(np.abs(tst.test_stat))*2

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
