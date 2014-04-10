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




Monte Carlo with 100 replications
---------------------------------
results
598948
time 11.1642833312
[-0.72505981  0.26514944  0.45681704]
[ 0.74884796  0.22005569  0.3004892 ]
reject at [0.2, 0.1, 0.05] (row 1: normal, row 2: bootstrap)
[[ 0.55  0.24  0.01]
 [ 0.29  0.16  0.06]]
bw [ 0.11492364  0.11492364]
tst.test_stat -1.40274609515
Not Significant
tst.boots_results min, max -2.03386582198 2.32562183511
lower tail bootstrap p-value 0.077694235589
aymp.normal p-value (2-sided) 0.160692566481

mean and std in Li and Wang for n=1 are -0.764 and 0.621
results look reasonable now

Power
-----
true model: quadratic, estimated model: linear
498198
time 8.4588166674
[ 0.50374364  0.3991975   0.25373434]
[ 1.21353172  0.28669981  0.25461368]
reject at [0.2, 0.1, 0.05] (row 1: normal, row 2: bootstrap)
[[ 0.66  0.78  0.82]
 [ 0.46  0.61  0.74]]
bw [ 0.11492364  0.11492364]
tst.test_stat 0.505426717024
Not Significant
tst.boots_results min, max -1.67050998463 3.39835350718
lower tail bootstrap p-value 0.892230576441
upper tail bootstrap p-value 0.107769423559
aymp.normal p-value (2-sided) 0.613259157709
aymp.normal p-value (upper) 0.306629578855


"""

from __future__ import print_function


if __name__ == '__main__':

    import time

    import numpy as np
    from scipy import stats

    from statsmodels.regression.linear_model import OLS
    #from statsmodels.nonparametric.api import KernelReg
    import statsmodels.sandbox.nonparametric.kernel_extras as smke

    seed = np.random.randint(999999)
    #seed = 661176
    print(seed)
    np.random.seed(seed)

    sig_e = 0.1 #0.5 #0.1
    nobs, k_vars = 100, 1

    t0 = time.time()

    b_res = []
    for i in range(100):
        x = np.random.uniform(0, 1, size=(nobs, k_vars))
        x.sort(0)

        order = 2
        exog = x**np.arange(1, order + 1)
        beta = np.array([2, -0.2])[:order+1-1] # 1. / np.arange(1, order + 2)
        y_true = np.dot(exog, beta)
        y = y_true + sig_e * np.random.normal(size=nobs)
        endog = y

        mod_ols = OLS(endog, exog[:,:1])
        #res_ols = mod_ols.fit()
        #'cv_ls'[1000, 0.5]
        bw_lw = [1./np.sqrt(12.) * nobs**(-0.2)]*2  #(-1. / 5.)
        tst = smke.TestFForm(endog, exog[:,:1], bw=bw_lw, var_type='c',
                             fform=lambda x,p: mod_ols.predict(p,x),
                             estimator=lambda y,x: OLS(y,x).fit().params,
                             nboot=399)
        b_res.append([tst.test_stat,
                      stats.norm.sf(tst.test_stat),
                      (tst.boots_results > tst.test_stat).mean()])
    t1 = time.time()
    b_res = np.asarray(b_res)

    print('time', (t1 - t0) / 60.)
    print(b_res.mean(0))
    print(b_res.std(0))
    print('reject at [0.2, 0.1, 0.05] (row 1: normal, row 2: bootstrap)')
    print((b_res[:,1:,None] >= [0.2, 0.1, 0.05]).mean(0))

    print('bw', tst.bw)
    print('tst.test_stat', tst.test_stat)
    print(tst.sig)
    print('tst.boots_results min, max', tst.boots_results.min(), tst.boots_results.max())
    print('lower tail bootstrap p-value', (tst.boots_results < tst.test_stat).mean())
    print('upper tail bootstrap p-value', (tst.boots_results >= tst.test_stat).mean())
    from scipy import stats
    print('aymp.normal p-value (2-sided)', stats.norm.sf(np.abs(tst.test_stat))*2)
    print('aymp.normal p-value (upper)', stats.norm.sf(tst.test_stat))

    res_ols = mod_ols.fit()

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
