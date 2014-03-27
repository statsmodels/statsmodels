# -*- coding: utf-8 -*-
"""Kernel Regression and Significance Test

Warning: SLOW, 11 minutes on my computer

Created on Thu Jan 03 20:20:47 2013

Author: Josef Perktold

results - this version
----------------------

>>> exec(open('ex_kernel_regression_censored1.py').read())
bw
[ 0.3987821   0.50933458]
[0.39878209999999997, 0.50933457999999998]

sig_test - default
Not Significant
pvalue
0.11
test statistic 0.000434305313291
bootstrap critical values
[ 0.00043875  0.00046808  0.0005064   0.00054151]

sig_test - pivot=True, nboot=200, nested_res=50
pvalue
0.01
test statistic 6.17877171579
bootstrap critical values
[ 5.5658345   5.74761076  5.87386858  6.46012041]
times: 8.34599995613 20.6909999847 666.373999834

"""

from __future__ import print_function
import time

import numpy as np
import statsmodels.nonparametric.api as nparam
import statsmodels.nonparametric.kernel_regression as smkr

if __name__ == '__main__':
    t0 = time.time()
    #example from test file
    nobs = 200
    np.random.seed(1234)
    C1 = np.random.normal(size=(nobs, ))
    C2 = np.random.normal(2, 1, size=(nobs, ))
    noise = np.random.normal(size=(nobs, ))
    Y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
    #self.write2file('RegData.csv', (Y, C1, C2))

    #CODE TO PRODUCE BANDWIDTH ESTIMATION IN R
    #library(np)
    #data <- read.csv('RegData.csv', header=FALSE)
    #bw <- npregbw(formula=data$V1 ~ data$V2 + data$V3,
    #                bwmethod='cv.aic', regtype='lc')
    model = nparam.KernelReg(endog=[Y], exog=[C1, C2],
                             reg_type='lc', var_type='cc', bw='aic')
    mean, marg = model.fit()
    #R_bw = [0.4017893, 0.4943397]  # Bandwidth obtained in R
    bw_expected = [0.3987821, 0.50933458]
    #npt.assert_allclose(model.bw, bw_expected, rtol=1e-3)
    print('bw')
    print(model.bw)
    print(bw_expected)

    print('\nsig_test - default')
    print(model.sig_test([1], nboot=100))
    t1 = time.time()
    res0 = smkr.TestRegCoefC(model, [1])
    print('pvalue')
    print((res0.t_dist >= res0.test_stat).mean())
    print('test statistic', res0.test_stat)
    print('bootstrap critical values')
    probs = np.array([0.9, 0.95, 0.975, 0.99])
    bsort0 = np.sort(res0.t_dist)
    nrep0 = len(bsort0)
    print(bsort0[(probs * nrep0).astype(int)])

    t2 = time.time()
    print('\nsig_test - pivot=True, nboot=200, nested_res=50')
    res1 = smkr.TestRegCoefC(model, [1], pivot=True, nboot=200, nested_res=50)
    print('pvalue')
    print((res1.t_dist >= res1.test_stat).mean())
    print('test statistic', res1.test_stat)
    print('bootstrap critical values')
    probs = np.array([0.9, 0.95, 0.975, 0.99])
    bsort1 = np.sort(res1.t_dist)
    nrep1 = len(bsort1)
    print(bsort1[(probs * nrep1).astype(int)])
    t3 = time.time()

    print('times:', t1-t0, t2-t1, t3-t2)


#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.plot(x, y, 'o', alpha=0.5)
#    ax.plot(x, y_cens, 'o', alpha=0.5)
#    ax.plot(x, y_true, lw=2, label='DGP mean')
#    ax.plot(x, sm_mean, lw=2, label='model 0 mean')
#    ax.plot(x, mean2, lw=2, label='model 2 mean')
#    ax.legend()
#
#    plt.show()
