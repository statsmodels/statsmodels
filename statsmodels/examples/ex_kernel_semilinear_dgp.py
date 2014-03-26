# -*- coding: utf-8 -*-
"""

Created on Sun Jan 06 09:50:54 2013

Author: Josef Perktold
"""

from __future__ import print_function


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    #from statsmodels.nonparametric.api import KernelReg
    import statsmodels.sandbox.nonparametric.kernel_extras as smke
    import statsmodels.sandbox.nonparametric.dgp_examples as dgp

    class UnivariateFunc1a(dgp.UnivariateFunc1):

        def het_scale(self, x):
            return 0.5

    seed = np.random.randint(999999)
    #seed = 430973
    #seed = 47829
    seed = 648456 #good seed for het_scale = 0.5
    print(seed)
    np.random.seed(seed)

    nobs, k_vars = 300, 3
    x = np.random.uniform(-2, 2, size=(nobs, k_vars))
    xb = x.sum(1) / 3  #beta = [1,1,1]

    k_vars_lin = 2
    x2 = np.random.uniform(-2, 2, size=(nobs, k_vars_lin))

    funcs = [#dgp.UnivariateFanGijbels1(),
             #dgp.UnivariateFanGijbels2(),
             #dgp.UnivariateFanGijbels1EU(),
             #dgp.UnivariateFanGijbels2(distr_x=stats.uniform(-2, 4))
             UnivariateFunc1a(x=xb)
             ]

    res = []
    fig = plt.figure()
    for i,func in enumerate(funcs):
        #f = func()
        f = func
        y = f.y + x2.sum(1)
        model = smke.SemiLinear(y, x2, x, 'ccc', k_vars_lin)
        mean, mfx = model.fit()
        ax = fig.add_subplot(1, 1, i+1)
        f.plot(ax=ax)
        xb_est = np.dot(model.exog, model.b)
        sortidx = np.argsort(xb_est) #f.x)
        ax.plot(f.x[sortidx], mean[sortidx], 'o', color='r', lw=2, label='est. mean')
#        ax.plot(f.x, mean0, color='g', lw=2, label='est. mean')
        ax.legend(loc='upper left')
        res.append((model, mean, mfx))

    print('beta', model.b)
    print('scale - est', (y - (xb_est+mean)).std())
    print('scale - dgp realised, true', (y - (f.y_true + x2.sum(1))).std(), \
                                        2 * f.het_scale(1))
    fittedvalues = xb_est + mean
    resid = np.squeeze(model.endog) - fittedvalues
    print('corrcoef(fittedvalues, resid)', np.corrcoef(fittedvalues, resid)[0,1])
    print('variance of components, var and as fraction of var(y)')
    print('fitted values', fittedvalues.var(), fittedvalues.var() / y.var())
    print('linear       ', xb_est.var(), xb_est.var() / y.var())
    print('nonparametric', mean.var(), mean.var() / y.var())
    print('residual     ', resid.var(), resid.var() / y.var())
    print('\ncovariance decomposition fraction of var(y)')
    print(np.cov(fittedvalues, resid) / model.endog.var(ddof=1))
    print('sum', (np.cov(fittedvalues, resid) / model.endog.var(ddof=1)).sum())
    print('\ncovariance decomposition, xb, m, resid as fraction of var(y)')
    print(np.cov(np.column_stack((xb_est, mean, resid)), rowvar=False) / model.endog.var(ddof=1))

    fig.suptitle('Kernel Regression')
    fig.show()

    alpha = 0.7
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f.x[sortidx], f.y[sortidx], 'o', color='b', lw=2, alpha=alpha, label='observed')
    ax.plot(f.x[sortidx], f.y_true[sortidx], 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
    ax.plot(f.x[sortidx], mean[sortidx], 'o', color='r', lw=2, alpha=alpha, label='est. mean')
    ax.legend(loc='upper left')

    sortidx = np.argsort(xb_est + mean)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f.x[sortidx], y[sortidx], 'o', color='b', lw=2, alpha=alpha, label='observed')
    ax.plot(f.x[sortidx], f.y_true[sortidx], 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
    ax.plot(f.x[sortidx], (xb_est + mean)[sortidx], 'o', color='r', lw=2, alpha=alpha, label='est. mean')
    ax.legend(loc='upper left')
    ax.set_title('Semilinear Model - observed and total fitted')

    fig = plt.figure()
#    ax = fig.add_subplot(1, 2, 1)
#    ax.plot(f.x, f.y, 'o', color='b', lw=2, alpha=alpha, label='observed')
#    ax.plot(f.x, f.y_true, 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
#    ax.plot(f.x, mean, 'o', color='r', lw=2, alpha=alpha, label='est. mean')
#    ax.legend(loc='upper left')
    sortidx0 = np.argsort(xb)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(f.y[sortidx0], 'o', color='b', lw=2, alpha=alpha, label='observed')
    ax.plot(f.y_true[sortidx0], 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
    ax.plot(mean[sortidx0], 'o', color='r', lw=2, alpha=alpha, label='est. mean')
    ax.legend(loc='upper left')
    ax.set_title('Single Index Model (sorted by true xb)')

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(y - xb_est, 'o', color='b', lw=2, alpha=alpha, label='observed')
    ax.plot(f.y_true, 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
    ax.plot(mean, 'o', color='r', lw=2, alpha=alpha, label='est. mean')
    ax.legend(loc='upper left')
    ax.set_title('Single Index Model (nonparametric)')

    plt.figure()
    plt.plot(y, xb_est+mean, '.')
    plt.title('observed versus fitted values')

    plt.show()



