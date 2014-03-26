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
#        mod0 = smke.SingleIndexModel(endog=[f.y], exog=[xb], #reg_type='ll',
#                          var_type='c')#, bw='cv_ls')
#        mean0, mfx0 = mod0.fit()
        model = smke.SingleIndexModel(endog=[f.y], exog=x, #reg_type='ll',
                          var_type='ccc')#, bw='cv_ls')
        mean, mfx = model.fit()
        ax = fig.add_subplot(1, 1, i+1)
        f.plot(ax=ax)
        xb_est = np.dot(model.exog, model.b)
        sortidx = np.argsort(xb_est) #f.x)
        ax.plot(f.x[sortidx], mean[sortidx], 'o', color='r', lw=2, label='est. mean')
#        ax.plot(f.x, mean0, color='g', lw=2, label='est. mean')
        ax.legend(loc='upper left')
        res.append((model, mean, mfx))

    fig.suptitle('Kernel Regression')
    fig.show()

    alpha = 0.7
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f.x[sortidx], f.y[sortidx], 'o', color='b', lw=2, alpha=alpha, label='observed')
    ax.plot(f.x[sortidx], f.y_true[sortidx], 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
    ax.plot(f.x[sortidx], mean[sortidx], 'o', color='r', lw=2, alpha=alpha, label='est. mean')
    ax.legend(loc='upper left')

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
    ax.plot(f.y[sortidx], 'o', color='b', lw=2, alpha=alpha, label='observed')
    ax.plot(f.y_true[sortidx], 'o', color='g', lw=2, alpha=alpha, label='dgp. mean')
    ax.plot(mean[sortidx], 'o', color='r', lw=2, alpha=alpha, label='est. mean')
    ax.legend(loc='upper left')
    ax.set_title('Single Index Model (sorted by estimated xb)')
    plt.show()
