# -*- coding: utf-8 -*-
"""

Created on Sun Jan 06 09:50:54 2013

Author: Josef Perktold
"""

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    #from statsmodels.nonparametric.api import KernelReg
    import statsmodels.sandbox.nonparametric.kernel_extras as smke
    import statsmodels.nonparametric.dgp_examples as dgp


    seed = np.random.randint(999999)
    seed = 430973
    print seed
    np.random.seed(seed)

    nobs, k_vars = 200, 3
    x = np.random.uniform(-2, 2, size=(nobs, k_vars))
    xb = x.sum(1) / 3  #beta = [1,1,1]

    funcs = [#dgp.UnivariateFanGijbels1(),
             #dgp.UnivariateFanGijbels2(),
             #dgp.UnivariateFanGijbels1EU(),
             #dgp.UnivariateFanGijbels2(distr_x=stats.uniform(-2, 4))
             dgp.UnivariateFunc1(x=xb)
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
