# -*- coding: utf-8 -*-
"""script to try out Censored kernel regression

Created on Wed Jan 02 13:43:44 2013

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np
import statsmodels.nonparametric.api as nparam

if __name__ == '__main__':

    np.random.seed(500)
    nobs = [250, 1000][0]
    sig_fac = 1
    x = np.random.uniform(-2, 2, size=nobs)
    x.sort()
    x2 = x**2 + 0.02 * np.random.normal(size=nobs)
    y_true = np.sin(x*5)/x + 2*x - 3 * x2
    y = y_true + sig_fac * (np.sqrt(np.abs(3+x))) * np.random.normal(size=nobs)
    cens_side = ['left', 'right', 'random'][2]
    if cens_side == 'left':
        c_val = 0.5
        y_cens = np.clip(y, c_val, 100)
    elif cens_side == 'right':
        c_val = 3.5
        y_cens = np.clip(y, -100, c_val)
    elif cens_side == 'random':
        c_val = 3.5 + 3 * np.random.randn(nobs)
        y_cens = np.minimum(y, c_val)

    model = nparam.KernelCensoredReg(endog=[y_cens],
                             #exog=[np.column_stack((x, x**2))], reg_type='lc',
                             exog=[x, x2], reg_type='ll',
                             var_type='cc', bw='aic', #'cv_ls', #[0.23, 434697.22], #'cv_ls',
                             censor_val=c_val[:,None],
                             #defaults=nparam.EstimatorSettings(efficient=True)
                             )

    sm_bw = model.bw

    sm_mean, sm_mfx = model.fit()

#    model1 = nparam.KernelReg(endog=[y],
#                             exog=[x], reg_type='lc',
#                             var_type='c', bw='cv_ls')
#    mean1, mfx1 = model1.fit()

    model2 = nparam.KernelReg(endog=[y_cens],
                             exog=[x, x2], reg_type='ll',
                             var_type='cc', bw='aic',# 'cv_ls'
                             )

    mean2, mfx2 = model2.fit()

    print(model.bw)
    #print model1.bw
    print(model2.bw)

    ix = np.argsort(y_cens)
    ix_rev = np.zeros(nobs, int)
    ix_rev[ix] = np.arange(nobs)
    ix_rev = model.sortix_rev

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'o', alpha=0.5)
    ax.plot(x, y_cens, 'o', alpha=0.5)
    ax.plot(x, y_true, lw=2, label='DGP mean')
    ax.plot(x, sm_mean[ix_rev], lw=2, label='model 0 mean')
    ax.plot(x, mean2, lw=2, label='model 2 mean')
    ax.legend()

    plt.show()
