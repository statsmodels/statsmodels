# -*- coding: utf-8 -*-
"""script to check KernelCensoredReg based on test file

Created on Thu Jan 03 20:20:47 2013

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np
import statsmodels.nonparametric.api as nparam

if __name__ == '__main__':
    #example from test file
    nobs = 200
    np.random.seed(1234)
    C1 = np.random.normal(size=(nobs, ))
    C2 = np.random.normal(2, 1, size=(nobs, ))
    noise = 0.1 * np.random.normal(size=(nobs, ))
    y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
    y[y>0] = 0  # censor the data
    model = nparam.KernelCensoredReg(endog=[y], exog=[C1, C2],
                                     reg_type='ll', var_type='cc',
                                     bw='cv_ls', censor_val=0)
    sm_mean, sm_mfx = model.fit()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sortidx = np.argsort(y)
    ax.plot(y[sortidx], 'o', alpha=0.5)
    #ax.plot(x, y_cens, 'o', alpha=0.5)
    #ax.plot(x, y_true, lw=2, label='DGP mean')
    ax.plot(sm_mean[sortidx], lw=2, label='model 0 mean')
    #ax.plot(x, mean2, lw=2, label='model 2 mean')
    ax.legend()

    plt.show()
