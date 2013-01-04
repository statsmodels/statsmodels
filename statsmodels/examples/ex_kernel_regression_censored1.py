# -*- coding: utf-8 -*-
"""

Created on Thu Jan 03 20:20:47 2013

Author: Josef Perktold
"""

import numpy as np
import statsmodels.nonparametric.api as nparam

if __name__ == '__main__':
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
    #R_bw = [0.4017893, 0.4943397]  # Bandwidth obtained in R
    bw_expected = [0.3987821, 0.50933458]
    #npt.assert_allclose(model.bw, bw_expected, rtol=1e-3)
    print model.bw
    print bw_expected

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'o', alpha=0.5)
    ax.plot(x, y_cens, 'o', alpha=0.5)
    ax.plot(x, y_true, lw=2, label='DGP mean')
    ax.plot(x, sm_mean, lw=2, label='model 0 mean')
    ax.plot(x, mean2, lw=2, label='model 2 mean')
    ax.legend()

    plt.show()
