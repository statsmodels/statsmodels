# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 10:51:39 2011

@author: josef
"""

import numpy as np

from scikits.statsmodels.sandbox.nonparametric import smoothers, kernels



#DGP: simple polynomial
order = 2
sigma_noise = 0.5
nobs = 100
lb, ub = -1, 2
x = np.linspace(lb, ub, nobs)
exog = x[:,None]**np.arange(order+1)
y_true = exog.sum(1)
y = y_true + sigma_noise * np.random.randn(nobs)


#xind = np.argsort(x)
pmod = smoothers.PolySmoother(2, x)
pmod.fit(y)  #no return
y_pred = pmod.predict(x)
error = y - y_pred
mse = (error*error).mean()
print mse

doplot = 1
if doplot:
    import matplotlib.pyplot as plt
    plt.plot(y, '.')
    plt.plot(y_true, '-', label='true')
    plt.plot(y_pred, '-', label='poly')
    plt.legend()

    plt.show()