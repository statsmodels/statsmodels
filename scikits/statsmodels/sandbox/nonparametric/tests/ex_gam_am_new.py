# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 13:45:43 2011

@author: josef
"""

import time

import numpy as np
#import matplotlib.pyplot as plt

from scipy import stats

from scikits.statsmodels.sandbox.gam import AdditiveModel
from scikits.statsmodels.sandbox.gam import Model as GAM #?
from scikits.statsmodels.genmod import families
from scikits.statsmodels.genmod.generalized_linear_model import GLM

np.random.seed(8765993) 
#seed is chosen for nice result, not randomly
#other seeds are pretty off in the prediction

#DGP: simple polynomial
order = 3
sigma_noise = 0.25
nobs = 500
lb, ub = -0.5, 2.5
x1 = np.linspace(lb, ub, nobs)
x2 = np.sin(x1)
x = np.column_stack((x1/x1.max(), x2))
exog = (x[:,:,None]**np.arange(order+1)[None, None, :]).reshape(nobs, -1)
y_true = exog.sum(1) / 2. 
z = y_true #alias check
d = x
y = y_true + sigma_noise * np.random.randn(nobs)

example = 1

if example == 1:
    m = AdditiveModel(d)
    m.fit(y)
    
    y_pred = m.results.predict(d)

    
for ss in m.smoothers:
    print ss.params 

if example > 0:
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(exog)    
    
    y_pred = m.results.mu + m.results.alpha #m.results.predict(d)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(y, '.')
    plt.plot(y_true, 'b-', label='true')
    plt.plot(y_pred, 'r-', label='GAM')
    plt.legend(loc='upper left')
    plt.title('gam.GAM Poisson')

    counter = 2
    for ii, xx in zip(['z', 'x1', 'x2'], [z, x[:,0], x[:,1]]):
        sortidx = np.argsort(xx)
        #plt.figure()
        plt.subplot(2, 2, counter)
        plt.plot(xx[sortidx], y[sortidx], '.')
        plt.plot(xx[sortidx], y_true[sortidx], 'b-', label='true')
        plt.plot(xx[sortidx], y_pred[sortidx], 'r-', label='GAM')
        plt.legend(loc='upper left')
        plt.title('gam.GAM Poisson ' + ii)
        counter += 1
        
    plt.show()