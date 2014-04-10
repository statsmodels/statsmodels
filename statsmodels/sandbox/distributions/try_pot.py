# -*- coding: utf-8 -*-
"""
Created on Wed May 04 06:09:18 2011

@author: josef
"""
from __future__ import print_function
import numpy as np

def mean_residual_life(x, frac=None, alpha=0.05):
    '''emprirical mean residual life or expected shortfall

    Parameters
    ----------


    todo: check formula for std of mean
    doesn't include case for all observations
    last observations std is zero
    vectorize loop using cumsum
    frac doesn't work yet

    '''

    axis = 0  #searchsorted is 1d only
    x = np.asarray(x)
    nobs = x.shape[axis]
    xsorted = np.sort(x, axis=axis)
    if frac is None:
        xthreshold = xsorted
    else:
        xthreshold = xsorted[np.floor(nobs * frac).astype(int)]
    #use searchsorted instead of simple index in case of ties
    xlargerindex = np.searchsorted(xsorted, xthreshold, side='right')

    #replace loop with cumsum ?
    result = []
    for i in range(len(xthreshold)-1):
        k_ind = xlargerindex[i]
        rmean = x[k_ind:].mean()
        rstd = x[k_ind:].std()   #this doesn't work for last observations, nans
        rmstd = rstd/np.sqrt(nobs-k_ind)    #std error of mean, check formula
        result.append((k_ind, xthreshold[i], rmean, rmstd))

    res = np.array(result)
    crit = 1.96  # todo: without loading stats, crit = -stats.t.ppf(0.05)
    confint = res[:,1:2] + crit * res[:,-1:] * np.array([[-1,1]])
    return np.column_stack((res, confint))

expected_shortfall = mean_residual_life #alias


if __name__ == "__main__":
    rvs = np.random.standard_t(5, size= 10)
    res = mean_residual_life(rvs)
    print(res)
    rmean = [rvs[i:].mean() for i in range(len(rvs))]
    print(res[:,2] - rmean[1:])

'''
>>> mean_residual_life(rvs, frac= 0.5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "E:\Josef\eclipsegworkspace\statsmodels-josef-experimental-030\scikits\statsmodels\sandbox\distributions\try_pot.py", line 35, in mean_residual_life
    for i in range(len(xthreshold)-1):
TypeError: object of type 'numpy.float64' has no len()
>>> mean_residual_life(rvs, frac= [0.5])
array([[ 1.        , -1.16904459,  0.35165016,  0.41090978, -1.97442776,
        -0.36366142],
       [ 1.        , -1.16904459,  0.35165016,  0.41090978, -1.97442776,
        -0.36366142],
       [ 1.        , -1.1690445
'''
