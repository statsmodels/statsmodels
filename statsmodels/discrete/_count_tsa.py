# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:54:42 2018

Author: Josef Perktold
"""

import numpy as np
try:
    import numba
    has_numba = True
except ImportError:
    has_numba = False
    import warnings
    warnings.warn('numba not available for jit')



def _predict_exparma(endog, exog, params, k_ar, k_ma):
    """ARMA type model with log-link, exponential mean function

    Both past observed endog (AR) and past predicted values or mean (MA) enter
    the current prediction.

    usecase count or Poisson time series models, e.g. INAR,
    or general nonlinear model with non-negative endog

    This is currently just a function to try out numba

    """

    nobs = endog.shape[0]
    k_exog = exog.shape[1]
    k_init = max(k_ar, k_ma)
    if k_ar > 0:
        p_ar = params[:k_ar]
    if k_ma > 0:
        p_ma = params[k_ar : k_ar + k_ma]
    p_ex = params[k_ar + k_ma:]

    predicted = np.empty(nobs, dtype=np.float)
    predicted[:k_init] = 1    # need something proper here
    for i in range(k_init, nobs):
        pred_i = 0.0
        for k in range(1, k_ar+1):
            pred_i += endog[i-k] * p_ar[k-1]
        for k in range(1, k_ma+1):
            pred_i += predicted[i-k] * p_ma[k-1]
        for k in range(k_exog):
            pred_i += exog[i, k] * p_ex[k]
        predicted[i] = np.exp(pred_i)

    return predicted

if has_numba:
    predict_exparma = numba.jit(_predict_exparma)
else:
    predict_exparma = _predict_exparma


