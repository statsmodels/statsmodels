# -*- coding: utf-8 -*-
"""Correlation and Covariance Structures

Created on Sat Dec 17 20:46:05 2011

Author: Josef Perktold
License: BSD-3


Reference
---------
quick reading of some section on mixed effects models in S-plus and of
outline for GEE.

"""

import numpy as np

def corr_equi(k_vars, rho):
    '''create equicorrelated correlation matrix with rho on off diagonal

    Parameters
    ----------
    k_vars : int
        number of variables, correlation matrix will be (k_vars, k_vars)
    rho : float
        correlation between any two random variables

    Returns
    -------
    corr : ndarray (k_vars, k_vars)
        correlation matrix

    '''
    corr = np.empty((k_vars, k_vars))
    corr.fill(rho)
    corr[np.diag_indices_from(corr)] = 1
    return corr

def corr_ar(k_vars, ar):
    '''create autoregressive correlation matrix

    This might be MA, not AR, process if used for residual process - check

    Parameters
    ----------
    ar : array_like, 1d
        AR lag-polynomial including 1 for lag 0


    '''
    from scipy.linalg import toeplitz
    if len(ar) < k_vars:
        ar_ = np.zeros(k_vars)
        ar_[:len(ar)] = ar
        ar = ar_

    return toeplitz(ar)


def corr_arma(k_vars, ar, ma):
    '''create arma correlation matrix

    converts arma to autoregressive lag-polynomial with k_var lags

    ar and arma might need to be switched for generating residual process

    Parameters
    ----------
    ar : array_like, 1d
        AR lag-polynomial including 1 for lag 0
    ma : array_like, 1d
        MA lag-polynomial

    '''
    from scipy.linalg import toeplitz
    from scikits.statsmodels.tsa.arima_process import arma2ar

    ar = arma2ar(ar, ma, nobs=k_vars)[:k_vars]  #bug in arma2ar

    return toeplitz(ar)


def corr2cov(corr, std):
    '''convert correlation matrix to covariance matrix

    Parameters
    ----------
    corr : ndarray, (k_vars, k_vars)
        correlation matrix
    std : ndarray, (k_vars,)
        standard deviation for the vector of random variables

    '''
    cov = corr * std[:,None] * std[None, :]  #same as outer product
    return cov
