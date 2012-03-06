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
    from statsmodels.tsa.arima_process import arma2ar

    ar = arma2ar(ar, ma, nobs=k_vars)[:k_vars]  #bug in arma2ar

    return toeplitz(ar)


def corr2cov(corr, std):
    '''convert correlation matrix to covariance matrix

    Parameters
    ----------
    corr : ndarray, (k_vars, k_vars)
        correlation matrix
    std : ndarray, (k_vars,) or scalar
        standard deviation for the vector of random variables. If scalar, then
        it is assumed that all variables have the same scale given by std.

    '''
    if np.size(std) == 1:
        std = std*np.ones(corr.shape[0])
    cov = corr * std[:,None] * std[None, :]  #same as outer product
    return cov


def whiten_ar(x, ar_coefs):
    """
    Whiten a series of columns according to an AR(p) covariance structure.

    This drops the initial conditions (Cochran-Orcut ?)
    Uses loop, so for short ar polynomials only, use lfilter otherwise

    This needs to improve, option on method, full additional to conditional

    Parameters
    ----------
    x : array-like, (nobs,) or (nobs, k_vars)
        The data to be whitened along axis 0
    ar_coefs : array
        coefficients of AR lag- polynomial,   TODO: ar or ar_coefs?

    Returns
    -------
    x_new : ndarray
        transformed array

    """

    rho = ar_coefs

    x = np.array(x, np.float64)  #make copy
    #_x = x.copy()
    #dimension handling is not DRY
    # I think previous code worked for 2d because of single index rows in np
    if x.ndim == 2:
        rho = rho[:, None]
    for i in range(self.order):
        _x[(i+1):] = _x[(i+1):] - rho[i] * x[0:-(i+1)]

    return _x[self.order:]


def yule_walker_acov(acov, order=1, method="unbiased", df=None, inv=False):
    """
    Estimate AR(p) parameters from acovf using Yule-Walker equation.


    Parameters
    ----------
    acov : array-like, 1d
        auto-covariance
    order : integer, optional
        The order of the autoregressive process.  Default is 1.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is False.

    Returns
    -------
    rho : ndarray
        The estimated autoregressive coefficients
    sigma
        TODO
    Rinv : ndarray
        inverse of the Toepliz matrix

    """


    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:]*rho).sum()
    if inv == True:
        return rho, np.sqrt(sigmasq), np.linalg.inv(R)
    else:
        return rho, np.sqrt(sigmasq)


class ARCovariance(object):
    '''
    experimental class for Covariance of AR process
    classmethod? staticmethods?
    '''

    def __init__(self, ar=None, ar_coefs=None, sigma=1.):
        if ar is not None:
            self.ar = ar
            self.ar_coefs = -ar[1:]
            self.k_lags = len(ar)
        elif ar_coefs is not None:
            self.arcoefs = ar_coefs
            self.ar = np.hstack(([1], -ar_coefs))
            self.k_lags = len(self.ar)

    @classmethod
    def fit(cls, cov, order, **kwds):
        rho, sigma = yule_walker_acov(cov, order=order, **kwds)
        return cls(ar_coefs=rho)

    def whiten(self, x):
        return whiten_ar(x, self.ar_coefs)

    def corr(self, k_vars=None):
        if k_vars is None:
            k_vars = len(self.ar)   #this could move into corr_arr
        return corr_ar(k_vars, self.ar)

    def cov(self, k_vars=None):
        return cov2corr(corr(self, k_vars=None), self.sigma)
