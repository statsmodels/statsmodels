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


    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure.

        Parameters
        ----------
        X : array-like
            The data to be whitened

        Returns
        -------
        TODO
        """
#TODO: notation for AR process
        X = np.asarray(X, np.float64)
        _X = X.copy()
        #dimension handling is not DRY
        # I think previous code worked for 2d because of single index rows in np
        if X.ndim == 1:
            for i in range(self.order):
                _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
            return _X[self.order:]
        elif X.ndim == 2:
            for i in range(self.order):
                _X[(i+1):,:] = _X[(i+1):,:] - self.rho[i] * X[0:-(i+1),:]
                return _X[self.order:,:]

def yule_walker(X, order=1, method="unbiased", df=None, inv=False, demean=True):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    Parameters
    ----------
    X : array-like
        1d array
    order : integer, optional
        The order of the autoregressive process.  Default is 1.
    method : string, optional
       Method can be "unbiased" or "mle" and this determines denominator in
       estimate of autocorrelation function (ACF) at lag k. If "mle", the
       denominator is n=X.shape[0], if "unbiased" the denominator is n-k.
       The default is unbiased.
    df : integer, optional
       Specifies the degrees of freedom. If `df` is supplied, then it is assumed
       the X has `df` degrees of freedom rather than `n`.  Default is None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho
        The autoregressive coefficients
    sigma
        TODO

    Examples
    --------
    >>> import scikits.statsmodels.api as sm
    >>> from scikits.statsmodels.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma = sm.regression.yule_walker(data.endog,       \
                                       order=4, method="mle")

    >>> rho
    array([ 1.28310031, -0.45240924, -0.20770299,  0.04794365])
    >>> sigma
    16.808022730464351

    """
#TODO: define R better, look back at notes and technical notes on YW.
#First link here is useful
#http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YuleWalkerAndMore.htm
    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError("ACF estimation method must be 'unbiased' or 'MLE'")
    X = np.array(X)
    if demean:
        X -= X.mean()                  # automatically demean's X
    n = df or X.shape[0]

    if method == "unbiased":        # this is df_resid ie., n - p
        denom = lambda k: n - k
    else:
        denom = lambda k: n
    if X.ndim > 1 and X.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")
    r = np.zeros(order+1, np.float64)
    r[0] = (X**2).sum() / denom(0)
    for k in range(1,order+1):
        r[k] = (X[0:-k]*X[k:]).sum() / denom(k)
    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:]*rho).sum()
    if inv == True:
        return rho, np.sqrt(sigmasq), np.linalg.inv(R)
    else:
        return rho, np.sqrt(sigmasq)
