# -*- coding: utf-8 -*-
"""some measures for evaluation of prediction, tests and model selection

Created on Tue Nov 08 15:23:20 2011

Author: Josef Perktold
License: BSD-3

"""


import numpy as np

def mse(x1, x2, axis=0):
    '''mean squared error
    '''
    return np.mean((x1-x2)**2, axis=axis)

def rmse(x1, x2, axis=0):
    '''root mean squared error
    '''
    return np.sqrt(mse(x1, x2, axis=axis))

def maxabs(x1, x2, axis=0):
    '''maximum absolute error
    '''
    return np.max(np.abs(x1-x2), axis=axis)

def meanabs(x1, x2, axis=0):
    '''mean absolute error
    '''
    return np.mean(np.abs(x1-x2), axis=axis)

def medianabs(x1, x2, axis=0):
    '''median absolute error
    '''
    return np.median(np.abs(x1-x2), axis=axis)

def bias(x1, x2, axis=0):
    '''bias, mean error
    '''
    return np.mean(x1-x2, axis=axis)

def medianbias(x1, x2, axis=0):
    '''median bias, median error
    '''
    return np.median(x1-x2, axis=axis)

def vare(x1, x2, ddof=0, axis=0):
    '''variance of error
    '''
    return np.var(x1-x2, ddof=0, axis=axis)

def stde(x1, x2, ddof=0, axis=0):
    '''variance of error
    '''
    return np.std(x1-x2, ddof=0, axis=axis)

def iqr(x1, x2, axis=0):
    '''interquartile range of error

    rounded index, no interpolations

    this could use newer numpy function instead
    '''
    if axis is None:
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)
        axis = 0
    xdiff = np.sort(x1 - x2)
    nobs = x1.shape[axis]
    idx = np.round((nobs-1) * np.array([0.25, 0.75])).astype(int)
    sl = [slice(None)] * xdiff.ndim
    sl[axis] = idx
    iqr = np.diff(xdiff[sl], axis=axis)
    iqr = np.squeeze(iqr) #drop reduced dimension
    if iqr.size == 1:
        return iqr #[0]
    else:
        return iqr


# Information Criteria
#---------------------

def aic(llf, nobs, df_modelwc):
    '''Akaike information criterion

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion

    '''
    return -2. * llf + 2. * df_modelwc

def aicc(llf, nobs, df_modelwc):
    '''Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aicc : float
        information criterion

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion#AICc

    '''
    return -2. * llf + 2. * df_modelwc * nobs / (nobs - df_modelwc - 1.)
    #float division

def bic(llf, nobs, df_modelwc):
    '''Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    References
    ----------
    http://en.wikipedia.org/wiki/Bayesian_information_criterion

    '''
    return -2. * llf + np.log(nobs) * df_modelwc

def hqic(llf, nobs, df_modelwc):
    '''Hannan-Quinn information criterion (HQC)

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    hqic : float
        information criterion

    References
    ----------
    Wikipedia doesn't say much

    '''
    return -2. * llf + 2 * np.log(np.log(nobs)) * df_modelwc


#IC based on residual sigma

def aic_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Akaike information criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. The information criteria should be used to compare
    only comparable models.

    For example, AIC is defined in terms of the loglikelihood as

    -2 llf + 2 k

    in terms of sigma_hat

    log(sigma_hat^2) + 2 k / n

    in terms of the determinant of Sigma_hat

    log(|sigma_hat|) + 2 k / n

    Note: In our definition we do not divide by n in the log-likelihood
    version.

    TODO: Latex math
    reference for example lecture notes by Herman Bierens

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aic(0, nobs, df_modelwc) / nobs


def aicc_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aicc : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable models.

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion#AICc

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aicc(0, nobs, df_modelwc) / nobs
    #float division


def bic_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable models.

    References
    ----------
    http://en.wikipedia.org/wiki/Bayesian_information_criterion

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + bic(0, nobs, df_modelwc) / nobs


def hqic_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Hannan-Quinn information criterion (HQC)

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    hqic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable models.

    References
    ----------
    xxx

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + hqic(0, nobs, df_modelwc) / nobs


#from var_model.py, VAR only? separates neqs and k_vars per equation
#def fpe_sigma():
#    ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)


__all__ = [maxabs, meanabs, medianabs, medianbias, mse, rmse, stde, vare,
           aic, aic_sigma, aicc, aicc_sigma, bias, bic, bic_sigma,
           hqic, hqic_sigma, iqr]
