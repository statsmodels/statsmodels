# -*- coding: utf-8 -*-
"""some measures for evaluation of prediction, tests and model selection

Created on Tue Nov 08 15:23:20 2011

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
import scipy
import scipy.stats


def mse(x1, x2, axis=0):
    """mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1-x2)**2, axis=axis)


def rmse(x1, x2, axis=0):
    """root mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))


def maxabs(x1, x2, axis=0):
    """maximum absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    maxabs : ndarray or float
       maximum absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.max(np.abs(x1-x2), axis=axis)


def meanabs(x1, x2, axis=0):
    """mean absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    meanabs : ndarray or float
       mean absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(np.abs(x1-x2), axis=axis)


def medianabs(x1, x2, axis=0):
    """median absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianabs : ndarray or float
       median absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(np.abs(x1-x2), axis=axis)


def bias(x1, x2, axis=0):
    """bias, mean error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    bias : ndarray or float
       bias, or mean difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(x1-x2, axis=axis)


def medianbias(x1, x2, axis=0):
    """median bias, median error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianbias : ndarray or float
       median bias, or median difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(x1-x2, axis=axis)


def vare(x1, x2, ddof=0, axis=0):
    """variance of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    vare : ndarray or float
       variance of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.var(x1-x2, ddof=ddof, axis=axis)


def stde(x1, x2, ddof=0, axis=0):
    """standard deviation of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    stde : ndarray or float
       standard deviation of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.std(x1-x2, ddof=ddof, axis=axis)


def iqr(x1, x2, axis=0):
    """interquartile range of error

    rounded index, no interpolations

    this could use newer numpy function instead

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.

    This uses ``numpy.asarray`` to convert the input, in contrast to the other
    functions in this category.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
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
    iqr = np.squeeze(iqr)  # drop reduced dimension
    return iqr


# Information Criteria
# ---------------------

def aic(llf, nobs, df_modelwc):
    """Akaike information criterion

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

    """
    return -2. * llf + 2. * df_modelwc


def aicc(llf, nobs, df_modelwc):
    """Akaike information criterion (AIC) with small sample correction

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

    """
    return -2. * llf + 2. * df_modelwc * nobs / (nobs - df_modelwc - 1.)


def bic(llf, nobs, df_modelwc):
    """Bayesian information criterion (BIC) or Schwarz criterion

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

    """
    return -2. * llf + np.log(nobs) * df_modelwc


def hqic(llf, nobs, df_modelwc):
    """Hannan-Quinn information criterion (HQC)

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

    """
    return -2. * llf + 2 * np.log(np.log(nobs)) * df_modelwc


# IC based on residual sigma

def aic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Akaike information criterion

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

    :math:`-2 llf + 2 k`

    in terms of :math:`\hat{\sigma}^2`

    :math:`log(\hat{\sigma}^2) + 2 k / n`

    in terms of the determinant of :math:`\hat{\Sigma}`

    :math:`log(\|\hat{\Sigma}\|) + 2 k / n`

    Note: In our definition we do not divide by n in the log-likelihood
    version.

    TODO: Latex math

    reference for example lecture notes by Herman Bierens

    See Also
    --------

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion

    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aic(0, nobs, df_modelwc) / nobs


def aicc_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Akaike information criterion (AIC) with small sample correction

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
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion#AICc

    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aicc(0, nobs, df_modelwc) / nobs


def bic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Bayesian information criterion (BIC) or Schwarz criterion

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
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    http://en.wikipedia.org/wiki/Bayesian_information_criterion

    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + bic(0, nobs, df_modelwc) / nobs


def hqic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Hannan-Quinn information criterion (HQC)

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
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    xxx

    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + hqic(0, nobs, df_modelwc) / nobs


def nash_sutcliff(mod, obs, axis=0, varient='standard', j=1):
    """Nash–Sutcliffe model efficiency coefficient (NSE)

    Parameters
    ----------
    mod : array_like
        Model estimates.
    obs : array_like
        Observations.
    axis : int
       axis along which the summary statistic is calculated
    varient : str
        Varient of the Nash-Sutcliffe model to use. Valid options are
        {'standard', 'modified', 'relative'}. Default is 'standard'. The
        standard option is equivalent to calculating the `rsquared` value.
    j : int
        The exponent to be used in the computation of the modified varient
        Nash-Sutcliffe effciency. The default value is j=1.

    Returns
    -------
    nse : ndarray or float
       Nash–Sutcliffe model efficiency coefficient along given axis.

    Notes
    -----
    If ``mod`` and ``obs`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    References
    ----------
    Krause, P., Boyle, D. P., and Bäse, F.: Comparison of different efficiency
    criteria for hydrological model assessment, Adv. Geosci., 5, 89-97, 2005.

    Legates and McCabe, 1999. Evaluating the use of "goodness-of-fit" measures
    in hydrologic and hydroclimatic model validation. Water Resources Research.
    v35 i1. 233-241.

    http://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    """
    mod = np.asanyarray(mod)
    obs = np.asanyarray(obs)

    # Mean values
    mean_obs = np.mean(obs, axis=axis)

    if varient.lower() in ('standard'):
        n = ((obs - mod) ** 2).sum(axis=axis)
        d = ((obs - mean_obs) ** 2).sum(axis=axis)
    elif varient.lower() == 'modified':
        if j <= 0:
            raise ValueError('Invalid value for j, must greater than zero')
        n = (np.abs(obs - mod) ** j).sum(axis=axis)
        d = (np.abs(obs - mean_obs) ** j).sum(axis=axis)
    elif varient.lower() == 'relative':
        n = (((obs - mod) / obs) ** 2).sum(axis=axis)
        d = (((obs - mean_obs) / mean_obs) ** 2).sum(axis=axis)
    else:
        raise ValueError('Unknown value for varient: %s' % varient)

    return 1 - (n / d)


def kling_gupta(mod, obs, axis=0, s=(1., 1., 1.), method='2009'):
    """Kling-Gupta model efficiency coefficient (NSE)

    Parameters
    ----------
    mod : array_like
        Model estimates.
    obs : array_like
        Observations.
    axis : int
        axis along which the summary statistic is calculated
    s : tuple of 3 ints
        Scaling factors.
    method : str


    Returns
    -------
    kge : ndarray or float
       Kling-Gupta model efficiency coefficient along given axis.

    Notes
    -----
    If ``mod`` and ``obs`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    References
    ----------
    Hoshin V. Gupta, Harald Kling, Koray K. Yilmaz, Guillermo F. Martinez,
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling, Journal of Hydrology,
    Volume 377, Issues 1-2, 20 October 2009, Pages 80-91,
    DOI: 10.1016/j.jhydrol.2009.08.003. ISSN 0022-1694.

    Kling, H., M. Fuchs, and M. Paulin (2012), Runoff conditions in the
    upper Danube basin under an ensemble of climate change scenarios,
    Journal of Hydrology, Volumes 424-425, 6 March 2012, Pages 264-277,
    DOI:10.1016/j.jhydrol.2012.01.011.
    """
    mod = np.asanyarray(mod)
    obs = np.asanyarray(obs)

    # Mean values
    mean_mod = np.mean(mod, axis=axis)
    mean_obs = np.mean(obs, axis=axis)

    # Standard deviations
    sigma_mod = np.std(mod, axis=axis)
    sigma_obs = np.std(obs, axis=axis)

    # Pearson product-moment correlation coefficient
    r = np.corrcoef(mod, obs)[0, 1]  # needs to support axis!

    # alpha is a measure of relative variability between modeled and observed
    # values (See Ref1)
    alpha = sigma_mod / sigma_obs

    # beta is the ratio between the mean of the simulated values to the mean of
    # observations
    beta = mean_mod / mean_obs

    # cv_mod is the coefficient of variation of the simulated values
    # [dimensionless]
    # cv_obs is the coefficient of variation of the observations
    # [dimensionless]
    cv_mod = sigma_mod / mean_mod
    cv_obs = sigma_obs / mean_obs

    # gamma is the variability ratio, which is used instead of alpha (See Ref2)
    gamma = cv_mod / cv_obs

    # Variability ratio depending on 'method'
    if method == '2012':
        vr = gamma
    elif method == '2009':
        vr = alpha
    else:
        raise ValueError('Unknown method %s' % method)

    # KGE Computation
    kge = 1 - np.sqrt((s[0] * (r - 1)) ** 2
                      + (s[1] * (vr - 1)) ** 2
                      + (s[2] * (beta - 1)) ** 2)
    return kge


# from var_model.py, VAR only? separates neqs and k_vars per equation
# def fpe_sigma():
#     ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)


__all__ = [maxabs, meanabs, medianabs, medianbias, mse, rmse, stde, vare,
           aic, aic_sigma, aicc, aicc_sigma, bias, bic, bic_sigma,
           hqic, hqic_sigma, iqr]
