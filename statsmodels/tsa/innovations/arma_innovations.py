import numpy as np

from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from . import _arma_innovations


def arma_loglike(endog, ar_params=None, ma_params=None, sigma2=1, prefix=None):
    """
    Compute loglikelihood of the given data assuming an ARMA process

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    loglike : numeric
        The joint loglikelihood.

    """
    llf_obs = arma_loglikeobs(endog, ar_params=ar_params, ma_params=ma_params,
                              sigma2=sigma2, prefix=prefix)
    return np.sum(llf_obs)


def arma_loglikeobs(endog, ar_params=None, ma_params=None, sigma2=1,
                    prefix=None):
    """
    Compute loglikelihood for each observation assuming an ARMA process

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    loglikeobs : array of numeric
        Array of loglikelihood values for each observation.

    """
    endog = np.array(endog)
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = prefix_dtype_map[prefix]

    endog = np.ascontiguousarray(endog, dtype=dtype)
    ar_params = np.asfortranarray(ar_params, dtype=dtype)
    ma_params = np.asfortranarray(ma_params, dtype=dtype)
    sigma2 = np.asscalar(dtype(sigma2))

    func = getattr(_arma_innovations, prefix + 'arma_loglikeobs_fast')
    return func(endog, ar_params, ma_params, sigma2)


def arma_score(endog, ar_params=None, ma_params=None, sigma2=1,
               prefix=None):
    """
    Compute the score (gradient of the loglikelihood function)

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    ----------
    score : array
        Score, evaluated at the given parameters.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    ar_params = [] if ar_params is None else ar_params
    ma_params = [] if ma_params is None else ma_params

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglike(endog, params[:p], params[p:p + q], params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)


def arma_scoreobs(endog, ar_params=None, ma_params=None, sigma2=1,
                  prefix=None):
    """
    Compute the score per observation (gradient of the loglikelihood function)

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    ----------
    scoreobs : array
        Score per observation, evaluated at the given parameters.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    ar_params = [] if ar_params is None else ar_params
    ma_params = [] if ma_params is None else ma_params

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglikeobs(endog, params[:p], params[p:p + q],
                               params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)
