import warnings

import numpy as np
from scipy.linalg.blas import find_best_blas_type

from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from statsmodels.tools.validation import array_like
from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map

from . import _arma_innovations

NON_STATIONARY_ERROR = """\
The model's autoregressive parameters (ar_params) indicate that the process
 is non-stationary. The innovations algorithm cannot be used.
"""


def _prefix_dtype(prefix):
    if prefix not in prefix_dtype_map:
        valid = ", ".join(f'"{key}"' for key in prefix_dtype_map)
        raise ValueError(f"prefix must be one of {valid}, got {prefix!r}.")
    return prefix_dtype_map[prefix]


def arma_innovations(endog, ar_params=None, ma_params=None, sigma2=1,
                     normalize=False, prefix=None):
    """
    Compute innovations using a given ARMA process

    Parameters
    ----------
    endog : array_like
        The observed time-series process, may be univariate or multivariate.
        If 2-d, each column is filtered separately using the same ARMA
        process.
    ar_params : array_like, optional
        Autoregressive parameters.
    ma_params : array_like, optional
        Moving average parameters.
    sigma2 : float, optional
        The ARMA innovation variance, which must be positive. Default is 1.
    normalize : bool, optional
        Whether or not to normalize the returned innovations. Default is False.
    prefix : {"s", "d", "c", "z"}, optional
        The BLAS prefix associated with the datatype: single or double
        precision, real ("s", "d") or complex ("c", "z"). Default is to find
        the best datatype based on given input. This argument is typically
        only used internally.

    Returns
    -------
    innovations : ndarray
        Innovations (one-step-ahead prediction errors) for the given `endog`
        series with predictions based on the given ARMA process, with the
        same shape as `endog`. If `normalize=True`, then the returned
        innovations have been "whitened" by dividing through by the square
        root of `innovations_mse`, so that they have variance `sigma2`.
    innovations_mse : ndarray
        Mean square error for the innovations divided by `sigma2`. The mean
        square error is ``sigma2 * innovations_mse``.
    """
    # Parameters
    endog = np.require(endog,  requirements="W")
    squeezed = endog.ndim == 1
    if squeezed:
        endog = endog[:, None]

    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    nobs, k_endog = endog.shape

    # Get BLAS prefix
    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = _prefix_dtype(prefix)

    # Make arrays contiguous for BLAS calls
    endog = np.asfortranarray(endog, dtype=dtype)
    # The Cython routines need writable arrays, so read-only parameters (for
    # example a pandas Series under copy-on-write) are copied
    ar_params = np.require(ar_params, dtype=dtype, requirements=["F", "W"])
    ma_params = np.require(ma_params, dtype=dtype, requirements=["F", "W"])
    sigma2 = dtype(sigma2).item()
    if not sigma2.real > 0:
        # 0 or NaN would give a misleading "non-stationary" error below
        raise ValueError(
            f"sigma2 must be a positive innovation variance, got {sigma2}."
        )
    # The lag polynomials are built from the converted parameters, so that
    # they have the dtype that matches the prefix
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # Get the appropriate functions
    arma_transformed_acovf_fast = getattr(
        _arma_innovations, prefix + "arma_transformed_acovf_fast")
    arma_innovations_algo_fast = getattr(
        _arma_innovations, prefix + "arma_innovations_algo_fast")
    arma_innovations_filter = getattr(
        _arma_innovations, prefix + "arma_innovations_filter")

    # Run the innovations algorithm for ARMA coefficients
    arma_acovf = arima_process.arma_acovf(
        ar, ma, sigma2=sigma2, nobs=nobs, dtype=dtype) / sigma2
    acovf, acovf2 = arma_transformed_acovf_fast(ar, ma, arma_acovf)
    theta, v = arma_innovations_algo_fast(nobs, ar_params, ma_params,
                                          acovf, acovf2)
    v = np.array(v)
    if (np.any(v < 0) or
            not np.isfinite(theta).all() or
            not np.isfinite(v).all()):
        # This is defensive code that is hard to hit
        raise ValueError(NON_STATIONARY_ERROR)

    # Run the innovations filter across each series
    u = []
    for i in range(k_endog):
        u_i = np.array(arma_innovations_filter(endog[:, i], ar_params,
                                               ma_params, theta))
        u.append(u_i)
    u = np.vstack(u).T
    if normalize:
        u /= v[:, None]**0.5

    # Post-processing
    if squeezed:
        u = u.squeeze()

    return u, v


def arma_loglike(endog, ar_params=None, ma_params=None, sigma2=1, prefix=None):
    """
    Compute the log-likelihood of the given data assuming an ARMA process

    Parameters
    ----------
    endog : array_like
        The observed time-series process, 1-d or a single column.
    ar_params : array_like, optional
        Autoregressive parameters.
    ma_params : array_like, optional
        Moving average parameters.
    sigma2 : float, optional
        The ARMA innovation variance. Default is 1.
    prefix : {"s", "d", "c", "z"}, optional
        The BLAS prefix associated with the datatype: single or double
        precision, real ("s", "d") or complex ("c", "z"). Default is to find
        the best datatype based on given input. This argument is typically
        only used internally.

    Returns
    -------
    float
        The joint loglikelihood.
    """
    llf_obs = arma_loglikeobs(endog, ar_params=ar_params, ma_params=ma_params,
                              sigma2=sigma2, prefix=prefix)
    return np.sum(llf_obs)


def arma_loglikeobs(endog, ar_params=None, ma_params=None, sigma2=1,
                    prefix=None):
    """
    Compute the log-likelihood for each observation assuming an ARMA process

    Parameters
    ----------
    endog : array_like
        The observed time-series process, 1-d or a single column.
    ar_params : array_like, optional
        Autoregressive parameters.
    ma_params : array_like, optional
        Moving average parameters.
    sigma2 : float, optional
        The ARMA innovation variance. Default is 1.
    prefix : {"s", "d", "c", "z"}, optional
        The BLAS prefix associated with the datatype: single or double
        precision, real ("s", "d") or complex ("c", "z"). Default is to find
        the best datatype based on given input. This argument is typically
        only used internally.

    Returns
    -------
    ndarray
        Array of loglikelihood values for each observation.
    """
    # A single column is squeezed to 1-d. The dtype is kept, so that the BLAS
    # prefix is found from the data.
    endog = array_like(endog, "endog", dtype=None, ndim=1)
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = _prefix_dtype(prefix)

    endog = np.ascontiguousarray(endog, dtype=dtype)
    # The Cython routines need writable arrays, so read-only parameters (for
    # example a pandas Series under copy-on-write) are copied
    ar_params = np.require(ar_params, dtype=dtype, requirements=["F", "W"])
    ma_params = np.require(ma_params, dtype=dtype, requirements=["F", "W"])
    sigma2 = dtype(sigma2).item()

    func = getattr(_arma_innovations, prefix + "arma_loglikeobs_fast")
    return func(endog, ar_params, ma_params, sigma2)


def _warn_unused_prefix(prefix, func_name):
    # The score is the complex-step derivative of the loglikelihood, which is
    # always evaluated in double precision (complex128), so prefix is not used
    if prefix is not None:
        warnings.warn(
            f"prefix is not used by {func_name} and is deprecated. It will be "
            "removed after statsmodels 0.17 is released.",
            FutureWarning,
            stacklevel=3,
        )


def arma_score(endog, ar_params=None, ma_params=None, sigma2=1,
               prefix=None):
    """
    Compute the score (gradient of the log-likelihood function)

    Parameters
    ----------
    endog : array_like
        The observed time-series process, 1-d or a single column.
    ar_params : array_like, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : array_like, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e., they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : float, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        Not used. The score is always computed in double precision.

        .. deprecated:: 0.16.0

           `prefix` has no effect. It is deprecated and will be removed after
           statsmodels 0.17 is released.

    Returns
    -------
    ndarray
        Score, evaluated at the given parameters. The elements are the
        derivatives with respect to `ar_params`, `ma_params` and `sigma2`, in
        that order.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    _warn_unused_prefix(prefix, "arma_score")
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

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
    Compute the score (gradient) per observation

    Parameters
    ----------
    endog : array_like
        The observed time-series process, 1-d or a single column.
    ar_params : array_like, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : array_like, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e., they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : float, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        Not used. The score is always computed in double precision.

        .. deprecated:: 0.16.0

           `prefix` has no effect. It is deprecated and will be removed after
           statsmodels 0.17 is released.

    Returns
    -------
    ndarray
        Score per observation, evaluated at the given parameters, with one
        row per observation. The columns are the derivatives with respect to
        `ar_params`, `ma_params` and `sigma2`, in that order.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglikeobs` method.
    """
    _warn_unused_prefix(prefix, "arma_scoreobs")
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglikeobs(endog, params[:p], params[p:p + q],
                               params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)
