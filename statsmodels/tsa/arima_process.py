"""
ARMA process and estimation with scipy.signal.lfilter

Notes
-----
* written without textbook, works but not sure about everything
  briefly checked and it looks to be standard least squares, see below

* theoretical autocorrelation function of general ARMA
  Done, relatively easy to guess solution, time consuming to get
  theoretical test cases, example file contains explicit formulas for
  acovf of MA(1), MA(2) and ARMA(1,1)

Properties:
Judge, ... (1985): The Theory and Practice of Econometrics

Author: josefpktd
License: BSD
"""
from statsmodels.compat.numpy import ComplexWarning

import functools
import inspect
import textwrap
import warnings

import numpy as np
from scipy import linalg, optimize, signal

from statsmodels.tools.docstring import Docstring, Parameter
from statsmodels.tools.docstring_helpers import Appender
from statsmodels.tools.validation import (
    array_like,
    int_like,
    is_int_like,
    string_like,
)

__all__ = [
    "arma2ar",
    "arma2ma",
    "arma_acf",
    "arma_acovf",
    "arma_generate_sample",
    "arma_impulse_response",
    "deconvolve",
    "index2lpol",
    "lpol2index",
]


NONSTATIONARY_ERROR = """\
The model's autoregressive parameters (ar) indicate that the process
 is non-stationary. arma_acovf can only be used with stationary processes.
"""


def _validate_lags(value, name):
    # The number of values to return must be a positive integer. 0 used to
    # raise an IndexError in the module level functions, and to select the
    # default in the ArmaProcess methods.
    value = int_like(value, name)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


# The start of the FutureWarning of _require_1d
_NOT_1D_DEPRECATED = r"\w+ must be 1-D\. Passing an array with"


def _require_1d(*names):
    """
    Check that the lag polynomial arguments of a function are 1-D

    Parameters
    ----------
    *names : str
        The names of the arguments that must be 1-D.

    Returns
    -------
    callable
        A decorator. The decorated function runs as before for 1-D input.
        For input with more than one dimension, it is run in the same way.
        If it returns, the result is returned with a FutureWarning, as this
        input is deprecated. If it raises, a ValueError that names the
        argument is raised instead.
    """

    def decorator(func):
        params = list(inspect.signature(func).parameters)
        positions = {name: params.index(name) for name in names}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            not_1d = []
            for name, pos in positions.items():
                value = args[pos] if pos < len(args) else kwargs.get(name)
                if np.ndim(value) > 1:
                    not_1d.append((name, np.shape(value)))
            if not not_1d:
                return func(*args, **kwargs)
            try:
                with warnings.catch_warnings():
                    # The functions that this one calls would warn again
                    warnings.filterwarnings(
                        "ignore", _NOT_1D_DEPRECATED, FutureWarning
                    )
                    result = func(*args, **kwargs)
            except Exception as exc:
                name, shape = not_1d[0]
                raise ValueError(
                    f"{name} must be 1-D, got an array with shape {shape}"
                ) from exc
            for name, shape in not_1d:
                warnings.warn(
                    f"{name} must be 1-D. Passing an array with {len(shape)} "
                    "dimensions is deprecated and will raise a ValueError after "
                    "statsmodels 0.17 is released.",
                    FutureWarning,
                    stacklevel=2,
                )
            return result

        return wrapper

    return decorator


@_require_1d("ar", "ma")
def arma_generate_sample(
    ar, ma, nsample, scale=1, distrvs=None, axis=0, burnin=0
):
    """
    Simulate data from an ARMA

    Parameters
    ----------
    ar : array_like
        The coefficient for autoregressive lag polynomial, including zero lag.
        Must be 1-D.
    ma : array_like
        The coefficient for moving-average lag polynomial, including zero lag.
        Must be 1-D.
    nsample : int or tuple of ints
        If nsample is an integer, then this creates a 1d timeseries of
        length size. If nsample is a tuple, creates a len(nsample)
        dimensional time series where time is indexed along the input
        variable ``axis``. All series are independent unless ``distrvs``
        generates dependent data.
    scale : float, optional
        The standard deviation of noise.
    distrvs : callable, optional
        A function that generates the random numbers, and takes ``size``
        as argument. The default is np.random.standard_normal.
    axis : int, optional
        See nsample for details.
    burnin : int, optional
        Number of observation at the beginning of the sample to drop.
        Used to reduce dependence on initial values.

    Returns
    -------
    ndarray
        Random sample(s) from an ARMA process.

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR parameters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)
    >>> model = sm.tsa.ARIMA(y, order=(2, 0, 2), trend='n').fit()
    >>> model.params
    array([ 0.790456  , -0.2314185 ,  0.70071724,  0.40606905,  0.98005244])
    """
    distrvs = np.random.standard_normal if distrvs is None else distrvs
    if np.ndim(nsample) == 0:
        nsample = [nsample]
    if burnin:
        # handle burin time for nd arrays
        # maybe there is a better trick in scipy.fft code
        newsize = list(nsample)
        newsize[axis] += burnin
        newsize = tuple(newsize)
        fslice = [slice(None)] * len(newsize)
        fslice[axis] = slice(burnin, None, None)
        fslice = tuple(fslice)
    else:
        newsize = tuple(nsample)
        fslice = tuple([slice(None)] * np.ndim(newsize))
    eta = scale * distrvs(size=newsize)
    return signal.lfilter(ma, ar, eta, axis=axis)[fslice]


@_require_1d("ar", "ma")
def arma_acovf(ar, ma, nobs=10, sigma2=1, dtype=None):
    """
    Theoretical autocovariances of stationary ARMA processes

    Parameters
    ----------
    ar : array_like
        The coefficients for autoregressive lag polynomial, including zero lag.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    ma : array_like
        The coefficients for moving-average lag polynomial, including zero lag.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    nobs : int, optional
        The number of terms (lags plus zero lag) to include in returned acovf.
    sigma2 : float, optional
        Variance of the innovation term.
    dtype : str, optional
        Numpy dtype to use for the output. If None (the default), the dtype
        is inferred from the common type of `ar`, `ma`, and `sigma2`.

    Returns
    -------
    ndarray
        The autocovariance of ARMA process given by ar, ma.

    See Also
    --------
    arma_acf : Autocorrelation function for ARMA processes.
    acovf : Sample autocovariance estimation.

    References
    ----------
    .. [*] Brockwell, Peter J., and Richard A. Davis. 2009. Time Series:
        Theory and Methods. 2nd ed. 1991. New York, NY: Springer.
    """
    if dtype is None:
        dtype = np.common_type(np.array(ar), np.array(ma), np.array(sigma2))

    nobs = int_like(nobs, "nobs", optional=False)
    if nobs < 1:
        raise ValueError(f"nobs must be a positive integer, got {nobs}")

    p = len(ar) - 1
    q = len(ma) - 1
    m = max(p, q) + 1

    if sigma2.real < 0:
        raise ValueError("Must have positive innovation variance.")

    # Short-circuit for trivial corner-case
    if p == q == 0:
        out = np.zeros(nobs, dtype=dtype)
        out[0] = sigma2
        return out
    elif p > 0 and np.max(np.abs(np.roots(ar))) >= 1:
        raise ValueError(NONSTATIONARY_ERROR)

    # Get the moving average representation coefficients that we need
    ma_coeffs = arma2ma(ar, ma, lags=m)

    # Solve for the first m autocovariances via the linear system
    # described by (BD, eq. 3.3.8)
    A = np.zeros((m, m), dtype=dtype)
    b = np.zeros((m, 1), dtype=dtype)
    # We need a zero-right-padded version of ar params
    tmp_ar = np.zeros(m, dtype=dtype)
    tmp_ar[: p + 1] = ar
    for k in range(m):
        A[k, : (k + 1)] = tmp_ar[: (k + 1)][::-1]
        A[k, 1 : m - k] += tmp_ar[(k + 1) : m]
        b[k] = sigma2 * np.dot(ma[k : q + 1], ma_coeffs[: max((q + 1 - k), 0)])
    acovf = np.zeros(max(nobs, m), dtype=dtype)
    try:
        acovf[:m] = np.linalg.solve(A, b)[:, 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError(NONSTATIONARY_ERROR) from exc

    # Iteratively apply (BD, eq. 3.3.9) to solve for remaining autocovariances
    if nobs > m:
        zi = signal.lfiltic([1], ar, acovf[:m:][::-1])
        acovf[m:] = signal.lfilter(
            [1], ar, np.zeros(nobs - m, dtype=dtype), zi=zi
        )[0]

    return acovf[:nobs]


@_require_1d("ar", "ma")
def arma_acf(ar, ma, lags=10):
    """
    Theoretical autocorrelation function of an ARMA process

    Parameters
    ----------
    ar : array_like
        Coefficients for autoregressive lag polynomial, including zero lag.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    ma : array_like
        Coefficients for moving-average lag polynomial, including zero lag.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    lags : int, optional
        The number of terms (lags plus zero lag) to include in returned acf.

    Returns
    -------
    ndarray
        The autocorrelations of ARMA process given by ar and ma.

    See Also
    --------
    arma_acovf : Autocovariances from ARMA processes.
    acf : Sample autocorrelation function estimation.
    acovf : Sample autocovariance function estimation.
    """
    acovf = arma_acovf(ar, ma, lags)
    return acovf / acovf[0]


@_require_1d("ar", "ma")
def arma_pacf(ar, ma, lags=10):
    """
    Theoretical partial autocorrelation function of an ARMA process

    Parameters
    ----------
    ar : array_like
        The coefficients for autoregressive lag polynomial, including zero lag.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    ma : array_like
        The coefficients for moving-average lag polynomial, including zero lag.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    lags : int, optional
        The number of terms (lags plus zero lag) to include in returned pacf.
        Must be a positive integer.

        .. versionchanged:: 0.15.1

           ``lags=0`` raises a ``ValueError``. It used to raise an
           ``IndexError``. An integral float such as 7.0 is accepted.

    Returns
    -------
    ndarray
        The partial autocorrelation of ARMA process given by ar and ma.

    Notes
    -----
    Solves the Yule-Walker equation for each lag order up to ``lags``.
    """
    lags = _validate_lags(lags, "lags")
    # TODO: Should use rank 1 inverse update
    apacf = np.zeros(lags)
    acov = arma_acf(ar, ma, lags=lags + 1)

    apacf[0] = 1.0
    for k in range(2, lags + 1):
        r = acov[:k]
        apacf[k - 1] = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])[-1]
    return apacf


# The values of normalization in arma_periodogram. Omitting it gives
# "sqrt(2pi)" with a FutureWarning; the default will be "2pi" after 0.17.
_PERIODOGRAM_NORMALIZATIONS = ("sqrt(2pi)", "2pi")


def _periodogram_normalization(normalization, func, stacklevel):
    # Validate normalization, and warn if it is omitted. stacklevel is the
    # stacklevel of the warning in the caller of this function.
    normalization = string_like(
        normalization,
        "normalization",
        optional=True,
        options=_PERIODOGRAM_NORMALIZATIONS,
    )
    if normalization is None:
        warnings.warn(
            f"The spectral density returned by {func} is divided by "
            "sqrt(2 * pi), not by 2 * pi as in the usual definition. The "
            "default will change to 2 * pi after statsmodels 0.17 is "
            'released. Set normalization="2pi" to switch now, or '
            'normalization="sqrt(2pi)" to keep the current result and '
            "silence this warning.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        normalization = "sqrt(2pi)"
    return normalization


@_require_1d("ar", "ma")
def arma_periodogram(ar, ma, worN=None, whole=0, *, normalization=None):
    """
    Periodogram for ARMA process given by lag-polynomials ar and ma

    Parameters
    ----------
    ar : array_like
        The autoregressive lag-polynomial with leading 1 and lhs sign.
        Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result, such as a column with shape
           (n, 1), is deprecated and will raise a ValueError after
           statsmodels 0.17 is released. Other 2-D input raises a
           ValueError.
    ma : array_like
        The moving average lag-polynomial with leading 1. Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result, such as a column with shape
           (n, 1), is deprecated and will raise a ValueError after
           statsmodels 0.17 is released. Other 2-D input raises a
           ValueError.
    worN : int or array_like, optional
        An option for scipy.signal.freqz (read "w or N").
        If None, then compute at 512 frequencies around the unit circle.
        If a single integer, the compute at that many frequencies.
        Otherwise, compute the response at frequencies given in worN.
        An integer must be positive.

        .. versionchanged:: 0.15.1

           ``worN=0`` raises a ``ValueError``. It used to return empty
           arrays. A negative integer raises the same ``ValueError``, as
           does a 0-d integer array that holds 0 or a negative integer. A
           float, such as 2.0, is a frequency.
    whole : bool, optional
        An option for scipy.signal.freqz.
        Normally, frequencies are computed from 0 to pi (upper-half of
        unit-circle.  If whole is non-zero compute frequencies from 0 to 2*pi.
    normalization : {"sqrt(2pi)", "2pi"}, optional
        The constant that ``abs(ma(z)) ** 2 / abs(ar(z)) ** 2`` is divided
        by, see the Notes. "2pi" gives the spectral density of the process.
        "sqrt(2pi)" divides by ``sqrt(2 * pi)`` instead, which is the result
        of earlier versions. If not given, "sqrt(2pi)" is used and a
        ``FutureWarning`` is issued.

        .. versionadded:: 0.15.1

        .. deprecated:: 0.15.1

           The default will change to "2pi" after statsmodels 0.17 is
           released. Set ``normalization="2pi"`` to switch now, or
           ``normalization="sqrt(2pi)"`` to keep the current result and
           silence the warning.

    Returns
    -------
    w : ndarray
        The frequencies.
    sd : ndarray
        The periodogram, also known as the spectral density.

    Notes
    -----
    With ``normalization="2pi"``, the spectral density is computed as
    ``abs(ma(z)) ** 2 / abs(ar(z)) ** 2 / (2 * pi)`` with
    ``z = exp(-1j * w)``, where ``ar(z)`` and ``ma(z)`` are the lag
    polynomials. This is the spectral density of the process with a unit
    innovation variance, whose integral over ``[-pi, pi]`` is the variance
    of the process. ``normalization="sqrt(2pi)"`` divides by
    ``sqrt(2 * pi)`` instead.

    This uses signal.freqz, which does not use fft. There is a fft version
    somewhere.
    """
    # An integer is the number of frequencies, as in signal.freqz. A float is
    # a frequency (strict).
    if is_int_like(worN, strict=True):
        worN = int(worN)
        if worN < 1:
            raise ValueError(f"worN must be a positive integer, got {worN}")
    w, h = signal.freqz(ma, ar, worN=worN, whole=whole)
    # The caller is 3 frames up, as _require_1d wraps this function
    normalization = _periodogram_normalization(
        normalization, "arma_periodogram", stacklevel=3
    )
    scale = 2 * np.pi if normalization == "2pi" else np.sqrt(2 * np.pi)
    sd = np.abs(h) ** 2 / scale
    if np.any(np.isnan(h)):
        # this happens with unit root or seasonal unit root'
        import warnings

        warnings.warn(
            "Warning: nan in frequency response h, maybe a unit root",
            RuntimeWarning,
            # 3 skips the wrapper of _require_1d
            stacklevel=3,
        )
    return w, sd


@_require_1d("ar", "ma")
def arma_impulse_response(ar, ma, leads=100):
    """
    Compute the impulse response function (MA representation) for ARMA process

    Parameters
    ----------
    ar : array_like
        The auto regressive lag polynomial. Must be 1-D.
    ma : array_like
        The moving average lag polynomial. Must be 1-D.
    leads : int, optional
        The number of observations to calculate.
        Must be a positive integer.

        .. versionchanged:: 0.15.1

           ``leads=0`` raises a ``ValueError``. It used to raise an
           ``IndexError``. An integral float such as 7.0 is accepted.

    Returns
    -------
    ndarray
        The impulse response function with ``leads`` elements.

    Notes
    -----
    This is the same as finding the MA representation of an ARMA(p,q).
    By reversing the role of ar and ma in the function arguments, the
    returned result is the AR representation of an ARMA(p,q), i.e

    ma_representation = arma_impulse_response(ar, ma, leads=100)
    ar_representation = arma_impulse_response(ma, ar, leads=100)

    Fully tested against MATLAB

    Examples
    --------
    AR(1)

    >>> arma_impulse_response([1.0, -0.8], [1.], leads=10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    this is the same as

    >>> 0.8**np.arange(10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    MA(2)

    >>> arma_impulse_response([1.0], [1., 0.5, 0.2], leads=10)
    array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])

    ARMA(1,2)

    >>> arma_impulse_response([1.0, -0.8], [1., 0.5, 0.2], leads=10)
    array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,
            0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])
    """
    leads = _validate_lags(leads, "leads")
    impulse = np.zeros(leads)
    impulse[0] = 1.0
    return signal.lfilter(ma, ar, impulse)


def arma2ma(ar, ma, lags=100):
    """
    A finite-lag approximate MA representation of an ARMA process

    Parameters
    ----------
    ar : array_like
        The auto regressive lag polynomial. Must be 1-D.
    ma : array_like
        The moving average lag polynomial. Must be 1-D.
    lags : int, optional
        The number of coefficients to calculate.
        Must be a positive integer.

        .. versionchanged:: 0.15.1

           ``lags=0`` raises a ``ValueError``. It used to raise an
           ``IndexError``. An integral float such as 7.0 is accepted.

    Returns
    -------
    ndarray
        The coefficients of MA lag polynomial with ``lags`` elements.

    Notes
    -----
    Equivalent to ``arma_impulse_response(ar, ma, leads=lags)``
    """
    lags = _validate_lags(lags, "lags")
    return arma_impulse_response(ar, ma, leads=lags)


@_require_1d("ar", "ma")
def arma2ar(ar, ma, lags=100):
    """
    A finite-lag AR approximation of an ARMA process

    Parameters
    ----------
    ar : array_like
        The auto regressive lag polynomial. Must be 1-D.
    ma : array_like
        The moving average lag polynomial. Must be 1-D.
    lags : int, optional
        The number of coefficients to calculate.
        Must be a positive integer.

        .. versionchanged:: 0.15.1

           ``lags=0`` raises a ``ValueError``. It used to raise an
           ``IndexError``. An integral float such as 7.0 is accepted.

    Returns
    -------
    ndarray
        The coefficients of AR lag polynomial with ``lags`` elements.

    Notes
    -----
    Equivalent to ``arma_impulse_response(ma, ar, leads=lags)``
    """
    lags = _validate_lags(lags, "lags")
    return arma_impulse_response(ma, ar, leads=lags)


# moved from sandbox.tsa.try_fi
@_require_1d("ar_des")
def ar2arma(ar_des, p, q, n=20, mse="ar", start=None):
    """
    Find arma approximation to ar process

    This finds the ARMA(p,q) coefficients that minimize the integrated
    squared difference between the impulse_response functions (MA
    representation) of the AR and the ARMA process. This does not check
    whether the MA lag polynomial of the ARMA process is invertible, neither
    does it check the roots of the AR lag polynomial.

    Parameters
    ----------
    ar_des : array_like
        The target impulse response/AR-representation to approximate,
        including lag zero, e.g. the (zero-padded or truncated) coefficients
        of the original AR lag polynomial. Must have ``n`` elements.
        Must be 1-D.
    p : int
        The length of desired AR lag polynomials.
    q : int
        The length of desired MA lag polynomials.
    n : int, optional
        The number of terms of the impulse_response function to include in the
        objective function for the approximation.
    mse : str, optional
        Not used.
    start : array_like, optional
        Initial values to use when finding the approximation.

    Returns
    -------
    ar_app : ndarray
        The coefficients of the AR lag polynomials of the approximation.
    ma_app : ndarray
        The coefficients of the MA lag polynomials of the approximation.
    res : tuple
        The result of optimize.leastsq.

    Notes
    -----
    Extension is possible if we want to match autocovariance instead
    of impulse response function.

    .. versionchanged:: 0.15.1

       ``ar_des`` can be a tuple. It used to raise a TypeError.
    """

    # TODO: convert MA lag polynomial, ma_app, to be invertible, by mirroring
    # TODO: roots outside the unit interval to ones that are inside. How to do
    # TODO: this?

    # p,q = pq
    def msear_err(arma, ar_des):
        ar, ma = np.r_[1, arma[: p - 1]], np.r_[1, arma[p - 1 :]]
        ar_approx = arma_impulse_response(ma, ar, n)
        return ar_des - ar_approx  # ((ar - ar_approx)**2).sum()

    if start is None:
        arma0 = np.r_[-0.9 * np.ones(p - 1), np.zeros(q - 1)]
    else:
        arma0 = start
    # leastsq would unpack a tuple ar_des into separate arguments
    res = optimize.leastsq(msear_err, arma0, args=(ar_des,), maxfev=5000)
    arma_app = np.atleast_1d(res[0])
    ar_app = np.r_[1, arma_app[: p - 1]]
    ma_app = np.r_[1, arma_app[p - 1 :]]
    return ar_app, ma_app, res


_arma_docs = {"ar": arma2ar.__doc__, "ma": arma2ma.__doc__}


@_require_1d("ar")
def lpol2index(ar):
    """
    Remove zeros from lag polynomial

    Parameters
    ----------
    ar : array_like
        coefficients of lag polynomial. Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input is deprecated and will raise a ValueError after
           statsmodels 0.17 is released. It returns the rows with a
           non-zero element and their index.

    Returns
    -------
    coeffs : ndarray
        non-zero coefficients of lag polynomial
    index : ndarray
        index (lags) of lag polynomial with non-zero elements
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        ar = array_like(ar, "ar")
    index = np.nonzero(ar)[0]
    coeffs = ar[index]
    return coeffs, index


@_require_1d("coeffs", "index")
def index2lpol(coeffs, index):
    """
    Expand coefficients to lag poly

    Parameters
    ----------
    coeffs : array_like
        non-zero coefficients of lag polynomial. Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result, such as a row with shape
           (1, n), is deprecated and will raise a ValueError after
           statsmodels 0.17 is released. Other 2-D input raises a
           ValueError.
    index : array_like of int
        index (lags) of lag polynomial with non-zero elements. Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.

    Returns
    -------
    ar : ndarray
        coefficients of lag polynomial

    Notes
    -----
    .. versionchanged:: 0.15.1

       ``index`` can be a tuple. It used to raise an IndexError.
    """
    n = max(index)
    ar = np.zeros(n + 1)
    # A tuple would index more than one dimension
    ar[np.asarray(index)] = coeffs
    return ar


def lpol_fima(d, n=20):
    """
    MA representation of fractional integration

    .. math:: (1-L)^{-d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int, optional
        number of terms to calculate, including lag zero

    Returns
    -------
    ma : ndarray
        coefficients of lag polynomial
    """
    # psi_j = psi_{j-1} * (j - 1 + d) / j with psi_0 = 1, this is valid for all
    # d, including d == 0 and d < 0 where the gamma function is negative
    j = np.arange(1, n)
    return np.r_[1.0, np.cumprod((j - 1 + d) / j)][:n]


# moved from sandbox.tsa.try_fi
def lpol_fiar(d, n=20):
    """
    AR representation of fractional integration

    .. math:: (1-L)^{d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int, optional
        number of terms to calculate, including lag zero

    Returns
    -------
    ar : ndarray
        coefficients of lag polynomial

    Notes
    -----
    first coefficient is 1, for 0 < d < 1 the other coefficients are negative,
    ar(L)*x_t
    """
    # pi_j = pi_{j-1} * (j - 1 - d) / j with pi_0 = 1, this is valid for all d,
    # including d == 0 and d < 0
    j = np.arange(1, n)
    return np.r_[1.0, np.cumprod((j - 1 - d) / j)][:n]


# moved from sandbox.tsa.try_fi
def lpol_sdiff(s):
    """
    Return coefficients for seasonal difference (1-L^s)

    just a trivial convenience function

    Parameters
    ----------
    s : int
        number of periods in season

    Returns
    -------
    sdiff : list, length s+1
    """
    return [1] + [0] * (s - 1) + [-1]


@_require_1d("num", "den")
def deconvolve(num, den, n=None):
    """
    Deconvolves divisor out of signal, division of polynomials for n terms

    calculates den^{-1} * num

    Parameters
    ----------
    num : array_like
        signal or lag polynomial. Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    den : array_like
        coefficients of lag polynomial (linear filter). Must be 1-D.

        .. deprecated:: 0.15.1

           2-D input that returned a result is deprecated and will raise a
           ValueError after statsmodels 0.17 is released. Other 2-D input
           raises a ValueError.
    n : int, optional
        number of terms of quotient

    Returns
    -------
    quot : ndarray
        quotient or filtered series. An empty array if ``den`` is longer than
        ``num`` and ``n`` is not given.

        .. versionchanged:: 0.15.1

           The empty quotient is an ndarray with the dtype of the quotient
           of the other cases, float64 or complex128. It used to be an
           empty list, as it still is for the deprecated 2-D input.
    rem : ndarray
        remainder. A copy of ``num`` if ``den`` is longer than ``num`` and
        ``n`` is not given.

        .. versionchanged:: 0.15.1

           The copy of ``num`` has the dtype of the remainder of the other
           cases, float64 or complex128. It used to be ``num`` with its own
           dtype, the same object if ``num`` was a 1-D ndarray.

    Notes
    -----
    If num is a time series, then this applies the linear filter den^{-1}.
    If both num and den are lag polynomials, then this calculates the
    quotient polynomial for n terms and also returns the remainder.

    This is copied from scipy.signal.signaltools and added n as optional
    parameter.
    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    N = len(num)
    D = len(den)
    if D > N and n is None:
        if num.ndim > 1 or den.ndim > 1:
            # The deprecated 2-D input keeps its result
            quot = []
            rem = num
        else:
            # The dtype of the quotient and the remainder in the other branch
            dtype = np.result_type(num, den, float)
            quot = np.zeros(0, dtype=dtype)
            rem = num.astype(dtype)
    else:
        if n is None:
            n = N - D + 1
        impulse = np.zeros(n, float)
        impulse[0] = 1
        quot = signal.lfilter(num, den, impulse)
        num_approx = signal.convolve(den, quot, mode="full")
        if len(num) < len(num_approx):  # 1d only ?
            num = np.concatenate((num, np.zeros(len(num_approx) - len(num))))
        rem = num - num_approx
    return quot, rem


def _arma_lags_method_doc(func, name, description, changes=(), keep=()):
    """
    Build the docstring of an ArmaProcess method from a module level function

    Parameters
    ----------
    func : callable
        The module level function that the method calls.
    name : str
        The name of the parameter of the method that sets the number of
        values that it returns.
    description : str
        The description of the parameter.
    changes : sequence of str, optional
        Earlier changes of the parameter, for its versionchanged note.
    keep : sequence of str, optional
        The parameters of func that the method also takes. They are
        documented after the parameter, as in func.

    Returns
    -------
    str
        The docstring of func with the parameter of the method, and the
        parameters in keep, in place of the parameters of func. The
        description of the parameter states the default value, that the
        value must be positive, and the behavior changes.
    """
    if func.__doc__ is None or not func.__doc__.strip():
        # No docstring (python -OO) or a blank one (Nuitka), which Docstring
        # does not parse
        return func.__doc__
    ds = Docstring(func.__doc__)
    desc = textwrap.wrap(
        f"{description} If not given, the ``nobs`` attribute of the process "
        "is used. Must be a positive integer.",
        width=72,
    )
    change = (
        f"``{name}=0`` raises a ``ValueError``. It used to select the "
        "``nobs`` attribute, as omitting the value does."
    )
    note = textwrap.wrap(
        " ".join([*changes, change]),
        width=72,
        initial_indent="    ",
        subsequent_indent="    ",
    )
    desc += ["", ".. versionchanged:: 0.15.1", "", *note]
    params = [Parameter(name, "int, optional", desc)]
    params += [param for param in ds._ds["Parameters"] if param.name in keep]
    ds.replace_block("Parameters", params)
    return str(ds)


_generate_sample_doc = Docstring(arma_generate_sample.__doc__)
_generate_sample_doc.remove_parameters(["ar", "ma", "nsample"])
_generate_sample_doc.insert_parameters(
    None,
    Parameter(
        "nsample",
        "int or tuple of int, optional",
        [
            "If nsample is an integer, then this creates a 1d timeseries of",
            "length size. If nsample is a tuple, creates a len(nsample)",
            "dimensional time series where time is indexed along the input",
            "variable ``axis``. All series are independent unless ``distrvs``",
            "generates dependent data. If not given, the ``nobs`` attribute",
            "of the process is used.",
            "",
            ".. versionchanged:: 0.15.1",
            "",
            "    The default is the ``nobs`` attribute of the process. It used",
            "    to be 100, whatever the value of ``nobs``.",
        ],
    ),
)
_generate_sample_doc.replace_block("Notes", [])
_generate_sample_doc.replace_block("Examples", [])


class ArmaProcess:
    r"""
    Theoretical properties of an ARMA process for specified lag-polynomials

    Parameters
    ----------
    ar : array_like, optional
        Coefficient for autoregressive lag polynomial, including zero lag.
        Must be entered using the signs from the lag polynomial representation.
        See the notes for more information about the sign. If not provided,
        defaults to ``[1.0]``. Must be 1-D.
    ma : array_like, optional
        Coefficient for moving-average lag polynomial, including zero lag.
        If not provided, defaults to ``[1.0]``. Must be 1-D.
    nobs : int, optional
        Length of simulated time series. Used, for example, if a sample is
        generated. See example.

    Notes
    -----
    Both the AR and MA components must include the coefficient on the
    zero-lag. In almost all cases these values should be 1. Further, due to
    using the lag-polynomial representation, the AR parameters should
    have the opposite sign of what one would write in the ARMA representation.
    See the examples below.

    The ARMA(p,q) process is described by

    .. math::

        y_{t}=\phi_{1}y_{t-1}+\ldots+\phi_{p}y_{t-p}+\theta_{1}\epsilon_{t-1}
               +\ldots+\theta_{q}\epsilon_{t-q}+\epsilon_{t}

    and the parameterization used in this function uses the lag-polynomial
    representation,

    .. math::

        \left(1-\phi_{1}L-\ldots-\phi_{p}L^{p}\right)y_{t} =
            \left(1+\theta_{1}L+\ldots+\theta_{q}L^{q}\right)\epsilon_{t}

    Examples
    --------
    ARMA(2,2) with AR coefficients 0.75 and -0.25, and MA coefficients 0.65 and 0.35

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> arma_process = sm.tsa.ArmaProcess(ar, ma)
    >>> arma_process.isstationary
    True
    >>> arma_process.isinvertible
    True
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    >>> y = arma_process.generate_sample(250)
    >>> model = sm.tsa.ARIMA(y, order=(2, 0, 2), trend='n').fit()
    >>> model.params
    array([ 0.790456  , -0.2314185 ,  0.70071724,  0.40606905,  0.98005244])

    The same ARMA(2,2) Using the from_coeffs class method

    >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    """

    # TODO: Check unit root behavior
    @_require_1d("ar", "ma")
    def __init__(self, ar=None, ma=None, nobs=100):
        if ar is None:
            ar = np.array([1.0])
        if ma is None:
            ma = np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexWarning)
            self.ar = array_like(ar, "ar")
            self.ma = array_like(ma, "ma")
        self.arcoefs = -self.ar[1:]
        self.macoefs = self.ma[1:]
        self.arpoly = np.polynomial.Polynomial(self.ar)
        self.mapoly = np.polynomial.Polynomial(self.ma)
        self.nobs = nobs

    @classmethod
    @_require_1d("maroots", "arroots")
    def from_roots(cls, maroots=None, arroots=None, nobs=100):
        """
        Create ArmaProcess from AR and MA polynomial roots

        Parameters
        ----------
        maroots : array_like, optional
            Roots for the MA polynomial
            1 + theta_1*z + theta_2*z^2 + ..... + theta_n*z^n.
            Must be 1-D.
        arroots : array_like, optional
            Roots for the AR polynomial
            1 - phi_1*z - phi_2*z^2 - ..... - phi_n*z^n.
            Must be 1-D.
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Returns
        -------
        ArmaProcess
            Class instance initialized with arcoefs and macoefs.

        Examples
        --------
        >>> arroots = [2, -2]
        >>> maroots = [3, 1.5]
        >>> arma_process = sm.tsa.ArmaProcess.from_roots(
        ...     arroots=arroots, maroots=maroots
        ... )
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
        if arroots is not None and len(arroots):
            arpoly = np.polynomial.polynomial.Polynomial.fromroots(arroots)
            arcoefs = arpoly.coef[1:] / arpoly.coef[0]
        else:
            arcoefs = []

        if maroots is not None and len(maroots):
            mapoly = np.polynomial.polynomial.Polynomial.fromroots(maroots)
            macoefs = mapoly.coef[1:] / mapoly.coef[0]
        else:
            macoefs = []

        # As from_coeffs will create a polynomial with constant 1/-1,(MA/AR)
        # we need to scale the polynomial coefficients accordingly
        return cls(np.r_[1, arcoefs], np.r_[1, macoefs], nobs=nobs)

    @classmethod
    @_require_1d("arcoefs", "macoefs")
    def from_coeffs(cls, arcoefs=None, macoefs=None, nobs=100):
        """
        Create ArmaProcess from an ARMA representation

        Parameters
        ----------
        arcoefs : array_like, optional
            Coefficient for autoregressive lag polynomial, not including zero
            lag. The sign is inverted to conform to the usual time series
            representation of an ARMA process in statistics. See the class
            docstring for more information. Must be 1-D.
        macoefs : array_like, optional
            Coefficient for moving-average lag polynomial, excluding zero lag.
            Must be 1-D.
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Returns
        -------
        ArmaProcess
            Class instance initialized with arcoefs and macoefs.

        Examples
        --------
        >>> arparams = [.75, -.25]
        >>> maparams = [.65, .35]
        >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
        arcoefs = [] if arcoefs is None else arcoefs
        macoefs = [] if macoefs is None else macoefs
        return cls(
            np.r_[1, -np.asarray(arcoefs)],
            np.r_[1, np.asarray(macoefs)],
            nobs=nobs,
        )

    @classmethod
    def from_estimation(cls, model_results, nobs=None):
        """
        Create an ArmaProcess from the results of an ARIMA estimation

        Parameters
        ----------
        model_results : ARIMAResults instance
            A fitted model.
        nobs : int, optional
            If None, nobs is taken from the results.

            .. versionchanged:: 0.15.1

               ``nobs=0`` is used as given, as in the constructor. It used
               to select the nobs of the results, as omitting the value
               does.

        Returns
        -------
        ArmaProcess
            Class instance initialized from model_results.

        See Also
        --------
        statsmodels.tsa.arima.model.ARIMA
            The models class used to create the ArmaProcess
        """
        if nobs is None:
            nobs = model_results.nobs
        return cls(
            model_results.polynomial_reduced_ar,
            model_results.polynomial_reduced_ma,
            nobs=nobs,
        )

    def __mul__(self, oth):
        if isinstance(oth, self.__class__):
            ar = (self.arpoly * oth.arpoly).coef
            ma = (self.mapoly * oth.mapoly).coef
        else:
            try:
                aroth, maoth = oth
                arpolyoth = np.polynomial.Polynomial(aroth)
                mapolyoth = np.polynomial.Polynomial(maoth)
                ar = (self.arpoly * arpolyoth).coef
                ma = (self.mapoly * mapolyoth).coef
            except Exception as exc:
                raise TypeError("Other type is not a valid type") from exc
        return self.__class__(ar, ma, nobs=self.nobs)

    def __repr__(self):
        msg = "ArmaProcess({0}, {1}, nobs={2}) at {3}"
        return msg.format(
            self.ar.tolist(), self.ma.tolist(), self.nobs, hex(id(self))
        )

    def __str__(self):
        return f"ArmaProcess\nAR: {self.ar.tolist()}\nMA: {self.ma.tolist()}"

    def _check_lags(self, value, name):
        # None selects nobs. This used to be "value or self.nobs", which
        # also selected nobs for 0.
        if value is None:
            return self.nobs
        return _validate_lags(value, name)

    @Appender(
        _arma_lags_method_doc(
            arma_acovf,
            "nobs",
            "The number of terms (lags plus zero lag) to include in returned "
            "acovf.",
        )
    )
    def acovf(self, nobs=None):
        nobs = self._check_lags(nobs, "nobs")
        return arma_acovf(self.ar, self.ma, nobs=nobs)

    @Appender(
        _arma_lags_method_doc(
            arma_acf,
            "lags",
            "The number of terms (lags plus zero lag) to include in returned acf.",
        )
    )
    def acf(self, lags=None):
        lags = self._check_lags(lags, "lags")
        return arma_acf(self.ar, self.ma, lags=lags)

    @Appender(
        _arma_lags_method_doc(
            arma_pacf,
            "lags",
            "The number of terms (lags plus zero lag) to include in returned "
            "pacf.",
        )
    )
    def pacf(self, lags=None):
        lags = self._check_lags(lags, "lags")
        return arma_pacf(self.ar, self.ma, lags=lags)

    @Appender(
        _arma_lags_method_doc(
            arma_periodogram,
            "nobs",
            "The number of frequencies, equally spaced on [0, pi), at which the "
            "periodogram is computed.",
            keep=["normalization"],
        )
    )
    def periodogram(self, nobs=None, *, normalization=None):
        nobs = self._check_lags(nobs, "nobs")
        normalization = _periodogram_normalization(
            normalization, "ArmaProcess.periodogram", stacklevel=2
        )
        return arma_periodogram(
            self.ar, self.ma, worN=nobs, normalization=normalization
        )

    @Appender(
        _arma_lags_method_doc(
            arma_impulse_response,
            "leads",
            "The number of observations to calculate.",
        )
    )
    def impulse_response(self, leads=None):
        leads = self._check_lags(leads, "leads")
        return arma_impulse_response(self.ar, self.ma, leads=leads)

    @Appender(
        _arma_lags_method_doc(
            arma2ma,
            "lags",
            "The number of coefficients to calculate.",
            ["Omitting ``lags`` no longer raises an ``AttributeError``."],
        )
    )
    def arma2ma(self, lags=None):
        lags = self._check_lags(lags, "lags")
        return arma2ma(self.ar, self.ma, lags=lags)

    @Appender(
        _arma_lags_method_doc(
            arma2ar,
            "lags",
            "The number of coefficients to calculate.",
            ["Omitting ``lags`` no longer raises an ``AttributeError``."],
        )
    )
    def arma2ar(self, lags=None):
        lags = self._check_lags(lags, "lags")
        return arma2ar(self.ar, self.ma, lags=lags)

    @property
    def arroots(self):
        """Roots of autoregressive lag-polynomial"""
        return self.arpoly.roots()

    @property
    def maroots(self):
        """Roots of moving average lag-polynomial"""
        return self.mapoly.roots()

    @property
    def isstationary(self):
        """
        Arma process is stationary if AR roots are outside unit circle

        Returns
        -------
        bool
             True if autoregressive roots are outside unit circle.
        """
        return bool(np.all(np.abs(self.arroots) > 1.0))

    @property
    def isinvertible(self):
        """
        Arma process is invertible if MA roots are outside unit circle

        Returns
        -------
        bool
             True if moving average roots are outside unit circle.
        """
        return bool(np.all(np.abs(self.maroots) > 1))

    def invertroots(self, retnew=False):
        """
        Make MA polynomial invertible by inverting roots inside unit circle

        Parameters
        ----------
        retnew : bool, optional
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial.

        Returns
        -------
        manew : ndarray
            A new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : bool
            True if the MA lag-polynomial was already invertible, returned if
            retnew is false.
        armaprocess : ArmaProcess
            If retnew is true, then return a new instance with invertible
            MA-polynomial.
        """
        # TODO: variable returns like this?
        pr = self.maroots
        mainv = self.ma
        invertible = self.isinvertible
        if not invertible:
            pr[np.abs(pr) < 1] = 1.0 / pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef / pnew.coef[0]

        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
        else:
            return mainv, invertible

    @Appender(str(_generate_sample_doc))
    def generate_sample(
        self, nsample=None, scale=1.0, distrvs=None, axis=0, burnin=0
    ):
        # The default used to be 100, whatever the value of nobs
        if nsample is None:
            nsample = self.nobs
        return arma_generate_sample(
            self.ar, self.ma, nsample, scale, distrvs, axis=axis, burnin=burnin
        )
