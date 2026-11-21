"""
Linear Filters for time series analysis and testing


TODO:
* check common sequence in signature of filter functions (ar,ma,x) or (x,ar,ma)

Created on Sat Oct 23 17:18:03 2010

Author: Josef-pktd
"""
# not original copied from various experimental scripts
# version control history is there

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
import warnings

import numpy as np
from scipy import signal
import scipy.fftpack as fft
from scipy.signal._signaltools import _centered as trim_centered

from statsmodels.tools.validation import PandasWrapper, array_like

if TYPE_CHECKING:
    from statsmodels.tools.typing import ArrayLike1D, ArrayLike2D


class CycleTrendResult(NamedTuple):
    """
    Result of :func:`cffilter`, :func:`hamilton_filter`, and
    :func:`hpfilter`: a cycle/trend decomposition.

    Parameters
    ----------
    cycle : ndarray, Series, or DataFrame
        The estimated cyclical component. See the docstring of the
        function that produced this result for the precise definition
        (e.g., :func:`hamilton_filter` leaves the first ``p + h - 1``
        values as ``NaN``).
    trend : ndarray, Series, or DataFrame
        The estimated trend component, matching ``cycle`` in shape.
    """

    cycle: ArrayLike1D | ArrayLike2D
    trend: ArrayLike1D | ArrayLike2D


def _pad_nans(x, head=None, tail=None):
    # convolution_filter, the only caller, passes 1-d or 2-d x, and a head
    # and a tail that are both positive when both are given
    if np.ndim(x) == 1:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[np.nan] * head, x, [np.nan] * tail]
        elif tail is None:
            return np.r_[[np.nan] * head, x]
        else:  # head is None
            return np.r_[x, [np.nan] * tail]
    # x is 2-d
    if head is None and tail is None:
        return x
    elif head and tail:
        return np.r_[[[np.nan] * x.shape[1]] * head, x,
                     [[np.nan] * x.shape[1]] * tail]
    elif tail is None:
        return np.r_[[[np.nan] * x.shape[1]] * head, x]
    else:  # head is None
        return np.r_[x, [[np.nan] * x.shape[1]] * tail]

# original changes and examples in sandbox.tsa.try_var_convolve

# do not do these imports, here just for copied fftconvolve
# get rid of these imports
# from scipy.fftpack import fft, ifft, ifftshift, fft2, ifft2, fftn, \
#     ifftn, fftfreq
# from numpy import product,array


# previous location in sandbox.tsa.try_var_convolve
def fftconvolveinv(in1, in2, mode="full"):
    """
    Convolve two N-dimensional arrays using FFT. See convolve

    Parameters
    ----------
    in1 : array_like
        First input array.
    in2 : array_like
        Second input array, used as the inverse filter.
    mode : str, optional
        Convolution mode, one of 'full', 'same', or 'valid'.

    Returns
    -------
    ndarray
        The convolved array.

    Notes
    -----
    copied from scipy.signal.signaltools, but here used to try out inverse
    filter. does not work or I cannot get it to work

    2010-10-23:
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it does not work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn
    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complexfloating) or
                      np.issubdtype(in2.dtype, np.complexfloating))
    size = s1+s2-1

    # Always use 2**n-sized FFT
    fsize = (2 ** np.ceil(np.log2(size))).astype(int)
    IN1 = fft.fftn(in1, fsize)
    # IN1 *= fftn(in2,fsize) # JP: this looks like the only change I made
    IN1 /= fft.fftn(in2, fsize)  # use inverse filter
    # note the inverse is elementwise not matrix inverse
    # is this correct, NO  does not seem to work for VARMA
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.prod(s1, axis=0) > np.prod(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret, osize)
    elif mode == "valid":
        return trim_centered(ret, abs(s2-s1)+1)


# code duplication with fftconvolveinv
def fftconvolve3(in1, in2=None, in3=None, mode="full"):
    """
    Convolve two N-dimensional arrays using FFT. See convolve

    Parameters
    ----------
    in1 : array_like
        First input array.
    in2 : array_like, optional
        Second input array. At least one of `in2` and `in3` must be given.
    in3 : array_like, optional
        Third input array. At least one of `in2` and `in3` must be given.
    mode : str, optional
        Convolution mode, one of 'full', 'same', or 'valid'.

    Returns
    -------
    ndarray
        The convolved array.

    Notes
    -----
    For use with arma  (old version: in1=num in2=den in3=data)

    * better for consistency with other functions in1=data in2=num in3=den
    * note in2 and in3 need to have consistent dimension/shape
      since I'm using max of in2, in3 shapes and not the sum

    copied from scipy.signal.signaltools, but here used to try out inverse
    filter does not work or I cannot get it to work

    2010-10-23
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it does not work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn
    """
    if (in2 is None) and (in3 is None):
        raise ValueError("at least one of in2 and in3 needs to be given")
    s1 = np.array(in1.shape)
    if in2 is not None:
        s2 = np.array(in2.shape)
    else:
        s2 = 0
    if in3 is not None:
        s3 = np.array(in3.shape)
        s2 = max(s2, s3)  # try this looks reasonable for ARMA
        # s2 = s3

    complex_result = (np.issubdtype(in1.dtype, np.complexfloating) or
                      np.issubdtype(in2.dtype, np.complexfloating))
    size = s1+s2-1

    # Always use 2**n-sized FFT
    fsize = (2 ** np.ceil(np.log2(size))).astype(int)
    # convolve shorter ones first, not sure if it matters
    IN1 = in1.copy()  # TODO: Is this correct?
    if in2 is not None:
        IN1 = fft.fftn(in2, fsize)
    if in3 is not None:
        IN1 /= fft.fftn(in3, fsize)  # use inverse filter
    # note the inverse is elementwise not matrix inverse
    # is this correct, NO  does not seem to work for VARMA
    IN1 *= fft.fftn(in1, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.prod(s1, axis=0) > np.prod(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret, osize)
    elif mode == "valid":
        return trim_centered(ret, abs(s2-s1)+1)


# original changes and examples in sandbox.tsa.try_var_convolve
# examples and tests are there
def recursive_filter(x, ar_coeff, init=None):
    """
    Autoregressive, or recursive, filtering.

    Parameters
    ----------
    x : array_like
        Time-series data, either 1-D with shape (n,) or a single column with
        shape (n, 1), such as a one-column DataFrame. Both are filtered along
        the n observations. An array with shape (1, ..., 1) is a single
        observation.

        .. versionchanged:: 0.15.1

           An (n, 1) column is now filtered along its rows, the same as 1-D
           input. Previously no recursion was applied, so the column was
           returned unchanged. ``init`` now works with an (n, 1) column, a
           row and a single observation, where it raised a ValueError. Other
           input with more than one column or more than two dimensions now
           raises a ValueError.

        .. deprecated:: 0.15.1

           A row, an array with shape (1, n) or (1, ..., 1, n) with n > 1, is
           filtered along its last axis, as a 1-D series of n observations,
           and returned with the shape of ``x``. ``x`` must have the correct
           shape, 1-D or an (n, 1) column, so a row now warns and will raise
           a ValueError in statsmodels 0.17.0. Pass the row as a 1-D array
           instead.
    ar_coeff : array_like
        AR coefficients in reverse time order. See Notes for details.
    init : array_like, optional
        Initial values of the time-series prior to the first value of y.
        The default is zero.

        .. versionchanged:: 0.15.1

           ``ar_coeff`` and ``init`` can be (n_coeff, 1) columns, which
           raised a ValueError from NumPy or SciPy. Other 2-d input raises a
           ValueError that names the argument.

    Returns
    -------
    ndarray, Series or DataFrame
        The filtered series, with the same shape as ``x``. A Series input
        returns a Series and a DataFrame returns a DataFrame, each with the
        index and name or columns of ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, an (n, 1) column, a row with shape (1, n) or
        (1, ..., 1, n), or a single observation with shape (1, ..., 1), if
        ``ar_coeff`` or ``init`` is not 1-D or an (n_coeff, 1) column, or if
        ``init`` and ``ar_coeff`` have different lengths.

    Warns
    -----
    FutureWarning
        If ``x`` is a row with shape (1, n) or (1, ..., 1, n), which is
        deprecated. A (1, 1) or (1, ..., 1) array is a single observation and
        does not warn.

    Notes
    -----
    Computes the recursive filter ::

        y[n] = ar_coeff[0] * y[n-1] + ...
                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]

    where n_coeff = len(ar_coeff). The values before the first observation
    are ``init`` in reverse time order, ``y[-1] = init[0]``, ...,
    ``y[-n_coeff] = init[n_coeff - 1]``, or zero if ``init`` is None. This is
    the recursive filter of ``stats::filter`` in R. For example, with
    ``ar_coeff=[0.5, 0.25]``, ``init=[1, 2]`` and ``x`` zero, y starts
    ``1, 0.75, 0.625``.
    """
    pw = PandasWrapper(x)
    x = array_like(x, "x")
    shape = x.shape
    if x.ndim > 1:
        # an (n, 1) column, or a single observation with shape (1, ..., 1)
        is_column = (x.ndim == 2 and shape[1] == 1) or x.size == 1
        if not is_column:
            if any(nobs != 1 for nobs in shape[:-1]):
                raise ValueError(
                    f"x must be 1-D or an (n, 1) column, but has shape {shape}. "
                    "Filter each series separately."
                )
            warnings.warn(
                f"x must be 1-D or an (n, 1) column, but has shape {shape}. "
                "A row with shape (1, n) or (1, ..., 1, n) is filtered as a "
                "1-D series of n observations, but this is deprecated and "
                "will raise a ValueError in statsmodels 0.17.0. Pass the row "
                "as a 1-D array instead.",
                FutureWarning,
                stacklevel=2,
            )
        # lfilter works along the last axis, so filter x as 1-D and give the
        # result the shape of x
        x = x.ravel()
    ar_coeff = array_like(ar_coeff, "ar_coeff", ndim=1)

    if init is not None:  # integer init are treated differently in lfiltic
        init = array_like(init, "init", ndim=1)
        if len(init) != len(ar_coeff):
            raise ValueError("ar_coeff must be the same length as init")

    if init is not None:
        zi = signal.lfiltic([1], np.r_[1, -ar_coeff], init, x)
    else:
        zi = None

    y = signal.lfilter([1.], np.r_[1, -ar_coeff], x, zi=zi)

    if init is not None:
        result = y[0]
    else:
        result = y

    return pw.wrap(result.reshape(shape))


def convolution_filter(x, filt, nsides=2):
    """
    Linear filtering via convolution. Centered and backward displaced moving
    weighted average.

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    filt : array_like
        Linear filter coefficients in reverse time-order. Should have the
        same number of dimensions as x though if 1d and ``x`` is 2d will be
        coerced to 2d.
    nsides : int, optional
        If 2, a centered moving average is computed using the filter
        coefficients. If 1, the filter coefficients are for past values only.
        Both methods use scipy.signal.convolve.

        .. versionchanged:: 0.15.1

           nsides=1 works with a 2d filter, and with 2d x and a filter of
           length one, where it raised a ValueError.

    Returns
    -------
    ndarray, Series or DataFrame
        The filtered array, with the number of observations of ``x``. It is
        1d if ``x`` is 1d, and otherwise has one column per column of ``x``.
        The values that the filter cannot compute at the ends are NaN. A
        Series or DataFrame ``x`` returns a Series or DataFrame with the
        index and the name or columns of ``x``.

    Notes
    -----
    The filters are the convolution filters of ``stats::filter`` in R.

    If nsides == 1, x is filtered with its current and past values ::

        y[n] = filt[0]*x[n] + filt[1]*x[n-1] + ...
               + filt[n_filt-1]*x[n-n_filt+1]

    where n_filt is len(filt). The first n_filt - 1 values are NaN.

    If nsides == 2, x is filtered around lag 0 ::

        y[n] = filt[0]*x[n+m] + ... + filt[m]*x[n] + ...
               + filt[n_filt-1]*x[n+m-n_filt+1]

    where n_filt is len(filt) and m = n_filt // 2. The first n_filt - 1 - m
    and the last m values are NaN. If n_filt is even, then more of the filter
    is forward in time than backward.

    If filt is 1d or (nlags,1) one lag polynomial is applied to all
    variables (columns of x). If filt is 2d, (nlags, nvars) each series is
    independently filtered with its own lag polynomial, uses loop over nvar.
    This is different than the usual 2d vs 2d convolution.

    Filtering is done with scipy.signal.convolve, so it will be reasonably
    fast for medium sized data. For large data fft convolution would be
    faster.
    """
    # for nsides shift the index instead of using 0 for 0 lag this
    # allows correct handling of NaNs
    if nsides == 1:
        trim_head = len(filt) - 1 or None
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt)/2.) - 1) or None
        trim_tail = int(np.ceil(len(filt)/2.) - len(filt) % 2) or None
    else:  # pragma : no cover
        raise ValueError("nsides must be 1 or 2")

    pw = PandasWrapper(x)
    x = array_like(x, "x", maxdim=2)
    filt = array_like(filt, "filt", ndim=x.ndim)

    if filt.ndim == 1 or min(filt.shape) == 1:
        result = signal.convolve(x, filt, mode="valid")
    else:  # filt.ndim == 2
        nlags = filt.shape[0]
        nvar = x.shape[1]
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        # nsides only sets the NaN padding, as for a 1d filter
        for i in range(nvar):
            # could also use np.convolve, but easier for swiching to fft
            result[:, i] = signal.convolve(x[:, i], filt[:, i], mode="valid")
    result = _pad_nans(result, trim_head, trim_tail)
    return pw.wrap(result)


# previously located in sandbox.tsa.garch
def miso_lfilter(ar, ma, x, useic=False):
    """
    Filter multiple time series into a single time series.

    Uses a convolution to merge inputs, and then lfilter to produce output.

    Parameters
    ----------
    ar : array_like
        The coefficients of autoregressive lag polynomial including lag zero,
        ar(L) in the expression ar(L)y_t.
    ma : array_like
        The coefficients of the moving average lag polynomial, ma(L) in
        ma(L)x_t, with one row per lag and one column per column of ``x``.
        A 1-d ``ma`` is a single column.
    x : array_like
        The input data series, time in rows, variables in columns. A 1-d
        ``x`` is a single variable.

        .. versionchanged:: 0.15.1

           ``x``, ``ar`` and ``ma`` are converted with ``array_like``, so a
           list or tuple ``x``, a column ``ar`` and a 1-d ``x`` or ``ma``
           work where they raised, and ``useic`` can be an ndarray. A 2-d
           ``ma`` without one column per column of a 2-d ``x`` is
           deprecated.
    useic : array_like or bool, optional
        Initial conditions for the AR filter, as accepted by
        ``scipy.signal.lfiltic``: the values of the filtered series before
        the first observation in reverse time order (``useic[0]`` is
        ``y[-1]``), of length ``ar.shape[0] - 1``. Use the default, False,
        for zero initial conditions.

        .. versionchanged:: 0.15.1

           Without AR lags, ``ar = [1.0]``, there are no initial conditions,
           and a non-zero ``useic`` is ignored, as values beyond
           ``ar.shape[0] - 1`` are, where it raised a ValueError. A scalar
           ``useic`` is the initial value ``y[-1]``, with zero for any
           further lags, where it raised an IndexError for one AR lag (and
           a ValueError for any number of lags with scipy 1.16 or later).

        .. deprecated:: 0.15.1

           ``useic=True`` is not a set of initial values. With two or more
           AR lags it is used as ``y[-1] = 1``, with zero for the further
           lags, as before, but this is deprecated and will raise a
           ValueError in statsmodels 0.17.0. With no or one AR lag, where it
           raised an error from scipy, it raises a ValueError.

    Returns
    -------
    y : ndarray
        The filtered output series.
    inp : ndarray, 1d
        The combined input series.

    Raises
    ------
    ValueError
        If ``ma`` or ``x`` is 1-d and ``ma`` does not have one column per
        column of ``x``, or if ``useic`` is True and ``ar`` has no or one AR
        lag.

    Warns
    -----
    FutureWarning
        If ``ma`` and ``x`` are 2-d and ``ma`` does not have one column per
        column of ``x``. The result does not filter each column of ``x`` with
        its own column of ``ma``. This is deprecated and will raise a
        ValueError in statsmodels 0.17.0.
    FutureWarning
        If ``useic`` is True and ``ar`` has two or more AR lags. True is used
        as ``y[-1] = 1``, with zero for the further lags. This is deprecated
        and will raise a ValueError in statsmodels 0.17.0.

    Notes
    -----
    x is filtered along its rows, there is no choice of axis
    Use of signal.lfilter requires that ar lag polynomial contains
    floating point numbers
    does not cut off invalid starting and final values

    miso_lfilter finds array y such that:

            ar(L)y_t = ma(L)x_t

    with shapes y (nobs,), x (nobs, nvars), ar (narlags,), and
    ma (narlags, nvars).
    """
    # a 2-d ma and x with different numbers of columns ran before, so they
    # are deprecated rather than rejected
    both_2d = np.ndim(ma) == 2 and np.ndim(x) == 2
    ma = array_like(ma, "ma", ndim=2)
    ar = array_like(ar, "ar", ndim=1)
    # dtype=None keeps complex x complex, as before
    x = array_like(x, "x", ndim=2, dtype=None)
    if ma.shape[1] != x.shape[1]:
        msg = (
            "ma must have one column per column of x, but ma has "
            f"{ma.shape[1]} columns and x has {x.shape[1]}"
        )
        if not both_2d:
            raise ValueError(msg)
        warnings.warn(
            f"{msg}. The result does not filter each column of x with its "
            "own column of ma. This is deprecated and will raise a "
            "ValueError in statsmodels 0.17.0.",
            FutureWarning,
            stacklevel=2,
        )
    if isinstance(useic, (bool, np.bool_)) and useic:
        msg = (
            "useic must be False or the initial values of the filtered "
            "series, not True"
        )
        if ar.shape[0] <= 2:
            # this raised an error from scipy
            raise ValueError(msg)
        warnings.warn(
            f"{msg}. True is used as y[-1] = 1, with zero for the further "
            "lags. This is deprecated and will raise a ValueError in "
            "statsmodels 0.17.0.",
            FutureWarning,
            stacklevel=2,
        )
    inp = signal.correlate(x, ma[::-1, :])[:, x.shape[1] - 1]
    # for testing 2d equivalence between convolve and correlate
    #  inp2 = signal.convolve(x, ma[:,::-1])[:, x.shape[1]-1]
    #  np.testing.assert_almost_equal(inp2, inp)
    nobs = x.shape[0]
    # cut of extra values at end

    # TODO: initialize also x for correlate
    # useic can be an ndarray; all-zero values are the default zero values
    if np.any(useic):
        # b = [1.0] as in lfilter, so that zi has len(ar) - 1 values, also
        # for ar = [1.0]. A scalar useic is y[-1], and lfiltic needs it 1-d.
        zi = signal.lfiltic([1.0], ar, np.atleast_1d(useic))
        return signal.lfilter([1], ar, inp, zi=zi)[0][:nobs], inp[:nobs]
    else:
        return signal.lfilter([1], ar, inp)[:nobs], inp[:nobs]
