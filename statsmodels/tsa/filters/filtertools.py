# -*- coding: utf-8 -*-
"""Linear Filters for time series analysis and testing


TODO:
* check common sequence in signature of filter functions (ar,ma,x) or (x,ar,ma)

Created on Sat Oct 23 17:18:03 2010

Author: Josef-pktd
"""
#not original copied from various experimental scripts
#version control history is there

from statsmodels.compat.python import range
import numpy as np
import scipy.fftpack as fft
from scipy import signal
from scipy.signal.signaltools import _centered as trim_centered
from ._utils import _maybe_get_pandas_wrapper


def _pad_nans(x, head=None, tail=None):
    if np.ndim(x) == 1:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[np.nan] * head, x, [np.nan] * tail]
        elif tail is None:
            return np.r_[[np.nan] * head, x]
        elif head is None:
            return np.r_[x, [np.nan] * tail]
    elif np.ndim(x) == 2:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[[np.nan] * x.shape[1]] * head, x,
                         [[np.nan] * x.shape[1]] * tail]
        elif tail is None:
            return np.r_[[[np.nan] * x.shape[1]] * head, x]
        elif head is None:
            return np.r_[x, [[np.nan] * x.shape[1]] * tail]
    else:
        raise ValueError("Nan-padding for ndim > 2 not implemented")

#original changes and examples in sandbox.tsa.try_var_convolve

# don't do these imports, here just for copied fftconvolve
#get rid of these imports
#from scipy.fftpack import fft, ifft, ifftshift, fft2, ifft2, fftn, \
#     ifftn, fftfreq
#from numpy import product,array

def fftconvolveinv(in1, in2, mode="full"):
    """Convolve two N-dimensional arrays using FFT. See convolve.

    copied from scipy.signal.signaltools, but here used to try out inverse filter
    doesn't work or I can't get it to work

    2010-10-23:
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it doesn't work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    size = s1+s2-1

    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))
    IN1 = fft.fftn(in1,fsize)
    #IN1 *= fftn(in2,fsize) #JP: this looks like the only change I made
    IN1 /= fft.fftn(in2,fsize)  # use inverse filter
    # note the inverse is elementwise not matrix inverse
    # is this correct, NO  doesn't seem to work for VARMA
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1,axis=0) > np.product(s2,axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret,osize)
    elif mode == "valid":
        return trim_centered(ret,abs(s2-s1)+1)

#code duplication with fftconvolveinv
def fftconvolve3(in1, in2=None, in3=None, mode="full"):
    """Convolve two N-dimensional arrays using FFT. See convolve.

    for use with arma  (old version: in1=num in2=den in3=data

    * better for consistency with other functions in1=data in2=num in3=den
    * note in2 and in3 need to have consistent dimension/shape
      since I'm using max of in2, in3 shapes and not the sum

    copied from scipy.signal.signaltools, but here used to try out inverse
    filter doesn't work or I can't get it to work

    2010-10-23
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it doesn't work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn
    """
    if (in2 is None) and (in3 is None):
        raise ValueError('at least one of in2 and in3 needs to be given')
    s1 = np.array(in1.shape)
    if not in2 is None:
        s2 = np.array(in2.shape)
    else:
        s2 = 0
    if not in3 is None:
        s3 = np.array(in3.shape)
        s2 = max(s2, s3) # try this looks reasonable for ARMA
        #s2 = s3


    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    size = s1+s2-1

    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))
    #convolve shorter ones first, not sure if it matters
    if not in2 is None:
        IN1 = fft.fftn(in2, fsize)
    if not in3 is None:
        IN1 /= fft.fftn(in3, fsize)  # use inverse filter
    # note the inverse is elementwise not matrix inverse
    # is this correct, NO  doesn't seem to work for VARMA
    IN1 *= fft.fftn(in1, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fft.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1,axis=0) > np.product(s2,axis=0):
            osize = s1
        else:
            osize = s2
        return trim_centered(ret,osize)
    elif mode == "valid":
        return trim_centered(ret,abs(s2-s1)+1)

#original changes and examples in sandbox.tsa.try_var_convolve
#examples and tests are there
def recursive_filter(x, ar_coeff, init=None):
    '''
    Autoregressive, or recursive, filtering.

    Parameters
    ----------
    x : array-like
        Time-series data. Should be 1d or n x 1.
    ar_coeff : array-like
        AR coefficients in reverse time order. See Notes
    init : array-like
        Initial values of the time-series prior to the first value of y.
        The default is zero.

    Returns
    -------
    y : array
        Filtered array, number of columns determined by x and ar_coeff. If a
        pandas object is given, a pandas object is returned.

    Notes
    -----

    Computes the recursive filter ::

        y[n] = ar_coeff[0] * y[n-1] + ...
                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]

    where n_coeff = len(n_coeff).
    '''
    _pandas_wrapper = _maybe_get_pandas_wrapper(x)
    x = np.asarray(x).squeeze()
    ar_coeff = np.asarray(ar_coeff).squeeze()

    if x.ndim > 1 or ar_coeff.ndim > 1:
        raise ValueError('x and ar_coeff have to be 1d')

    if init is not None:  # integer init are treated differently in lfiltic
        if len(init) != len(ar_coeff):
            raise ValueError("ar_coeff must be the same length as init")
        init = np.asarray(init, dtype=float)

    if init is not None:
        zi = signal.lfiltic([1], np.r_[1, -ar_coeff], init, x)
    else:
        zi = None

    y = signal.lfilter([1.], np.r_[1, -ar_coeff], x, zi=zi)

    if init is not None:
        result = y[0]
    else:
        result = y

    if _pandas_wrapper:
        return _pandas_wrapper(result)
    return result


def convolution_filter(x, filt, nsides=2):
    '''
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

    Returns
    -------
    y : ndarray, 2d
        Filtered array, number of columns determined by x and filt. If a
        pandas object is given, a pandas object is returned. The index of
        the return is the exact same as the time period in ``x``

    Notes
    -----
    In nsides == 1, x is filtered ::

        y[n] = filt[0]*x[n-1] + ... + filt[n_filt-1]*x[n-n_filt]

    where n_filt is len(filt).

    If nsides == 2, x is filtered around lag 0 ::

        y[n] = filt[0]*x[n - n_filt/2] + ... + filt[n_filt / 2] * x[n]
               + ... + x[n + n_filt/2]

    where n_filt is len(filt). If n_filt is even, then more of the filter
    is forward in time than backward.

    If filt is 1d or (nlags,1) one lag polynomial is applied to all
    variables (columns of x). If filt is 2d, (nlags, nvars) each series is
    independently filtered with its own lag polynomial, uses loop over nvar.
    This is different than the usual 2d vs 2d convolution.

    Filtering is done with scipy.signal.convolve, so it will be reasonably
    fast for medium sized data. For large data fft convolution would be
    faster.
    '''
    # for nsides shift the index instead of using 0 for 0 lag this
    # allows correct handling of NaNs
    if nsides == 1:
        trim_head = len(filt) - 1
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt)/2.) - 1) or None
        trim_tail = int(np.ceil(len(filt)/2.) - len(filt) % 2) or None
    else:  # pragma : no cover
        raise ValueError("nsides must be 1 or 2")

    _pandas_wrapper = _maybe_get_pandas_wrapper(x)
    x = np.asarray(x)
    filt = np.asarray(filt)
    if x.ndim > 1 and filt.ndim == 1:
        filt = filt[:, None]
    if x.ndim > 2:
        raise ValueError('x array has to be 1d or 2d')

    if filt.ndim == 1 or min(filt.shape) == 1:
        result = signal.convolve(x, filt, mode='valid')
    elif filt.ndim == 2:
        nlags = filt.shape[0]
        nvar = x.shape[1]
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        if nsides == 2:
            for i in range(nvar):
                # could also use np.convolve, but easier for swiching to fft
                result[:, i] = signal.convolve(x[:, i], filt[:, i],
                                               mode='valid')
        elif nsides == 1:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], np.r_[0, filt[:, i]],
                                               mode='valid')
    result = _pad_nans(result, trim_head, trim_tail)
    if _pandas_wrapper:
        return _pandas_wrapper(result)
    return result


#copied from sandbox.tsa.garch
def miso_lfilter(ar, ma, x, useic=False): #[0.1,0.1]):
    '''
    use nd convolution to merge inputs,
    then use lfilter to produce output

    arguments for column variables
    return currently 1d

    Parameters
    ----------
    ar : array_like, 1d, float
        autoregressive lag polynomial including lag zero, ar(L)y_t
    ma : array_like, same ndim as x, currently 2d
        moving average lag polynomial ma(L)x_t
    x : array_like, 2d
        input data series, time in rows, variables in columns

    Returns
    -------
    y : array, 1d
        filtered output series
    inp : array, 1d
        combined input series

    Notes
    -----
    currently for 2d inputs only, no choice of axis
    Use of signal.lfilter requires that ar lag polynomial contains
    floating point numbers
    does not cut off invalid starting and final values

    miso_lfilter find array y such that::

            ar(L)y_t = ma(L)x_t

    with shapes y (nobs,), x (nobs,nvars), ar (narlags,), ma (narlags,nvars)

    '''
    ma = np.asarray(ma)
    ar = np.asarray(ar)
    #inp = signal.convolve(x, ma, mode='valid')
    #inp = signal.convolve(x, ma)[:, (x.shape[1]+1)//2]
    #Note: convolve mixes up the variable left-right flip
    #I only want the flip in time direction
    #this might also be a mistake or problem in other code where I
    #switched from correlate to convolve
    # correct convolve version, for use with fftconvolve in other cases
    #inp2 = signal.convolve(x, ma[:,::-1])[:, (x.shape[1]+1)//2]
    inp = signal.correlate(x, ma[::-1,:])[:, (x.shape[1]+1)//2]
    #for testing 2d equivalence between convolve and correlate
    #np.testing.assert_almost_equal(inp2, inp)
    nobs = x.shape[0]
    # cut of extra values at end

    #todo initialize also x for correlate
    if useic:
        return signal.lfilter([1], ar, inp,
                #zi=signal.lfilter_ic(np.array([1.,0.]),ar, ic))[0][:nobs], inp[:nobs]
                zi=signal.lfiltic(np.array([1.,0.]),ar, useic))[0][:nobs], inp[:nobs]
    else:
        return signal.lfilter([1], ar, inp)[:nobs], inp[:nobs]
    #return signal.lfilter([1], ar, inp), inp
