# -*- coding: utf-8 -*-
"""Linear Filters for time series analysis and testing


TODO:
* check common sequence in signature of filter functions (ar,ma,x) or (x,ar,ma)

Created on Sat Oct 23 17:18:03 2010

Author: Josef-pktd
"""
#not original copied from various experimental scripts
#version control history is there

import numpy as np
import scipy.fftpack as fft
from scipy import signal
from scipy.signal.signaltools import _centered as trim_centered

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
def arfilter(x, a):
    '''apply an autoregressive filter to a series x

    x can be 2d, a can be 1d, 2d, or 3d

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    a : array_like
        autoregressive filter coefficients, ar lag polynomial
        see Notes

    Returns
    -------
    y : ndarray, 2d
        filtered array, number of columns determined by x and a

    Notes
    -----

    In general form this uses the linear filter ::

        y = a(L)x

    where
    x : nobs, nvars
    a : nlags, nvars, npoly

    Depending on the shape and dimension of a this uses different
    Lag polynomial arrays

    case 1 : a is 1d or (nlags,1)
        one lag polynomial is applied to all variables (columns of x)
    case 2 : a is 2d, (nlags, nvars)
        each series is independently filtered with its own
        lag polynomial, uses loop over nvar
    case 3 : a is 3d, (nlags, nvars, npoly)
        the ith column of the output array is given by the linear filter
        defined by the 2d array a[:,:,i], i.e. ::

            y[:,i] = a(.,.,i)(L) * x
            y[t,i] = sum_p sum_j a(p,j,i)*x(t-p,j)
                     for p = 0,...nlags-1, j = 0,...nvars-1,
                     for all t >= nlags

    All filtering is done with scipy.signal.convolve, so it will be reasonably
    fast for medium sized arrays. For large arrays fft convolution would be
    faster.

    Note: maybe convert to axis=1, Not

    TODO:
        initial conditions,
        make sure tests for 3d case are done, I don't remember how much I
        tested the 3d case

    '''
    x = np.asarray(x)
    a = np.asarray(a)
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim > 2:
        raise ValueError('x array has to be 1d or 2d')
    nvar = x.shape[1]
    nlags = a.shape[0]
    ntrim = nlags//2
    # for x is 2d with ncols >1

    if a.ndim == 1:
        # case: identical ar filter (lag polynomial)
        return signal.convolve(x, a[:,None], mode='valid')
        # alternative:
        #return signal.lfilter(a,[1],x.astype(float),axis=0)
    elif a.ndim == 2:
        if min(a.shape) == 1:
            # case: identical ar filter (lag polynomial)
            return signal.convolve(x, a, mode='valid')

        # case: independent ar
        #(a bit like recserar in gauss, but no x yet)
        result = np.zeros((x.shape[0]-nlags+1, nvar))
        for i in range(nvar):
            # could also use np.convolve, but easier for swiching to fft
            result[:,i] = signal.convolve(x[:,i], a[:,i], mode='valid')
        return result

    elif a.ndim == 3:
        # case: vector autoregressive with lag matrices
#        #not necessary:
#        if np.any(a.shape[1:] != nvar):
#            raise ValueError('if 3d shape of a has to be (nobs,nvar,nvar)')
        yf = signal.convolve(x[:,:,None], a)
        yvalid = yf[ntrim:-ntrim, yf.shape[1]//2,:]
        return yvalid


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
