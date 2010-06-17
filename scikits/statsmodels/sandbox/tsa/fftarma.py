# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:53:25 2009

Author: josef-pktd

generate arma sample using fft with all the lfilter it looks slow
to get the ma representation first

apply arma filter (in ar representation) to time series to get white noise
but seems slow to be useful for fast estimation for nobs=10000

change/check: instead of using marep, use fft-transform of ar and ma
    separately, use ratio check theory is correct and example works
    DONE : feels much faster than lfilter
    -> use for estimation of ARMA
    -> use pade (scipy.misc) approximation to get starting polynomial
       from autocorrelation (is autocorrelation of AR(p) related to marep?)
       check if pade is fast, not for larger arrays ?
       maybe pade doesn't do the right thing for this, not tried yet
       scipy.pade([ 1.    ,  0.6,  0.25, 0.125, 0.0625, 0.1],2)
       raises LinAlgError: singular matrix
       also doesn't have roots inside unit circle ??
    -> even without initialization, it might be fast for estimation
    -> how do I enforce stationarity and invertibility,
       need helper function

get function drop imag if close to zero from numpy/scipy source, where?

"""

import numpy as np
import numpy.fft as fft
#import scipy.fftpack as fft
from scipy import signal
from try_var_convolve import maxabs

nobs = 10000
ar = [1, 0.0]
ma = [1, 0.0]
ar2 = np.zeros(nobs)
ar2[:2] = [1, -0.9]



uni = np.zeros(nobs)
uni[0]=1.
#arrep = signal.lfilter(ma, ar, ar2)
#marep = signal.lfilter([1],arrep, uni)
# same faster:
arcomb = np.convolve(ar, ar2, mode='same')
marep = signal.lfilter(ma,arcomb, uni) #[len(ma):]
print marep[:10]
mafr = fft.fft(marep)

rvs = np.random.normal(size=nobs)
datafr = fft.fft(rvs)
y = fft.ifft(mafr*datafr)
print np.corrcoef(np.c_[y[2:], y[1:-1], y[:-2]],rowvar=0)

arrep = signal.lfilter([1],marep, uni)
print arrep[:20]  # roundtrip to ar
arfr = fft.fft(arrep)
yfr = fft.fft(y)
x = fft.ifft(arfr*yfr).real  #imag part is e-15
# the next two are equal, roundtrip works
print x[:5]
print rvs[:5]
print np.corrcoef(np.c_[x[2:], x[1:-1], x[:-2]],rowvar=0)


# ARMA filter using fft with ratio of fft of ma/ar lag polynomial
# seems much faster than using lfilter

#padding, note arcomb is already full length
arcombp = np.zeros(nobs)
arcombp[:len(arcomb)] = arcomb
map_ = np.zeros(nobs)    #rename: map was shadowing builtin
map_[:len(ma)] = ma
ar0fr = fft.fft(arcombp)
ma0fr = fft.fft(map_)
y2 = fft.ifft(ma0fr/ar0fr*datafr)
#the next two are (almost) equal in real part, almost zero but different in imag
print y2[:10]
print y[:10]
print maxabs(y, y2)  # from chfdiscrete
#1.1282071239631782e-014
