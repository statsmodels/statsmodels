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
from scikits.statsmodels.sandbox.archive.linalg_decomp_1 import OneTimeProperty
from scikits.statsmodels.tsa.arima_process import ArmaProcess



#trying to convert old experiments to a class


class ArmaFft(ArmaProcess):
    '''fft tools for arma processes


    check whether we don't want to fix maxlags, and create new instance if
    maxlag changes. usage for different lengths of timeseries ?
    or fix frequency and length for fft

    check default frequencies w, terminology norw  n_or_w

    '''

    def __init__(self, ar, ma, n):
        #duplicates now that are subclassing ArmaProcess
        super(ArmaFft, self).__init__(ar, ma)

        self.ar = np.asarray(ar)
        self.ma = np.asarray(ma)
        self.nobs = nobs
        #could make the polynomials into cached attributes
        self.arpoly = np.polynomial.Polynomial(ar)
        self.mapoly = np.polynomial.Polynomial(ma)
        self.nar = len(ar)  #1d only currently
        self.nma = len(ma)
        self.arroots = self.arpoly.roots()
        self.maroots = self.mapoly.roots()

    def padarr(self, arr, maxlag):
        '''pad 1d array with zeros at end to have length maxlag
        function that is a method, no self used
        '''
        return np.r_[arr, np.zeros(maxlag-len(arr))]


    def pad(self, maxlag):
        arpad = np.r_[self.ar, np.zeros(maxlag-self.nar)]
        mapad = np.r_[self.ma, np.zeros(maxlag-self.nma)]
        return arpad, mapad

    def fftar(self, n):
        return fft.fft(self.padarr(self.ar, n))

    def fftma(self, n):
        return fft.fft(self.padarr(self.ma, n))

    #@OneTimeProperty  # not while still debugging things
    def fftarma(self, n):
        n = self.nobs
        return (self.fftma(n) / self.fftar(n))

    def spd(self, n):
        hw = self.fftarma(n)  #not sure, need to check normalization
        #return (hw*hw.conj()).real[n//2-1:]  * 0.5 / np.pi #doesn't show in plot
        return hw * 0.5 / np.pi

    def spdshift(self, n):
        #size = s1+s2-1
        mapadded = self.padarr(self.ma, n)
        arpadded = self.padarr(self.ar, n)
        hw = fft.fft(fft.fftshift(mapadded)) / fft.fft(fft.fftshift(arpadded))
        #return np.abs(spd)[n//2-1:]
        return (hw*hw.conj()).real[n//2-1:]

    def spddirect(self, n):
        #size = s1+s2-1
        #abs looks wrong
        return np.abs(fft.fft(self.ma, n) / fft.fft(self.ar, n))[n//2-1:]

    def spddirect2(self, n):
        #size = s1+s2-1
        hw = (fft.fft(np.r_[self.ma[::-1],self.ma], n)
                / fft.fft(np.r_[self.ar[::-1],self.ar], n))
        return (hw*hw.conj()).real[n//2-1:]

    def spdroots(self, w):
        '''spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)
        '''
        return self.spdroots_(self.arroots, self.maroots, w)

    def spdroots_(self, arroots, maroots, w):
        '''spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)

        this should go into a function
        '''
        w = np.atleast_2d(w).T
        cosw = np.cos(w)
        num = 1 + maroots**2 + 2* maroots * cosw
        den = 1 + arroots**2 + 2* arroots * cosw
        print 'num.shape, den.shape', num.shape, den.shape
        hw = 0.5 / np.pi * num.prod(-1) / den.prod(-1) #or use expsumlog
        return np.squeeze(hw)

    def filter(self, x):
        n = x.shape[0]
        if n == self.fftarma:
            fftarma = self.fftarma
        else:
            fftarma = self.fftma(n) / self.fftar(n)
            #print 'not yet, currently needs same length'
        tmpfft = fftarma * fft.fft(x)
        return fft.ifft(tmpfft)


    def acf2spdfreq(self, acovf, nfreq=100, w=None):
        '''
        not really a method
        just for comparison, not efficient for large n or long acf
        '''
        if w is None:
            w = np.linspace(0, np.pi, nfreq)[:, None]
        nac = len(acovf)
        hw = 0.5 / np.pi * (acovf[0] +
                            2 * (acovf[1:] * np.cos(w*np.arange(1,nac))).sum(1))
        return hw

    def spdmapoly(self, w, twosided=False):
        '''ma only, need division for ar, use LagPolynomial
        '''
        if w is None:
            w = np.linspace(0, np.pi, nfreq)
        0.5 / np.pi * self.mapoly(np.exp(w*1j))






def spdar1(ar, w):
    if len(ar) == 1:
        rho = ar
    else:
        rho = ar[1]
    return 0.5 / np.pi /(1 + rho*rho - 2 * rho * np.cos(w))

nobs = 200  #10000
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

ar = [1, -0.4]
ma = [1, 0.2]

arma1 = ArmaFft([1, -0.5,0,0,0,00, -0.7, 0.3], [1, 0.8], nobs)

nfreq = nobs
w = np.linspace(0, np.pi, nfreq)
w2 = np.linspace(0, 2*np.pi, nfreq)

import matplotlib.pyplot as plt

plt.figure()
spd1 = arma1.spd(2**10)
print spd1.shape
_ = plt.plot(spd1)
plt.title('spd fft')

plt.figure()
spd2 = arma1.spdshift(2**10)
print spd2.shape
_ = plt.plot(spd2)
plt.title('spd fft shift')

plt.figure()
spd3 = arma1.spddirect(2**10)
print spd3.shape
_ = plt.plot(spd3)
plt.title('spd fft direct')

plt.figure()
spd3b = arma1.spddirect2(2**10)
print spd3b.shape
_ = plt.plot(spd3b)
plt.title('spd fft direct mirrored')

plt.figure()
spdr = arma1.spdroots(w)
print spdr.shape
plt.plot(w, spdr)
plt.title('spd from roots')

plt.figure()
spdar1 = spdar1(arma1.ar, w)
print spdar1.shape
_ = plt.plot(w, spdar1)
plt.title('spd ar1')


plt.figure()
wper, spdper = arma1.arma_periodogram(nfreq)
print spdper.shape
_ = plt.plot(w, spdper)
plt.title('periodogram')

startup = 1000
rvs = arma1.generate_sample(startup+10000)[startup:]
import matplotlib.mlab as mlb
plt.figure()
sdm, wm = mlb.psd(x)
print 'sdm.shape', sdm.shape
sdm = sdm.ravel()
plt.plot(wm, sdm)
plt.title('matplotlib')

from nitime.algorithms import LD_AR_est
#yule_AR_est(s, order, Nfreqs)
wnt, spdnt = LD_AR_est(rvs, 10, 512)
plt.figure()
print 'spdnt.shape', spdnt.shape
_ = plt.plot(spdnt.ravel())
print spdnt[:10]
plt.title('nitime')


plt.show()
