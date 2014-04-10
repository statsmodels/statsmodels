'''Levinson Durbin recursion adjusted from nitime

'''

from statsmodels.compat.python import range
import numpy as np

from statsmodels.tsa.stattools import acovf

def levinson_durbin_nitime(s, order=10, isacov=False):
    '''Levinson-Durbin recursion for autoregressive processes

    '''
    #from nitime

##    if sxx is not None and type(sxx) == np.ndarray:
##        sxx_m = sxx[:order+1]
##    else:
##        sxx_m = ut.autocov(s)[:order+1]
    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s)[:order+1]  #not tested

    phi = np.zeros((order+1, order+1), 'd')
    sig = np.zeros(order+1)
    # initial points for the recursion
    phi[1,1] = sxx_m[1]/sxx_m[0]
    sig[1] = sxx_m[0] - phi[1,1]*sxx_m[1]
    for k in range(2,order+1):
        phi[k,k] = (sxx_m[k]-np.dot(phi[1:k,k-1], sxx_m[1:k][::-1]))/sig[k-1]
        for j in range(1,k):
            phi[j,k] = phi[j,k-1] - phi[k,k]*phi[k-j,k-1]
        sig[k] = sig[k-1]*(1 - phi[k,k]**2)

    sigma_v = sig[-1]; arcoefs = phi[1:,-1]
    return sigma_v, arcoefs, pacf, phi  #return everything

import nitime.utils as ut


sxx=None
order = 10

npts = 2048*10
sigma = 1
drop_transients = 1024
coefs = np.array([0.9, -0.5])

# Generate AR(2) time series
X, v, _ = ut.ar_generator(npts, sigma, coefs, drop_transients)

s = X

import statsmodels.api as sm
sm.tsa.stattools.pacf(X)
