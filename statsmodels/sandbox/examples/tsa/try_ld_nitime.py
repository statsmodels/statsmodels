'''Levinson Durbin recursion adjusted from nitime

'''

import numpy as np

import nitime.utils as ut

import statsmodels.api as sm

sxx=None
order = 10

npts = 2048*10
sigma = 1
drop_transients = 1024
coefs = np.array([0.9, -0.5])

# Generate AR(2) time series
X, v, _ = ut.ar_generator(npts, sigma, coefs, drop_transients)

s = X

sm.tsa.stattools.pacf(X)
