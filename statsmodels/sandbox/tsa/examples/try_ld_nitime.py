'''Levinson Durbin recursion adjusted from nitime

'''

from statsmodels.compat.python import range
import numpy as np

from statsmodels.tsa.stattools import acovf

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
