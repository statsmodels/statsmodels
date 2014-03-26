# -*- coding: utf-8 -*-
"""

Created on Wed Jul 03 23:01:44 2013

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np

from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess
from statsmodels.miscmodels.tmodel import TArma
from statsmodels.tsa.arima_model import ARMA

nobs = 500
ar = [1, -0.6, -0.1]
ma = [1, 0.7]
dist = lambda n: np.random.standard_t(3, size=n)
np.random.seed(8659567)
x = arma_generate_sample(ar, ma, nobs, sigma=1, distrvs=dist,
                         burnin=500)

mod = TArma(x)
order = (2, 1)
res = mod.fit(order=order)
res2 = mod.fit_mle(order=order, start_params=np.r_[res[0], 5, 1], method='nm')

print(res[0])
proc = ArmaProcess.from_coeffs(res[0][:order[0]], res[0][:order[1]])

print(ar, ma)
proc.nobs = nobs
# TODO: bug nobs is None, not needed ?, used in ArmaProcess.__repr__
print(proc.ar, proc.ma)

print(proc.ar_roots(), proc.ma_roots())

from statsmodels.tsa.arma_mle import Arma
modn = Arma(x)
resn = modn.fit_mle(order=order)

moda = ARMA(x, order=order)
resa = moda.fit( trend='nc')

print('\nparameter estimates')
print('ls  ', res[0])
print('norm', resn.params)
print('t   ', res2.params)
print('A   ', resa.params)

print('\nstandard deviation of parameter estimates')
#print 'ls  ', res[0]  #TODO: not available yet
print('norm', resn.bse)
print('t   ', res2.bse)
print('A   ', resa.bse)
print('A/t-1', resa.bse / res2.bse[:3] - 1)

print('other bse')
print(resn.bsejac)
print(resn.bsejhj)
print(res2.bsejac)
print(res2.bsejhj)

print(res2.t_test(np.eye(len(res2.params))))

# TArma has no fittedvalues and resid
# TODO: check if lag is correct or if fitted `x-resid` is shifted
resid = res2.model.geterrors(res2.params)
fv = res[2]['fvec']  #resid returned from leastsq?

import matplotlib.pyplot as plt
plt.plot(x, 'o', alpha=0.5)
plt.plot(x-resid)
plt.plot(x-fv)
#plt.show()
