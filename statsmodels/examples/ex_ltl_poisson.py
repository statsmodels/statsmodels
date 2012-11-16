# -*- coding: utf-8 -*-
"""

Created on Fri Jul 27 03:57:30 2012

Author: Josef Perktold
"""

import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.robust.least_trimmed_squares import LTLikelihood
from statsmodels.discrete.discrete_model import Poisson

from statsmodels.datasets import randhie

rand_data = randhie.load()
endog_randhie = rand_data.endog
exog_randhie = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
exog_randhie = add_constant(exog_randhie, prepend=False)

#randhie raises numpy.linalg.linalg.LinAlgError: Singular matrix in LTL

np.random.seed(98765678)
nobs = 50
rvs = np.random.randn(nobs,5)
data_exog = rvs
data_exog = add_constant(data_exog, prepend=True)
beta = np.concatenate(([2], 0.31 * np.ones(rvs.shape[1])))
xbeta = 1 + 0.31*rvs.sum(1)
data_endog = np.random.poisson(np.exp(xbeta))

outl_type = ['random', 'biased'][0]

if not outl_type == 'biased':
    n_outl = 2 #nobs // 10
    outl_slice = slice(None, 5*n_outl, 5)
    outl_true = range(nobs)[outl_slice]
    data_endog[outl_slice] += 10  #add outliers
else:
    #biased outliers
    idx_big = np.argsort(np.argsort(rvs, 0), 0)
    outl_mask = (idx_big < 5).any(1)
    data_endog[outl_mask] += 8
    outl_true = np.nonzero(outl_mask)[0]

k_trim = len(outl_true)

endog, exog = data_endog, data_exog

print 'Poisson\n'

print 'true params'
print beta

print "\nMaximum Likelihood"
res_poisson = Poisson(endog, exog).fit(disp=False)
print 'parameters'
print res_poisson.params
print 'bias'
print res_poisson.params - beta

print "\nTrimmed Maximum Likelihood"
mod_poisson_ltl = LTLikelihood(endog, exog)
best = mod_poisson_ltl.fit(k_trimmed=k_trim)
outl = np.nonzero(~best[-1])[0]

print 'parameters'
print best[0].params
print 'bias'
print best[0].params - beta

print "number of outliers:", k_trim
print "\nindices of outliers detected"
print outl, (outl == outl_true).all()
print "extra outliers detected:", set(outl) - set(outl_true)
print "outliers not detected:", set(outl_true) - set(outl)

doplot = 0
if doplot:
    import matplotlib.pyplot as plt
    sort_idx = np.argsort(res_poisson.fittedvalues)
    sort_idx = np.argsort(xbeta)

    plt.plot(endog[sort_idx], 'bo', label='observed')
    #ym = endog.copy()
    #ym = endog.astype(float)
    #ym[best[-1]] = np.nan
    outl_ma = np.ma.array(endog, mask=best[-1])
    #plt.plot((endog * (~best[-1]))[sort_idx], 'ro')
    plt.plot(outl_ma[sort_idx], 'ro', label='outliers')

    #fvp = res_poisson.fittedvalues[sort_idx]
    fvp = res_poisson.predict(exog)[sort_idx]
    plt.plot(fvp, 'g-', lw=2, label='MLE predicted mean')

    fv = np.nan * np.ones(nobs)
    fv[best[-1]] = best[0].fittedvalues
    fv = best[0].predict(exog)
    plt.plot(fv[sort_idx], 'r-', lw=2, label='LTL predicted mean')
    plt.plot(np.exp(xbeta)[sort_idx], 'b-', lw=2, label='true mean')
    plt.legend()
    plt.title('Poisson Regression with outliers\n(sorted by true mean)')
    plt.show()

mod_poisson_ltl_e = LTLikelihood(endog, exog)
import time
t0 = time.time()
res_e = mod_poisson_ltl_e.fit_exact(2)
t1 = time.time()
from scipy.misc import comb
print t1 - t0, mod_poisson_ltl_e.temp.n_est_calls, round(comb(nobs, 2))
print np.nonzero(~res_e[-1])[0]
