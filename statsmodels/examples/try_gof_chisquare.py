# -*- coding: utf-8 -*-
"""

Created on Thu Feb 28 15:37:53 2013

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np
from scipy import stats
from statsmodels.stats.gof import (chisquare, chisquare_power,
                                  chisquare_effectsize)

from numpy.testing import assert_almost_equal


nobs = 30000
n_bins = 5
probs = 1./np.arange(2, n_bins + 2)
probs /= probs.sum()
#nicer
probs = np.round(probs, 2)
probs[-1] = 1 - probs[:-1].sum()
print("probs", probs)
probs_d = probs.copy()
delta = 0.01
probs_d[0] += delta
probs_d[1] -= delta
probs_cs = probs.cumsum()
#rvs = np.random.multinomial(n_bins, probs, size=10)
#rvs = np.round(np.random.randn(10), 2)
rvs = np.argmax(np.random.rand(nobs,1) < probs_cs, 1)
print(probs)
print(np.bincount(rvs) * (1. / nobs))


freq = np.bincount(rvs)
print(stats.chisquare(freq, nobs*probs))
print('null', chisquare(freq, nobs*probs))
print('delta', chisquare(freq, nobs*probs_d))
chisq_null, pval_null = chisquare(freq, nobs*probs)

# effect size ?
d_null = ((freq / float(nobs) - probs)**2 / probs).sum()
print(d_null)
d_delta = ((freq / float(nobs) - probs_d)**2 / probs_d).sum()
print(d_delta)
d_null_alt = ((probs - probs_d)**2 / probs_d).sum()
print(d_null_alt)

print('\nchisquare with value')
chisq, pval = chisquare(freq, nobs*probs_d)
print(stats.ncx2.sf(chisq_null, n_bins, 0.001 * nobs))
print(stats.ncx2.sf(chisq, n_bins, 0.001 * nobs))
print(stats.ncx2.sf(chisq, n_bins, d_delta * nobs))
print(chisquare(freq, nobs*probs_d, value=np.sqrt(d_delta)))
print(chisquare(freq, nobs*probs_d, value=np.sqrt(chisq / nobs)))
print()

assert_almost_equal(stats.chi2.sf(d_delta * nobs, n_bins - 1),
                    chisquare(freq, nobs*probs_d)[1], decimal=13)

crit = stats.chi2.isf(0.05, n_bins - 1)
power = stats.ncx2.sf(crit, n_bins-1, 0.001**2 * nobs)
#> library(pwr)
#> tr = pwr.chisq.test(w =0.001, N =30000 , df = 5-1, sig.level = 0.05, power = NULL)
assert_almost_equal(power, 0.05147563, decimal=7)
effect_size = 0.001
power = chisquare_power(effect_size, nobs, n_bins, alpha=0.05)
assert_almost_equal(power, 0.05147563, decimal=7)
print(chisquare(freq, nobs*probs, value=0, ddof=0))
d_null_alt = ((probs - probs_d)**2 / probs).sum()
print(chisquare(freq, nobs*probs, value=np.sqrt(d_null_alt), ddof=0))


#Monte Carlo to check correct size and power of test

d_delta_r = chisquare_effectsize(probs, probs_d)
n_rep = 10000
nobs = 3000
res_boots = np.zeros((n_rep, 6))
for i in range(n_rep):
    rvs = np.argmax(np.random.rand(nobs,1) < probs_cs, 1)
    freq = np.bincount(rvs)
    res1 = chisquare(freq, nobs*probs)
    res2 = chisquare(freq, nobs*probs_d)
    res3 = chisquare(freq, nobs*probs_d, value=d_delta_r)
    res_boots[i] = [res1[0], res2[0], res3[0], res1[1], res2[1], res3[1]]

alpha = np.array([0.01, 0.05, 0.1, 0.25, 0.5])
chi2_power = chisquare_power(chisquare_effectsize(probs, probs_d), 3000, n_bins,
                             alpha=[0.01, 0.05, 0.1, 0.25, 0.5])
print((res_boots[:, 3:] < 0.05).mean(0))
reject_freq = (res_boots[:, 3:, None] < alpha).mean(0)
reject = (res_boots[:, 3:, None] < alpha).sum(0)

desired = np.column_stack((alpha, chi2_power, alpha)).T

print('relative difference Monte Carlo rejection and expected (in %)')
print((reject_freq / desired - 1) * 100)
