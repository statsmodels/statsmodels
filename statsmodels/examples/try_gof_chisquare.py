# -*- coding: utf-8 -*-
"""

Created on Thu Feb 28 15:37:53 2013

Author: Josef Perktold
"""

import numpy as np
from scipy import stats
from statsmodels.stats.gof import chisquare, chisquare_power

from numpy.testing import assert_almost_equal


nobs = 30000
n_bins = 5
probs = 1./np.arange(2, n_bins + 2)
probs /= probs.sum()
#nicer
probs = np.round(probs, 2)
probs[-1] = 1 - probs[:-1].sum()
print "probs", probs
probs_d = probs.copy()
delta = 0.01
probs_d[0] += delta
probs_d[1] -= delta
probs_cs = probs.cumsum()
#rvs = np.random.multinomial(n_bins, probs, size=10)
#rvs = np.round(np.random.randn(10), 2)
rvs = np.argmax(np.random.rand(nobs,1) < probs_cs, 1)
print probs
print np.bincount(rvs) * (1. / nobs)


freq = np.bincount(rvs)
print stats.chisquare(freq, nobs*probs)
print 'null', chisquare(freq, nobs*probs)
print 'delta', chisquare(freq, nobs*probs_d)
chisq_null, pval_null = chisquare(freq, nobs*probs)

# effect size ?
d_null = ((freq / float(nobs) - probs)**2 / probs).sum()
print d_null
d_delta = ((freq / float(nobs) - probs_d)**2 / probs_d).sum()
print d_delta
d_null_alt = ((probs - probs_d)**2 / probs_d).sum()
print d_null_alt

chisq, pval = chisquare(freq, nobs*probs_d)
print stats.ncx2.sf(chisq_null, n_bins, 0.001 * nobs)
print stats.ncx2.sf(chisq, n_bins, 0.001 * nobs)
print stats.ncx2.sf(chisq, n_bins, d_delta * nobs)

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
print chisquare(freq, nobs*probs, value=0, ddof=0)
d_null_alt = ((probs - probs_d)**2 / probs).sum()
print chisquare(freq, nobs*probs, value=np.sqrt(d_null_alt), ddof=0)
