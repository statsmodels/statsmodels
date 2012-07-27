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
nobs = 500
rvs = np.random.randn(nobs,5)
data_exog = rvs
data_exog = add_constant(data_exog, prepend=True)
beta = np.concatenate(([1], 0.1 * np.ones(rvs.shape[1])))
xbeta = 1 + 0.1*rvs.sum(1)
data_endog = np.random.poisson(np.exp(xbeta))

outl_type = ['random', 'biased'][1]

if not outl_type == 'biased':
    n_outl = nobs // 10
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

print "Maximum Likelihood"
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
