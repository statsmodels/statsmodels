# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:28:26 2015

Author: Josef Perktold
License: BSD-3

"""

from __future__ import division

import numpy as np
from scipy import stats

from statsmodels.stats._proportion_exact import ExactTwoProportion

ex = 2
n1, n2 = 15, 15
yo1 = 7
yo2 = 12

if ex == 2:
    n1, n2 = 69, 88
    yo1 = 67
    yo2 = 76

pt = ExactTwoProportion(yo1, n1, yo2, n2)

print(pt.chisquare_proportion_indep(yo1, yo2))
print(pt.chisquare_proportion_indep(yo1, yo2, alternative='todo'))
print(pt.chisquare_proportion_indep(yo1, yo2, alternative='todo'))

print(pt.chisquare_proportion_indep(np.arange(0, 16), np.arange(0, 16))[1])
stats.binom.pmf(np.arange(n1 + 1), n1, 0.5)

if ex == 2:
    alternative='2-sided' #'todo'
else:
    alternative='todo'
sto, pvo = pt.chisquare_proportion_indep(yo1, yo2, alternative=alternative)

ys1 = np.arange(n1 + 1)
ys2 = np.arange(n2 + 1)

st, pv =  pt.chisquare_proportion_indep(ys1[None, :], ys2[:,None], alternative=alternative)

prob1 = stats.binom.pmf(np.arange(n1 + 1), n1, pt.prob_pooled)
prob2 = stats.binom.pmf(np.arange(n2 + 1), n1, pt.prob_pooled)

prob = prob1[None, :] * prob2[:, None]
pvemp = prob[pv < pvo].sum()
print(pvemp)

st, pv =  pt.chisquare_proportion_indep(ys1[None, :], ys2[:,None],
                                     prob_var=pt.prob_pooled,
                                     alternative=alternative)

prob1 = stats.binom.pmf(np.arange(n1 + 1), n1, pt.prob_pooled)
prob2 = stats.binom.pmf(np.arange(n2 + 1), n1, pt.prob_pooled)

prob = prob1[None, :] * prob2[:, None]
pvemp = prob[pv < pvo].sum()
print(pvemp)

res = []
for prob0 in np.linspace(0.001, 0.999, 1001):
    #print(prob,)
    st, pv =  pt.chisquare_proportion_indep(ys1[None, :], ys2[:,None],
                                     #prob_var=prob0,
                                     alternative=alternative)

    prob1 = stats.binom.pmf(np.arange(n1 + 1), n1, prob0)
    prob2 = stats.binom.pmf(np.arange(n2 + 1), n2, prob0)

    prob = prob1[None, :] * prob2[:, None]
    pvemp = prob[pv <= pvo].sum()
    #print(pvemp)
    res.append([prob0, pvemp])

res =np.array(res)
pvm_ind = res[:,1].argmax(0)
print(res[pvm_ind])

print(pt.pvalue_exactdist_mle())
import time
t0 = time.time()
pres = pt.pvalue_exact_sup()
t1 = time.time()
print(pres[:2])
print('time', t1 - t0)

t0 = time.time()
presbb = pt.pvalue_exact_sup(grid=('bb', 0.0001, 101))
t1 = time.time()
print(pres[:2])
print('time', t1 - t0)
