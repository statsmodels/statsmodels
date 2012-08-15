# -*- coding: utf-8 -*-
"""

Created on Wed Jul 18 05:30:39 2012

Author: Josef Perktold
"""
import time
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels import robust
from statsmodels.robust.least_trimmed_squares import (lts, LTS, subsample,
                                        scale_lts, EfficientLTS, LTSRLM)
#from statsmodels.tools.tools import add_constant

seed = np.random.randint(0, 999999)
#494249
#seed = 258729 #difference between rlm-tm and rlm-tm-w
print 'seed:', seed
np.random.seed(seed)

nobs, k_vars = 200, 4
n_outl_mix = 10, 5
n_outl_mix = 5, 5 #70, 20
n_outl = sum(n_outl_mix)

beta0 = np.ones(k_vars)
beta1 = np.ones(k_vars)
beta1[:3] = 0
from scipy import stats
distr0 = stats.norm(scale=0.5).rvs
distr1 = stats.t(3, scale=1).rvs

exog = np.column_stack((np.ones(nobs), np.random.randn(nobs, k_vars-1)))

endog_true = np.hstack((np.dot(exog[:nobs-n_outl_mix[0]], beta0),
                        np.dot(exog[nobs-n_outl_mix[0]:], beta1)))
noise = np.hstack((distr1(size=n_outl_mix[1])**2,
                   distr0(size=nobs - n_outl_mix[1]) ))
endog = endog_true + noise
min_outl = range(n_outl_mix[1]) + range(nobs)[nobs-n_outl_mix[0]:]

nstarts = 100

np.random.seed(963678)
t0 = time.time()
bestw = lts(endog, exog, k_trimmed=None, max_nstarts=nstarts, max_nrefine=100, max_exact=0)
t1 = time.time()
outl = np.nonzero(bestw[-1])[0]
print bestw[0], outl + 1, t1-t0#, (outl == min_outl).all(), bestw[1].n_est_calls


#np.random.seed(963678)
t2 = time.time()
mod_lts = LTS(endog, exog)
bestw2 = mod_lts.fit(random_search_options=dict(max_nstarts=nstarts, n_keep=30))
t3 = time.time()
outl = np.nonzero(~bestw2[-1])[0]
print bestw2[0].ssr, outl + 1, t3-t2#, (outl == min_outl).all(), mod_lts.temp.n_est_calls
print "len(mod_lts.temp.best_stage1)", len(mod_lts.temp.best_stage1)

print 'sum and len of all_dict.values()'
print 'lts', sum(bestw[1].all_dict.values()), len(bestw[1].all_dict.values())
print 'LTS', sum(mod_lts.all_dict.values()), len(mod_lts.all_dict.values())
print mod_lts.temp.n_refine_steps
print len(mod_lts.temp.ssr_keep)

#for (ssr,ii) in mod_lts.temp.best_stage1: print ssr, np.nonzero(ii)[0]


res_ols = OLS(endog, exog).fit()
print res_ols.params, 'ols'

inl = slice(n_outl_mix[0], nobs-n_outl_mix[0])
res_olst = OLS(endog[inl], exog[inl]).fit()
print res_olst.params, 'ols-t'

#mod_lts_e = LTS(endog, exog)
#best_e = mod_lts.fit_exact(1)
print bestw2[0].params, 'bestw2'


nstarts = 500
mod_lts = LTS(endog, exog)
k_trimmed = n_outl
so = dict(max_nstarts=nstarts, n_keep=30, max_nrefine_st1=1)
#making so to search harder didn't change the result
bestwt = mod_lts.fit(k_trimmed=k_trimmed, random_search_options=so)
bestwt_outl = np.nonzero(~bestwt[-1])[0]
print bestwt[0].params, 'bestwt'
print bestwt_outl

hsal = RLM(endog, exog, M=robust.norms.HuberT()).fit()   # default M
print hsal.params, 'rlm'
print np.nonzero(hsal.weights < 0.9)[0]
hsalw = RLM(endog, exog, M=robust.norms.HuberT()).fit(weights=bestw2[-1].astype(float))   # default M
print hsalw.params, 'rlm-w'
print np.nonzero(hsalw.weights < 0.9)[0]

hsalta = RLM(endog, exog, M=robust.norms.TrimmedMean()).fit()
print hsalta.params, 'rlm-tm'
print np.nonzero(hsalta.weights < 0.9)[0]

hsalt = RLM(endog, exog, M=robust.norms.TrimmedMean()).fit(weights=bestw2[-1].astype(float))   # default M
print hsalt.params, 'rlm-tm-w'
print np.nonzero(hsalt.weights < 0.9)[0]

restb3a = RLM(endog, exog, M=robust.norms.TukeyBiweight(c=4.685)).fit()
print restb3a.params, 'rlm-tb'
restb3 = RLM(endog, exog, M=robust.norms.TukeyBiweight(c=4.685)).fit(weights=bestw2[-1].astype(float))
print restb3.params, 'rlm-tb-w'
restb4 = RLM(endog, exog, M=robust.norms.TukeyBiweight(c=4.685 *0.75)).fit(weights=bestw2[-1].astype(float))
print restb4.params, 'rlm-tb-0.75-w'

print "fixing the scale"
scale_adj = scale_lts(bestw2[0].ssr, (~bestw2[1]).sum(), len(bestw2[1]), k_vars, distr='norm')
init = dict(scale=scale_adj)
restb3b = RLM(endog, exog, M=robust.norms.TukeyBiweight(c=4.685)).fit(
            weights=bestw2[-1].astype(float), update_scale=False, init=init)
print restb3b.params
restb3c = RLM(endog, exog, M=robust.norms.TrimmedMean()).fit(
            weights=bestw2[-1].astype(float), update_scale=False, init=init)
print restb3c.params

mod_mm = LTSRLM(endog, exog, M=robust.norms.TukeyBiweight(c=4.685))
res_mm = mod_mm.fit()
print res_mm.params, 'MM-E, LTSRLM'

breakdown, efficiency = 0.5, 0.9
mod_elts = EfficientLTS(endog, exog)
elts, elts_mask = mod_elts.fit(breakdown, efficiency, random_search_options=None, maxiter=10)
print elts.params, 'elts'
#print np.nonzero(~elts_mask)
print (~elts_mask).sum()

import matplotlib.pyplot as plt
s_idx = np.argsort(endog)
#s_idx = np.argsort(exog[:,1])
plt.figure()
plt.plot(endog[s_idx], 'o')
plt.plot(hsal.predict(exog)[s_idx])
plt.figure()
plt.plot(noise)
#plt.show()

for ii in range(5):
    noise = np.hstack((distr1(size=n_outl_mix[1])**2,
                       distr0(size=nobs - n_outl_mix[1]) ))
    endog = endog_true + noise
    mod_lts = LTS(endog, exog)
    bestwt = mod_lts.fit(k_trimmed=k_trimmed, random_search_options=so)
    #bestwt_outl = np.nonzero(~bestwt[-1])[0]
    print bestwt[0].params, 'bestwt',
    hsalt = RLM(endog, exog, M=robust.norms.TrimmedMean()).fit(weights=bestwt[-1].astype(float))   # default M
    print hsalt.params, 'rlm-tm-w'
    #print np.nonzero(hsalt.weights < 0.9)[0]
