# -*- coding: utf-8 -*-
"""

Created on Wed Jul 18 05:30:39 2012

Author: Josef Perktold
"""

import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels import robust
from statsmodels.robust.least_trimmed_squares import lts, LTS, subsample
#from statsmodels.tools.tools import add_constant

from statsmodels.robust.tests.results.results_lts import (
        endog_stackloss, exog_stackloss, endog_wood, exog_wood,
        endog_aircraft, exog_aircraft, endog_salin, exog_salin)






outl_sal = [1, 5, 8, 9, 10, 11, 13, 16, 23, 24, 25, 28] #1-based indices
inl_sal = [i for i in range(28) if i+1 not in outl_sal]





#print np.array(list(subsample(10, 8))) #this always has identical idx, mutable

for ii in subsample(10, 8):
    print (2**(ii*np.arange(10))).sum(), ii.sum(), ii



h1 = RLM(endog_stackloss, exog_stackloss,\
            M = robust.norms.HuberT()).fit()   # default M

h2 = RLM(endog_stackloss, exog_stackloss,\
            M = robust.norms.HuberT()).fit(cov="H2")
h3 = RLM(endog_stackloss, exog_stackloss,\
            M = robust.norms.HuberT()).fit(cov="H3")

for res in [h1, h2, h3]:
    print np.nonzero(res.weights < 0.99)[0] + 1, res.params

endog, exog = endog_stackloss, exog_stackloss
nobs = endog.shape[0]
k_trimmed = 9
k_accept = 21-k_trimmed
best_idx_all = []
best = (np.inf, np.zeros(exog.shape[1]), np.nan * np.zeros(nobs))
for ii in subsample(21, 21-k_trimmed, max_nrep=5):
    iin = ii.copy()
    for ib in range(20):
        res_t_ols = OLS(endog[iin], exog[iin]).fit()
        print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
        r = endog - res_t_ols.predict(exog)
        #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
        idx3 = np.argsort(np.abs(r))[k_accept:]
        ii2 = np.ones(21, bool)
        ii2[idx3] = False
        if (ii2 == iin).all():
            if res_t_ols.ssr < best[0]:
                #update best result so far
                best = (res_t_ols.ssr, res_t_ols.params, ~ii2)
                best_idx_all.append(tuple(np.nonzero(~ii2)[0]))
            break
        else:
            iin = ii2
    else:
        print "maxiter 20 reached"

#combinations Skipper
#int(np.round(comb(21, 12)))

print '\nbest result'
print best[:2], np.nonzero(best[-1])[0] + 1

nobs = 28
nobs_used = 28

inl_sal = np.array(inl_sal)
idx0 = nobs - nobs_used
idx = inl_sal[inl_sal > (nobs - nobs_used)] - (nobs - nobs_used)
res_t_ols = OLS(endog_salin[idx], exog_salin[idx]).fit()

best = lts(endog_salin, exog_salin, k_trimmed=None, max_nstarts=5, max_nrefine=20)
print best[0], np.nonzero(best[-1])[0] + 1

hsal = RLM(endog_salin, exog_salin, M=robust.norms.HuberT()).fit()   # default M
print np.nonzero(hsal.weights<0.99)[0] + 1

best = lts(endog_salin, exog_salin, k_trimmed=3, max_nstarts=1, max_nrefine=20, max_exact=1000)
print best[0], np.nonzero(best[-1])[0] + 1

bestr = lts(endog_salin, exog_salin, k_trimmed=3, max_nstarts=5, max_nrefine=20, max_exact=0)
print bestr[0], np.nonzero(bestr[-1])[0] + 1

bestr = lts(endog_salin, exog_salin, k_trimmed=12, max_nstarts=500, max_nrefine=20, max_exact=0)
print bestr[0], np.nonzero(bestr[-1])[0] + 1

besta = lts(endog_aircraft, exog_aircraft, k_trimmed=None, max_nstarts=100, max_nrefine=20, max_exact=0)
print besta[0], np.nonzero(besta[-1])[0] + 1

set(best_idx_all)
s_all0 = set((i for j in best_idx_all for i in j))
s_all = reduce(set.union, map(set, best_idx_all))
reduce(set.intersection, map(set, best_idx_all))
[s_all - set(j) for j in best_idx_all]

#idea break when we have checked a outlier set already
print (0, 2, 3, 5, 10, 12, 14, 19, 20).__hash__()

bestw = lts(endog_wood, exog_wood, k_trimmed=None, max_nstarts=100, max_nrefine=20, max_exact=0)
print bestw[0], np.nonzero(bestw[-1])[0] + 1

bestw2 = LTS(endog_wood, exog_wood).fit()
print bestw2[0], np.nonzero(bestw2[-1])[0] + 1
