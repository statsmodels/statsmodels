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
from statsmodels.robust.least_trimmed_squares import lts, LTS, subsample
#from statsmodels.tools.tools import add_constant

from statsmodels.robust.tests.results.results_lts import (
        endog_stackloss, exog_stackloss, endog_wood, exog_wood,
        endog_aircraft, exog_aircraft, endog_salin, exog_salin)


#compared with
#Douglas M. Hawkins, 1994, The feasible solution algorithm
#for least trimmed squares regression,
#Computational Statistics & Data Analysis 17 (1994) 185-196, North-Holland

#zero-based index
min_outl_wood = [ 0,  3,  4,  5,  6,  7, 18]
min_outl_stackloss = [ 0,  1,  2,  3,  7, 12, 13, 19, 20]
min_outl_aircraft = [ 1,  2,  3, 11, 15, 17, 18, 20, 21]
min_outl_salin = [ 0,  4,  7,  8,  9, 10, 12, 15, 22, 23, 24, 27]
ssr_wood = 1.168
ssr_stackloss = 1.6371
ssr_aircraft = 36.034
ssr_salin = 0.70

endog, exog = endog_wood, exog_wood
min_outl = min_outl_wood

endog, exog = endog_stackloss, exog_stackloss
min_outl = min_outl_stackloss

endog, exog, min_outl = endog_salin, exog_salin, min_outl_salin

endog, exog, min_outl = endog_aircraft, exog_aircraft, min_outl_aircraft


expand = 1
if expand:
    m = 4
    endog = np.tile(endog, m)
    exog = np.tile(exog, (m,1))
    min_outl = [ 2,  8,  9, 10, 12, 13, 15, 17, 18, 20, 21, 25, 31, 32, 33, 35, 36,
           38, 40, 41, 43, 44, 48, 54, 55, 56, 58, 59, 61, 63, 64, 66, 67, 71,
           77, 78, 79, 81, 82, 84, 86, 87, 89, 90]
    #mo = min_outl * m

nstarts = 100

np.random.seed(963678)
t0 = time.time()
bestw = lts(endog, exog, k_trimmed=None, max_nstarts=nstarts, max_nrefine=100, max_exact=0)
t1 = time.time()
outl = np.nonzero(bestw[-1])[0]
print bestw[0], outl + 1, t1-t0, (outl == min_outl).all(), bestw[1].n_est_calls


#np.random.seed(963678)
t2 = time.time()
mod_lts = LTS(endog, exog)
bestw2 = mod_lts.fit(random_search_options=dict(max_nstarts=nstarts, n_keep=30))
t3 = time.time()
outl = np.nonzero(~bestw2[-1])[0]
print bestw2[0].ssr, outl + 1, t3-t2, (outl == min_outl).all(), mod_lts.temp.n_est_calls
print "len(mod_lts.temp.best_stage1)", len(mod_lts.temp.best_stage1)

print 'sum and len of all_dict.values()'
print 'lts', sum(bestw[1].all_dict.values()), len(bestw[1].all_dict.values())
print 'LTS', sum(mod_lts.all_dict.values()), len(mod_lts.all_dict.values())
print mod_lts.temp.n_refine_steps
print len(mod_lts.temp.ssr_keep)

#for (ssr,ii) in mod_lts.temp.best_stage1: print ssr, np.nonzero(ii)[0]

if expand:
    #aircraft only (I also checked salin)
    k_trim = len(min_outl_aircraft) * 4
    nobs = len(endog_aircraft)
    bestw = lts(endog, exog, k_trimmed=k_trim, max_nstarts=nstarts, max_nrefine=100, max_exact=0)
    print (np.remainder(np.nonzero(bestw[-1])[0], nobs) == min_outl_aircraft*4).all()
