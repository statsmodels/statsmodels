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

min_outl_wood = [ 0,  3,  4,  5,  6,  7, 18]  #zero-based index

np.random.seed(963678)
t0 = time.time()
bestw = lts(endog_wood, exog_wood, k_trimmed=None, max_nstarts=100, max_nrefine=20, max_exact=0)
t1 = time.time()
outl = np.nonzero(bestw[-1])[0]
print bestw[0], outl + 1, t1-t0, (outl == min_outl_wood).all(), bestw[1].n_est_calls


np.random.seed(963678)
t2 = time.time()
mod_lts = LTS(endog_wood, exog_wood)
bestw2 = mod_lts.fit(random_search_options=dict(max_nstarts=100, n_keep=10))
t3 = time.time()
outl = np.nonzero(~bestw2[-1])[0]
print bestw2[0].ssr, outl + 1, t3-t2, (outl == min_outl_wood).all(), mod_lts.temp.n_est_calls
print "len(mod_lts.temp.best_stage1)", len(mod_lts.temp.best_stage1)
