# -*- coding: utf-8 -*-
"""

Created on Wed Jul 18 05:30:39 2012

Author: Josef Perktold
"""

import itertools

import numpy as np
from scipy.misc import comb

from statsmodels.regression.linear_model import OLS


def subsample(n, k, max_nrep=20):
    idx = np.ones(n, bool)
    idx[:(n-k)] = False
    for i in xrange(max_nrep):
        np.random.shuffle(idx)
        yield idx


def lts(endog, exog, k_trimmed=None, max_nstarts=5, max_nrefine=20, max_exact=100):
    nobs = endog.shape[0]
    nobs2, k_vars = exog.shape
    if k_trimmed is None:
        k_trimmed = nobs - int(np.trunc(nobs+k_vars)//2)
        #k_trimmed = nobs - (nobs - k_vars) + 1
    k_start = k_vars + 1
    k_accept = nobs - k_trimmed
    best = (np.inf, np.zeros(exog.shape[1]), np.nan * np.zeros(nobs))
    all_dict = {}
    if comb(nobs, k_accept) <= max_exact:
        #index array
        iterator = itertools.combinations(range(nobs), k_accept)
    else:
        #boolean array
        #iterator = subsample(nobs, nobs - k_trimmed, max_nrep=max_nstarts)
        iterator = subsample(nobs, k_start, max_nrep=max_nstarts)
    for ii in iterator:
        if type(ii) is tuple:
            iin = np.zeros(nobs, bool)
            iin[list(ii)] = True
        else:
            iin = ii.copy()
        for ib in range(max_nrefine):
            res_t_ols = OLS(endog[iin], exog[iin]).fit()
            #print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
            r = endog - res_t_ols.predict(exog)
            #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
            idx3 = np.argsort(np.abs(r))[k_accept:]
            ii2 = np.ones(nobs, bool)
            ii2[idx3] = False
            if (ii2 == iin).all():
                if res_t_ols.ssr < best[0]:
                    #update best result so far
                    res_t_ols.all_dict = all_dict
                    best = (res_t_ols.ssr, res_t_ols, ~ii2)
                break
            else:
                iin = ii2
                outl = tuple(np.nonzero(iin)[0])
                if outl in all_dict:
                    all_dict[outl] += 1
                    break
                else:
                    all_dict[outl] = 1
        else:
            print "maxiter 20 reached"

    return best


