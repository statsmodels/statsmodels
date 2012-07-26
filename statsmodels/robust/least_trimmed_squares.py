# -*- coding: utf-8 -*-
"""

Created on Wed Jul 18 05:30:39 2012

Author: Josef Perktold
"""

import itertools

import numpy as np
from scipy.misc import comb

from statsmodels.regression.linear_model import OLS

class Holder(object):
    pass


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
    n_est_calls = 0
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
            n_est_calls += 1
            #print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
            r = endog - res_t_ols.predict(exog)
            #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
            idx3 = np.argsort(np.abs(r))[k_accept:]
            ii2 = np.ones(nobs, bool)
            ii2[idx3] = False
            if (ii2 == iin).all():
                if res_t_ols.ssr < best[0]:
                    #update best result so far

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

    best[1].all_dict = all_dict
    best[1].n_est_calls = n_est_calls
    return best



class LTS(object):
    '''


    TODO: variation: trim based on likelihood contribution, loglike_obs,
          and use llf instead of ssr
    '''


    def __init__(self, endog, exog, est_model=OLS):
        self.endog = endog
        self.exog = exog
        self.est_model = est_model
        self.nobs, self.k_vars = self.exog.shape
        #TODO: all_dict might not be useful anymore with new algorithm
        self.all_dict = {} #store tried outliers
        self.temp = Holder()
        self.temp.n_est_calls = 0

    def _refine_step(self, iin, k_accept):
        endog, exog = self.endog, self.exog
        nobs = self.nobs

        res_trimmed = self.est_model(endog[iin], exog[iin]).fit()
        self.temp.n_est_calls += 1
        #print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
        r = endog - res_trimmed.predict(exog)
        #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
        #partial sort would be enough: need only the smallest k_outlier
        #values
        #TODO: another version: use resid_se and outlier test to determin
        #k_outliers
        idx3 = np.argsort(np.abs(r))[k_accept:]

        ii2 = np.ones(nobs, bool)
        ii2[idx3] = False
        return res_trimmed, ii2

    def refine(self, iin, k_accept, max_nrefine=2):
        '''
        concentration step
        '''

        endog, exog = self.endog, self.exog
        #nobs = self.nobs
        #all_dict = self.all_dict

        for ib in range(max_nrefine):
            res_trimmed, ii2 = self._refine_step(iin, k_accept)
            if (ii2 == iin).all():
                converged = True
                break
            else:
                iin = ii2
                #for debugging
                outl = tuple(np.nonzero(iin)[0])
                all_dict = self.all_dict
                if outl in all_dict:
                    all_dict[outl] += 1
                else:
                    all_dict[outl] = 1
                 #remove stopping on already evaluated (seen before)
#                outl = tuple(np.nonzero(iin)[0])
#                if outl in all_dict:
#                    all_dict[outl] += 1
#                    break
#                else:
#                    all_dict[outl] = 1
        else:
            #max_nrefine reached
            converged = False

        return res_trimmed, ii2, converged


    def fit_random(self, k_trimmed, max_nstarts=10, k_start=None, n_keep=10):
        #currently random only,
        #TODO: where does exact, full enumeration go
        endog, exog = self.endog, self.exog
        nobs, k_vars = exog.shape  #instead of using attributes?

        if k_start is None:
            #TODO: check, this should be k_vars, exactly determined
            k_start = k_vars + 1
        k_accept = nobs - k_trimmed

        #stage 1
        best_stage1 = []
        ssr_keep = [np.inf] * n_keep
        #need sorted list that allows inserting and deletes worst
        #use: add if it is better than n_keep-worst (i.e. min_keep)

        iterator = subsample(nobs, k_start, max_nrep=max_nstarts)

        for ii in iterator:
            iin = ii.copy()   #TODO: do I still need a copy
            res_trimmed, ii2, converged = self.refine(iin, k_accept, max_nrefine=2)
            if res_trimmed.ssr < ssr_keep[n_keep-1]:
                best_stage1.append((res_trimmed.ssr, ii2))
                #update minkeep, shouldn't grow longer than n_keep
                #we don't drop extra indices in best_stage1
                ssr_keep.append(res_trimmed.ssr)
                ssr_keep.sort()  #inplace python sort
                del ssr_keep[n_keep:]   #remove extra

        #stage 2 : refine best_stage1 to convergence
        ssr_best = np.inf
        for (ssr, start_mask) in best_stage1:
            if ssr > ssr_keep[n_keep-1]: continue
            res_trimmed, ii2, converged = self.refine(start_mask, k_accept,
                                                      max_nrefine=100)
            if not converged:
                #warning ?
                print "refine step did not converge, max_nrefine limit reached"
            if res_trimmed.ssr < ssr_best:
                ssr_best = res_trimmed.ssr
                res_best = (res_trimmed, ii2)

        self.temp.best_stage1 = best_stage1
        self.temp.ssr_keep = ssr_keep

        return res_best

    def fit(self, k_trimmed=None, max_exact=100, random_search_options=None):
        nobs, k_vars = self.nobs, self.k_vars

        if k_trimmed is None:
            k_trimmed = nobs - int(np.trunc(nobs + k_vars + 1)//2)

        self.k_accept = k_accept = nobs - k_trimmed

        if comb(nobs, k_accept) <= max_exact:
            #index array
            raise NotImplementedError
            #iterator = itertools.combinations(range(nobs), k_accept)
        else:
            #boolean array
            options = dict(max_nstarts=10, k_start=None, n_keep=10)
            if not random_search_options is None:
                options.update(random_search_options)
            return self.fit_random(k_trimmed, **options)
