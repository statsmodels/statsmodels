# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:53:21 2018

Author: Josef Perktold
License: BSD-3
"""

from collections import defaultdict
import numpy as np

from statsmodels.base._penalties import SCADSmoothed


class ScreeningResults(object):
    def __init__(self, screener, **kwds):
        self.screener = screener
        self.__dict__.update(**kwds)


class VariableScreening(object):
    """Ultra-high, conditional sure independence screening

    This is an adjusted version of Fan's sure independence screening.


    TODO Notes
    penalization weights need to be used to avoid penalizing always kept
    exog. The length of weights vector depends on the number of variables
    included in the candidate model.
    This is not included yet. We SCAD penalize all parameters, but large
    parameters are not penalized by the SCAD penalty function.

    Does user provide Penalized class or base class?
    -> refactor so we use Penalized class with pen_weight=0 as base class

    pen_weight is currently not a tuning option that can be changed by user.

    pearson_resid: inconsistency in whether penalized or half-trimmed params
    are used. Related issue: GLM resid_pearson does not include freq_weights.

    freq_weights are not supported in this. Candidate ranking uses
    moment condition with resid_pearson without freq_weights.

    fit_kwds are missing, e.g. we need to avoid irls in GLM

    variable names: do we keep track of those?

    currently only supports numpy arrays, now exog type check or conversion

    extensions to lazy or patched exog.
    Actually, that should work already by calling screen_vars with patches
    of exog, and then with a combined exog of best subset exog.

    currently only single columns are selected, no terms (multi column exog)

    """

    def __init__(self, model, base_class, **kwargs):

        self.model = model
        self.model_class = model.__class__
        self.init_kwds = model._get_init_kwds()
        # pen_weight and penal are explicitly included
        # TODO: check what we want to do here
        self.init_kwds.pop('pen_weight', None)
        self.init_kwds.pop('penal', None)

        self.endog = model.endog
        self.exog_keep = model.exog
        self.penal = SCADSmoothed(0.1, c0=0.0001)

        # this is only needed for initial start_params
        self.base_class = base_class

        # option for screening algorithm
        self.k_add = 5
        self.k_max_add = 10
        self.threshold_trim = 1e-4
        self.k_max_included = 20

    def ranking_measure(self, res_pen, exog, keep=None):
        endog = self.endog
        # TODO: does it really help to change/trim params
        # we are not reestimating with trimmed model
        p = res_pen.params.copy()
        if keep is not None:
            p[~keep] = 0
        if hasattr(res_pen, 'resid_pearson'):
            # this is different from the else
            # here we use the resid_pearson from res_pen which includes
            # dropped variables/params
            resid_pearson = res_pen.resid_pearson
        else:
            predicted = res_pen.model.predict(p)
            # this is currently hardcoded for Poisson
            resid_pearson = (endog - predicted) / np.sqrt(predicted)

        mom_cond = np.abs(resid_pearson.dot(exog))**2
        return mom_cond

    def screen_exog(self, exog, endog=None, maxiter=5, disp=0):
        model_class = self.model_class
        if endog is None:
            # allow a different endog than used in model
            endog = self.endog
        x0 = self.exog_keep
        x1 = exog
        k0 = x0.shape[1]
        # TODO: remove the need for x, use x1 separately from x0
        # needs change to idx to be based on x1 (candidate variables)
        x = np.column_stack((x0, x1))
        nobs, k_vars = x.shape

        history = defaultdict(list)
        idx_nonzero = [0]
        keep = np.array([True])
        idx_excl = np.arange(1, k_vars)
        #start_params = [endog.mean()]
        res_pen = model_class(endog, x0, **self.init_kwds).fit(disp=disp)
        start_params = res_pen.params

        for _ in range(maxiter):
            # This does not work, droping the Poisson fit creates problems

            mom_cond = self.ranking_measure(res_pen, x1, keep=keep)
            mcs = np.sort(mom_cond)[::-1]
            #print(mcs[:10])

            threshold = mcs[max((self.k_max_add, k0 + self.k_add))]
            #idx = np.concatenate((idx_nonzero, np.nonzero(mom_cond > threshold)[0] + 1))
            idx = np.concatenate((idx_nonzero, idx_excl[mom_cond > threshold]))
            start_params2 = np.zeros(len(idx))
            start_params2[:len(start_params)] = start_params

            res_pen = model_class(endog, x[:, idx], penal=self.penal,
                                       pen_weight=nobs * 10,
                                       **self.init_kwds).fit(method='bfgs',
                                                start_params=start_params2,
                                                warn_convergence=False, disp=disp,
                                                skip_hessian=True)

            keep = np.abs(res_pen.params) > self.threshold_trim
            # use largest params to keep
            if len(res_pen.params) > self.k_max_included:
                # TODO we can use now np.partition with partial sort
                thresh_params = np.sort(np.abs(res_pen.params))[-self.k_max_included]
                keep2 = np.abs(res_pen.params) > thresh_params
                keep = np.logical_and(keep, keep2)
            idx_nonzero = idx[keep]
            # TODO: problem with order between using mask and indexing
            #idx_nonzero.sort()

            if disp:
                print(idx_nonzero)
            x0 = x[:, idx_nonzero]
            start_params = res_pen.params[keep]

            # use mask to get excluded
            mask_excl = np.ones(k_vars, dtype=bool)
            mask_excl[idx_nonzero] = False
            idx_excl = np.nonzero(mask_excl)[0]
            x1 = x[:, idx_excl]
            history['idx_nonzero'].append(idx_nonzero)
            history['keep'].append(keep)
            history['params_keep'].append(start_params)

        # final esimate
        res_final = model_class(endog, x[:, idx_nonzero], penal=self.penal,
                                pen_weight=nobs * 10,
                                **self.init_kwds).fit(method='bfgs',
                                          start_params=res_pen.params[keep],
                                          warn_convergence=False, disp=disp)

        res = ScreeningResults(self,
                               results_pen = res_pen,
                               results_final = res_final,
                               idx_nonzero = idx_nonzero,
                               idx_excl = idx_excl,
                               start_params = start_params,
                               history = history)
        return res
