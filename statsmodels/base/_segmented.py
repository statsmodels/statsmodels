# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:32:38 2015

Author: Josef Perktold
License: BSD-3
"""
from __future__ import division

import numpy as np
from scipy import optimize

from statsmodels.regression.linear_model import OLS


def segmented(model, idx):
    """segmented regression with a single knot or break point

    """

    exog_k = np.asarray(model.exog[:, idx], order='F')
    exog = model.exog.copy(order='F')

    # TODO: generalize, option for objective, callback
    def ssr(a):
        exog[:, -1] = np.clip(exog_k - a, 0, np.inf)
        ssr_ = OLS(model.endog, exog).fit().ssr
        return ssr_

    brack = np.percentile(exog_k, [70, 85])
    print('brack', brack)
    res_optim = optimize.brent(ssr, brack=brack)
    # TODO check convergence

    # redo, don't rely on having the correct last estimate in the loop:
    exog[:, -1] = np.clip(exog_k - res_optim, 0, np.inf)
    res_best = OLS(model.endog, exog).fit()
    # attach extra result
    res_best.knot_location = res_optim
    return res_best


class Segmented(object):
    """class to search for a variable transformation in regression

    Warning: This class makes in place transformation.
    However, if copy_data is True, then the original model remains unchanged.


    TODO: There might still be a problem with boundary segments in use of
    bounds. brent can go outside of bounds, but I think local minimum forces
    argmin knot to be inside bounds. We need to make sure outside bound are
    close enough to the boundary of the support of exog_k.

    Currently there is no safeguard if user requests too many knots compared
    to the number of observations. (But linear model uses pinv.)


    Implementation notes:
    This became more complicated than necessary to be able to reuse exog
    arrays with inplace modification. (This might not improve performance by
    much if exog has few columns.)

    Looping over columns should reduce optimization problems, make local
    minima less likely and avoid non-unique knot indices or restriction on
    knot order. CORRECTION: brent doesn't work that way, fminbound does. brent
    doesn't preserve order.

    We need to create new models in loop because models are not designed for
    modifications to exog, at least not without "cheating" in the case of OLS.

    TODO: pandas and name handling. The final model does not yet recuperate
    the exog names or row indices.

    Extensions:
    This only handles regression splines where only one column
    is affected by a knot location. B-splines, others need adjustments to
    several columns if a knot moves. That is not the main use case for this.

    not yet: projection on left out exog in original outcome model, currently
    assumes that we have full exog for OLS, or that this is part of backfitting.
    """

    def __init__(self, model, exog_source, target_indices, bounds=None,
                 degree=1, copy_data=True):
        self.model_base = model
        self.model_class = model.__class__
        # currently args and kwds are not supported, no GLM yet
        if copy_data:
            self.endog = model.endog.copy()
            self.exog = model.exog.copy()
        else:
            self.endog = model.endog
            self.exog = model.exog

        self.exog_source = exog_source
        self.target_indices = target_indices
        self.k_cols = len(target_indices)
        self.nobs = model.endog.shape[0]
        self.degree = degree
        self.bounds = bounds

        # to initialize splines
        perc = np.linspace(1, 100, self.k_cols + 2)
        self.quantiles = np.percentile(exog_source, perc)

    @classmethod
    def from_model(cls, model, exog_add, k_knots=1, degree=1):
        """adds a variable for linear segment, splines of degree

        No missing value handling.

        Parameters
        ----------
        model : instance of a model
        exog_add : array_like, 1-D
           variable to add
        k_knots : int
           number of interior knots, the number of segments is k_nots + 1
        degree : int
           degree (terminology?) of spline, linear, degree=1, is default

        Returns
        -------
        seg : instance of class Segmented

        """
        exog_add = np.asarray(exog_add)
        exog = model.exog
        nobs, k_vars = exog.shape
        # TODO: check number of observations per segment

        # we want bounds to be interior
        perc = np.linspace(0, 100, k_knots + 2)
        n_min = k_vars + degree + k_knots + 2  # 2 higher than perfect fit
        p_min = np.ceil(n_min) * 100 / nobs
        q = np.percentile(exog_add, np.clip(perc, p_min, 100 - p_min))
        # the following part needs to be used for predict given q = bounds[1:-1]
        if degree == 1:
            vander = exog_add
        else:
            vander = exog_add[:, None]**np.arange(1, degree + 1)
        columns = [cls.transform(exog_add,  qi, degree=1) for qi in q[1:-1]]
        exog = np.column_stack([exog, vander] + columns)

        target_indices = list(range(k_vars + degree, exog.shape[1]))
        mod_base = model.__class__(model.endog, exog)
        seg = Segmented(mod_base, exog_add, target_indices, bounds=q,
                        degree=degree, copy_data=False)

        return seg


    def get_objective(self, exog_k, target_idx):
        endog = self.endog
        exog = self.exog  #.copy() no copy, use inplacce
        def ssr(tparam):
            exog[:, target_idx] = self.transform(exog_k,  tparam, self.degree)
            # TODO: Refactor OLS, or cheat to not create new model
            ssr_ = self.model_class(endog, exog).fit().ssr
            return ssr_

        return ssr

    @staticmethod
    def transform(exog_k,  tparam, degree):
        """linear spline

        We call this from class method so we cannot have instance attributes
        """
        if degree == 1:
            return np.maximum(exog_k - tparam, 0)
        else:
            return np.maximum((exog_k - tparam)**degree, 0)


    def _fit_one(self, bounds):
        pass


    def _fit_all(self, bounds=None, maxiter=1, method='brent'):
        exog_k = self.exog_source
        if bounds is None:
            if self.bounds is not None:
                bounds = self.bounds
            else:
                raise ValueError('bounds not specified')

        for it in range(maxiter):
            for k in range(self.k_cols):
                obj = self.get_objective(exog_k, self.target_indices[k])

                if method == 'brent':
                    # Note: brent with 2 value brackets uses it for (a, b), not for (a,c)
                    brack = bounds[k : k+2]
                    #brack = bounds[k],  bounds[k+2]
                    res_optim = optimize.brent(obj, brack=brack)
                else:
                    brack = bounds[k],  bounds[k+2]
                    res_optim = optimize.fminbound(obj, *brack)
                tparam = res_optim   # we will want more returns, check convergence
                # note self.exog is supposed to be modified inplace
                # don't rely on last call of optimizer
                objvalue = obj(tparam)  # for checking, assert
                bounds[k+1] = tparam  # new knot

        self.bounds = bounds
        return objvalue

    def get_results(self):
        endog = self.endog
        exog = self.exog
        res = self.model_class(endog, exog).fit()
        res.knot_locations = self.bounds[1:-1]
        return res


    def add_knot(self, maxiter=1):
        """insert knot and reoptimize

        This only insert into interior segments and needs at least 3 segments
        to start with.

        """
        # start with current state of model
        # use new Segmented instance with augmented exog
        endog = self.endog
        exog = np.column_stack((self.exog, self.exog_source))
        target_indices = list(self.target_indices) + [exog.shape[1] - 1]
        target_indices.sort()
        # sort does not affect self.target_indices on py 3.4
        #TODO: add test to make sure

        # We need to insert a knot and then reoptimize all
        # TODO: Problem current `bounds` handling assumes sorted knots
        mod_new = self.model_class(endog, exog)
        seg = Segmented(mod_new, self.exog_source, target_indices, copy_data=False)


        import copy
        bounds = list(copy.copy(self.bounds))
        # try each inner segment
        res = []
        bounds_all = []
        for k in range(1, len(bounds)):
            low, upp = bounds[k : k+2]
            #bounds = sorted(list(bounds) + [(low + upp) / 2])
            bounds = list(bounds) + [(low + upp) / 2]
            objvalue = seg._fit_all(bounds, maxiter=maxiter)
            res.append(objvalue)
            bounds_all.append(bounds)

        # get best
        bidx = np.argmin(res)
        seg._fit_all(bounds_all[bidx], maxiter=maxiter)
        return seg, (res, bounds_all)

    def segmented(model, idx, k_segments=1):
        """segmented regression with a single knot or break point

        """

        exog_k = np.asarray(model.exog[:, idx], order='F')
        exog = model.exog.copy(order='F')

        # TODO: generalize, option for objective, callback
        def ssr(a):
            exog[:, -1] = np.clip(exog_k - a, 0, np.inf)
            ssr_ = OLS(model.endog, exog).fit().ssr
            return ssr_

        brack = np.percentile(exog_k, [70, 85])
        print('brack', brack)
        res_optim = optimize.brent(ssr, brack=brack)
        # TODO check convergence

        # redo, don't rely on having the correct last estimate in the loop:
        exog[:, -1] = np.clip(exog_k - res_optim, 0, np.inf)
        res_best = OLS(model.endog, exog).fit()
        # attach extra result
        res_best.knot_location = res_optim
        return res_best

