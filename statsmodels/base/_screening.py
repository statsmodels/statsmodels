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

    Parameters
    ----------
    model : instance of penalizing model
        examples: GLMPenalized, PoissonPenalized and LogitPenalized
    pen_weight : None or float
        penalization weight use in SCAD penalized MLE
    k_add : int
        number of exog to add
    k_max_add : int
        maximum number of variables to add during variable addition, i.e.
        forward selection. default is 30
    threshold_trim : float
        threshold for trimming parameters to zero, default is 1e-4
    k_max_included : int
        maximum total number of variables to include in model.
    ranking_attr : str
        This determines the result attribute or model method that is used for
        the ranking of exog to include. The availability of attributes depends
        on the model.
        Default is 'resid_pearson', 'score_factor' can be used in GLM.
    ranking_project : bool
        If ranking_project is True, then the exog candidates for inclusion are
        first projected on the already included exog before the computation
        of the ranking measure. This brings the ranking measure closer to
        the statistic of a score test for variable addition.

    Notes
    -----
    Status: experimental, tested only on a limited set of models and
    with a limited set of model options.

    TODOs and current limitations:

    penalization weights need to be used to avoid penalizing always kept
    exog. The length of weights vector depends on the number of variables
    included in the candidate model.
    This is not included yet. We SCAD penalize all parameters, but large
    parameters are not penalized by the SCAD penalty function.

    pearson_resid: GLM resid_pearson does not include freq_weights.

    freq_weights are not supported in this. Candidate ranking uses
    moment condition with resid_pearson or others without freq_weights.

    fit_kwds are missing, e.g. we need to avoid irls in GLM, this is done
    by using start_params and a gradient fit_method.

    variable names: do we keep track of those?

    currently only supports numpy arrays, no exog type check or conversion

    currently only single columns are selected, no terms (multi column exog)

    """

    def __init__(self, model, pen_weight=None, k_add=5, k_max_add=30,
                 threshold_trim=1e-4, k_max_included=20,
                 ranking_attr='resid_pearson', ranking_project=True):

        self.model = model
        self.model_class = model.__class__
        self.init_kwds = model._get_init_kwds()
        # pen_weight and penal are explicitly included
        # TODO: check what we want to do here
        self.init_kwds.pop('pen_weight', None)
        self.init_kwds.pop('penal', None)

        self.endog = model.endog
        self.exog_keep = model.exog
        self.k_keep = model.exog.shape[1]
        self.nobs = len(self.endog)
        self.penal = SCADSmoothed(0.1, c0=0.0001)

        if pen_weight is not None:
            self.pen_weight = pen_weight
        else:
            self.pen_weight = self.nobs * 10

        # option for screening algorithm
        self.k_add = k_add
        self.k_max_add = k_max_add
        self.threshold_trim = threshold_trim
        self.k_max_included = k_max_included
        self.ranking_attr = ranking_attr
        self.ranking_project = ranking_project

    def ranking_measure(self, res_pen, exog, keep=None):
        """compute measure for ranking exog candidates for inclusion

        """
        endog = self.endog

        if self.ranking_project:
            ex_incl = res_pen.model.exog[:, keep]
            exog = exog - ex_incl.dot(np.linalg.pinv(ex_incl).dot(exog))

        if self.ranking_attr == 'predicted_poisson':
            # I keep this for more experiments

            # TODO: does it really help to change/trim params
            # we are not reestimating with trimmed model
            p = res_pen.params.copy()
            if keep is not None:
                p[~keep] = 0
            predicted = res_pen.model.predict(p)
            # this is currently hardcoded for Poisson
            resid_factor = (endog - predicted) / np.sqrt(predicted)
        elif self.ranking_attr[:6] == 'model.':
            # use model method, this is intended for score_factor
            attr = self.ranking_attr.split('.')[1]
            resid_factor = getattr(res_pen.model, attr)(res_pen.params)
            if resid_factor.ndim == 2:
                # for score_factor when extra params are in model
                resid_factor = resid_factor[:, 0]
            mom_cond = np.abs(resid_factor.dot(exog))**2
        else:
            # use results attribute
            resid_factor = getattr(res_pen, self.ranking_attr)
            mom_cond = np.abs(resid_factor.dot(exog))**2
        return mom_cond

    def screen_exog(self, exog, endog=None, maxiter=5, method='bfgs',
                    disp=False):
        """screen and select variables (columns) in exog

        Parameters
        ----------
        exog : ndarray
            candidate explanatory variables that are screened for inclusion in
            the model
        endog : ndarray (optional)
            use a new endog in the screening model.
            This is not tested yet, and might not work correctly
        maxiter : int
            number of screening iterations
        method : str
            optimization method to use in fit, needs to be only of the gradient
            optimizers
        disp : bool
            display option for fit during optimization

        Returns
        -------
        res_screen : instance of ScreeningResults
            The attribute `results_final` contains is the results instance
            with the final model selection.
            `idx_nonzero` contains the index of the selected exog in the full
            exog, combined exog that are always kept plust exog_candidates.


        """
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
        converged = False
        idx_old = []
        for it in range(maxiter):
            mom_cond = self.ranking_measure(res_pen, x1, keep=keep)
            mcs = np.sort(mom_cond)[::-1]

            threshold = mcs[max((self.k_max_add, k0 + self.k_add))]
            idx = np.concatenate((idx_nonzero, idx_excl[mom_cond > threshold]))
            start_params2 = np.zeros(len(idx))
            start_params2[:len(start_params)] = start_params

            res_pen = model_class(endog, x[:, idx], penal=self.penal,
                                       pen_weight=self.pen_weight,
                                       **self.init_kwds).fit(method=method,
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
            history['idx_added'].append(idx)

            if (len(idx_nonzero) == len(idx_old) and
                    (idx_nonzero == idx_old).all()):
                converged = True
                break
            idx_old = idx_nonzero


        # final esimate
        res_final = model_class(endog, x[:, idx_nonzero], penal=self.penal,
                                pen_weight=self.pen_weight,
                                **self.init_kwds).fit(method=method,
                                          start_params=start_params,
                                          warn_convergence=False, disp=disp)

        res = ScreeningResults(self,
                               results_pen = res_pen,
                               results_final = res_final,
                               idx_nonzero = idx_nonzero,
                               idx_excl = idx_excl,
                               start_params = start_params,
                               history = history,
                               converged = converged,
                               iterations = it)
        return res

    def screen_exog_iterator(self, exog_iterator):
        """
        batched version of screen exog

        This screens variables in a two step process:

        In the first step screen_exog is used on each element of the
        exog_iterator, and the batch winners are collected.

        In the second step all batch winners are combined into a new array
        of exog candidates and `screen_exog` is used to select a final
        model.

        Parameters
        ----------
        exog_iterator : iterator over ndarrays

        Returns
        -------
        res_screen_final : instance of ScreeningResults
            This is the instance returned by the second round call to
            `screen_exog`. Additional attributes are added to provide
            more information about the batched selection process.
            The index of final nonzero variables is
            `idx_nonzero_batches` which is a 2-dimensional array with batch
            index in the first column and variable index withing batch in the
            second column. They can be used jointly as index for the data
            in the exog_iterator.

        """
        # res_batches = []
        res_idx = []
        exog_winner = []
        exog_idx = []
        for ex in exog_iterator:
            res_screen = self.screen_exog(ex, maxiter=20)
            # avoid storing res_screen, only for debugging
            # res_batches.append(res_screen)
            res_idx.append(res_screen.idx_nonzero)
            exog_winner.append(ex[:, res_screen.idx_nonzero[1:] - self.k_keep])
            exog_idx.append(res_screen.idx_nonzero[1:] - self.k_keep)

        exog_winner = np.column_stack(exog_winner)
        res_screen_final = self.screen_exog(exog_winner, maxiter=20)


        exog_winner_names = ['var%d_%d' % (bidx, idx)
                             for bidx, batch in enumerate(exog_idx)
                             for idx in batch]

        idx_full = [(bidx, idx)
                    for bidx, batch in enumerate(exog_idx)
                    for idx in batch]
        ex_final_idx = res_screen_final.idx_nonzero[1:] - self.k_keep
        final_names = np.array(exog_winner_names)[ex_final_idx]
        res_screen_final.idx_nonzero_batches = np.array(idx_full)[ex_final_idx]
        res_screen_final.exog_final_names = final_names
        history = {'res_idx': res_idx,
                   'exog_idx': exog_idx}
        res_screen_final.history = history
        return res_screen_final
