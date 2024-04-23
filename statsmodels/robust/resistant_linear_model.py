"""
Created on Apr. 19, 2024 12:17:03 p.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from statsmodels.base.model import Model
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.testing import Holder

from statsmodels.robust.robust_linear_model import RLM
import statsmodels.robust.norms as rnorms
import statsmodels.robust.scale as rscale
from statsmodels.robust.covariance import  _get_detcov_startidx


class RLMDetS(Model):
    """S-estimator for linear model with deterministic starts.
    """

    def __init__(self, endog, exog, norm=None, breakdown_point=0.5,
                 col_indices=None, include_endog=False):
        super().__init__(endog, exog)

        if norm is None:
            norm = rnorms.TukeyBiweight(c=1.547)
            scale_bias = 0.1995
            self.mscale = rscale.MScale(norm, scale_bias)
        else:
            raise ValueError()

        self.norm = norm
        # need tuning params for given breakdown point
        # if type(norm) is a class then create an instance
        # if norm is an instance, then just use it
        # self.norm._set_tuning_param(???)
        # data for robust mahalanobis distance of starting sets
        self.breakdown_point = breakdown_point

        if col_indices is None:
            exog_start = self.exog[:, 1:]
        else:
            exog_start = self.exog[:, col_indices]

        if include_endog:
            self.data_start = np.column_stack((endog, exog_start))
        else:
            self.data_start = exog_start

    def _get_start_params(self, h):
        # I think we should use iterator with yield
        starts = _get_detcov_startidx(
            self.data_start, h, options_start=None, methods_cov="all")

        start_params_all = [
            OLS(self.endog[idx], self.exog[idx]).fit().params
            for (idx, method) in starts
            ]
        return start_params_all

    def _fit_once(self, start_params, maxiter=100):
        mod = RLM(self.endog, self.exog, M=self.norm)
        res = mod.fit(start_params=start_params,
                      scale_est=self.mscale,
                      maxiter=maxiter)
        return res

    def fit(self, h, maxiter=100, maxiter_step=5):

        res = {}
        for ii, sp in enumerate(self._get_start_params(h)):
            res_ii = self._fit_once(sp, maxiter=maxiter_step)
            res[ii] = Holder(
                scale=res_ii.scale,
                params=res_ii.params,
                method=ii,  # method  # TODO need start set method
                )

        scale_all = np.array([i.scale for i in res.values()])
        scale_sorted = np.argsort(scale_all)
        best_idx = scale_sorted[0]

        # TODO: iterate until convergence if start fits are not converged
        res_best = self._fit_once(res[best_idx].params, maxiter=maxiter)

        # TODO: add extra start and convergence info
        res_best._results.results_iter = res
        # results instance of _fit_once has RLM as `model`
        res_best.model_dets = self
        return res_best


class RLMDetSMM(RLMDetS):
    """MM-estimator with S-estimator starting values

    """

    def fit(self, h=None, binding=False):
        norm_m = rnorms.TukeyBiweight(c=4.685061)
        res_s = super().fit(h)
        mod_m = RLM(res_s.model.endog, res_s.model.exog, M=norm_m)
        res_mm = mod_m.fit(
            start_params=np.asarray(res_s.params),
            start_scale=res_s.scale,
            update_scale=False
            )

        if not binding:
            # we can compute this first and skip MM if scale decrease
            mod_sm = RLM(res_s.model.endog, res_s.model.exog, M=norm_m)
            res_sm = mod_sm.fit(
                start_params=res_s.params,
                scale_est=self.mscale
                )

        if not binding and res_sm.scale < res_mm.scale:
            return res_sm
        else:
            return res_mm
