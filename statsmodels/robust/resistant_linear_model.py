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


    Notes
    -----
    This estimator combines the method of Fast-S regression (Saliban-Barrera
    et al 2006) using starting sets similar to the deterministic estimation of
    multivariate location and scatter DetS and DetMM of Hubert et al (2012).


    References
    ----------

    .. [1] Hubert, Mia, Peter J. Rousseeuw, and Tim Verdonck. 2012. “A
       Deterministic Algorithm for Robust Location and Scatter.” Journal of
       Computational and Graphical Statistics 21 (3): 618–37.
       https://doi.org/10.1080/10618600.2012.672100.

    .. [2] Hubert, Mia, Peter Rousseeuw, Dina Vanpaemel, and Tim Verdonck.
       2015. “The DetS and DetMM Estimators for Multivariate Location and
       Scatter.” Computational Statistics & Data Analysis 81 (January): 64–75.
       https://doi.org/10.1016/j.csda.2014.07.013.

    .. [3] Rousseeuw, Peter J., Stefan Van Aelst, Katrien Van Driessen, and
        Jose Agulló. 2004. “Robust Multivariate Regression.”
        Technometrics 46 (3): 293–305.

    .. [4] Salibian-Barrera, Matías, and Víctor J. Yohai. 2006. “A Fast
       Algorithm for S-Regression Estimates.” Journal of Computational and
       Graphical Statistics 15 (2): 414–27.


    """

    def __init__(self, endog, exog, norm=None, breakdown_point=0.5,
                 col_indices=None, include_endog=False):
        super().__init__(endog, exog)

        if norm is None:
            norm = rnorms.TukeyBiweight()

        tune = norm.get_tuning(bp=breakdown_point)
        c = tune[0]
        scale_bias = tune[2]
        norm = norm._set_tuning_param(c, inplace=False)
        self.mscale = rscale.MScale(norm, scale_bias)

        self.norm = norm
        self.breakdown_point = breakdown_point

        # TODO: detect constant
        if col_indices is None:
            exog_start = self.exog[:, 1:]
        else:
            exog_start = self.exog[:, col_indices]

        # data for robust mahalanobis distance of starting sets
        if include_endog:
            self.data_start = np.column_stack((endog, exog_start))
        else:
            self.data_start = exog_start

    def _get_start_params(self, h):
        # I think we should use iterator with yield

        if self.data_start.shape[1] == 0 and self.exog.shape[1] == 1:
            quantiles = np.quantile(self.endog, [0.25, 0.5, 0.75])
            start_params_all = [np.atleast_1d([q]) for q in quantiles]
            return start_params_all


        starts = _get_detcov_startidx(
            self.data_start, h, options_start=None, methods_cov="all")

        start_params_all = [
            OLS(self.endog[idx], self.exog[idx]).fit().params
            for (idx, method) in starts
            ]
        return start_params_all

    def _fit_one(self, start_params, maxiter=100):
        mod = RLM(self.endog, self.exog, M=self.norm)
        res = mod.fit(start_params=start_params,
                      scale_est=self.mscale,
                      maxiter=maxiter)
        return res

    def fit(self, h, maxiter=100, maxiter_step=5, start_params_extra=None):

        start_params_all = self._get_start_params(h)
        if start_params_extra:
            start_params_all.extend(start_params_extra)
        res = {}
        for ii, sp in enumerate(start_params_all):
            res_ii = self._fit_one(sp, maxiter=maxiter_step)
            res[ii] = Holder(
                scale=res_ii.scale,
                params=res_ii.params,
                method=ii,  # method  # TODO need start set method
                )

        scale_all = np.array([i.scale for i in res.values()])
        scale_sorted = np.argsort(scale_all)
        best_idx = scale_sorted[0]

        # TODO: iterate until convergence if start fits are not converged
        res_best = self._fit_one(res[best_idx].params, maxiter=maxiter)

        # TODO: add extra start and convergence info
        res_best._results.results_iter = res
        # results instance of _fit_once has RLM as `model`
        res_best.model_dets = self
        return res_best


class RLMDetSMM(RLMDetS):
    """MM-estimator with S-estimator starting values.

    Parameters
    ----------
    endog : array-like, 1-dim
        Dependent, endogenous variable.
    exog array-like, 1-dim
        Inependent, exogenous regressor variables.
    norm : robust norm
        Redescending robust norm used for S- and MM-estimation.
        Default is TukeyBiweight.
    efficiency : float in (0, 1)
        Asymptotic efficiency of the MM-estimator (used in second stage).
    breakdown_point : float in (0, 0.5)
        Breakdown point of the preliminary S-estimator.
    col_indices : None or array-like of ints
        Index of columns of exog to use in the mahalanobis distance computation
        for the starting sets of the S-estimator.
        Default is all exog except first column (constant). Todo: will change
        when we autodetect the constant column
    include_endog : bool
        If true, then the endog variable is combined with the exog variables
        to compute the the mahalanobis distances for the starting sets of the
        S-estimator.

    """
    def __init__(self, endog, exog, norm=None, efficiency=0.95,
                 breakdown_point=0.5, col_indices=None, include_endog=False):
        super().__init__(
            endog,
            exog,
            norm=norm,
            breakdown_point=breakdown_point,
            col_indices=col_indices,
            include_endog=include_endog
            )

        self.efficiency = efficiency
        if norm is None:
            norm = rnorms.TukeyBiweight()

        c = norm.get_tuning(eff=efficiency)[0]
        norm = norm._set_tuning_param(c, inplace=False)
        self.norm_mean = norm

    def fit(self, h=None, scale_binding=False, start=None):
        """Estimate the model

        Parameters
        ----------
        h : int
            The size of the initial sets for the S-estimator.
            Default is ....  (todo)
        scale_binding : bool
            If true, then the scale is fixed in the second stage M-estimation,
            i.e. this is the MM-estimator.
            If false, then the high breakdown point M-scale is used also in the
            second stage M-estimation if that estimated scale is smaller than
            the scale of the preliminary, first stage S-estimato.
        start : tuple or None
            If None, then the starting parameters and scale for the second
            stage M-estimation are taken from the fist stage S-estimator.
            Alternatively, the starting parameters and starting scale can be
            provided by the user as tuple (start_params, start_scale). In this
            case the first stage S-estimation in skipped.
        maxiter, other optimization parameters are still missing (todo)

        Returns
        -------
        results instance

        Notes
        -----
        If scale_binding is false, then the estimator is a standard
        MM-estimator with fixed scale in the second stage M-estimation.
        If scale_binding is true, then the estimator will try to find an
        estimate with lower M-scale using the same scale-norm rho as in the
        first stage S-estimator. If the estimated scale, is not smaller than
        then the scale estimated in the first stage S-estimator, then the
        fixed scale MM-estimator is returned.


        """
        norm_m = self.norm_mean
        if start is None:
            res_s = super().fit(h)
            start_params = np.asarray(res_s.params)
            start_scale = res_s.scale
        else:
            start_params, start_scale = start
            res_s = None

        mod_m = RLM(self.endog, self.exog, M=norm_m)
        res_mm = mod_m.fit(
            start_params=start_params,
            start_scale=start_scale,
            update_scale=False
            )

        if not scale_binding:
            # we can compute this first and skip MM if scale decrease
            mod_sm = RLM(self.endog, self.exog, M=norm_m)
            res_sm = mod_sm.fit(
                start_params=start_params,
                scale_est=self.mscale
                )

        if not scale_binding and res_sm.scale < res_mm.scale:
            res = res_sm
        else:
            res = res_mm

        res._results.results_dets = res_s
        return res
