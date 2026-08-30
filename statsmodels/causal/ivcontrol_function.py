# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:06:01 2023

Author: Josef Perktold
License: BSD-3
"""

import numpy as np

from statsmodels.base.model import LikelihoodModel
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.genmod.generalized_linear_model import GLM


def momcond(params, modelcf=None):
    mod_first = modelcf.model_first

    y2 = modelcf.endog
    x2 = modelcf.explan

    params1 = params[:2]
    params2 = params[2:]
    mom1 = mod_first.score_obs(params1, scale=1)
    # we need resid, control function as function of params
    controlf = modelcf._control_function(
        None,
        params=params1,
        model_first=mod_first,
        )

    mod2 = GLM(y2, np.column_stack((x2, controlf)), family=modelcf.family)
    mom2 = mod2.score_obs(params2)

    return np.column_stack([mom1, mom2])


class GMMIVCF(GMM):
    """GMM for model with with 2-stage control function approach
    """
    def momcond(self, params):
        return momcond(params, modelcf=self.model)


class GLMIVCF(LikelihoodModel):
    """GLM with endogenous regressor with 2-stage control function approach
    """

    def __init__(self, endog, explan, endog_explan, instruments,
                 family=None,
                 family_first=None,
                 cf_kwds=None,
                 ):
        super().__init__(
            endog,
            explan=explan,
            endog_explan=endog_explan,
            instruments=instruments,
            )

        self.family = family
        self.family_first = family_first
        cf_kwds_ = {
            "resid": "resid_generalized",
            "interaction": None,
            "callback": None,
            }
        cf_kwds_.update(cf_kwds if cf_kwds is not None else {})
        self.cf_kwds = cf_kwds_

        y1 = self.endog_explan
        x1 = self.instruments
        self.model_first = GLM(y1, x1, family=family_first)

    def _control_function(self, results_first, params=None, model_first=None):
        if results_first is not None:
            res = results_first
            mod = results_first.model
        else:
            if params is None:
                raise ValueError("params cannot be None if results_first is")
            mod = model_first
            res = None

        cf_kwds = self.cf_kwds

        if params is None:
            # we can use results attributes
            if cf_kwds["resid"] == "resid_generalized":
                resid = mod.score_factor(res.params, scale=1)
            elif cf_kwds["resid"] == "resid_response":
                resid = res.resid_response

        else:
            # need resid as function of params for derivatives (numdiff)
            if cf_kwds["resid"] == "resid_generalized":
                resid = mod.score_factor(params, scale=1)
            elif cf_kwds["resid"] == "resid_response":
                resid = mod.endog - mod.predict(params)

        if cf_kwds["interaction"] is not None:
            ex = cf_kwds["interaction"]
            if ex.ndim < 2:
                ex = ex[:, None]
            cf = np.column_stack((resid, resid[:, None] * ex))
        elif cf_kwds["callback"] is not None:
            cf = cf_kwds["callback"](resid)
        else:
            cf = resid

        return cf

    def fit(self):
        y2 = self.endog
        x2 = self.explan

        res_first = self.model_first.fit()
        controlf = self._control_function(res_first)
        res_outcome = GLM(y2,
                          np.column_stack((x2, controlf)),
                          family=self.family
                          ).fit()

        params = np.concatenate([res_first.params, res_outcome.params])
        mod_gmm = GMMIVCF(y2, x2, None, k_moms=len(params),
                          param_names=[f"s{i}" for i in range(len(params))])
        mod_gmm.model = self
        res_gmm = mod_gmm.fit(start_params=params, maxiter=0)
        res_out = res_outcome._results  # res_outcome is wrapped results
        res_out._cache = {}
        k_out = len(res_outcome.params)
        cov2 = res_gmm.cov_params()[-k_out:, -k_out:]
        res_out.cov_params_default = cov2
        res_out.normalized_cov_params = None

        res_gmm.results_first = res_first
        res_gmm.results_outcome = res_outcome

        return res_gmm
