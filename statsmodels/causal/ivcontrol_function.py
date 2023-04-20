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
    resid = mod_first.endog - mod_first.predict(params1)
    mod2 = GLM(y2, np.column_stack((x2, resid)), family=modelcf.family)
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
                 ):
        super().__init__(
            endog,
            explan=explan,
            endog_explan=endog_explan,
            instruments=instruments,
            )

        self.family = family
        self.family_first = family_first

        y1 = self.endog_explan
        x1 = self.instruments
        self.model_first = GLM(y1, x1, family=family_first)

    def fit(self):
        y2 = self.endog
        x2 = self.explan

        res_first = self.model_first.fit()
        res_outcome = GLM(y2,
                          np.column_stack((x2, res_first.resid_response)),
                          family=self.family
                          ).fit()

        params = np.concatenate([res_first.params, res_outcome.params])
        mod_gmm = GMMIVCF(y2, x2, None, k_moms=len(params),
                          param_names=[f"s{i}" for i in range(len(params))])
        mod_gmm.model = self
        res_gmm = mod_gmm.fit(start_params=params, maxiter=0)
        return res_gmm
