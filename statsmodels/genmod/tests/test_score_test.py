# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:39:15 2018

Author: Josef Perktold
"""


import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import statsmodels.stats._diagnostic_other as diao
from statsmodels.base._parameter_inference import score_test


class TestScoreTest(object):
    # compares score to wald, and regression test for pvalues
    rtol_ws = 5e-3
    atol_ws = 0
    dispersed = False  # Poisson correctly specified
    # regression numbers
    res_pvalue = [0.31786373532550893, 0.32654081685271297]

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 500, 5

        np.random.seed(786452)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        x2 = np.random.randn(nobs, 2)
        xx = np.column_stack((x, x2))

        if cls.dispersed:
            het = np.random.randn(nobs)
            y = np.random.poisson(np.exp(x.sum(1) * 0.5 + het))
            #y_mc = np.random.negative_binomial(np.exp(x.sum(1) * 0.5), 2)
        else:
            y = np.random.poisson(np.exp(x.sum(1) * 0.5))

        cls.exog_extra = x2
        cls.model_full = GLM(y, xx, family=families.Poisson())
        cls.model_drop = GLM(y, x, family=families.Poisson())

    def test_wald_score(self):
        mod_full = self.model_full
        mod_drop = self.model_drop
        restriction = 'x5=0, x6=0'
        res_full = mod_full.fit()
        res_constr = mod_full.fit_constrained('x5=0, x6=0')
        res_drop = mod_drop.fit()

        wald = res_full.wald_test(restriction)
        lm_constr = np.hstack(score_test(res_constr))
        lm_extra = np.hstack(score_test(res_drop, exog_extra=self.exog_extra))

        res_wald = np.hstack([wald.statistic.squeeze(), wald.pvalue, [wald.df_denom]])
        assert_allclose(lm_constr, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_extra, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_constr, lm_extra, rtol=1e-12, atol=1e-14)
        # regression number
        assert_allclose(lm_constr[1], self.res_pvalue[0], rtol=1e-12, atol=1e-14)

        cov_type='HC0'
        res_full_hc = mod_full.fit(cov_type=cov_type, start_params=res_full.params)
        wald = res_full_hc.wald_test(restriction)
        lm_constr = np.hstack(score_test(res_constr, cov_type=cov_type))
        lm_extra = np.hstack(score_test(res_drop, exog_extra=self.exog_extra,
                                        cov_type=cov_type))

        res_wald = np.hstack([wald.statistic.squeeze(), wald.pvalue, [wald.df_denom]])
        assert_allclose(lm_constr, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_extra, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_constr, lm_extra, rtol=1e-13)
        # regression number
        assert_allclose(lm_constr[1], self.res_pvalue[1], rtol=1e-12, atol=1e-14)


class TestScoreTestDispersed(TestScoreTest):
    rtol_ws = 0.11
    atol_ws = 0.015
    dispersed = True  # Poisson is mis-specified
    res_pvalue = [5.412978775609189e-14, 0.05027602575743518]
