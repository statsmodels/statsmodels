# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:39:15 2018

Author: Josef Perktold
"""


import numpy as np
from numpy.testing import assert_allclose

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.stats._diagnostic_other as diao
from statsmodels.base._parameter_inference import score_test


class TestScoreTest(object):
    # compares score to wald, and regression test for pvalues
    rtol_ws = 5e-3
    atol_ws = 0
    rtol_wooldridge = 0.004
    dispersed = False  # Poisson correctly specified
    # regression numbers
    res_pvalue = [0.31786373532550893, 0.32654081685271297]
    skip_wooldridge = False
    res_disptest = np.array([[ 0.13927919,  0.44461477],
                             [ 0.13927919,  0.44461477],
                             [ 0.21295545,  0.41568086],
                             [ 0.14935018,  0.88127732],
                             [ 0.14935018,  0.88127732],
                             [ 0.14841761,  0.8820132 ],
                             [ 0.22836064,  0.81936588]])
    res_disptest_g = [0.052247629593715761, 0.81919738867722225]

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

        wald = res_full.wald_test(restriction, scalar=True)
        lm_constr = np.hstack(score_test(res_constr))
        lm_extra = np.hstack(score_test(res_drop, exog_extra=self.exog_extra))

        res_wald = np.hstack([wald.statistic, wald.pvalue, [wald.df_denom]])
        assert_allclose(lm_constr, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_extra, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_constr, lm_extra, rtol=1e-12, atol=1e-14)
        # regression number
        assert_allclose(lm_constr[1], self.res_pvalue[0], rtol=1e-12, atol=1e-14)

        cov_type='HC0'
        res_full_hc = mod_full.fit(cov_type=cov_type, start_params=res_full.params)
        wald = res_full_hc.wald_test(restriction, scalar=True)
        lm_constr = np.hstack(score_test(res_constr, cov_type=cov_type))
        lm_extra = np.hstack(score_test(res_drop, exog_extra=self.exog_extra,
                                        cov_type=cov_type))

        res_wald = np.hstack([wald.statistic, wald.pvalue, [wald.df_denom]])
        assert_allclose(lm_constr, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_extra, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_constr, lm_extra, rtol=1e-13)
        # regression number
        assert_allclose(lm_constr[1], self.res_pvalue[1], rtol=1e-12, atol=1e-14)

        if not self.skip_wooldridge:
            # compare with Wooldridge auxiliary regression
            # does not work for Poisson, even with family attribute
            # diao.lm_test_glm assumes fittedvalues is mean (not linear pred)
            lm_wooldridge = diao.lm_test_glm(res_drop, self.exog_extra)
            assert_allclose(lm_wooldridge.pval1, self.res_pvalue[0],
                            rtol=1e-12, atol=1e-14)
            assert_allclose(lm_wooldridge.pval3, self.res_pvalue[1],
                            rtol=self.rtol_wooldridge)
            # smoke test
            lm_wooldridge.summary()

    def test_dispersion(self):
        res_drop = self.model_drop.fit()
        res_test = diao.dispersion_poisson(res_drop)
        assert_allclose(res_test[0], self.res_disptest, rtol=1e-6, atol=1e-14)
        # constant only dispersion
        ex = np.ones((res_drop.model.endog.shape[0], 1))
        # ex = np.column_stack((np.ones(res_drop.model.endog.shape[0]),
        #                      res_drop.predict()))  # or **2
        # dispersion_poisson_generic might not be correct
        # or not clear what the alternative hypothesis is
        # choosing different `ex` implies different alternative hypotheses
        res_test = diao.dispersion_poisson_generic(res_drop, ex)
        assert_allclose(res_test, self.res_disptest_g, rtol=1e-6, atol=1e-14)


class TestScoreTestDispersed(TestScoreTest):
    rtol_ws = 0.11
    atol_ws = 0.015
    rtol_wooldridge = 0.03
    dispersed = True  # Poisson is mis-specified
    res_pvalue = [5.412978775609189e-14, 0.05027602575743518]
    res_disptest = np.array([[  1.26473634e+02,   0.00000000e+00],
                             [  1.26473634e+02,   0.00000000e+00],
                             [  1.19393621e+02,   0.00000000e+00],
                             [  4.53940519e+00,   5.64131397e-06],
                             [  4.53940519e+00,   5.64131397e-06],
                             [  2.98154154e+00,   2.86801135e-03],
                             [  4.27569194e+00,   1.90544551e-05]])
    res_disptest_g = [17.670784788586968, 2.6262956791721383e-05]


class TestScoreTestPoisson(TestScoreTest):
    # compares score to wald, and regression test for pvalues
    rtol_ws = 5e-3
    atol_ws = 0
    rtol_wooldridge = 0.004
    dispersed = False  # Poisson correctly specified
    # regression numbers
    res_pvalue = [0.31786373532550893, 0.32654081685271297]
    skip_wooldridge = False
    res_disptest = np.array([[ 0.13927919,  0.44461477],
                             [ 0.13927919,  0.44461477],
                             [ 0.21295545,  0.41568086],
                             [ 0.14935018,  0.88127732],
                             [ 0.14935018,  0.88127732],
                             [ 0.14841761,  0.8820132 ],
                             [ 0.22836064,  0.81936588]])
    res_disptest_g = [0.052247629593715761, 0.81919738867722225]

    @classmethod
    def setup_class(cls):
        # copy-paste except for model
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
        cls.model_full = Poisson(y, xx)
        cls.model_drop = Poisson(y, x)

    def test_wald_score(self):
        super(TestScoreTestPoisson, self).test_wald_score()


class TestScoreTestPoissonDispersed(TestScoreTestPoisson):
    rtol_ws = 0.11
    atol_ws = 0.015
    rtol_wooldridge = 0.03
    dispersed = True  # Poisson is mis-specified
    res_pvalue = [5.412978775609189e-14, 0.05027602575743518]
    res_disptest = np.array([[  1.26473634e+02,   0.00000000e+00],
                             [  1.26473634e+02,   0.00000000e+00],
                             [  1.19393621e+02,   0.00000000e+00],
                             [  4.53940519e+00,   5.64131397e-06],
                             [  4.53940519e+00,   5.64131397e-06],
                             [  2.98154154e+00,   2.86801135e-03],
                             [  4.27569194e+00,   1.90544551e-05]])
    res_disptest_g = [17.670784788586968, 2.6262956791721383e-05]
