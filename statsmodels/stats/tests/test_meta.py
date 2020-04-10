# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:08:37 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
import pandas as pd

from numpy.testing import assert_equal, assert_allclose

from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM

from statsmodels.stats.meta_analysis import (
    effectsize_smd, combine_effects, _fit_tau_iterative,
    _fit_tau_mm, _fit_tau_iter_mm)

from .results import results_meta


class TestMetaK1(object):

    @classmethod
    def setup_class(cls):

        cls.eff = np.array([61.00, 61.40, 62.21, 62.30, 62.34, 62.60, 62.70,
                            62.84, 65.90])
        cls.var_eff = np.array([0.2025, 1.2100, 0.0900, 0.2025, 0.3844, 0.5625,
                                0.0676, 0.0225, 1.8225])

    def test_tau_kacker(self):
        # test iterative and two-step methods, Kacker 2004
        # PM CA DL C2 from table 1 first row p. 135
        # test for PM and DL are also against R metafor in other tests
        eff, var_eff = self.eff, self.var_eff
        t_PM, t_CA, t_DL, t_C2 = [0.8399, 1.1837, 0.5359, 0.9352]

        tau2, converged = _fit_tau_iterative(eff, var_eff,
                                             tau2_start=0.1, atol=1e-8)
        assert_equal(converged, True)
        assert_allclose(np.sqrt(tau2), t_PM, atol=6e-5)

        k = len(eff)
        # cochrane uniform weights
        tau2_ca = _fit_tau_mm(eff, var_eff, np.ones(k) / k)
        assert_allclose(np.sqrt(tau2_ca), t_CA, atol=6e-5)

        # DL one step, and 1 iteration, reduced agreement with Kacker
        tau2_dl = _fit_tau_mm(eff, var_eff, 1 / var_eff)
        assert_allclose(np.sqrt(tau2_dl), t_DL, atol=1e-3)

        tau2_dl_, converged = _fit_tau_iter_mm(eff, var_eff, tau2_start=0,
                                               maxiter=1)
        assert_equal(converged, False)
        assert_allclose(tau2_dl_, tau2_dl, atol=1e-10)

        # C2 two step, start with CA
        tau2_c2, converged = _fit_tau_iter_mm(eff, var_eff,
                                              tau2_start=tau2_ca,
                                              maxiter=1)
        assert_equal(converged, False)
        assert_allclose(np.sqrt(tau2_c2), t_C2, atol=6e-5)

    def test_pm(self):
        res = results_meta.exk1_metafor
        eff, var_eff = self.eff, self.var_eff

        tau2, converged = _fit_tau_iterative(eff, var_eff,
                                             tau2_start=0.1, atol=1e-8)
        assert_equal(converged, True)
        assert_allclose(tau2, res.tau2, atol=1e-10)

        # compare with WLS, PM weights
        mod_wls = WLS(eff, np.ones(len(eff)), weights=1 / (var_eff + tau2))
        res_wls = mod_wls.fit(cov_type="fixed_scale")

        assert_allclose(res_wls.params, res.b, atol=1e-13)
        assert_allclose(res_wls.bse, res.se, atol=1e-10)
        ci_low, ci_upp = res_wls.conf_int()[0]
        assert_allclose(ci_low, res.ci_lb, atol=1e-10)
        assert_allclose(ci_upp, res.ci_ub, atol=1e-10)

        # need stricter atol to match metafor,
        # I also used higher precision in metafor
        res3 = combine_effects(eff, var_eff, method_re="pm", atol=1e-7)
        # TODO: asserts below are copy paste, DRY?
        assert_allclose(res3.tau2, res.tau2, atol=1e-10)
        assert_allclose(res3.mean_effect_re, res.b, atol=1e-13)
        assert_allclose(res3.sd_eff_w_re, res.se, atol=1e-10)

        ci = res3.conf_int(use_t=False)[1]
        assert_allclose(ci[0], res.ci_lb, atol=1e-10)
        assert_allclose(ci[1], res.ci_ub, atol=1e-10)

        assert_allclose(res3.q, res.QE, atol=1e-10)
        # the following doesn't pass yet
        # assert_allclose(res3.i2, res.I2 / 100, atol=1e-10)  # percent in R
        # assert_allclose(res3.h2, res.H2, atol=1e-10)
        q, pv, df = res3.test_homogeneity()
        assert_allclose(pv, res.QEp, atol=1e-10)
        assert_allclose(q, res.QE, atol=1e-10)
        assert_allclose(df, 9 - 1, atol=1e-10)



    def test_dl(self):
        res = results_meta.exk1_dl
        eff, var_eff = self.eff, self.var_eff

        tau2 = _fit_tau_mm(eff, var_eff, 1 / var_eff)
        assert_allclose(tau2, res.tau2, atol=1e-10)

        res3 = combine_effects(eff, var_eff, method_re="dl")
        assert_allclose(res3.tau2, res.tau2, atol=1e-10)
        assert_allclose(res3.mean_effect_re, res.b, atol=1e-13)
        assert_allclose(res3.sd_eff_w_re, res.se, atol=1e-10)
        ci = res3.conf_int(use_t=False)  # fe, re, fe_wls, re_wls
        assert_allclose(ci[1][0], res.ci_lb, atol=1e-10)
        assert_allclose(ci[1][1], res.ci_ub, atol=1e-10)

        assert_allclose(res3.q, res.QE, atol=1e-10)
        assert_allclose(res3.i2, res.I2 / 100, atol=1e-10)  # percent in R
        assert_allclose(res3.h2, res.H2, atol=1e-10)
        q, pv, df = res3.test_homogeneity()
        assert_allclose(pv, res.QEp, atol=1e-10)
        assert_allclose(q, res.QE, atol=1e-10)
        assert_allclose(df, 9 - 1, atol=1e-10)

        # compare FE estimate
        res_fe = results_meta.exk1_fe
        assert_allclose(res3.mean_effect_fe, res_fe.b, atol=1e-13)
        assert_allclose(res3.sd_eff_w_fe, res_fe.se, atol=1e-10)

        assert_allclose(ci[0][0], res_fe.ci_lb, atol=1e-10)
        assert_allclose(ci[0][1], res_fe.ci_ub, atol=1e-10)

        # compare FE, RE with HKSJ adjustment
        res_dls = results_meta.exk1_dl_hksj
        res_fes = results_meta.exk1_fe_hksj

        assert_allclose(res3.mean_effect_re, res_dls.b, atol=1e-13)
        assert_allclose(res3.mean_effect_fe, res_fes.b, atol=1e-13)

        assert_allclose(res3.sd_eff_w_fe * np.sqrt(res3.scale_hksj_fe),
                        res_fes.se, atol=1e-10)
        assert_allclose(res3.sd_eff_w_re * np.sqrt(res3.scale_hksj_re),
                        res_dls.se, atol=1e-10)
        assert_allclose(np.sqrt(res3.var_hksj_fe), res_fes.se, atol=1e-10)
        assert_allclose(np.sqrt(res3.var_hksj_re), res_dls.se, atol=1e-10)

        # metafor uses t distribution for hksj
        ci = res3.conf_int(use_t=True)  # fe, re, fe_wls, re_wls
        assert_allclose(ci[3][0], res_dls.ci_lb, atol=1e-10)
        assert_allclose(ci[3][1], res_dls.ci_ub, atol=1e-10)
        assert_allclose(ci[2][0], res_fes.ci_lb, atol=1e-10)
        assert_allclose(ci[2][1], res_fes.ci_ub, atol=1e-10)
