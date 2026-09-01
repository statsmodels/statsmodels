"""
Created on Feb 3, 2022 1:04:22 PM

Author: Josef Perktold
License: BSD-3
"""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.discrete.discrete_model import Probit
from statsmodels.regression.linear_model import OLS
from statsmodels.treatment.treatment_effects import TreatmentEffect

from .results import results_teffects as res_st

cur_dir = Path(__file__).parent.resolve()

file_name = "cataneo2.csv"
file_path = Path(cur_dir).joinpath("results", file_name)

dta_cat = pd.read_csv(file_path)

formula = "mbsmoke_ ~ mmarried_ + mage + mage2 + fbaby_ + medu"
res_probit = Probit.from_formula(formula, dta_cat).fit()

methods = [
    ("ra", res_st.results_ra),
    ("ipw", res_st.results_ipw),
    ("aipw", res_st.results_aipw),
    ("aipw_wls", res_st.results_aipw_wls),
    ("ipw_ra", res_st.results_ipwra),
    ]

method_labels = [
    ("ra", "RA"),
    ("ipw", "IPW"),
    ("aipw", "AIPW"),
    ("aipw_wls", "AIPW-WLS"),
    ("ipw_ra", "IPW-RA"),
    ]


class TestTEffects:

    @classmethod
    def setup_class(cls):
        formula_outcome = "bweight ~ prenatal1_ + mmarried_ + mage + fbaby_"
        mod = OLS.from_formula(formula_outcome, dta_cat)
        tind = np.asarray(dta_cat["mbsmoke_"])
        cls.teff = TreatmentEffect(mod, tind, results_select=res_probit)

    def test_aux(self):
        prob = res_probit.predict()
        assert prob.shape == (4642,)

    @pytest.mark.parametrize("case", method_labels)
    def test_method_label(self, case):
        # each estimator must label its own results, not report "IPW"
        meth, label = case
        res = getattr(self.teff, meth)(return_results=True)
        assert res.method == label

    @pytest.mark.parametrize("case", methods)
    def test_effects(self, case):
        meth, res2 = case
        teff = self.teff

        res1 = getattr(teff, meth)(return_results=False)
        assert_allclose(res1[:2], res2.table[:2, 0], rtol=1e-4)

        # if meth in ["ipw", "aipw", "aipw_wls", "ra", "ipw_ra"]:
        res0 = getattr(teff, meth)(return_results=True)
        assert_allclose(res1, res0.effect, rtol=1e-4)
        res1 = res0.results_gmm
        # TODO: check ra and ipw difference 5e-6, others pass at 1e-12
        assert_allclose(res0.start_params, res1.params, rtol=1e-5)
        assert_allclose(res1.params[:2], res2.table[:2, 0], rtol=1e-5)
        assert_allclose(res1.bse[:2], res2.table[:2, 1], rtol=1e-3)
        assert_allclose(res1.tvalues[:2], res2.table[:2, 2], rtol=1e-3)
        assert_allclose(res1.pvalues[:2], res2.table[:2, 3],
                        rtol=1e-4, atol=1e-15)
        ci = res1.conf_int()
        assert_allclose(ci[:2, 0], res2.table[:2, 4], rtol=5e-4)
        assert_allclose(ci[:2, 1], res2.table[:2, 5], rtol=5e-4)

        # test all GMM params
        # constant is in different position in Stata, `idx` rearanges
        k_p = len(res1.params)
        if k_p == 8:
            # IPW, no outcome regression
            idx = [0, 1, 7, 2, 3, 4, 5, 6]
        elif k_p == 18:
            idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10, 17, 12, 13, 14,
                   15, 16]
        elif k_p == 12:
            # RA, no selection regression
            idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10]
        else:
            idx = np.arange(k_p)

        # TODO: check if improved optimization brings values closer
        assert_allclose(res1.params, res2.table[idx, 0], rtol=1e-4)
        assert_allclose(res1.bse, res2.table[idx, 1], rtol=0.05)

        # test effects on the treated, no Stata reference values for aipw
        if not meth.startswith("aipw"):
            table = res2.table_t

            res1 = getattr(teff, meth)(return_results=False, effect_group=1)
            assert_allclose(res1[:2], table[:2, 0], rtol=1e-4)

            res0 = getattr(teff, meth)(return_results=True, effect_group=1)
            # TODO: check ipw difference 1e-5, others pass at 1e-12
            assert_allclose(res1, res0.effect, rtol=2e-5)
            res1 = res0.results_gmm
            # TODO: check ra difference 4e-5, others pass at 1e-12
            assert_allclose(res0.start_params, res1.params, rtol=5e-5)
            assert_allclose(res1.params[:2], table[:2, 0], rtol=5e-5)
            assert_allclose(res1.bse[:2], table[:2, 1], rtol=1e-3)
            assert_allclose(res1.tvalues[:2], table[:2, 2], rtol=1e-3)
            assert_allclose(res1.pvalues[:2], table[:2, 3],
                            rtol=1e-4, atol=1e-15)
            ci = res1.conf_int()
            assert_allclose(ci[:2, 0], table[:2, 4], rtol=5e-4)
            assert_allclose(ci[:2, 1], table[:2, 5], rtol=5e-4)

            # consistency check, effect on untreated,  not in Stata
            res1 = getattr(teff, meth)(return_results=False, effect_group=0)
            res0 = getattr(teff, meth)(return_results=True, effect_group=0)
            assert_allclose(res1, res0.effect, rtol=1e-12)
            assert_allclose(res0.start_params, res0.results_gmm.params,
                            rtol=1e-12)

    @pytest.mark.parametrize("meth", ["aipw", "aipw_wls"])
    @pytest.mark.parametrize("effect_group", [1, 0])
    def test_aipw_effect_group(self, meth, effect_group):
        # no Stata reference values, check against direct computation
        teff = self.teff
        res1 = getattr(teff, meth)(return_results=False,
                                   effect_group=effect_group)
        res0 = getattr(teff, meth)(return_results=True,
                                   effect_group=effect_group)
        assert_allclose(res1, res0.effect, rtol=1e-12)
        assert_allclose(res0.start_params, res0.results_gmm.params,
                        rtol=1e-12)
        assert res0.effect_group == effect_group

        tind = teff.treatment
        endog = teff.model_pool.endog
        exog = teff.model_pool.exog
        prob = res_probit.predict()
        if meth == "aipw":
            fit0 = teff.results0.predict(exog)
            fit1 = teff.results1.predict(exog)
        else:
            fit0 = teff.results_ipwwls0.predict(exog)
            fit1 = teff.results_ipwwls1.predict(exog)
        if effect_group == 0:
            # ATC by symmetry: swap treatment and control
            tind, prob = 1 - tind, 1 - prob
            fit0, fit1 = fit1, fit0
        treated = tind == 1
        odds = prob / (1 - prob)
        pom_t = endog[treated].mean()
        pom_c = (fit0[treated].sum()
                 + (odds * (endog - fit0))[~treated].sum()) / treated.sum()
        if effect_group == 0:
            pom_t, pom_c = pom_c, pom_t
        assert_allclose(res1, [pom_t - pom_c, pom_c, pom_t], rtol=1e-12)


@pytest.mark.parametrize("meth", ["ipw_ra", "aipw_wls"])
def test_select_params_not_six(meth):
    # GMM moment conditions used to hardcode 6 selection parameters
    formula_sel = "mbsmoke_ ~ mmarried_ + mage + fbaby_"
    res_sel = Probit.from_formula(formula_sel, dta_cat).fit(disp=0)
    formula_outcome = "bweight ~ prenatal1_ + mmarried_ + mage + fbaby_"
    mod = OLS.from_formula(formula_outcome, dta_cat)
    tind = np.asarray(dta_cat["mbsmoke_"])
    teff = TreatmentEffect(mod, tind, results_select=res_sel)

    res1 = getattr(teff, meth)(return_results=False)
    res0 = getattr(teff, meth)(return_results=True)
    assert_allclose(res1, res0.effect, rtol=1e-12)
    assert_allclose(res0.start_params, res0.results_gmm.params, rtol=1e-12)
