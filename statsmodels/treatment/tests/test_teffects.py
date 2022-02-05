"""
Created on Feb 3, 2022 1:04:22 PM

Author: Josef Perktold
License: BSD-3
"""

import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Probit
from statsmodels.treatment.treatment_effects import (
    TreatmentEffect
    )

from .results import results_teffects as res_st


cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'cataneo2.csv'
file_path = os.path.join(cur_dir, 'results', file_name)

dta_cat = pd.read_csv(file_path)

formula = 'mbsmoke_ ~ mmarried_ + mage + mage2 + fbaby_ + medu'
res_probit = Probit.from_formula(formula, dta_cat).fit()

methods = [
    ("ra", res_st.results_ra),
    ("ipw", res_st.results_ipw),
    ("aipw", res_st.results_aipw),
    ("aipw_wls", res_st.results_aipw_wls),
    ("ipw_ra", res_st.results_ipwra),
    ]


class TestTEffects():

    @classmethod
    def setup_class(cls):
        formula_outcome = 'bweight ~ prenatal1_ + mmarried_ + mage + fbaby_'
        mod = OLS.from_formula(formula_outcome, dta_cat)
        tind = np.asarray(dta_cat['mbsmoke_'])
        cls.teff = TreatmentEffect(mod, tind, results_select=res_probit)

    def test_aux(self):
        prob = res_probit.predict()
        assert prob.shape == (4642,)

    @pytest.mark.parametrize('case', methods)
    def test_effects(self, case):
        meth, res2 = case
        teff = self.teff

        res1 = getattr(teff, meth)(return_results=False)
        res1 = np.asarray(res1).squeeze()
        assert_allclose(res1[:2], res2.table[:2, 0], rtol=1e-4)

        if meth in ["ipw", "aipw"]:
            res1 = getattr(teff, meth)(return_results=True)
            assert_allclose(res1.params[:2], res2.table[:2, 0], rtol=1e-5)
            assert_allclose(res1.bse[:2], res2.table[:2, 1], rtol=5e-4)
            assert_allclose(res1.tvalues[:2], res2.table[:2, 2], rtol=5e-4)
            assert_allclose(res1.pvalues[:2], res2.table[:2, 3],
                            rtol=1e-4, atol=1e-15)
            ci = res1.conf_int()
            assert_allclose(ci[:2, 0], res2.table[:2, 4], rtol=1e-4)
            assert_allclose(ci[:2, 1], res2.table[:2, 5], rtol=1e-4)

            # all GMM params
            # constant is in different position
            # assert_allclose(res1.params, res2.table[:, 0], rtol=1e-4)
