# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:15:46 2023

Author: Josef Perktold
License: BSD-3
"""

import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from statsmodels.genmod import families
from statsmodels.causal.ivcontrol_function import GLMIVCF

path = os.path.abspath(__file__)
dir_path = os.path.join(os.path.dirname(path), "results")


def test_basic():
    csv_path = os.path.join(dir_path, "temp_residinclude.csv")
    dta = pd.read_csv(csv_path, dtype=float)
    dta["const"] = 1.

    y2 = dta["Y"]
    x2 = dta[["const", "X"]]
    y1 = dta["X"]
    x1 = dta[["const", "Z"]]

    endog, explan, endog_explan, instruments = y2, x2, y1, x1
    mod = GLMIVCF(endog, explan, endog_explan, instruments,
                  family=families.Binomial())
    res = mod.fit()

    params2 = [0.235849056603773, 0.455640305098354, 1.524725025365007,
               -0.162472360933428, 1.474839056522943]
    bse2 = [0.0412338339697916, 0.0630056866271331, 0.4112119420601235,
            0.8200832474987831, 0.8740459928452390]
    tvalues2 = [5.719794496348780, 7.231733030620353, 3.707881190722025,
                -0.198116912434147, 1.687370079601843]
    pvalues2 = [1.06652968526624e-08, 4.76868950935508e-13,
                2.09000666546514e-04, 8.42953592884409e-01,
                9.15322133567022e-02]

    assert_allclose(res.params, params2, rtol=1e-7)
    assert_allclose(res.bse, bse2, rtol=1e-6)
    assert_allclose(res.tvalues, tvalues2, rtol=1e-5)
    assert_allclose(res.pvalues, pvalues2, rtol=1e-5)

    # with probit in first stage, control function is still resid_reponse
    mod = GLMIVCF(endog, explan, endog_explan, instruments,
                  family=families.Binomial(),
                  family_first=families.Binomial(link=families.links.Probit()),
                  cf_kwds = {"resid": "resid_response"},
                  )
    res = mod.fit()
    sli = slice(-3, None, None)
    assert_allclose(res.params[sli], params2[sli], rtol=1e-7)
    assert_allclose(res.bse[sli], bse2[sli], rtol=1e-6)
    assert_allclose(res.tvalues[sli], tvalues2[sli], rtol=1e-5)
    assert_allclose(res.pvalues[sli], pvalues2[sli], rtol=1e-5)

    # check attached outcome model
    res_out = res.results_outcome
    assert_allclose(res_out.params, params2[sli], rtol=1e-7)
    assert_allclose(res_out.bse, bse2[sli], rtol=1e-6)
    assert_allclose(res_out.tvalues, tvalues2[sli], rtol=1e-5)
    assert_allclose(res_out.pvalues, pvalues2[sli], rtol=1e-5)

    mean = y2.mean()
    assert_allclose(res_out.predict().mean(), mean, rtol=1e-10)
    mout = res_out.get_prediction(which="mean", average=True).predicted
    assert_allclose(mout, mean, rtol=1e-10)

    # t_test for endogeneity
    constr = np.zeros(len(res.params))
    constr[-1] = 1
    tt1 = res.t_test(constr).summary_frame().to_numpy()
    tt2 = res_out.t_test("x2").summary_frame().to_numpy()
    assert_allclose(tt1, tt2, rtol=1e-10)
