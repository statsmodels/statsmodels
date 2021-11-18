# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:48:37 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose

from statsmodels.tools.tools import add_constant

from statsmodels.base._prediction_inference import PredictionResultsMonotonic

from statsmodels.discrete.discrete_model import (
    NegativeBinomialP,
    )
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP
    )

from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp


# copied from `test_gmm_poisson.TestGMMAddOnestep`
XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()
endog_name = 'docvis'
exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
endog = DATA[endog_name]
exog = DATA[exog_names]


class CheckPredict():

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        # Note we have alpha, stata has lnalpha
        sl1 = slice(self.k_infl, -1, None)
        sl2 = slice(0, -(self.k_infl + 1), None)
        assert_allclose(res1.params[sl1], res2.params[sl2], rtol=self.rtol)
        assert_allclose(res1.bse[sl1], res2.bse[sl2], rtol=30 * self.rtol)
        assert_allclose(res1.params[-1], np.exp(res2.params[-1]),
                        rtol=self.rtol)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        ex = np.asarray(exog).mean(0)

        # test for which="mean"
        rdf = res2.results_margins_atmeans
        pred = res1.get_prediction(ex, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"][0], rtol=1e-4)
        assert_allclose(pred.se, rdf["se"][0], rtol=1e-4,  atol=1e-4)
        if isinstance(pred, PredictionResultsMonotonic):
            # default method is endpoint transformation for non-ZI models
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"][0], rtol=1e-3,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"][0], rtol=1e-3,  atol=1e-4)

            ci = pred.conf_int(method="delta")[0]
            assert_allclose(ci[0], rdf["ll"][0], rtol=1e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"][0], rtol=1e-4,  atol=1e-4)
        else:
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"][0], rtol=1e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"][0], rtol=1e-4,  atol=1e-4)

        rdf = res2.results_margins_mean
        pred = res1.get_prediction(average=True, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"][0], rtol=3e-4)  # self.rtol)
        assert_allclose(pred.se, rdf["se"][0], rtol=3e-3,  atol=1e-4)
        if isinstance(pred, PredictionResultsMonotonic):
            # default method is endpoint transformation for non-ZI models
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"][0], rtol=1e-3,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"][0], rtol=1e-3,  atol=1e-4)

            ci = pred.conf_int(method="delta")[0]
            assert_allclose(ci[0], rdf["ll"][0], rtol=1e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"][0], rtol=1e-4,  atol=1e-4)
        else:
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"][0], rtol=5e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"][0], rtol=5e-4,  atol=1e-4)

        # test for which="prob"
        rdf = res2.results_margins_atmeans
        pred = res1.get_prediction(ex, which="prob", y_values=np.arange(2),
                                   **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"][1:3], rtol=3e-4)  # self.rtol)
        assert_allclose(pred.se, rdf["se"][1:3], rtol=3e-3,  atol=1e-4)

        ci = pred.conf_int()
        assert_allclose(ci[:, 0], rdf["ll"][1:3], rtol=5e-4,  atol=1e-4)
        assert_allclose(ci[:, 1], rdf["ul"][1:3], rtol=5e-4,  atol=1e-4)

        rdf = res2.results_margins_mean
        pred = res1.get_prediction(which="prob", y_values=np.arange(2),
                                   average=True, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"][1:3], rtol=5e-3)  # self.rtol)
        assert_allclose(pred.se, rdf["se"][1:3], rtol=3e-3,  atol=5e-4)

        ci = pred.conf_int()
        assert_allclose(ci[:, 0], rdf["ll"][1:3], rtol=5e-4,  atol=1e-3)
        assert_allclose(ci[:, 1], rdf["ul"][1:3], rtol=5e-4,  atol=5e-3)


class TestNegativeBinomialPPredict(CheckPredict):

    @classmethod
    def setup_class(cls):
        # using newton has results much closer to Stata than bfgs
        res1 = NegativeBinomialP(endog, exog).fit(method="newton", maxiter=300)
        cls.res1 = res1
        cls.res2 = resp.results_nb_docvis
        cls.pred_kwds_mean = {}
        cls.k_infl = 0
        cls.rtol = 1e-8


class TestZINegativeBinomialPPredict(CheckPredict):

    @classmethod
    def setup_class(cls):
        exog_infl = add_constant(DATA["aget"], prepend=False)
        mod_zinb = ZeroInflatedNegativeBinomialP(endog, exog,
                                                 exog_infl=exog_infl, p=2)

        sp = np.array([
            -6.58, -1.28, 0.19, 0.08, 0.22, -0.05, 0.03, 0.17, 0.27, 0.68,
            0.62])
        # using newton. bfgs has non-invertivle hessian at convergence
        # start_params not needed, but speed up
        res1 = mod_zinb.fit(start_params=sp, method="newton", maxiter=300)
        cls.res1 = res1
        cls.res2 = resp.results_zinb_docvis
        cls.pred_kwds_mean = {"exog_infl": exog_infl.mean(0)}
        cls.k_infl = 2
        cls.rtol = 1e-4
