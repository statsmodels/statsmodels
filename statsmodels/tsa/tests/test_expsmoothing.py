import numpy as np
from statsmodels.tsa._expsmoothing import (exp_smoothing, ses, brown_linear,
                                           holt_des, damp_es, seasonal_es)
from statsmodels.tsa.filters.seasonal import seasonal_decompose
from statsmodels.datasets import co2
from results.results_expsmoothing import ExpSmoothingResults
expected = ExpSmoothingResults()


class TestExpSmoothing:
    @classmethod
    def setupClass(cls):
        dta = co2.load_pandas().data
        dta.co2.interpolate(inplace=True)
        cls.dta = dta.resample('MS')  # shorter better to test

        # decompose it - to use appropriate methods for results
        cls.decomposed = seasonal_decompose(cls.dta)

    def test_ses_ndarray(self):
        dta = self.decomposed.resid.dropna()  # no trend, no season

        # without forecasting, smoke only
        results = ses(dta.values, alpha=.9)
        expected_res = expected.ses()

        # with forecasting
        results = ses(dta.values, alpha=.9, forecast=10)

        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.forecasts,
                                       expected_res.forecasts, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)

    def test_brown_linear_ndarray(self):
        # smoke tests, no reference implementation
        dta = self.decomposed.trend + self.decomposed.resid

        results = brown_linear(dta.values, alpha=.9)

        results = brown_linear(dta.values, alpha=.9, forecast=10)

    def test_holt_des_ndarray(self):
        # model with trend for additive
        dta = self.decomposed.resid + self.decomposed.trend
        results = holt_des(dta.values, .9, .2, initial=None)

        # model with exponential trend for multiplicative
        dta = self.decomposed.resid + np.exp(self.decomposed.trend)
        results = holt_des(dta.values, .9, .2, trend='multiplicative',
                           initial=None)

    def test_damp_es_ndarray(self):
        dta = self.dta
        # used when linear trend is decaying and with no seasonality
        trend = self.decomposed.trend[::-1]
        dta = self.decomposed.resid + trend
        results = damp_es(dta.values, .9, .5)

        # decaying exponential trend
        dta = self.decomposed.resid + 1./self.decomposed.trend
        results = damp_es(dta.values, .9, .5, trend='m')


    #def test_seasonal_es_ndarray(self):
    #    dta = self.decomposed.resid + self.decomposed.seasonal
    #    results = seasonal_es(dta.values, alpha=.9, delta=.5, cycle=12)


    #    dta = self.decomposed.resid + np.exp(self.decomposed.seasonal)
    #    results = seasonal_es(dta.values, alpha=.9, delta=.5, cycle=12,
    #                          seasonal='a')


    def exp_smoothing_ndarray(self):
        pass
