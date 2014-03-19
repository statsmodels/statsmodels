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
        results = ses(dta.values, alpha=.9, forecast=48)

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

        results = brown_linear(dta.values, alpha=.9, forecast=48)

    def test_holt_des_ndarray(self):
        # model with trend for additive
        dta = (self.decomposed.resid + self.decomposed.trend).dropna()
        results = holt_des(dta.values, .86, .523, initial=None)
        expected_res = expected.holt_des()

        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        results = holt_des(dta.values, .86, .523, initial=None,
                           forecast=48)
        np.testing.assert_almost_equal(results.forecasts,
                                       expected_res.forecasts, 8)

        # model with exponential trend for multiplicative
        expected_res = expected.holt_des_mult()
        results = holt_des(dta.values, .86, .523, trend='mult',
                           initial=None)

        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        results = holt_des(dta.values, .86, .523, initial=None,
                           forecast=48, trend='m')
        np.testing.assert_almost_equal(results.forecasts,
                                       expected_res.forecasts, 8)

    def test_damp_es_ndarray(self):
        dta = self.dta
        # used when linear trend is decaying and with no seasonality
        trend = self.decomposed.trend[::-1]
        dta = (self.decomposed.resid + trend.values).dropna()

        results = damp_es(dta.values, alpha=.7826, gamma=.0299, trend='a',
                          damp=.98, forecast=48,
                          initial={'bt' : -0.25091078545258960197,
                                   'st' : 372.02473545186114733951})
        # NOTE: technically you should only used optimized initial params
        # with damped. took these from R
        expected_res = expected.damped_trend()
        # NOTE: This error seems a little big to be rounding error only
        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 1)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 1)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 1)
        np.testing.assert_allclose(results.forecasts,
                                   expected_res.forecasts, rtol=1e-3)


        results = damp_es(dta.values, alpha=.782599999999999962341,
                          gamma=0.029899999999999999495, trend='m',
                          damp=.98, forecast=48,
                          initial={'bt' : 0.999321789870789678467,
                                   'st' : 372.028495898940150254930})
        expected_res = expected.damped_mult_trend()
        # NOTE: technically you should only used optimized initial params
        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 3)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 1)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 2)
        np.testing.assert_almost_equal(results.forecasts,
                                       expected_res.forecasts, 3)



    def test_seasonal_es_ndarray(self):
        dta = (self.decomposed.resid + self.decomposed.seasonal).dropna()
        results = seasonal_es(dta.values, alpha=.0043, delta=.2586, cycle=12,
                              damp=False)

        expected_res = expected.hw_seas()
        #np.testing.assert_(not hasattr(results, "trend"))
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        results = seasonal_es(dta.values, alpha=.0043, delta=.2586,
                              cycle=12, forecast=48, damp=False)
        np.testing.assert_almost_equal(results.forecasts,
                                       expected_res.forecasts, 8)

        # multiplicative
        # shift to be positive
        np.testing.assert_raises(ValueError, seasonal_es,
                                 dta.values, alpha=.0043, delta=.2586,
                                 cycle=12, damp=False, season='m')

        results = seasonal_es(dta.values + 5, alpha=.0043, delta=.2586,
                              cycle=12,
                              damp=False, season='m')

        expected_res = expected.hw_seas_mult()
        #np.testing.assert_(not hasattr(results, "trend"))
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        results = seasonal_es(dta.values + 5, alpha=.0043, delta=.2586,
                              cycle=12, forecast=48, damp=False,
                              season='m')
        #NOTE: This loses a little precision each cycle in R, why?
        # is it numerical or is there are reason the forecasts shrink?
        np.testing.assert_almost_equal(results.forecasts,
                                       expected_res.forecasts, 3)


    def exp_smoothing_ndarray(self):

        pass
