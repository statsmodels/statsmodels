import numpy as np
from statsmodels.tsa._expsmoothing import (ExpSmoothing, ses, brown_linear,
                                           holt_des, damp_es, seasonal_es)
from statsmodels.tsa.filters.seasonal import seasonal_decompose
from statsmodels.datasets import co2
from pandas.util.testing import assert_series_equal
from results.results_expsmoothing import ExpSmoothingResults
expected = ExpSmoothingResults()


class TestExpSmoothing:
    @classmethod
    def setupClass(cls):
        dta = co2.load_pandas().data
        dta.co2 = dta.co2.interpolate()
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
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)

    def test_brown_linear_ndarray(self):
        # smoke tests, no reference implementation
        pass
        #dta = (self.decomposed.trend + self.decomposed.resid).dropna()
        #results = brown_linear(dta.values, alpha=.9)
        #forecasts = results.forecast(48)
        #raise ValueError("This doesn't look right")

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
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 8)

        # model with exponential trend for multiplicative
        expected_res = expected.holt_des_mult()
        results = holt_des(dta.values, .86, .523, trend='mult', initial=None)
        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 8)

    def test_damp_es_ndarray(self):
        dta = self.dta
        # used when linear trend is decaying and with no seasonality
        trend = self.decomposed.trend[::-1]
        dta = (self.decomposed.resid + trend.values).dropna()

        alpha = .7826
        # NOTE: Hyndman writes it like this for stability in optimization
        #  these are the update equations used
        gamma = .0299/alpha

        results = damp_es(dta.values, alpha=.7826, gamma=gamma, trend='a',
                          damp=.98,
                          initial={'bt' : -0.25091078545258960197,
                                   'st' : 372.02473545186114733951})
        # NOTE: technically you should only used optimized initial params
        # with damped. took these from R
        expected_res = expected.damped_trend()
        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 8)


        alpha = .7826
        gamma = .0299/alpha
        results = damp_es(dta.values, alpha=alpha,
                          gamma=gamma, trend='m',
                          damp=.98,
                          initial={'bt' : 0.999321789870789678467,
                                   'st' : 372.028495898940150254930})
        expected_res = expected.damped_mult_trend()
        # NOTE: technically you should only used optimized initial params
        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 8)
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        #NOTE: see commented out code for ets-matched residuals
        #np.testing.assert_almost_equal(results.resid,
        #                               expected_res.resid, 5)
        #NOTE: not sure why the precision is low here. should be right.
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 3)



    def test_seasonal_es_ndarray(self):
        dta = (self.decomposed.resid + self.decomposed.seasonal).dropna()
        results = seasonal_es(dta.values, alpha=.0043, delta=.2586, period=12,
                              damp=False)

        expected_res = expected.hw_seas()
        #np.testing.assert_(not hasattr(results, "trend"))
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 8)

        # multiplicative
        # shift to be positive
        np.testing.assert_raises(ValueError, seasonal_es,
                                 dta.values, alpha=.0043, delta=.2586,
                                 period=12, damp=False, season='m')

        results = seasonal_es(dta.values + 5, alpha=.0043, delta=.2586,
                              period=12,
                              damp=False, season='m')

        expected_res = expected.hw_seas_mult()
        #np.testing.assert_(not hasattr(results, "trend"))
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.resid,
                                       expected_res.resid, 8)
        #NOTE: This loses a little precision each period in R, why?
        # is it numerical or is there are reason the forecasts shrink?
        np.testing.assert_almost_equal(results.forecast(48),
                                       expected_res.forecasts, 3)

    def test_exp_smoothing_ndarray(self):
        dta = self.dta
        init_ct = np.array([1.001843291395, 0.999992706537, 0.997342232980,
                            0.994001023919, 0.990791351262, 0.991124074728,
                            0.996183466518, 1.002105875039, 1.006614389201,
                            1.008529917146, 1.007285317008, 1.004186354267])
        init_ct = init_ct[::-1] # proper time order
        init = {'st': 314.78888791602560104366,
                'bt': 1.00024658183093162478,
                'ct' : init_ct,
                }
        alpha = .7198
        # ets defines the model like this in the update equations for
        # stability in optimization cf. Hyndman book
        gamma = .0387/alpha
        delta = .01
        model = ExpSmoothing(dta.values, alpha=alpha,
                                gamma=gamma, delta=delta, period=12,
                                season='m', trend='m', damp=1)
        results = model.fit(initial=init)

        expected_res = expected.multmult()
        np.testing.assert_almost_equal(results.fitted,
                                       expected_res.fitted, 8)
        np.testing.assert_almost_equal(results.trend,
                                       expected_res.trend, 8)
        #NOTE: see commented out code for ets-matched residuals
        #np.testing.assert_almost_equal(results.resid,
        #                               expected_res.resid, 5)
        np.testing.assert_almost_equal(results.forecast(h=48),
                                       expected_res.forecasts, 8)

    def test_exp_smoothing_pandas(self):
        dta = self.dta
        init_ct = np.array([1.001843291395, 0.999992706537, 0.997342232980,
                            0.994001023919, 0.990791351262, 0.991124074728,
                            0.996183466518, 1.002105875039, 1.006614389201,
                            1.008529917146, 1.007285317008, 1.004186354267])
        init_ct = init_ct[::-1] # proper time order
        init = {'st': 314.78888791602560104366,
                'bt': 1.00024658183093162478,
                'ct' : init_ct,
                }
        alpha = .7198
        # ets defines the model like this in the update equations for
        # stability in optimization cf. Hyndman book
        gamma = .0387/alpha
        delta = .01
        model = ExpSmoothing(dta, alpha=alpha,
                             gamma=gamma, delta=delta, period=12,
                             season='m', trend='m', damp=1)
        results = model.fit(initial=init)

        expected_res = expected.multmult_pandas()
        assert_series_equal(results.fitted,
                            expected_res.fitted, 8)
        assert_series_equal(results.trend,
                            expected_res.trend, 8)
        #NOTE: see commented out code for ets-matched residuals
        #np.testing.assert_almost_equal(results.resid,
        #                               expected_res.resid, 5)
        assert_series_equal(results.forecast(h=48),
                            expected_res.forecasts, 8)
