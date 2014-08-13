"""
Tests for SARIMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from statsmodels.tsa.statespace import sarimax
from .results import results_sarimax
from statsmodels.tools import add_constant
from numpy.testing import assert_almost_equal
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class SARIMAXTests(sarimax.SARIMAX):
    def test_loglike(self):
        assert_almost_equal(
            self.result.llf,
            self.true['loglike'], 4
        )

    def test_aic(self):
        assert_almost_equal(
            self.result.aic,
            self.true['aic'], 3
        )

    def test_bic(self):
        assert_almost_equal(
            self.result.bic,
            self.true['bic'], 3
        )


class ARIMA(SARIMAXTests):
    """
    ARIMA model

    Stata arima documentation, Example 1
    """
    def __init__(self, true, *args, **kwargs):
        self.true = true
        endog = true['data']

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        super(ARIMA, self).__init__(endog, order=(1, 1, 1), trend='c',
                                    *args, **kwargs)

        # Stata estimates the mean of the process, whereas SARIMAX estimates
        # the intercept of the process. Get the intercept.
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'],
                       true['params_variance']]

        self.update(params)


class TestARIMAStationary(ARIMA):
    def __init__(self):
        super(TestARIMAStationary, self).__init__(
            results_sarimax.wpi1_stationary
        )
        self.result = self.filter()


class TestARIMADiffuse(ARIMA):
    def __init__(self):
        super(TestARIMADiffuse, self).__init__(results_sarimax.wpi1_diffuse)
        self.initialize_approximate_diffuse(self.true['initial_variance'])
        self.result = self.filter()


class AdditiveSeasonal(SARIMAXTests):
    """
    ARIMA model with additive seasonal effects

    Stata arima documentation, Example 2
    """
    def __init__(self, true, *args, **kwargs):
        self.true = true
        endog = np.log(true['data'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        super(AdditiveSeasonal, self).__init__(
            endog, order=(1, 1, (1, 0, 0, 1)), trend='c', *args, **kwargs
        )

        # Stata estimates the mean of the process, whereas SARIMAX estimates
        # the intercept of the process. Get the intercept.
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'],
                       true['params_variance']]

        self.update(params)


class TestAdditionSeasonal(AdditiveSeasonal):
    def __init__(self):
        super(TestAdditionSeasonal, self).__init__(
            results_sarimax.wpi1_seasonal
        )
        self.result = self.filter()


class Airline(SARIMAXTests):
    """
    Multiplicative SARIMA model: "Airline" model

    Stata arima documentation, Example 3
    """
    def __init__(self, true, *args, **kwargs):
        self.true = true
        endog = np.log(true['data'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        super(Airline, self).__init__(
            endog, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
            trend='n', *args, **kwargs
        )

        params = np.r_[true['params_ma'], true['params_seasonal_ma'],
                       true['params_variance']]

        self.update(params)


class TestAirlineHamilton(Airline):
    def __init__(self):
        super(TestAirlineHamilton, self).__init__(
            results_sarimax.air2_stationary
        )
        self.result = self.filter()


class TestAirlineHarvey(Airline):
    def __init__(self):
        super(TestAirlineHarvey, self).__init__(
            results_sarimax.air2_stationary, hamilton_representation=False
        )
        self.result = self.filter()


class TestAirlineStateDifferencing(Airline):
    def __init__(self):
        super(TestAirlineStateDifferencing, self).__init__(
            results_sarimax.air2_stationary, simple_differencing=False,
            hamilton_representation=False
        )
        self.result = self.filter()

    def test_bic(self):
        # Due to diffuse component of the state (which technically changes the
        # BIC calculation - see Durbin and Koopman section 7.4), this is the
        # best we can do for BIC
        assert_almost_equal(
            self.result.bic,
            self.true['bic'], 0
        )


class Friedman(SARIMAXTests):
    """
    ARMAX model: Friedman quantity theory of money

    Stata arima documentation, Example 4
    """
    def __init__(self, true, *args, **kwargs):
        self.true = true
        endog = np.r_[true['data']['consump']]
        exog = add_constant(true['data']['m2'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        super(Friedman, self).__init__(
            endog, exog=exog, order=(1, 0, 1), *args, **kwargs
        )

        params = np.r_[true['params_exog'], true['params_ar'],
                       true['params_ma'], true['params_variance']]

        self.update(params)


class TestFriedmanMLERegression(Friedman):
    def __init__(self):
        super(TestFriedmanMLERegression, self).__init__(
            results_sarimax.friedman2_mle
        )
        self.result = self.filter()


class TestFriedmanStateRegression(Friedman):
    def __init__(self):
        # Remove the regression coefficients from the parameters, since they
        # will be estimated as part of the state vector
        true = dict(results_sarimax.friedman2_mle)

        true['mle_params_exog'] = true['params_exog'][:]
        true['mle_se_exog'] = true['se_exog'][:]

        true['params_exog'] = []
        true['se_exog'] = []

        super(TestFriedmanStateRegression, self).__init__(
            true, mle_regression=False
        )

        self.result = self.filter()

    def test_regression_parameters(self):
        assert_almost_equal(
            self.result.filtered_state[-2:, -1],
            self.true['mle_params_exog'], 1
        )

    # Loglikelihood (and so aic, bic) is slightly different when states are
    # integrated into the state vector
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass


class TestFriedmanPredict(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, prediction

    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This follows the given Stata example, although it is not truly forecasting
    because it compares using the actual data (which is available in the
    example but just not used in the parameter MLE estimation) against dynamic
    prediction of that data. Here `test_predict` matches the first case, and
    `test_dynamic_predict` matches the second.
    """
    def __init__(self):
        super(TestFriedmanPredict, self).__init__(
            results_sarimax.friedman2_predict
        )

        self.result = self.filter()

    # loglike, aic, bic are not the point of this test (they could pass, but we
    # would have to modify the data so that they were calculated to
    # exclude the last 15 observations)
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_predict(self):
        assert_almost_equal(
            self.result.predict()[0][0],
            self.true['predict'], 3
        )

    def test_dynamic_predict(self):
        dynamic = len(self.true['data']['consump'])-15-1
        assert_almost_equal(
            self.result.predict(dynamic=dynamic)[0][0],
            self.true['dynamic_predict'], 3
        )


class TestFriedmanForecast(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, forecasts

    Variation on:
    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This is a variation of the Stata example, in which the endogenous data is
    actually made to be missing so that the predict command must forecast.

    As another unit test, we also compare against the case in State when
    predict is used against missing data (so forecasting) with the dynamic
    option also included. Note, however, that forecasting in State space models
    amounts to running the Kalman filter against missing datapoints, so it is
    not clear whether "dynamic" forecasting (where instead of missing
    datapoints for lags, we plug in previous forecasted endog values) is
    meaningful.
    """
    def __init__(self):
        true = dict(results_sarimax.friedman2_predict)

        true['forecast_data'] = {
            'consump': true['data']['consump'][-15:],
            'm2': true['data']['m2'][-15:]
        }
        true['data'] = {
            'consump': true['data']['consump'][:-15],
            'm2': true['data']['m2'][:-15]
        }

        super(TestFriedmanForecast, self).__init__(true)

        self.result = self.filter()

    # loglike, aic, bic are not the point of this test (they could pass, but we
    # would have to modify the data so that they were calculated to
    # exclude the last 15 observations)
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_forecast(self):
        end = len(self.true['data']['consump'])+15-1
        exog = add_constant(self.true['forecast_data']['m2'])
        assert_almost_equal(
            self.result.predict(end=end, exog=exog)[0][0],
            self.true['forecast'], 3
        )

    def test_dynamic_forecast(self):
        end = len(self.true['data']['consump'])+15-1
        dynamic = len(self.true['data']['consump'])-1
        exog = add_constant(self.true['forecast_data']['m2'])
        assert_almost_equal(
            self.result.predict(end=end, dynamic=dynamic, exog=exog)[0][0],
            self.true['dynamic_forecast'], 3
        )
