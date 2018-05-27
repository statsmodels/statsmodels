import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly


class Forecast:
    def __init__(self, endog, model, test_sample=0.2, **spec):
        if type(test_sample) == str:
            self.endog_training = endog[:test_sample][:-1]
            self.endog_test = endog[test_sample:]
        else:
            if type(test_sample) == float | type(test_sample) == int:
                if test_sample > 0.0 & test_sample < 1.0:
                    # here test_sample is containing the number of observations
                    # to consider for the endog_test
                    test_sample = int(test_sample * len(endog))
            self.endog_training = endog[:-test_sample]
            self.endog_test = endog[-test_sample:]
        self.endog_training = endog[:test_sample][:-1]
        self.endog_test = endog[test_sample:]
        self.model = model(self.endog_training, **spec)
        self.results = self.model.fit()

    @cache_readonly
    def nobs_training(self, endog_training):
        return len(endog_training)

    @cache_readonly
    def nobs_test(self, endog_test):
        return len(endog_test)

    @cache_readonly
    def forecast_val(self):
        """
        (array) The model forecast values
        """
        return self.results.forecast(self.nobs_test)

    # In this case we'll be computing accuracy using forecast errors
    # instead of residual values. The forecast error is calculated on test set.
    @cache_readonly
    def forecast_error(self):
        """
        (array) The model forecast errors
        """
        return self.endog_test - self.forecast_val

    @cache_readonly
    def mae(self):
        """
        (float) Mean Absolute Error
        """
        return np.mean(np.abs(self.forecast_error))

    @cache_readonly
    def rmse(self):
        """
        (float) Root mean squared error
        """
        return np.sqrt(np.mean(self.forecast_error ** 2))

    @cache_readonly
    def mape(self):
        """
        (float) Mean absolute percentage error
        """
        return np.mean(np.abs((self.forecast_error) / self.endog_test)) * 100

    @cache_readonly
    def smape(self):
        """
        (float) symmetric Mean absolute percentage error
        """
        return np.mean(
                200*np.abs(self.forecast_error) /
                np.abs(self.endog_test + self.forecast_val))

    @cache_readonly
    def mase(self):
        """
        (float) Mean Absolute Scaled Error
        """
        # for non-seasonal time series
        e_j = self.forecast_error  # not sure if this the correct approach
        sum_v = 0
        for val in range(2, self.nobs_training):
            sum_v += (self.endog_training[val] - self.endog_training[val - 1])
        q_j = e_j*(self.nobs_training-1)/sum_v
        return np.mean(np.abs(q_j))


class ForecastSet:
    """docstring for second class."""

    def __init__(self, endog, test_sample):
        self.endog = endog
        self.test_sample = test_sample
        self.models = []

    def add(self, model, **spec):
        fcast = Forecast(self.endog, model, self.test_sample, **spec)
        self.models.append(fcast)

    def select(self, measure='mae'):
        """selection based on criteria provided by the user"""
        measure_vals = np.zeros(len(self.models))
        for mod in range(len(self.models)):
            measure_vals[mod] = self.models[mod].measure
        min_measure = measure_vals.min()
        model = np.where(measure_vals == min_measure)
        return self.models[model]
