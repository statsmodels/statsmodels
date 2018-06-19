"""Classes to hold the Forecast results individually and in sets."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly


class Forecast(object):
    """Class to hold the data of a single forecast model."""

    def __init__(self, endog, model, test_sample=0.2, **spec):
        """Intialize the data for the Forecast class."""
        # TODO: make date selection of test sample more robust
        if type(test_sample) == str:
            self.endog_training = endog[:test_sample][:-1]
            self.endog_test = endog[test_sample:]
        else:
            if type(test_sample) == float or type(test_sample) == int:
                if test_sample > 0.0 and test_sample < 1.0:
                    # here test_sample is containing the number of observations
                    # to consider for the endog_test
                    test_sample = int(test_sample * len(endog))
            self.endog_training = endog[:-test_sample]
            self.endog_test = endog[-test_sample:]
        self.model = model(self.endog_training, **spec)
        self.results = self.model.fit()

    @property
    def resid(self):
        """(array) The list of residuals while fitting the model."""
        return self.results.resid

    @property
    def fittedvalues(self):
        """(array) The list of fitted values of the time-series model."""
        return self.results.fittedvalues

    @cache_readonly
    def nobs_training(self):
        """(int) Number of observations in the training set."""
        return len(self.endog_training)

    @cache_readonly
    def nobs_test(self):
        """(int) Number of observations in the test set."""
        return len(self.endog_test)

    @cache_readonly
    def forecasts(self):
        """(array) The model forecast values."""
        return self.results.forecast(self.nobs_test)

    # In this case we'll be computing accuracy using forecast errors
    # instead of residual values. The forecast error is calculated on test set.
    @cache_readonly
    def forecasts_error(self):
        """(array) The model forecast errors."""
        return self.endog_test - self.forecasts

    @cache_readonly
    def mae(self):
        """(float) Mean Absolute Error."""
        return np.mean(np.abs(self.forecasts_error))

    @cache_readonly
    def rmse(self):
        """(float) Root mean squared error."""
        return np.sqrt(np.mean(self.forecasts_error ** 2))

    @cache_readonly
    def mape(self):
        """(float) Mean absolute percentage error."""
        return np.mean(np.abs((self.forecasts_error) / self.endog_test)) * 100

    @cache_readonly
    def smape(self):
        """(float) symmetric Mean absolute percentage error."""
        return np.mean(
                200*np.abs(self.forecasts_error) /
                np.abs(self.endog_test + self.forecasts))

    @cache_readonly
    def mase(self):
        """(float) Mean Absolute Scaled Error."""
        # for non-seasonal time series
        e_j = self.forecasts_error  # not sure if this the correct approach
        sum_v = 0
        for val in range(2, self.nobs_training):
            sum_v += (self.endog_training[val] - self.endog_training[val - 1])
        q_j = e_j*(self.nobs_training-1)/sum_v
        return np.mean(np.abs(q_j))


class ForecastSet(object):
    """Class to hold various Forecast objects.

    The class holds various Forecast objects and selects
    the best Forecast object based on some evaluation measure.
    """

    def __init__(self, endog, test_sample):
        """Initialize the values of the ForecastSet class."""
        self.endog = endog
        self.test_sample = test_sample
        self.models = []

    def add(self, model, **spec):
        """Add a Forecast object to this ForecastSet."""
        fcast = Forecast(self.endog, model, self.test_sample, **spec)
        self.models.append(fcast)

    def select(self, measure='mae'):
        """Select the best forecast based on criteria provided by the user."""
        measure_vals = np.zeros(len(self.models))
        for mod in range(len(self.models)):
            measure_vals[mod] = getattr(self.models[mod], measure)
        min_measure = measure_vals.min()
        model = np.where(measure_vals == min_measure)
        # print(self.models[0][0])
        return self.models[model[0][0]]
