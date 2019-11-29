"""
Tests for prediction of state space models

Author: Chad Fulton
License: Simplified-BSD
"""


import numpy as np
import pandas as pd

from statsmodels.tsa.statespace import sarimax
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_


def test_predict_dates():
    index = pd.date_range(start='1950-01-01', periods=11, freq='D')
    np.random.seed(324328)
    endog = pd.Series(np.random.normal(size=10), index=index[:-1])

    # Basic test
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res = mod.filter(mod.start_params)

    # In-sample prediction should have the same index
    pred = res.predict()
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[:-1].values)
    # Out-of-sample forecasting should extend the index appropriately
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])

    # Simple differencing in the SARIMAX model should eliminate dates of
    # series eliminated due to differencing
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter(mod.start_params)
    pred = res.predict()
    # In-sample prediction should lose the first index value
    assert_equal(mod.nobs, endog.shape[0] - 1)
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[1:-1].values)
    # Out-of-sample forecasting should still extend the index appropriately
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])

    # Simple differencing again, this time with a more complex differencing
    # structure
    mod = sarimax.SARIMAX(endog, order=(1, 2, 0), seasonal_order=(0, 1, 0, 4),
                          simple_differencing=True)
    res = mod.filter(mod.start_params)
    pred = res.predict()
    # In-sample prediction should lose the first 6 index values
    assert_equal(mod.nobs, endog.shape[0] - (4 + 2))
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[4 + 2:-1].values)
    # Out-of-sample forecasting should still extend the index appropriately
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])


def test_memory_no_predicted():
    # Tests for forecasts with memory_no_predicted is set
    endog = [0.5, 1.2, 0.4, 0.6]

    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res1 = mod.filter([0.5, 1.])
    mod.ssm.memory_no_predicted = True
    res2 = mod.filter([0.5, 1.])

    # Make sure we really didn't store all of the values in res2
    assert_equal(res1.predicted_state.shape, (1, 5))
    assert_(res2.predicted_state is None)
    assert_equal(res1.predicted_state_cov.shape, (1, 1, 5))
    assert_(res2.predicted_state_cov is None)

    # Check that we can't do dynamic in-sample prediction
    assert_raises(ValueError, res2.predict, dynamic=True)
    assert_raises(ValueError, res2.get_prediction, dynamic=True)

    # Make sure the point forecasts are the same
    assert_allclose(res1.forecast(10), res2.forecast(10))

    # Make sure the confidence intervals are the same
    fcast1 = res1.get_forecast(10)
    fcast2 = res1.get_forecast(10)

    assert_allclose(fcast1.summary_frame(), fcast2.summary_frame())
