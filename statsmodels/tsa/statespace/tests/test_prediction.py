"""
Tests for prediction of state space models

Author: Chad Fulton
License: Simplified-BSD
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd

import warnings
from statsmodels.tsa.statespace import sarimax
from numpy.testing import assert_equal, assert_allclose, assert_raises
from nose.exc import SkipTest


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
