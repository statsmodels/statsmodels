"""unit test for automatic sarimax forecasting."""
import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.automatic import sarimax
from statsmodels import datasets
from numpy.testing import assert_equal


current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')


def test_smoke_non_stepwise():
    """Test function for non-stepwise auto_order."""
    intercept, p, d, q = sarimax.auto_order(macrodata['infl'], d=0)
# p, q = sarimax.auto_order(macrodata.infl, d=0, enforce_stationarity=False)
    desired_intercept = False
    desired_p = 2
    desired_d = 0
    desired_q = 2
    assert_equal(intercept, desired_intercept)
    assert_equal(p, desired_p)
    assert_equal(d, desired_d)
    assert_equal(q, desired_q)


def test_smoke_stepwise():
    """Test function for stepwise auto_order."""
    # p, q = sarimax.auto_order(
    #             macrodata['infl'], stepwise=True, enforce_stationarity=False)
    intercept, p, d, q = sarimax.auto_order(macrodata['infl'], stepwise=True)
    desired_intercept = False
    desired_p = 2
    desired_d = 0
    desired_q = 2
    assert_equal(intercept, desired_intercept)
    assert_equal(p, desired_p)
    assert_equal(d, desired_d)
    assert_equal(q, desired_q)


def test_auto_order_cpi():
    """Test function for auto_order against auto.arima."""
    intercept, p, d, q = sarimax.auto_order(macrodata['cpi'], d=2)
    desired_p = 1
    desired_d = 2
    desired_q = 2
    assert_equal(p, desired_p)
    assert_equal(d, desired_d)
    assert_equal(q, desired_q)

def test_auto_order_infl():
    """Test function for auto_order against auto.arima."""
    intercept, p, d, q = sarimax.auto_order(macrodata['infl'], stepwise=True,
                                            d=1)
    desired_p = 2
    desired_d = 1
    desired_q = 2
    assert_equal(p, desired_p)
    assert_equal(d, desired_d)
    assert_equal(q, desired_q)
