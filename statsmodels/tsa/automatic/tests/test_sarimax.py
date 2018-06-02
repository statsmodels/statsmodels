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


def test_non_stepwise():
    """test function for non-stepwise auto_order."""
    p, q, P, Q = sarimax.auto_order(macrodata['infl'], d=0)
# p, q = sarimax.auto_order(macrodata.infl, d=0, enforce_stationarity=False)
    desired_p = 2
    desired_q = 2
    desired_P = 0
    desired_Q = 0
    assert_equal(p, desired_p)
    assert_equal(q, desired_q)
    assert_equal(P, desired_P)
    assert_equal(Q, desired_Q)


def test_stepwise():
    """test function for stepwise auto_order."""
    # p, q = sarimax.auto_order(
    #             macrodata['infl'], stepwise=True, enforce_stationarity=False)
    p, q = sarimax.auto_order(macrodata['infl'], stepwise=True)
    desired_p = 2
    desired_q = 2
    assert_equal(p, desired_p)
    assert_equal(q, desired_q)
