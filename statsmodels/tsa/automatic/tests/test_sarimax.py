"""unit test for automatic sarimax forecasting."""
import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.automatic import sarimax
from statsmodels import datasets


current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')

def test_non_stepwise():
    """Auto order function for SARIMAX models."""
    p, q = sarimax.auto_order(macrodata.infl, d=0)
    # p, q = sarimax.auto_order(macrodata.infl, d=0, enforce_stationarity=False)
    desired_p = 2
    desired_q = 2
    assert_equal(p, desired_p)
    assert_equal(q, desired_q)

