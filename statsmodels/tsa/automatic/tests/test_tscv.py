"""Unit test for Time Series Cross Validation."""
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels import datasets
from statsmodels.tsa.automatic import tscv
from numpy.testing import assert_equal, assert_allclose

macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
dataset = macrodata['infl']

curdir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(curdir, "results/result_tscv.csv")
data = pd.read_csv(data_file, header=None)


def forecast_fixed_ar1(endog, horizon=1):
    """Test Function."""
    mod = sm.tsa.SARIMAX(endog, order=(1, 0, 0), trend='c',
                         concentrate_scale=True)
    res = mod.filter([0.75, 0.5])
    return res.forecast(horizon)


def test_tscv_unit():
    e = tscv.evaluate(dataset, forecast_fixed_ar1, roll_window=30)
    for i in range(30, 203):
        assert_allclose(float(data.iloc[i, 1]), e[i])
