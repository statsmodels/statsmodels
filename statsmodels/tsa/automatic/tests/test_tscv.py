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
dataset_cpi = macrodata['cpi']

curdir = os.path.dirname(os.path.abspath(__file__))
new_data = pd.read_csv(os.path.join(curdir, 'CPIAPPNS.csv'))
new_data.index = pd.PeriodIndex(start='04-2010', end='03-2018', freq='M')
new_data = new_data['CPIAPPNS']

curdir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(curdir, "results/result_tscv.csv")
data = pd.read_csv(data_file, header=None)

curdir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(curdir, "results/result_tscv_cpi.csv")
data_cpi = pd.read_csv(data_file, header=None)

curdir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(curdir, "results/result_tscv_CPIAPPNS.csv")
data_CPIAPPNS = pd.read_csv(data_file, header=None)


def forecast_fixed_ar1(endog, horizon=1):
    """Forecast Function."""
    mod = sm.tsa.SARIMAX(endog, order=(1, 0, 0), trend='c',
                         concentrate_scale=True)
    res = mod.filter([0.75, 0.5])
    return res.forecast(horizon)


def test_tscv_unit_1():
    """Test Function."""
    e = tscv.evaluate(dataset, forecast_fixed_ar1, roll_window=30)
    for i in range(30, 203):
        assert_allclose(float(data.iloc[i, 1]), e[i])


def test_tscv_unit_2():
    """Test Function."""
    e = tscv.evaluate(dataset_cpi, forecast_fixed_ar1, roll_window=30)
    for i in range(30, 203):
        assert_allclose(float(data_cpi.iloc[i, 1]), e[i])


def test_tscv_unit_3_CPIAPPNS():
    """Test Function."""
    e = tscv.evaluate(new_data, forecast_fixed_ar1, roll_window=30)
    for i in range(30, 96):
        assert_allclose(float(data_CPIAPPNS.iloc[i, 1]), e[i])
