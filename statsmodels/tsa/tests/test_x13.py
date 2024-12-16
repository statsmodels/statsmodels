from statsmodels.compat.pandas import MONTH_END

import numpy as np
import pandas as pd
import pytest

from statsmodels.datasets import co2, macrodata
from statsmodels.tsa.x13 import (
    _find_x12,
    x13_arima_analysis,
    x13_arima_select_order,
)

x13path = _find_x12()

dta = macrodata.load_pandas().data
index = pd.period_range(start="1959Q1", end="2009Q3", freq="Q")
dta.index = index
quarterly_data = dta.dropna()

dta = co2.load_pandas().data
dta["co2"] = dta.co2.interpolate()
monthly_data = dta.resample(MONTH_END)
# change in pandas 0.18 resample is deferred object
if not isinstance(monthly_data, (pd.DataFrame, pd.Series)):
    monthly_data = monthly_data.mean()

monthly_start_data = dta.resample("MS")
if not isinstance(monthly_start_data, (pd.DataFrame, pd.Series)):
    monthly_start_data = monthly_start_data.mean()

data = (
    monthly_data,
    monthly_start_data,
    monthly_data.co2,
    monthly_start_data.co2,
    quarterly_data.realgdp,
    quarterly_data[["realgdp"]],
)
ids = (
    "monthly",
    "monthly_start",
    "monthly_co2",
    "monthly_start_co2",
    "series",
    "dataframe",
)


@pytest.fixture(params=data, ids=ids)
def dataset(request):
    return request.param


@pytest.mark.parametrize("use_numpy", [True, False])
def test_x13_arima_select_order(dataset, use_numpy):
    if use_numpy:
        index = dataset.index
        dataset = np.squeeze(np.asarray(dataset))
        start = index[0]
        if isinstance(index, pd.DatetimeIndex):
            freq = index.inferred_freq
        elif isinstance(index, pd.PeriodIndex):
            start = start.to_timestamp()
            freq = index.freq
        else:
            raise NotImplementedError()
        assert freq is not None
    else:
        freq = start = None
    res = x13_arima_select_order(dataset, start=start, freq=freq)
    assert isinstance(res.order, tuple)
    assert isinstance(res.sorder, tuple)


@pytest.mark.matplotlib
def test_x13_arima_plot(dataset):
    res = x13_arima_analysis(dataset)
    res.plot()


def test_x13_arima_plot_no_pandas(dataset):
    res = x13_arima_analysis(dataset)
    res.plot()
