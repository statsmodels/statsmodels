import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.seasonal import MSTL


@pytest.fixture(scope="module")
def data():
    t = np.arange(1, 1000)
    trend = 0.0001 * t**2 + 100
    daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / (24 * 7))
    y = trend + daily_seasonality + weekly_seasonality
    return y


@pytest.fixture(scope="module")
def data_pd(data):
    ts = pd.date_range(start="2000-01-01", periods=len(data), freq="H")
    return pd.Series(data=data, index=ts)


def test_return_pandas_series_when_input_pandas_and_len_periods_one(data_pd):
    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)


def test_seasonal_is_datafame_when_input_pandas_and_multiple_periods(data_pd):
    mod = MSTL(endog=data_pd, periods=(3, 5))
    res = mod.fit()
    assert isinstance(res.seasonal, pd.DataFrame)


@pytest.mark.parametrize(
    "data, periods, windows, expected",
    [
        (data, 3, None, 1),
        (data, (3, 6), None, 2),
        (data, (3, 6, 1e6), None, 2),
    ],
    indirect=["data"],
)
def test_number_of_seasonal_components(data, periods, windows, expected):
    mod = MSTL(endog=data, periods=periods, windows=windows)
    res = mod.fit()
    n_seasonal_components = (
        res.seasonal.shape[1] if res.seasonal.ndim > 1 else res.seasonal.ndim
    )
    assert n_seasonal_components == expected


@pytest.mark.parametrize(
    "periods, windows",
    [((3, 5), 1), (7, (3, 5))],
)
def test_raise_value_error_when_periods_and_windows_diff_lengths(
    periods, windows
):
    with pytest.raises(
        ValueError, match="Periods and windows must have same length"
    ):
        MSTL(endog=[1, 2, 3, 4, 5], periods=periods, windows=windows)


@pytest.mark.parametrize(
    "data, lmbda",
    [(data, 0.1), (data, 1), (data, -3.0), (data, "auto")],
    indirect=["data"],
)
def test_fit_with_box_cox(data, lmbda):
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda=lmbda)
    mod.fit()


def test_auto_fit_with_box_cox(data):
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda="auto")
    mod.fit()
    assert hasattr(mod, "est_lmbda")
    assert isinstance(mod.est_lmbda, float)


def test_stl_kwargs_smoke(data):
    stl_kwargs = {
        "period": 12,
        "seasonal": 15,
        "trend": 17,
        "low_pass": 15,
        "seasonal_deg": 0,
        "trend_deg": 1,
        "low_pass_deg": 1,
        "seasonal_jump": 2,
        "trend_jump": 2,
        "low_pass_jump": 3,
        "robust": False,
        "inner_iter": 3,
        "outer_iter": 3,
    }
    periods = (5, 6, 7)
    mod = MSTL(
        endog=data, periods=periods, lmbda="auto", stl_kwargs=stl_kwargs
    )
    mod.fit()


@pytest.mark.matplotlib
def test_plot(data, data_pd, close_figures):
    mod = MSTL(endog=data, periods=5)
    res = mod.fit()
    res.plot()

    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    res.plot()
