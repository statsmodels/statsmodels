import contextlib
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms
from statsmodels.tsa.tsatools import freq_to_period


def _expect_long_period_warning(periods, nobs):
    # MSTL warns and drops the periods that are at least half the sample length
    if np.max(periods) >= nobs / 2:
        return pytest.warns(UserWarning, match="A period.s. is larger than half")
    return contextlib.nullcontext()


@pytest.fixture
def mstl_results():
    cur_dir = Path(__file__).parent.resolve()
    file_path = cur_dir / "results/mstl_test_results.csv"
    return pd.read_csv(file_path)


@pytest.fixture
def data_pd():
    cur_dir = Path(__file__).parent.resolve()
    file_path = cur_dir / "results/mstl_elec_vic.csv"
    return pd.read_csv(file_path, index_col=["ds"], parse_dates=["ds"])


@pytest.fixture
def data(data_pd):
    return data_pd["y"].values


def test_return_pandas_series_when_input_pandas_and_len_periods_one(data_pd):
    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)


def test_numpy_integer_periods_and_windows(data):
    res = MSTL(endog=data, periods=5, windows=7).fit()
    res_np = MSTL(endog=data, periods=np.int64(5), windows=np.int64(7)).fit()
    assert_allclose(res_np.seasonal, res.seasonal)
    assert_allclose(res_np.trend, res.trend)


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
    with _expect_long_period_warning(periods, len(data)):
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
def test_raise_value_error_when_periods_and_windows_diff_lengths(periods, windows):
    with pytest.raises(ValueError, match="Periods and windows must have same length"):
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
    mod = MSTL(endog=data, periods=periods, lmbda="auto", stl_kwargs=stl_kwargs)
    mod.fit()


def test_stl_kwargs_not_consumed(data):
    # inner_iter and outer_iter were popped from the model's kwargs on the
    # first fit, so a second fit silently used the STL defaults
    stl_kwargs = {"inner_iter": 1, "outer_iter": 0, "period": 12}
    mod = MSTL(endog=data, periods=(24, 24 * 7), stl_kwargs=stl_kwargs)
    res1 = mod.fit()
    res2 = mod.fit()
    assert stl_kwargs == {"inner_iter": 1, "outer_iter": 0, "period": 12}
    assert_allclose(res2.trend, res1.trend)
    assert_allclose(res2.seasonal, res1.seasonal)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot(data, data_pd, close_figures):
    mod = MSTL(endog=data, periods=5)
    res = mod.fit()
    res.plot()

    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    res.plot()


def test_output_similar_to_R_implementation(data_pd, mstl_results):
    mod = MSTL(
        endog=data_pd,
        periods=(24, 24 * 7),
        stl_kwargs={
            "seasonal_deg": 0,
            "seasonal_jump": 1,
            "trend_jump": 1,
            "trend_deg": 1,
            "low_pass_jump": 1,
            "low_pass_deg": 1,
            "inner_iter": 2,
            "outer_iter": 0,
        },
    )
    res = mod.fit()

    expected_observed = mstl_results["Data"]
    expected_trend = mstl_results["Trend"]
    expected_seasonal = mstl_results[["Seasonal24", "Seasonal168"]]
    expected_resid = mstl_results["Remainder"]

    assert_allclose(res.observed, expected_observed)
    assert_allclose(res.trend, expected_trend)
    assert_allclose(res.seasonal, expected_seasonal)
    assert_allclose(res.resid, expected_resid)


@pytest.mark.slow
@pytest.mark.parametrize(
    "data, periods_ordered, windows_ordered, periods_not_ordered, "
    "windows_not_ordered",
    [
        (data, (12, 24, 24 * 7), (11, 15, 19), (12, 24 * 7, 24), (11, 19, 15)),
        (
            data,
            (12, 24, 24 * 7 * 1e6),
            (11, 15, 19),
            (12, 24 * 7 * 1e6, 24),
            (11, 19, 15),
        ),
        (data, (12, 24, 24 * 7), None, (12, 24 * 7, 24), None),
    ],
    indirect=["data"],
)
def test_output_invariant_to_period_order(
    data,
    periods_ordered,
    windows_ordered,
    periods_not_ordered,
    windows_not_ordered,
):
    with _expect_long_period_warning(periods_ordered, len(data)):
        mod1 = MSTL(endog=data, periods=periods_ordered, windows=windows_ordered)
    res1 = mod1.fit()
    with _expect_long_period_warning(periods_not_ordered, len(data)):
        mod2 = MSTL(
            endog=data, periods=periods_not_ordered, windows=windows_not_ordered
        )
    res2 = mod2.fit()

    assert_equal(res1.observed, res2.observed)
    assert_equal(res1.trend, res2.trend)
    assert_equal(res1.seasonal, res2.seasonal)
    assert_equal(res1.resid, res2.resid)


def test_period_from_inferred_freq():
    # A DatetimeIndex without a freq gives the period of its inferred freq,
    # without the warning pandas 3.1 gives when inferred_freq is read
    rng = np.random.default_rng(20261121)
    dates = pd.date_range("2000-01-01", periods=72, freq="MS")
    t = np.arange(72)
    y = 10 + np.sin(2 * np.pi * t / 12) + 0.1 * rng.standard_normal(72)
    endog = pd.Series(y, index=list(dates))
    assert endog.index.freq is None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mod = MSTL(endog)
    assert [str(w.message) for w in record] == []
    expected = MSTL(y, periods=12)
    assert_allclose(mod.fit().seasonal, expected.fit().seasonal, rtol=1e-10)


def _monthly_series(nobs=72, seed=20261128):
    rng = np.random.default_rng(seed)
    t = np.arange(nobs)
    y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 12) + 0.5 * np.cos(2 * np.pi * t / 4)
    return y + 0.2 * rng.standard_normal(nobs)


def test_infer_period_from_index():
    # without periods, the period is that of the index frequency, as given by
    # freq_to_period: 12 for monthly data
    index = pd.date_range("2000-01-01", periods=72, freq="MS")
    y = pd.Series(_monthly_series(), index=index)
    assert freq_to_period(index.freqstr) == 12
    mod = MSTL(y)
    assert list(mod.periods) == [12]
    res = mod.fit()
    expected = MSTL(y.to_numpy(), periods=12).fit()
    assert_allclose(res.seasonal.to_numpy(), expected.seasonal, rtol=1e-12)
    assert_allclose(res.trend.to_numpy(), expected.trend, rtol=1e-12)


@pytest.mark.parametrize("kind", ["ndarray", "irregular-index"])
def test_infer_period_raises(kind):
    y = _monthly_series()
    if kind == "irregular-index":
        days = np.cumsum(np.r_[0, np.arange(1, y.shape[0]) % 3 + 28])
        index = pd.Timestamp("2000-01-01") + pd.to_timedelta(days, unit="D")
        y = pd.Series(y, index=index)
        assert y.index.inferred_freq is None
    with pytest.raises(ValueError, match="Unable to determine period from endog"):
        MSTL(y)


def test_empty_windows_raises():
    with pytest.raises(ValueError, match="Periods and windows must have same length"):
        MSTL(_monthly_series(), periods=(4, 12), windows=[])


def test_2d_endog_raises():
    y = _monthly_series()
    with pytest.raises(ValueError, match="y must be a 1d array"):
        MSTL(np.c_[y, y], periods=12)


def test_str():
    # the periods are sorted together with their windows
    mod = MSTL(
        _monthly_series(), periods=(12, 4), windows=(15, 11), lmbda=0.5, iterate=3
    )
    expected = "MSTL(endog, periods=(4, 12), windows=(11, 15), lmbda=0.5, iterate=3)"
    assert str(mod) == expected
    # the default windows are 7 + 4 * i for i = 1, ..., number of periods
    mod = MSTL(_monthly_series(), periods=(12, 4))
    assert list(mod.periods) == [4, 12]
    assert list(mod.windows) == [11, 15]
    assert str(mod).startswith(f"MSTL(endog, periods={mod.periods}, windows=(11, 15),")


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_dataframe_seasonal(close_figures):
    # with several periods and pandas input the seasonal component is a
    # DataFrame, plotted as one panel per column between the trend and the
    # residual
    index = pd.date_range("2000-01-01", periods=72, freq="MS")
    res = MSTL(pd.Series(_monthly_series(), index=index), periods=(4, 12)).fit()
    assert isinstance(res.seasonal, pd.DataFrame)
    fig = res.plot()
    axes = fig.axes
    assert len(axes) == 5
    # the observed panel is titled with the name of the observed Series
    assert axes[0].get_title() == res.observed.name == "observed"
    assert axes[1].get_ylabel() == "Trend"
    for ax, col in zip(axes[2:4], ["seasonal_4", "seasonal_12"], strict=True):
        assert ax.get_ylabel() == col.capitalize()
        ydata = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert_allclose(ydata, res.seasonal[col].to_numpy(), rtol=1e-12)
    assert axes[4].get_ylabel() == "Resid"


@pytest.mark.parametrize("form", SERIES_FORMS + ("int",))
def test_input_forms(form):
    # endog is documented as array_like: every form of a 1-d series gives the
    # values of the ndarray, and pandas input gives pandas output with its
    # index
    y = _monthly_series()
    forms = series_forms(y)
    forms["int"] = np.round(10 * y).astype(np.int64)
    y_form = forms[form]
    y_float = np.asarray(y_form, dtype=float).squeeze()
    res = MSTL(y_form, periods=(4, 12)).fit()
    expected = MSTL(y_float, periods=(4, 12)).fit()
    for attr in ("observed", "seasonal", "trend", "resid", "weights"):
        actual = getattr(res, attr)
        assert_allclose(np.asarray(actual), getattr(expected, attr), rtol=1e-12)
        if isinstance(y_form, (pd.Series, pd.DataFrame)):
            assert isinstance(actual, (pd.Series, pd.DataFrame))
            pd.testing.assert_index_equal(actual.index, y_form.index)
        else:
            assert isinstance(actual, np.ndarray)
    if isinstance(y_form, (pd.Series, pd.DataFrame)):
        assert list(res.seasonal.columns) == ["seasonal_4", "seasonal_12"]
    # the components make up the series
    seasonal = np.asarray(res.seasonal).sum(axis=1)
    combined = np.asarray(res.trend) + seasonal + np.asarray(res.resid)
    assert_allclose(combined, y_float, rtol=1e-12)
    assert_allclose(np.asarray(res.observed), y_float, rtol=1e-15)


@pytest.mark.parametrize("container", ["series", "dataframe"])
@pytest.mark.parametrize("index_type", ["period", "datetime"])
@pytest.mark.parametrize(
    "freq, datetime_freq, period",
    [("M", "MS", 12), ("Q", "QS", 4), ("W", "W", 52), ("D", "D", 7)],
)
def test_periods_from_index(freq, datetime_freq, period, index_type, container):
    # a PeriodIndex has no inferred_freq, so without periods MSTL raised
    # "Unable to determine period from endog". The period is that of the
    # frequency of the index, as in seasonal_decompose, and the result is the
    # decomposition with that period. A DatetimeIndex gives the period of its
    # inferred frequency, as before.
    y = np.random.default_rng(20261121).standard_normal(3 * period)
    if index_type == "period":
        index = pd.period_range("2000-01-01", periods=y.shape[0], freq=freq)
    else:
        index = pd.date_range("2000-01-01", periods=y.shape[0], freq=datetime_freq)
    if container == "series":
        endog = pd.Series(y, index=index)
    else:
        endog = pd.DataFrame({"y": y}, index=index)
    mod = MSTL(endog)
    assert list(mod.periods) == [period]
    res = mod.fit()
    expected = MSTL(y, periods=period).fit()
    assert_allclose(res.seasonal, expected.seasonal, rtol=1e-12)
    assert_allclose(res.trend, expected.trend, rtol=1e-12)
    assert res.seasonal.index.equals(index)


@pytest.mark.parametrize(
    "convert", [np.int64, np.array, float], ids=["int64", "0d-array", "float"]
)
def test_int_like_periods_and_windows(data, convert):
    # A 0-d integer array or an integral float is a single period or window,
    # as an int is. They raised a TypeError. With one period, MSTL is STL.
    res = MSTL(endog=data, periods=convert(5), windows=convert(7)).fit()
    assert res.seasonal.ndim == 1
    expected = STL(data, period=5, seasonal=7).fit()
    assert_allclose(res.seasonal, expected.seasonal, rtol=1e-12)
    assert_allclose(res.trend, expected.trend, rtol=1e-12)


@pytest.mark.parametrize("value", [True, np.True_], ids=["True", "np.True_"])
@pytest.mark.parametrize("arg", ["periods", "windows"])
def test_bool_periods_and_windows(data, value, arg):
    # A bool is neither a period nor a window
    kwargs = {"periods": 5, "windows": 7, arg: value}
    with pytest.raises(TypeError):
        MSTL(endog=data, **kwargs)
