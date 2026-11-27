from statsmodels.compat.pandas import MONTH_END

import datetime as dt
import re
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import statsmodels.datasets
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tools.testing import warns_or_none
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.exponential_smoothing import (
    ExponentialSmoothing,
)
from statsmodels.tsa.tests._contract import series_forms


@pytest.fixture(scope="module")
def data(request):
    rs = np.random.RandomState(987654321)
    err = rs.standard_normal(500)
    index = pd.date_range("1980-1-1", freq=MONTH_END, periods=500)
    fourier = Fourier(12, 1)
    terms = fourier.in_sample(index)
    det = np.squeeze(np.asarray(terms @ np.array([[2], [1]])))
    for i in range(1, 500):
        err[i] += 0.9 * err[i - 1] + det[i]
    return pd.Series(err, index=index)


def test_smoke(data):
    stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    res.forecast(37)
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res.stl, STL)
    assert isinstance(res.result, DecomposeResult)
    assert isinstance(res.model, ARIMA)
    assert hasattr(res.model_result, "forecast")
    assert res.result.nobs == 500


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_sharex(data, close_figures):
    stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    plt = res.result.plot()
    grouper_view = plt.axes[0].get_shared_x_axes()
    sibs = grouper_view.get_siblings(plt.axes[1])
    assert len(sibs) == 4


MODELS = [
    (ARIMA, {"order": (2, 0, 0), "trend": "c"}),
    (ExponentialSmoothing, {"trend": True}),
    (AutoReg, {"lags": 2}),
    (ETSModel, {}),
]
MODELS = MODELS[-1:]
IDS = [str(c[0]).split(".")[-1][:-2] for c in MODELS]


@pytest.mark.parametrize("config", MODELS, ids=IDS)
@pytest.mark.parametrize("horizon", [1, 7, 23])
def test_equivalence_forecast(data, config, horizon, close_figures):
    model, kwargs = config

    stl = STL(data)
    stl_fit = stl.fit()
    resids = data - stl_fit.seasonal
    mod = model(resids, **kwargs)
    fit_kwarg = {}
    if model is ETSModel:
        fit_kwarg["disp"] = False
        fit_kwarg["param_index"] = "integer"
    res = mod.fit(**fit_kwarg)
    stlf = STLForecast(data, model, model_kwargs=kwargs).fit(fit_kwargs=fit_kwarg)

    seasonal = np.asarray(stl_fit.seasonal)[-12:]
    seasonal = np.tile(seasonal, 1 + horizon // 12)
    fcast = res.forecast(horizon) + seasonal[:horizon]
    actual = stlf.forecast(horizon)
    assert_allclose(actual, fcast, rtol=1e-4)
    if not hasattr(res, "get_prediction"):
        return
    pred = stlf.get_prediction(data.shape[0], data.shape[0] + horizon - 1)
    assert isinstance(pred, PredictionResults)
    assert_allclose(pred.predicted_mean, fcast, rtol=1e-4)

    half = data.shape[0] // 2
    stlf.get_prediction(half, data.shape[0] + horizon - 1)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=True)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=half // 2)
    if hasattr(data, "index"):
        loc = data.index[half + half // 2]
        a = stlf.get_prediction(
            half, data.shape[0] + horizon - 1, dynamic=loc.strftime("%Y-%m-%d")
        )
        b = stlf.get_prediction(
            half, data.shape[0] + horizon - 1, dynamic=loc.to_pydatetime()
        )
        c = stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=loc)
        assert_allclose(a.predicted_mean, b.predicted_mean, rtol=1e-4)
        assert_allclose(a.predicted_mean, c.predicted_mean, rtol=1e-4)


def test_exceptions(data):
    class BadModel:
        def __init__(self, *args, **kwargs):
            pass

    with pytest.raises(AttributeError, match="model must expose"):
        STLForecast(data, BadModel)

    class NoForecast(BadModel):
        def fit(self, *args, **kwargs):
            return BadModel()

    with pytest.raises(AttributeError, match="The model's result"):
        STLForecast(data, NoForecast).fit()

    class BadResult:
        def forecast(self, *args, **kwargs):
            pass

    class FakeModel(BadModel):
        def fit(self, *args, **kwargs):
            return BadResult()

    with pytest.raises(AttributeError, match="The model result does not"):
        STLForecast(data, FakeModel).fit().summary()

    class BadResultSummary(BadResult):
        def summary(self, *args, **kwargs):
            return object()

    class FakeModelSummary(BadModel):
        def fit(self, *args, **kwargs):
            return BadResultSummary()

    with pytest.raises(TypeError, match="The model result's summary"):
        STLForecast(data, FakeModelSummary).fit().summary()


@pytest.fixture
def sunspots():
    df = statsmodels.datasets.sunspots.load_pandas().data
    df.index = np.arange(df.shape[0])
    return df.iloc[:, 0]


def test_get_prediction(sunspots):
    # GH7309
    stlf_model = STLForecast(
        sunspots, model=ARIMA, model_kwargs={"order": (2, 2, 0)}, period=11
    )
    # The default optimizer may stop short of convergence on some platforms
    with warns_or_none(ConvergenceWarning, match="Maximum Likelihood optimization"):
        stlf_res = stlf_model.fit()
    pred = stlf_res.get_prediction()
    assert pred.predicted_mean.shape == (309,)
    assert pred.var_pred_mean.shape == (309,)


@pytest.mark.parametrize("not_implemented", [True, False])
def test_no_var_pred(sunspots, not_implemented, close_figures):
    class DummyPred:
        def __init__(self, predicted_mean, row_labels):
            self.predicted_mean = predicted_mean
            self.row_labels = row_labels

            def f():
                raise NotImplementedError

            if not_implemented:
                self.forecast = property(f)

    class DummyRes:
        def __init__(self, res):
            self._res = res

        def forecast(self, *args, **kwargs):
            return self._res.forecast(*args, **kwargs)

        def get_prediction(self, *args, **kwargs):
            pred = self._res.get_prediction(*args, **kwargs)

            return DummyPred(pred.predicted_mean, pred.row_labels)

    class DummyMod:
        def __init__(self, y):
            self._mod = ARIMA(y)

        def fit(self, *args, **kwargs):
            res = self._mod.fit(*args, **kwargs)
            return DummyRes(res)

    stl_mod = STLForecast(sunspots, model=DummyMod, period=11)
    stl_res = stl_mod.fit()
    with pytest.warns(UserWarning, match="The variance of"):
        pred = stl_res.get_prediction()
    assert np.all(np.isnan(pred.var_pred_mean))


NOBS = 120
PERIOD = 12
AR_KWARGS = {"lags": 2, "trend": "c"}


@pytest.fixture(scope="module")
def monthly():
    # Integer-valued so that the int dtype form holds the same values
    rng = np.random.default_rng([20261120, 2])
    t = np.arange(NOBS)
    y = 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / PERIOD)
    return np.round(100 * (y + rng.standard_normal(NOBS)))


def _stl_form(y, form):
    if form == "ndarray":
        return y
    if form == "int":
        return y.astype(np.int64)
    return series_forms(y)[form]


def _stl_ar_by_hand(y):
    # STL followed by an AutoReg of the deseasonalized data, fit by OLS
    stl_res = STL(y, period=PERIOD).fit()
    ar_res = AutoReg(stl_res.trend + stl_res.resid, **AR_KWARGS).fit()
    return stl_res.seasonal, ar_res


def _seasonal_by_hand(seasonal, origin, locs):
    # The STL seasonal is used before origin. From origin on, the last full
    # period before origin repeats.
    locs = np.asarray(locs)
    repeated = seasonal[origin - PERIOD + (locs - origin) % PERIOD]
    in_sample = seasonal[np.minimum(locs, origin - 1)]
    return np.where(locs < origin, in_sample, repeated)


# list and tuple are left out: STLForecast.fit raises TypeError for them
@pytest.mark.parametrize("form", ["ndarray", "int", "series", "column", "dataframe"])
def test_forecast_assembled_by_hand(monthly, form):
    x = _stl_form(monthly, form)
    res = STLForecast(x, AutoReg, model_kwargs=AR_KWARGS, period=PERIOD).fit()
    seasonal, ar_res = _stl_ar_by_hand(monthly)
    steps = 2 * PERIOD + 5
    locs = np.arange(NOBS, NOBS + steps)
    expected = ar_res.forecast(steps) + _seasonal_by_hand(seasonal, NOBS, locs)
    fcast = res.forecast(steps)
    assert_allclose(np.asarray(fcast), expected, rtol=1e-10)
    if form in ("series", "dataframe"):
        assert isinstance(fcast, pd.Series)
        index = x.index
        expected_index = pd.date_range(
            index[-1] + index.freq, periods=steps, freq=index.freq
        )
        pd.testing.assert_index_equal(fcast.index, expected_index)
    else:
        assert isinstance(fcast, np.ndarray)


# column and dataframe are left out: get_prediction raises ValueError for
# 2-d endog when it builds a Series from it
@pytest.mark.parametrize("form", ["ndarray", "int", "series"])
@pytest.mark.parametrize("dynamic", [False, 6])
def test_get_prediction_assembled_by_hand(monthly, form, dynamic):
    x = _stl_form(monthly, form)
    res = STLForecast(x, AutoReg, model_kwargs=AR_KWARGS, period=PERIOD).fit()
    seasonal, ar_res = _stl_ar_by_hand(monthly)
    start, end = 100, NOBS + 17
    pred = res.get_prediction(start, end, dynamic=dynamic)
    ar_pred = ar_res.get_prediction(start, end, dynamic=dynamic)
    # Dynamic prediction repeats the seasonal from the dynamic origin
    origin = NOBS if dynamic is False else start + dynamic
    locs = np.arange(start, end + 1)
    expected = ar_pred.predicted_mean + _seasonal_by_hand(seasonal, origin, locs)
    assert_allclose(np.asarray(pred.predicted_mean), expected, rtol=1e-10)
    # The seasonal is treated as known, so the variance is the model's
    assert_allclose(
        np.asarray(pred.var_pred_mean), ar_pred.var_pred_mean, rtol=1e-10
    )
    if form == "series":
        expected_index = pd.date_range(
            x.index[start], periods=end - start + 1, freq=x.index.freq
        )
        pd.testing.assert_index_equal(pred.predicted_mean.index, expected_index)


def test_date_string_index(monthly):
    # An index of date strings is read as dates, so that dates can locate
    # the seasonal part of the prediction
    dates = pd.date_range("2000-01-01", periods=NOBS, freq="MS")
    y = pd.Series(monthly, index=dates.strftime("%Y-%m-%d"))
    # AutoReg also reads the strings as dates, which have no frequency
    with pytest.warns(ValueWarning, match="No frequency information"):
        res = STLForecast(y, AutoReg, model_kwargs=AR_KWARGS, period=PERIOD).fit()
    pred = res.get_prediction(
        pd.Timestamp("2008-05-01"),
        pd.Timestamp("2009-12-01"),
        dynamic=dt.datetime(2009, 1, 1),
    )
    # Observations 100 to 119, with dynamic prediction from 108
    seasonal, ar_res = _stl_ar_by_hand(monthly)
    ar_pred = ar_res.get_prediction(100, NOBS - 1, dynamic=8)
    locs = np.arange(100, NOBS)
    expected = ar_pred.predicted_mean + _seasonal_by_hand(seasonal, 108, locs)
    assert_allclose(np.asarray(pred.predicted_mean), expected, rtol=1e-10)


def test_unparseable_string_index(monthly):
    # An index that cannot be read as dates is replaced by positions
    y = pd.Series(monthly, index=[f"obs{i}" for i in range(NOBS)])
    # AutoReg warns that the index is unsupported, and pandas >= 2.0 warns
    # that it cannot infer a date format while the labels are tried as dates
    match = "unsupported index|Could not infer format"
    with pytest.warns(UserWarning, match=match) as record:
        res = STLForecast(y, AutoReg, model_kwargs=AR_KWARGS, period=PERIOD).fit()
    assert any(
        issubclass(w.category, ValueWarning) and "unsupported index" in str(w.message)
        for w in record
    )
    pred = res.get_prediction(100, NOBS - 1, dynamic=8)
    seasonal, ar_res = _stl_ar_by_hand(monthly)
    ar_pred = ar_res.get_prediction(100, NOBS - 1, dynamic=8)
    locs = np.arange(100, NOBS)
    expected = ar_pred.predicted_mean + _seasonal_by_hand(seasonal, 108, locs)
    assert_allclose(np.asarray(pred.predicted_mean), expected, rtol=1e-10)


def test_fit_options_and_properties(monthly):
    dates = pd.date_range("2000-01-01", periods=NOBS, freq="MS")
    y = pd.Series(monthly, index=dates)
    stl_kwargs = {"seasonal": 13, "robust": True}
    mod = STLForecast(y, AutoReg, model_kwargs=AR_KWARGS, **stl_kwargs)
    res = mod.fit(inner_iter=3, outer_iter=4, fit_kwargs={"cov_type": "HC0"})

    stl = STL(y, **stl_kwargs)
    stl_res = stl.fit(inner_iter=3, outer_iter=4)
    ar_res = AutoReg(stl_res.trend + stl_res.resid, **AR_KWARGS).fit(cov_type="HC0")
    # The period is read from the monthly index
    assert res.period == 12
    assert isinstance(res.stl, STL)
    assert res.stl.config == stl.config
    assert isinstance(res.result, DecomposeResult)
    assert_allclose(res.result.seasonal, stl_res.seasonal, rtol=1e-10)
    assert_allclose(res.result.trend, stl_res.trend, rtol=1e-10)
    assert isinstance(res.model, AutoReg)
    assert_allclose(res.model.endog, ar_res.model.endog, rtol=1e-10)
    assert_allclose(res.model_result.params, ar_res.params, rtol=1e-10)
    assert_allclose(res.model_result.bse, ar_res.bse, rtol=1e-10)

    smry = res.summary()
    assert isinstance(smry, Summary)
    title = ar_res.summary().tables[0].title
    assert smry.tables[0].title == "STL Decomposition and " + title
    config = smry.tables[-1].as_text()
    assert "STL Configuration" in config
    assert re.search(r"Period:\s+12\b", config)
    assert re.search(r"Seasonal:\s+13\b", config)
    assert re.search(r"Robust:\s+True\b", config)
    assert re.search(rf"Trend Length:\s+{stl.config['trend']}\b", config)


def test_summary_stl_configuration(data):
    # The STL configuration table lists the period, seasonal and robust
    # settings on the left and the trend and low pass settings on the right,
    # with the values of STL.config. The left column has one entry fewer, so
    # its last row is blank
    res = STLForecast(data, AutoReg, model_kwargs={"lags": 2}).fit()
    config = res.stl.config
    left = ["period", "seasonal", "seasonal_deg", "seasonal_jump", "robust"]
    right = ["trend", "trend_deg", "trend_jump"]
    right += ["low_pass", "low_pass_deg", "low_pass_jump"]
    table = res.summary().tables[-1]
    rows = [[str(cell.data).strip() for cell in row] for row in table]
    assert len(rows) == len(right)
    for row, key in zip(rows, left + [None], strict=True):
        if key is None:
            assert row[:2] == ["", ""]
        else:
            assert row[0].lower() == key.replace("_", " ") + ":"
            assert row[1] == str(config[key])
    for row, key in zip(rows, right, strict=True):
        assert row[2].lower().startswith(key.replace("_", " "))
        assert row[3] == str(config[key])


def _seasonal_data():
    rng = np.random.default_rng(20261120)
    t = np.arange(120)
    y = 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 12)
    return y + rng.standard_normal(120)


def _stl_ar_reference(y, start, steps):
    # STL of the values, an AR(2) with a constant fit to the deseasonalized
    # values, and the seasonal component of the last period repeated out of
    # sample. Returns the forecasts and the predictions from start
    stl_res = STL(y, period=12).fit()
    seasonal = np.asarray(stl_res.seasonal)
    deseasonalized = np.asarray(stl_res.trend + stl_res.resid)
    ar_res = AutoReg(deseasonalized, lags=2, trend="c").fit()
    oos_seasonal = np.resize(seasonal[-12:], steps)
    forecast = ar_res.forecast(steps) + oos_seasonal
    end = y.shape[0] + steps - 1
    predicted = ar_res.predict(start, end)
    predicted = predicted + np.r_[seasonal[start:], oos_seasonal]
    return forecast, predicted


@pytest.mark.parametrize("container", [list, tuple])
def test_sequence_endog(container):
    y = _seasonal_data()
    endog = container(y.tolist())
    model_kwargs = {"lags": 2, "trend": "c"}
    stlf = STLForecast(endog, AutoReg, model_kwargs=model_kwargs, period=12)
    res = stlf.fit()
    forecast, predicted = _stl_ar_reference(y, 100, 14)
    assert_allclose(res.forecast(14), forecast, rtol=1e-10)
    pred = res.get_prediction(100, 133)
    assert_allclose(pred.predicted_mean, predicted, rtol=1e-10)


@pytest.mark.parametrize("form", ["column", "dataframe"])
def test_one_column_endog(form):
    # An (n, 1) column or a one-column DataFrame gives the results of the
    # 1-d data
    y = _seasonal_data()
    if form == "column":
        endog = y[:, None]
    else:
        index = pd.date_range("2000-01-01", periods=120, freq="MS")
        endog = pd.DataFrame({"y": y}, index=index)
    model_kwargs = {"lags": 2, "trend": "c"}
    stlf = STLForecast(endog, AutoReg, model_kwargs=model_kwargs, period=12)
    res = stlf.fit()
    forecast, predicted = _stl_ar_reference(y, 100, 14)
    assert_allclose(res.forecast(14), forecast, rtol=1e-10)
    pred = res.get_prediction(100, 133)
    assert_allclose(pred.predicted_mean, predicted, rtol=1e-10)


class _ValuesAR:
    # An AR(2) with a constant fit to the values of endog, without its index
    def __init__(self, endog):
        self._model = AutoReg(np.asarray(endog), lags=2, trend="c")

    def fit(self):
        return self._model.fit()


def test_non_date_index_no_warning():
    # An index that does not hold dates is replaced by a RangeIndex without
    # the warning that pandas gives when it cannot infer a date format. The
    # model ignores the index, so any warning comes from STLForecast
    y = _seasonal_data()
    endog = pd.Series(y, index=[f"obs{i}" for i in range(120)])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = STLForecast(endog, _ValuesAR, period=12).fit()
        forecast = res.forecast(14)
        pred = res.get_prediction(100, 133)
    assert [str(w.message) for w in record] == []
    expected_forecast, expected_predicted = _stl_ar_reference(y, 100, 14)
    assert_allclose(forecast, expected_forecast, rtol=1e-10)
    assert_allclose(pred.predicted_mean, expected_predicted, rtol=1e-10)


@pytest.mark.parametrize("kind", ["multiindex", "tuples", "bool"])
def test_index_values_not_dates(kind):
    # pandas raises a TypeError, not a ValueError, for index values that
    # cannot be dates; the index is replaced by a RangeIndex
    y = _seasonal_data()
    loc = np.arange(120)
    if kind == "multiindex":
        index = pd.MultiIndex.from_arrays([loc // 12, loc % 12])
    elif kind == "tuples":
        index = pd.Index([(i // 12, i % 12) for i in loc], tupleize_cols=False)
    else:
        index = pd.Index(loc % 2 == 0)
    endog = pd.Series(y, index=index)
    res = STLForecast(endog, _ValuesAR, period=12).fit()
    expected_forecast, expected_predicted = _stl_ar_reference(y, 100, 14)
    assert_allclose(res.forecast(14), expected_forecast, rtol=1e-10)
    pred = res.get_prediction(100, 133)
    assert_allclose(pred.predicted_mean, expected_predicted, rtol=1e-10)


@pytest.mark.parametrize("index_type", ["datetime", "period", "no_freq"])
@pytest.mark.parametrize("date_type", ["str", "datetime", "timestamp", "datetime64"])
def test_forecast_date_steps(index_type, date_type):
    # A date gives the forecasts up to and including it: those of the model
    # fit to the deseasonalized data plus the seasonal component of the last
    # period repeated. The model infers the frequency of an index without
    # one, so the number of steps is taken from its forecast
    rng = np.random.default_rng(20261120)
    t = np.arange(120)
    y = 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 12)
    y = y + rng.standard_normal(120)
    if index_type == "period":
        index = pd.period_range("2000-01", periods=120, freq="M")
    else:
        index = pd.date_range("2000-01-01", periods=120, freq="MS")
        if index_type == "no_freq":
            index = pd.DatetimeIndex(index.to_list())
    date = pd.Timestamp("2010-03-01")
    if date_type == "str":
        date = "2010-03-01"
    elif date_type == "datetime":
        date = date.to_pydatetime()
    elif date_type == "datetime64":
        date = date.to_datetime64()
    model_kwargs = {"order": (1, 0, 0), "trend": "c"}
    stlf = STLForecast(
        pd.Series(y, index=index), ARIMA, model_kwargs=model_kwargs, period=12
    )
    if index_type == "no_freq":
        with pytest.warns(ValueWarning, match="No frequency information"):
            res = stlf.fit()
    else:
        res = stlf.fit()
    forecast = res.forecast(date)

    stl_res = STL(y, period=12).fit()
    seasonal = np.asarray(stl_res.seasonal)
    deseasonalized = np.asarray(stl_res.trend + stl_res.resid)
    ar = ARIMA(deseasonalized, order=(1, 0, 0), trend="c")
    ar_res = ar.filter(np.asarray(res.model_result.params))
    expected = ar_res.forecast(3) + seasonal[-12:-9]
    assert_allclose(forecast, expected, rtol=1e-10)
    if index_type == "period":
        assert forecast.index[-1] == pd.Period("2010-03", freq="M")
    else:
        assert forecast.index[-1] == pd.Timestamp("2010-03-01")


def test_get_prediction_dynamic_key_covers_several_dates():
    # A dynamic date key that covers several dates (all of April 2001 in daily
    # data) gives a slice of locations, which raised a TypeError. Dynamic
    # prediction begins at the first date of the key.
    rng = np.random.default_rng(20261121)
    index = pd.date_range("2000-01-01", periods=500, freq="D")
    seasonal = np.tile([1.0, 2.0, 0.0, -1.0, -2.0, 0.5, -0.5], 72)[:500]
    y = pd.Series(seasonal + rng.standard_normal(500).cumsum(), index=index)
    mod = STLForecast(y, ARIMA, model_kwargs={"order": (1, 0, 0)}, period=7)
    res = mod.fit(fit_kwargs={"method": "yule_walker"})
    loc = 366 + 31 + 28 + 31
    assert index[loc] == pd.Timestamp("2001-04-01")
    pred = res.get_prediction(start=10, dynamic="2001-04")
    expected = res.get_prediction(start=10, dynamic=loc - 10)
    assert_allclose(pred.predicted_mean, expected.predicted_mean, rtol=1e-12)
    assert_allclose(pred.var_pred_mean, expected.var_pred_mean, rtol=1e-12)
    pd.testing.assert_index_equal(pred.predicted_mean.index, index[10:])


@pytest.mark.parametrize(
    "steps",
    [np.int64(17), np.array(17), 17.0],
    ids=["int64", "0d-array", "float"],
)
@pytest.mark.parametrize("form", ["ndarray", "series"])
def test_forecast_int_like_steps(monthly, form, steps):
    # A 0-d integer array and an integral float are a number of steps, as an
    # int is. They are passed to the model as an int.
    x = _stl_form(monthly, form)
    res = STLForecast(x, AutoReg, model_kwargs=AR_KWARGS, period=PERIOD).fit()
    seasonal, ar_res = _stl_ar_by_hand(monthly)
    locs = np.arange(NOBS, NOBS + 17)
    expected = ar_res.forecast(17) + _seasonal_by_hand(seasonal, NOBS, locs)
    assert_allclose(np.asarray(res.forecast(steps)), expected, rtol=1e-10)


@pytest.mark.parametrize("steps", [True, np.True_], ids=["True", "np.True_"])
def test_forecast_bool_steps(monthly, steps):
    # A bool is not a number of steps: True was one step. It is the location
    # of the last forecast, which raises for a date index.
    x = _stl_form(monthly, "series")
    res = STLForecast(x, AutoReg, model_kwargs=AR_KWARGS, period=PERIOD).fit()
    with pytest.raises(TypeError):
        res.forecast(steps)


@pytest.mark.parametrize(
    "dynamic, expected",
    [(np.True_, True), (np.False_, False), (np.array(True), True)],
    ids=["np.True_", "np.False_", "0-d True"],
)
def test_get_prediction_dynamic_numpy_bool(dynamic, expected):
    # A NumPy bool reached the integer checks: np.True_ was an offset of 1
    # and np.False_ an offset of 0 (dynamic from the start)
    rng = np.random.default_rng(20261126)
    t = np.arange(96)
    y = pd.Series(
        10 + np.sin(2 * np.pi * t / 12) + np.cumsum(0.3 * rng.standard_normal(96)),
        index=pd.date_range("2000-01-01", periods=96, freq=MONTH_END),
    )
    res = STLForecast(y, AutoReg, model_kwargs={"lags": 2}, period=12).fit()
    pred = res.get_prediction(start=40, end=80, dynamic=dynamic).predicted_mean
    ref = res.get_prediction(start=40, end=80, dynamic=expected).predicted_mean
    assert_allclose(pred, ref, rtol=1e-12)
    other = res.get_prediction(start=40, end=80, dynamic=not expected)
    assert np.max(np.abs(pred - other.predicted_mean)) > 1e-3
