from statsmodels.compat.pandas import MONTH_END

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import statsmodels.datasets
from statsmodels.tools.sm_exceptions import ConvergenceWarning
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
