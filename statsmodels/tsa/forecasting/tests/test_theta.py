from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tests._contract import series_forms

SMOKE_PARAMS = list(product(["array", "dataframe", "series"], [True, False]))
SMOKE_IDS = [f"type: {typ}, exponential: {exp}" for typ, exp in SMOKE_PARAMS]


@pytest.fixture(params=SMOKE_PARAMS, ids=SMOKE_IDS)
def data(request):
    rs = np.random.RandomState([3290328901, 323293105, 121029109])
    scale = 0.01 if request.param[1] else 1
    y = np.cumsum(scale + scale * rs.standard_normal(300))
    if request.param[1]:
        y = np.exp(y)
    index = pd.date_range("2000-01-01", periods=300)
    if request.param[0] == "array":
        return y
    elif request.param[0] == "dataframe":
        return pd.DataFrame(y[:, None], columns=["y"], index=index)
    else:
        return pd.Series(y, name="y", index=index)


@pytest.fixture(params=["datetime", "period", "range", "nofreq"])
def indexed_data(request):
    rs = np.random.RandomState([3290328901, 323293105, 121029109])
    scale = 0.01
    y = np.cumsum(scale + scale * rs.standard_normal(300))
    y = np.exp(y)
    if request.param == "datetime":
        index = pd.date_range("2000-1-1", periods=300)
    elif request.param == "period":
        index = pd.period_range("2000-1-1", periods=300, freq="M")
    elif request.param == "range":
        index = pd.RangeIndex(100, 100 + 2 * 300, 2)
    else:  # request.param == "nofreq"
        index = pd.date_range("2000-1-1", periods=1000)
        locs = np.unique(rs.randint(0, 1000, size=500))
        index = index[locs[:300]]
    return pd.Series(y, index=index, name=f"y_{request.param}")


@pytest.mark.smoke
@pytest.mark.parametrize("period", [None, 4, 12])
@pytest.mark.parametrize("use_mle", [True, False])
@pytest.mark.parametrize("deseasonalize", [True, False])
@pytest.mark.parametrize("use_test", [True, False])
@pytest.mark.parametrize("diff", [True, False])
@pytest.mark.parametrize("model", ["auto", "additive", "multiplicative"])
def test_smoke(data, period, use_mle, deseasonalize, use_test, diff, model):
    if period is None and isinstance(data, np.ndarray):
        return
    res = ThetaModel(
        data,
        period=period,
        deseasonalize=deseasonalize,
        use_test=use_test,
        difference=diff,
        method=model,
    ).fit(use_mle=use_mle)
    assert "b0" in str(res.summary())
    res.forecast(36)
    res.forecast_components(47)
    assert res.model.use_test is (use_test and res.model.deseasonalize)
    assert res.model.difference is diff


@pytest.mark.smoke
def test_alt_index(indexed_data):
    idx = indexed_data.index
    date_like = not hasattr(idx, "freq") or getattr(idx, "freq", None) is None
    period = 12 if date_like else None
    res = ThetaModel(indexed_data, period=period).fit()
    if hasattr(idx, "freq") and idx.freq is None:
        with pytest.warns(UserWarning, match="Only PeriodIndexes, DatetimeIndexes"):
            res.forecast_components(37)
        with pytest.warns(UserWarning, match="Only PeriodIndexes, DatetimeIndexes"):
            res.forecast(23)
    else:
        res.forecast_components(37)
        res.forecast(23)


def test_no_freq():
    rs = np.random.RandomState(3231298)
    idx = pd.date_range("2000-1-1", periods=300)
    locs = []
    for i in range(100):
        locs.append(2 * i + int((i % 2) == 1))
    y = pd.Series(rs.standard_normal(100), index=idx[locs])
    with pytest.raises(ValueError, match="You must specify a period or"):
        ThetaModel(y)


def test_forecast_errors(data):
    res = ThetaModel(data, period=12).fit()
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        res.forecast(-1)
    with pytest.raises(ValueError, match="theta must be a float"):
        res.forecast(7, theta=0.99)
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        res.forecast_components(0)


def test_pi_width():
    # GH 7075
    rs = np.random.RandomState(1233091)
    y = np.arange(100) + rs.standard_normal(100)

    th = ThetaModel(y, period=12, deseasonalize=False)
    res = th.fit()
    pi = res.prediction_intervals(24)
    d = np.squeeze(np.diff(np.asarray(pi), axis=1))
    assert np.all(np.diff(d) > 0)


# GH7544
@pytest.mark.parametrize("period", [4, 12])
def test_forecast_seasonal_alignment(data, period):
    res = ThetaModel(
        data,
        period=period,
        deseasonalize=True,
        use_test=False,
        difference=False,
    ).fit(use_mle=False)
    seasonal = res._seasonal
    comp = res.forecast_components(32)
    index = np.arange(data.shape[0], data.shape[0] + comp.shape[0])
    expected = seasonal[index % period]
    np.testing.assert_allclose(comp.seasonal, expected)


def test_auto():
    rs = np.random.RandomState(232387289)
    m = 250
    e = rs.standard_normal(m)
    s = 10 * np.sin(np.linspace(0, np.pi, 12))
    s = np.tile(s, (m // 12 + 1))[:m]
    idx = pd.period_range("2000-01-01", freq="M", periods=m)
    x = e + s
    y = pd.DataFrame(10 + x - x.min(), index=idx)

    tm = ThetaModel(y, method="auto")
    assert tm.method == "mul"
    res = tm.fit()

    tm = ThetaModel(y, method="mul")
    assert tm.method == "mul"
    res2 = tm.fit()

    np.testing.assert_allclose(res.params, res2.params, rtol=1e-4)

    tm = ThetaModel(y - y.mean(), method="auto")
    assert tm.method == "add"
    res3 = tm.fit()

    assert not np.allclose(res.params, res3.params)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_predict(close_figures):
    rs = np.random.RandomState([3290328901, 323293105, 121029109])
    y = np.cumsum(0.01 + 0.01 * rs.standard_normal(300))
    index = pd.date_range("2000-01-01", periods=300)
    y = pd.Series(np.exp(y), index=index, name="y")

    res = ThetaModel(y, period=7).fit()

    fig = res.plot_predict(steps=12)
    ax = fig.axes[0]
    # forecast line covers the requested horizon
    forecast_lines = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == 12]
    assert len(forecast_lines) >= 1
    fc = res.forecast(steps=12)
    np.testing.assert_allclose(forecast_lines[0].get_ydata(), fc.values)

    # alpha=None must omit the confidence interval fill, and the legend
    fig_noci = res.plot_predict(steps=12, alpha=None)
    assert len(fig_noci.axes[0].collections) == 0
    assert fig_noci.axes[0].get_legend() is None
    fig_ci = res.plot_predict(steps=12, alpha=0.05)
    assert len(fig_ci.axes[0].collections) >= 1

    fig_hist = res.plot_predict(steps=12, in_sample=True)
    in_sample_lines = [ln for ln in fig_hist.axes[0].get_lines()
                       if len(ln.get_ydata()) == len(y)]
    assert len(in_sample_lines) >= 1


def test_theta_invalid_period_raises():
    # periods below 2 previously slipped past validation: negative periods
    # passed the seasonality test silently and were treated as non-seasonal
    y = np.arange(1.0, 25.0)
    with pytest.raises(ValueError, match="period must be >= 2"):
        ThetaModel(y, period=-3)
    with pytest.raises(ValueError, match="period must be >= 2"):
        ThetaModel(y, period=1)


NOBS = 96
PERIOD = 12


@pytest.fixture(scope="module")
def seasonal_y():
    # Positive, trending and seasonal. Integer-valued so that the int dtype
    # form holds the same values.
    rng = np.random.default_rng([20261120, 3])
    t = np.arange(NOBS)
    y = 50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / PERIOD)
    return np.round(10 * (y + rng.standard_normal(NOBS)))


def _ses_level(y, alpha):
    # The SES level after the last observation, starting from the first one
    level = y[0]
    for obs in y:
        level = alpha * obs + (1 - alpha) * level
    return level


def _theta_closed_form(y, params, steps, theta=2.0, method=None):
    """
    The Theta forecast at fixed b0 and alpha

    SES of the (deseasonalized) data plus (theta - 1) / theta times the drift
    term of Hyndman and Billah (2003), reseasonalized when method is given.
    Also returns the data the SES and the OLS trend are fit to.
    """
    nobs = y.shape[0]
    adj = y
    if method is not None:
        dec = seasonal_decompose(y, model=method, period=PERIOD)
        adj = y - dec.seasonal if method == "additive" else y / dec.seasonal
        season = dec.seasonal[(nobs + np.arange(steps)) % PERIOD]
    b0, alpha = params
    h = np.arange(1, steps + 1)
    drift = b0 * (h - 1 + 1 / alpha - (1 - alpha) ** nobs / alpha)
    weight = 1.0 if np.isinf(theta) else (theta - 1) / theta
    fcast = weight * drift + _ses_level(adj, alpha)
    if method == "additive":
        fcast = fcast + season
    elif method == "multiplicative":
        fcast = fcast * season
    return fcast, adj


@pytest.mark.parametrize("theta", [1.0, 2.0, 3.5, np.inf])
def test_forecast_closed_form(seasonal_y, theta):
    # b0 is the OLS slope on 0, ..., T-1. theta=1 gives the SES forecast and
    # theta=inf the full drift.
    res = ThetaModel(seasonal_y, deseasonalize=False).fit()
    slope = np.polyfit(np.arange(NOBS), seasonal_y, 1)[0]
    assert_allclose(res.params["b0"], slope, rtol=1e-9)
    expected, _ = _theta_closed_form(seasonal_y, res.params, 18, theta=theta)
    assert_allclose(res.forecast(18, theta=theta), expected, rtol=1e-9)


@pytest.mark.parametrize("method", ["additive", "add", "multiplicative", "mul"])
def test_forecast_deseasonalized_closed_form(seasonal_y, method):
    full = "additive" if method.startswith("add") else "multiplicative"
    res = ThetaModel(seasonal_y, period=PERIOD, use_test=False, method=method).fit()
    expected, adj = _theta_closed_form(seasonal_y, res.params, 18, method=full)
    slope = np.polyfit(np.arange(NOBS), adj, 1)[0]
    assert_allclose(res.params["b0"], slope, rtol=1e-9)
    assert_allclose(res.forecast(18), expected, rtol=1e-9)

    comp = res.forecast_components(18)
    assert list(comp.columns) == ["trend", "ses", "seasonal"]
    dec = seasonal_decompose(seasonal_y, model=full, period=PERIOD)
    season = dec.seasonal[(NOBS + np.arange(18)) % PERIOD]
    assert_allclose(comp["seasonal"], season, rtol=1e-12)
    assert_allclose(comp["ses"], _ses_level(adj, res.params["alpha"]), rtol=1e-9)


@pytest.mark.parametrize("difference,reject", [(False, True), (True, False)])
def test_seasonality_test_decision(difference, reject):
    # The data are deseasonalized only if the period-th autocorrelation is
    # significant at 10%, using the differences if difference is True
    rng = np.random.default_rng([20261120, 5])
    y = 100 + np.cumsum(rng.standard_normal(NOBS))
    x = np.diff(y) if difference else y
    rho = acf(x, nlags=PERIOD, fft=True)
    stat = x.shape[0] * rho[-1] ** 2 / np.sum(rho[:-1] ** 2)
    assert bool(stat > stats.chi2.ppf(0.9, 1)) is reject

    mod = ThetaModel(y, period=PERIOD, difference=difference, method="additive")
    res = mod.fit()
    assert mod.difference is difference
    season = res.forecast_components(PERIOD)["seasonal"]
    if reject:
        dec = seasonal_decompose(y, model="additive", period=PERIOD)
        expected = dec.seasonal[(NOBS + np.arange(PERIOD)) % PERIOD]
        assert_allclose(season, expected, rtol=1e-12)
    else:
        # Not deseasonalized, so the additive seasonal component is zero
        assert_allclose(season, np.zeros(PERIOD))


# list and tuple are left out: ThetaModel.fit raises AttributeError for them
@pytest.mark.parametrize("form", ["ndarray", "int", "series", "column", "dataframe"])
def test_input_forms_and_index(seasonal_y, form):
    if form == "ndarray":
        x = seasonal_y
    elif form == "int":
        x = seasonal_y.astype(np.int64)
    else:
        x = series_forms(seasonal_y)[form]
    res = ThetaModel(x, period=PERIOD, use_test=False, method="additive").fit()
    expected, _ = _theta_closed_form(seasonal_y, res.params, 18, method="additive")
    fcast = res.forecast(18)
    assert isinstance(fcast, pd.Series)
    assert fcast.name == "forecast"
    assert_allclose(fcast, expected, rtol=1e-9)
    # pandas input continues its index, other input continues the positions
    if form in ("series", "dataframe"):
        index = x.index
        expected_index = pd.date_range(
            index[-1] + index.freq, periods=18, freq=index.freq
        )
    else:
        expected_index = pd.RangeIndex(NOBS, NOBS + 18)
    pd.testing.assert_index_equal(fcast.index, expected_index)
    pd.testing.assert_index_equal(res.forecast_components(18).index, expected_index)
    pd.testing.assert_index_equal(res.prediction_intervals(18).index, expected_index)


@pytest.mark.parametrize("form", ["series", "dataframe"])
def test_period_from_index(seasonal_y, form):
    x = series_forms(seasonal_y)[form]
    assert ThetaModel(x).period == 12
    x.index = pd.date_range("2000-01-01", periods=NOBS, freq="QS")
    assert ThetaModel(x).period == 4


def test_ndarray_needs_period(seasonal_y):
    with pytest.raises(ValueError, match="You must specify a period"):
        ThetaModel(seasonal_y)
    # The period is not needed when the data are not deseasonalized
    mod = ThetaModel(seasonal_y, deseasonalize=False)
    assert mod.period is None
    assert not mod.deseasonalize
    assert not mod.use_test


def test_invalid_method(seasonal_y):
    with pytest.raises(ValueError, match="must be one of"):
        ThetaModel(seasonal_y, period=PERIOD, method="geometric")


@pytest.mark.parametrize("use_mle", [True, False])
def test_prediction_interval_one_step(seasonal_y, use_mle):
    res = ThetaModel(seasonal_y, period=PERIOD, use_test=False).fit(use_mle=use_mle)
    fcast = res.forecast(12, theta=3)
    pi = res.prediction_intervals(12, theta=3, alpha=0.1)
    assert list(pi.columns) == ["lower", "upper"]
    # Symmetric about the forecast, and the one-step variance is sigma2
    assert_allclose((pi["lower"] + pi["upper"]) / 2, fcast, rtol=1e-12)
    half_width = stats.norm.ppf(0.95) * np.sqrt(res.sigma2)
    assert_allclose(pi["upper"].iloc[0] - fcast.iloc[0], half_width, rtol=1e-10)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_predict_in_sample_ndarray(seasonal_y, close_figures):
    res = ThetaModel(seasonal_y, period=PERIOD, use_test=False).fit()
    fig = res.plot_predict(steps=12, alpha=None, in_sample=True)
    in_sample, forecast = fig.axes[0].get_lines()
    # An ndarray is plotted against its positions, followed by the forecast
    assert_allclose(in_sample.get_xdata(), np.arange(NOBS))
    assert_allclose(in_sample.get_ydata(), seasonal_y)
    assert_allclose(forecast.get_xdata(), np.arange(NOBS, NOBS + 12))
    assert_allclose(forecast.get_ydata(), res.forecast(12), rtol=1e-12)
