from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns

from itertools import product
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def test_period_from_inferred_freq():
    # A DatetimeIndex without a freq gives the period of its inferred freq,
    # without the warning pandas 3.1 gives when inferred_freq is read
    rng = np.random.default_rng(20261121)
    dates = pd.date_range("2000-01-01", periods=96, freq="MS")
    y = pd.Series(100 + rng.standard_normal(96).cumsum(), index=list(dates))
    assert y.index.freq is None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mod = ThetaModel(y)
    assert [str(w.message) for w in record] == []
    assert mod.period == 12


def test_forecast_inferred_freq():
    # A regular DatetimeIndex without a freq is extended with the inferred
    # freq, as the same index with the freq. The forecasts used to have
    # integer labels, with a warning.
    rng = np.random.default_rng(20261121)
    y = np.exp(np.cumsum(0.01 + 0.01 * rng.standard_normal(120)))
    index = pd.date_range("2000-01-31", periods=120, freq=MONTH_END)
    no_freq = pd.DatetimeIndex(list(index))
    assert no_freq.freq is None
    res = ThetaModel(pd.Series(y, index=no_freq)).fit()
    expected = ThetaModel(pd.Series(y, index=index)).fit()
    assert res.model.period == 12
    fcast_index = pd.date_range(index[-1], periods=8, freq=MONTH_END)[1:]
    with pytest_warns(None):
        fcast = res.forecast(7)
        components = res.forecast_components(7)
    pd.testing.assert_index_equal(fcast.index, fcast_index)
    pd.testing.assert_series_equal(fcast, expected.forecast(7), rtol=1e-12)
    pd.testing.assert_index_equal(components.index, fcast_index)
    pd.testing.assert_frame_equal(
        components, expected.forecast_components(7), rtol=1e-12
    )


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
    dec = None
    if method is not None:
        dec = seasonal_decompose(y, model=method, period=PERIOD)
        adj = y - dec.seasonal if method == "additive" else y / dec.seasonal
    b0, alpha = params
    h = np.arange(1, steps + 1)
    drift = b0 * (h - 1 + 1 / alpha - (1 - alpha) ** nobs / alpha)
    weight = 1.0 if np.isinf(theta) else (theta - 1) / theta
    fcast = weight * drift + _ses_level(adj, alpha)
    if dec is not None:
        season = dec.seasonal[(nobs + np.arange(steps)) % PERIOD]
        fcast = fcast + season if method == "additive" else fcast * season
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


def test_forecast_closed_form_short_series():
    # In a long series (1 - alpha) ** T is negligible, so the drift term is
    # also checked on a short series where it is not
    rng = np.random.default_rng([20261120, 30])
    nobs = 14
    y = np.round(10 * (20 + 0.4 * np.arange(nobs) + 2 * rng.standard_normal(nobs)))
    res = ThetaModel(y, deseasonalize=False).fit()
    alpha = res.params["alpha"]
    assert (1 - alpha) ** nobs / alpha > 1e-3
    expected, _ = _theta_closed_form(y, res.params, 6, theta=3.0)
    assert_allclose(res.forecast(6, theta=3.0), expected, rtol=1e-9)


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
    # The one-step variance on the deseasonalized scale is sigma2.
    assert_allclose((pi["lower"] + pi["upper"]) / 2, fcast, rtol=1e-12)
    half_width = stats.norm.ppf(0.95) * np.sqrt(res.sigma2)
    season = seasonal_decompose(seasonal_y, model="mul", period=PERIOD).seasonal
    half_width *= season[len(seasonal_y) % PERIOD]
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


@pytest.mark.parametrize("alpha", [0.3, 1.0])
def test_forecast_trend_closed_form(alpha):
    # At fixed parameters, the trend component is
    # b0 * (h - 1 + 1 / alpha - (1 - alpha)**T / alpha). T = 12 is short
    # enough that (1 - alpha)**T matters for alpha = 0.3, and alpha = 1 gives
    # b0 * h. The forecast is (theta - 1) / theta times the trend plus the
    # one-step SES forecast
    from statsmodels.tsa.forecasting.theta import ThetaModelResults

    y = np.arange(1.0, 13.0)
    model = ThetaModel(y, deseasonalize=False)
    b0, one_step = 0.4, 12.5
    res = ThetaModelResults(b0, alpha, 1.0, one_step, np.empty(0), False, model)
    h = np.arange(1, 8)
    expected = b0 * (h - 1 + 1 / alpha - (1 - alpha) ** 12 / alpha)
    if alpha == 1.0:
        np.testing.assert_allclose(expected, b0 * h, rtol=1e-12)
    comp = res.forecast_components(7)
    np.testing.assert_allclose(comp.trend, expected, rtol=1e-12)
    np.testing.assert_allclose(comp.ses, one_step, rtol=1e-12)
    expected_forecast = 0.5 * expected + one_step
    np.testing.assert_allclose(res.forecast(7), expected_forecast, rtol=1e-12)


@pytest.mark.parametrize("method", ["additive", "multiplicative", None])
def test_sigma2_deseasonalized(method):
    # Without MLE, sigma2 is estimated from an IMA(1,1) with drift fit to the
    # deseasonalized data, which the forecasts and the MLE estimates use.
    # method=None does not deseasonalize. The data are an IMA(1,1) with drift
    # plus a seasonal pattern
    rng = np.random.default_rng(20261120)
    e = rng.standard_normal(121)
    t = np.arange(120)
    y = 50 + np.cumsum(0.3 + e[1:] - 0.5 * e[:-1])
    y = y + 5 * np.sin(2 * np.pi * t / 12)
    if method is None:
        model = ThetaModel(y, deseasonalize=False)
        deseasonalized = y
    else:
        model = ThetaModel(y, period=12, use_test=False, method=method)
        seasonal = seasonal_decompose(y, model=method, period=12).seasonal
        if method == "multiplicative":
            deseasonalized = y / seasonal
        else:
            deseasonalized = y - seasonal
    res = model.fit(use_mle=False)
    # The same estimator on the same data. Both are the results of an
    # optimizer, and where it stops depends on the platform: on macOS
    # (Accelerate) the two differ by 3e-7 relative, and noise of a few ulps
    # in the data moves sigma2 by up to 7e-7. Fitting to the seasonal data,
    # the bug the test guards against, makes sigma2 3.6 to 4.5 times larger.
    ima = SARIMAX(deseasonalized, order=(0, 1, 1), trend="c")
    expected = ima.fit(disp=False).params[-1]
    np.testing.assert_allclose(res.sigma2, expected, rtol=1e-4)


@pytest.mark.parametrize("use_mle", [True, False])
@pytest.mark.parametrize("method", ["additive", "multiplicative", None])
def test_prediction_intervals_ima_variance(use_mle, method):
    # The Theta model is an IMA(1,1) with drift whose moving average
    # coefficient is alpha - 1, so the psi weights after the first are alpha
    # and the h-step forecast variance is sigma2 * (1 + (h - 1) * alpha**2).
    # The drift does not change the variance. The reference is the forecast
    # variance of SARIMAX at the parameters of the Theta model, which has
    # reached its steady state after the training sample
    rng = np.random.default_rng(1129)
    e = rng.standard_normal(301)
    y = 50 + np.cumsum(0.1 + e[1:] - 0.6 * e[:-1])[:-1]
    if method is None:
        model = ThetaModel(y, deseasonalize=False)
        adjusted = y
    else:
        y *= np.resize([0.7, 1.1, 1.4, 0.8], len(y))
        model = ThetaModel(y, period=4, method=method, use_test=False)
        season = seasonal_decompose(y, model=method, period=4).seasonal
        adjusted = y / season if method == "multiplicative" else y - season
    res = model.fit(use_mle=use_mle)
    b0, alpha = res.params
    steps = 6
    pi = res.prediction_intervals(steps)
    half_width = (pi["upper"] - pi["lower"]) / 2
    variance = (half_width / stats.norm.ppf(0.975)) ** 2

    ima = SARIMAX(adjusted, order=(0, 1, 1), trend="c")
    ima_res = ima.filter([b0, alpha - 1, res.sigma2])
    expected = ima_res.get_forecast(steps).var_pred_mean
    closed_form = res.sigma2 * (1 + np.arange(steps) * alpha**2)
    if method == "multiplicative":
        # Var(s * X) = s**2 * Var(X) conditional on the seasonal factors.
        factors = season[(len(y) + np.arange(steps)) % 4]
        expected *= factors**2
        closed_form *= factors**2
    np.testing.assert_allclose(variance, expected, rtol=1e-8)
    np.testing.assert_allclose(variance, closed_form, rtol=1e-8)
    # The intervals are centered on the forecast
    center = (pi["upper"] + pi["lower"]) / 2
    np.testing.assert_allclose(center, res.forecast(steps), rtol=1e-12)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("use_mle", [True, False])
def test_sequence_endog(container, use_mle):
    # A list or a tuple gives the results of the same data in an ndarray. The
    # data are an IMA(1,1) with drift plus a seasonal pattern, so that the MLE
    # of the moving average coefficient is inside the invertible region and
    # determined to more than a few digits
    rng = np.random.default_rng(20261120)
    e = rng.standard_normal(121)
    t = np.arange(120)
    y = 50 + np.cumsum(0.3 + e[1:] - 0.5 * e[:-1])
    y = y + 5 * np.sin(2 * np.pi * t / 12)
    expected = ThetaModel(y, period=12).fit(use_mle=use_mle)
    res = ThetaModel(container(y.tolist()), period=12).fit(use_mle=use_mle)
    # alpha, the MLE estimates and sigma2 are the results of optimizers, and
    # where they stop depends on the memory layout of the data on macOS
    # (Accelerate): alpha differed by 5e-10 relative without MLE there. Noise
    # of a few ulps in these data moves the results by up to 9e-7 relative. A
    # wrong conversion of the input changes the index, the number of
    # observations or the values completely.
    rtol = 1e-4
    np.testing.assert_allclose(res.params, expected.params, rtol=rtol)
    pd.testing.assert_frame_equal(
        res.forecast_components(15), expected.forecast_components(15), rtol=rtol
    )
    pd.testing.assert_series_equal(
        res.forecast(15), expected.forecast(15), rtol=rtol
    )
    pd.testing.assert_frame_equal(
        res.prediction_intervals(15),
        expected.prediction_intervals(15),
        rtol=rtol,
    )

    # The table of the parameter estimates prints them to full precision, so
    # only the model table, with the sample and the number of observations,
    # is compared as text
    def without_time(table):
        lines = table.as_text().splitlines()
        return [ln for ln in lines if "Date:" not in ln and "Time:" not in ln]

    assert without_time(res.summary().tables[0]) == without_time(
        expected.summary().tables[0]
    )


@pytest.mark.parametrize(
    ("method", "error", "match"),
    [
        ("geometric", ValueError, "^method must be one of"),
        (1, TypeError, "^method must be a string"),
    ],
)
def test_invalid_method_message(method, error, match):
    # The error names the method argument
    y = np.arange(1.0, 49.0)
    with pytest.raises(error, match=match):
        ThetaModel(y, period=12, method=method)


@pytest.mark.parametrize("use_mle", [False, True])
def test_summary_parameter_table(use_mle):
    # Each estimate is in the row of its own parameter (0.15.0 showed the
    # first value in a row labelled "Parameters", the second in the row of
    # the first parameter, and dropped the label of the last), with four
    # significant digits
    rng = np.random.default_rng(20261124)
    index = pd.date_range("2000-01-01", periods=60, freq="MS")
    y = pd.Series(50 + np.cumsum(rng.standard_normal(60)), index=index)
    res = ThetaModel(y, deseasonalize=False).fit(use_mle=use_mle)
    lines = res.summary().tables[1].as_text().splitlines()
    # title, ====, header, ----, one row per parameter, ----
    assert lines[2].split() == ["Estimate"]
    rows = [line.split() for line in lines[4:-1]]
    assert [row[0] for row in rows] == list(res.params.index)
    values = np.array([float(row[1]) for row in rows])
    # four significant digits: relative rounding error below 5e-4
    assert_allclose(values, res.params.to_numpy(), rtol=5e-4)
    digits = [len(row[1].replace("-", "").replace(".", "").lstrip("0")) for row in rows]
    assert max(digits) <= 4
