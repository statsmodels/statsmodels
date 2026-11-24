from statsmodels.compat.numpy import NP_LT_1_24
from statsmodels.compat.python import PYTHON_IMPL_WASM

from itertools import combinations
from typing import NamedTuple
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from scipy import stats

from statsmodels.datasets import danish_data, longley
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl import model as ardl_model
from statsmodels.tsa.ardl.model import (
    ARDL,
    UECM,
    ARDLResults,
    ardl_select_order,
)
from statsmodels.tsa.deterministic import DeterministicProcess, Seasonality

dane_data = danish_data.load_pandas().data


class Dataset(NamedTuple):
    y: pd.Series
    x: pd.DataFrame


@pytest.fixture(scope="module", params=[None, 0, 3, [1, 2, 4]])
def lags(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        2,
        [1, 3, 5],
        {"lry": 3, "ibo": 2, "ide": 1},
        {"lry": 3, "ibo": [2], "ide": [1, 3]},
    ],
)
def order(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        2,
        {"lry": 3, "ibo": 2, "ide": 1},
    ],
)
def uecm_order(request):
    return request.param


@pytest.fixture(scope="module", params=[None, 3])
def uecm_lags(request):
    return request.param


@pytest.fixture(scope="module", params=["n", "c", "ct"])
def trend(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def causal(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 3])
def maxlag(request):
    return request.param


@pytest.fixture(scope="module", params=[2, {"lry": 3, "ibo": 2, "ide": 1}])
def maxorder(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def seasonal(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def use_numpy(request):
    return request.param


@pytest.fixture(scope="module", params=[None, 10])
def hold_back(request):
    return request.param


@pytest.fixture(scope="module")
def data():
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    return Dataset(y, x)


@pytest.fixture(scope="module", params=[None, 2])
def fixed(request):
    if request.param is None:
        return None
    index = dane_data.lrm.index
    gen = np.random.default_rng(0)
    return pd.DataFrame(
        gen.standard_t(10, (dane_data.lrm.shape[0], 2)),
        index=index,
        columns=["z0", "z1"],
    )


def check_results(res: ARDLResults):
    model: ARDL = res.model
    n, k = model._x.shape
    assert model.score(res.params).shape == (k,)
    assert model.hessian(res.params).shape == (k, k)
    assert isinstance(model.loglike(res.params), float)
    assert model.information(res.params).shape == (k, k)
    assert isinstance(model.exog_names, list)
    assert model.ar_lags is None or isinstance(model.ar_lags, list)
    assert isinstance(model.seasonal, bool)
    assert isinstance(model.hold_back, int)
    assert isinstance(model.ardl_order, tuple)
    assert isinstance(model.df_model, int)
    assert isinstance(model.nobs, int)
    assert isinstance(model.endog_names, str)
    assert isinstance(model.k_constant, int)
    res.summary()
    res.test_heteroskedasticity()
    res.diagnostic_summary()
    assert res.ar_lags is None or isinstance(res.ar_lags, list)
    assert res.df_resid == (res.nobs - res.df_model)
    assert res.scale == res.sigma2
    assert isinstance(res.fpe, float)


def _convert_to_numpy(data, fixed, order, seasonal, use_numpy):
    y = data.y
    x = data.x
    z = fixed
    period = None
    if use_numpy:
        y = np.asarray(y)
        x = np.asarray(x)
        if isinstance(order, dict):
            order = dict(enumerate(order.values()))
        if fixed is not None:
            z = np.asarray(fixed)
        period = 4 if seasonal else None
    return y, x, z, order, period


def test_model_init(
    data: Dataset, lags, order, trend, causal, fixed, use_numpy, seasonal
):
    y, x, z, order, period = _convert_to_numpy(data, fixed, order, seasonal, use_numpy)

    mod = ARDL(
        y,
        lags,
        x,
        order,
        trend,
        causal=causal,
        fixed=z,
        seasonal=seasonal,
        period=period,
    )
    res = mod.fit()
    check_results(res)
    res.predict()
    fixed_oos = None
    if z is not None:
        fixed_oos = np.array(z)[:12]
        if isinstance(z, pd.DataFrame):
            fixed_oos = pd.DataFrame(fixed_oos, columns=z.columns)
    exog_oos = None
    if x is not None:
        exog_oos = np.array(x)[:12]
        if isinstance(x, pd.DataFrame):
            exog_oos = pd.DataFrame(exog_oos, columns=x.columns)

    res.forecast(12, exog=exog_oos, fixed=fixed_oos)


def test_ardl_order_exceptions(data):
    with pytest.raises(ValueError, match="lags must be a non-negative"):
        ARDL(data.y, -1)
    with pytest.raises(ValueError, match="All values in lags must be positive"):
        ARDL(data.y, [-1, 0, 2])
    with pytest.raises(ValueError, match="integer orders must be at least"):
        ARDL(data.y, 2, data.x, order=0, causal=True)
    with pytest.raises(ValueError, match="integer orders must be at least"):
        ARDL(data.y, 2, data.x, -1, causal=False)
    with pytest.raises(ValueError, match="sequence orders must be strictly positive"):
        ARDL(data.y, 2, data.x, {"lry": [0, 1], "ibo": 3, "ide": 0}, causal=True)
    with pytest.raises(TypeError, match="sequence orders must contain non-negative"):
        ARDL(
            data.y,
            2,
            data.x,
            {"lry": [1, "apple"], "ibo": 3, "ide": 1},
            causal=True,
        )
    with pytest.raises(ValueError, match="sequence orders must contain distinct"):
        ARDL(
            data.y,
            2,
            data.x,
            {"lry": [1, 1, 2, 3], "ibo": 3, "ide": [1, 1, 1]},
            causal=True,
        )
    with pytest.raises(ValueError, match="sequence orders must be strictly positive"):
        ARDL(data.y, 2, data.x, [0, 1, 2], causal=True)


def test_ardl_order_keys_exceptions(data):
    with pytest.raises(
        ValueError, match="order dictionary contains keys for exogenous"
    ):
        ARDL(
            data.y,
            2,
            data.x,
            {"lry": [1, 2], "ibo": 3, "other": 4},
            causal=False,
        )
    with pytest.warns(SpecificationWarning, match="exog contains variables that"):
        ARDL(data.y, 2, data.x, {"lry": [1, 2]}, causal=False)


def test_ardl_deterministic_exceptions(data):
    with pytest.raises(TypeError):
        ARDL(data.y, 2, data.x, 2, deterministic="seasonal")
    deterministic = DeterministicProcess(data.y.index, constant=True, order=1)
    with pytest.warns(SpecificationWarning, match="When using deterministic, trend"):
        ARDL(data.y, 2, data.x, 2, deterministic=deterministic, trend="ct")


def test_ardl_holdback_exceptions(data):
    with pytest.raises(ValueError, match="hold_back must be >="):
        ARDL(data.y, 2, data.x, 2, hold_back=1)


def test_ardl_fixed_exceptions(data):
    rs = np.random.RandomState(992711)
    fixed = rs.standard_normal((2, 200))
    with pytest.raises(ValueError, match="fixed must be an"):
        ARDL(data.y, 2, data.x, 2, fixed=fixed)
    fixed = rs.standard_normal((dane_data.lrm.shape[0], 2))
    fixed[20, 0] = -np.inf
    with pytest.raises(ValueError, match="fixed must be an"):
        ARDL(data.y, 2, data.x, 2, fixed=fixed)


def test_ardl_select_order(
    data: Dataset,
    maxlag,
    maxorder,
    trend,
    causal,
    fixed,
    use_numpy,
    seasonal,
    hold_back,
):
    y, x, z, maxorder, period = _convert_to_numpy(
        data, fixed, maxorder, seasonal, use_numpy
    )
    res = ardl_select_order(
        y,
        maxlag,
        x,
        maxorder,
        trend,
        fixed=fixed,
        causal=causal,
        hold_back=hold_back,
        period=period,
        seasonal=seasonal,
        glob=seasonal,
    )
    assert isinstance(res.model, ARDL)
    assert isinstance(res.aic, pd.Series)
    assert isinstance(res.bic, pd.Series)
    assert isinstance(res.hqic, pd.Series)
    assert res.period == period
    assert res.trend == trend
    assert res.seasonal == seasonal
    assert isinstance(res.dl_lags, dict)
    assert res.ar_lags is None or isinstance(res.ar_lags, list)


def test_ardl_select_order_invalid_ic_raises(data):
    with pytest.raises(ValueError, match="ic"):
        ardl_select_order(data.y, 2, data.x, 2, ic="not-a-real-ic")


def test_ardl_no_regressors(data):
    res = ARDL(
        data.y,
        None,
        data.x,
        {"lry": None, "ibo": None, "ide": None},
        trend="n",
    ).fit()
    assert res.params.shape[0] == 0
    check_results(res)


def test_ardl_only_y_lag(data):
    res = ARDL(data.y, 3, data.x, None, trend="n").fit()
    assert res.params.shape[0] == 3
    check_results(res)


def test_ardl_only_x(data):
    res = ARDL(data.y, None, data.x, {"lry": 1, "ibo": 2, "ide": 3}, trend="n").fit()
    assert res.params.shape[0] == 9
    res = ARDL(
        data.y,
        None,
        data.x,
        {"lry": 1, "ibo": 2, "ide": 3},
        trend="n",
        causal=True,
    ).fit()
    assert res.params.shape[0] == 6
    check_results(res)


def test_ardl_only_trend(data):
    res = ARDL(data.y, None, data.x, None, trend="c").fit()
    assert res.params.shape[0] == 1
    check_results(res)


def test_ardl_only_seasonal(data):
    res = ARDL(data.y, None, data.x, None, trend="n", seasonal=True).fit()
    assert res.params.shape[0] == 4
    check_results(res)


def test_ardl_only_deterministic(data):
    deterministic = DeterministicProcess(data.y.index, constant=True, order=3)
    res = ARDL(data.y, None, data.x, None, trend="n", deterministic=deterministic).fit()
    assert res.params.shape[0] == 4
    check_results(res)


def test_ardl_no_endog_exog(data):
    res = ARDL(data.y, None, data.x, None, trend="ct", seasonal=True).fit()
    assert res.params.shape[0] == 5
    check_results(res)


def test_ardl_no_exog(data):
    res = ARDL(data.y, [1, 4], data.x, None, trend="ct", seasonal=True).fit()
    assert res.params.shape[0] == 7
    check_results(res)


def test_ardl_parameter_names(data):
    mod = ARDL(data.y, 2, data.x, 2, causal=True, trend="c")
    expected = [
        "const",
        "lrm.L1",
        "lrm.L2",
        "lry.L1",
        "lry.L2",
        "ibo.L1",
        "ibo.L2",
        "ide.L1",
        "ide.L2",
    ]
    assert mod.exog_names == expected
    mod = ARDL(np.asarray(data.y), 2, np.asarray(data.x), 2, causal=False, trend="ct")
    expected = [
        "const",
        "trend",
        "y.L1",
        "y.L2",
        "x0.L0",
        "x0.L1",
        "x0.L2",
        "x1.L0",
        "x1.L1",
        "x1.L2",
        "x2.L0",
        "x2.L1",
        "x2.L2",
    ]
    assert mod.exog_names == expected
    mod = ARDL(
        np.asarray(data.y),
        [2],
        np.asarray(data.x),
        None,
        causal=False,
        trend="n",
        seasonal=True,
        period=4,
    )
    expected = ["s(1,4)", "s(2,4)", "s(3,4)", "s(4,4)", "y.L2"]
    assert mod.exog_names == expected


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_diagnostics_plot(data, close_figures):
    import matplotlib.figure

    res = ARDL(
        data.y,
        2,
        data.x,
        {"lry": 3, "ibo": 2, "ide": [1, 3]},
        trend="ct",
        seasonal=True,
    ).fit()

    fig = res.plot_diagnostics()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_against_autoreg(data, trend, seasonal):
    ar = AutoReg(data.y, 3, trend=trend, seasonal=seasonal)
    ardl = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    ar_res = ar.fit()
    ardl_res = ardl.fit()
    assert_allclose(ar_res.params, ardl_res.params)
    assert ar_res.ar_lags == ardl_res.ar_lags
    assert ar.trend == ardl.trend
    assert ar.seasonal == ardl.seasonal

    ar_fcast = ar_res.forecast(12)
    ardl_fcast = ardl_res.forecast(12)
    assert_allclose(ar_fcast, ardl_fcast)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)

    ar_fcast = ar_res.predict()
    ardl_fcast = ardl_res.predict()
    assert_allclose(ar_fcast, ardl_fcast)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)


@pytest.mark.parametrize("start", [None, 0, 2, 4])
@pytest.mark.parametrize("end", [None, 20])
@pytest.mark.parametrize("dynamic", [20, True])
def test_against_autoreg_predict_start_end(data, trend, seasonal, start, end, dynamic):
    ar = AutoReg(data.y, 3, trend=trend, seasonal=seasonal)
    ardl = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    ar_res = ar.fit()
    ardl_res = ardl.fit()

    ar_fcast = ar_res.predict(start=start, end=end, dynamic=dynamic)
    ardl_fcast = ardl_res.predict(start=start, end=end, dynamic=dynamic)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)
    assert_allclose(ar_fcast, ardl_fcast)


def test_invalid_init(data):
    with pytest.raises(ValueError, match="lags must be a non-negative"):
        ARDL(data.y, -1)
    with pytest.raises(ValueError, match="All values in lags must be positive"):
        ARDL(data.y, [-1, 1, 2])
    with pytest.raises(ValueError, match="All values in lags must be positive"):
        ARDL(data.y, [1, 2, 2, 3])
    with pytest.raises(ValueError, match="hold_back must be "):
        ARDL(data.y, 3, data.x, 4, hold_back=3)


def test_prediction_oos_no_new_data(data):
    res = ARDL(data.y, 2, data.x, 3, causal=True).fit()
    val = res.forecast(1)
    assert val.shape[0] == 1
    res = ARDL(data.y, [3], data.x, [3]).fit()
    val = res.forecast(3)
    assert val.shape[0] == 3


@pytest.mark.parametrize("index_type", ["numpy", "range", "date", "period"])
@pytest.mark.parametrize("dynamic", [False, True, 1, 4, 10])
@pytest.mark.parametrize("start", [8, 10])
def test_prediction_oos_start(index_type, dynamic, start):
    y = np.array([2, 7, 1, 3, -1, 5, 2, 4], dtype=float)
    index = pd.RangeIndex(13)
    if index_type == "date":
        index = pd.date_range("2000-01-01", periods=13, freq="MS")
    elif index_type == "period":
        index = pd.period_range("2000-01", periods=13, freq="M")
    if index_type != "numpy":
        y = pd.Series(y, index=index[:8])
    model = ARDL(y, 2)
    # y[t] = 1 + 0.5*y[t-1] - 0.25*y[t-2], with final observations 2, 4.
    expected = np.array([2.5, 1.25, 1, 1.1875, 1.34375])
    end = 12
    if index_type in ("date", "period"):
        prediction = model.predict([1, 0.5, -0.25], index[start], index[end], dynamic)
    else:
        prediction = model.predict([1, 0.5, -0.25], start, end, dynamic)
    assert_allclose(prediction, expected[start - 8 :])
    if index_type != "numpy":
        assert_index_equal(prediction.index, index[start : end + 1])
    if index_type in ("date", "period"):
        prediction = model.predict(
            [1, 0.5, -0.25], index[start], index[end], dynamic=index[start + 1]
        )
        assert_allclose(prediction, expected[start - 8 :])
        with pytest.raises(ValueError, match="Dynamic prediction cannot begin prior"):
            model.predict(
                [1, 0.5, -0.25], index[start], index[end], dynamic=index[start - 1]
            )


@pytest.mark.parametrize("causal", [False, True])
def test_prediction_oos_start_exog_fixed(causal):
    y = np.array([2, 7, 1, 3, -1, 5, 2, 4], dtype=float)
    x = np.arange(13.0)[:, None]
    fixed = (np.arange(13.0) % 3)[:, None]
    model = ARDL(y, 2, x[:8], [int(causal)], fixed=fixed[:8], causal=causal)
    # Scalar recurrence: y[t] = 1 + .5*y[t-1] - .25*y[t-2]
    #                              + 2*x[t-causal] + 3*fixed[t].
    expected = [2.0, 4.0]
    for t in range(8, 13):
        expected.append(
            1 + 0.5 * expected[-1] - 0.25 * expected[-2]
            + 2 * x[t - int(causal), 0] + 3 * fixed[t, 0]
        )
    prediction = model.predict(
        [1, 0.5, -0.25, 2, 3], 10, 12, exog_oos=x[8:], fixed_oos=fixed[8:]
    )
    assert_allclose(prediction, expected[4:])


def test_prediction_oos_start_longley():
    data = longley.load_pandas()
    x = data.exog[["GNPDEFL", "GNP"]]
    res = ARDL(
        data.endog.iloc[:11], 2, x.iloc[:11], {"GNPDEFL": 1, "GNP": 2}
    ).fit()
    full = res.forecast(5, exog=x.iloc[11:])
    prediction = res.predict(13, 13, exog_oos=x.iloc[11:])
    assert_allclose(prediction, full.iloc[2:3])
    assert_index_equal(prediction.index, full.index[2:3])


@pytest.mark.parametrize("start", [8, 10])
def test_get_prediction_oos_start(start, use_numpy):
    y = np.array([2, 7, 1, 3, -1, 5, 2, 4], dtype=float)
    index = pd.date_range("2000-01-01", periods=13, freq="MS")
    if not use_numpy:
        y = pd.Series(y, index=index[:8])
    res = ARDL(y, 1).fit()
    const, phi = res.params
    prediction = res.get_prediction(start, 12)
    horizons = np.arange(start - 7, 6)
    expected_mean = const * (1 - phi**horizons) / (1 - phi) + phi**horizons * 4
    expected_var = [res.sigma2 * sum(phi ** (2 * i) for i in range(h)) for h in horizons]
    assert_allclose(prediction.predicted_mean, expected_mean)
    assert_allclose(prediction.var_pred_mean, expected_var)


def test_prediction_exceptions(data, fixed, use_numpy):
    y, x, z, order, _ = _convert_to_numpy(data, None, 3, False, use_numpy)
    res = ARDL(y, 2, x, 3, causal=False).fit()
    with pytest.raises(ValueError, match="exog_oos must be"):
        res.forecast(1)
    if isinstance(x, pd.DataFrame):
        exog_oos = np.asarray(data.x)[:12]
        with pytest.raises(TypeError, match="exog_oos must be a DataFrame when"):
            res.forecast(12, exog=exog_oos)
        with pytest.raises(ValueError, match="must have the same columns"):
            res.forecast(12, exog=data.x.iloc[:12, :1])


def test_prediction_replacements(data, fixed):
    res = ARDL(data.y, 4, data.x, [1, 3]).fit()
    direct = res.predict()
    alt = res.predict(exog=data.x)
    assert_allclose(direct, alt)
    assert_index_equal(direct.index, alt.index)

    res = ARDL(data.y, 4, data.x, [1, 3], fixed=fixed).fit()
    direct = res.predict()
    alt = res.predict(fixed=fixed)
    assert_allclose(direct, alt)
    assert_index_equal(direct.index, alt.index)


def test_prediction_wrong_shape(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, x, [1, 3]).fit()
    with pytest.raises(ValueError, match="exog must have the same number"):
        res.predict(exog=np.asarray(data.x)[:, :1])
    with pytest.raises(ValueError, match="exog must have the same number of rows"):
        res.predict(exog=np.asarray(data.x)[:-2])
    res = ARDL(data.y, 4, data.x, [1, 3]).fit()
    with pytest.raises(ValueError, match="exog must have the same columns"):
        res.predict(exog=data.x.iloc[:, :1])
    with pytest.raises(ValueError, match="exog must have the same number of rows"):
        res.predict(exog=data.x.iloc[:-2])


def test_prediction_wrong_shape_fixed(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, fixed=x).fit()
    with pytest.raises(ValueError, match="fixed must have the same number"):
        res.predict(fixed=np.asarray(data.x)[:, :1])
    with pytest.raises(ValueError, match="fixed must have the same number of rows"):
        res.predict(fixed=np.asarray(data.x)[:-2])
    res = ARDL(data.y, 4, fixed=data.x).fit()
    with pytest.raises(ValueError, match="fixed must have the same number"):
        res.predict(fixed=data.x.iloc[:, :1])
    with pytest.raises(ValueError, match="fixed must have the same number of rows"):
        res.predict(fixed=data.x.iloc[:-2])


def test_insuficient_oos(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, fixed=x).fit()
    with pytest.raises(ValueError, match="fixed_oos must be provided"):
        res.forecast(12)
    with pytest.raises(ValueError, match="fixed_oos must have at least"):
        res.forecast(12, fixed=x[:11])
    res = ARDL(data.y, 4, data.x, 3, causal=True).fit()
    with pytest.raises(ValueError, match="exog_oos must be provided"):
        res.forecast(12)
    with pytest.raises(ValueError, match="exog_oos must have at least"):
        res.forecast(12, exog=data.x.iloc[-10:])


def test_insuficient_data(data):
    with pytest.raises(ValueError, match=r"The number of regressors \(36\)"):
        ARDL(data.y, 20, data.x, 4)


def test_forecast_date(data):
    res = ARDL(data.y, 3).fit()
    numeric = res.forecast(12)
    date = res.forecast("1990-07-01")
    assert_allclose(numeric, date)
    assert_index_equal(numeric.index, date.index)


def test_get_prediction(data):
    res = ARDL(data.y, 3).fit()
    ar_res = AutoReg(data.y, 3).fit()
    pred = res.get_prediction(end="2020-01-01")
    ar_pred = ar_res.get_prediction(end="2020-01-01")
    assert_allclose(pred.predicted_mean, ar_pred.predicted_mean)
    assert_allclose(pred.var_pred_mean, ar_pred.var_pred_mean)


def test_apply_forecast_matches_ardl_order():
    """
    ARDLResults.apply() used to inherit AutoRegResults.apply(), which
    hardcodes a plain AutoReg(...) reconstruction and so silently drops the
    exog lag structure ('order'). The applied model's design matrix ended up
    narrower than the original params, abd any predict/forecast call raised a matmul shape mismatch
    """
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    y_train, x_train = y.iloc[:45], x.iloc[:45]
    sel_res = ardl_select_order(y_train, 3, x_train, 3, ic="aic", trend="c")
    assert sel_res.model.ardl_order == (3, 0, 3, 3)
    res = sel_res.model.fit()

    y_apply, x_apply = y.iloc[:50], x.iloc[:50]
    x_oos = x.iloc[50:55]
    new_res = res.apply(endog=y_apply, exog=x_apply)
    forecast = new_res.forecast(steps=5, exog=x_oos)
    assert forecast.shape == (5, )
    assert np.all(np.isfinite(forecast))

    predicted = new_res.predict()
    assert predicted.shape[0] == y_apply.shape[0]


def test_apply_design_matrix_matches_fresh_fit():
    """
    Beyond not crashing, the applied model's exog lag columns must be numerically
    identical to an independently constructed ARDL model fit directly on the new data
    under the same order
    """
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    orig = ARDL(y.iloc[:45], 3, x.iloc[:45], order=3, trend="c")
    res = orig.fit()

    y_apply, x_apply = y.iloc[:50], x.iloc[:50]
    new_res = res.apply(endog=y_apply, exog=x_apply)

    fresh = ARDL(y_apply, 3, x_apply, order=orig._order, trend="c")
    assert new_res.model._x.shape == fresh._x.shape
    assert_allclose(new_res.model._x, fresh._x)


def test_append_matches_apply():
    """
    append()ing the tail of a series should give the same applied model as
    apply()ing the full series, since both describe the same combined dataset under
    the same order
    """
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    res = ARDL(y.iloc[:45], 3, x.iloc[:45], order=3, trend="c").fit()

    applied = res.apply(endog=y.iloc[:50], exog=x.iloc[:50])
    appended = res.append(endog=y.iloc[45:50], exog=x.iloc[45:50])

    assert_allclose(applied.model._x, appended.model._x)
    x_oos = x.iloc[50:55]
    assert_allclose(
        applied.forecast(steps=5, exog=x_oos),
        appended.forecast(steps=5, exog=x_oos)
    )


def test_apply_refit_reestimates_params():
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    res = ARDL(y.iloc[:45], 3, x.iloc[:45], order=3, trend="c").fit()

    y_apply, x_apply = y.iloc[:50], x.iloc[:50]
    applied_fixed = res.apply(endog=y_apply, exog=x_apply, refit=False)
    applied_refit = res.apply(endog=y_apply, exog=x_apply, refit=True)

    # refit=False just carries the original params over unchanged
    assert_allclose(applied_fixed.params, res.params)
    # refit=True re-estimates on the new data, matching a fresh fit
    fresh = ARDL(y_apply, 3, x_apply, order=res.model._order, trend="c").fit()
    assert_allclose(applied_refit.params, fresh.params)
    assert not np.allclose(applied_refit.params, res.params)


def test_apply_unexpected_exog_raises():
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    res_no_exog = ARDL(y.iloc[:45], 3, trend="c").fit()

    with pytest.raises(ValueError, match="exog must be None"):
        res_no_exog.apply(endog=y.iloc[:50], exog=x.iloc[:50])


def test_apply_no_exog_roundtrip():
    """
    apply()ing a no-exog model with exog=None -- the ordinary, documented
    case for such a model -- used to crash. ARDLResults.apply reconstructs
    the clone with order=existing._order, and _order for a no-exog model is
    `{}` (the *resolved* empty order, not the raw None the user originally
    passed), which _format_order's `order in (0, None)` guard does not
    match; it fell through to array_like(None, ...) and raised. append()
    delegates to apply() and hit the identical crash.
    """
    y = dane_data.lrm
    res_no_exog = ARDL(y.iloc[:45], 3, trend="c").fit()

    applied = res_no_exog.apply(endog=y.iloc[:50], exog=None)
    assert applied.model.exog is None
    assert np.all(np.isfinite(applied.params))

    appended = res_no_exog.append(endog=y.iloc[45:50], exog=None)
    assert appended.model.exog is None
    assert_allclose(applied.model._x, appended.model._x)


def test_apply_missing_exog_raises():
    """
    apply()'s own explicit check for this case ("exog must be provided
    when the original model contained exog variables") used to be
    unreachable: order=existing._order is keyed by the original exog's
    columns, so reconstructing with exog=None made _format_order raise its
    own array_like error first, wrapped into a generic message that never
    mentioned exog at all.
    """
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    res = ARDL(y.iloc[:45], 3, x.iloc[:45], order=3, trend="c").fit()

    with pytest.raises(ValueError, match="exog must be provided"):
        res.apply(endog=y.iloc[:50], exog=None)


def test_apply_wrong_number_of_exog_columns_raises():
    """
    apply()'s own explicit check for a column-count mismatch used to be
    reachable only when the new exog's columns were a superset of the
    original order dict's keys (e.g. more columns, same names/positions
    reused) -- the much more common case, fewer or renamed columns, made
    _format_order raise its own "extra keys" error first, about the order
    dict rather than about exog itself.
    """
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    res = ARDL(y.iloc[:45], 3, x.iloc[:45], order=3, trend="c").fit()

    with pytest.raises(ValueError, match="number of exog variables"):
        res.apply(endog=y.iloc[:50], exog=x.iloc[:50, :2])


@pytest.mark.parametrize("causal", [True, False])
def test_uecm_apply_append_keep_uecm(causal):
    # UECMResults inherits ARDLResults.apply, which rebuilt the model as an
    # ARDL and paired it with the UECM parameters
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    res = UECM(y.iloc[:45], 3, x.iloc[:45], 2, trend="c", causal=causal).fit()

    fresh = UECM(y.iloc[:50], 3, x.iloc[:50], 2, trend="c", causal=causal)
    expected = fresh.predict(res.params)

    applied = res.apply(endog=y.iloc[:50], exog=x.iloc[:50])
    assert isinstance(applied.model, UECM)
    assert list(applied.params.index) == list(res.params.index)
    assert_allclose(applied.model._x, fresh._x)
    assert_allclose(applied.predict(), expected)

    appended = res.append(endog=y.iloc[45:50], exog=x.iloc[45:50])
    assert isinstance(appended.model, UECM)
    assert_allclose(appended.predict(), expected)

    refit = res.apply(endog=y.iloc[:50], exog=x.iloc[:50], refit=True)
    assert isinstance(refit.model, UECM)
    assert_allclose(refit.params, fresh.fit().params)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
@pytest.mark.smoke
@pytest.mark.parametrize("trend", ["n", "c", "ct"])
@pytest.mark.parametrize("seasonal", [True, False])
def test_ardl_smoke_plots(data, seasonal, trend, close_figures):
    from matplotlib.figure import Figure

    mod = ARDL(
        data.y,
        3,
        trend=trend,
        seasonal=seasonal,
    )
    res = mod.fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, Figure)
    fig = res.plot_predict(end=100)
    assert isinstance(fig, Figure)
    fig = res.plot_predict(end=75, alpha=None, in_sample=False)
    assert isinstance(fig, Figure)
    assert isinstance(res.summary(), Summary)


def test_uecm_model_init(
    data: Dataset,
    uecm_lags,
    uecm_order,
    trend,
    causal,
    fixed,
    use_numpy,
    seasonal,
):
    y, x, z, uecm_order, period = _convert_to_numpy(
        data, fixed, uecm_order, seasonal, use_numpy
    )

    mod = UECM(
        y,
        uecm_lags,
        x,
        uecm_order,
        trend,
        causal=causal,
        fixed=z,
        seasonal=seasonal,
        period=period,
    )
    res = mod.fit()
    check_results(res)
    res.predict()

    ardl = ARDL(
        y,
        uecm_lags,
        x,
        uecm_order,
        trend,
        causal=causal,
        fixed=z,
        seasonal=seasonal,
        period=period,
    )
    uecm = UECM.from_ardl(ardl)
    uecm_res = uecm.fit()
    check_results(uecm_res)
    uecm_res.predict()


def test_from_ardl_none(data):
    with pytest.warns(SpecificationWarning):
        mod = UECM.from_ardl(ARDL(data.y, 2, data.x, {"lry": 2, "ide": 2, "ibo": None}))
    assert mod.ardl_order == (2, 2, 2)


def test_uecm_model_formula(
    data: Dataset,
    uecm_lags,
    uecm_order,
    trend,
    causal,
    fixed,
    seasonal,
):
    fmla = "lrm ~ lry + ibo + ide"
    df = pd.concat([data.y, data.x], axis=1)
    if fixed is not None:
        fmla += " | " + " + ".join(fixed.columns)
        df = pd.concat([df, fixed], axis=1)
    mod = UECM.from_formula(
        fmla,
        df,
        uecm_lags,
        uecm_order,
        trend,
        causal=causal,
        seasonal=seasonal,
    )
    res = mod.fit()
    check_results(res)
    res.predict()


def test_uecm_errors(data):
    with pytest.raises(TypeError, match="order must be None"):
        UECM(data.y, 2, data.x, [0, 1, 2])
    with pytest.raises(TypeError, match="lags must be an"):
        UECM(data.y, [1, 2], data.x, 2)
    with pytest.raises(TypeError, match="order values must be positive"):
        UECM(data.y, 2, data.x, {"ibo": [1, 2]})
    with pytest.raises(ValueError, match="Model must contain"):
        UECM(data.y, 2, data.x, None)
    with pytest.raises(ValueError, match="All included exog"):
        UECM(data.y, 2, data.x, {"lry": 2, "ide": 2, "ibo": 0})
    with pytest.raises(ValueError, match="hold_back must be"):
        UECM(data.y, 3, data.x, 5, hold_back=4)
    with pytest.raises(ValueError, match="The number of"):
        UECM(data.y, 20, data.x, 4)
    ardl = ARDL(data.y, 2, data.x, {"lry": [1, 2], "ide": 2, "ibo": 2})
    with pytest.raises(ValueError, match="UECM can only be created from"):
        UECM.from_ardl(ardl)
    ardl = ARDL(data.y, 2, data.x, {"lry": [0, 2], "ide": 2, "ibo": 2})
    with pytest.raises(ValueError, match="UECM can only be created from"):
        UECM.from_ardl(ardl)
    ardl = ARDL(data.y, [1, 3], data.x, 2)
    with pytest.raises(ValueError, match="UECM can only be created from"):
        UECM.from_ardl(ardl)
    res = UECM(data.y, 2, data.x, 2).fit()
    with pytest.raises(NotImplementedError):
        res.predict(end=100)
    with pytest.raises(NotImplementedError):
        res.predict(dynamic=True)
    with pytest.raises(NotImplementedError):
        res.predict(dynamic=25)


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("use_t", [True, False])
def test_uecm_ci_repr(use_numpy, use_t):
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    if use_numpy:
        y = np.asarray(y)
        x = np.asarray(x)
    mod = UECM(y, 3, x, 3)
    res = mod.fit(use_t=use_t)
    if use_numpy:
        ci_params = res.params[:5].copy()
        ci_params /= ci_params[1]
    else:
        ci_params = res.params.iloc[:5].copy()
        ci_params /= ci_params["lrm.L1"]
    assert_allclose(res.ci_params, ci_params)
    assert res.ci_bse.shape == (5,)
    assert res.ci_tvalues.shape == (5,)
    assert res.ci_pvalues.shape == (5,)
    assert "Cointegrating Vector" in str(res.ci_summary())
    assert res.ci_conf_int().shape == (5, 2)
    assert res.ci_cov_params().shape == (5, 5)
    assert res.ci_resids.shape == dane_data.lrm.shape


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_bounds_test(case):
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    expected = {
        1: 0.7109023,
        2: 5.116768,
        3: 6.205875,
        4: 5.430622,
        5: 6.785325,
    }
    bounds_result = res.bounds_test(case)
    assert_allclose(bounds_result.statistic, expected[case])
    assert "BoundsTestResult" in str(bounds_result)


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_bounds_test_simulation(case):
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    bounds_result = res.bounds_test(
        case=case, asymptotic=False, rng=[1, 2, 3, 4], nsim=10_000
    )
    assert (bounds_result.pvalue >= 0.0).all()
    assert (bounds_result.pvalue <= 1.0).all()
    assert (bounds_result.critical_values > 0.0).all().all()


@pytest.mark.parametrize(
    "rng",
    [None, np.random.RandomState(0), 0, [1, 2], np.random.default_rng([1, 2])],
)
def test_bounds_test_rng(rng):
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    bounds_result = res.bounds_test(case=3, asymptotic=False, rng=rng, nsim=10_000)
    assert (bounds_result.pvalue >= 0.0).all()
    assert (bounds_result.pvalue <= 1.0).all()
    assert (bounds_result.critical_values > 0.0).all().all()


def test_bounds_test_simulate_order():
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    bounds_result = res.bounds_test(3)
    assert "BoundsTestResult" in str(bounds_result)
    bounds_result_sim = res.bounds_test(3, asymptotic=False, nsim=10_000, rng=[1, 2, 3])
    assert_allclose(bounds_result.statistic, bounds_result_sim.statistic)
    assert (bounds_result_sim.pvalue > bounds_result.pvalue).all()


def _bounds_fixed_data(nobs=140):
    # Cointegrated y and x1, x2 with a stationary effect of z1, z2, z3 and of
    # quarterly seasonal dummies. The same data was used with the R reference
    rng = np.random.default_rng(10575)
    e = rng.standard_normal((nobs, 4))
    x1 = np.cumsum(e[:, 0]) + 0.05 * np.arange(nobs)
    x2 = np.cumsum(e[:, 1])
    z1 = rng.standard_normal(nobs)
    z2 = 0.5 * np.r_[0.0, e[:-1, 2]] + e[:, 2]
    z3 = rng.uniform(-1.0, 1.0, nobs)
    pos = np.arange(nobs) % 4
    y = np.zeros(nobs)
    for t in range(1, nobs):
        y[t] = (
            y[t - 1]
            - 0.1 * (y[t - 1] - 0.4 * x1[t - 1] + 0.3 * x2[t - 1] - 0.5)
            + 0.6 * z1[t]
            + 0.4 * z2[t]
            - 0.5 * z3[t]
            + 0.5 * (pos[t] == 2)
            - 0.4 * (pos[t] == 3)
            + 0.3 * (x1[t] - x1[t - 1])
            + e[t, 3]
        )
    index = pd.period_range("1980Q1", periods=nobs, freq="Q")
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "z1": z1, "z2": z2, "z3": z3}, index=index
    )


# Reference values from R, ARDL 0.2.5 (CRAN), with the data of _bounds_fixed_data
# written to csv with 17 digits. For each specification, the F statistic and
# the p-value of the I(1) bound for cases 1 to 5 of
#
#   uecm(y ~ x1 + x2 [- 1 | + trend(y)] | <fixed>, data, order=<order>)
#   bounds_f_test(<model>, case=<case>)
#
# The cases 1, 4 and 5 add "- 1" (case 1) or "+ trend(y)" (cases 4 and 5) to
# the formula. For the seasonal specifications, <fixed> are the explicit
# quarterly dummies s1, s2, s3 (s0, s1, s2, s3 for case 1, which has no
# constant), where s_i = 1 if the (zero based) position of the observation
# modulo 4 is i. R uses the bounds tables of Kripfganz and Schneider, so the
# p-values only agree with statsmodels to about 0.003.
BOUNDS_FIXED_REF = {
    "none": (
        [
            3.540334722642,
            2.640859741313,
            3.496908717074,
            2.733365220736,
            3.640491073069,
        ],
        [0.065191, 0.221479, 0.174347, 0.376394, 0.306278],
    ),
    "z1": (
        [
            2.194698978773,
            1.648222586499,
            2.165738388022,
            2.086999059559,
            2.778353197338,
        ],
        [0.265632, 0.589421, 0.492546, 0.620963, 0.525298],
    ),
    "z123": (
        [
            3.011727218399,
            2.519489127349,
            3.350264734258,
            2.584100689190,
            3.361605910168,
        ],
        [0.117480, 0.254961, 0.198853, 0.428388, 0.371219],
    ),
    "z123_b": (
        [
            2.269351154249,
            1.897824603936,
            2.508557632225,
            1.912166292452,
            2.463065676919,
        ],
        [0.247656, 0.478230, 0.391478, 0.694087, 0.616822],
    ),
    "seasonal": (
        [
            3.425337586493,
            2.576102370312,
            3.425337586493,
            2.681794340056,
            3.572929058717,
        ],
        [0.074224, 0.239045, 0.186010, 0.393332, 0.321163],
    ),
    "seasonal_z1": (
        [
            2.124326580262,
            1.595698339355,
            2.124326580262,
            2.053107281405,
            2.735146441131,
        ],
        [0.284319, 0.613221, 0.505307, 0.635354, 0.537445],
    ),
}
# name: (lags of y, order of x1 and x2, fixed columns, seasonal)
BOUNDS_FIXED_SPECS = {
    "none": (2, {"x1": 2, "x2": 1}, [], False),
    "z1": (2, {"x1": 2, "x2": 1}, ["z1"], False),
    "z123": (2, {"x1": 2, "x2": 1}, ["z1", "z2", "z3"], False),
    "z123_b": (3, {"x1": 1, "x2": 2}, ["z1", "z2", "z3"], False),
    "seasonal": (2, {"x1": 2, "x2": 1}, [], True),
    "seasonal_z1": (2, {"x1": 2, "x2": 1}, ["z1"], True),
}
BOUNDS_CASE_TREND = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("spec", sorted(BOUNDS_FIXED_SPECS))
def test_bounds_test_fixed_seasonal_r(spec, case):
    # The fixed regressors and the seasonal terms were omitted from the model
    # used to compute the statistic
    data = _bounds_fixed_data()
    lags, order, fixed, seasonal = BOUNDS_FIXED_SPECS[spec]
    mod = UECM(
        data["y"],
        lags,
        data[["x1", "x2"]],
        order,
        trend=BOUNDS_CASE_TREND[case],
        fixed=data[fixed] if fixed else None,
        seasonal=seasonal,
    )
    result = mod.fit().bounds_test(case)
    stat, pvalue = BOUNDS_FIXED_REF[spec]
    assert_allclose(result.statistic, stat[case - 1], rtol=1e-8)
    assert_allclose(result.pvalue["upper"], pvalue[case - 1], atol=0.005)


@pytest.mark.parametrize("case", [3, 5])
def test_bounds_test_fixed_seasonal_numpy(case):
    data = _bounds_fixed_data()
    kwargs = {"trend": BOUNDS_CASE_TREND[case]}
    expected = UECM(
        data["y"],
        2,
        data[["x1", "x2"]],
        {"x1": 2, "x2": 1},
        fixed=data[["z1", "z2"]],
        seasonal=True,
        **kwargs,
    ).fit()
    mod = UECM(
        np.asarray(data["y"]),
        2,
        np.asarray(data[["x1", "x2"]]),
        {0: 2, 1: 1},
        fixed=np.asarray(data[["z1", "z2"]]),
        seasonal=True,
        period=4,
        **kwargs,
    )
    result = mod.fit().bounds_test(case)
    result_pd = expected.bounds_test(case)
    assert_allclose(result.statistic, result_pd.statistic, rtol=1e-10)
    assert_allclose(result.pvalue, result_pd.pvalue, rtol=1e-10)


@pytest.mark.parametrize("seasonal", [False, True])
@pytest.mark.parametrize("use_fixed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_bounds_test_matches_levels_wald(causal, use_fixed, seasonal):
    # If the trend of the model is the one of the case, the statistic is the F
    # statistic that the coefficients on the levels of the model are all 0
    data = _bounds_fixed_data()
    mod = UECM(
        data["y"],
        2,
        data[["x1", "x2"]],
        {"x1": 2, "x2": 1},
        trend="c",
        fixed=data[["z1", "z2"]] if use_fixed else None,
        causal=causal,
        seasonal=seasonal,
    )
    res = mod.fit()
    ndet = mod._blocks["deterministic"].shape[1]
    nlvl = mod._blocks["levels"].shape[1]
    loc = np.arange(ndet, ndet + nlvl)
    params = np.asarray(res.params)[loc]
    cov = np.asarray(res.cov_params())[np.ix_(loc, loc)]
    expected = params @ np.linalg.solve(cov, params) / nlvl
    assert_allclose(res.bounds_test(3).statistic, expected, rtol=1e-10)


def test_bounds_test_deterministic_not_implemented():
    data = _bounds_fixed_data()
    det = DeterministicProcess(
        data.index, constant=True, additional_terms=[Seasonality(4)]
    )
    mod = UECM(
        data["y"],
        2,
        data[["x1", "x2"]],
        {"x1": 2, "x2": 1},
        trend="n",
        deterministic=det,
    )
    with pytest.raises(NotImplementedError, match="deterministic"):
        mod.fit().bounds_test(3)


# Pesaran, Shin & Smith (2001), Table CI(iii), Case III, at the 10%, 5% and 1%
# levels, indexed by k, the number of regressors excluding the dependent
# variable. The statsmodels tables were simulated independently, so they
# differ from the published values by up to about 0.07.
PSS_CASE_3 = {
    1: ([4.04, 4.94, 6.84], [4.78, 5.73, 7.84]),
    2: ([3.17, 3.79, 5.15], [4.14, 4.85, 6.36]),
    3: ([2.72, 3.23, 4.29], [3.77, 4.35, 5.61]),
    10: ([1.83, 2.06, 2.54], [2.94, 3.24, 3.86]),
}


@pytest.mark.parametrize("k", sorted(PSS_CASE_3))
def test_bounds_test_crit_vals_pss(k):
    # GH-10546: the tables were indexed by the number of variables including y
    rng = np.random.default_rng(10546)
    data = np.cumsum(rng.standard_normal((200, k + 1)), axis=0)
    y = pd.Series(data[:, 0], name="y")
    x = pd.DataFrame(data[:, 1:], columns=[f"x{i}" for i in range(k)])
    res = UECM(y, 1, x, 1, trend="c").fit()
    bounds_result = res.bounds_test(case=3)
    crit_vals = bounds_result.critical_values.loc[[90.0, 95.0, 99.0]]
    lower, upper = PSS_CASE_3[k]
    assert_allclose(crit_vals["lower"], lower, atol=0.1)
    assert_allclose(crit_vals["upper"], upper, atol=0.1)


def test_bounds_test_pvalue_matches_simulation():
    # GH-10546: with one regressor, the asymptotic p-values were those for two.
    # For this seed the asymptotic and the simulated p-values differ by 0.0003,
    # and by 0.045 if the table for one regressor too many is used. Other
    # seeds of this process differ by up to 0.016 in a sample of 200 (finite
    # sample error), so the seed and the tolerance are tied to each other.
    rng = np.random.default_rng(3)
    nobs = 200
    x = np.cumsum(rng.standard_normal(nobs))
    y = np.zeros(nobs)
    for t in range(1, nobs):
        y[t] = y[t - 1] - 0.05 * (y[t - 1] - 0.5 * x[t - 1]) + rng.standard_normal()
    res = UECM(pd.Series(y), 1, pd.DataFrame({"x": x}), 1, trend="c").fit()
    bounds_result = res.bounds_test(case=3)
    bounds_result_sim = res.bounds_test(case=3, asymptotic=False, nsim=5_000, rng=0)
    assert_allclose(bounds_result.pvalue, bounds_result_sim.pvalue, atol=0.01)


def _pss_data(k, seed, nobs=120):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((nobs, k)), axis=0)
    y = np.zeros(nobs)
    e = rng.standard_normal(nobs)
    for t in range(1, nobs):
        y[t] = y[t - 1] - 0.1 * (y[t - 1] - x[t - 1].sum()) + e[t]
    return pd.Series(y, name="y"), pd.DataFrame(x, columns=[f"x{i}" for i in range(k)])


# Reference values from the CRAN package ARDL 0.2.5 (R 4.6.1), which indexes
# the PSS tables by the number of regressors, excluding y. For the simulated
# data the model was fit on the same data, written with 17 digits, using
#
#   m <- ardl(y ~ x0 + x1 + trend(y), data=d, order=c(2, 1, 1))  # no trend(y): 3
#   bounds_f_test(m, case=case)               # statistic, p-value (upper bound)
#   bounds_f_test(m, case=case, alpha=a)$tab  # bounds at a = 0.1, 0.05, 0.01
#
# and for the Danish data (``denmark`` in ARDL) using
#
#   ardl(LRM ~ LRY + IBO + IDE [+ trend(LRM)], data=denmark, order=c(3,3,3,3))
#
# The statistics agree to rounding error. The tables and p-value response
# surfaces were simulated independently, so the bounds differ from those of ARDL
# by up to 0.03 here and the p-values by up to 0.001. The p-values of ARDL are
# for the upper bound (I(1) regressors).
# (k, case, seed) -> (statistic, upper p-value, lower and upper bounds at 10%, 5%, 1%)
PSS_ARDL_R = {
    (2, 3, 20250101): (
        5.0519595811347,
        0.039012807407573,
        [3.164478061, 3.790645235, 5.165537747],
        [4.098947825, 4.806770545, 6.337465242],
    ),
    (3, 4, 20250102): (
        3.218610401748,
        0.192376185646,
        [2.955746507, 3.407295070, 4.331279426],
        [3.731001848, 4.199901754, 5.220473297],
    ),
    (5, 5, 20250103): (
        4.235000741521,
        0.051746621004,
        [2.756956243, 3.159420396, 3.999982287],
        [3.792142719, 4.259334499, 5.206415588],
    ),
}


@pytest.mark.parametrize("k, case, seed", sorted(PSS_ARDL_R))
def test_bounds_test_vs_r_ardl(k, case, seed):
    # GH-10546: the p-value and tables for k > 1 regressors, against ARDL in R
    stat, pvalue, lower, upper = PSS_ARDL_R[(k, case, seed)]
    y, x = _pss_data(k, seed)
    trend = "c" if case == 3 else "ct"
    res = UECM(y, 2, x, 1, trend=trend).fit()
    bounds_result = res.bounds_test(case=case)
    assert_allclose(bounds_result.statistic, stat, rtol=1e-8)
    assert_allclose(bounds_result.pvalue["upper"], pvalue, atol=0.005)
    assert bounds_result.pvalue["lower"] < bounds_result.pvalue["upper"]
    crit_vals = bounds_result.critical_values.loc[[90.0, 95.0, 99.0]]
    assert_allclose(crit_vals["lower"], lower, atol=0.05)
    assert_allclose(crit_vals["upper"], upper, atol=0.05)


@pytest.mark.parametrize(
    "case, stat, pvalue, lower, upper",
    [
        (
            3,
            5.993052959250,
            0.005424240728,
            [2.7204500, 3.2243925, 4.3325749],
            [3.7649065, 4.3542357, 5.5462945],
        ),
        (
            4,
            5.07063440668,
            0.01291889795,
            [2.9557465, 3.4072951, 4.3312794],
            [3.7310018, 4.1999018, 5.2204733],
        ),
    ],
)
def test_bounds_test_vs_r_ardl_danish(case, stat, pvalue, lower, upper):
    # See PSS_ARDL_R for the R calls; the statsmodels p-values for the upper
    # bound are 0.005308 and 0.013299, those with k one too large 0.0021 and
    # 0.0068
    res = UECM(dane_data.lrm, 3, dane_data[["lry", "ibo", "ide"]], 3).fit()
    bounds_result = res.bounds_test(case=case)
    assert_allclose(bounds_result.statistic, stat, rtol=1e-8)
    assert_allclose(bounds_result.pvalue["upper"], pvalue, rtol=0.05)
    crit_vals = bounds_result.critical_values.loc[[90.0, 95.0, 99.0]]
    assert_allclose(crit_vals["lower"], lower, atol=0.05)
    assert_allclose(crit_vals["upper"], upper, atol=0.05)


# Reference values for UECMs without lagged differences of the dependent
# variable, which the UECM defines as lags in (None, 0, 1). They were computed
# with the CRAN package ARDL 0.2.5 (R 4.6.1) on the Danish data (lrm, lry, ibo,
# ide), where ``bounds_f_test`` is applied to
#
#   m <- uecm(lrm ~ lry + ibo + ide, data = ts(danish), order = c(p, q1, q2, q3))
#   bounds_f_test(m, case = case, pvalue = TRUE, exact = FALSE)
#
# using ``- 1`` in the formula for case 1 and ``+ trend(lrm)`` for cases 4 and
# 5. R's ``order`` is (p, q1, q2, q3) so that lags=p and order={"lry": q1, ...}
# in statsmodels, except that p=1 in R has no lagged differences, which is
# lags in (None, 0, 1) in statsmodels. R only reports the p-value for the
# upper (I(1)) bound, and interpolates it linearly in a table of simulated
# critical values, so only the statistic is compared tightly.
UECM_BOUNDS_ORDERS = {
    "A": {"lry": 1, "ibo": 1, "ide": 1},
    "B": {"lry": 1, "ibo": 3, "ide": 2},
    "C": {"lry": 1, "ibo": 1, "ide": 1},
    "D": {"lry": 2, "ibo": 1, "ide": 3},
}
# (statistic, p-value of upper bound) by case
UECM_BOUNDS_R = {
    "A": {
        1: (1.9432957566916351, 0.36287272254480135),
        2: (4.8144488803156964, 0.0070907355695709181),
        3: (5.4869229704429063, 0.010819220971760888),
        4: (4.6924170205040081, 0.023467875999393494),
        5: (5.7387913709648135, 0.021112138280176964),
    },
    "B": {
        1: (1.4481839378587602, 0.56059763874767199),
        2: (6.7011928959160612, 0.00074591538238529703),
        3: (7.8113641520124864, 0.0008499446520620936),
        4: (6.6096462743357502, 0.0009660310080628051),
        5: (8.2106581226912301, 0.0009121472341234244),
    },
    "C": {
        1: (2.866152280082872, 0.12805153309903969),
        2: (7.6795561299124016, 0.00041522066012287438),
        3: (8.8358479889672736, 0.00058703744889628494),
        4: (7.7725330968737918, 0.00047398337170748313),
        5: (9.3931953005845052, 0.00053002856715460072),
    },
    "D": {
        1: (3.0274938465848855, 0.10468737489476289),
        2: (6.4510893339483442, 0.00083045241841425536),
        3: (7.2358881879706587, 0.00099762563124191576),
        4: (6.1225320160599539, 0.0022308094015068124),
        5: (7.4450742068913236, 0.0020834410519250278),
    },
}


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "spec, lags",
    [
        ("A", 0),
        ("A", None),
        ("A", 1),
        ("B", 0),
        ("B", None),
        ("B", 1),
        ("C", 2),
        ("D", 2),
    ],
)
def test_bounds_test_matches_r(spec, lags, case):
    trend = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}[case]
    res = UECM(
        dane_data.lrm,
        lags,
        dane_data[["lry", "ibo", "ide"]],
        UECM_BOUNDS_ORDERS[spec],
        trend=trend,
    ).fit()
    stat, pvalue = UECM_BOUNDS_R[spec][case]
    bounds_result = res.bounds_test(case)
    assert_allclose(bounds_result.statistic, stat, rtol=1e-8)
    assert_allclose(bounds_result.pvalue["upper"], pvalue, atol=5e-3)
    assert bounds_result.pvalue["lower"] < bounds_result.pvalue["upper"]


@pytest.mark.parametrize("lags", [0, None])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_bounds_test_no_lagged_dendog(lags, causal, trend, case):
    # The UECM always includes the lagged level of the dependent variable, so
    # that lags=0 and lags=None are the same model as lags=1
    kwargs = {
        "exog": dane_data[["lry", "ibo", "ide"]],
        "order": {"lry": 1, "ibo": 2, "ide": 1},
        "causal": causal,
        "trend": trend,
    }
    expected = UECM(dane_data.lrm, 1, **kwargs).fit().bounds_test(case)
    res = UECM(dane_data.lrm, lags, **kwargs).fit()
    assert res.model.ar_lags is None
    bounds_result = res.bounds_test(case)
    assert_allclose(bounds_result.statistic, expected.statistic, rtol=1e-9)
    assert_allclose(bounds_result.pvalue, expected.pvalue, rtol=1e-9)
    assert_allclose(bounds_result.critical_values, expected.critical_values)


@pytest.mark.parametrize("lags", [0, 2])
@pytest.mark.parametrize("case, trend", [(1, "n"), (3, "c"), (5, "ct")])
def test_bounds_test_matches_wald_test(lags, case, trend):
    # The rebuilt model must have the lag structure of the original model, so
    # that for the matching trend the bounds statistic is the F statistic of
    # the test that all level terms are 0
    res = UECM(
        dane_data.lrm,
        lags,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
        trend=trend,
    ).fit()
    levels = [n for n in res.params.index if n.endswith(".L1") and n[:2] != "D."]
    assert len(levels) == 4
    wald = res.wald_test(", ".join(f"{name} = 0" for name in levels), use_f=True)
    bounds_result = res.bounds_test(case)
    assert_allclose(bounds_result.statistic, np.squeeze(wald.statistic), rtol=1e-10)


def test_bounds_test_from_ardl_no_lagged_dendog():
    exog = dane_data[["lry", "ibo", "ide"]]
    ardl = ARDL(dane_data.lrm, 0, exog, 2)
    res = UECM.from_ardl(ardl).fit()
    expected = UECM(dane_data.lrm, 0, exog, 2).fit()
    assert_allclose(
        res.bounds_test(3).statistic, expected.bounds_test(3).statistic, rtol=1e-12
    )


def test_resids_ardl_uecm():
    ardl_mod = ARDL(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    ardl_res = ardl_mod.fit()
    uecm_mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    uecm_res = uecm_mod.fit()

    assert_allclose(uecm_res.resid, ardl_res.resid)


@pytest.mark.parametrize("y_lags", [None, 1, 2])
@pytest.mark.parametrize("x_lags", [None, 1, 2])
@pytest.mark.parametrize("causal", [True, False])
def test_ardl_trend_ctt(data, y_lags, x_lags, causal):
    """Test ARDL with trend='ctt'."""
    res = ARDL(data.y, y_lags, data.x, x_lags, trend="ctt", causal=causal).fit()
    n_x = data.x.shape[1]
    n_params = 3
    n_params += y_lags or 0
    n_params += n_x * (int(not causal) + x_lags) if x_lags else 0
    assert res.params.shape[0] == n_params
    check_results(res)


def test_uecm_resid():
    """Test that UECMResults.resid is computed correctly using model._y."""
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()

    # Assert that the override fix is working
    np.testing.assert_allclose(res.resid, res.model._y - res.fittedvalues)


@pytest.mark.parametrize("model", [ARDL, UECM])
@pytest.mark.parametrize("lags", [0, 3])
@pytest.mark.parametrize("pandas", [True, False])
def test_ardl_uecm_summary_after_remove_data(model, lags, pandas):
    """Test that UECMResults.resid is computed correctly using model._y."""
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    order = {"lry": 1, "ibo": 3, "ide": 2}
    if not pandas:
        y = np.array(y)
        x = np.array(x)
        order = dict(enumerate(order.values()))
    mod = model(y, lags, x, order)
    res = mod.fit()
    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


@pytest.mark.parametrize("model", [ARDL, UECM])
@pytest.mark.parametrize("trend", ["n", "c", "ct"])
def test_ardl_uecm_rsquared(data, model, trend):
    res = model(data.y, 3, data.x, 2, trend=trend).fit()
    assert res.model.k_constant == int(trend != "n")
    # OLS detects the constant in the design matrix itself
    ols_res = OLS(res.model._y, res.model._x).fit()
    assert_allclose(res.rsquared, ols_res.rsquared)
    assert_allclose(res.ess, ols_res.ess)


def test_ardl_k_constant_deterministic(data):
    deterministic = DeterministicProcess(data.y.index, constant=True, order=1)
    mod = ARDL(data.y, 2, data.x, 2, trend="n", deterministic=deterministic)
    assert mod.k_constant == 1
    res = mod.fit()
    ols_res = OLS(mod._y, mod._x).fit()
    assert_allclose(res.rsquared, ols_res.rsquared)


def _select_order_loop(*args, **kwargs):
    # The previous implementation: fit each model separately
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ardl_model, "_ardl_select_ssr", lambda *a: None)
        return ardl_select_order(*args, **kwargs)


def _select_order_qr(*args, use_qr=True, **kwargs):
    # ardl_select_order, checking whether all models come from one QR
    calls = []
    orig = ardl_model._ardl_select_ssr

    def select_ssr(*a):
        calls.append(orig(*a))
        return calls[-1]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ardl_model, "_ardl_select_ssr", select_ssr)
        res = ardl_select_order(*args, **kwargs)
    assert len(calls) == 1
    assert (calls[0] is not None) == use_qr
    return res


def _assert_select_order_equal(res, expected, rtol):
    for ic in ("aic", "bic", "hqic"):
        actual = getattr(res, ic)
        desired = getattr(expected, ic)
        assert_allclose(actual.index, desired.index, rtol=rtol)
        assert list(actual) == list(desired)
    assert res.model.ardl_order == expected.model.ardl_order
    assert res.ar_lags == expected.ar_lags
    assert res.dl_lags == expected.dl_lags


def test_ardl_select_order_qr(
    data, maxlag, maxorder, trend, causal, fixed, seasonal, hold_back
):
    # one QR gives the same criteria as fitting each model separately
    y, x, z, maxorder, period = _convert_to_numpy(
        data, fixed, maxorder, seasonal, False
    )
    args = (y, maxlag, x, maxorder, trend)
    kwargs = dict(
        fixed=fixed,
        causal=causal,
        hold_back=hold_back,
        period=period,
        seasonal=seasonal,
        glob=seasonal,
    )
    res = _select_order_qr(*args, **kwargs)
    expected = _select_order_loop(*args, **kwargs)
    _assert_select_order_equal(res, expected, rtol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
def test_ardl_select_order_qr_simulated(trend):
    rng = np.random.default_rng(10545)
    nobs = 500
    x = np.cumsum(rng.standard_normal((nobs, 2)), axis=0)
    e = rng.standard_normal(nobs)
    y = np.zeros(nobs)
    for t in range(2, nobs):
        y[t] = 0.5 * y[t - 1] - 0.2 * y[t - 2] + 0.3 * x[t - 1, 0] + e[t]
    args = (y, 4, x, {0: 2, 1: 2}, trend)
    for ic in ("aic", "bic"):
        res = _select_order_qr(*args, ic=ic)
        expected = _select_order_loop(*args, ic=ic)
        _assert_select_order_equal(res, expected, rtol=1e-10)
        assert res.model.ardl_order[:2] == (2, 1)
    res = _select_order_qr(*args, glob=True, hold_back=4)
    expected = _select_order_loop(*args, glob=True, hold_back=4)
    _assert_select_order_equal(res, expected, rtol=1e-10)


def test_ardl_select_order_qr_fallback(data):
    # Singular regressors and exact fits are fitted model by model
    # A constant in fixed and in the trend
    fixed = np.ones((data.y.shape[0], 1))
    always = np.ones((data.y.shape[0], 2))
    select = [np.asarray(data.x)]
    y = np.asarray(data.y)
    assert ardl_model._ardl_select_ssr(y, always, select, False) is None
    res = _select_order_qr(data.y, 2, data.x, 2, "c", fixed=fixed, use_qr=False)
    expected = _select_order_loop(data.y, 2, data.x, 2, "c", fixed=fixed)
    _assert_select_order_equal(res, expected, rtol=0)

    # An exact fit
    y = 1.0 + 0.5 * data.x["lry"] - 0.2 * data.x["ibo"].shift(1).fillna(0.0)
    res = _select_order_qr(y, 2, data.x, 2, "c", use_qr=False)
    expected = _select_order_loop(y, 2, data.x, 2, "c")
    _assert_select_order_equal(res, expected, rtol=0)


@pytest.mark.skipif(PYTHON_IMPL_WASM, reason="linalg error not raised on WASM")
def test_ardl_select_order_nan(data):
    # Data that are not finite use the model by model fits, which raise
    y = data.y.copy()
    y.iloc[20] = np.nan
    # NumPy 1.23 raises a SystemError instead of the LinAlgError
    error = SystemError if NP_LT_1_24 else np.linalg.LinAlgError
    with pytest.raises(error):
        ardl_select_order(y, 2, data.x, 2, "c")


def _contract_ardl_data(n=120):
    rng = np.random.default_rng(20261123)
    x = rng.standard_normal((n, 2))
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.5 * y[t - 1] - 0.2 * y[t - 2] + x[t, 0] + 0.5 * x[t - 1, 1] + e[t]
        )
    return 10 + y, x


def _ardl_design(y, x, lags, order, start):
    # [const, y.L1, ..., y.Llags, x0.L0, ..., x0.Lorder, x1.L0, ...] for
    # observations start, start + 1, ..., nobs - 1
    n = y.shape[0]
    cols = [np.ones(n - start)]
    cols += [y[start - j : n - j] for j in range(1, lags + 1)]
    for i in range(x.shape[1]):
        cols += [x[start - j : n - j, i] for j in range(order + 1)]
    return np.column_stack(cols)


def test_ardl_causal_forecast_without_exog_ndarray():
    # With only lagged exog, the one-step forecast does not need exog_oos.
    # The two-step forecast needs x[T + 1], which is unknown
    y, x = _contract_ardl_data()
    res = ARDL(y, 2, x, [1, 2]).fit()
    p = np.asarray(res.params)
    reg = [1.0, y[-1], y[-2], x[-1, 0], x[-2, 0], x[-1, 1], x[-2, 1]]
    fcast = res.forecast(1)
    assert isinstance(fcast, np.ndarray)
    assert_allclose(fcast[0], np.dot(reg, p), rtol=1e-10)
    with pytest.raises(ValueError, match="exog_oos must be provided"):
        res.forecast(2)


def test_ardl_apply_ndarray_exog():
    # An ndarray exog applies the fitted parameters to the new data, and one
    # with the wrong number of columns raises
    y, x = _contract_ardl_data()
    res = ARDL(y[:80], 2, x[:80], 1).fit()
    new_res = res.apply(y[80:], exog=x[80:])
    expected = _ardl_design(y[80:], x[80:], 2, 1, 2) @ np.asarray(res.params)
    assert_allclose(new_res.fittedvalues, expected, rtol=1e-10)
    with pytest.raises(ValueError, match="number of exog variables"):
        res.apply(y[80:], exog=np.column_stack([x[80:], x[80:, :1]]))


def test_ardl_apply_deterministic_ndarray():
    # The deterministic process of the model is applied to the positions of
    # the new data, so the time trend restarts at 1
    y, x = _contract_ardl_data()
    index = pd.date_range("2000-01-01", periods=y.shape[0], freq="MS")
    dp = DeterministicProcess(index[:80], constant=True, order=1)
    res = ARDL(y[:80], 1, x[:80], 1, trend="n", deterministic=dp).fit()
    new_res = res.apply(y[80:], exog=x[80:])
    design = _ardl_design(y[80:], x[80:], 1, 1, 1)
    trend = np.arange(2.0, 41.0)
    design = np.column_stack([design[:, 0], trend, design[:, 1:]])
    expected = design @ np.asarray(res.params)
    assert_allclose(new_res.fittedvalues, expected, rtol=1e-10)


def test_ardl_apply_deterministic_pandas():
    # The deterministic process of the model is applied to the index of the
    # new data, so calendar dummies follow the dates of the new data
    from statsmodels.tsa.deterministic import CalendarSeasonality

    y, x = _contract_ardl_data()
    index = pd.date_range("2000-01-01", periods=y.shape[0], freq="MS")
    endog = pd.Series(y, index=index, name="y")
    exog = pd.DataFrame(x, index=index, columns=["x0", "x1"])
    terms = [CalendarSeasonality("MS", "A")]
    dp = DeterministicProcess(index[:80], constant=True, additional_terms=terms)
    res = ARDL(endog[:80], 1, exog[:80], 1, trend="n", deterministic=dp).fit()
    new_res = res.apply(endog[80:], exog=exog[80:])
    design = _ardl_design(y[80:], x[80:], 1, 1, 1)
    months = np.asarray(index[81:].month)
    dummies = np.column_stack([months == m for m in range(2, 13)]).astype(float)
    design = np.column_stack([design[:, 0], dummies, design[:, 1:]])
    expected = design @ np.asarray(res.params)
    assert_allclose(np.asarray(new_res.fittedvalues), expected, rtol=1e-10)
    assert list(new_res.fittedvalues.index) == list(index[81:])


def test_ardl_apply_construction_error():
    # An error when building the model for the new data is raised with an
    # explanation in front of the original message
    y, x = _contract_ardl_data()
    res = ARDL(y, 2, x, 1).fit()
    with pytest.raises(ValueError, match="An exception occurred during the") as err:
        res.apply(y[:4], exog=x[:4])
    assert "The number of regressors" in err.value.args[1]


@pytest.mark.parametrize("end", [None, 100])
def test_ardl_get_prediction_in_sample(end):
    # In sample, the variance of the prediction is sigma2 and the mean is
    # the fitted value
    y, x = _contract_ardl_data()
    index = pd.date_range("2000-01-01", periods=y.shape[0], freq="MS")
    endog = pd.Series(y, index=index, name="y")
    exog = pd.DataFrame(x, index=index, columns=["x0", "x1"])
    res = ARDL(endog, 2, exog, 1).fit()
    if end is not None:
        end = index[end]
    pred = res.get_prediction(start=index[2], end=end)
    resid = y[2:] - _ardl_design(y, x, 2, 1, 2) @ np.asarray(res.params)
    sigma2 = resid @ resid / resid.shape[0]
    n_pred = 118 if end is None else 99
    assert_allclose(
        np.asarray(pred.var_pred_mean), np.full(n_pred, sigma2), rtol=1e-10
    )
    assert_allclose(
        np.asarray(pred.predicted_mean),
        y[2 : 2 + n_pred] - resid[:n_pred],
        rtol=1e-10,
    )
    assert list(pred.predicted_mean.index) == list(index[2 : 2 + n_pred])


def _contract_ardl_endog_forms(y):
    from statsmodels.tsa.tests._contract import series_forms

    forms = dict(series_forms(y))
    forms["period"] = pd.Series(
        y, index=pd.period_range("2000-01", periods=y.shape[0], freq="M")
    )
    forms["range"] = pd.Series(y, index=pd.RangeIndex(y.shape[0]))
    return forms


_ENDOG_FORMS = ["list", "tuple", "series", "column", "dataframe", "period", "range"]


@pytest.mark.parametrize("form", _ENDOG_FORMS)
def test_ardl_endog_input_forms(form):
    # endog is documented as array_like: every form of a 1-D series gives the
    # OLS estimates of the ndarray, and pandas input gives pandas predictions
    y, x = _contract_ardl_data()
    endog = _contract_ardl_endog_forms(y)[form]
    res = ARDL(endog, 2, x, 1).fit()
    expected = np.linalg.lstsq(_ardl_design(y, x, 2, 1, 2), y[2:], rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)
    pred = res.predict()
    if isinstance(endog, (pd.Series, pd.DataFrame)):
        assert isinstance(pred, pd.Series)
        assert list(pred.index) == list(endog.index)
    else:
        assert isinstance(pred, np.ndarray)


def test_ardl_endog_integer():
    # Integer endog gives the result of the same values as floats
    y, x = _contract_ardl_data()
    y_int = np.round(4 * y).astype(np.int64)
    res = ARDL(y_int, 2, x, 1).fit()
    expected = ARDL(y_int.astype(float), 2, x, 1).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)


@pytest.mark.parametrize("form", ["list", "dataframe", "integer", "1d"])
def test_ardl_exog_input_forms(form):
    # exog is documented as array_like
    y, x = _contract_ardl_data()
    if form == "integer":
        x = np.round(4 * x).astype(np.int64)
    elif form == "1d":
        x = x[:, :1]
    if form == "list":
        exog = x.tolist()
    elif form == "dataframe":
        exog = pd.DataFrame(x, columns=["a", "b"])
    elif form == "1d":
        exog = x[:, 0]
    else:
        exog = x
    res = ARDL(y, 2, exog, 1).fit()
    expected = np.linalg.lstsq(_ardl_design(y, x, 2, 1, 2), y[2:], rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)


@pytest.mark.parametrize(
    "order",
    [1, [0, 1], (0, 1), np.array([0, 1]), {0: 1, 1: 1}, {0: [0, 1], 1: (0, 1)}],
    ids=["int", "list", "tuple", "ndarray", "dict", "dict-of-sequences"],
)
def test_ardl_order_forms(order):
    # order is documented as int, sequence of int, or dict
    y, x = _contract_ardl_data()
    res = ARDL(y, 2, x, order).fit()
    expected = np.linalg.lstsq(_ardl_design(y, x, 2, 1, 2), y[2:], rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)
    assert res.model.dl_lags == {0: [0, 1], 1: [0, 1]}


@pytest.mark.parametrize(
    "lags", [[1, 2], (1, 2), np.array([1, 2]), np.int64(2)], ids=str
)
def test_ardl_lags_forms(lags):
    # lags is documented as int or sequence of int
    y, x = _contract_ardl_data()
    res = ARDL(y, lags, x, 1).fit()
    expected = np.linalg.lstsq(_ardl_design(y, x, 2, 1, 2), y[2:], rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)


@pytest.mark.parametrize(
    "trend, names",
    [
        ("n", []),
        ("c", ["const"]),
        ("t", ["trend"]),
        ("ct", ["const", "trend"]),
        ("ctt", ["const", "trend", "trend_squared"]),
    ],
)
def test_ardl_uecm_trend_values(trend, names):
    # Each documented trend adds its terms in front of the other regressors
    y, x = _contract_ardl_data()
    res = ARDL(y, 1, x[:, :1], 1, trend=trend).fit()
    assert res.model.exog_names == names + ["y.L1", "x0.L0", "x0.L1"]
    t = np.arange(2.0, 121.0)
    terms = {"n": [], "c": [1.0], "t": [t], "ct": [1.0, t], "ctt": [1.0, t, t**2]}
    det = [np.broadcast_to(term, (119,)) for term in terms[trend]]
    reg = np.column_stack(det + [y[:-1], x[1:, 0], x[:-1, 0]])
    expected = np.linalg.lstsq(reg, y[1:], rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-8)
    ures = UECM(y, 1, x[:, :1], 1, trend=trend).fit()
    assert ures.model.exog_names == names + ["y.L1", "x0.L1", "D.x0.L0"]
    # The UECM is a reparametrization of the same ARDL
    assert_allclose(ures.llf, res.llf, rtol=1e-10)


@pytest.mark.parametrize("func", ["ARDL", "UECM", "ardl_select_order"])
def test_ardl_invalid_trend(func):
    y, x = _contract_ardl_data()
    with pytest.raises(ValueError, match="trend must be one of"):
        if func == "ARDL":
            ARDL(y, 1, x, 1, trend="x")
        elif func == "UECM":
            UECM(y, 1, x, 1, trend="x")
        else:
            ardl_select_order(y, 1, x, 1, trend="x")


def test_ardl_missing_values():
    # missing="drop" drops the observations with NaNs, "raise" raises
    from statsmodels.tools.sm_exceptions import MissingDataError

    y, x = _contract_ardl_data()
    y_nan = y.copy()
    y_nan[5] = np.nan
    res = ARDL(y_nan, 1, x, 1, missing="drop").fit()
    expected = ARDL(np.delete(y, 5), 1, np.delete(x, 5, axis=0), 1).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)
    with pytest.raises(MissingDataError):
        ARDL(y_nan, 1, x, 1, missing="raise")
    with pytest.raises(ValueError, match="missing must be one of"):
        ARDL(y_nan, 1, x, 1, missing="x")


@pytest.mark.parametrize("form", ["ndarray", "list", "dataframe", "1d", "series"])
def test_ardl_fixed_input_forms(form):
    # fixed is documented as array_like and is included without lags
    y, x = _contract_ardl_data()
    fixed = np.random.default_rng(20261124).standard_normal((120, 1))
    index = pd.date_range("2000-01-01", periods=120, freq="MS")
    fixed_form = {
        "ndarray": fixed,
        "list": fixed.tolist(),
        "dataframe": pd.DataFrame(fixed, index=index, columns=["f"]),
        "1d": fixed[:, 0],
        "series": pd.Series(fixed[:, 0], index=index, name="f"),
    }[form]
    endog = pd.Series(y, index=index, name="y")
    exog = pd.DataFrame(x, index=index, columns=["x0", "x1"])
    res = ARDL(endog, 2, exog, 1, fixed=fixed_form).fit()
    reg = np.column_stack([_ardl_design(y, x, 2, 1, 2), fixed[2:]])
    expected = np.linalg.lstsq(reg, y[2:], rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)


@pytest.mark.parametrize("form", _ENDOG_FORMS)
def test_ardl_select_order_endog_input_forms(form):
    # endog is documented as array_like: every form selects the same model
    # with the same criteria
    y, x = _contract_ardl_data()
    expected = ardl_select_order(y, 2, x, 1, ic="aic")
    res = ardl_select_order(_contract_ardl_endog_forms(y)[form], 2, x, 1, ic="aic")
    assert res.model.ardl_order == expected.model.ardl_order
    assert_allclose(
        np.asarray(res.aic.index), np.asarray(expected.aic.index), rtol=1e-10
    )


def _uecm_design(y, x, start):
    # UECM(y, 2, x, 2): [const, y.L1, x0.L1, x1.L1, D.y.L1, D.x0.L0, D.x0.L1,
    # D.x1.L0, D.x1.L1] for observations start, ..., nobs - 1
    n = y.shape[0]
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.r_[np.full((1, x.shape[1]), np.nan), np.diff(x, axis=0)]
    cols = [np.ones(n - start), y[start - 1 : n - 1]]
    cols += [x[start - 1 : n - 1, i] for i in range(x.shape[1])]
    cols += [dy[start - 1 : n - 1]]
    for i in range(x.shape[1]):
        cols += [dx[start - j : n - j, i] for j in range(2)]
    return np.column_stack(cols), dy[start:]


def _contract_uecm_data():
    y, x = _contract_ardl_data()
    return y.cumsum() / 10, x.cumsum(0)


@pytest.mark.parametrize("form", _ENDOG_FORMS)
def test_uecm_endog_input_forms(form):
    # endog is documented as array_like: every form of a 1-D series gives the
    # OLS estimates of the regression of the difference
    y, x = _contract_uecm_data()
    res = UECM(_contract_ardl_endog_forms(y)[form], 2, x, 2).fit()
    reg, dy = _uecm_design(y, x, 2)
    expected = np.linalg.lstsq(reg, dy, rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)


@pytest.mark.parametrize("form", ["list", "dataframe"])
def test_uecm_exog_input_forms(form):
    # exog is documented as a DataFrame or a 2-d array_like
    y, x = _contract_uecm_data()
    exog = x.tolist() if form == "list" else pd.DataFrame(x, columns=["a", "b"])
    res = UECM(y, 2, exog, 2).fit()
    reg, dy = _uecm_design(y, x, 2)
    expected = np.linalg.lstsq(reg, dy, rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)


@pytest.mark.parametrize(
    "order", [2, np.int64(2), {0: 2, 1: 2}], ids=["int", "np.int64", "dict"]
)
def test_uecm_order_forms(order):
    # order is documented as int or dict; a sequence raises, as does a
    # sequence of lags
    y, x = _contract_uecm_data()
    res = UECM(y, 2, x, order).fit()
    reg, dy = _uecm_design(y, x, 2)
    expected = np.linalg.lstsq(reg, dy, rcond=None)[0]
    assert_allclose(np.asarray(res.params), expected, rtol=1e-10)
    with pytest.raises(TypeError, match="order must be None"):
        UECM(y, 2, x, [1, 2])
    with pytest.raises(TypeError, match="lags must be an integer or None"):
        UECM(y, [1, 2], x, 2)


def test_uecm_results_ci_types():
    # pandas input gives pandas output of the cointegrating vector, labelled
    # by the level variables, and ndarray input gives ndarrays
    y, x = _contract_uecm_data()
    index = pd.date_range("2000-01-01", periods=120, freq="MS")
    endog = pd.Series(y, index=index, name="y")
    exog = pd.DataFrame(x, index=index, columns=["a", "b"])
    res = UECM(endog, 2, exog, 2).fit()
    res_np = UECM(y, 2, x, 2).fit()
    labels = ["const", "y", "a", "b"]
    for attr in ("ci_params", "ci_bse", "ci_tvalues", "ci_pvalues"):
        value = getattr(res, attr)
        assert isinstance(value, pd.Series)
        assert list(value.index) == labels
        value_np = getattr(res_np, attr)
        assert isinstance(value_np, np.ndarray)
        assert_allclose(np.asarray(value), value_np, rtol=1e-10)
    # The cointegrating vector is normalized by the coefficient on y.L1
    params = np.asarray(res.params)[:4]
    assert_allclose(np.asarray(res.ci_params), params / params[1], rtol=1e-12)
    ci = res.ci_conf_int()
    assert isinstance(ci, pd.DataFrame)
    assert list(ci.columns) == ["lower", "upper"]
    assert_allclose(res_np.ci_conf_int(), np.asarray(ci), rtol=1e-10)
    assert isinstance(res.ci_cov_params(), pd.DataFrame)
    assert isinstance(res.ci_summary(), Summary)
    resids = res.ci_resids
    assert isinstance(resids, pd.Series)
    assert list(resids.index) == list(index)
    expected = np.column_stack([np.ones(120), y, x]) @ (params / params[1])
    assert_allclose(np.asarray(resids), expected, rtol=1e-10)
    assert_allclose(res_np.ci_resids, expected, rtol=1e-10)


@pytest.mark.parametrize("to_int", [int, np.int64, np.int32])
@pytest.mark.parametrize("causal", [False, True])
def test_ardl_order_dict_numpy_int(to_int, causal):
    # numpy integer values in an order dict raised TypeError
    rng = np.random.default_rng(20261128)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 2)), axis=0)
    order = {0: to_int(2), 1: to_int(1)}
    res = ARDL(y, 1, x, order, causal=causal).fit()

    first = int(causal)
    assert res.model.dl_lags == {0: list(range(first, 3)), 1: list(range(first, 2))}
    t = np.arange(2, nobs)
    reg = [np.ones(nobs - 2), y[t - 1]]
    reg += [x[t - j, 0] for j in range(first, 3)]
    reg += [x[t - j, 1] for j in range(first, 2)]
    ols = OLS(y[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)


@pytest.mark.parametrize("to_int", [int, np.int64, np.int32])
def test_uecm_order_dict_numpy_int(to_int):
    # numpy integer values in an order dict raised TypeError
    rng = np.random.default_rng(20261129)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 2)), axis=0)
    res = UECM(y, 1, x, {0: to_int(2), 1: to_int(1)}).fit()

    # D.y on const, y.L1, x0.L1, x1.L1, D.x0.L0, D.x0.L1 and D.x1.L0
    t = np.arange(2, nobs)
    dx = np.r_[np.full((1, 2), np.nan), np.diff(x, axis=0)]
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 1]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 1]]
    ols = OLS(y[t] - y[t - 1], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)


@pytest.mark.parametrize("to_int", [np.int64, np.int32])
def test_ardl_select_order_maxorder_dict_numpy_int(to_int):
    # The same selection as with int values, which raised TypeError
    rng = np.random.default_rng(20261130)
    y = np.cumsum(rng.standard_normal(150))
    x = np.cumsum(rng.standard_normal((150, 2)), axis=0)
    res = ardl_select_order(y, 2, x, {0: to_int(2), 1: to_int(1)})
    expected = ardl_select_order(y, 2, x, {0: 2, 1: 1})
    assert_allclose(res.bic.index, expected.bic.index, rtol=1e-12)
    assert list(res.bic) == list(expected.bic)
    assert res.model.ardl_order == expected.model.ardl_order


@pytest.mark.parametrize("use_pandas", [True, False])
def test_uecm_excluded_exog_by_hand(use_pandas):
    # An exog variable excluded with None has no level, no differences and no
    # term in the cointegrating relationship
    rng = np.random.default_rng(20261132)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 3)), axis=0)
    if use_pandas:
        endog = pd.Series(y, name="y")
        exog = pd.DataFrame(x, columns=["a", "b", "c"])
        order = {"a": 2, "b": None, "c": 1}
    else:
        endog, exog = y, x
        order = {0: 2, 1: None, 2: 1}
    res = UECM(endog, 1, exog, order).fit()

    a, c = ("a", "c") if use_pandas else ("x0", "x2")
    names = ["const", "y.L1", f"{a}.L1", f"{c}.L1", f"D.{a}.L0", f"D.{a}.L1"]
    names.append(f"D.{c}.L0")
    assert res.model.exog_names == names
    t = np.arange(2, nobs)
    dx = np.r_[np.full((1, 3), np.nan), np.diff(x, axis=0)]
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 2]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 2]]
    params = OLS(y[t] - y[t - 1], np.column_stack(reg)).fit().params
    assert_allclose(res.params, params, rtol=1e-8)
    # The relationship is normalized by the coefficient of y.L1
    ci_params = params[:4] / params[1]
    assert_allclose(res.ci_params, ci_params, rtol=1e-8)
    ci_resids = np.column_stack([np.ones(nobs), y, x[:, 0], x[:, 2]]) @ ci_params
    assert_allclose(res.ci_resids, ci_resids, rtol=1e-8)
    if use_pandas:
        assert list(res.ci_params.index) == ["const", "y", "a", "c"]


def _ardl_uecm_regressors(y, x, model, lags, order, hold_back):
    # The regressors of an ARDL or a UECM with a constant, built by hand
    n = y.shape[0]
    lags = lags or 0
    t = np.arange(hold_back, n)
    cols = [np.ones(t.shape[0])]
    if model == "ARDL":
        cols += [y[t - j] for j in range(1, lags + 1)]
        cols += [x[t - j] for j in range(order + 1)]
        return y[t], np.column_stack(cols)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.r_[np.nan, np.diff(x)]
    cols += [y[t - 1], x[t - 1]]
    cols += [dy[t - j] for j in range(1, lags)]
    cols += [dx[t - j] for j in range(order)]
    return dy[t], np.column_stack(cols)


@pytest.mark.parametrize(
    "model, lags, order, hold_back, nobs",
    [
        ("ARDL", 1, 4, None, 96),
        ("ARDL", None, 3, None, 97),
        ("ARDL", 1, 4, 6, 94),
        ("ARDL", 4, 1, None, 96),
        ("UECM", 1, 3, None, 97),
        ("UECM", 3, 1, None, 97),
    ],
)
def test_ardl_nobs_exog_lag_hold_back(model, lags, order, hold_back, nobs):
    # When the largest exog lag exceeded the endog lag and hold_back was
    # None, nobs counted the observations dropped by the exog lags, and
    # llf, the ICs, sigma2, df_resid and the p-values used it
    rng = np.random.default_rng(20261124)
    y = np.cumsum(rng.standard_normal(100))
    x = np.cumsum(rng.standard_normal(100))
    mod_cls = ARDL if model == "ARDL" else UECM
    res = mod_cls(y, lags, x[:, None], order, hold_back=hold_back).fit()

    hb = hold_back if hold_back is not None else 100 - nobs
    dep, reg = _ardl_uecm_regressors(y, x, model, lags, order, hb)
    ols = OLS(dep, reg).fit()
    k = reg.shape[1]
    assert res.nobs == nobs == dep.shape[0]
    assert res.df_resid == nobs - k
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.llf, ols.llf, rtol=1e-10)
    assert_allclose(res.sigma2, ols.ssr / nobs, rtol=1e-10)
    # The ICs count the variance as a parameter
    assert_allclose(res.aic, -2 * ols.llf + 2 * (k + 1), rtol=1e-10)
    assert_allclose(res.bic, -2 * ols.llf + np.log(nobs) * (k + 1), rtol=1e-10)
    hqic = -2 * ols.llf + 2 * np.log(np.log(nobs)) * (k + 1)
    assert_allclose(res.hqic, hqic, rtol=1e-10)
    assert_allclose(res.pvalues, ols.pvalues, rtol=1e-8)
    assert_allclose(res.conf_int(), ols.conf_int(), rtol=1e-8)


def _bic_by_hand(y, reg):
    # BIC as ARDLResults computes it, with the variance counted as a parameter
    nobs, k = reg.shape
    resid = y - reg @ np.linalg.lstsq(reg, y, rcond=None)[0]
    ssr = resid @ resid
    llf = -nobs / 2 * (np.log(2 * np.pi * ssr / nobs) + 1)
    return -2 * llf + np.log(nobs) * (k + 1)


@pytest.mark.parametrize("singular", [False, True])
def test_ardl_select_order_glob_all_subsets(singular):
    # glob=True never fitted the model with all candidate lags, which is the
    # true model here
    rng = np.random.default_rng(20261125)
    nobs = 300
    x = rng.standard_normal(nobs)
    e = rng.standard_normal(nobs)
    y = np.zeros(nobs)
    for t in range(2, nobs):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + x[t] + 0.8 * x[t - 1] + e[t]
    # A second constant makes the regressors singular, so that the models
    # are fitted one by one
    fixed = np.ones((nobs, 1)) if singular else None
    res = ardl_select_order(y, 2, x[:, None], 1, "c", fixed=fixed, glob=True)

    # All 2**4 subsets of the candidate lags, on the sample after 2 lags
    t = np.arange(2, nobs)
    always = [np.ones(nobs - 2)] * (2 if singular else 1)
    cands = {
        ("y", 1): y[t - 1],
        ("y", 2): y[t - 2],
        ("x", 0): x[t],
        ("x", 1): x[t - 1],
    }
    bics = {}
    for size in range(len(cands) + 1):
        for subset in combinations(cands, size):
            reg = np.column_stack(always + [cands[c] for c in subset])
            bics[subset] = _bic_by_hand(y[t], reg)
    assert len(res.bic) == 16
    assert_allclose(res.bic.index, np.sort(list(bics.values())), rtol=1e-8)
    best = min(bics, key=bics.get)
    assert best == tuple(cands)
    assert res.model.ar_lags == [1, 2]
    assert res.model.dl_lags == {0: [0, 1]}


@pytest.mark.parametrize("singular", [False, True])
def test_ardl_select_order_glob_no_candidates(singular):
    # With no candidate lags, glob=True fitted no model and raised
    # UnboundLocalError
    rng = np.random.default_rng(20261126)
    y = 1.0 + rng.standard_normal(80)
    fixed = np.ones((80, 1)) if singular else None
    res = ardl_select_order(y, 0, None, 0, "c", fixed=fixed, glob=True)
    reg = np.ones((80, 2 if singular else 1))
    assert_allclose(res.bic.index, [_bic_by_hand(y, reg)], rtol=1e-8)
    assert res.model.ar_lags is None
    assert res.model.dl_lags == {}


def _dane_uecm_res():
    exog = dane_data[["lry", "ibo", "ide"]]
    return UECM(dane_data.lrm, 3, exog, {"lry": 1, "ibo": 3, "ide": 2}).fit()


@pytest.mark.parametrize("case", [0, 6, -1, 2.5, "3", None])
def test_bounds_test_invalid_case(case):
    # Values outside 1, ..., 5 raised UnboundLocalError
    res = _dane_uecm_res()
    with pytest.raises(ValueError, match="case must be one of 1, 2, 3, 4 or 5"):
        res.bounds_test(case)


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("to_case", [np.int64, float])
def test_bounds_test_case_types(case, to_case):
    # Integer values of other types keep working; the statistics of
    # test_bounds_test, given to 7 significant digits
    expected = {1: 0.7109023, 2: 5.116768, 3: 6.205875, 4: 5.430622, 5: 6.785325}
    bounds_result = _dane_uecm_res().bounds_test(to_case(case))
    assert_allclose(bounds_result.statistic, expected[case], rtol=1e-6)


@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(2, 42), pd.RangeIndex(10, 90, 2), pd.RangeIndex(40, 0, -1)],
    ids=["start", "step", "decreasing"],
)
def test_ardl_uecm_range_index_labels(index):
    # The default end was the last label of the index used as a location, so
    # fittedvalues and get_prediction raised (or UECM.predict stopped early),
    # and forecasts were labelled as if the index were 0, 1, ..., nobs - 1
    rng = np.random.default_rng(20261206)
    nobs = 40
    y = rng.standard_normal(nobs).cumsum()
    x = rng.standard_normal((nobs + 3, 1))
    start = index.start + nobs * index.step
    oos_index = pd.RangeIndex(start, start + 3 * index.step, index.step)
    endog = pd.Series(y, index=index)
    exog = pd.DataFrame(x[:nobs], index=index, columns=["x"])
    exog_oos = pd.DataFrame(x[nobs:], index=oos_index, columns=["x"])
    # The same models on the default index
    default_exog = pd.DataFrame(x[:nobs], columns=["x"])
    default_exog_oos = pd.DataFrame(x[nobs:], columns=["x"])

    res = ARDL(endog, 2, exog, 1, trend="ct").fit()
    expected = ARDL(pd.Series(y), 2, default_exog, 1, trend="ct").fit()
    fitted = res.fittedvalues
    assert_index_equal(fitted.index, index[2:])
    assert_allclose(fitted.to_numpy(), expected.fittedvalues.to_numpy(), rtol=1e-10)
    pred = res.predict()
    assert_index_equal(pred.index, index)
    assert_allclose(pred.to_numpy(), expected.predict().to_numpy(), rtol=1e-10)
    pred_res = res.get_prediction()
    assert_index_equal(pred_res.predicted_mean.index, index)
    expected_var = expected.get_prediction().var_pred_mean.to_numpy()
    assert_allclose(pred_res.var_pred_mean.to_numpy(), expected_var, rtol=1e-10)
    fcast = res.forecast(3, exog=exog_oos)
    assert_index_equal(fcast.index, oos_index)
    expected_fcast = expected.forecast(3, exog=default_exog_oos).to_numpy()
    assert_allclose(fcast.to_numpy(), expected_fcast, rtol=1e-10)

    # UECM predicts in sample only; its hold back is 2
    res = UECM(endog, 2, exog, 1).fit()
    expected = UECM(pd.Series(y), 2, default_exog, 1).fit()
    fitted = res.fittedvalues
    assert_index_equal(fitted.index, index[2:])
    assert_allclose(fitted.to_numpy(), expected.fittedvalues.to_numpy(), rtol=1e-10)
    pred = res.predict()
    assert_index_equal(pred.index, index)
    assert_allclose(pred.to_numpy(), expected.predict().to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("to_int", [np.int64, np.int32])
@pytest.mark.parametrize("causal", [False, True])
def test_ardl_order_dict_numpy_int_keys(to_int, causal):
    # numpy integer keys of an order dict raised AssertionError for ndarray
    # exog
    rng = np.random.default_rng(20261201)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 2)), axis=0)
    res = ARDL(y, 1, x, {to_int(0): 2, to_int(1): [1]}, causal=causal).fit()

    first = int(causal)
    assert res.model.dl_lags == {0: list(range(first, 3)), 1: [1]}
    assert all(type(key) is int for key in res.model.dl_lags)
    t = np.arange(2, nobs)
    reg = [np.ones(nobs - 2), y[t - 1]]
    reg += [x[t - j, 0] for j in range(first, 3)]
    reg += [x[t - 1, 1]]
    ols = OLS(y[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)


@pytest.mark.parametrize("to_int", [np.int64, np.int32])
def test_uecm_order_dict_numpy_int_keys(to_int):
    # numpy integer keys of an order dict raised AssertionError for ndarray
    # exog
    rng = np.random.default_rng(20261202)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 2)), axis=0)
    res = UECM(y, 1, x, {to_int(0): 2, to_int(1): 1}).fit()

    # D.y on const, y.L1, x0.L1, x1.L1, D.x0.L0, D.x0.L1 and D.x1.L0
    t = np.arange(2, nobs)
    dx = np.r_[np.full((1, 2), np.nan), np.diff(x, axis=0)]
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 1]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 1]]
    ols = OLS(y[t] - y[t - 1], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)


@pytest.mark.parametrize("to_int", [np.int64, np.int32])
def test_ardl_select_order_maxorder_dict_numpy_int_keys(to_int):
    # The same selection as with int keys, which raised AssertionError
    rng = np.random.default_rng(20261203)
    y = np.cumsum(rng.standard_normal(150))
    x = np.cumsum(rng.standard_normal((150, 2)), axis=0)
    res = ardl_select_order(y, 2, x, {to_int(0): 2, to_int(1): 1})
    expected = ardl_select_order(y, 2, x, {0: 2, 1: 1})
    assert_allclose(res.bic.index, expected.bic.index, rtol=1e-12)
    assert list(res.bic) == list(expected.bic)
    assert res.model.ardl_order == expected.model.ardl_order


@pytest.mark.parametrize("endog_int", [True, False])
@pytest.mark.parametrize("exog_int", [True, False])
@pytest.mark.parametrize("use_pandas", [True, False])
def test_uecm_integer_data(endog_int, exog_int, use_pandas):
    # The levels of exog were truncated to integers when endog was an integer
    # array, and integer data warned about casting NaN to an integer
    rng = np.random.default_rng(20261127)
    nobs = 120
    y = np.round(4 * np.cumsum(rng.standard_normal(nobs)))
    x = np.cumsum(rng.standard_normal(nobs))
    if exog_int:
        x = np.round(3 * x)
    endog = y.astype(np.int64) if endog_int else y
    exog = x.astype(np.int64) if exog_int else x
    if use_pandas:
        endog = pd.Series(endog, name="y")
        exog = pd.DataFrame({"x": exog})
    else:
        exog = exog[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = UECM(endog, 2, exog, 2).fit()

    # D.y on const, y.L1, x.L1, D.y.L1, D.x.L0 and D.x.L1
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.r_[np.nan, np.diff(x)]
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1], dy[t - 1], dx[t], dx[t - 1]]
    ols = OLS(dy[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.bse, ols.bse, rtol=1e-8)


@pytest.mark.parametrize("kind", ["ndarray", "dataframe", "mixed"])
def test_uecm_bool_exog(kind):
    # np.diff and DataFrame.diff take the xor of bool values, so the
    # differences of bool exog were their absolute values
    rng = np.random.default_rng(20261128)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    xb = rng.standard_normal((nobs, 2)) > 0
    x = xb.astype(float)
    if kind == "ndarray":
        exog = xb
    else:
        exog = pd.DataFrame(xb, columns=["a", "b"])
        if kind == "mixed":
            # A float column next to the bool one
            x[:, 1] = rng.standard_normal(nobs)
            exog["b"] = x[:, 1]
    res = UECM(y, 2, exog, 2).fit()

    # D.y on const, y.L1, the levels x.L1 and the differences D.y.L1, D.x.L0
    # and D.x.L1, with the bool values as 0 and 1
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.vstack([np.full((1, 2), np.nan), np.diff(x, axis=0)])
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 1], dy[t - 1]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 1], dx[t - 1, 1]]
    ols = OLS(dy[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.bse, ols.bse, rtol=1e-8)


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_ardl_select_order_glob_no_lags_label(causal, singular):
    # With glob=True, the model without any candidate lags was listed with
    # exog order 0, so the selected model included lag 0 of every exog
    # variable (causal=True raised ValueError) when that model had the
    # smallest BIC, as it does here
    rng = np.random.default_rng(20261123)
    nobs = 100
    y = rng.standard_normal(nobs)
    x = rng.standard_normal((nobs, 2))
    fixed = np.ones((nobs, 1)) if singular else None
    res = ardl_select_order(
        y, 1, x, int(causal), "c", fixed=fixed, causal=causal, glob=True
    )

    def lags_of(order, first):
        # The lags that an ARDL order includes
        if order is None:
            return []
        if isinstance(order, int):
            return list(range(first, order + 1))
        return list(order)

    # Each listed order against the regression on the lags it includes, on
    # the sample after 1 lag
    t = np.arange(1, nobs)
    always = [np.ones(nobs - 1)] * (2 if singular else 1)
    assert len(res.bic) == 8
    for value, (lags, order) in res.bic.items():
        cols = [y[t - lag] for lag in lags_of(lags, 1)]
        for key, exog_order in order.items():
            cols += [x[t - lag, key] for lag in lags_of(exog_order, int(causal))]
        reg = np.column_stack(always + cols)
        assert_allclose(value, _bic_by_hand(y[t], reg), rtol=1e-8)
    assert res.bic.iloc[0] == (0, {0: None, 1: None})
    assert res.model.ar_lags is None
    assert res.model.dl_lags == {}


@pytest.mark.parametrize(
    "order",
    [
        {0.0: 1, 1.0: 1},
        {np.float64(0): 1, np.float64(1): 1},
        {0: 1, 1.0: 2},
        {True: 1, 0: 1},
        {np.True_: 1, 0: 1},
    ],
    ids=["float", "np.float64", "one-float", "bool", "np.bool"],
)
@pytest.mark.parametrize("func", ["ARDL", "UECM", "select"])
def test_ardl_order_dict_non_integer_keys(func, order):
    # Float and bool keys equal to a column index raised AssertionError,
    # IndexError or ValueError from the construction of the regressors
    rng = np.random.default_rng(20261124)
    y = rng.standard_normal(100).cumsum()
    x = rng.standard_normal((100, 2)).cumsum(0)
    with pytest.raises(TypeError, match="order dictionary keys must be integer"):
        if func == "ARDL":
            ARDL(y, 1, x, order)
        elif func == "UECM":
            UECM(y, 1, x, order)
        else:
            ardl_select_order(y, 1, x, order)


def test_ardl_order_dict_float_column_labels():
    # Float keys select the columns of a DataFrame with float labels
    rng = np.random.default_rng(20261124)
    nobs = 100
    y = rng.standard_normal(nobs)
    x = rng.standard_normal((nobs, 2))
    exog = pd.DataFrame(x, columns=[0.0, 1.0])
    res = ARDL(y, 1, exog, {0.0: 1, 1.0: 2}).fit()
    assert res.model.dl_lags == {0.0: [0, 1], 1.0: [0, 1, 2]}

    # y on const, y.L1, x0.L0, x0.L1, x1.L0, x1.L1 and x1.L2
    t = np.arange(2, nobs)
    reg = [np.ones(nobs - 2), y[t - 1], x[t, 0], x[t - 1, 0]]
    reg += [x[t, 1], x[t - 1, 1], x[t - 2, 1]]
    ols = OLS(y[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-10)


def _ardl_columns_res(columns, seed):
    # An ARDL with a constant, 2 lags of y, lags 0-2 of the first exog column
    # and lag 0 of the second
    rng = np.random.default_rng(seed)
    nobs = 60
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 2)), axis=0)
    a, b = columns
    res = ARDL(y, 2, pd.DataFrame(x, columns=columns), {a: 2, b: 0}).fit()
    names = ["const", "y.L1", "y.L2", f"{a}.L0", f"{a}.L1", f"{a}.L2", f"{b}.L0"]
    assert res.model.exog_names == names
    return rng, y, x, res


@pytest.mark.parametrize("columns", [["a", "b"], [1, 0]], ids=["str", "int"])
def test_ardl_predict_replacement_exog_columns_by_name(columns):
    # The columns of a replacement exog DataFrame were used by position, so
    # that a DataFrame with the columns in another order gave wrong
    # predictions
    rng, y, x, res = _ardl_columns_res(columns, 20261201)
    nobs = y.shape[0]
    x2 = x + rng.standard_normal((nobs, 2))
    exog2 = pd.DataFrame(x2, columns=columns)
    a, b = columns
    pred = np.asarray(res.predict(exog=exog2[[b, a]]))

    # The regressors const, y.L1, y.L2, a.L0, a.L1, a.L2 and b.L0 built from
    # the replacement x2
    t = np.arange(2, nobs)
    reg = [np.ones(nobs - 2), y[t - 1], y[t - 2]]
    reg += [x2[t, 0], x2[t - 1, 0], x2[t - 2, 0], x2[t, 1]]
    expected = np.column_stack(reg) @ np.asarray(res.params)
    assert np.all(np.isnan(pred[:2]))
    assert_allclose(pred[2:], expected, rtol=1e-10)
    # The same as with the columns in the original order
    assert_allclose(res.predict(exog=exog2), pred, rtol=1e-12)
    prediction = res.get_prediction(exog=exog2[[b, a]])
    assert_allclose(prediction.predicted_mean, pred, rtol=1e-12)


@pytest.mark.parametrize("columns", [["a", "b"], [1, 0]], ids=["str", "int"])
def test_ardl_forecast_exog_oos_columns_by_name(columns):
    # The columns of an exog_oos DataFrame (and of a replacement exog) were
    # used by position, so that a DataFrame with the columns in another order
    # gave wrong forecasts
    rng, y, x, res = _ardl_columns_res(columns, 20261202)
    nobs = y.shape[0]
    x_oos = rng.standard_normal((3, 2))
    x2 = x + rng.standard_normal((nobs, 2))
    a, b = columns
    oos = pd.DataFrame(x_oos, columns=columns)[[b, a]]
    exog2 = pd.DataFrame(x2, columns=columns)[[b, a]]

    def forecasts(x_in):
        # y[t] = c + p1 y[t-1] + p2 y[t-2] + a0 xa[t] + a1 xa[t-1]
        #        + a2 xa[t-2] + b0 xb[t]
        c, p1, p2, a0, a1, a2, b0 = np.asarray(res.params)
        xa = np.r_[x_in[:, 0], x_oos[:, 0]]
        xb = np.r_[x_in[:, 1], x_oos[:, 1]]
        path = list(y)
        for t in range(nobs, nobs + 3):
            path.append(
                c + p1 * path[t - 1] + p2 * path[t - 2] + a0 * xa[t]
                + a1 * xa[t - 1] + a2 * xa[t - 2] + b0 * xb[t]
            )
        return np.array(path[nobs:])

    expected = forecasts(x)
    fcast = np.asarray(res.forecast(3, exog=oos))
    assert_allclose(fcast, expected, rtol=1e-10)
    # The same as with the columns in the original order
    assert_allclose(res.forecast(3, exog=oos[columns]), fcast, rtol=1e-12)
    pred = res.predict(start=nobs + 1, end=nobs + 2, exog_oos=oos)
    assert_allclose(pred, expected[1:], rtol=1e-10)
    # With a replacement exog, the lags of a in the forecasts come from it
    pred = res.predict(start=nobs, end=nobs + 2, exog=exog2, exog_oos=oos)
    assert_allclose(pred, forecasts(x2), rtol=1e-10)


def _as_exog_form(values, form):
    # A 1-d exog in the given form
    if form == "1d":
        return values
    if form == "series":
        return pd.Series(values, name="x")
    return values[:, None]


@pytest.mark.parametrize("repl", ["1d", "series", "column"])
@pytest.mark.parametrize("form", ["1d", "series"])
def test_ardl_predict_1d_exog(form, repl):
    # With a 1-d array or a Series exog, predict and forecast with a
    # replacement exog or exog_oos raised IndexError
    rng = np.random.default_rng(20261203)
    nobs = 60
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal(nobs))
    x2 = x + rng.standard_normal(nobs)
    x_oos = rng.standard_normal(3)
    res = ARDL(y, 2, _as_exog_form(x, form), 2).fit()
    # The single variable has the key 0
    base = "0" if form == "series" else "x0"
    assert res.model.exog_names[3:] == [f"{base}.L{i}" for i in range(3)]
    params = np.asarray(res.params)

    # The replacement equal to the original exog gives the predictions
    pred = res.predict(exog=_as_exog_form(x, repl))
    assert_allclose(pred, res.predict(), rtol=1e-12)
    # The regressors const, y.L1, y.L2, x.L0, x.L1 and x.L2 built from the
    # replacement x2
    pred = np.asarray(res.predict(exog=_as_exog_form(x2, repl)))
    t = np.arange(2, nobs)
    reg = [np.ones(nobs - 2), y[t - 1], y[t - 2], x2[t], x2[t - 1], x2[t - 2]]
    assert np.all(np.isnan(pred[:2]))
    assert_allclose(pred[2:], np.column_stack(reg) @ params, rtol=1e-10)

    def forecasts(x_in):
        # y[t] = c + p1 y[t-1] + p2 y[t-2] + b0 x[t] + b1 x[t-1] + b2 x[t-2]
        c, p1, p2, b0, b1, b2 = params
        xx = np.r_[x_in, x_oos]
        path = list(y)
        for t in range(nobs, nobs + 3):
            path.append(
                c + p1 * path[t - 1] + p2 * path[t - 2] + b0 * xx[t]
                + b1 * xx[t - 1] + b2 * xx[t - 2]
            )
        return np.array(path[nobs:])

    oos = _as_exog_form(x_oos, repl)
    fcast = np.asarray(res.forecast(3, exog=oos))
    assert_allclose(fcast, forecasts(x), rtol=1e-10)
    pred = res.predict(start=nobs + 1, end=nobs + 2, exog_oos=oos)
    assert_allclose(pred, forecasts(x)[1:], rtol=1e-10)
    exog2 = _as_exog_form(x2, repl)
    pred = res.predict(start=nobs, end=nobs + 2, exog=exog2, exog_oos=oos)
    assert_allclose(pred, forecasts(x2), rtol=1e-10)


@pytest.mark.parametrize("form", ["1d", "series"])
def test_ardl_predict_1d_exog_wrong_columns(form):
    rng = np.random.default_rng(20261204)
    nobs = 40
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal(nobs))
    res = ARDL(y, 2, _as_exog_form(x, form), 2).fit()
    msg = "must have the same number of columns as the original data, 1"
    with pytest.raises(ValueError, match="exog " + msg):
        res.predict(exog=np.column_stack([x, x]))
    with pytest.raises(ValueError, match="exog_oos " + msg):
        res.forecast(2, exog=np.ones((2, 2)))


@pytest.mark.parametrize("mixed", [False, True])
def test_uecm_nullable_bool_exog(mixed):
    # The differences of a nullable "boolean" column were not numeric (the
    # first one is NA), so UECM raised TypeError; ARDL worked
    rng = np.random.default_rng(20261124)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    xb = rng.standard_normal((nobs, 2)) > 0
    x = xb.astype(float)
    exog = pd.DataFrame(xb, columns=["a", "b"]).astype("boolean")
    if mixed:
        # A float column next to the nullable bool one
        x[:, 1] = rng.standard_normal(nobs)
        exog["b"] = x[:, 1]
    res = UECM(y, 2, exog, 2).fit()

    # D.y on const, y.L1, the levels x.L1 and the differences D.y.L1, D.x.L0
    # and D.x.L1, with the bool values as 0 and 1
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.vstack([np.full((1, 2), np.nan), np.diff(x, axis=0)])
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 1], dy[t - 1]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 1], dx[t - 1, 1]]
    ols = OLS(dy[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.bse, ols.bse, rtol=1e-8)


@pytest.mark.parametrize(
    "kind", ["ndarray-uint8", "ndarray-int8", "frame-uint8", "frame-UInt8"]
)
def test_uecm_integer_exog(kind):
    # np.diff and DataFrame.diff wrap around for unsigned integers (np.diff
    # also when the difference of signed integers does not fit in their type),
    # so a change from 3 to 1 of a uint8 exog was differenced as 254
    rng = np.random.default_rng(20261125)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    if kind == "ndarray-int8":
        # Differences of -200 and 200 do not fit in int8
        x = np.where(rng.standard_normal((nobs, 2)) > 0, 100.0, -100.0)
    else:
        x = rng.integers(0, 5, size=(nobs, 2)).astype(float)
    if kind == "ndarray-uint8":
        exog = x.astype(np.uint8)
    elif kind == "ndarray-int8":
        exog = x.astype(np.int8)
    elif kind == "frame-uint8":
        exog = pd.DataFrame(x.astype(np.uint8), columns=["a", "b"])
    else:
        exog = pd.DataFrame(x, columns=["a", "b"]).astype("UInt8")
    res = UECM(y, 2, exog, 2).fit()

    # D.y on const, y.L1, the levels x.L1 and the differences D.y.L1, D.x.L0
    # and D.x.L1, with the differences of the float64 values
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.vstack([np.full((1, 2), np.nan), np.diff(x, axis=0)])
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 1], dy[t - 1]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 1], dx[t - 1, 1]]
    ols = OLS(dy[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.bse, ols.bse, rtol=1e-8)


@pytest.mark.parametrize(
    "dtype, categories", [(int, None), (int, [50, 10, 20]), (float, None)]
)
def test_uecm_categorical_exog(dtype, categories):
    # A categorical column has no diff, so UECM raised TypeError; ARDL uses
    # the values of the categories (not their codes) as the levels
    rng = np.random.default_rng(20261126)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.column_stack(
        [rng.choice([10.0, 20.0, 50.0], size=nobs), rng.standard_normal(nobs)]
    )
    exog = pd.DataFrame(
        {"a": pd.Categorical(x[:, 0].astype(dtype), categories=categories)}
    )
    exog["b"] = x[:, 1]
    res = UECM(y, 2, exog, 2).fit()

    # D.y on const, y.L1, the levels x.L1 and the differences D.y.L1, D.x.L0
    # and D.x.L1, with the differences of the values of the categories
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.vstack([np.full((1, 2), np.nan), np.diff(x, axis=0)])
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], x[t - 1, 1], dy[t - 1]]
    reg += [dx[t, 0], dx[t - 1, 0], dx[t, 1], dx[t - 1, 1]]
    ols = OLS(dy[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.bse, ols.bse, rtol=1e-8)
    # The UECM is a reparametrization of the ARDL with the same lags
    ardl_res = ARDL(y, 2, exog, 2).fit()
    assert_allclose(res.ssr, ardl_res.ssr, rtol=1e-10)


@pytest.mark.parametrize("kind", ["ndarray", "list", "series", "series-category"])
def test_uecm_1d_exog(kind):
    # A 1-d exog or a Series raised "could not broadcast input array from
    # shape (n-1,) into shape (n-1,1)"; ARDL treats it as the variable 0
    rng = np.random.default_rng(20261127)
    nobs = 150
    y = np.cumsum(rng.standard_normal(nobs))
    if kind == "series-category":
        x = rng.choice([10.0, 20.0, 50.0], size=nobs)
        exog = pd.Series(pd.Categorical(x.astype(int)), name="x")
    else:
        x = rng.standard_normal(nobs)
        exog = {"ndarray": x, "list": list(x), "series": pd.Series(x, name="x")}[
            kind
        ]
    res = UECM(y, 2, exog, 2).fit()

    # D.y on const, y.L1, the level x.L1 and the differences D.y.L1, D.x.L0
    # and D.x.L1
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.r_[np.nan, np.diff(x)]
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1], dy[t - 1], dx[t], dx[t - 1]]
    ols = OLS(dy[t], np.column_stack(reg)).fit()
    assert_allclose(res.params, ols.params, rtol=1e-8)
    assert_allclose(res.bse, ols.bse, rtol=1e-8)
    # The names of the variable 0 in ARDL: x0 for an array, 0 for a Series
    base = "0" if kind.startswith("series") else "x0"
    expected = ["const", "y.L1", f"{base}.L1", "D.y.L1"]
    expected += [f"D.{base}.L0", f"D.{base}.L1"]
    assert res.model.exog_names == expected
    ardl_res = ARDL(y, 2, exog, 2).fit()
    assert ardl_res.model.exog_names[3:] == [f"{base}.L{i}" for i in range(3)]
    # The UECM is a reparametrization of the ARDL with the same lags
    assert_allclose(res.ssr, ardl_res.ssr, rtol=1e-10)


@pytest.mark.parametrize(
    "kind", ["ndarray", "ndarray-frame", "frame", "frame-bool", "series"]
)
def test_uecm_predict_replacement_exog(kind):
    # The replacement exog was ignored: predict(exog=x2) equalled predict()
    rng = np.random.default_rng(20261128)
    nobs = 80
    y = np.cumsum(rng.standard_normal(nobs))
    k = 1 if kind == "series" else 2
    if kind == "frame-bool":
        x = (rng.standard_normal((nobs, k)) > 0).astype(float)
        x2 = (rng.standard_normal((nobs, k)) > 0).astype(float)
    else:
        x = np.cumsum(rng.standard_normal((nobs, k)), axis=0)
        x2 = x + rng.standard_normal((nobs, k))
    if kind == "ndarray":
        exog, exog2 = x, x2
    elif kind == "ndarray-frame":
        # A DataFrame replaces an array by position
        exog, exog2 = x, pd.DataFrame(x2)
    elif kind == "series":
        exog, exog2 = pd.Series(x[:, 0], name="a"), pd.Series(x2[:, 0], name="a")
    else:
        dtype = bool if kind == "frame-bool" else float
        exog = pd.DataFrame(x.astype(dtype), columns=["a", "b"])
        # The columns are matched by name
        exog2 = pd.DataFrame(x2.astype(dtype), columns=["a", "b"])[["b", "a"]]
    res = UECM(y, 2, exog, 2).fit()
    pred = np.asarray(res.predict(exog=exog2))

    # The regressors const, y.L1, x.L1, D.y.L1, D.x.L0 and D.x.L1 with the
    # replacement x2, the bool values as 0 and 1
    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx2 = np.vstack([np.full((1, k), np.nan), np.diff(x2, axis=0)])
    reg = [np.ones(nobs - 2), y[t - 1]] + [x2[t - 1, i] for i in range(k)]
    reg += [dy[t - 1]]
    for i in range(k):
        reg += [dx2[t, i], dx2[t - 1, i]]
    expected = np.column_stack(reg) @ np.asarray(res.params)
    assert np.all(np.isnan(pred[:2]))
    assert_allclose(pred[2:], expected, rtol=1e-10)
    # The UECM is a reparametrization of the ARDL with the same lags, so its
    # predicted differences are those of the ARDL's predicted levels
    ardl_pred = ARDL(y, 2, x, 2).fit().predict(exog=x2)
    assert_allclose(pred[2:], ardl_pred[2:] - y[1:-1], rtol=1e-8)
    assert_allclose(res.get_prediction(exog=exog2).predicted_mean, pred, rtol=1e-12)


def test_uecm_predict_replacement_fixed():
    # The replacement fixed was ignored: predict(fixed=z2) equalled predict()
    rng = np.random.default_rng(20261129)
    nobs = 80
    y = np.cumsum(rng.standard_normal(nobs))
    x = np.cumsum(rng.standard_normal((nobs, 1)), axis=0)
    z = rng.standard_normal((nobs, 2))
    z2 = rng.standard_normal((nobs, 2))
    res = UECM(y, 2, x, 2, fixed=z).fit()
    pred = res.predict(fixed=z2.tolist())

    t = np.arange(2, nobs)
    dy = np.r_[np.nan, np.diff(y)]
    dx = np.r_[np.nan, np.diff(x[:, 0])]
    reg = [np.ones(nobs - 2), y[t - 1], x[t - 1, 0], dy[t - 1], dx[t], dx[t - 1]]
    reg += [z2[t, 0], z2[t, 1]]
    expected = np.column_stack(reg) @ res.params
    assert_allclose(pred[2:], expected, rtol=1e-10)
    # With both replaced, the levels and differences use the replacement exog
    x2 = x + 1.0
    both = res.predict(exog=x2, fixed=z2)
    assert_allclose(both[2:], expected + res.params[2], rtol=1e-10)


def test_uecm_predict_replacement_errors():
    rng = np.random.default_rng(20261130)
    nobs = 50
    y = np.cumsum(rng.standard_normal(nobs))
    x = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["a", "b"])
    res = UECM(y, 2, x, 2, fixed=rng.standard_normal((nobs, 1))).fit()
    with pytest.raises(TypeError, match="exog must be a DataFrame"):
        res.predict(exog=np.asarray(x))
    with pytest.raises(ValueError, match="exog must have the same columns"):
        res.predict(exog=x.rename(columns={"b": "c"}))
    with pytest.raises(ValueError, match="exog must have the same shape"):
        res.predict(exog=x.iloc[:-2])
    with pytest.raises(ValueError, match="fixed must have the same shape"):
        res.predict(fixed=np.ones((nobs, 2)))
    res = UECM(y, 2, np.asarray(x), 2).fit()
    with pytest.raises(ValueError, match="exog must have the same shape"):
        res.predict(exog=np.asarray(x)[:, :1])


@pytest.mark.parametrize("glob", [False, True])
@pytest.mark.parametrize("use_pandas", [False, True])
def test_ardl_select_order_maxorder_none_no_warning(use_pandas, glob):
    # A variable that maxorder excludes with None is excluded from the
    # selected model without a warning that it is missing from the order
    rng = np.random.default_rng(20261124)
    nobs = 200
    x = rng.standard_normal((nobs, 2))
    e = rng.standard_normal(nobs)
    y = np.zeros(nobs)
    for t in range(1, nobs):
        y[t] = 0.5 * y[t - 1] + x[t, 0] + 0.8 * x[t - 1, 0] + x[t, 1] + e[t]
    if use_pandas:
        exog = pd.DataFrame(x, columns=["a", "b"])
        maxorder = {"a": 1, "b": None}
        key = "a"
    else:
        exog = x
        maxorder = {0: 1, 1: None}
        key = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpecificationWarning)
        sel = ardl_select_order(y, 1, exog, maxorder, glob=glob)
    assert sel.model.ar_lags == [1]
    assert sel.model.dl_lags == {key: [0, 1]}

    # y on const, y.L1, x0.L0 and x0.L1, without the excluded x1
    t = np.arange(1, nobs)
    reg = np.column_stack([np.ones(nobs - 1), y[t - 1], x[t, 0], x[t - 1, 0]])
    ols = OLS(y[t], reg).fit()
    assert_allclose(sel.model.fit().params, ols.params, rtol=1e-10)


def test_predict_dynamic_key_covers_several_dates():
    # A dynamic date key that covers several dates (all of April 2001 in daily
    # data) gives a slice of locations, which raised a TypeError. Dynamic
    # prediction begins at the first date of the key.
    rng = np.random.default_rng(20261121)
    index = pd.date_range("2000-01-01", periods=500, freq="D")
    y = pd.Series(rng.standard_normal(500).cumsum(), index=index)
    x = pd.Series(rng.standard_normal(500), index=index, name="x")
    res = ARDL(y, 1, x, 1).fit()
    loc = 366 + 31 + 28 + 31
    assert index[loc] == pd.Timestamp("2001-04-01")
    pred = res.predict(start=10, dynamic="2001-04")
    expected = res.predict(start=10, dynamic=loc - 10)
    assert_allclose(pred, expected, rtol=1e-12)
    assert_index_equal(pred.index, index[10:])


def test_ardl_test_serial_correlation():
    # Inherited from AutoRegResults, whose statistics were all NaN since
    # 0.14.0
    rng = np.random.default_rng(20261123)
    nobs = 200
    x = rng.standard_normal(nobs)
    e = rng.standard_normal(nobs)
    y = np.zeros(nobs)
    for t in range(1, nobs):
        y[t] = 0.5 * y[t - 1] + x[t] - 0.5 * x[t - 1] + e[t]
    res = ARDL(y, 1, x[:, None], 1, trend="c").fit()
    sc = res.test_serial_correlation(lags=8, model_df="regressors")

    reg = np.column_stack([np.ones(nobs - 1), y[:-1], x[1:], x[:-1]])
    resid = OLS(y[1:], reg).fit().resid
    # "ar" counts the one lag of y
    sc_ar = res.test_serial_correlation(lags=8, model_df="ar")
    assert_allclose(sc_ar["DF"], np.maximum(np.arange(1, 9) - 1, 0))
    lb = acorr_ljungbox(resid, lags=8, model_df=4)
    df = np.maximum(np.arange(1, 9) - 4, 0)
    assert_allclose(sc["Ljung-Box"], lb["lb_stat"], rtol=1e-8)
    assert_allclose(sc["DF"], df)
    pvalue = stats.chi2.sf(lb["lb_stat"].to_numpy()[4:], df[4:])
    assert_allclose(sc["LB P-value"].iloc[4:], pvalue, rtol=1e-8)
    assert np.all(np.isnan(sc["LB P-value"].iloc[:4]))
