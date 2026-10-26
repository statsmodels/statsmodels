from statsmodels.compat.python import PYTHON_IMPL_WASM

from typing import NamedTuple

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest

from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
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
    # GH-10546: with one regressor, the asymptotic p-values were those for two
    rng = np.random.default_rng(0)
    nobs = 500
    x = np.cumsum(rng.standard_normal(nobs))
    y = np.zeros(nobs)
    for t in range(1, nobs):
        y[t] = y[t - 1] - 0.05 * (y[t - 1] - 0.5 * x[t - 1]) + rng.standard_normal()
    res = UECM(pd.Series(y), 1, pd.DataFrame({"x": x}), 1, trend="c").fit()
    bounds_result = res.bounds_test(case=3)
    bounds_result_sim = res.bounds_test(case=3, asymptotic=False, nsim=20_000, rng=0)
    assert_allclose(bounds_result.pvalue, bounds_result_sim.pvalue, atol=0.003)


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
    assert_allclose(bounds_result.statistic, expected.statistic, rtol=1e-12)
    assert_allclose(bounds_result.pvalue, expected.pvalue, rtol=1e-12)
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
    with pytest.raises(np.linalg.LinAlgError):
        ardl_select_order(y, 2, data.x, 2, "c")
