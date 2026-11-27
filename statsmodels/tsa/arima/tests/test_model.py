"""
Tests for ARIMA model.

Tests are primarily limited to checking that the model is constructed correctly
and that it is calling the appropriate parameter estimators correctly. Tests of
correctness of parameter estimation routines are left to the individual
estimators' test functions.

Author: Chad Fulton
License: BSD-3
"""

from statsmodels.compat.platform import PLATFORM_WIN32

import io

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.datasets import macrodata
from statsmodels.iolib.summary import Summary
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import innovations, innovations_mle
from statsmodels.tsa.arima.estimators.statespace import statespace
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.model import ARIMA

dta = macrodata.load_pandas().data
dta.index = pd.date_range(start="1959-01-01", end="2009-07-01", freq="QS")


def test_default_trend():
    # Test that we are setting the trend default correctly
    endog = dta["infl"].iloc[:50]

    # Defaults when only endog is specified
    mod = ARIMA(endog)
    # with no integration, default trend a constant
    assert_equal(mod._spec_arima.trend_order, 0)
    assert_allclose(mod.exog, np.ones((mod.nobs, 1)))

    # Defaults with integrated model
    mod = ARIMA(endog, order=(0, 1, 0))
    # with no integration, default trend is none
    assert_equal(mod._spec_arima.trend_order, None)
    assert_equal(mod.exog, None)


def test_trend_array():
    # GH 9145, a polynomial trend given as an ndarray raised ValueError
    endog = dta["infl"].iloc[:50]

    mod = ARIMA(endog, trend=np.array([0, 1, 1]))
    desired = ARIMA(endog, trend=[0, 1, 1])
    assert_equal(mod._spec_arima.trend_terms, desired._spec_arima.trend_terms)
    assert_allclose(mod.exog, desired.exog)


@pytest.mark.parametrize("trend", [np.array(["c", "t"]), [[1, 0], [0, 1]]])
def test_trend_array_invalid(trend):
    endog = dta["infl"].iloc[:50]
    with pytest.raises(ValueError, match="Valid trend inputs are"):
        ARIMA(endog, trend=trend)


def test_invalid():
    # Tests that invalid options raise errors
    # (note that this is only invalid options specific to `ARIMA`, and not
    # invalid options that would raise errors in SARIMAXSpecification).
    endog = dta["infl"].iloc[:50]
    mod = ARIMA(endog, order=(1, 0, 0))

    # Need valid method
    with pytest.raises(ValueError):
        mod.fit(method="not_a_method")

    # Can only use certain methods with fixed parameters
    # (e.g., 'statespace' and 'hannan-rissanen')
    with mod.fix_params({"ar.L1": 0.5}):
        with pytest.raises(ValueError):
            mod.fit(method="yule_walker")

    # Cannot override model-level values in fit
    with pytest.raises(ValueError):
        mod.fit(method="statespace", method_kwargs={"enforce_stationarity": False})

    # start_params only valid for MLE methods
    with pytest.raises(ValueError):
        mod.fit(method="yule_walker", start_params=[0.5, 1.0])

    # has_exog and gls=False with non-statespace method
    mod2 = ARIMA(endog, order=(1, 0, 0), trend="c")
    with pytest.raises(ValueError):
        mod2.fit(method="yule_walker", gls=False)

    # non-stationary parameters
    mod3 = ARIMA(np.arange(100) * 1.0, order=(1, 0, 0), trend="n")
    with pytest.raises(ValueError):
        mod3.fit(method="hannan_rissanen")

    # non-invertible parameters
    mod3 = ARIMA(np.arange(20) * 1.0, order=(0, 0, 1), trend="n")
    with pytest.raises(ValueError):
        mod3.fit(method="hannan_rissanen")


def test_yule_walker():
    # Test for basic use of Yule-Walker estimation
    endog = dta["infl"].iloc[:50]

    # AR(2), no trend (since trend would imply GLS estimation)
    desired_p = yule_walker(endog, ar_order=2, demean=False).parameters
    mod = ARIMA(endog, order=(2, 0, 0), trend="n")
    res = mod.fit(method="yule_walker")
    assert_allclose(res.params, desired_p.params)


def test_burg():
    # Test for basic use of Yule-Walker estimation
    endog = dta["infl"].iloc[:50]

    # AR(2), no trend (since trend would imply GLS estimation)
    desired_p = burg(endog, ar_order=2, demean=False).parameters
    mod = ARIMA(endog, order=(2, 0, 0), trend="n")
    res = mod.fit(method="burg")
    assert_allclose(res.params, desired_p.params)


def test_hannan_rissanen():
    # Test for basic use of Hannan-Rissanen estimation
    endog = dta["infl"].diff().iloc[1:101]

    # ARMA(1, 1), no trend (since trend would imply GLS estimation)
    desired_p = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=False).parameters
    mod = ARIMA(endog, order=(1, 0, 1), trend="n")
    res = mod.fit(method="hannan_rissanen")
    assert_allclose(res.params, desired_p.params)


def test_innovations():
    # Test for basic use of Yule-Walker estimation
    endog = dta["infl"].iloc[:50]

    # MA(2), no trend (since trend would imply GLS estimation)
    desired_p = innovations(endog, ma_order=2, demean=False).parameters
    mod = ARIMA(endog, order=(0, 0, 2), trend="n")
    res = mod.fit(method="innovations")
    assert_allclose(res.params, desired_p[-1].params)


def test_innovations_arma():
    # Test for basic use of innovations estimation with AR terms
    endog = dta["infl"].diff().iloc[1:101]

    # ARMA(1, 1), no trend (since trend would imply GLS estimation)
    desired_p = innovations(endog, ar_order=1, ma_order=1, demean=False).parameters
    mod = ARIMA(endog, order=(1, 0, 1), trend="n")
    res = mod.fit(method="innovations")
    assert_allclose(res.params, desired_p[-1].params)

    # The recursion level is passed through method_kwargs
    desired_p = innovations(
        endog, ar_order=1, ma_order=1, demean=False, m=10
    ).parameters
    res = mod.fit(method="innovations", method_kwargs={"m": 10})
    assert_allclose(res.params, desired_p[-1].params)

    # Estimates that imply a non-stationary, non-invertible process are only
    # accepted without enforce_stationarity and enforce_invertibility
    endog = dta["infl"].iloc[:50]
    mod = ARIMA(endog, order=(1, 0, 1), trend="n")
    with pytest.raises(ValueError, match="Non-stationary autoregressive"):
        mod.fit(method="innovations")
    desired_p = innovations(endog, ar_order=1, ma_order=1, demean=False).parameters
    mod = ARIMA(
        endog,
        order=(1, 0, 1),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(method="innovations")
    assert_allclose(res.params, desired_p[-1].params)


def test_innovations_mle():
    # Test for basic use of Yule-Walker estimation
    endog = dta["infl"].iloc[:100]

    # ARMA(1, 1), no trend (since trend would imply GLS estimation)
    desired_p = innovations_mle(endog, order=(1, 0, 1), demean=False).parameters
    mod = ARIMA(endog, order=(1, 0, 1), trend="n")
    res = mod.fit(method="innovations_mle")
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-5)

    # SARMA(1, 0)x(1, 0)4, no trend (since trend would imply GLS estimation)
    desired_p = innovations_mle(
        endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), demean=False
    ).parameters
    mod = ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend="n")
    res = mod.fit(method="innovations_mle")
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-5)


@pytest.mark.parametrize("trend", ["n", "c"])
def test_innovations_mle_start_params_not_implemented(trend):
    # ARIMA.fit does not pass starting parameters through for innovations_mle.
    # The error should say so, not raise a TypeError from the estimator.
    endog = dta["infl"].iloc[:100]
    mod = ARIMA(endog, order=(1, 0, 0), trend=trend)
    with pytest.raises(NotImplementedError, match="innovations_mle"):
        mod.fit(method="innovations_mle", start_params=[0.5, 1.0])

    if trend == "n":
        # The statespace method still accepts starting parameters.
        res = mod.fit(method="statespace", start_params=[0.5, 1.0])
        assert res.params.shape == (2,)


def test_statespace():
    # Test for basic use of Yule-Walker estimation
    endog = dta["infl"].iloc[:100]

    # ARMA(1, 1), no trend
    desired_p = statespace(endog, order=(1, 0, 1), include_constant=False).parameters
    mod = ARIMA(endog, order=(1, 0, 1), trend="n")
    res = mod.fit(method="statespace")
    # Note: tol changes required due to precision issues on Windows
    rtol = 1e-7 if not PLATFORM_WIN32 else 1e-3
    assert_allclose(res.params, desired_p.params, rtol=rtol, atol=1e-4)

    # ARMA(1, 2), with trend
    desired_p = statespace(endog, order=(1, 0, 2), include_constant=True).parameters
    mod = ARIMA(endog, order=(1, 0, 2), trend="c")
    res = mod.fit(method="statespace")
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-4)

    # SARMA(1, 0)x(1, 0)4, no trend
    desired_p, _ = statespace(
        endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), include_constant=False
    )
    mod = ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend="n")
    res = mod.fit(method="statespace")
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-4)


def test_low_memory():
    # Basic test that the low_memory option is working
    endog = dta["infl"].iloc[:50]

    mod = ARIMA(endog, order=(1, 0, 0), concentrate_scale=True)
    res1 = mod.fit()
    res2 = mod.fit(low_memory=True)

    # Check that the models produce the same results
    assert_allclose(res2.params, res1.params)
    assert_allclose(res2.llf, res1.llf)

    # Check that the model's basic memory conservation option was not changed
    assert_equal(mod.ssm.memory_conserve, 0)

    # Check that low memory was actually used (just check a couple)
    assert res2.llf_obs is None
    assert res2.predicted_state is None
    assert res2.filtered_state is None
    assert res2.smoothed_state is None


def check_cloned(mod, endog, exog=None):
    mod_c = mod.clone(endog, exog=exog)

    assert_allclose(mod.nobs, mod_c.nobs)
    assert mod._index.equals(mod_c._index)
    assert_equal(mod.k_params, mod_c.k_params)
    assert_allclose(mod.start_params, mod_c.start_params)
    p = mod.start_params
    assert_allclose(mod.loglike(p), mod_c.loglike(p))
    assert_allclose(mod.concentrate_scale, mod_c.concentrate_scale)


def test_clone():
    endog = dta["infl"].iloc[:50]
    exog = np.arange(endog.shape[0])

    # Basic model
    check_cloned(ARIMA(endog), endog)
    check_cloned(ARIMA(endog.values), endog.values)
    # With trends
    check_cloned(ARIMA(endog, trend="c"), endog)
    check_cloned(ARIMA(endog, trend="t"), endog)
    check_cloned(ARIMA(endog, trend="ct"), endog)
    # With exog
    check_cloned(ARIMA(endog, exog=exog), endog, exog=exog)
    check_cloned(ARIMA(endog, exog=exog, trend="c"), endog, exog=exog)
    # Concentrated scale
    check_cloned(
        ARIMA(endog, exog=exog, trend="c", concentrate_scale=True), endog, exog=exog
    )

    # Higher order (use a different dataset to avoid warnings about
    # non-invertible start params)
    endog = dta["realgdp"].iloc[:100]
    exog = np.arange(endog.shape[0])
    check_cloned(
        ARIMA(
            endog,
            order=(2, 1, 1),
            seasonal_order=(1, 1, 2, 4),
            exog=exog,
            trend=[0, 0, 1],
            concentrate_scale=True,
        ),
        endog,
        exog=exog,
    )


def test_constant_integrated_model_error():
    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 1, 0), trend="c")

    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 0, 0), seasonal_order=(1, 1, 0, 6), trend="c")

    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 2, 0), trend="t")

    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 1, 0), seasonal_order=(1, 1, 0, 6), trend="t")


def test_forecast():
    # Numpy
    endog = dta["infl"].iloc[:100].values

    mod = ARIMA(endog[:50], order=(1, 1, 0), trend="t")
    res = mod.filter([0.2, 0.3, 1.0])

    endog2 = endog.copy()
    endog2[50:] = np.nan
    mod2 = mod.clone(endog2)
    res2 = mod2.filter(res.params)

    assert_allclose(res.forecast(50), res2.fittedvalues[-50:])


def test_forecast_with_exog():
    # Numpy
    endog = dta["infl"].iloc[:100].values
    exog = np.arange(len(endog)) ** 2

    mod = ARIMA(endog[:50], order=(1, 1, 0), exog=exog[:50], trend="t")
    res = mod.filter([0.2, 0.05, 0.3, 1.0])

    endog2 = endog.copy()
    endog2[50:] = np.nan
    mod2 = mod.clone(endog2, exog=exog)
    print(mod.param_names)
    print(mod2.param_names)
    res2 = mod2.filter(res.params)

    assert_allclose(res.forecast(50, exog=exog[50:]), res2.fittedvalues[-50:])


def test_append():
    endog = dta["infl"].iloc[:100].values
    mod = ARIMA(endog[:50], trend="c")
    res = mod.fit()
    res_e = res.append(endog[50:])
    mod2 = ARIMA(endog)
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_append_with_exog():
    # Numpy
    endog = dta["infl"].iloc[:100].values
    exog = np.arange(len(endog))
    mod = ARIMA(endog[:50], exog=exog[:50], trend="c")
    res = mod.fit()
    res_e = res.append(endog[50:], exog=exog[50:])
    mod2 = ARIMA(endog, exog=exog, trend="c")
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_append_with_exog_and_trend():
    # Numpy
    endog = dta["infl"].iloc[:100].values
    exog = np.arange(len(endog)) ** 2
    mod = ARIMA(endog[:50], exog=exog[:50], trend="ct")
    res = mod.fit()
    res_e = res.append(endog[50:], exog=exog[50:])
    mod2 = ARIMA(endog, exog=exog, trend="ct")
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_append_with_exog_pandas():
    # Pandas
    endog = dta["infl"].iloc[:100]
    exog = pd.Series(np.arange(len(endog)), index=endog.index)
    mod = ARIMA(endog.iloc[:50], exog=exog.iloc[:50], trend="c")
    res = mod.fit()
    res_e = res.append(endog.iloc[50:], exog=exog.iloc[50:])
    mod2 = ARIMA(endog, exog=exog, trend="c")
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_extend_with_exog_constant_over_window():
    # GH 8991: extending with `exog` that is constant over the extension window
    # (but not over the full sample) must not raise, since `extend` reuses the
    # already-validated specification of the original model.
    endog = dta["infl"].iloc[:100].values
    exog = np.r_[np.arange(50.0), np.ones(50) * 3.0]
    mod = ARIMA(endog[:50], exog=exog[:50], trend="c")
    res = mod.fit()

    # Raised ValueError before GH 8991 was fixed
    res_e = res.extend(endog[50:], exog=exog[50:])

    mod2 = ARIMA(endog, exog=exog, trend="c")
    res2 = mod2.filter(res.params)

    assert_allclose(res_e.llf_obs, res2.llf_obs[50:])


@pytest.mark.parametrize(
    "order, trend", [((1, 0, 0), "ct"), ((1, 1, 0), "t"), ((1, 0, 0), [0, 1, 0, 1])]
)
@pytest.mark.parametrize("trend_offset", [1, 0, 50])
@pytest.mark.parametrize("use_exog", [False, True])
def test_extend_append_apply_with_time_trend(order, trend, trend_offset, use_exog):
    # ARIMA keeps its trend_offset, so the time trend of forecasts and of
    # extended results continues from the end of the sample. Before,
    # trend_offset was not stored on the model, so the trend restarted at 1
    # in every out-of-sample period.
    endog = dta["infl"].iloc[:100].values + 0.1 * np.arange(100)
    exog = np.cos(np.arange(100.0)) if use_exog else None

    def model(start, end, offset=trend_offset):
        return ARIMA(
            endog[start:end],
            exog=None if exog is None else exog[start:end],
            order=order,
            trend=trend,
            trend_offset=offset,
        )

    params = model(0, 100).start_params
    res_full = model(0, 100).filter(params)
    res = model(0, 50).filter(params)
    assert_equal(res.model.trend_offset, trend_offset)

    res_ext = res.extend(
        endog[50:70], exog=None if exog is None else exog[50:70]
    ).extend(endog[70:], exog=None if exog is None else exog[70:])
    assert_equal(res_ext.model.trend_offset, trend_offset + 70)
    assert_allclose(res_ext.llf_obs, res_full.llf_obs[70:])

    res_app = res.append(endog[50:], exog=None if exog is None else exog[50:])
    assert_allclose(res_app.llf, res_full.llf)

    fcast_exog = np.ones(5) if use_exog else None
    desired = res_full.get_forecast(5, exog=fcast_exog)
    for actual in [
        res_ext.get_forecast(5, exog=fcast_exog),
        res_app.get_forecast(5, exog=fcast_exog),
    ]:
        assert_allclose(actual.predicted_mean, desired.predicted_mean)
        assert_allclose(actual.conf_int(), desired.conf_int())

    # apply uses the same trend_offset on the new data
    res_apply = res.apply(endog[50:], exog=None if exog is None else exog[50:])
    res_new = model(50, 100).filter(params)
    assert_allclose(res_apply.llf, res_new.llf)
    assert_allclose(
        res_apply.forecast(5, exog=fcast_exog), res_new.forecast(5, exog=fcast_exog)
    )


def test_forecast_with_trend_offset():
    # With trend_offset=0 the time trend is 0, ..., nobs - 1 in the sample, so
    # the one-step forecast of a regression on a linear trend with AR(1)
    # errors is c + b * nobs + phi * (y[-1] - c - b * (nobs - 1))
    endog = 10 + 0.5 * np.arange(100.0) + np.sin(np.arange(100.0))
    c, b, phi = 10.0, 0.5, 0.5
    mod = ARIMA(endog, order=(1, 0, 0), trend="ct", trend_offset=0)
    res = mod.filter([c, b, phi, 1.0])

    desired = c + b * 100 + phi * (endog[-1] - c - b * 99)
    assert_allclose(res.forecast(1), desired)


def test_cov_type_none():
    endog = dta["infl"].iloc[:100].values
    mod = ARIMA(endog[:50], trend="c")
    res = mod.fit(cov_type="none")
    assert_allclose(res.cov_params(), np.nan)


def test_nonstationary_gls_error():
    # GH-6540
    endog = pd.read_csv(
        io.StringIO("""\
data\n
9.112\n9.102\n9.103\n9.099\n9.094\n9.090\n9.108\n9.088\n9.091\n9.083\n9.095\n
9.090\n9.098\n9.093\n9.087\n9.088\n9.083\n9.095\n9.077\n9.082\n9.082\n9.081\n
9.081\n9.079\n9.088\n9.096\n9.081\n9.098\n9.081\n9.094\n9.091\n9.095\n9.097\n
9.108\n9.104\n9.098\n9.085\n9.093\n9.094\n9.092\n9.093\n9.106\n9.097\n9.108\n
9.100\n9.106\n9.114\n9.111\n9.097\n9.099\n9.108\n9.108\n9.110\n9.101\n9.111\n
9.114\n9.111\n9.126\n9.124\n9.112\n9.120\n9.142\n9.136\n9.131\n9.106\n9.112\n
9.119\n9.125\n9.123\n9.138\n9.133\n9.133\n9.137\n9.133\n9.138\n9.136\n9.128\n
9.127\n9.143\n9.128\n9.135\n9.133\n9.131\n9.136\n9.120\n9.127\n9.130\n9.116\n
9.132\n9.128\n9.119\n9.119\n9.110\n9.132\n9.130\n9.124\n9.130\n9.135\n9.135\n
9.119\n9.119\n9.136\n9.126\n9.122\n9.119\n9.123\n9.121\n9.130\n9.121\n9.119\n
9.106\n9.118\n9.124\n9.121\n9.127\n9.113\n9.118\n9.103\n9.112\n9.110\n9.111\n
9.108\n9.113\n9.117\n9.111\n9.100\n9.106\n9.109\n9.113\n9.110\n9.101\n9.113\n
9.111\n9.101\n9.097\n9.102\n9.100\n9.110\n9.110\n9.096\n9.095\n9.090\n9.104\n
9.097\n9.099\n9.095\n9.096\n9.085\n9.097\n9.098\n9.090\n9.080\n9.093\n9.085\n
9.075\n9.067\n9.072\n9.062\n9.068\n9.053\n9.051\n9.049\n9.052\n9.059\n9.070\n
9.058\n9.074\n9.063\n9.057\n9.062\n9.058\n9.049\n9.047\n9.062\n9.052\n9.052\n
9.044\n9.060\n9.062\n9.055\n9.058\n9.054\n9.044\n9.047\n9.050\n9.048\n9.041\n
9.055\n9.051\n9.028\n9.030\n9.029\n9.027\n9.016\n9.023\n9.031\n9.042\n9.035\n
"""),
        index_col=None,
    )
    mod = ARIMA(
        endog,
        order=(18, 0, 39),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with pytest.raises(ValueError, match="Roots of the autoregressive"):
        mod.fit(method="hannan_rissanen", low_memory=True, cov_type="none")


@pytest.mark.parametrize(
    "ar_order, ma_order, fixed_params",
    [
        (1, 1, {}),
        (1, 1, {"ar.L1": 0}),
        (2, 3, {"ar.L2": -1, "ma.L1": 2}),
        ([0, 1], 0, {"ar.L2": 0}),
        ([1, 5], [0, 0, 1], {"ar.L5": -10, "ma.L3": 5}),
    ],
)
def test_hannan_rissanen_with_fixed_params(ar_order, ma_order, fixed_params):
    # Test for basic uses of Hannan-Rissanen estimation with fixed parameters
    endog = dta["infl"].diff().iloc[1:101]

    desired_p = hannan_rissanen(
        endog,
        ar_order=ar_order,
        ma_order=ma_order,
        demean=False,
        fixed_params=fixed_params,
    ).parameters
    # no constant or trend (since constant or trend would imply GLS estimation)
    mod = ARIMA(
        endog,
        order=(ar_order, 0, ma_order),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with mod.fix_params(fixed_params):
        res = mod.fit(method="hannan_rissanen")

    assert_allclose(res.params, desired_p.params)


@pytest.mark.parametrize(
    "random_state_type", [7, np.random.RandomState, np.random.default_rng]
)
def test_reproducible_simulation(random_state_type):
    rg = np.random.RandomState(12345)
    x = rg.randn(100)
    res = ARIMA(x, order=(1, 0, 0)).fit()

    def get_random_state(val):
        if isinstance(random_state_type, int):
            return 7
        return random_state_type(7)

    rng = get_random_state(random_state_type)
    sim1 = res.simulate(1, rng=rng)
    rng = get_random_state(random_state_type)
    sim2 = res.simulate(1, rng=rng)
    assert_allclose(sim1, sim2)


def test_alternative_estimators_seasonal_differencing():
    # Seasonal differencing only (P=0, D=1, Q=0) should be accepted by
    # non-seasonal estimators; seasonal AR or MA terms should be rejected.
    rg = np.random.RandomState(12345)
    endog = rg.standard_normal(48)

    order = (1, 0, 1)
    seasonal_order = (0, 1, 0, 12)

    # hannan_rissanen should accept seasonal-differencing-only models
    mod_hr = ARIMA(endog, order=order, seasonal_order=seasonal_order)
    try:
        with pytest.warns(UserWarning, match="Provided"):
            mod_hr.fit(method="hannan_rissanen")
    except Exception as exc:
        pytest.fail(
            f"hannan_rissanen failed on seasonal-differencing-only model: {exc}"
        )

    # yule_walker: AR-only model with seasonal differencing
    mod_yw = ARIMA(endog, order=(2, 0, 0), seasonal_order=seasonal_order)
    try:
        with pytest.warns(UserWarning, match="Provided"):
            mod_yw.fit(method="yule_walker")
    except Exception as exc:
        pytest.fail(f"yule_walker failed on seasonal-differencing-only model: {exc}")

    # Seasonal AR term (P=1) should still be rejected by hannan_rissanen
    with pytest.raises(ValueError, match="seasonal"):
        ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit(
            method="hannan_rissanen"
        )


def test_summary_after_remove_data():
    # summary() must still work after remove_data() has been called
    endog = dta["infl"].iloc[:50]
    mod = ARIMA(endog, order=(1, 0, 0), concentrate_scale=True)
    res = mod.fit()

    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


def _contract_arima_data(n=120):
    rng = np.random.default_rng(20261125)
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.2 * y[t - 2] + e[t]
    return 10 + y, rng.standard_normal((n, 2))


def _yule_walker_by_hand(x, order):
    # Yule-Walker equations with the (biased) autocovariances of x, which is
    # not demeaned
    n = x.shape[0]
    r = np.array([x[k:] @ x[: n - k] / n for k in range(order + 1)])
    toeplitz = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    phi = np.linalg.solve(toeplitz, r[1:])
    return np.r_[phi, r[0] - phi @ r[1:]]


def test_arima_integrated_non_statespace_method():
    # Methods that cannot handle integration are applied to the differenced
    # endog
    y, _ = _contract_arima_data()
    y = np.cumsum(y)
    mod = ARIMA(y, order=(2, 1, 0))
    with pytest.warns(UserWarning, match="has been differenced"):
        res = mod.fit(method="yule_walker")
    assert_allclose(res.params, _yule_walker_by_hand(np.diff(y), 2), rtol=1e-10)


@pytest.mark.parametrize(
    "order, seasonal_order",
    [((1, 1, 0), (0, 0, 0, 0)), ((1, 0, 0), (1, 1, 0, 4)), ((1, 1, 0), (0, 1, 0, 4))],
)
def test_arima_integrated_innovations_mle(order, seasonal_order):
    # innovations_mle, the only one of these methods that is given the whole
    # order, receives the differenced endog and the orders without the
    # differencing, so the result equals the estimator applied to the
    # differenced series with d = D = 0
    from statsmodels.tsa.statespace.tools import diff

    y, _ = _contract_arima_data()
    y = np.cumsum(y)
    mod = ARIMA(y, order=order, seasonal_order=seasonal_order)
    with pytest.warns(UserWarning, match="has been differenced"):
        res = mod.fit(method="innovations_mle")
    p, d, q = order
    sp, sd, sq, s = seasonal_order
    dy = diff(y, k_diff=d, k_seasonal_diff=sd, seasonal_periods=s)
    expected, _ = innovations_mle(
        dy, order=(p, 0, q), seasonal_order=(sp, 0, sq, s), demean=False
    )
    # Both are the results of an optimizer, on objectives that agree up to
    # rounding. Where it stops depends on the platform (up to 4.9e-6 relative
    # on macOS), while differencing the wrong number of times changes the
    # estimates completely.
    assert_allclose(res.params, expected.params, rtol=1e-4)


def test_arima_return_params_non_statespace_method():
    y, _ = _contract_arima_data()
    params = ARIMA(y, order=(2, 0, 0), trend="n").fit(
        method="yule_walker", return_params=True
    )
    assert isinstance(params, np.ndarray)
    assert_allclose(params, _yule_walker_by_hand(y, 2), rtol=1e-10)


def test_arima_return_params_statespace():
    y, x = _contract_arima_data()
    mod = ARIMA(y, x, order=(1, 0, 0))
    params = mod.fit(return_params=True)
    assert isinstance(params, np.ndarray)
    assert_allclose(params, mod.fit().params, rtol=1e-12)


def test_arima_low_memory_non_statespace_method():
    # The results are filtered with memory conservation, which is reset
    # afterwards
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y, _ = _contract_arima_data()
    mod = ARIMA(y, order=(2, 0, 0), trend="n")
    conserve_memory = mod.ssm.conserve_memory
    res = mod.fit(method="yule_walker", low_memory=True)
    assert mod.ssm.conserve_memory == conserve_memory
    assert res.filter_results.conserve_memory != conserve_memory
    assert res.smoothed_state is None
    params = _yule_walker_by_hand(y, 2)
    assert_allclose(res.params, params, rtol=1e-10)
    # The loglikelihood is that of the state space model at these parameters
    expected = SARIMAX(y, order=(2, 0, 0)).loglike(params)
    assert_allclose(res.llf, expected, rtol=1e-10)
    full = mod.fit(method="yule_walker")
    assert full.smoothed_state is not None
    assert_allclose(res.forecast(3), full.forecast(3), rtol=1e-10)


def test_arima_gls_kwargs():
    # gls_kwargs are passed to the GLS estimator
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.arima.estimators.gls import gls

    y, x = _contract_arima_data()
    mod = ARIMA(y, x, order=(1, 0, 0), trend="n")
    with pytest.warns(ConvergenceWarning, match="failed to converge in 1"):
        res = mod.fit(method="yule_walker", gls=True, gls_kwargs={"max_iter": 1})
    with pytest.warns(ConvergenceWarning, match="failed to converge in 1"):
        expected = gls(
            y,
            x,
            order=(1, 0, 0),
            include_constant=False,
            arma_estimator="yule_walker",
            max_iter=1,
        )
    assert res.fit_details.iterations == 1
    assert_allclose(res.params, expected.parameters.params, rtol=1e-10)
    converged = mod.fit(method="yule_walker", gls=True)
    assert converged.fit_details.iterations > 1


def test_arima_impulse_responses_kwargs():
    # anchor and exog are passed on through kwargs. The impulse responses of
    # an AR(1) are phi ** h, whatever the anchor
    y, x = _contract_arima_data()
    res = ARIMA(y, x, order=(1, 0, 0)).fit()
    phi = res.params[res.model.param_names.index("ar.L1")]
    expected = phi ** np.arange(4)
    for anchor in ("start", "end", 10):
        irf = res.impulse_responses(3, anchor=anchor, exog=x[-3:])
        assert_allclose(irf, expected, rtol=1e-10)


def _contract_arima_endog_forms(y):
    from statsmodels.tsa.tests._contract import series_forms

    forms = dict(series_forms(y))
    forms["period"] = pd.Series(
        y, index=pd.period_range("2000-01", periods=y.shape[0], freq="M")
    )
    forms["range"] = pd.Series(y, index=pd.RangeIndex(y.shape[0]))
    return forms


@pytest.mark.parametrize(
    "form", ["list", "tuple", "series", "column", "dataframe", "period", "range"]
)
def test_arima_endog_input_forms(form):
    # endog is documented as array_like: every form of a 1-D series gives the
    # estimates of the ndarray, and pandas input gives pandas forecasts that
    # continue the index
    y, _ = _contract_arima_data()
    endog = _contract_arima_endog_forms(y)[form]
    res = ARIMA(endog, order=(2, 0, 0), trend="n").fit(method="yule_walker")
    params = _yule_walker_by_hand(y, 2)
    assert_allclose(np.asarray(res.params), params, rtol=1e-10)
    fcast = res.forecast(2)
    assert_allclose(
        np.asarray(fcast)[0], params[0] * y[-1] + params[1] * y[-2], rtol=1e-10
    )
    if not isinstance(endog, (pd.Series, pd.DataFrame)):
        assert isinstance(fcast, np.ndarray)
    elif isinstance(endog.index, pd.RangeIndex):
        assert list(fcast.index) == [120, 121]
    elif isinstance(endog.index, pd.PeriodIndex):
        expected = pd.period_range("2010-01", periods=2, freq="M")
        assert list(fcast.index) == list(expected)
    else:
        expected = pd.date_range("2010-01-01", periods=2, freq="MS")
        assert list(fcast.index) == list(expected)


def test_arima_endog_integer():
    # Integer endog gives the result of the same values as floats
    y, _ = _contract_arima_data()
    y_int = np.round(4 * y).astype(np.int64)
    res = ARIMA(y_int, order=(2, 0, 0), trend="n").fit(method="yule_walker")
    assert_allclose(res.params, _yule_walker_by_hand(y_int.astype(float), 2), rtol=1e-10)


@pytest.mark.parametrize("form", ["list", "dataframe", "1d"])
def test_arima_exog_input_forms(form):
    # exog is documented as array_like. Compare the likelihood at fixed
    # parameters with that of the ndarray
    y, x = _contract_arima_data()
    if form == "1d":
        x = x[:, :1]
    if form == "list":
        exog = x.tolist()
    elif form == "dataframe":
        exog = pd.DataFrame(x, columns=["a", "b"])
    else:
        exog = x[:, 0]
    mod = ARIMA(y, exog, order=(1, 0, 0), trend="n")
    expected = ARIMA(y, x, order=(1, 0, 0), trend="n")
    params = np.r_[np.linspace(0.5, -0.5, x.shape[1]), 0.6, 1.2]
    assert_allclose(mod.loglike(params), expected.loglike(params), rtol=1e-12)
    if form == "dataframe":
        assert mod.exog_names == ["a", "b"]


@pytest.mark.parametrize(
    "order", [(2, 0, 0), [2, 0, 0], np.array([2, 0, 0]), ([1, 2], 0, 0)], ids=str
)
def test_arima_order_forms(order):
    # order is documented as a tuple; lists, arrays and lists of lags work too
    y, _ = _contract_arima_data()
    res = ARIMA(y, order=order, trend="n").fit(method="yule_walker")
    assert_allclose(res.params, _yule_walker_by_hand(y, 2), rtol=1e-10)


@pytest.mark.parametrize(
    "trend, n_trend",
    [("n", 0), ("c", 1), ("t", 1), ("ct", 2), ("ctt", 3), ([1, 0, 1], 2)],
    ids=str,
)
def test_arima_trend_values(trend, n_trend):
    # The trend is part of exog: the polynomial in t = 1, 2, ..., nobs
    y, _ = _contract_arima_data()
    mod = ARIMA(y, order=(1, 0, 0), trend=trend)
    assert mod.k_trend == n_trend
    powers = {"n": [], "c": [0], "t": [1], "ct": [0, 1], "ctt": [0, 1, 2]}
    powers = powers.get(trend, [0, 2]) if isinstance(trend, str) else [0, 2]
    t = np.arange(1.0, 121.0)
    expected = np.column_stack([t**p for p in powers]) if powers else None
    if expected is None:
        assert mod.exog is None
    else:
        assert_allclose(mod.exog, expected, rtol=1e-12)


def test_arima_invalid_trend():
    y, _ = _contract_arima_data()
    with pytest.raises(ValueError, match="Valid trend inputs are"):
        ARIMA(y, order=(1, 0, 0), trend="x")
    with pytest.raises(ValueError, match="trend terms of lower order than"):
        ARIMA(y, order=(1, 1, 0), trend="c")


def test_arima_missing_values():
    # missing="raise" raises if endog contains NaNs
    from statsmodels.tools.sm_exceptions import MissingDataError

    y, _ = _contract_arima_data()
    y_nan = y.copy()
    y_nan[5] = np.nan
    with pytest.raises(MissingDataError):
        ARIMA(y_nan, order=(1, 0, 0), missing="raise")
    with pytest.raises(ValueError, match="missing must be one of"):
        ARIMA(y_nan, order=(1, 0, 0), missing="x")


def _start_params_model(fixed):
    import contextlib

    rng = np.random.default_rng(20261135)
    e = rng.standard_normal(100)
    y = np.empty(100)
    y[0] = e[0]
    for t in range(1, 100):
        y[t] = 0.5 * y[t - 1] + e[t]
    mod = ARIMA(10 + y, order=(1, 0, 0))
    if fixed:
        return mod, mod.fix_params({"const": 10.0})
    return mod, contextlib.nullcontext()


@pytest.mark.parametrize(
    "fixed, includes_fixed, start_params",
    [
        (False, False, [0.5]),
        (False, False, [10.0, 0.5]),
        (False, True, [10.0, 0.5]),
        (True, True, [10.0, 0.5]),
        (True, False, []),
        (True, False, [0.5, 1.0, 1.0]),
        (True, False, [10.0, 0.5, 1.0, 1.0]),
    ],
)
def test_fit_start_params_wrong_length(fixed, includes_fixed, start_params):
    # These lengths raised an IndexError or a ValueError from numpy, or
    # are too long for the free parameters
    mod, fix = _start_params_model(fixed)
    n_params = 2 if fixed and not includes_fixed else 3
    msg = f"start_params has {len(start_params)} elements, but the model has"
    with fix, pytest.raises(ValueError, match=f"{msg} {n_params} "):
        mod.fit(start_params=start_params, includes_fixed=includes_fixed)


@pytest.mark.parametrize("includes_fixed", [False, True])
def test_fit_start_params_too_long_deprecated(includes_fixed):
    # Extra elements returned a result with as many parameters as elements.
    # They are deprecated and the result is unchanged
    from statsmodels.tsa.statespace import sarimax

    mod, _ = _start_params_model(False)
    start_params = [10.0, 0.5, 1.0, 9.0, 9.0]
    with pytest.warns(FutureWarning, match="start_params has 5 elements, but"):
        res = mod.fit(start_params=start_params, includes_fixed=includes_fixed)
    expected = sarimax.SARIMAX.fit(
        mod, start_params=start_params, includes_fixed=includes_fixed, disp=0
    )
    assert res.params.shape == (5,)
    assert_allclose(res.params, expected.params, rtol=1e-12)
    assert_allclose(res.forecast(3), expected.forecast(3), rtol=1e-12)


@pytest.mark.parametrize(
    "includes_fixed, start_params, expected_start",
    [(True, [10.0, 0.5, 1.0, 9.0], [10.0, 0.5, 1.0]), (False, [0.5], [0.5, 0.5])],
)
def test_fit_start_params_fixed_deprecated(includes_fixed, start_params, expected_start):
    # With fixed parameters, extra elements were ignored and a single
    # element was used for every free parameter
    mod, fix = _start_params_model(True)
    with fix:
        with pytest.warns(FutureWarning, match="This is deprecated and will raise"):
            res = mod.fit(start_params=start_params, includes_fixed=includes_fixed)
        expected = mod.fit(start_params=expected_start, includes_fixed=includes_fixed)
    assert res.params.shape == (3,)
    assert_allclose(res.params, expected.params, rtol=1e-12)


@pytest.mark.parametrize(
    "fixed, includes_fixed, start_params",
    [
        (False, False, [10.0, 0.5, 1.0]),
        (False, True, [10.0, 0.5, 1.0]),
        (True, False, [0.5, 1.0]),
        (True, True, [10.0, 0.5, 1.0]),
    ],
)
def test_fit_start_params_right_length(fixed, includes_fixed, start_params):
    # The right number of elements does not warn
    import warnings

    mod, fix = _start_params_model(fixed)
    with fix, warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        res = mod.fit(start_params=start_params, includes_fixed=includes_fixed)
    assert res.params.shape == (3,)
