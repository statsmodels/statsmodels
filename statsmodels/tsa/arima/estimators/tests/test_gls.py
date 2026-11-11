from contextlib import nullcontext

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
from statsmodels.tsa.arima_process import arma_acovf


@pytest.mark.parametrize("arma_estimator", ["innovations_mle", "statespace"])
@pytest.mark.parametrize("component", ["ar", "ma"])
@pytest.mark.parametrize("lag, seasonal", [(1, False), (2, False), (4, True)])
def test_gls_whitening_lags(arma_estimator, component, lag, seasonal):
    # Compare one feasible GLS step with the closed-form matrix estimator.
    # For an AR process with just lag s, gamma(k*s) = phi**k / (1-phi**2)
    # and all other autocovariances are zero (unit innovation variance).
    # For an MA process with just lag s, gamma(0) = 1 + theta**2,
    # gamma(s) = theta, and all other autocovariances are zero.
    rng = np.random.default_rng(24680)
    nobs = 80
    innovations = rng.standard_normal(nobs + 200)
    errors = innovations.copy()
    for t in range(lag, len(errors)):
        previous = errors if component == "ar" else innovations
        errors[t] += 0.6 * previous[t - lag]
    exog = np.column_stack([np.ones(nobs), rng.standard_normal(nobs)])
    endog = exog @ np.array([2.0, 0.7]) + errors[-nobs:]

    order = (0, 0, 0)
    seasonal_order = (0, 0, 0, 0)
    if seasonal:
        seasonal_order = (1, 0, 0, lag) if component == "ar" else (0, 0, 1, lag)
    else:
        order = ([lag], 0, 0) if component == "ar" else (0, 0, [lag])

    result = gls(
        endog,
        exog,
        order=order,
        seasonal_order=seasonal_order,
        arma_estimator=arma_estimator,
        n_iter=1,
    )
    prefix = "seasonal_" if seasonal else ""
    coefficient = getattr(result.parameters, f"{prefix}{component}_params")[0]
    autocovariances = np.zeros(nobs)
    if component == "ar":
        autocovariances[::lag] = (
            coefficient ** np.arange(len(autocovariances[::lag]))
            / (1 - coefficient**2)
        )
    else:
        autocovariances[0] = 1 + coefficient**2
        autocovariances[lag] = coefficient
    covariance = toeplitz(autocovariances)
    precision_exog = np.linalg.solve(covariance, exog)
    precision_endog = np.linalg.solve(covariance, endog)
    expected = np.linalg.solve(exog.T @ precision_exog, exog.T @ precision_endog)

    assert_allclose(result.parameters.exog_params, expected, rtol=1e-10, atol=1e-10)


def _lag_polynomial(lags, coefficients, period, sign):
    # Coefficients of 1 + sign * sum_j coefficients[j] * L**(period * lags[j]),
    # where sign is -1 for an AR and +1 for an MA polynomial. `lags` is an int
    # (all lags up to it) or a list of lags, as in the `order` argument.
    lags = list(range(1, lags + 1)) if isinstance(lags, int) else list(lags)
    if len(lags) == 0:
        return np.ones(1)
    polynomial = np.zeros(period * max(lags) + 1)
    polynomial[0] = 1.0
    polynomial[period * np.array(lags)] = sign * np.asarray(coefficients)
    return polynomial


def _difference(x, k_diff, k_seasonal_diff, period):
    for _ in range(k_diff):
        x = x[1:] - x[:-1]
    for _ in range(k_seasonal_diff):
        x = x[period:] - x[:-period]
    return x


_PRODUCT_SPECS = [
    # Non-seasonal and seasonal factors in the same polynomial
    ((1, 0, 0), (1, 0, 0, 4)),
    ((0, 0, 1), (0, 0, 1, 4)),
    ((1, 0, 1), (1, 0, 1, 4)),
    (([2], 0, 0), (1, 0, 0, 3)),
    # Differencing: D=1 alone, D=1 with an AR factor, and d=1
    ((0, 0, 0), (0, 1, 1, 4)),
    ((1, 0, 0), (0, 1, 1, 4)),
    ((1, 1, 0), (0, 0, 1, 4)),
]


@pytest.mark.parametrize(
    "order, seasonal_order, arma_estimator",
    [
        pytest.param(
            order,
            seasonal_order,
            arma_estimator,
            id=f"{arma_estimator}-{order}x{seasonal_order}",
        )
        for arma_estimator in ["innovations_mle", "statespace"]
        for order, seasonal_order in _PRODUCT_SPECS
    ]
    # Hannan-Rissanen does not support seasonal terms; sparse lags only
    + [
        pytest.param(
            ([2], 0, [2]),
            (0, 0, 0, 0),
            "hannan_rissanen",
            id="hannan_rissanen-([2], 0, [2])x(0, 0, 0, 0)",
        )
    ],
)
def test_gls_whitening_polynomial_products(order, seasonal_order, arma_estimator):
    # The reference does not use the innovations algorithm of the estimator.
    # The data and the regressors are differenced here (differencing is not
    # part of the whitening polynomials), the AR and MA polynomials of the
    # differenced error process are the products (np.convolve) of the
    # non-seasonal and the seasonal factors built from the parameters that
    # the estimator found, the autocovariances come from arma_acovf, and the
    # GLS normal equations are solved with np.linalg.solve.
    rng = np.random.default_rng(13579)
    nobs = 80
    k_diff, k_seasonal_diff, period = order[1], seasonal_order[1], seasonal_order[3]

    # Simulate regression errors with all ARMA coefficients nonzero (0.5 for
    # AR, 0.4 for MA), integrated if there is differencing
    ar = np.convolve(
        _lag_polynomial(order[0], 0.5, 1, -1),
        _lag_polynomial(seasonal_order[0], 0.5, period, -1),
    )
    ma = np.convolve(
        _lag_polynomial(order[2], 0.4, 1, 1),
        _lag_polynomial(seasonal_order[2], 0.4, period, 1),
    )
    integration = np.ones(1)
    for _ in range(k_diff):
        integration = np.convolve(integration, [1.0, -1.0])
    for _ in range(k_seasonal_diff):
        seasonal_difference = np.zeros(period + 1)
        seasonal_difference[[0, -1]] = [1.0, -1.0]
        integration = np.convolve(integration, seasonal_difference)
    innovations = rng.standard_normal(nobs + 200)
    stationary = lfilter(ma, ar, innovations)[200:]
    errors = lfilter([1.0], integration, stationary)
    exog = rng.standard_normal((nobs, 2))
    endog = exog @ np.array([0.7, -0.4]) + errors

    integrated = k_diff + k_seasonal_diff > 0
    with pytest.warns(SpecificationWarning) if integrated else nullcontext():
        result = gls(
            endog,
            exog,
            order=order,
            seasonal_order=seasonal_order,
            include_constant=False,
            arma_estimator=arma_estimator,
            n_iter=1,
        )
    params = result.parameters
    ar_poly = np.convolve(
        _lag_polynomial(order[0], params.ar_params, 1, -1),
        _lag_polynomial(seasonal_order[0], params.seasonal_ar_params, period, -1),
    )
    ma_poly = np.convolve(
        _lag_polynomial(order[2], params.ma_params, 1, 1),
        _lag_polynomial(seasonal_order[2], params.seasonal_ma_params, period, 1),
    )

    endog_diff = _difference(endog, k_diff, k_seasonal_diff, period)
    exog_diff = np.column_stack(
        [_difference(col, k_diff, k_seasonal_diff, period) for col in exog.T]
    )
    covariance = toeplitz(arma_acovf(ar_poly, ma_poly, nobs=len(endog_diff)))
    precision_exog = np.linalg.solve(covariance, exog_diff)
    precision_endog = np.linalg.solve(covariance, endog_diff)
    expected = np.linalg.solve(
        exog_diff.T @ precision_exog, exog_diff.T @ precision_endog
    )

    assert_allclose(params.exog_params, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.low_precision(
    "Test against Example 6.6.1 in Brockwell and Davis (2016)"
)
def test_brockwell_davis_example_661():
    endog = oshorts.copy()
    exog = np.ones_like(endog)

    # Here we restrict the iterations to 1 and test against the values in the
    # text (set tolerance=1 to suppress to warning that it didn't converge)
    res = gls(endog, exog, order=(0, 0, 1), max_iter=1, tolerance=1).parameters
    assert_allclose(res.exog_params, -4.745, atol=1e-3)
    assert_allclose(res.ma_params, -0.818, atol=1e-3)
    assert_allclose(res.sigma2, 2041, atol=1)

    # Here we do not restrict the iterations and test against the values in
    # the last row of Table 6.2 (note: this table does not report sigma2)
    res = gls(endog, exog, order=(0, 0, 1)).parameters
    assert_allclose(res.exog_params, -4.780, atol=1e-3)
    assert_allclose(res.ma_params, -0.848, atol=1e-3)


@pytest.mark.low_precision(
    "Test against Example 6.6.2 in Brockwell and Davis (2016)"
)
def test_brockwell_davis_example_662():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    res = gls(endog, exog, order=(2, 0, 0)).parameters

    # Parameter values taken from Table 6.3 row 2, except for sigma2 and the
    # last digit of the exog_params[0], which were given in the text
    assert_allclose(res.exog_params, [10.091, -0.0216], atol=1e-3)
    assert_allclose(res.ar_params, [1.005, -0.291], atol=1e-3)
    assert_allclose(res.sigma2, 0.4571, atol=1e-3)


def test_integrated():
    # Get the lake data
    endog1 = lake.copy()
    exog1 = np.c_[np.ones_like(endog1), np.arange(1, len(endog1) + 1) * 1.0]

    endog2 = np.r_[0, np.cumsum(endog1)]
    exog2 = np.c_[[0, 0], np.cumsum(exog1, axis=0).T].T

    # Estimate without integration
    p1 = gls(endog1, exog1, order=(1, 0, 0)).parameters

    # Estimate with integration
    with pytest.warns(UserWarning, match="Provided `endog` and `exog"):
        p2 = gls(endog2, exog2, order=(1, 1, 0)).parameters

    assert_allclose(p1.params, p2.params)


def test_integrated_invalid():
    # Test for invalid versions of integrated model
    # - include_constant=True is invalid if integration is present
    endog = lake.copy()
    exog = np.arange(1, len(endog) + 1) * 1.0
    with pytest.raises(ValueError):
        gls(endog, exog, order=(1, 1, 0), include_constant=True)


def test_results():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    # Test for results output
    _result = gls(endog, exog, order=(1, 0, 0))
    p, res = _result.parameters, _result.other_results

    assert ("params" in res)
    assert ("converged" in res)
    assert ("differences" in res)
    assert ("iterations" in res)
    assert ("arma_estimator" in res)
    assert ("arma_results" in res)

    assert (res.converged)
    assert (res.iterations > 0)
    assert_equal(res.arma_estimator, "innovations_mle")
    assert_equal(len(res.params), res.iterations + 1)
    assert_equal(len(res.differences), res.iterations + 1)
    assert_equal(len(res.arma_results), res.iterations + 1)
    assert_equal(res.params[-1], p)


def test_iterations():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    # Test for n_iter usage
    res = gls(endog, exog, order=(1, 0, 0), n_iter=1).other_results
    assert_equal(res.iterations, 1)
    assert_equal(res.converged, None)


def test_misc():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    # Test for warning if iterations fail to converge
    with pytest.warns(UserWarning, match="Feasible GLS failed to converg"):
        gls(endog, exog, order=(2, 0, 0), max_iter=0)


@pytest.mark.todo("Low priority: test full GLS against another package")
@pytest.mark.smoke
def test_alternate_arma_estimators_valid():
    # Test that we can use (valid) alternate ARMA estimators
    # Note that this does not test the results of the alternative estimators,
    # and so it is labeled as a smoke test / TODO. However, assuming those
    # estimators are tested elsewhere, the main testable concern from their
    # inclusion in the feasible GLS step is that produce results at all.
    # Thus, for example, we specify n_iter=1, and ignore the actual results.
    # Nonetheless, it would be good to test against another package.

    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    res_yw = gls(
        endog, exog=exog, order=(1, 0, 0), arma_estimator="yule_walker", n_iter=1
    ).other_results
    assert_equal(res_yw.arma_estimator, "yule_walker")

    res_b = gls(endog, exog=exog, order=(1, 0, 0), arma_estimator="burg", n_iter=1).other_results
    assert_equal(res_b.arma_estimator, "burg")

    res_i = gls(
        endog, exog=exog, order=(0, 0, 1), arma_estimator="innovations", n_iter=1
    ).other_results
    assert_equal(res_i.arma_estimator, "innovations")

    res_hr = gls(
        endog, exog=exog, order=(1, 0, 1), arma_estimator="hannan_rissanen", n_iter=1
    ).other_results
    assert_equal(res_hr.arma_estimator, "hannan_rissanen")

    res_ss = gls(
        endog, exog=exog, order=(1, 0, 1), arma_estimator="statespace", n_iter=1
    ).other_results
    assert_equal(res_ss.arma_estimator, "statespace")

    # Finally, default method is innovations
    res_imle = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1).other_results
    assert_equal(res_imle.arma_estimator, "innovations_mle")


def test_alternate_arma_estimators_invalid():
    # Test that specifying an invalid ARMA estimators raises an error
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    # Test for invalid estimator
    with pytest.raises(ValueError):
        gls(
            endog,
            exog,
            order=(0, 0, 1),
            arma_estimator="invalid_estimator",
        )

    # Yule-Walker, Burg can only handle consecutive AR
    with pytest.raises(ValueError):
        gls(endog, exog, order=(0, 0, 1), arma_estimator="yule_walker")
    with pytest.raises(ValueError):
        gls(
            endog,
            exog,
            order=(0, 0, 0),
            seasonal_order=(1, 0, 0, 4),
            arma_estimator="yule_walker",
        )
    with pytest.raises(ValueError):
        gls(endog, exog, order=([0, 1], 0, 0), arma_estimator="yule_walker")

    with pytest.raises(ValueError):
        gls(endog, exog, order=(0, 0, 1), arma_estimator="burg")
    with pytest.raises(ValueError):
        gls(
            endog,
            exog,
            order=(0, 0, 0),
            seasonal_order=(1, 0, 0, 4),
            arma_estimator="burg",
        )
    with pytest.raises(ValueError):
        gls(endog, exog, order=([0, 1], 0, 0), arma_estimator="burg")

    # Innovations (MA) can only handle consecutive MA
    with pytest.raises(ValueError):
        gls(endog, exog, order=(1, 0, 0), arma_estimator="innovations")
    with pytest.raises(ValueError):
        gls(
            endog,
            exog,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 1, 4),
            arma_estimator="innovations",
        )
    with pytest.raises(ValueError):
        gls(endog, exog, order=(0, 0, [0, 1]), arma_estimator="innovations")

    # Hannan-Rissanen can't handle seasonal components
    with pytest.raises(ValueError):
        gls(
            endog,
            exog,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 1, 4),
            arma_estimator="hannan_rissanen",
        )


def test_arma_kwargs():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]

    # Test with the default method for scipy.optimize.minimize (BFGS)
    res1_imle = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1).other_results
    assert_equal(res1_imle.arma_estimator_kwargs, {})
    assert_equal(
        res1_imle.arma_results[1].minimize_results.message,
        "Optimization terminated successfully.",
    )

    # Now specify a different method (L-BFGS-B)
    arma_estimator_kwargs = {"minimize_kwargs": {"method": "L-BFGS-B"}}
    res2_imle = gls(
        endog,
        exog=exog,
        order=(1, 0, 1),
        n_iter=1,
        arma_estimator_kwargs=arma_estimator_kwargs,
    ).other_results
    assert_equal(res2_imle.arma_estimator_kwargs, arma_estimator_kwargs)
    msg = res2_imle.arma_results[1].minimize_results.message
    if isinstance(msg, bytes):
        msg = msg.decode("utf-8")
    assert msg.startswith("CONVERGENCE: REL")
    assert msg.endswith("FACTR*EPSMCH")
