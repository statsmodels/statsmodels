import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
    _package_fixed_and_free_params_info,
    _stitch_fixed_and_free_params,
    _validate_fixed_params,
    hannan_rissanen,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.innovations.arma_innovations import arma_innovations


@pytest.mark.low_precision(
    "Test against Example 5.1.7 in Brockwell and Davis (2016)"
)
def test_brockwell_davis_example_517():
    # Get the lake data
    endog = lake.copy()

    # BD do not implement the "bias correction" third step that they describe,
    # so we can't use their results to test that. Thus here `unbiased=False`.
    # Note: it's not clear why BD use initial_order=22 (and they don't mention
    # that they do this), but it is the value that allows the test to pass.
    hr = hannan_rissanen(
        endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False
    ).parameters
    assert_allclose(hr.ar_params, [0.6961], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3788], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4774, atol=1e-4)


def test_itsmr():
    # This is essentially a high precision version of
    # test_brockwell_davis_example_517, where the desired values were computed
    # from R itsmr::hannan; see results/results_hr.R
    endog = lake.copy()
    hr = hannan_rissanen(
        endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False
    ).parameters

    assert_allclose(hr.ar_params, [0.69607715], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3787969217], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


def test_initial_order():
    endog = np.arange(20) * 1.0
    with pytest.raises(ValueError, match="initial_ar_order"):
        hannan_rissanen(endog, ar_order=2, ma_order=0, initial_ar_order=1)
    with pytest.raises(ValueError, match="initial_ar_order"):
        hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=1)
    with pytest.raises(ValueError, match="initial_ar_order"):
        hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=20)


def test_invalid_orders():
    endog = np.arange(2) * 1.0
    with pytest.raises(ValueError, match="ar_order"):
        hannan_rissanen(endog, ar_order=2)
    with pytest.raises(ValueError, match="ma_order"):
        hannan_rissanen(endog, ma_order=2)


@pytest.mark.todo("Improve checks on valid order parameters.")
@pytest.mark.smoke
def test_nonconsecutive_lags():
    rs = np.random.default_rng(392039)
    endog = rs.standard_normal(200) * 1.0
    hannan_rissanen(endog, ar_order=[1, 4])
    hannan_rissanen(endog, ma_order=[1, 3])
    hannan_rissanen(endog, ar_order=[1, 4], ma_order=[1, 3])
    hannan_rissanen(endog, ar_order=[0, 0, 1])
    hannan_rissanen(endog, ma_order=[0, 0, 1])
    hannan_rissanen(endog, ar_order=[0, 0, 1], ma_order=[0, 0, 1])

    hannan_rissanen(endog, ar_order=0, ma_order=0)


def test_unbiased_error():
    # Test that we get the appropriate error when we specify unbiased=True
    # but the second-stage yields non-stationary parameters.
    endog = np.arange(1000) * 1.0
    with pytest.raises(ValueError, match="Cannot perform third step"):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)


def test_unbiased_default_skips_non_stationary_step2():
    # With unbiased=None, a non-stationary second step skips the third step.
    # The third step would fall back to the second-step result with an
    # EstimationWarning, so any EstimationWarning is an error here: only
    # skipping the third step passes.
    endog = np.arange(1000) * 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", EstimationWarning)
        res_none = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=None)
        res_false = hannan_rissanen(
            endog, ar_order=1, ma_order=1, unbiased=False
        )
    assert not (res_false.parameters.is_stationary
                and res_false.parameters.is_invertible)
    assert_allclose(res_none.parameters.params, res_false.parameters.params)
    assert_allclose(res_none.other_results.resid,
                    res_false.other_results.resid)


@pytest.mark.parametrize("unbiased", [None, True])
@pytest.mark.parametrize(
    "ar_poly, ma_poly, ar_order, ma_order, seed, outside",
    [
        # Near-unit MA root: step 2 is invertible but the step 3 correction
        # is not.
        pytest.param([1, -0.5], [1, 0.9], 1, 1, 53, "ma", id="arma11-ma-root"),
        # Step 2 is (0.6417, -0.3945); the step 3 correction makes the AR
        # parameter 1.0297, which is non-stationary, the MA part is fine.
        pytest.param([1, -0.9], [1, -0.5], 1, 1, 58, "ar", id="arma11-ar-root"),
        pytest.param([1], [1, 0.95], 0, 1, 1, "ma", id="ma1"),
        pytest.param(
            [1, 0, -0.5], [1, 0, 0, 0.9], [2], [2, 3], 19, "ma",
            id="sparse-ar2-ma23",
        ),
    ],
)
def test_unbiased_fallback_to_step2(
    unbiased, ar_poly, ma_poly, ar_order, ma_order, seed, outside
):
    # Step 2 is stationary and invertible, the step 3 correction is not: the
    # step 2 results are returned with a warning. The cases cover both halves
    # of the gate (stationarity, invertibility), a model without AR part and
    # non-consecutive lags. The seeds were found by a search with
    # default_rng(seed) for series of 100 observations.
    rng = np.random.default_rng(seed)
    endog = arma_generate_sample(
        ar_poly, ma_poly, 100, distrvs=rng.standard_normal
    )
    res_step2 = hannan_rissanen(
        endog, ar_order=ar_order, ma_order=ma_order, unbiased=False
    )
    assert res_step2.parameters.is_stationary
    assert res_step2.parameters.is_invertible
    # The independent step 3 leaves the region through the AR or the MA
    # polynomial, as intended by the case.
    spec = res_step2.other_results.spec
    step3 = SARIMAXParams(spec=spec)
    step3.ar_params, step3.ma_params = _bias_correction_reference(
        endog - endog.mean(),
        res_step2.parameters.ar_params,
        res_step2.parameters.ma_params,
        spec.ar_lags,
        spec.ma_lags,
    )
    assert step3.is_stationary == (outside != "ar")
    assert step3.is_invertible == (outside != "ma")
    with pytest.warns(EstimationWarning, match="third step"):
        res = hannan_rissanen(
            endog, ar_order=ar_order, ma_order=ma_order, unbiased=unbiased
        )
    assert_allclose(res.parameters.params, res_step2.parameters.params)
    assert_allclose(res.other_results.resid, res_step2.other_results.resid)


def test_set_default_unbiased():
    # setting unbiased=None with stationary and invertible parameters should
    # yield the exact same results as setting unbiased=True
    endog = lake.copy()
    _result = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=None)
    p_1, other_results_2 = _result.parameters, _result.other_results

    # unbiased=True
    _result = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)
    p_2, other_results_1 = _result.parameters, _result.other_results

    assert_allclose(p_1.ar_params, p_2.ar_params)
    assert_allclose(p_1.ma_params, p_2.ma_params)
    assert_allclose(p_1.sigma2, p_2.sigma2)
    assert_allclose(other_results_1.resid, other_results_2.resid)

    # unbiased=False
    p_3 = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=False).parameters
    assert not np.array_equal(p_1.ar_params, p_3.ar_params)


def _bias_correction_reference(
    endog, ar_params, ma_params, ar_lags=None, ma_lags=None
):
    # Hannan-Rissanen step 3 from Brockwell and Davis (2016), section 5.1.4;
    # `ar_lags` and `ma_lags` are the lags of the parameters (default: 1, ...)
    ar_params, ma_params = np.asarray(ar_params), np.asarray(ma_params)
    if ar_lags is None:
        ar_lags = list(range(1, len(ar_params) + 1))
    if ma_lags is None:
        ma_lags = list(range(1, len(ma_params) + 1))
    p, q, nobs = max(ar_lags, default=0), max(ma_lags, default=0), len(endog)
    m = max(p, q)
    # polynomials with zeros for the lags that are not in the model
    phi = np.zeros(p)
    theta = np.zeros(q)
    phi[np.array(ar_lags, dtype=int) - 1] = ar_params
    theta[np.array(ma_lags, dtype=int) - 1] = ma_params
    z = np.zeros(nobs)
    v = np.zeros(nobs)
    w = np.zeros(nobs)
    for t in range(m, nobs):
        z[t] = (
            endog[t]
            - sum(phi[j] * endog[t - 1 - j] for j in range(p))
            - sum(theta[j] * z[t - 1 - j] for j in range(q))
        )
        v[t] = z[t] + sum(phi[j] * v[t - 1 - j] for j in range(p))
        w[t] = z[t] - sum(theta[j] * w[t - 1 - j] for j in range(q))
    exog = np.array(
        [
            [v[t - lag] for lag in ar_lags] + [w[t - lag] for lag in ma_lags]
            for t in range(m, nobs)
        ]
    )
    correction = np.linalg.lstsq(exog, z[m:], rcond=None)[0]
    k_ar = len(ar_lags)
    return ar_params + correction[:k_ar], ma_params + correction[k_ar:]


@pytest.mark.parametrize(
    "ar_poly, ma_poly, ar_order, ma_order",
    [
        pytest.param([1, -0.5], [1, 0.4], 1, 1, id="arma11"),
        pytest.param([1, -0.5], [1, 0.8], 1, 1, id="arma11-large-ma"),
        pytest.param([1, -0.6, 0.2], [1, 0.5], 2, 1, id="arma21"),
        pytest.param([1, -0.4], [1, 0.5, 0.3], 1, 2, id="arma12"),
        pytest.param(
            [1, -0.5, 0, 0.3], [1, 0.4, 0.3], [1, 0, 1], [1, 1],
            id="sparse-ar13-ma12",
        ),
    ],
)
def test_unbiased_brockwell_davis_reference(
    ar_poly, ma_poly, ar_order, ma_order
):
    # Compare step 3 with an independent loop (ITSM/itsmr lack step 3)
    rng = np.random.default_rng(12345)
    endog = arma_generate_sample(
        ar_poly, ma_poly, 500, distrvs=rng.standard_normal
    )

    step2 = hannan_rissanen(
        endog, ar_order=ar_order, ma_order=ma_order, unbiased=False
    ).parameters
    assert step2.is_stationary
    assert step2.is_invertible
    spec = step2.spec
    desired_ar, desired_ma = _bias_correction_reference(
        endog - endog.mean(),
        step2.ar_params,
        step2.ma_params,
        spec.ar_lags,
        spec.ma_lags,
    )

    with warnings.catch_warnings():
        # the third step must not fall back to the second step
        warnings.simplefilter("error", EstimationWarning)
        actual = hannan_rissanen(
            endog, ar_order=ar_order, ma_order=ma_order, unbiased=True
        ).parameters
    assert_allclose(actual.ar_params, desired_ar)
    assert_allclose(actual.ma_params, desired_ma)

    # One Gauss-Newton step moves the step 2 estimates towards the maximum
    # likelihood estimates
    k = spec.k_ar_params + spec.k_ma_params
    mle = ARIMA(
        endog, order=(ar_order, 0, ma_order), trend="n"
    ).fit().params[:k]
    distance_step2 = np.linalg.norm(step2.params[:k] - mle)
    distance_step3 = np.linalg.norm(actual.params[:k] - mle)
    assert distance_step3 < distance_step2


@pytest.mark.parametrize(
    "ar_order, ma_order, fixed_params, invalid_fixed_params",
    [
        # no fixed param
        (2, [1, 0, 1], None, None),
        ([0, 1], 0, {}, None),
        # invalid fixed params
        (1, 3, {"ar.L2": 1, "ma.L2": 0}, ["ar.L2"]),
        ([0, 1], [0, 0, 1], {"ma.L1": 0, "sigma2": 1}, ["ma.L2", "sigma2"]),
        (0, 0, {"ma.L1": 0, "ar.L1": 0}, ["ar.L1", "ma.L1"]),
        (5, [1, 0], {"random_param": 0, "ar.L1": 0}, ["random_param"]),
        # valid fixed params
        (0, 2, {"ma.L1": -1, "ma.L2": 1}, None),
        (1, 0, {"ar.L1": 0}, None),
        ([1, 0, 1], 3, {"ma.L2": 1, "ar.L3": -1}, None),
        # all fixed
        (2, 2, {"ma.L1": 1, "ma.L2": 1, "ar.L1": 1, "ar.L2": 1}, None),
    ],
)
def test_validate_fixed_params(ar_order, ma_order, fixed_params, invalid_fixed_params):
    # test validation with both _validate_fixed_params and directly with
    # hannan_rissanen

    rs = np.random.RandomState(2381235)
    endog = rs.normal(size=100)
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)

    if invalid_fixed_params is None:
        _validate_fixed_params(fixed_params, spec.param_names)
        hannan_rissanen(
            endog,
            ar_order=ar_order,
            ma_order=ma_order,
            fixed_params=fixed_params,
            unbiased=False,
        )
    else:
        valid_params = sorted(set(spec.param_names) - {"sigma2"})
        msg = (
            f"Invalid fixed parameter(s): {invalid_fixed_params}. "
            f"Please select among {valid_params}."
        )
        # using direct `assert` to test error message instead of `match` since
        # the error message contains regex characters
        with pytest.raises(ValueError) as e:
            _validate_fixed_params(fixed_params, spec.param_names)
            assert e.msg == msg
        with pytest.raises(ValueError) as e:
            hannan_rissanen(
                endog,
                ar_order=ar_order,
                ma_order=ma_order,
                fixed_params=fixed_params,
                unbiased=False,
            )
            assert e.msg == msg


@pytest.mark.parametrize(
    "fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch",
    [
        (
            {},
            [1],
            [],
            Bunch(
                # lags
                fixed_ar_lags=[],
                fixed_ma_lags=[],
                free_ar_lags=[1],
                free_ma_lags=[],
                # ixs
                fixed_ar_ix=np.array([], dtype=int),
                fixed_ma_ix=np.array([], dtype=int),
                free_ar_ix=np.array([0], dtype=int),
                free_ma_ix=np.array([], dtype=int),
                # fixed params
                fixed_ar_params=np.array([]),
                fixed_ma_params=np.array([]),
            ),
        ),
        (
            {"ar.L2": 0.1, "ma.L1": 0.2},
            [2],
            [1, 3],
            Bunch(
                # lags
                fixed_ar_lags=[2],
                fixed_ma_lags=[1],
                free_ar_lags=[],
                free_ma_lags=[3],
                # ixs
                fixed_ar_ix=np.array([1], dtype=int),
                fixed_ma_ix=np.array([0], dtype=int),
                free_ar_ix=np.array([], dtype=int),
                free_ma_ix=np.array([2], dtype=int),
                # fixed params
                fixed_ar_params=np.array([0.1]),
                fixed_ma_params=np.array([0.2]),
            ),
        ),
        (
            {"ma.L5": 0.1, "ma.L10": 0.2},
            [],
            [5, 10],
            Bunch(
                # lags
                fixed_ar_lags=[],
                fixed_ma_lags=[5, 10],
                free_ar_lags=[],
                free_ma_lags=[],
                # ixs
                fixed_ar_ix=np.array([], dtype=int),
                fixed_ma_ix=np.array([4, 9], dtype=int),
                free_ar_ix=np.array([], dtype=int),
                free_ma_ix=np.array([], dtype=int),
                # fixed params
                fixed_ar_params=np.array([]),
                fixed_ma_params=np.array([0.1, 0.2]),
            ),
        ),
    ],
)
def test_package_fixed_and_free_params_info(
    fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch
):
    actual_bunch = _package_fixed_and_free_params_info(
        fixed_params, spec_ar_lags, spec_ma_lags
    )
    assert isinstance(actual_bunch, Bunch)
    assert len(actual_bunch) == len(expected_bunch)
    assert actual_bunch.keys() == expected_bunch.keys()

    # check lags
    lags = ["fixed_ar_lags", "fixed_ma_lags", "free_ar_lags", "free_ma_lags"]
    for k in lags:
        assert isinstance(actual_bunch[k], list)
        assert actual_bunch[k] == expected_bunch[k]

    # check lags
    ixs = ["fixed_ar_ix", "fixed_ma_ix", "free_ar_ix", "free_ma_ix"]
    for k in ixs:
        assert isinstance(actual_bunch[k], np.ndarray)
        assert actual_bunch[k].dtype in [np.int64, np.int32]
        assert_allclose(actual_bunch[k], expected_bunch[k])

    params = ["fixed_ar_params", "fixed_ma_params"]
    for k in params:
        assert isinstance(actual_bunch[k], np.ndarray)
        assert_allclose(actual_bunch[k], expected_bunch[k])


@pytest.mark.parametrize(
    "fixed_lags, free_lags, fixed_params, free_params, "
    "spec_lags, expected_all_params",
    [
        ([], [], [], [], [], []),
        ([2], [], [0.2], [], [2], [0.2]),
        ([], [1], [], [0.2], [1], [0.2]),
        ([1], [3], [0.2], [-0.2], [1, 3], [0.2, -0.2]),
        ([3], [1, 2], [0.2], [0.3, -0.2], [1, 2, 3], [0.3, -0.2, 0.2]),
        ([3, 1], [2, 4], [0.3, 0.1], [0.5, 0.0], [1, 2, 3, 4], [0.1, 0.5, 0.3, 0.0]),
        (
            [3, 10],
            [1, 2],
            [0.2, 0.5],
            [0.3, -0.2],
            [1, 2, 3, 10],
            [0.3, -0.2, 0.2, 0.5],
        ),
        # edge case where 'spec_lags' is somehow not sorted
        (
            [3, 10],
            [1, 2],
            [0.2, 0.5],
            [0.3, -0.2],
            [3, 1, 10, 2],
            [0.2, 0.3, 0.5, -0.2],
        ),
    ],
)
def test_stitch_fixed_and_free_params(
    fixed_lags, free_lags, fixed_params, free_params, spec_lags, expected_all_params
):
    actual_all_params = _stitch_fixed_and_free_params(
        fixed_lags, fixed_params, free_lags, free_params, spec_lags
    )
    assert actual_all_params == expected_all_params


@pytest.mark.parametrize(
    "fixed_params",
    [
        {"ar.L1": 0.69607715},  # fix ar
        {"ma.L1": 0.37879692},  # fix ma
        {"ar.L1": 0.69607715, "ma.L1": 0.37879692},  # no free params
    ],
)
def test_itsmr_with_fixed_params(fixed_params):
    # This test is a variation of test_itsmr where we fix 1 or more parameters
    # for Example 5.1.7 in Brockwell and Davis (2016) and check that free
    # parameters are still correct'.

    endog = lake.copy()
    hr = hannan_rissanen(
        endog,
        ar_order=1,
        ma_order=1,
        demean=True,
        initial_ar_order=22,
        unbiased=False,
        fixed_params=fixed_params,
    ).parameters

    assert_allclose(hr.ar_params, [0.69607715], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3787969217], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


def test_unbiased_error_with_fixed_params():
    # unbiased=True with fixed params should throw NotImplementedError for now
    rs = np.random.RandomState(383382)
    endog = rs.normal(size=1000)
    msg = (
        "Third step of Hannan-Rissanen estimation to remove parameter bias"
        " is not yet implemented for the case with fixed parameters."
    )
    with pytest.raises(NotImplementedError, match=msg):
        hannan_rissanen(
            endog, ar_order=1, ma_order=1, unbiased=True, fixed_params={"ar.L1": 0}
        )


def test_set_default_unbiased_with_fixed_params():
    # setting unbiased=None with fixed params should yield the exact same
    # results as setting unbiased=False
    rs = np.random.RandomState(590193)
    endog = rs.normal(size=1000)
    # unbiased=None
    _result = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=None, fixed_params={"ar.L1": 0.69607715}
    )
    p_1, other_results_2 = _result.parameters, _result.other_results
    # unbiased=False
    _result = hannan_rissanen(
        endog,
        ar_order=1,
        ma_order=1,
        unbiased=False,
        fixed_params={"ar.L1": 0.69607715},
    )
    p_2, other_results_1 = _result.parameters, _result.other_results

    assert_allclose(p_1.ar_params, p_2.ar_params)
    assert_allclose(p_1.ma_params, p_2.ma_params)
    assert_allclose(p_1.sigma2, p_2.sigma2)
    assert_allclose(other_results_1.resid, other_results_2.resid)


def test_hannan_rissanen_seasonal():
    # Seasonal AR or MA terms are not supported, but seasonal differencing is.
    # A pure AR model is the least squares regression of the demeaned
    # seasonal difference on its lags
    x = np.random.default_rng(20261131).standard_normal(200).cumsum()
    with pytest.raises(ValueError, match="does not support seasonal"):
        hannan_rissanen(x, ar_order=1, seasonal_order=(1, 0, 0, 4))
    p, _ = hannan_rissanen(x, ar_order=1, seasonal_order=(0, 1, 0, 4))
    dx = x[4:] - x[:-4]
    dx = dx - dx.mean()
    reg, dep = dx[:-1], dx[1:]
    phi = (reg @ dep) / (reg @ reg)
    resid = dep - phi * reg
    assert_allclose(p.ar_params, [phi], rtol=1e-10)
    assert_allclose(p.sigma2, resid @ resid / (resid.shape[0] - 1), rtol=1e-10)


@pytest.mark.parametrize(
    "ar_order", [[1, 2], (1, 2), np.array([1, 2])], ids=["list", "tuple", "ndarray"]
)
@pytest.mark.parametrize(
    "form", ["list", "tuple", "series", "column", "dataframe", "integer"]
)
def test_hannan_rissanen_input_forms(form, ar_order):
    # endog is documented as array_like and ar_order as int or a sequence
    from statsmodels.tsa.tests._contract import series_forms

    e = np.random.default_rng(20261132).standard_normal(201)
    x = np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.5 * x[t - 1] + e[t + 1] + 0.3 * e[t]
    if form == "integer":
        x = np.round(4 * x)
        endog = x.astype(np.int64)
    else:
        endog = series_forms(x)[form]
    p, _ = hannan_rissanen(endog, ar_order=ar_order, ma_order=1)
    expected, _ = hannan_rissanen(x, ar_order=2, ma_order=1)
    assert_allclose(p.params, expected.params, rtol=1e-12)
