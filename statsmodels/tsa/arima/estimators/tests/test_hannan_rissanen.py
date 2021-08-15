import numpy as np

import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen


@pytest.mark.low_precision('Test against Example 5.1.7 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_517():
    # Get the lake data
    endog = lake.copy()

    # BD do not implement the "bias correction" third step that they describe,
    # so we can't use their results to test that. Thus here `unbiased=False`.
    # Note: it's not clear why BD use initial_order=22 (and they don't mention
    # that they do this), but it is the value that allows the test to pass.
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True,
                            initial_ar_order=22, unbiased=False)
    assert_allclose(hr.ar_params, [0.6961], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3788], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4774, atol=1e-4)


def test_itsmr():
    # This is essentially a high precision version of
    # test_brockwell_davis_example_517, where the desired values were computed
    # from R itsmr::hannan; see results/results_hr.R
    endog = lake.copy()
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True,
                            initial_ar_order=22, unbiased=False)

    assert_allclose(hr.ar_params, [0.69607715], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3787969217], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_initial_order():
    endog = np.arange(20) * 1.0
    # TODO: shouldn't allow initial_ar_order <= ar_order
    hannan_rissanen(endog, ar_order=2, ma_order=0, initial_ar_order=1)
    # TODO: shouldn't allow initial_ar_order <= ma_order
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=1)
    # TODO: shouldn't allow initial_ar_order >= dataset
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=20)


@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_invalid_orders():
    endog = np.arange(2) * 1.0
    # TODO: shouldn't allow ar_order >= dataset
    hannan_rissanen(endog, ar_order=2)
    # TODO: shouldn't allow ma_order >= dataset
    hannan_rissanen(endog, ma_order=2)


@pytest.mark.todo('Improve checks on valid order parameters.')
@pytest.mark.smoke
def test_nonconsecutive_lags():
    endog = np.arange(20) * 1.0
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
    endog = (np.arange(1000) * 1.0)
    with pytest.raises(ValueError, match='Cannot perform third step'):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)


def test_set_default_unbiased():
    # setting unbiased=None with stationary and invertible parameters should
    # yield the exact same results as setting unbiased=True
    endog = lake.copy()
    p_1, other_results_2 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=None
    )

    # unbiased=True
    p_2, other_results_1 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=True
    )

    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)

    # unbiased=False
    p_3, _ = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=False
    )
    assert not np.array_equal(p_1.ar_params, p_3.ar_params)


@pytest.mark.parametrize(
    "ar_order, ma_order, fixed_params, err_msg",
    [
        # no fixed param
        (2, [1, 0, 1], None, None),
        ([0, 1], 0, {}, None),
        # invalid fixed param names
        (1, 3, {"ar.L2": 1, "ma.L2": 0}, "Invalid fixed parameter"),
        ([0, 1], [0, 1], {"sigma2": 1}, "Invalid fixed parameter"),
        (0, 0, {"ma.L1": 0, "ar.L1": 0}, "Invalid fixed parameter"),
        (5, [1], {"random_param": 0, "ar.L1": 0}, "Invalid fixed parameter"),
        # invalid fixed param values
        (1, 3, {"ar.L1": np.nan, "ma.L2": 0}, "includes NaN or Inf values"),
        ([0, 1], 3, {"ma.L3": np.inf}, "includes NaN or Inf values"),
        # valid fixed params
        (0, 2, {"ma.L1": -1, "ma.L2": 1}, None),
        (1, 0, {"ar.L1": 0}, None),
        ([1, 0, 1], 3, {"ma.L2": 1, "ar.L3": -1}, None),
        # all fixed
        (2, 2, {"ma.L1": 1, "ma.L2": 1, "ar.L1": 1, "ar.L2": 1}, None)
    ]
)
def test_validate_fixed_params(ar_order, ma_order, fixed_params, err_msg):
    # test validation with hannan_rissanen

    endog = np.random.normal(size=100)

    if err_msg is None:
        hannan_rissanen(
            endog, ar_order=ar_order, ma_order=ma_order,
            fixed_params=fixed_params, unbiased=False
        )
    else:
        with pytest.raises(ValueError, match=err_msg):
            hannan_rissanen(
                endog, ar_order=ar_order, ma_order=ma_order,
                fixed_params=fixed_params, unbiased=False
            )


@pytest.mark.parametrize(
    "fixed_params",
    [
        {"ar.L1": 0.69607715},  # fix ar
        {"ma.L1": 0.37879692},  # fix ma
        {"ar.L1": 0.69607715, "ma.L1": 0.37879692},  # no free params
    ]
)
def test_itsmr_with_fixed_params(fixed_params):
    # This test is a variation of test_itsmr where we fix 1 or more parameters
    # for Example 5.1.7 in Brockwell and Davis (2016) and check that free
    # parameters are still correct'.

    endog = lake.copy()
    hr, _ = hannan_rissanen(
        endog, ar_order=1, ma_order=1, demean=True,
        initial_ar_order=22, unbiased=False,
        fixed_params=fixed_params
    )

    assert_allclose(hr.ar_params, [0.69607715], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3787969217], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


def test_unbiased_error_with_fixed_params():
    # unbiased=True with fixed params should throw NotImplementedError for now
    endog = np.random.normal(size=1000)
    msg = (
        "Third step of Hannan-Rissanen estimation to remove parameter bias"
        " is not yet implemented for the case with fixed parameters."
    )
    with pytest.raises(NotImplementedError, match=msg):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True,
                        fixed_params={"ar.L1": 0})


def test_set_default_unbiased_with_fixed_params():
    # setting unbiased=None with fixed params should yield the exact same
    # results as setting unbiased=False
    endog = np.random.normal(size=1000)
    # unbiased=None
    p_1, other_results_2 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=None,
        fixed_params={"ar.L1": 0.69607715}
    )
    # unbiased=False
    p_2, other_results_1 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=False,
        fixed_params={"ar.L1": 0.69607715}
    )

    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)
