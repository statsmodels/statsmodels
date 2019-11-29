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
