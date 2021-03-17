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

    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True,
                            initial_ar_order=22, unbiased=False, fixed_params={"ar":0.695})

    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4774, atol=1e-4)


def test_itsmr():

    endog = lake.copy()
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True,
                            initial_ar_order=22, unbiased=False, fixed_params={"ma": 0.378})

    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


def test_nonconsecutive_lags():
    endog = np.arange(20) * 1.0
    hannan_rissanen(endog, ar_order=[1, 4], fixed_params={'ar':2})
    hannan_rissanen(endog, ma_order=[1, 3], fixed_params={'ma':2})
    hannan_rissanen(endog, ar_order=[0, 0, 1], fixed_params={'ar':2})
    hannan_rissanen(endog, ar_order=[0, 0, 1], ma_order=[0, 0, 1], fixed_params={'ar':2})

    hannan_rissanen(endog, ar_order=0, ma_order=0)


def test_unbiased_error():
    # Test that we get the appropriate error when we specify unbiased=True
    # but the second-stage yields non-stationary parameters.
    endog = (np.arange(1000) * 1.0)
    with pytest.raises(ValueError, match='Cannot perform third step'):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True, fixed_params={"ar":2})
