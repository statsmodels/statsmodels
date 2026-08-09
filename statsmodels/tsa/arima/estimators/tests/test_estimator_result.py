import numpy as np

from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators._base import ARMAEstimationResult
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson
from statsmodels.tsa.arima.estimators.gls import gls
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
    innovations,
    innovations_mle,
)
from statsmodels.tsa.arima.estimators.statespace import statespace
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker


def _check(result):
    assert isinstance(result, ARMAEstimationResult)
    assert len(result) == 2
    parameters, other_results = result
    assert result.parameters is parameters
    assert result.other_results is other_results


def test_burg_returns_estimator_result():
    _check(burg(lake, ar_order=2))


def test_yule_walker_returns_estimator_result():
    _check(yule_walker(lake, ar_order=2))


def test_hannan_rissanen_returns_estimator_result():
    _check(hannan_rissanen(lake, ar_order=1, ma_order=1))


def test_innovations_returns_estimator_result():
    _check(innovations(lake, ma_order=2))


def test_innovations_mle_returns_estimator_result():
    _check(innovations_mle(lake, order=(1, 0, 0)))


def test_durbin_levinson_returns_estimator_result():
    _check(durbin_levinson(lake, ar_order=2))


def test_statespace_returns_estimator_result():
    _check(statespace(lake, order=(1, 0, 0)))


def test_gls_returns_estimator_result():
    endog = lake.copy()
    exog = np.ones_like(endog)
    _check(gls(endog, exog, order=(1, 0, 0), max_iter=2))
