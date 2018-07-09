"""unit test for automatic sarimax forecasting."""
import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose

from statsmodels.tsa.automatic import Forecast
from statsmodels.tsa.automatic import ForecastSet

current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')


def test_forecast_smoke():
    f1 = Forecast(macrodata['infl'], sm.tsa.SARIMAX,
                  test_sample=0.3, **{'order': (1, 0, 0)})
    desired_nobs_test = 60
    desired_nobs_training = 143
    # desired_resid
    # desired_fittedvalues
    # desired_forecasts_error
    desired_mae = 2.5427849274382734
    desired_rmse = 3.1849960758607656
    desired_mape = 81.97352912558263
    desired_smape = 151.64134150240392
    desired_mase = 250.74684701127433

    assert_equal(f1.nobs_test, desired_nobs_test)
    assert_equal(f1.nobs_training, desired_nobs_training)
    assert_allclose(f1.mae, desired_mae)
    assert_allclose(f1.rmse, desired_rmse)
    assert_allclose(f1.mape, desired_mape)
    assert_allclose(f1.smape, desired_smape)
    assert_allclose(f1.mase, desired_mase)


def test_forecast_set_smoke():
    fs1 = ForecastSet(macrodata['infl'], test_sample=0.2)
    fs1.add(model=sm.tsa.SARIMAX, **{'order': (1, 0, 0)})
    fs1.add(model=sm.tsa.SARIMAX, **{'order': (1, 0, 1)})
    model = fs1.select()
    desired_mae = 2.0048760787792554
    desired_rmse = 2.9081962554338867
    desired_mape = 87.31706459730462
    desired_smape = 285.5613121872605
    desired_mase = 416.397339438769

    assert_allclose(model.mae, desired_mae)
    assert_allclose(model.rmse, desired_rmse)
    assert_allclose(model.mape, desired_mape)
    assert_allclose(model.smape, desired_smape)
    assert_allclose(model.mase, desired_mase)
