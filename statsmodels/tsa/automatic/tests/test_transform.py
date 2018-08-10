import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose

from statsmodels.tsa.automatic import transform

current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
data = macrodata['cpi']
new_data = pd.read_csv('../CPIAPPNS.csv')
new_data.index = pd.PeriodIndex(start='04-2010', end='03-2018', freq='M')
new_data = new_data['CPIAPPNS']


def test_transform():
    lambda_d0s1 = transform.predict_lambda(endog=data, lambda_range=(-1, 2),
                                           d=0, offset=False,
                                           seasonal_periods=1, gridsize=61)
    lambda_d1s1 = transform.predict_lambda(endog=data, lambda_range=(-1, 2),
                                           d=1, offset=False,
                                           seasonal_periods=1, gridsize=61)
    lambda_d1s12 = transform.predict_lambda(endog=data, lambda_range=(-1, 2),
                                            d=1, offset=False,
                                            seasonal_periods=12, gridsize=61)
    assert_allclose(lambda_d0s1, 0.35)
    assert_allclose(lambda_d1s1, 0.6)
    assert_allclose(lambda_d1s12, 0.6)
    # assert_equal(result, 0.35)


def test_unit_transform():
    lambda_d0s1 = transform.predict_lambda(endog=new_data,
                                           lambda_range=(-1, 2),
                                           d=0, offset=False,
                                           seasonal_periods=1, gridsize=61)
    lambda_d1s1 = transform.predict_lambda(endog=new_data,
                                           lambda_range=(-1, 2),
                                           d=1, offset=False,
                                           seasonal_periods=1, gridsize=61)
    assert_allclose(lambda_d0s1, 2.0)
    assert_allclose(lambda_d1s1, 2.0)
