import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose

from statsmodels.tsa.automatic import transform

macrodata = datasets.macrodata.load_pandas().data
data = sm.datasets.macrodata.load_pandas().data['cpi']

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
