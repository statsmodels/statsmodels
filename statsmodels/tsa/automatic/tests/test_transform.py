import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose

from statsmodels.tsa.automatic import transform

macrodata = datasets.macrodata.load_pandas().data


def test_transform():
    result = transform.predict_lambda(macrodata['cpi'], (-1, 2), d=0,
                                      seasonal_periods=1, offset=False,
                                      gridsize=61)
    assert_equal(result, 1)
