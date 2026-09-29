"""unit test for automatic sarimax forecasting."""
import os
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.automatic import exponentialsmoothing
from statsmodels import datasets
from numpy.testing import assert_equal


curdir = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
new_data = pd.read_csv(os.path.join(curdir, 'CPIAPPNS.csv'))
new_data.index = pd.PeriodIndex(start='04-2010', end='03-2018', freq='M')


def test_smoke_auto_es():
    """Test function for auto_es."""
    trend, seasonal = exponentialsmoothing.auto_es(macrodata['cpi'])
    desired_trend = 'add'
    desired_seasonal = None
    assert_equal(trend, desired_trend)
    assert_equal(seasonal, desired_seasonal)


def test_auto_es():
    """Test function for auto_es against ets()."""
    trend, seasonal = exponentialsmoothing.auto_es(macrodata['cpi'])
    desired_trend = 'add'
    desired_seasonal = None
    assert_equal(trend, desired_trend)
    assert_equal(seasonal, desired_seasonal)


def test_auto_es_infl():
    """Test function for auto_es against ets()."""
    trend, seasonal = exponentialsmoothing.auto_es(macrodata['infl'])
    desired_trend = None
    desired_seasonal = None
    assert_equal(trend, desired_trend)
    assert_equal(seasonal, desired_seasonal)


def test_auto_es_CPIAPPNS():
    """Test function for auto_es against ets()."""
    trend, seasonal = exponentialsmoothing.auto_es(new_data['CPIAPPNS'])
    desired_trend = None
    desired_seasonal = None
    assert_equal(trend, desired_trend)
    assert_equal(seasonal, desired_seasonal)
