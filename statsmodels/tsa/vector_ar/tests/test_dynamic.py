from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.util.testing import assert_frame_equal, assert_series_equal

PD_GT_19 = LooseVersion(pd.__version__) > LooseVersion('0.19.2')

pytestmark = pytest.mark.skipif(PD_GT_19,
                                reason='Requires pandas <= 0.19.2')

from statsmodels.tsa.vector_ar.dynamic import _window_ols


@pytest.fixture(params=(0.0, 0.01))
def ols_data(request):
    nobs = 500
    rs = np.random.RandomState(12345)
    x = pd.DataFrame(rs.standard_normal((nobs, 3)),
                     columns=['x{0}'.format(i) for i in range(1, 4)])
    e = rs.standard_normal(500)
    y = pd.Series(x.values.sum(1) + e, name='y')

    if request.param > 0.0:
        locs = rs.random_sample(nobs)
        y.loc[locs < request.param] = np.nan

    return {'y': y, 'x': x}


def assert_ols_equal(res1, res2):
    if isinstance(res1.beta, pd.Series):
        assert_series_equal(res1.beta, res2.beta)
        assert_allclose(res1.r2, res2.r2)
        assert_allclose(res1.nobs, res2.nobs)
    else:
        assert_frame_equal(res1.beta, res2.beta)
        assert_series_equal(res1.nobs, res2.nobs)
        assert_series_equal(res1.r2, res2.r2)
    assert_series_equal(res1.resid, res2.resid)


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_window_ols_full(ols_data):
    y, x = ols_data['y'], ols_data['x']
    res1 = _window_ols(y, x, window_type='full_sample')
    res2 = _window_ols(y, x)
    res3 = pd.ols(y=y, x=x, window_type='full_sample')
    assert_ols_equal(res1, res2)
    assert_ols_equal(res1, res3)


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_window_ols_rolling(ols_data):
    y, x = ols_data['y'], ols_data['x']
    res1 = _window_ols(y, x, window_type='rolling', window=100)
    res2 = pd.ols(y=y, x=x, window_type='rolling', window=100)
    assert_ols_equal(res1, res2)


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_window_ols_expanding(ols_data):
    y, x = ols_data['y'], ols_data['x']
    res1 = _window_ols(y, x, window_type='expanding')
    res2 = pd.ols(y=y, x=x, window_type='expanding')
    assert_ols_equal(res1, res2)
