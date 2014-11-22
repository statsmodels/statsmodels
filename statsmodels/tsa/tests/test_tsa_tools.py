'''tests for some time series analysis functions

'''

from statsmodels.compat.python import zip
import warnings
import numpy as np
from numpy.random import randn
from numpy.testing import assert_array_almost_equal, assert_equal, assert_raises
import pandas as pd
from nose import SkipTest
from pandas.util.testing import assert_frame_equal, assert_produces_warning

import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech, reintegrate, unvec
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
from statsmodels.regression.linear_model import OLS
from .results import savedrvs
from .results.datamlw_tls import mlacf, mlccf, mlpacf, mlywar

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.
x1000 = xo / 1000.

pdversion = pd.version.version.split('.')
pdmajor = int(pdversion[0])
pdminor = int(pdversion[1])

macro_data = sm.datasets.macrodata.load().data[['year',
                                                'quarter',
                                                'realgdp',
                                                'cpi']]


def skip_if_early_pandas():
    if pdmajor == 0 and pdminor <= 12:
        raise SkipTest("known failure of test on early pandas")


def test_acf():
    acf_x = tsa.acf(x100, unbiased=False)[:21]
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8)  # why only dec=8
    acf_x = tsa.acf(x1000, unbiased=False)[:21]
    assert_array_almost_equal(mlacf.acf1000.ravel(), acf_x, 8)  # why only dec=9


def test_ccf():
    ccf_x = tsa.ccf(x100[4:], x100[:-4], unbiased=False)[:21]
    assert_array_almost_equal(mlccf.ccf100.ravel()[:21][::-1], ccf_x, 8)
    ccf_x = tsa.ccf(x1000[4:], x1000[:-4], unbiased=False)[:21]
    assert_array_almost_equal(mlccf.ccf1000.ravel()[:21][::-1], ccf_x, 8)


def test_pacf_yw():
    pacfyw = tsa.pacf_yw(x100, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfyw, 1)
    pacfyw = tsa.pacf_yw(x1000, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfyw, 2)
    #assert False


def test_pacf_ols():
    pacfols = tsa.pacf_ols(x100, 20)
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfols, 8)
    pacfols = tsa.pacf_ols(x1000, 20)
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfols, 8)
    #assert False


def test_ywcoef():
    yw = -sm.regression.yule_walker(x100, 10, method='mle')[0]
    ar_coef = mlywar.arcoef100[1:]
    assert_array_almost_equal(ar_coef, yw, 8)
    yw = -sm.regression.yule_walker(x1000, 20, method='mle')[0]
    ar_coef = mlywar.arcoef1000[1:]
    assert_array_almost_equal(yw, ar_coef, 8)



def test_yule_walker_inter():
    # see 1869
    x = np.array([1, -1, 2, 2, 0, -2, 1, 0, -3, 0, 0])
    # it works
    result = sm.regression.yule_walker(x, 3)


def test_duplication_matrix():
    for k in range(2, 10):
        m = tools.unvech(randn(k * (k + 1) / 2))
        duplication = tools.duplication_matrix(k)
        assert_equal(vec(m), np.dot(duplication, vech(m)))


def test_elimination_matrix():
    for k in range(2, 10):
        m = randn(k, k)
        elimination = tools.elimination_matrix(k)
        assert_equal(vech(m), np.dot(elimination, vec(m)))


def test_commutation_matrix():
    m = randn(4, 3)
    commutation = tools.commutation_matrix(4, 3)
    assert_equal(vec(m.T), np.dot(commutation, vec(m)))


def test_vec():
    arr = np.array([[1, 2],
                    [3, 4]])
    assert_equal(vec(arr), [1, 3, 2, 4])


def test_vech():
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    assert_equal(vech(arr), [1, 4, 7, 5, 8, 9])


def test_unvec():
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    assert_equal(unvec(vec(arr)), arr)


def test_lagmat_trim():
    x = np.arange(20.0) + 1.0
    lags = 5
    base_lagmat = sm.tsa.lagmat(x, lags, trim=None)
    forward_lagmat = sm.tsa.lagmat(x, lags, trim='forward')
    assert_equal(base_lagmat[:-lags], forward_lagmat)
    backward_lagmat = sm.tsa.lagmat(x, lags, trim='backward')
    assert_equal(base_lagmat[lags:], backward_lagmat)
    both_lagmat = sm.tsa.lagmat(x, lags, trim='both')
    assert_equal(base_lagmat[lags:-lags], both_lagmat)


def test_lagmat_bad_trim():
    x = randn(10, 1)
    assert_raises(ValueError, sm.tsa.lagmat, x, 1, trim='unknown')


def test_add_lag_insert():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
    data = pd.DataFrame(data)
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3)
    assert_equal(lag_data.values, results)


def test_add_lag_noinsert():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :], lagmat))
    data = pd.DataFrame(data)
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False)
    assert_equal(lag_data.values, results)


def test_add_lag_noinsert_atend():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, -1], 3, trim='Both')
    results = np.column_stack((nddata[3:, :], lagmat))
    data = pd.DataFrame(data)
    lag_data = sm.tsa.add_lag(data, 'cpi', 3, insert=False)
    assert_equal(lag_data.values, results)
    # should be the same as insert
    lag_data2 = sm.tsa.add_lag(data, 'cpi', 3, insert=True)
    assert_equal(lag_data2.values, results)


def test_add_lag_ndarray():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
    lag_data = sm.tsa.add_lag(nddata, 2, 3)
    assert_equal(lag_data, results)


def test_add_lag_noinsert_ndarray():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :], lagmat))
    lag_data = sm.tsa.add_lag(nddata, 2, 3, insert=False)
    assert_equal(lag_data, results)


def test_add_lag_noinsertatend_ndarray():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, -1], 3, trim='Both')
    results = np.column_stack((nddata[3:, :], lagmat))
    lag_data = sm.tsa.add_lag(nddata, 3, 3, insert=False)
    assert_equal(lag_data, results)
    # should be the same as insert also check negative col number
    lag_data2 = sm.tsa.add_lag(nddata, -1, 3, insert=True)
    assert_equal(lag_data2, results)


def test_add_lag1d():
    data = randn(100)
    lagmat = sm.tsa.lagmat(data, 3, trim='Both')
    results = np.column_stack((data[3:], lagmat))
    lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
    assert_equal(results, lag_data)

    # add index
    data = data[:, None]
    lagmat = sm.tsa.lagmat(data, 3, trim='Both')  # test for lagmat too
    results = np.column_stack((data[3:], lagmat))
    lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
    assert_equal(results, lag_data)


def test_add_lag1d_drop():
    data = randn(100)
    lagmat = sm.tsa.lagmat(data, 3, trim='Both')
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True, insert=True)
    assert_equal(lagmat, lag_data)

    # no insert, should be the same
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True, insert=False)
    assert_equal(lagmat, lag_data)


def test_add_lag1d_struct():
    data = np.zeros(100, dtype=[('variable', float)])
    nddata = randn(100)
    data['variable'] = nddata

    lagmat = sm.tsa.lagmat(nddata, 3, trim='Both', original='in')
    lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=True)
    assert_equal(lagmat, np.array(lag_data.tolist()))

    lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=False)
    assert_equal(lagmat, np.array(lag_data.tolist()))

    lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
    assert_equal(lagmat, np.array(lag_data.tolist()))


def test_add_lag_1d_drop_struct():
    data = np.zeros(100, dtype=[('variable', float)])
    nddata = randn(100)
    data['variable'] = nddata

    lagmat = sm.tsa.lagmat(nddata, 3, trim='Both')
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True)
    assert_equal(lagmat, np.array(lag_data.tolist()))


def test_add_lag_drop_insert_struct():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :2], lagmat, nddata[3:, -1]))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, drop=True)
    assert_equal(np.array(lag_data.tolist()), results)


def test_add_lag_drop_noinsert_struct():
    data = macro_data
    nddata = data.view((float, 4))
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, np.array([0, 1, 3])], lagmat))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False, drop=True)
    assert_equal(np.array(lag_data.tolist()), results)


def test_add_lag1d_dataframe():
    data = np.zeros(100, dtype=[('variable', float)])
    nddata = randn(100)
    data['variable'] = nddata
    data = pd.DataFrame(data)
    lagmat = sm.tsa.lagmat(nddata, 3, trim='Both', original='in')
    lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=True)
    assert_equal(lagmat, lag_data.values)

    lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=False)
    assert_equal(lagmat, lag_data.values)

    lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
    assert_equal(lagmat, lag_data.values)


def test_add_lag_1d_drop_dataframe():
    data = np.zeros(100, dtype=[('variable', float)])
    nddata = randn(100)
    data['variable'] = nddata
    data = pd.DataFrame(data)

    lagmat = sm.tsa.lagmat(nddata, 3, trim='Both')
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True)
    assert_equal(lagmat, lag_data.values)


def test_add_lag_drop_insert_dataframe():
    data = macro_data
    nddata = data.view((float, 4))
    data = pd.DataFrame(data)
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :2], lagmat, nddata[3:, -1]))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, drop=True)
    assert_equal(lag_data.values, results)


def test_add_lag_drop_noinsert_dataframe():
    data = macro_data
    nddata = data.view((float, 4))
    data = pd.DataFrame(data)
    lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, np.array([0, 1, 3])], lagmat))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False, drop=True)
    assert_equal(lag_data.values, results)


def test_freq_to_period():
    from pandas.tseries.frequencies import to_offset

    freqs = ['A', 'AS-MAR', 'Q', 'QS', 'QS-APR', 'W', 'W-MON', 'B']
    expected = [1, 1, 4, 4, 4, 52, 52, 52]
    for i, j in zip(freqs, expected):
        assert_equal(tools.freq_to_period(i), j)
        assert_equal(tools.freq_to_period(to_offset(i)), j)


def test_add_lag_drop_insert_early():
    x = randn(5, 3)
    x_lags_incl = sm.tsa.add_lag(x, lags=2, insert=0, drop=False)
    x_lags_excl = sm.tsa.add_lag(x, lags=2, insert=0, drop=True)
    assert_equal(x_lags_incl[:, :2], x_lags_excl[:, :2])
    assert_equal(x_lags_incl[:, 3:], x_lags_excl[:, 2:])


def test_add_lag_negative_index():
    x = randn(5, 3)
    x_lags_incl = sm.tsa.add_lag(x, lags=2, insert=-1, drop=False)
    assert_equal(x_lags_incl[:, 0], x[2:, 0])
    assert_equal(x_lags_incl[:, 3], x[1:4, 0])
    assert_equal(x_lags_incl[:, 4], x[:3, 0])


def test_add_trend_prepend():
    n = 10
    x = randn(n, 1)
    trend_1 = sm.tsa.add_trend(x, trend='ct', prepend=True)
    trend_2 = sm.tsa.add_trend(x, trend='ct', prepend=False)
    assert_equal(trend_1[:, :2], trend_2[:, 1:])


def test_add_time_trend_dataframe():
    n = 10
    x = randn(n, 1)
    x = pd.DataFrame(x, columns=['col1'])
    trend_1 = sm.tsa.add_trend(x, trend='t')
    assert_array_almost_equal(np.asarray(trend_1['trend']),
                              np.arange(1.0, n + 1))


def test_add_trend_prepend_dataframe():
    # Skipped on pandas < 13.1 it seems
    skip_if_early_pandas()
    n = 10
    x = randn(n, 1)
    x = pd.DataFrame(x, columns=['col1'])
    trend_1 = sm.tsa.add_trend(x, trend='ct', prepend=True)
    trend_2 = sm.tsa.add_trend(x, trend='ct', prepend=False)
    assert_frame_equal(trend_1.iloc[:, :2], trend_2.iloc[:, 1:])


def test_add_trend_duplicate_name():
    x = pd.DataFrame(np.zeros((10, 1)), columns=['trend'])
    with warnings.catch_warnings(record=True) as w:
        assert_produces_warning(sm.tsa.add_trend(x, trend='ct'),
                                tools.ColumnNameConflict)
        y = sm.tsa.add_trend(x, trend='ct')
        # should produce a single warning
    np.testing.assert_equal(len(w), 1)
    assert 'const' in y.columns
    assert 'trend_0' in y.columns


def test_add_trend_c():
    x = np.zeros((10, 1))
    y = sm.tsa.add_trend(x, trend='c')
    assert np.all(y[:, 1] == 1.0)


def test_add_trend_ct():
    n = 20
    x = np.zeros((20, 1))
    y = sm.tsa.add_trend(x, trend='ct')
    assert np.all(y[:, 1] == 1.0)
    assert_equal(y[0, 2], 1.0)
    assert_array_almost_equal(np.diff(y[:, 2]), np.ones((n - 1)))


def test_add_trend_ctt():
    n = 10
    x = np.zeros((n, 1))
    y = sm.tsa.add_trend(x, trend='ctt')
    assert np.all(y[:, 1] == 1.0)
    assert y[0, 2] == 1.0
    assert_array_almost_equal(np.diff(y[:, 2]), np.ones((n - 1)))
    assert y[0, 3] == 1.0
    assert_array_almost_equal(np.diff(y[:, 3]), np.arange(3.0, 2.0 * n, 2.0))


def test_add_trend_t():
    n = 20
    x = np.zeros((20, 1))
    y = sm.tsa.add_trend(x, trend='t')
    assert y[0, 1] == 1.0
    assert_array_almost_equal(np.diff(y[:, 1]), np.ones((n - 1)))


def test_add_trend_no_input():
    n = 100
    y = sm.tsa.add_trend(x=None, trend='ct', nobs=n)
    assert np.all(y[:, 0] == 1.0)
    assert y[0, 1] == 1.0
    assert_array_almost_equal(np.diff(y[:, 1]), np.ones((n - 1)))


def test_reintegrate_1_diff():
    x = randn(10, 1)
    y = np.cumsum(x) + 1.0
    assert_array_almost_equal(y, reintegrate(np.diff(y), [y[0]]))


def test_reintegrate_2_diff():
    x = randn(10, 1)
    y = np.cumsum(x) + 1.0
    z = np.cumsum(y) + 1.0
    levels = [z[0], np.diff(z, 1)[0]]
    assert_array_almost_equal(z, reintegrate(np.diff(z, 2), levels))


def test_detrend_1d_order_0():
    x = randn(100)
    assert_array_almost_equal(x - x.mean(), sm.tsa.detrend(x, order=0))


def test_detrend_1d_order_1():
    n = 100
    x = randn(n)
    z = sm.tsa.add_trend(x=None, trend='ct', nobs=n)
    resid = OLS(x, z).fit().resid
    detrended = sm.tsa.detrend(x, order=1)
    assert_array_almost_equal(resid, detrended)


def test_detrend_2d_order_0():
    n = 100
    x = randn(n, 1)
    assert_array_almost_equal(x - x.mean(), sm.tsa.detrend(x, order=0))


def test_detrend_2d_order_2():
    n = 100
    x = randn(n, 1)
    z = sm.tsa.add_trend(x=None, trend='ctt', nobs=n)
    resid = OLS(x, z).fit().resid[:, None]
    detrended = sm.tsa.detrend(x, order=2)
    assert_array_almost_equal(resid, detrended)


def test_detrend_2d_order_2_axis_1():
    n = 100
    x = randn(1, n)
    z = sm.tsa.add_trend(x=None, trend='ctt', nobs=n)
    resid = OLS(x.T, z).fit().resid[None, :]
    detrended = sm.tsa.detrend(x, order=2, axis=1)
    assert_array_almost_equal(resid, detrended)

    pass


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[__file__, '-vvs'], exit=False)
