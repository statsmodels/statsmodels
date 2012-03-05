'''tests for some time series analysis functions

'''

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech

from results import savedrvs
from results.datamlw_tls import mlacf, mlccf, mlpacf, mlywar

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:]/1000.
x1000 = xo/1000.


def test_acf():
    acf_x = tsa.acf(x100, unbiased=False)[:21]
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8) #why only dec=8
    acf_x = tsa.acf(x1000, unbiased=False)[:21]
    assert_array_almost_equal(mlacf.acf1000.ravel(), acf_x, 8) #why only dec=9

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
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfols, 2)
    pacfols = tsa.pacf_ols(x1000, 20)
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfols, 5)
    #assert False

def test_ywcoef():
    assert_array_almost_equal(mlywar.arcoef100[1:],
                    -sm.regression.yule_walker(x100, 10, method='mle')[0], 8)
    assert_array_almost_equal(mlywar.arcoef1000[1:],
                    -sm.regression.yule_walker(x1000, 20, method='mle')[0], 8)

def test_duplication_matrix():
    for k in range(2, 10):
        m = tools.unvech(np.random.randn(k * (k + 1) / 2))
        Dk = tools.duplication_matrix(k)
        assert(np.array_equal(vec(m), np.dot(Dk, vech(m))))

def test_elimination_matrix():
    for k in range(2, 10):
        m = np.random.randn(k, k)
        Lk = tools.elimination_matrix(k)
        assert(np.array_equal(vech(m), np.dot(Lk, vec(m))))

def test_commutation_matrix():
    m = np.random.randn(4, 3)
    K = tools.commutation_matrix(4, 3)
    assert(np.array_equal(vec(m.T), np.dot(K, vec(m))))

def test_vec():
    arr = np.array([[1, 2],
                    [3, 4]])
    assert(np.array_equal(vec(arr), [1, 3, 2, 4]))

def test_vech():
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    assert(np.array_equal(vech(arr), [1, 4, 7, 5, 8, 9]))


def test_add_lag_insert():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,2],3,trim='Both')
    results = np.column_stack((nddata[3:,:3],lagmat,nddata[3:,-1]))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3)
    assert_equal(lag_data.view((float,len(lag_data.dtype.names))), results)

def test_add_lag_noinsert():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,2],3,trim='Both')
    results = np.column_stack((nddata[3:,:],lagmat))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False)
    assert_equal(lag_data.view((float,len(lag_data.dtype.names))), results)

def test_add_lag_noinsert_atend():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,-1],3,trim='Both')
    results = np.column_stack((nddata[3:,:],lagmat))
    lag_data = sm.tsa.add_lag(data, 'cpi', 3, insert=False)
    assert_equal(lag_data.view((float,len(lag_data.dtype.names))), results)
    # should be the same as insert
    lag_data2 = sm.tsa.add_lag(data, 'cpi', 3, insert=True)
    assert_equal(lag_data2.view((float,len(lag_data2.dtype.names))), results)

def test_add_lag_ndarray():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,2],3,trim='Both')
    results = np.column_stack((nddata[3:,:3],lagmat,nddata[3:,-1]))
    lag_data = sm.tsa.add_lag(nddata, 2, 3)
    assert_equal(lag_data, results)

def test_add_lag_noinsert_ndarray():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,2],3,trim='Both')
    results = np.column_stack((nddata[3:,:],lagmat))
    lag_data = sm.tsa.add_lag(nddata, 2, 3, insert=False)
    assert_equal(lag_data, results)

def test_add_lag_noinsertatend_ndarray():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,-1],3,trim='Both')
    results = np.column_stack((nddata[3:,:],lagmat))
    lag_data = sm.tsa.add_lag(nddata, 3, 3, insert=False)
    assert_equal(lag_data, results)
    # should be the same as insert also check negative col number
    lag_data2 = sm.tsa.add_lag(nddata, -1, 3, insert=True)
    assert_equal(lag_data2, results)

def test_add_lag1d():
    data = np.random.randn(100)
    lagmat = sm.tsa.lagmat(data,3,trim='Both')
    results = np.column_stack((data[3:],lagmat))
    lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
    assert_equal(results, lag_data)

    # add index
    data = data[:,None]
    lagmat = sm.tsa.lagmat(data,3,trim='Both') # test for lagmat too
    results = np.column_stack((data[3:],lagmat))
    lag_data = sm.tsa.add_lag(data,lags=3, insert=True)
    assert_equal(results, lag_data)

def test_add_lag1d_drop():
    data = np.random.randn(100)
    lagmat = sm.tsa.lagmat(data,3,trim='Both')
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True, insert=True)
    assert_equal(lagmat, lag_data)

    # no insert, should be the same
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True, insert=False)
    assert_equal(lagmat, lag_data)

def test_add_lag1d_struct():
    data = np.zeros(100, dtype=[('variable',float)])
    nddata = np.random.randn(100)
    data['variable'] = nddata

    lagmat = sm.tsa.lagmat(nddata,3,trim='Both', original='in')
    lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=True)
    assert_equal(lagmat, lag_data.view((float,4)))

    lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=False)
    assert_equal(lagmat, lag_data.view((float,4)))

    lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
    assert_equal(lagmat, lag_data.view((float,4)))

def test_add_lag_1d_drop_struct():
    data = np.zeros(100, dtype=[('variable',float)])
    nddata = np.random.randn(100)
    data['variable'] = nddata

    lagmat = sm.tsa.lagmat(nddata,3,trim='Both')
    lag_data = sm.tsa.add_lag(data, lags=3, drop=True)
    assert_equal(lagmat, lag_data.view((float,3)))

def test_add_lag_drop_insert():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,2],3,trim='Both')
    results = np.column_stack((nddata[3:,:2],lagmat,nddata[3:,-1]))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, drop=True)
    assert_equal(lag_data.view((float,len(lag_data.dtype.names))), results)

def test_add_lag_drop_noinsert():
    data = sm.datasets.macrodata.load().data[['year','quarter','realgdp','cpi']]
    nddata = data.view((float,4))
    lagmat = sm.tsa.lagmat(nddata[:,2],3,trim='Both')
    results = np.column_stack((nddata[3:,np.array([0,1,3])],lagmat))
    lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False, drop=True)
    assert_equal(lag_data.view((float,len(lag_data.dtype.names))), results)


if __name__ == '__main__':
    #running them directly
    # test_acf()
    # test_ccf()
    # test_pacf_yw()
    # test_pacf_ols()
    # test_ywcoef()

    import nose
    nose.runmodule()
