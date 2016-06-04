'''tests for some time series analysis functions

'''
import unittest
from nose.tools import assert_raises

from statsmodels.compat.python import zip

import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech

from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import mlacf, mlccf, mlpacf, \
    mlywar

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.
x1000 = xo / 1000.


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
    # assert False


def test_pacf_ols():
    pacfols = tsa.pacf_ols(x100, 20)
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfols, 8)
    pacfols = tsa.pacf_ols(x1000, 20)
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfols, 8)
    # assert False


def test_ywcoef():
    assert_array_almost_equal(mlywar.arcoef100[1:],
                              -
                              sm.regression.yule_walker(x100, 10, method='mle')[
                                  0], 8)
    assert_array_almost_equal(mlywar.arcoef1000[1:],
                              -sm.regression.yule_walker(x1000, 20,
                                                         method='mle')[0], 8)


def test_yule_walker_inter():
    # see 1869
    x = np.array([1, -1, 2, 2, 0, -2, 1, 0, -3, 0, 0])
    # it works
    result = sm.regression.yule_walker(x, 3)


def test_duplication_matrix():
    for k in range(2, 10):
        m = tools.unvech(np.random.randn(k * (k + 1) // 2))
        Dk = tools.duplication_matrix(k)
        assert (np.array_equal(vec(m), np.dot(Dk, vech(m))))


def test_elimination_matrix():
    for k in range(2, 10):
        m = np.random.randn(k, k)
        Lk = tools.elimination_matrix(k)
        assert (np.array_equal(vech(m), np.dot(Lk, vec(m))))


def test_commutation_matrix():
    m = np.random.randn(4, 3)
    K = tools.commutation_matrix(4, 3)
    assert (np.array_equal(vec(m.T), np.dot(K, vec(m))))


def test_vec():
    arr = np.array([[1, 2],
                    [3, 4]])
    assert (np.array_equal(vec(arr), [1, 3, 2, 4]))


def test_vech():
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    assert (np.array_equal(vech(arr), [1, 4, 7, 5, 8, 9]))


class TestLagmat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = sm.datasets.macrodata.load()
        cls.macro_data = data.data[['year', 'quarter', 'realgdp', 'cpi']]
        cls.random_data = np.random.randn(100)
        year = cls.macro_data['year']
        quarter = cls.macro_data['quarter']
        cls.macro_df = pd.DataFrame.from_records(cls.macro_data)
        index = [str(int(yr)) + '-Q' + str(int(qu))
                 for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter)]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    def test_add_lag_insert(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = sm.tsa.add_lag(data, 'realgdp', 3)
        assert_equal(lag_data.view((float, len(lag_data.dtype.names))), results)

    def test_add_lag_noinsert(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False)
        assert_equal(lag_data.view((float, len(lag_data.dtype.names))), results)

    def test_add_lag_noinsert_atend(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, -1], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = sm.tsa.add_lag(data, 'cpi', 3, insert=False)
        assert_equal(lag_data.view((float, len(lag_data.dtype.names))), results)
        # should be the same as insert
        lag_data2 = sm.tsa.add_lag(data, 'cpi', 3, insert=True)
        assert_equal(lag_data2.view((float, len(lag_data2.dtype.names))),
                     results)

    def test_add_lag_ndarray(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = sm.tsa.add_lag(nddata, 2, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_ndarray(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = sm.tsa.add_lag(nddata, 2, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsertatend_ndarray(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, -1], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = sm.tsa.add_lag(nddata, 3, 3, insert=False)
        assert_equal(lag_data, results)
        # should be the same as insert also check negative col number
        lag_data2 = sm.tsa.add_lag(nddata, -1, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_sep_return(self):
        data = self.random_data
        n = data.shape[0]
        lagmat, leads = sm.tsa.lagmat(data, 3, trim='none', original='sep')
        expected = np.zeros((n + 3, 4))
        for i in range(4):
            expected[i:i + n, i] = data
        expected_leads = expected[:, :1]
        expected_lags = expected[:, 1:]
        assert_equal(expected_lags, lagmat)
        assert_equal(expected_leads, leads)

    def test_add_lag1d(self):
        data = self.random_data
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

    def test_add_lag1d_drop(self):
        data = self.random_data
        lagmat = sm.tsa.lagmat(data, 3, trim='Both')
        lag_data = sm.tsa.add_lag(data, lags=3, drop=True, insert=True)
        assert_equal(lagmat, lag_data)

        # no insert, should be the same
        lag_data = sm.tsa.add_lag(data, lags=3, drop=True, insert=False)
        assert_equal(lagmat, lag_data)

    def test_add_lag1d_struct(self):
        data = np.zeros(100, dtype=[('variable', float)])
        nddata = self.random_data
        data['variable'] = nddata

        lagmat = sm.tsa.lagmat(nddata, 3, trim='Both', original='in')
        lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=True)
        assert_equal(lagmat, lag_data.view((float, 4)))

        lag_data = sm.tsa.add_lag(data, 'variable', lags=3, insert=False)
        assert_equal(lagmat, lag_data.view((float, 4)))

        lag_data = sm.tsa.add_lag(data, lags=3, insert=True)
        assert_equal(lagmat, lag_data.view((float, 4)))

    def test_add_lag_1d_drop_struct(self):
        data = np.zeros(100, dtype=[('variable', float)])
        nddata = self.random_data
        data['variable'] = nddata

        lagmat = sm.tsa.lagmat(nddata, 3, trim='Both')
        lag_data = sm.tsa.add_lag(data, lags=3, drop=True)
        assert_equal(lagmat, lag_data.view((float, 3)))

    def test_add_lag_drop_insert(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :2], lagmat, nddata[3:, -1]))
        lag_data = sm.tsa.add_lag(data, 'realgdp', 3, drop=True)
        assert_equal(lag_data.view((float, len(lag_data.dtype.names))), results)

    def test_add_lag_drop_noinsert(self):
        data = self.macro_data
        nddata = data.view((float, 4))
        lagmat = sm.tsa.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, np.array([0, 1, 3])], lagmat))
        lag_data = sm.tsa.add_lag(data, 'realgdp', 3, insert=False, drop=True)
        assert_equal(lag_data.view((float, len(lag_data.dtype.names))), results)

    def test_dataframe_without_pandas(self):
        data = self.macro_df
        both = sm.tsa.lagmat(data, 3, trim='both', original='in')
        both_np = sm.tsa.lagmat(data.values, 3, trim='both', original='in')
        assert_equal(both, both_np)

        lags = sm.tsa.lagmat(data, 3, trim='none', original='ex')
        lags_np = sm.tsa.lagmat(data.values, 3, trim='none', original='ex')
        assert_equal(lags, lags_np)

        lags, lead = sm.tsa.lagmat(data, 3, trim='forward', original='sep')
        lags_np, lead_np = sm.tsa.lagmat(data.values, 3, trim='forward', original='sep')
        assert_equal(lags, lags_np)
        assert_equal(lead, lead_np)

    def test_dataframe_both(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3,16))
        values[:n,:4] = data.values
        for lag in range(1,4):
            new_cols = [col + '.L.' + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag:n+lag,4*lag:4*(lag+1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values,columns=columns, index=index)
        expected = expected.iloc[3:]

        both = sm.tsa.lagmat(self.macro_df, 3, trim='both', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = sm.tsa.lagmat(self.macro_df, 3, trim='both', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        lags, lead = sm.tsa.lagmat(self.macro_df, 3, trim='both',
                                   original='sep', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_too_few_observations(self):
        assert_raises(ValueError, sm.tsa.lagmat, self.macro_df, 300, use_pandas=True)
        assert_raises(ValueError, sm.tsa.lagmat, self.macro_data, 300)

    def test_unknown_trim(self):
        assert_raises(ValueError, sm.tsa.lagmat, self.macro_df, 3,
                      trim='unknown', use_pandas=True)
        assert_raises(ValueError, sm.tsa.lagmat, self.macro_data, 3,
                      trim='unknown')

    def test_dataframe_forward(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3,16))
        values[:n,:4] = data.values
        for lag in range(1,4):
            new_cols = [col + '.L.' + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag:n+lag,4*lag:4*(lag+1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values,columns=columns, index=index)
        both = sm.tsa.lagmat(self.macro_df, 3, trim='forward', original='in',
                             use_pandas=True)
        assert_frame_equal(both, expected)
        lags = sm.tsa.lagmat(self.macro_df, 3, trim='forward', original='ex',
                             use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        lags, lead = sm.tsa.lagmat(self.macro_df, 3, trim='forward',
                                   original='sep', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_pandas_errors(self):
        assert_raises(ValueError, sm.tsa.lagmat, self.macro_df, 3, trim='none', use_pandas=True)
        assert_raises(ValueError, sm.tsa.lagmat, self.macro_df, 3,
                      trim='backward', use_pandas=True)
        assert_raises(ValueError, sm.tsa.lagmat, self.series, 3, trim='none', use_pandas=True)
        assert_raises(ValueError, sm.tsa.lagmat, self.series, 3,
                      trim='backward', use_pandas=True)

    def test_series_forward(self):
        expected = pd.DataFrame(index=self.series.index,
                                columns=['cpi', 'cpi.L.1', 'cpi.L.2',
                                         'cpi.L.3'])
        expected['cpi'] = self.series
        for lag in range(1, 4):
            expected['cpi.L.' + str(int(lag))] = self.series.shift(lag)
        expected = expected.fillna(0.0)

        both = sm.tsa.lagmat(self.series, 3, trim='forward', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = sm.tsa.lagmat(self.series, 3, trim='forward', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 1:])
        lags, lead = sm.tsa.lagmat(self.series, 3, trim='forward',
                                   original='sep', use_pandas=True)
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_series_both(self):
        expected = pd.DataFrame(index=self.series.index,
                                columns=['cpi', 'cpi.L.1', 'cpi.L.2',
                                         'cpi.L.3'])
        expected['cpi'] = self.series
        for lag in range(1, 4):
            expected['cpi.L.' + str(int(lag))] = self.series.shift(lag)
        expected = expected.iloc[3:]

        both = sm.tsa.lagmat(self.series, 3, trim='both', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = sm.tsa.lagmat(self.series, 3, trim='both', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 1:])
        lags, lead = sm.tsa.lagmat(self.series, 3, trim='both', original='sep', use_pandas=True)
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])


def test_freq_to_period():
    from pandas.tseries.frequencies import to_offset
    freqs = ['A', 'AS-MAR', 'Q', 'QS', 'QS-APR', 'W', 'W-MON', 'B', 'D', 'H']
    expected = [1, 1, 4, 4, 4, 52, 52, 5, 7, 24]
    for i, j in zip(freqs, expected):
        assert_equal(tools.freq_to_period(i), j)
        assert_equal(tools.freq_to_period(to_offset(i)), j)


def test_detrend():
    data = np.arange(5)
    assert_array_almost_equal(sm.tsa.detrend(data, order=1),
                              np.zeros_like(data))
    assert_array_almost_equal(sm.tsa.detrend(data, order=0), [-2, -1, 0, 1, 2])
    data = np.arange(10).reshape(5, 2)
    assert_array_almost_equal(sm.tsa.detrend(data, order=1, axis=0),
                              np.zeros_like(data))
    assert_array_almost_equal(sm.tsa.detrend(data, order=0, axis=0),
                              [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]])
    assert_array_almost_equal(sm.tsa.detrend(data, order=0, axis=1),
                              [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5],
                               [-0.5, 0.5], [-0.5, 0.5]])


if __name__ == '__main__':
    import nose

    nose.runmodule()
