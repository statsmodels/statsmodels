"""tests for some time series analysis functions"""

from statsmodels.compat.pandas import (
    PD_LT_2_2_0,
    QUARTER_END,
    YEAR_END,
    assert_frame_equal,
    assert_series_equal,
)

import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest

from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa import stattools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
    mlacf,
    mlccf,
    mlpacf,
    mlywar,
)
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import LagmatResult, vec, vech

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.0
x1000 = xo / 1000.0


def test_acf():
    acf_x = stattools.acf(x100, adjusted=False, fft=False, nlags=20)
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8)  # why only dec=8
    acf_x = stattools.acf(x1000, adjusted=False, fft=False, nlags=20)
    assert_array_almost_equal(mlacf.acf1000.ravel(), acf_x, 8)  # why only dec=9


def test_ccf():
    ccf_x = stattools.ccf(x100[4:], x100[:-4], adjusted=False)[:21]
    assert_array_almost_equal(mlccf.ccf100.ravel()[:21][::-1], ccf_x, 8)
    ccf_x = stattools.ccf(x1000[4:], x1000[:-4], adjusted=False)[:21]
    assert_array_almost_equal(mlccf.ccf1000.ravel()[:21][::-1], ccf_x, 8)


def test_pacf_yw():
    pacfyw = stattools.pacf_yw(x100, 20, method="mle")
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfyw, 1)
    pacfyw = stattools.pacf_yw(x1000, 20, method="mle")
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfyw, 2)
    # assert False


def test_pacf_ols():
    pacfols = stattools.pacf_ols(x100, 20)
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfols, 8)
    pacfols = stattools.pacf_ols(x1000, 20)
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfols, 8)
    # assert False


def test_ywcoef():
    assert_array_almost_equal(
        mlywar.arcoef100[1:],
        -regression.yule_walker(x100, 10, method="mle", result_object=False)[0],
        8,
    )
    assert_array_almost_equal(
        mlywar.arcoef1000[1:],
        -regression.yule_walker(x1000, 20, method="mle", result_object=False)[0],
        8,
    )


@pytest.mark.smoke
def test_yule_walker_inter():
    # see 1869
    x = np.array([1, -1, 2, 2, 0, -2, 1, 0, -3, 0, 0])
    # it works
    regression.yule_walker(x, 3, result_object=False)


def test_duplication_matrix():
    rs = np.random.RandomState(664831)
    for k in range(2, 10):
        m = tools.unvech(rs.randn(k * (k + 1) // 2))
        Dk = tools.duplication_matrix(k)
        assert np.array_equal(vec(m), np.dot(Dk, vech(m)))


def test_elimination_matrix():
    rs = np.random.RandomState(664832)
    for k in range(2, 10):
        m = rs.randn(k, k)
        Lk = tools.elimination_matrix(k)
        assert np.array_equal(vech(m), np.dot(Lk, vec(m)))


def test_commutation_matrix():
    rs = np.random.RandomState(664833)
    m = rs.randn(4, 3)
    K = tools.commutation_matrix(4, 3)
    assert np.array_equal(vec(m.T), np.dot(K, vec(m)))


def test_vec():
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(vec(arr), [1, 3, 2, 4])


def test_vech():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.array_equal(vech(arr), [1, 4, 7, 5, 8, 9])


class TestLagmat:
    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.macro_df = data.data[["year", "quarter", "realgdp", "cpi"]]
        cols = list(cls.macro_df.columns)
        cls.realgdp_loc = cols.index("realgdp")
        cls.cpi_loc = cols.index("cpi")
        rs = np.random.RandomState(12345)
        cls.random_data = rs.randn(100)

        index = [
            str(int(yr)) + "-Q" + str(int(qu))
            for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter, strict=True)
        ]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    def test_add_lag_insert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_atend(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, -1], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(data, self.cpi_loc, 3, insert=False)
        assert_equal(lag_data, results)
        # should be the same as insert
        lag_data2 = tools.add_lag(data, self.cpi_loc, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_add_lag_pandas_str_col(self):
        # the docstring example passes a column label; it used to crash with
        # "'<' not supported between instances of 'str' and 'int'"
        lag_data = tools.add_lag(self.macro_df, "realgdp", 3)
        expected = tools.add_lag(self.macro_df, self.realgdp_loc, 3)
        assert_equal(np.asarray(lag_data), np.asarray(expected))

        lag_data = tools.add_lag(self.macro_df, "realgdp", 3, drop=True)
        expected = tools.add_lag(self.macro_df, self.realgdp_loc, 3, drop=True)
        assert_equal(np.asarray(lag_data), np.asarray(expected))

    def test_add_lag_str_col_errors(self):
        with pytest.raises(KeyError, match="nope"):
            tools.add_lag(self.macro_df, "nope", 3)
        with pytest.raises(TypeError, match="pandas"):
            tools.add_lag(np.ones((5, 2)), "realgdp", 3)

    def test_add_lag_pandas_nonstr_label(self):
        # any non-int-like col is resolved as a label, not only strings
        df = self.macro_df.rename(columns={"realgdp": ("realgdp", "level")})
        label = ("realgdp", "level")
        loc = list(df.columns).index(label)

        lag_data = tools.add_lag(df, label, 3)
        expected = tools.add_lag(df, loc, 3)
        assert_equal(np.asarray(lag_data), np.asarray(expected))

    def test_add_lag_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(nddata, 2, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(nddata, 2, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsertatend_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, -1], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(nddata, 3, 3, insert=False)
        assert_equal(lag_data, results)
        # should be the same as insert also check negative col number
        lag_data2 = tools.add_lag(nddata, -1, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_sep_return(self):
        data = self.random_data
        n = data.shape[0]
        _result = stattools.lagmat(
            data, 3, trim="none", original="sep", result_object=False
        )
        lagmat, leads = _result.lags, _result.leads
        expected = np.zeros((n + 3, 4))
        for i in range(4):
            expected[i : i + n, i] = data
        expected_leads = expected[:, :1]
        expected_lags = expected[:, 1:]
        assert_equal(expected_lags, lagmat)
        assert_equal(expected_leads, leads)

    def test_sep_return_default_is_result_object(self):
        # LagmatResult is always used for original="sep".
        data = self.random_data
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = stattools.lagmat(data, 3, trim="none", original="sep")
            # The result object is used whenever it matches the legacy
            # tuple's contents, so result_object=False cannot opt out of
            # it here.
            opted_out = stattools.lagmat(
                data, 3, trim="none", original="sep", result_object=False
            )
            # Unpacking is a stable, non-deprecated part of the API.
            lags, leads = res
        assert isinstance(res, LagmatResult)
        assert isinstance(opted_out, LagmatResult)
        assert len(res) == 2
        assert isinstance(lags, np.ndarray)
        assert_array_almost_equal(lags, res.lags)
        assert_array_almost_equal(leads, res.leads)

    def test_sep_return_result_object_true(self):
        data = self.random_data
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = stattools.lagmat(
                data, 3, trim="none", original="sep", result_object=True
            )
            assert res[0] is res.lags
            assert res[1] is res.leads
        assert isinstance(res, LagmatResult)

    def test_non_sep_original_never_warns(self):
        # Only original="sep" returns more than one value, so the other
        # values must stay silent.  An explicit result_object=True still
        # returns the result object, with leads left as None because the
        # original was either excluded or folded into lags.
        data = self.random_data
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lags_ex = stattools.lagmat(data, 3, trim="none", original="ex")
            lags_in = stattools.lagmat(data, 3, trim="none", original="in")
            nt_ex = stattools.lagmat(
                data, 3, trim="none", original="ex", result_object=True
            )
            nt_in = stattools.lagmat(
                data, 3, trim="none", original="in", result_object=True
            )
        assert isinstance(lags_ex, np.ndarray)
        assert isinstance(lags_in, np.ndarray)
        for nt, expected in ((nt_ex, lags_ex), (nt_in, lags_in)):
            assert isinstance(nt, LagmatResult)
            assert_array_almost_equal(nt.lags, expected)
            assert nt.leads is None

    def test_add_lag1d(self):
        data = self.random_data
        lagmat = stattools.lagmat(data, 3, trim="Both")
        results = np.column_stack((data[3:], lagmat))
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(results, lag_data)

        # add index
        data = data[:, None]
        lagmat = stattools.lagmat(data, 3, trim="Both")  # test for lagmat too
        results = np.column_stack((data[3:], lagmat))
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(results, lag_data)

    def test_add_lag1d_drop(self):
        data = self.random_data
        lagmat = stattools.lagmat(data, 3, trim="Both")
        lag_data = tools.add_lag(data, lags=3, drop=True, insert=True)
        assert_equal(lagmat, lag_data)

        # no insert, should be the same
        lag_data = tools.add_lag(data, lags=3, drop=True, insert=False)
        assert_equal(lagmat, lag_data)

    def test_add_lag1d_struct(self):
        data = np.zeros(100, dtype=[("variable", float)])
        nddata = self.random_data
        data["variable"] = nddata

        lagmat = stattools.lagmat(nddata, 3, trim="Both", original="in")
        lag_data = tools.add_lag(data, 0, lags=3, insert=True)
        assert_equal(lagmat, lag_data)

        lag_data = tools.add_lag(data, 0, lags=3, insert=False)
        assert_equal(lagmat, lag_data)

        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(lagmat, lag_data)

    def test_add_lag_1d_drop_struct(self):
        data = np.zeros(100, dtype=[("variable", float)])
        nddata = self.random_data
        data["variable"] = nddata

        lagmat = stattools.lagmat(nddata, 3, trim="Both")
        lag_data = tools.add_lag(data, lags=3, drop=True)
        assert_equal(lagmat, lag_data)

    def test_add_lag_drop_insert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :2], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, drop=True)
        assert_equal(lag_data, results)

    def test_add_lag_drop_noinsert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, np.array([0, 1, 3])], lagmat))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, insert=False, drop=True)
        assert_equal(lag_data, results)

    def test_dataframe_without_pandas(self):
        data = self.macro_df
        both = stattools.lagmat(data, 3, trim="both", original="in")
        both_np = stattools.lagmat(data.values, 3, trim="both", original="in")
        assert_equal(both, both_np)

        lags = stattools.lagmat(data, 3, trim="none", original="ex")
        lags_np = stattools.lagmat(data.values, 3, trim="none", original="ex")
        assert_equal(lags, lags_np)

        _result = stattools.lagmat(
            data, 3, trim="forward", original="sep", result_object=False
        )
        lags, lead = _result.lags, _result.leads
        _result_np = stattools.lagmat(
            data.values, 3, trim="forward", original="sep", result_object=False
        )
        lags_np, lead_np = _result_np.lags, _result_np.leads
        assert_equal(lags, lags_np)
        assert_equal(lead, lead_np)

    def test_dataframe_both(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3, 16))
        values[:n, :4] = data.values
        for lag in range(1, 4):
            new_cols = [col + ".L." + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag : n + lag, 4 * lag : 4 * (lag + 1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values, columns=columns, index=index)
        expected = expected.iloc[3:]

        both = stattools.lagmat(
            self.macro_df, 3, trim="both", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.macro_df, 3, trim="both", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 4:])
        _result = stattools.lagmat(
            self.macro_df,
            3,
            trim="both",
            original="sep",
            use_pandas=True,
            result_object=False,
        )
        lags, lead = _result.lags, _result.leads
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_too_few_observations(self):
        with pytest.raises(ValueError):
            stattools.lagmat(self.macro_df, 300, use_pandas=True)
        with pytest.raises(ValueError):
            stattools.lagmat(self.macro_df.values, 300)

    def test_unknown_trim(self):
        with pytest.raises(ValueError):
            stattools.lagmat(
                self.macro_df,
                3,
                trim="unknown",
                use_pandas=True,
            )
        with pytest.raises(ValueError):
            stattools.lagmat(
                self.macro_df.values,
                3,
                trim="unknown",
            )

    def test_dataframe_forward(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3, 16))
        values[:n, :4] = data.values
        for lag in range(1, 4):
            new_cols = [col + ".L." + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag : n + lag, 4 * lag : 4 * (lag + 1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values, columns=columns, index=index)
        both = stattools.lagmat(
            self.macro_df, 3, trim="forward", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.macro_df, 3, trim="forward", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 4:])
        _result = stattools.lagmat(
            self.macro_df,
            3,
            trim="forward",
            original="sep",
            use_pandas=True,
            result_object=False,
        )
        lags, lead = _result.lags, _result.leads
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_pandas_errors(self):
        with pytest.raises(ValueError):
            stattools.lagmat(
                self.macro_df,
                3,
                trim="none",
                use_pandas=True,
            )
        with pytest.raises(ValueError):
            stattools.lagmat(
                self.macro_df,
                3,
                trim="backward",
                use_pandas=True,
            )
        with pytest.raises(ValueError):
            stattools.lagmat(
                self.series,
                3,
                trim="none",
                use_pandas=True,
            )
        with pytest.raises(ValueError):
            stattools.lagmat(
                self.series,
                3,
                trim="backward",
                use_pandas=True,
            )

    def test_series_forward(self):
        expected = pd.DataFrame(
            index=self.series.index,
            columns=["cpi", "cpi.L.1", "cpi.L.2", "cpi.L.3"],
        )
        expected["cpi"] = self.series
        for lag in range(1, 4):
            expected["cpi.L." + str(int(lag))] = self.series.shift(lag)
        expected = expected.fillna(0.0)

        both = stattools.lagmat(
            self.series, 3, trim="forward", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.series, 3, trim="forward", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 1:])
        _result = stattools.lagmat(
            self.series,
            3,
            trim="forward",
            original="sep",
            use_pandas=True,
            result_object=False,
        )
        lags, lead = _result.lags, _result.leads
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_series_both(self):
        expected = pd.DataFrame(
            index=self.series.index,
            columns=["cpi", "cpi.L.1", "cpi.L.2", "cpi.L.3"],
        )
        expected["cpi"] = self.series
        for lag in range(1, 4):
            expected["cpi.L." + str(int(lag))] = self.series.shift(lag)
        expected = expected.iloc[3:]

        both = stattools.lagmat(
            self.series, 3, trim="both", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.series, 3, trim="both", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 1:])
        _result = stattools.lagmat(
            self.series,
            3,
            trim="both",
            original="sep",
            use_pandas=True,
            result_object=False,
        )
        lags, lead = _result.lags, _result.leads
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_range_index_columns(self):
        # GH 8377
        df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
        df.columns = pd.RangeIndex(2)
        result = stattools.lagmat(df, maxlag=2, use_pandas=True)
        assert result.shape == (100, 4)
        assert list(result.columns) == ["0.L.1", "1.L.1", "0.L.2", "1.L.2"]

    def test_duplicate_column_names(self):
        # GH 8377
        df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
        df.columns = [0, "0"]
        with pytest.raises(ValueError, match="Columns names must be"):
            stattools.lagmat(df, maxlag=2, use_pandas=True)

    def test_lagmat_array_equiv(self):
        x = np.arange(100, dtype=float)
        scalar = stattools.lagmat(x, 5)
        array = stattools.lagmat(x, [1, 2, 3, 4, 5])
        assert_equal(scalar, array)

    def test_lagmat_array_exception(self):
        with pytest.raises(ValueError, match="All values in `maxlag` must be >=  0"):
            stattools.lagmat(np.arange(100, dtype=float), [-1])
        with pytest.raises(ValueError, match="`maxlag` must contain unique values"):
            stattools.lagmat(np.arange(100, dtype=float), [1, 2, 3, 3])

    def test_lagmat_array_dropped(self):
        x = np.arange(100, dtype=float)
        scalar = stattools.lagmat(x, 5)
        array = stattools.lagmat(x, [1, 3, 5])
        assert array.shape[0] == (scalar.shape[0])
        assert array.shape[1] == 3
        assert array.shape[1] < scalar.shape[1]
        assert_equal(scalar[:, [0, 2, 4]], array)

    @pytest.mark.parametrize("original", ["in", "ex", "sep"])
    def test_lagmat_0(self, original):
        x = np.arange(100, dtype=float)
        scalar = stattools.lagmat(x, 0, original=original, result_object=False)
        array = stattools.lagmat(x, [0], original=original, result_object=False)
        if original == "sep":
            scalar, x_scalar = scalar.lags, scalar.leads
            array, x_array = array.lags, array.leads
            assert x_scalar.shape == x_array.shape
            assert x_scalar.shape == (x.shape[0], 1)
        assert array.shape == scalar.shape
        expected_shape = (x.shape[0], 1) if original == "in" else (x.shape[0], 0)
        assert array.shape == expected_shape


ANNUAL = "A" if PD_LT_2_2_0 else YEAR_END
freqs = [
    YEAR_END,
    f"{ANNUAL}-MAR",
    QUARTER_END,
    "QS",
    "QS-APR",
    "W",
    "W-MON",
    "B",
    "D",
    "h",
    "2h",
    "min",
    "5min",
    "30s",
]
expected = [1, 1, 4, 4, 4, 52, 52, 5, 7, 24, 12, 1440, 288, 2880]
freq_expected = [(f, e) for f, e in zip(freqs, expected, strict=True)]


@pytest.mark.parametrize("freq_expected", freq_expected)
def test_freq_to_period(freq_expected):
    freq, expected = freq_expected
    assert_equal(tools.freq_to_period(freq), expected)
    assert_equal(tools.freq_to_period(to_offset(freq)), expected)


BUSINESS_MONTH_END = "BM" if PD_LT_2_2_0 else "BME"
BUSINESS_QUARTER_END = "BQ" if PD_LT_2_2_0 else "BQE"
BUSINESS_YEAR_END = "BA" if PD_LT_2_2_0 else "BYE"
CUSTOM_BUSINESS_MONTH_END = "CBM" if PD_LT_2_2_0 else "CBME"

business_freqs = [
    BUSINESS_MONTH_END,
    "BMS",
    f"2{BUSINESS_MONTH_END}",
    CUSTOM_BUSINESS_MONTH_END,
    "CBMS",
    BUSINESS_QUARTER_END,
    f"{BUSINESS_QUARTER_END}-APR",
    "BQS",
    BUSINESS_YEAR_END,
    "BYS",
]
business_expected = [12, 12, 12, 12, 12, 4, 4, 4, 1, 1]
business_freq_expected = [
    (f, e) for f, e in zip(business_freqs, business_expected, strict=True)
]


@pytest.mark.parametrize("freq_expected", business_freq_expected)
def test_freq_to_period_business(freq_expected):
    # Business month/quarter/year frequencies have the same seasonal
    # period as their calendar counterparts (GH 10525)
    freq, expected = freq_expected
    assert_equal(tools.freq_to_period(freq), expected)
    assert_equal(tools.freq_to_period(to_offset(freq)), expected)


def test_freq_to_period_business_seasonal_decompose():
    # A business month-end series is decomposed with period 12 without
    # an explicit period (GH 10525)
    idx = pd.date_range("2000-01-31", periods=48, freq=BUSINESS_MONTH_END)
    y = pd.Series(
        np.sin(np.arange(48) * 2 * np.pi / 12) + np.arange(48) / 10,
        index=idx,
    )
    res = seasonal_decompose(y)
    assert_allclose(
        res.seasonal.iloc[:12].to_numpy(),
        res.seasonal.iloc[12:24].to_numpy(),
        rtol=1e-10,
    )


def test_freq_to_period_nondivisible_multiplier():
    # A multiplier that does not divide a day evenly has no integer
    # observations-per-day period (GH 9619)
    with pytest.raises(ValueError, match="not understood"):
        tools.freq_to_period("7min")


@pytest.mark.parametrize("freq", ["0h", pd.offsets.Hour(-1)])
def test_freq_to_period_nonpositive_multiplier(freq):
    # A non-positive multiplier has no valid observations-per-day period
    # (GH 9619)
    with pytest.raises(ValueError, match="not understood"):
        tools.freq_to_period(freq)


@pytest.mark.parametrize("freq", ["ns", "us"])
def test_freq_to_period_unsupported_offset(freq):
    # Valid pandas offsets with rule codes "NS"/"US" have no
    # observations-per-day mapping and fall through to ValueError (GH 9619)
    with pytest.raises(ValueError, match="not understood"):
        tools.freq_to_period(freq)


class TestDetrend:
    @classmethod
    def setup_class(cls):
        cls.data_1d = np.arange(5.0)
        cls.data_2d = np.arange(10.0).reshape(5, 2)

    def test_detrend_1d(self):
        data = self.data_1d
        assert_array_almost_equal(tools.detrend(data, order=1), np.zeros_like(data))
        assert_array_almost_equal(tools.detrend(data, order=0), [-2, -1, 0, 1, 2])

    def test_detrend_negative_order(self):
        # a negative order used to build an empty vander matrix and silently
        # return the input unchanged
        with pytest.raises(ValueError, match="non-negative"):
            tools.detrend(self.data_1d, order=-1)

    def test_detrend_2d(self):
        data = self.data_2d
        assert_array_almost_equal(
            tools.detrend(data, order=1, axis=0), np.zeros_like(data)
        )
        assert_array_almost_equal(
            tools.detrend(data, order=0, axis=0),
            [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]],
        )
        assert_array_almost_equal(
            tools.detrend(data, order=0, axis=1),
            [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        )

    def test_detrend_series(self):
        data = pd.Series(self.data_1d, name="one")
        detrended = tools.detrend(data, order=1)
        assert_array_almost_equal(detrended.values, np.zeros_like(data))
        assert_series_equal(detrended, pd.Series(detrended.values, name="one"))
        detrended = tools.detrend(data, order=0)
        assert_array_almost_equal(detrended.values, pd.Series([-2, -1, 0, 1, 2]))
        assert_series_equal(detrended, pd.Series(detrended.values, name="one"))

    def test_detrend_dataframe(self):
        columns = ["one", "two"]
        index = list("abcde")
        data = pd.DataFrame(self.data_2d, columns=columns, index=index)

        detrended = tools.detrend(data, order=1, axis=0)
        assert_array_almost_equal(detrended.values, np.zeros_like(data))
        assert_frame_equal(
            detrended,
            pd.DataFrame(detrended.values, columns=columns, index=index),
        )

        detrended = tools.detrend(data, order=0, axis=0)
        assert_array_almost_equal(
            detrended.values, [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]]
        )
        assert_frame_equal(
            detrended,
            pd.DataFrame(detrended.values, columns=columns, index=index),
        )

        detrended = tools.detrend(data, order=0, axis=1)
        assert_array_almost_equal(
            detrended.values,
            [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        )
        assert_frame_equal(
            detrended,
            pd.DataFrame(detrended.values, columns=columns, index=index),
        )

    def test_detrend_dim_too_large(self):
        with pytest.raises(NotImplementedError):
            tools.detrend(np.ones((3, 3, 3)))


def _vec_ref(a):
    # The columns of a, stacked on top of one another
    return np.concatenate([a[:, j] for j in range(a.shape[1])])


def _vech_ref(a):
    # The lower triangle of a, column by column
    return np.concatenate([a[j:, j] for j in range(a.shape[0])])


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_vec_vech_identities(n):
    # Positive integer values, so that the products with the 0/1 matrices
    # below are exact
    rng = np.random.default_rng(20261119 + n)
    a = rng.integers(1, 10, size=(n, n)).astype(float)
    s = a + a.T
    nh = n * (n + 1) // 2
    assert_allclose(vec(a), _vec_ref(a), rtol=1e-15)
    assert_allclose(vech(a), _vech_ref(a), rtol=1e-15)
    assert_allclose(tools.unvec(_vec_ref(a)), a, rtol=1e-15)
    assert_allclose(tools.unvech(_vech_ref(s)), s, rtol=1e-15)

    dup = tools.duplication_matrix(n)
    elim = tools.elimination_matrix(n)
    assert dup.shape == (n * n, nh)
    assert elim.shape == (nh, n * n)
    # D vech(S) = vec(S) for a symmetric S, and L vec(A) = vech(A) for any A
    assert_allclose(dup @ _vech_ref(s), _vec_ref(s), rtol=1e-15)
    assert_allclose(elim @ _vec_ref(a), _vech_ref(a), rtol=1e-15)
    # L D = I, and the Moore-Penrose inverse of D maps vec(S) to vech(S)
    assert_allclose(elim @ dup, np.eye(nh), rtol=1e-15)
    assert_allclose(np.linalg.pinv(dup) @ _vec_ref(s), _vech_ref(s), rtol=1e-12)


@pytest.mark.parametrize("p, q", [(1, 1), (1, 3), (2, 3), (3, 2), (4, 4)])
def test_commutation_matrix_identities(p, q):
    rng = np.random.default_rng(20261119 + 10 * p + q)
    a = rng.integers(1, 10, size=(p, q)).astype(float)
    k = tools.commutation_matrix(p, q)
    # K vec(A) = vec(A')
    assert_allclose(k @ _vec_ref(a), _vec_ref(a.T), rtol=1e-15)
    # A permutation matrix, with K_{q,p} K_{p,q} = I
    assert_allclose(k.sum(0), np.ones(p * q), rtol=1e-15)
    assert_allclose(k.sum(1), np.ones(p * q), rtol=1e-15)
    assert_allclose(tools.commutation_matrix(q, p) @ k, np.eye(p * q), rtol=1e-15)
    # K_{r,p} (A kron B) K_{q,s} = B kron A for A (p, q) and B (r, s), see
    # Magnus and Neudecker (2019), Matrix Differential Calculus, Ch. 2
    b = rng.integers(1, 10, size=(2, 3)).astype(float)
    k_rp = tools.commutation_matrix(2, p)
    k_qs = tools.commutation_matrix(q, 3)
    assert_allclose(k_rp @ np.kron(a, b) @ k_qs, np.kron(b, a), rtol=1e-15)


@pytest.mark.parametrize("form", [list, tuple, pd.Series])
def test_unvech_input_forms(form):
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert_allclose(tools.unvech(form(v)), expected, rtol=1e-15)


def test_add_trend_recarray():
    x = np.rec.fromarrays([np.arange(5.0)], names="a")
    with pytest.raises(NotImplementedError):
        tools.add_trend(x, trend="c")


@pytest.mark.parametrize("label", [["a", "b", "c", "d"], ["a", "a", "a", "a"]])
def test_add_trend_non_numeric_column(label):
    # The range of a column of strings cannot be computed, so it is not taken
    # for a constant, even if all of its values are the same
    df = pd.DataFrame({"label": label, "x": [1.0, 2.0, 4.0, 8.0]})
    res = tools.add_trend(df, trend="ct", has_constant="raise")
    assert list(res.columns) == ["label", "x", "const", "trend"]
    assert list(res["label"]) == label
    assert_allclose(res["const"], np.ones(4), rtol=1e-15)
    assert_allclose(res["trend"], np.arange(1.0, 5.0), rtol=1e-15)


@pytest.mark.parametrize("form", SERIES_FORMS)
@pytest.mark.parametrize("trend", ["c", "t", "ct", "ctt"])
def test_add_trend_input_forms(form, trend):
    x = np.random.default_rng(3).standard_normal(50)
    value = series_forms(x)[form]
    res = tools.add_trend(value, trend=trend)
    t = np.arange(1.0, 51.0)
    columns = {"c": [t**0], "t": [t], "ct": [t**0, t], "ctt": [t**0, t, t**2]}
    expected = np.column_stack([x] + columns[trend])
    assert_allclose(np.asarray(res), expected, rtol=1e-15)
    if form in ("series", "dataframe"):
        assert isinstance(res, pd.DataFrame)
        pd.testing.assert_index_equal(res.index, value.index)
    else:
        assert isinstance(res, np.ndarray)


def _lags_of(x, col, lags):
    # Columns t-1, ..., t-lags of x[:, col], for t = lags, ..., nobs - 1
    nobs = x.shape[0]
    return np.column_stack([x[lags - j : nobs - j, col] for j in range(1, lags + 1)])


def test_add_lag_label_not_unique():
    df = pd.DataFrame(np.ones((10, 3)), columns=["a", "b", "a"])
    with pytest.raises(ValueError, match="does not uniquely identify"):
        tools.add_lag(df, "a", lags=1)


@pytest.mark.parametrize("insert, loc", [(0, 0), (2, 2), (3, 3), (-1, 3), (-3, 1)])
def test_add_lag_insert_position(insert, loc):
    # An int inserts the lags before that column, counting from the end if
    # it is negative
    x = np.random.default_rng(1).standard_normal((20, 3))
    expected = np.column_stack([x[2:, :loc], _lags_of(x, 1, 2), x[2:, loc:]])
    res = tools.add_lag(x, 1, lags=2, insert=insert)
    assert_allclose(res, expected, rtol=1e-15)


def test_add_lag_insert_past_last_column():
    x = np.random.default_rng(1).standard_normal((20, 3))
    expected = np.column_stack([x[2:], _lags_of(x, 1, 2)])
    with pytest.warns(ValueWarning, match="inserting at the last position"):
        res = tools.add_lag(x, 1, lags=2, insert=5)
    assert_allclose(res, expected, rtol=1e-15)


def test_add_lag_drop_column_after_lags():
    # With the lags inserted before col, col is dropped from the columns
    # that follow them
    x = np.random.default_rng(1).standard_normal((20, 4))
    expected = np.column_stack([x[2:, :1], _lags_of(x, 2, 2), x[2:, [1, 3]]])
    res = tools.add_lag(x, 2, lags=2, drop=True, insert=1)
    assert_allclose(res, expected, rtol=1e-15)


@pytest.mark.parametrize("form", SERIES_FORMS)
def test_add_lag_input_forms(form):
    x = np.random.default_rng(4).standard_normal(50)
    res = tools.add_lag(series_forms(x)[form], lags=2)
    assert isinstance(res, np.ndarray)
    assert_allclose(res, np.column_stack([x[2:], x[1:-1], x[:-2]]), rtol=1e-15)


@pytest.mark.parametrize("form", ["series", "column", "dataframe"])
def test_detrend_input_forms(form):
    x = np.random.default_rng(5).standard_normal(50)
    value = series_forms(x)[form]
    res = tools.detrend(value, order=1)
    # The residuals of the least squares fit on a constant and a trend
    design = np.column_stack([np.ones(50), np.arange(50.0)])
    expected = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    assert_allclose(np.squeeze(np.asarray(res)), expected, rtol=1e-8)
    assert type(res) is type(value)
    if form != "column":
        pd.testing.assert_index_equal(res.index, value.index)
    if form == "dataframe":
        assert list(res.columns) == ["x"]


def test_lagmat_negative_maxlag():
    # The message said that maxlag must be greater than 0, but 0 is allowed
    x = np.arange(10.0)
    with pytest.raises(ValueError, match=r"`maxlag` must be non-negative\. Got -1\."):
        tools.lagmat(x, -1)
    np.testing.assert_allclose(tools.lagmat(x, 0, original="in"), x[:, None])


@pytest.mark.parametrize("form", SERIES_FORMS)
@pytest.mark.parametrize("use_pandas", [False, True])
def test_lagmat_input_forms(form, use_pandas):
    x = np.random.default_rng(6).standard_normal(50)
    value = series_forms(x)[form]
    res = tools.lagmat(value, 2, trim="both", use_pandas=use_pandas)
    assert_allclose(np.asarray(res), np.column_stack([x[1:-1], x[:-2]]), rtol=1e-15)
    if use_pandas and form in ("series", "dataframe"):
        assert isinstance(res, pd.DataFrame)
        pd.testing.assert_index_equal(res.index, value.index[2:])
        assert list(res.columns) == ["x.L.1", "x.L.2"]
    else:
        assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("form", ["series", "column", "dataframe"])
def test_lagmat2ds_input_forms(form):
    x = np.random.default_rng(7).standard_normal(50)
    res = tools.lagmat2ds(series_forms(x)[form], 2, trim="both")
    assert isinstance(res, np.ndarray)
    assert_allclose(res, np.column_stack([x[2:], x[1:-1], x[:-2]]), rtol=1e-15)


@pytest.mark.parametrize("use_pandas", [False, True])
@pytest.mark.parametrize("maxlagex, dropex", [(1, 0), (3, 1), (2, 2)])
def test_lagmat2ds_maxlagex(maxlagex, dropex, use_pandas):
    rng = np.random.default_rng(2)
    nobs, maxlag0 = 30, 2
    x = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["a", "b"])

    def lag(col, j):
        # Lag j of a column, with zeros before the start of the sample
        out = np.zeros(nobs)
        out[j:] = x.iloc[: nobs - j, col]
        return out

    expected = np.column_stack(
        [lag(0, j) for j in range(maxlag0 + 1)]
        + [lag(1, j) for j in range(dropex, maxlagex + 1)]
    )
    res = tools.lagmat2ds(
        x, maxlag0, maxlagex=maxlagex, dropex=dropex, use_pandas=use_pandas
    )
    assert_allclose(np.asarray(res), expected, rtol=1e-15)
    if use_pandas:
        names = ["a"] + [f"a.L.{j}" for j in range(1, maxlag0 + 1)]
        names += [f"b.L.{j}" if j else "b" for j in range(dropex, maxlagex + 1)]
        assert list(res.columns) == names
        pd.testing.assert_index_equal(res.index, x.index)
    else:
        assert isinstance(res, np.ndarray)


def test_int_input_matches_float():
    xi = np.arange(30) % 7
    xf = xi.astype(float)
    assert_allclose(tools.add_trend(xi, "ct"), tools.add_trend(xf, "ct"), rtol=1e-15)
    assert_allclose(tools.add_lag(xi, lags=2), tools.add_lag(xf, lags=2), rtol=1e-15)
    assert_allclose(tools.detrend(xi), tools.detrend(xf), rtol=1e-12)
    assert_allclose(tools.lagmat(xi, 2), tools.lagmat(xf, 2), rtol=1e-15)
    assert_allclose(tools.lagmat2ds(xi, 2), tools.lagmat2ds(xf, 2), rtol=1e-15)


class TestAddTrend:
    @classmethod
    def setup_class(cls):
        cls.n = 200
        cls.arr_1d = np.arange(float(cls.n))
        cls.arr_2d = np.tile(np.arange(float(cls.n))[:, None], 2)
        cls.c = np.ones(cls.n)
        cls.t = np.arange(1.0, cls.n + 1)

    def test_series(self):
        s = pd.Series(self.arr_1d)
        appended = tools.add_trend(s)
        expected = pd.DataFrame(s)
        expected["const"] = self.c
        assert_frame_equal(expected, appended)

        prepended = tools.add_trend(s, prepend=True)
        expected = pd.DataFrame(s)
        expected.insert(0, "const", self.c)
        assert_frame_equal(expected, prepended)

        s = pd.Series(self.arr_1d)
        appended = tools.add_trend(s, trend="ct")
        expected = pd.DataFrame(s)
        expected["const"] = self.c
        expected["trend"] = self.t
        assert_frame_equal(expected, appended)

    def test_dataframe(self):
        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df)
        expected = df.copy()
        expected["const"] = self.c
        assert_frame_equal(expected, appended)

        prepended = tools.add_trend(df, prepend=True)
        expected = df.copy()
        expected.insert(0, "const", self.c)
        assert_frame_equal(expected, prepended)

        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df, trend="t")
        expected = df.copy()
        expected["trend"] = self.t
        assert_frame_equal(expected, appended)

        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df, trend="ctt")
        expected = df.copy()
        expected["const"] = self.c
        expected["trend"] = self.t
        expected["trend_squared"] = self.t**2
        assert_frame_equal(expected, appended)

    def test_duplicate_const(self):
        with pytest.raises(ValueError):
            tools.add_trend(
                x=self.c,
                trend="c",
                has_constant="raise",
            )
        with pytest.raises(ValueError):
            tools.add_trend(
                x=self.c,
                trend="ct",
                has_constant="raise",
            )
        df = pd.DataFrame(self.c)
        with pytest.raises(ValueError):
            tools.add_trend(x=df, trend="c", has_constant="raise")
        with pytest.raises(ValueError):
            tools.add_trend(x=df, trend="ct", has_constant="raise")

        skipped = tools.add_trend(self.c, trend="c")
        assert_equal(skipped, self.c[:, None])

        skipped_const = tools.add_trend(self.c, trend="ct", has_constant="skip")
        expected = np.vstack((self.c, self.t)).T
        assert_equal(skipped_const, expected)

        added = tools.add_trend(self.c, trend="c", has_constant="add")
        expected = np.vstack((self.c, self.c)).T
        assert_equal(added, expected)

        added = tools.add_trend(self.c, trend="ct", has_constant="add")
        expected = np.vstack((self.c, self.c, self.t)).T
        assert_equal(added, expected)

        with pytest.raises(
            ValueError,
            match=r"x contains one or more constant "
            r"columns. Column\(s\) col_1 are constant. Adding"
            r" a constant with trend='c' is not allowed.",
        ):
            add_trend_to = pd.DataFrame([self.t, self.c]).T
            add_trend_to.columns = ["col_0", "col_1"]
            tools.add_trend(x=add_trend_to, trend="c", has_constant="raise")

        with pytest.raises(
            ValueError,
            match=r"x contains one or more constant "
            r"columns. Column\(s\) 1 are constant. Adding"
            r" a constant with trend='ct' is not allowed.",
        ):
            add_trend_to = pd.DataFrame([self.t, self.c]).T
            tools.add_trend(x=add_trend_to, trend="ct", has_constant="raise")

    def test_dataframe_duplicate(self):
        df = pd.DataFrame(self.arr_2d, columns=["const", "trend"])
        tools.add_trend(df, trend="ct")
        tools.add_trend(df, trend="ct", prepend=True)

    def test_array(self):
        base = np.vstack((self.arr_1d, self.c, self.t, self.t**2)).T
        assert_equal(tools.add_trend(self.arr_1d), base[:, :2])
        assert_equal(tools.add_trend(self.arr_1d, trend="t"), base[:, [0, 2]])
        assert_equal(tools.add_trend(self.arr_1d, trend="ct"), base[:, :3])
        assert_equal(tools.add_trend(self.arr_1d, trend="ctt"), base)

        base = np.hstack(
            (
                self.c[:, None],
                self.t[:, None],
                self.t[:, None] ** 2,
                self.arr_2d,
            )
        )
        assert_equal(tools.add_trend(self.arr_2d, prepend=True), base[:, [0, 3, 4]])
        assert_equal(
            tools.add_trend(self.arr_2d, trend="t", prepend=True),
            base[:, [1, 3, 4]],
        )
        assert_equal(
            tools.add_trend(self.arr_2d, trend="ct", prepend=True),
            base[:, [0, 1, 3, 4]],
        )
        assert_equal(tools.add_trend(self.arr_2d, trend="ctt", prepend=True), base)

    def test_unknown_trend(self):
        with pytest.raises(ValueError):
            tools.add_trend(x=self.arr_1d, trend="unknown")

    def test_trend_n(self):
        assert_equal(tools.add_trend(self.arr_1d, "n"), self.arr_1d)
        assert tools.add_trend(self.arr_1d, "n") is not self.arr_1d
        assert_equal(tools.add_trend(self.arr_2d, "n"), self.arr_2d)
        assert tools.add_trend(self.arr_2d, "n") is not self.arr_2d

    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_trend_n_array_like(self, container, ndim):
        # A list was returned as a list, and a tuple raised AttributeError.
        # The other trends return an ndarray for both.
        if ndim == 1:
            data = self.arr_1d
            x = container(data.tolist())
        else:
            data = self.arr_2d
            x = container(container(row) for row in data.tolist())
        res = tools.add_trend(x, "n")
        assert isinstance(res, np.ndarray)
        assert_allclose(res, data, rtol=1e-15)
        assert isinstance(tools.add_trend(x, "c"), np.ndarray)

    def test_trend_n_pandas(self):
        # pandas input is returned as a copy of the same type
        s = pd.Series(self.arr_1d, name="y")
        res = tools.add_trend(s, "n")
        assert res is not s
        assert_series_equal(res, s)
        df = pd.DataFrame(self.arr_2d, columns=["a", "b"])
        res = tools.add_trend(df, "n")
        assert res is not df
        assert_frame_equal(res, df)


class TestLagmat2DS:
    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.macro_df = data.data[["year", "quarter", "realgdp", "cpi"]]
        rs = np.random.RandomState(12345)
        cls.random_data = rs.randn(100)
        index = [
            str(int(yr)) + "-Q" + str(int(qu))
            for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter, strict=True)
        ]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    @staticmethod
    def _prepare_expected(data, lags, trim="front"):
        t, k = data.shape
        expected = np.zeros((t + lags, (lags + 1) * k))
        for col in range(k):
            for i in range(lags + 1):
                if i < lags:
                    expected[i : -lags + i, (lags + 1) * col + i] = data[:, col]
                else:
                    expected[i:, (lags + 1) * col + i] = data[:, col]
        if trim == "front":
            expected = expected[:-lags]
        return expected

    def test_lagmat2ds_numpy(self):
        data = self.macro_df
        npdata = data.values
        lagmat = stattools.lagmat2ds(npdata, 2)
        expected = self._prepare_expected(npdata, 2)
        assert_array_equal(lagmat, expected)

        lagmat = stattools.lagmat2ds(npdata[:, :2], 3)
        expected = self._prepare_expected(npdata[:, :2], 3)
        assert_array_equal(lagmat, expected)

        npdata = self.series.values
        lagmat = stattools.lagmat2ds(npdata, 5)
        expected = self._prepare_expected(npdata[:, None], 5)
        assert_array_equal(lagmat, expected)

    def test_lagmat2ds_pandas(self):
        data = self.macro_df
        lagmat = stattools.lagmat2ds(data, 2)
        expected = self._prepare_expected(data.values, 2)
        assert_array_equal(lagmat, expected)

        lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, trim="both")
        expected = self._prepare_expected(data.values[:, :2], 3)
        expected = expected[3:]
        assert_array_equal(lagmat, expected)

        data = self.series
        lagmat = stattools.lagmat2ds(data, 5)
        expected = self._prepare_expected(data.values[:, None], 5)
        assert_array_equal(lagmat, expected)

    def test_lagmat2ds_use_pandas(self):
        data = self.macro_df
        lagmat = stattools.lagmat2ds(data, 2, use_pandas=True)
        expected = self._prepare_expected(data.values, 2)
        cols = []
        for c in data:
            for lags in range(3):
                if lags == 0:
                    cols.append(c)
                else:
                    cols.append(c + ".L." + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        assert_frame_equal(lagmat, expected)

        lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, use_pandas=True, trim="both")
        expected = self._prepare_expected(data.values[:, :2], 3)
        cols = []
        for c in data.iloc[:, :2]:
            for lags in range(4):
                if lags == 0:
                    cols.append(c)
                else:
                    cols.append(c + ".L." + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        expected = expected.iloc[3:]
        assert_frame_equal(lagmat, expected)

        data = self.series
        lagmat = stattools.lagmat2ds(data, 5, use_pandas=True)
        expected = self._prepare_expected(data.values[:, None], 5)

        cols = []
        c = data.name
        for lags in range(6):
            if lags == 0:
                cols.append(c)
            else:
                cols.append(c + ".L." + str(lags))

        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        assert_frame_equal(lagmat, expected)

    def test_3d_error(self):
        data = np.array(2)
        with pytest.raises(ValueError):
            stattools.lagmat2ds(data, 5)

        data = np.zeros((100, 2, 2))
        with pytest.raises(ValueError):
            stattools.lagmat2ds(data, 5)


def test_grangercausality():
    rs = np.random.RandomState(66483)
    data = rs.rand(100, 2)
    out = stattools.grangercausalitytests(data, maxlag=2)
    result, models = out[1]
    res2down, res2djoint, rconstr = models
    assert res2djoint.centered_tss is not res2djoint.uncentered_tss


def test_unintegrate_docstring_example():
    x = np.array([1, 3, 9.0, 19, 8.0])
    with pytest.warns(FutureWarning, match=r"tsatools\.unintegrate_levels is"):
        levels = tools.unintegrate_levels(x, 2)
    assert_allclose(levels, [1.0, 2.0], rtol=1e-12)
    with pytest.warns(FutureWarning, match=r"tsatools\.unintegrate is"):
        undiffed = tools.unintegrate(np.diff(x, 2), levels)
    assert_allclose(undiffed, x, rtol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_unintegrate_round_trip(d):
    # Integer-valued data, so that differencing and summing are exact
    rng = np.random.default_rng(20261119)
    x = rng.integers(-20, 20, size=25).cumsum().astype(float)
    with pytest.warns(FutureWarning, match=r"tsatools\.unintegrate_levels is"):
        levels = tools.unintegrate_levels(x, d)
    # The first value of the k-th difference, for k = 0, 1, 2
    first_values = [x[0], x[1] - x[0], x[2] - 2 * x[1] + x[0]]
    assert_allclose(levels, first_values[:d], rtol=1e-12)

    with pytest.warns(FutureWarning, match=r"tsatools\.unintegrate is") as record:
        undiffed = tools.unintegrate(np.diff(x, d), levels)
    # unintegrate recurses once per difference, but warns only once
    assert sum(issubclass(w.category, FutureWarning) for w in record) == 1
    assert_allclose(undiffed, x, rtol=1e-12)
