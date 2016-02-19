from datetime import datetime
from pandas import DatetimeIndex
import numpy.testing as npt
from statsmodels.tsa.base.datetools import (_date_from_idx,
                _idx_from_dates, date_parser, date_range_str, dates_from_str,
                dates_from_range, _infer_freq, _freq_to_pandas)
from pandas import DatetimeIndex, PeriodIndex

def test_date_from_idx():
    d1 = datetime(2008, 12, 31)
    idx = 15
    npt.assert_equal(_date_from_idx(d1, idx, 'Q'), datetime(2012, 9, 30))
    npt.assert_equal(_date_from_idx(d1, idx, 'A'), datetime(2023, 12, 31))
    npt.assert_equal(_date_from_idx(d1, idx, 'B'), datetime(2009, 1, 21))
    npt.assert_equal(_date_from_idx(d1, idx, 'D'), datetime(2009, 1, 15))
    npt.assert_equal(_date_from_idx(d1, idx, 'W'), datetime(2009, 4, 12))
    npt.assert_equal(_date_from_idx(d1, idx, 'M'), datetime(2010, 3, 31))

def test_idx_from_date():
    d1 = datetime(2008, 12, 31)
    idx = 15
    npt.assert_equal(_idx_from_dates(d1, datetime(2012, 9, 30), 'Q'), idx)
    npt.assert_equal(_idx_from_dates(d1, datetime(2023, 12, 31), 'A'), idx)
    npt.assert_equal(_idx_from_dates(d1, datetime(2009, 1, 21), 'B'), idx)
    npt.assert_equal(_idx_from_dates(d1, datetime(2009, 1, 15), 'D'), idx)
    # move d1 and d2 forward to end of week
    npt.assert_equal(_idx_from_dates(datetime(2009, 1, 4),
                      datetime(2009, 4, 17), 'W'), idx-1)
    npt.assert_equal(_idx_from_dates(d1, datetime(2010, 3, 31), 'M'), idx)

def test_regex_matching_month():
    t1 = "1999m4"
    t2 = "1999:m4"
    t3 = "1999:mIV"
    t4 = "1999mIV"
    result = datetime(1999, 4, 30)
    npt.assert_equal(date_parser(t1), result)
    npt.assert_equal(date_parser(t2), result)
    npt.assert_equal(date_parser(t3), result)
    npt.assert_equal(date_parser(t4), result)

def test_regex_matching_quarter():
    t1 = "1999q4"
    t2 = "1999:q4"
    t3 = "1999:qIV"
    t4 = "1999qIV"
    result = datetime(1999, 12, 31)
    npt.assert_equal(date_parser(t1), result)
    npt.assert_equal(date_parser(t2), result)
    npt.assert_equal(date_parser(t3), result)
    npt.assert_equal(date_parser(t4), result)

def test_dates_from_range():
    results = [datetime(1959, 3, 31, 0, 0),
               datetime(1959, 6, 30, 0, 0),
               datetime(1959, 9, 30, 0, 0),
               datetime(1959, 12, 31, 0, 0),
               datetime(1960, 3, 31, 0, 0),
               datetime(1960, 6, 30, 0, 0),
               datetime(1960, 9, 30, 0, 0),
               datetime(1960, 12, 31, 0, 0),
               datetime(1961, 3, 31, 0, 0),
               datetime(1961, 6, 30, 0, 0),
               datetime(1961, 9, 30, 0, 0),
               datetime(1961, 12, 31, 0, 0),
               datetime(1962, 3, 31, 0, 0),
               datetime(1962, 6, 30, 0, 0)]
    dt_range = dates_from_range('1959q1', '1962q2')
    npt.assert_(results == dt_range)

    # test with starting period not the first with length
    results = results[2:]
    dt_range = dates_from_range('1959q3', length=len(results))
    npt.assert_(results == dt_range)

    # check month
    results = [datetime(1959, 3, 31, 0, 0),
               datetime(1959, 4, 30, 0, 0),
               datetime(1959, 5, 31, 0, 0),
               datetime(1959, 6, 30, 0, 0),
               datetime(1959, 7, 31, 0, 0),
               datetime(1959, 8, 31, 0, 0),
               datetime(1959, 9, 30, 0, 0),
               datetime(1959, 10, 31, 0, 0),
               datetime(1959, 11, 30, 0, 0),
               datetime(1959, 12, 31, 0, 0),
               datetime(1960, 1, 31, 0, 0),
               datetime(1960, 2, 28, 0, 0),
               datetime(1960, 3, 31, 0, 0),
               datetime(1960, 4, 30, 0, 0),
               datetime(1960, 5, 31, 0, 0),
               datetime(1960, 6, 30, 0, 0),
               datetime(1960, 7, 31, 0, 0),
               datetime(1960, 8, 31, 0, 0),
               datetime(1960, 9, 30, 0, 0),
               datetime(1960, 10, 31, 0, 0),
               datetime(1960, 12, 31, 0, 0),
               datetime(1961, 1, 31, 0, 0),
               datetime(1961, 2, 28, 0, 0),
               datetime(1961, 3, 31, 0, 0),
               datetime(1961, 4, 30, 0, 0),
               datetime(1961, 5, 31, 0, 0),
               datetime(1961, 6, 30, 0, 0),
               datetime(1961, 7, 31, 0, 0),
               datetime(1961, 8, 31, 0, 0),
               datetime(1961, 9, 30, 0, 0),
               datetime(1961, 10, 31, 0, 0)]

    dt_range = dates_from_range("1959m3", length=len(results))


def test_infer_freq():
    d1 = datetime(2008, 12, 31)
    d2 = datetime(2012, 9, 30)

    b = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['B']).values
    d = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['D']).values
    w = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['W']).values
    m = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['M']).values
    a = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['A']).values
    q = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['Q']).values

    npt.assert_(_infer_freq(w) == 'W-SUN')
    npt.assert_(_infer_freq(a) == 'A-DEC')
    npt.assert_(_infer_freq(q) == 'Q-DEC')
    npt.assert_(_infer_freq(w[:3]) == 'W-SUN')
    npt.assert_(_infer_freq(a[:3]) == 'A-DEC')
    npt.assert_(_infer_freq(q[:3]) == 'Q-DEC')
    npt.assert_(_infer_freq(b[2:5]) == 'B')
    npt.assert_(_infer_freq(b[:3]) == 'D')
    npt.assert_(_infer_freq(b) == 'B')
    npt.assert_(_infer_freq(d) == 'D')
    npt.assert_(_infer_freq(m) == 'M')
    npt.assert_(_infer_freq(d[:3]) == 'D')
    npt.assert_(_infer_freq(m[:3]) == 'M')

def test_period_index():
    # tests 1285
    from pandas import PeriodIndex
    dates = PeriodIndex(start="1/1/1990", periods=20, freq="M")
    npt.assert_(_infer_freq(dates) == "M")
