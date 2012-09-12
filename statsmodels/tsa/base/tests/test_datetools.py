from datetime import datetime
import numpy.testing as npt
from statsmodels.tsa.base.datetools import (_date_from_idx,
                _idx_from_dates, date_parser, date_range_str, dates_from_str,
                dates_from_range, _infer_freq, _freq_to_pandas)

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

try:
    from pandas import DatetimeIndex
    _pandas_08x = True
except ImportError, err:
    _pandas_08x = False
    d1 = datetime(2008, 12, 31)
    d2 = datetime(2012, 9, 30)

def test_infer_freq():
    d1 = datetime(2008, 12, 31)
    d2 = datetime(2012, 9, 30)

    if _pandas_08x:
        b = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['B']).values
        d = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['D']).values
        w = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['W']).values
        m = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['M']).values
        a = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['A']).values
        q = DatetimeIndex(start=d1, end=d2, freq=_freq_to_pandas['Q']).values
    else:
        from pandas import DateRange

        b = DateRange(d1, d2, offset=_freq_to_pandas['B']).values
        d = DateRange(d1, d2, offset=_freq_to_pandas['D']).values
        w = DateRange(d1, d2, offset=_freq_to_pandas['W']).values
        m = DateRange(d1, d2, offset=_freq_to_pandas['M']).values
        a = DateRange(d1, d2, offset=_freq_to_pandas['A']).values
        q = DateRange(d1, d2, offset=_freq_to_pandas['Q']).values

    assert _infer_freq(b[2:5]) == 'B'
    assert _infer_freq(b[:3]) == 'D'

    assert _infer_freq(b) == 'B'
    assert _infer_freq(d) == 'D'
    assert _infer_freq(w) == 'W'
    assert _infer_freq(m) == 'M'
    assert _infer_freq(a) == 'A'
    assert _infer_freq(q) == 'Q'
    assert _infer_freq(d[:3]) == 'D'
    assert _infer_freq(w[:3]) == 'W'
    assert _infer_freq(m[:3]) == 'M'
    assert _infer_freq(a[:3]) == 'A'
    assert _infer_freq(q[:3]) == 'Q'

