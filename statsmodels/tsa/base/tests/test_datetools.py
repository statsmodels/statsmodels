import calendar
from datetime import datetime

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from statsmodels.tsa.base.datetools import (
    date_parser,
    date_range_str,
    dates_from_range,
    dates_from_str,
)


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
    dt_range = dates_from_range("1959q1", "1962q2")
    assert results == dt_range

    # test with starting period not the first with length
    results = results[2:]
    dt_range = dates_from_range("1959q3", length=len(results))
    assert results == dt_range

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
               datetime(1960, 2, 29, 0, 0),
               datetime(1960, 3, 31, 0, 0),
               datetime(1960, 4, 30, 0, 0),
               datetime(1960, 5, 31, 0, 0),
               datetime(1960, 6, 30, 0, 0),
               datetime(1960, 7, 31, 0, 0),
               datetime(1960, 8, 31, 0, 0),
               datetime(1960, 9, 30, 0, 0),
               datetime(1960, 10, 31, 0, 0),
               datetime(1960, 11, 30, 0, 0),
               datetime(1960, 12, 31, 0, 0),
               datetime(1961, 1, 31, 0, 0),
               datetime(1961, 2, 28, 0, 0),
               datetime(1961, 3, 31, 0, 0),
               datetime(1961, 4, 30, 0, 0),
               datetime(1961, 5, 31, 0, 0),
               datetime(1961, 6, 30, 0, 0),
               datetime(1961, 7, 31, 0, 0),
               datetime(1961, 8, 31, 0, 0),
               datetime(1961, 9, 30, 0, 0)]

    dt_range = dates_from_range("1959m3", length=len(results))
    assert results == dt_range


ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


@pytest.mark.parametrize("year", [1900, 1999, 2000, 2004])
@pytest.mark.parametrize("month", range(1, 13))
def test_date_parser_month_end(year, month):
    # Monthly dates are at the last day of the month, also in leap years
    expected = datetime(year, month, calendar.monthrange(year, month)[1])
    assert date_parser(f"{year}m{month}") == expected
    assert date_parser(f"{year}:M{ROMAN[month]}") == expected


@pytest.mark.parametrize("quarter", [1, 2, 3, 4])
def test_date_parser_quarter_end(quarter):
    month = 3 * quarter
    expected = datetime(1999, month, calendar.monthrange(1999, month)[1])
    assert date_parser(f"1999Q{quarter}") == expected
    assert date_parser(f"1999:q{ROMAN[quarter]}") == expected


@pytest.mark.parametrize("year", ["1999", "2000", "0099"])
def test_date_parser_annual(year):
    # Annual dates are at the end of the year
    assert date_parser(year) == datetime(int(year), 12, 31)


def test_date_parser_fallback():
    # Other strings are parsed by pandas.to_datetime, which gets the keywords
    assert date_parser("1999-04-15") == datetime(1999, 4, 15)
    assert date_parser("04/05/1999") == datetime(1999, 4, 5)
    assert date_parser("04/05/1999", dayfirst=True) == datetime(1999, 5, 4)


@pytest.mark.parametrize("end, length", [("1970", None), (None, 6)])
def test_annual_range(end, length):
    years = range(1965, 1971)
    assert date_range_str("1965", end, length) == [str(year) for year in years]
    expected = [datetime(year, 12, 31) for year in years]
    assert dates_from_range("1965", end, length) == expected


@pytest.mark.parametrize(
    "start, end, length, first, periods, freq",
    [
        ("1959q3", "1960q2", None, "1959Q3", 4, "Q"),
        ("1959Q4", None, 5, "1959Q4", 5, "Q"),
        ("1959q1", None, 8, "1959Q1", 8, "Q"),
        ("1959:m11", "1960:m2", None, "1959-11", 4, "M"),
        ("1959m3", None, 31, "1959-03", 31, "M"),
        ("1959m1", None, 24, "1959-01", 24, "M"),
    ],
)
def test_dates_from_range_period_ends(start, end, length, first, periods, freq):
    expected = pd.period_range(first, periods=periods, freq=freq)
    expected = [p.end_time.normalize().to_pydatetime() for p in expected]
    assert dates_from_range(start, end, length) == expected


@pytest.mark.parametrize(
    "start, length, match",
    [
        ("1965-01", None, "Date 1965-01 not understood"),
        ("1965q1", None, "length must be provided if end is None"),
        ("1965q1", 0, "length must be provided if end is None"),
    ],
)
def test_date_range_str_errors(start, length, match):
    with pytest.raises(ValueError, match=match):
        date_range_str(start, length=length)


@pytest.mark.parametrize("container", [list, tuple, np.array, pd.Series])
def test_dates_from_str_sequence(container):
    dates = dates_from_str(container(["1999q1", "1999m4", "1999", "1999-04-15"]))
    expected = [
        datetime(1999, 3, 31),
        datetime(1999, 4, 30),
        datetime(1999, 12, 31),
        datetime(1999, 4, 15),
    ]
    assert dates == expected


def _abbreviated_labels(periods):
    # Labels of pandas periods in the format of date_range_str, e.g. "1959q3"
    freq = periods.freqstr[0]
    if freq == "Q":
        return [f"{p.year}q{p.quarter}" for p in periods]
    if freq == "M":
        return [f"{p.year}m{p.month}" for p in periods]
    return [str(p.year) for p in periods]


@pytest.mark.parametrize(
    "start, end, length, expected",
    [
        # Ranges within one year
        ("1959q1", "1959q3", None, pd.period_range("1959Q1", "1959Q3", freq="Q")),
        ("1959m2", "1959m11", None, pd.period_range("1959-02", "1959-11", freq="M")),
        ("1965", "1965", None, pd.period_range("1965", "1965", freq="Y")),
        ("1959q3", None, 2, pd.period_range("1959Q3", periods=2, freq="Q")),
        ("1959m5", None, 1, pd.period_range("1959-05", periods=1, freq="M")),
        # A length that continues into the next year
        ("1959q4", None, 6, pd.period_range("1959Q4", periods=6, freq="Q")),
        ("1959m12", None, 14, pd.period_range("1959-12", periods=14, freq="M")),
        ("1959m8", None, 30, pd.period_range("1959-08", periods=30, freq="M")),
        # Ranges that were already supported
        ("1959q3", "1962q2", None, pd.period_range("1959Q3", "1962Q2", freq="Q")),
        ("1959m3", None, 31, pd.period_range("1959-03", periods=31, freq="M")),
        ("1965", "1968", None, pd.period_range("1965", "1968", freq="Y")),
    ],
)
def test_date_range_str_ranges(start, end, length, expected):
    assert date_range_str(start, end, length) == _abbreviated_labels(expected)


@pytest.mark.parametrize(
    "start, end, length",
    [
        ("1959q3", "1959q1", None),
        ("1959m3", "1958m12", None),
        ("1965", "1964", None),
        ("1959q3", None, -2),
        ("1965", None, -2),
    ],
)
def test_date_range_str_end_before_start(start, end, length):
    with pytest.raises(ValueError, match="end must not be before start"):
        date_range_str(start, end, length)


@pytest.mark.parametrize(
    "start, length, expected",
    [
        ("1959m12", 14, pd.period_range("1959-12", periods=14, freq="M")),
        ("1959q4", 6, pd.period_range("1959Q4", periods=6, freq="Q")),
    ],
)
def test_dates_from_range_next_year(start, length, expected):
    # The dates are at the end of each period
    dates = dates_from_range(start, length=length)
    assert dates == [p.end_time.normalize().to_pydatetime() for p in expected]
