from statsmodels.compat.pandas import (
    MONTH_END,
    PD_LT_2_2_0,
    PD_LT_3,
    QUARTER_END,
    YEAR_END,
    is_int_index,
)
from statsmodels.compat.pytest import pytest_warns

from collections.abc import Hashable
import warnings

import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.deterministic import (
    CalendarFourier,
    CalendarSeasonality,
    CalendarTimeTrend,
    DeterministicProcess,
    DeterministicTerm,
    Fourier,
    Seasonality,
    TimeTrend,
)


@pytest.fixture(scope="module")
def time_index(request):
    return pd.date_range("2000-01-01", periods=833, freq="B")


@pytest.fixture(scope="module", params=["range", "period", "datetime", "fib", "int64"])
def index(request):
    param = request.param
    if param in ("period", "datetime"):
        idx = pd.date_range("2000-01-01", periods=137, freq=MONTH_END)
        if param == "period":
            idx = idx.to_period("M")
    elif param == "range":
        idx = pd.RangeIndex(0, 123)
    elif param == "int64":
        idx = pd.Index(np.arange(123))
    elif param == "fib":
        fib = [0, 1]
        for _ in range(113):
            fib.append(fib[-2] + fib[-1])
        idx = pd.Index(fib)
    else:
        raise NotImplementedError
    return idx


@pytest.fixture(scope="module", params=[None, False, "list"])
def forecast_index(request):
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    if request.param is None:
        return None
    elif request.param == "list":
        return list(idx)
    return idx


@pytest.mark.smoke
def test_time_trend_smoke(index, forecast_index):
    tt = TimeTrend(True, 2)
    tt.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63 and forecast_index is None
    ):
        warn = UserWarning
    with pytest_warns(warn):
        tt.out_of_sample(steps, index, forecast_index)
    str(tt)
    hash(tt)
    assert isinstance(tt.order, int)
    assert isinstance(tt._constant, bool)
    assert TimeTrend.from_string("ctt") == tt
    assert TimeTrend.from_string("ct") != tt
    assert TimeTrend.from_string("t") != tt
    assert TimeTrend.from_string("n") != tt
    assert Seasonality(12) != tt
    tt0 = TimeTrend(False, 0)
    tt0.in_sample(index)
    str(tt0)


@pytest.mark.smoke
def test_seasonality_smoke(index, forecast_index):
    s = Seasonality(12)
    s.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63 and forecast_index is None
    ):
        warn = UserWarning
    with pytest_warns(warn):
        s.out_of_sample(steps, index, forecast_index)
    assert isinstance(s.period, int)
    str(s)
    hash(s)
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)) and index.freq:
        s = Seasonality.from_index(index)
        s.in_sample(index)
        s.out_of_sample(steps, index, forecast_index)
        Seasonality.from_index(list(index))


@pytest.mark.smoke
def test_fourier_smoke(index, forecast_index):
    f = Fourier(12, 2)
    f.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63 and forecast_index is None
    ):
        warn = UserWarning
    with pytest_warns(warn):
        f.out_of_sample(steps, index, forecast_index)
    assert isinstance(f.period, float)
    assert isinstance(f.order, int)
    str(f)
    hash(f)
    with pytest.raises(ValueError, match=r"2 \* order must be <= period"):
        Fourier(12, 7)


@pytest.mark.smoke
def test_calendar_time_trend_smoke(time_index, forecast_index):
    ct = CalendarTimeTrend(YEAR_END, order=2)
    ct.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    ct.out_of_sample(steps, time_index, forecast_index)
    str(ct)
    hash(ct)
    assert isinstance(ct.order, int)
    assert isinstance(ct.constant, bool)
    assert isinstance(ct.freq, str)
    assert ct.base_period is None


@pytest.mark.smoke
def test_calendar_fourier_smoke(time_index, forecast_index):
    cf = CalendarFourier(YEAR_END, 2)
    cf.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    cf.out_of_sample(steps, time_index, forecast_index)
    assert isinstance(cf.order, int)
    assert isinstance(cf.freq, str)
    str(cf)
    repr(cf)
    hash(cf)


params = CalendarSeasonality._supported
cs_params = [(k, k2) for k, v in params.items() for k2 in v.keys()]


@pytest.mark.parametrize("freq_period", cs_params)
def test_calendar_seasonality(time_index, forecast_index, freq_period):
    freq, period = freq_period
    cs = CalendarSeasonality(period, freq)
    cs.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    cs.out_of_sample(steps, time_index, forecast_index)
    assert isinstance(cs.period, str)
    assert isinstance(cs.freq, str)
    str(cs)
    repr(cs)
    hash(cs)
    cs2 = CalendarSeasonality(period, freq)
    assert cs == cs2


def test_forbidden_index():
    index = pd.RangeIndex(0, 10)
    ct = CalendarTimeTrend(YEAR_END, order=2)
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct.in_sample(index)


def test_calendar_time_trend_base(time_index):
    ct = CalendarTimeTrend(MONTH_END, True, order=3, base_period="1960-1-1")
    ct2 = CalendarTimeTrend(MONTH_END, True, order=3)
    assert ct != ct2
    str(ct)
    str(ct2)
    assert ct.base_period is not None
    assert ct2.base_period is None


def test_invalid_freq_period(time_index):
    with pytest.raises(ValueError, match="The combination of freq="):
        CalendarSeasonality("h", YEAR_END)
    cs = CalendarSeasonality("B", "W")
    with pytest.raises(ValueError, match="freq is B but index contains"):
        cs.in_sample(pd.date_range("2000-1-1", periods=10, freq="D"))


def test_check_index_type():
    ct = CalendarTimeTrend(YEAR_END, True, order=3)
    idx = pd.RangeIndex(0, 20)
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct._check_index_type(idx, pd.DatetimeIndex)
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct._check_index_type(idx, (pd.DatetimeIndex,))
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct._check_index_type(idx, (pd.DatetimeIndex, pd.PeriodIndex))
    idx = pd.Index([0, 1, 1, 2, 3, 5, 8, 13])
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        types = (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)
        ct._check_index_type(idx, types)


def test_unknown_freq():
    with pytest.raises(ValueError, match="freq is not understood by pandas"):
        CalendarTimeTrend("unknown", True, order=3)


# pandas 2.2 and 2.3 make periods of start-anchored freqs, as the end-anchored
# freqs anchored in December ("QS-FEB" as "QE-DEC"). Earlier and later
# versions have none.
PD_START_ANCHORED_PERIODS = not PD_LT_2_2_0 and PD_LT_3
_START_ANCHORED_SKIP = pytest.mark.skipif(
    PD_START_ANCHORED_PERIODS, reason="pandas has periods for start-anchored freqs"
)
# "YS-JUL", "BYS-JUL" and "YE-JUN" are "AS-JUL", "BAS-JUL" and "A-JUN" before
# pandas 2.2
YEAR_START_JUL = "AS-JUL" if PD_LT_2_2_0 else "YS-JUL"
BUSINESS_YEAR_START_JUL = "BAS-JUL" if PD_LT_2_2_0 else "BYS-JUL"
YEAR_END_JUN = "A-JUN" if PD_LT_2_2_0 else "YE-JUN"
CALENDAR_TERMS = [
    (CalendarFourier, {"order": 2}),
    (CalendarTimeTrend, {"order": 1}),
    (CalendarTimeTrend, {"order": 2, "base_period": "2000-02-15"}),
    (CalendarTimeTrend.from_string, {"trend": "ctt"}),
]


def _calendar_term_id(param):
    if isinstance(param, dict):
        return "-".join(f"{key}={value}" for key, value in param.items())
    return param.__qualname__


@pytest.mark.parametrize(
    "freq",
    [
        "SMS",
        "WOM-1MON",
        pytest.param(
            "C",
            marks=pytest.mark.skipif(PD_LT_2_2_0, reason="pandas has C periods"),
        ),
    ],
)
@pytest.mark.parametrize("map_start_anchored", [True, False])
@pytest.mark.parametrize("term, kwargs", CALENDAR_TERMS, ids=_calendar_term_id)
def test_calendar_term_freq_without_periods(freq, map_start_anchored, term, kwargs):
    # pandas has no periods with these freqs, and they have no end-anchored
    # equivalent, so an error from pandas, usually an AttributeError, was
    # raised when the terms were computed
    with pytest.raises(ValueError, match=f'freq "{freq}" is not supported by'):
        term(freq, map_start_anchored=map_start_anchored, **kwargs)


@pytest.mark.parametrize(
    "freq, end_freq",
    [
        ("MS", MONTH_END),
        ("2MS", f"2{MONTH_END}"),
        ("QS", f"{QUARTER_END}-DEC"),
        ("QS-JAN", f"{QUARTER_END}-DEC"),
        ("QS-FEB", f"{QUARTER_END}-JAN"),
        ("QS-DEC", f"{QUARTER_END}-NOV"),
        ("QS-APR", f"{QUARTER_END}-DEC"),
        ("QS-JUL", f"{QUARTER_END}-DEC"),
        ("QS-OCT", f"{QUARTER_END}-DEC"),
        ("2QS-APR", f"2{QUARTER_END}-DEC"),
        ("YS", YEAR_END),
        (YEAR_START_JUL, YEAR_END_JUN),
        ("BMS", MONTH_END),
        ("2BMS", f"2{MONTH_END}"),
        ("BQS", f"{QUARTER_END}-DEC"),
        ("BQS-APR", f"{QUARTER_END}-DEC"),
        ("BQS-FEB", f"{QUARTER_END}-JAN"),
        ("BYS", YEAR_END),
        (BUSINESS_YEAR_START_JUL, YEAR_END_JUN),
    ],
)
@pytest.mark.parametrize("term, kwargs", CALENDAR_TERMS, ids=_calendar_term_id)
def test_calendar_term_start_anchored_mapped(freq, end_freq, term, kwargs):
    # A start-anchored freq gives the terms of the end-anchored freq with the
    # same periods, on every pandas version, and keeps its name. The calendar
    # quarters ("QS-APR" etc.) are those of "QE-DEC", as in pandas 2.2 and 2.3.
    # The business-day freqs have the periods of their calendar twins.
    index = pd.date_range("1999-12-01", "2002-03-31", freq="D")
    created = term(freq, **kwargs)
    assert created.freq == pd.tseries.frequencies.to_offset(freq).freqstr
    terms = created.in_sample(index)
    expected = term(end_freq, **kwargs).in_sample(index)
    np.testing.assert_allclose(np.asarray(terms), np.asarray(expected), rtol=1e-12)
    assert terms.index.equals(expected.index)
    assert created == term(freq, **kwargs)
    assert created != term(end_freq, **kwargs)


@pytest.mark.parametrize(
    "freq, months, first_month",
    [
        ("MS", 1, 1),
        ("QS", 3, 1),
        ("QS-FEB", 3, 2),
        ("QS-APR", 3, 4),
        (YEAR_START_JUL, 12, 7),
        ("BMS", 1, 1),
        ("BQS-FEB", 3, 2),
        (BUSINESS_YEAR_START_JUL, 12, 7),
    ],
)
def test_calendar_term_start_anchored_periods(freq, months, first_month):
    # The periods of a start-anchored freq start on the first day of
    # first_month and every months months after it. tau is the fraction of the
    # period that has elapsed, and the time trend is the number of periods
    # since the base period plus 1 + tau.
    index = pd.date_range("1999-12-01 05:00", "2002-03-31", freq="17h")

    def period_start(month):
        # month counts the months since January of year 0
        start = (month - (first_month - 1)) // months * months + first_month - 1
        parts = {"year": start // 12, "month": start % 12 + 1, "day": 1}
        return start, pd.to_datetime(pd.DataFrame(parts)).to_numpy()

    month = np.asarray(12 * index.year + index.month - 1, dtype=np.int64)
    start, start_date = period_start(month)
    _, next_date = period_start(start + months)
    tau = (index.to_numpy() - start_date) / (next_date - start_date)
    assert np.all((tau >= 0) & (tau < 1))
    fourier = CalendarFourier(freq, 1).in_sample(index)
    expected = np.column_stack([np.sin(2 * np.pi * tau), np.cos(2 * np.pi * tau)])
    # sin and cos are zero, or close to it, at some points of each period, so
    # atol
    np.testing.assert_allclose(np.asarray(fourier), expected, rtol=1e-12, atol=1e-12)

    base_start, _ = period_start(np.array([12 * 2000 + 1], dtype=np.int64))
    trend = CalendarTimeTrend(freq, False, 1, base_period="2000-02-15").in_sample(index)
    expected = (start - base_start[0]) // months + 1 + tau
    np.testing.assert_allclose(np.asarray(trend)[:, 0], expected, rtol=1e-12)


@pytest.mark.parametrize(
    "freq",
    [
        pytest.param(freq, marks=_START_ANCHORED_SKIP)
        for freq in ["MS", "2MS", "QS", "QS-FEB", "YS", YEAR_START_JUL]
        + ["BQS", "BQS-FEB", "BYS", BUSINESS_YEAR_START_JUL]
    ]
    + ["BMS"],
)
@pytest.mark.parametrize("term, kwargs", CALENDAR_TERMS, ids=_calendar_term_id)
def test_calendar_term_start_anchored_not_mapped_raises(freq, term, kwargs):
    # Without the map, freq is passed to pandas, which has no periods for
    # start-anchored freqs except in versions 2.2 and 2.3, and none for "BMS"
    # in any version
    with pytest.raises(ValueError, match=f'freq "{freq}" is not supported by'):
        term(freq, map_start_anchored=False, **kwargs)


@pytest.mark.skipif(
    not PD_START_ANCHORED_PERIODS,
    reason="pandas has no periods for start-anchored freqs",
)
@pytest.mark.parametrize(
    "freq, pandas_freq",
    [
        ("MS", MONTH_END),
        ("QS", QUARTER_END),
        ("QS-FEB", QUARTER_END),
        ("YS", YEAR_END),
        ("YS-JUL", YEAR_END),
        ("BQS", QUARTER_END),
        ("BQS-FEB", QUARTER_END),
        ("BYS-JUL", YEAR_END),
    ],
)
@pytest.mark.parametrize(
    "term, kwargs",
    [term for term in CALENDAR_TERMS if "base_period" not in term[1]],
    ids=_calendar_term_id,
)
def test_calendar_term_start_anchored_not_mapped(freq, pandas_freq, term, kwargs):
    # Without the map, freq is passed to pandas, and pandas 2.2 and 2.3 make
    # periods of start-anchored freqs as the end-anchored ones anchored in
    # December, so "QS-FEB" and "YS-JUL" (and "BQS-FEB" and "BYS-JUL") get the
    # quarters and years of "QE-DEC" and "YE-DEC"
    index = pd.date_range("1999-12-01", "2002-03-31", freq="D")
    created = term(freq, map_start_anchored=False, **kwargs)
    terms = created.in_sample(index)
    expected = term(pandas_freq, **kwargs).in_sample(index)
    np.testing.assert_allclose(np.asarray(terms), np.asarray(expected), rtol=1e-12)
    assert created != term(freq, **kwargs)


@pytest.mark.skipif(
    not PD_START_ANCHORED_PERIODS,
    reason="pandas has no periods for start-anchored freqs",
)
@pytest.mark.parametrize(
    "freq",
    ["MS", "2MS", "QS", "QS-JAN", "QS-APR", "QS-JUL", "QS-OCT", "2QS-APR", "YS"]
    + ["BQS", "BQS-APR", "BYS"],
)
@pytest.mark.parametrize(
    "term, kwargs",
    [term for term in CALENDAR_TERMS if "base_period" not in term[1]],
    ids=_calendar_term_id,
)
def test_calendar_term_start_anchored_map_unchanged(freq, term, kwargs):
    # pandas 2.2 and 2.3 already give these freqs the periods of the map, so
    # the map does not change their terms, index or columns there
    index = pd.date_range("1999-12-01 05:00", "2002-03-31", freq="17h")
    terms = term(freq, **kwargs).in_sample(index)
    expected = term(freq, map_start_anchored=False, **kwargs).in_sample(index)
    pd.testing.assert_frame_equal(terms, expected, rtol=1e-15)


@pytest.mark.skipif(
    not PD_START_ANCHORED_PERIODS,
    reason="pandas has no periods for start-anchored freqs",
)
@pytest.mark.parametrize("freq", ["MS", "QS-FEB", "YS-JUL"])
def test_calendar_time_trend_start_anchored_not_mapped_base_period(freq):
    # Without the map, pandas 2.2 and 2.3 have periods of start-anchored freqs
    # but no period range, which the base period needs, and raise
    with pytest.raises(ValueError, match="period frequency"):
        CalendarTimeTrend(freq, base_period="2000-02-15", map_start_anchored=False)
    created = CalendarTimeTrend(freq, base_period="2000-02-15")
    assert created.base_period == "2000-02-15"


@pytest.mark.parametrize("term, kwargs", CALENDAR_TERMS, ids=_calendar_term_id)
def test_calendar_term_map_start_anchored_other_freq(term, kwargs):
    # The map does not change other freqs
    index = pd.date_range("1999-12-01", "2002-03-31", freq="D")
    created = term(MONTH_END, map_start_anchored=False, **kwargs)
    assert created == term(MONTH_END, **kwargs)
    terms = created.in_sample(index)
    expected = term(MONTH_END, **kwargs).in_sample(index)
    np.testing.assert_allclose(np.asarray(terms), np.asarray(expected), rtol=1e-15)


@pytest.mark.parametrize("term", [CalendarFourier, CalendarTimeTrend])
def test_calendar_term_business_day_freq(term):
    # Periods with freq "B" are deprecated in recent pandas, but still
    # supported, so checking the freq does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        created = term("B", order=1)
    assert created.freq == "B"


def test_calendar_seasonality_month_start():
    # The dummies are computed without periods, so "MS" is supported
    index = pd.date_range("2000-01-01", periods=24, freq="MS")
    terms = CalendarSeasonality("MS", "Y").in_sample(index)
    expected = CalendarSeasonality(MONTH_END, "Y").in_sample(index)
    np.testing.assert_allclose(terms, expected, rtol=1e-15)
    np.testing.assert_allclose(terms, np.tile(np.eye(12), (2, 1)), rtol=1e-15)


def test_invalid_formcast_index(index):
    tt = TimeTrend(order=4)
    with pytest.raises(ValueError, match="The number of values in forecast_"):
        tt.out_of_sample(10, index, pd.RangeIndex(11))


def test_seasonal_from_index_err():
    index = pd.Index([0, 1, 1, 2, 3, 5, 8, 12])
    with pytest.raises(TypeError):
        Seasonality.from_index(index)
    index = pd.date_range("2000-1-1", periods=10)[[0, 1, 2, 3, 5, 8]]
    with pytest.raises(ValueError):
        Seasonality.from_index(index)


def test_time_trend(index):
    tt = TimeTrend(constant=True)
    const = tt.in_sample(index)
    assert const.shape == (index.shape[0], 1)
    assert np.all(const == 1)
    pd.testing.assert_index_equal(const.index, index)
    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63
    ):
        warn = UserWarning
    with pytest_warns(warn):
        const_fcast = tt.out_of_sample(23, index)
    assert np.all(const_fcast == 1)

    tt = TimeTrend(constant=False)
    empty = tt.in_sample(index)
    assert empty.shape == (index.shape[0], 0)

    tt = TimeTrend(constant=False, order=2)
    t2 = tt.in_sample(index)
    assert t2.shape == (index.shape[0], 2)
    assert list(t2.columns) == ["trend", "trend_squared"]

    tt = TimeTrend(constant=True, order=2)
    final = tt.in_sample(index)
    expected = pd.concat([const, t2], axis=1)
    pd.testing.assert_frame_equal(final, expected)

    tt = TimeTrend(constant=True, order=2)
    short = tt.in_sample(index[:-50])
    with pytest_warns(warn):
        remainder = tt.out_of_sample(50, index[:-50])
    direct = tt.out_of_sample(steps=50, index=index[:-50], forecast_index=index[-50:])
    combined = pd.concat([short, remainder], axis=0)
    if isinstance(index, (pd.DatetimeIndex, pd.RangeIndex)):
        pd.testing.assert_frame_equal(combined, final)
    combined = pd.concat([short, direct], axis=0)
    pd.testing.assert_frame_equal(combined, final, check_index_type=False)


@pytest.mark.parametrize(
    "term, expected",
    [
        (TimeTrend(True, 0), "TimeTrend(Constant)"),
        (TimeTrend(False, 0), "TimeTrend(Empty)"),
        (TimeTrend(True, 1), "TimeTrend(Constant,Powers 1 to 1)"),
        (TimeTrend(False, 2), "TimeTrend(Powers 1 to 2)"),
        (TimeTrend(True, 3), "TimeTrend(Constant,Powers 1 to 3)"),
        (
            CalendarTimeTrend("D", True, 2),
            "CalendarTimeTrend(Constant,Powers 1 to 2, freq=D)",
        ),
    ],
)
def test_time_trend_str(term, expected):
    # The highest power was given as order + 1
    assert str(term) == expected
    # The terms have the powers 1, ..., order of the time trend
    index = pd.date_range("2000-01-01", periods=5, freq="D")
    trends = term.in_sample(index).iloc[:, int(term.constant) :].to_numpy()
    assert trends.shape[1] == term.order
    powers = np.arange(1, term.order + 1)
    np.testing.assert_allclose(trends, trends[:, :1] ** powers, rtol=1e-14)


@pytest.mark.parametrize("order", [-1, -3])
@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize(
    "create",
    [
        TimeTrend,
        lambda constant, order: CalendarTimeTrend("D", constant, order),
        lambda constant, order: DeterministicProcess(
            pd.RangeIndex(10), constant=constant, order=order
        ),
    ],
    ids=["TimeTrend", "CalendarTimeTrend", "DeterministicProcess"],
)
def test_time_trend_negative_order(create, constant, order):
    # A negative order was accepted, and computing the terms raised an
    # unrelated error, or gave "Powers 1 to 0" in the string
    with pytest.raises(ValueError, match=f"order must be non-negative, got {order}"):
        create(constant, order)


def test_seasonality(index):
    s = Seasonality(period=12)
    exog = s.in_sample(index)
    assert s.is_dummy
    assert exog.shape == (index.shape[0], 12)
    pd.testing.assert_index_equal(exog.index, index)
    assert np.all(exog.sum(axis=1) == 1.0)
    assert list(exog.columns) == [f"s({i},12)" for i in range(1, 13)]
    expected = np.zeros((index.shape[0], 12))
    for i in range(12):
        expected[i::12, i] = 1.0
    np.testing.assert_equal(expected, np.asarray(exog))

    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63
    ):
        warn = UserWarning
    with pytest_warns(warn):
        fcast = s.out_of_sample(steps=12, index=index)
    assert fcast.iloc[0, len(index) % 12] == 1.0
    assert np.all(fcast.sum(axis=1) == 1)

    s = Seasonality(period=7, initial_period=3)
    exog = s.in_sample(index)
    assert exog.iloc[0, 2] == 1.0
    assert exog.iloc[0].sum() == 1.0
    assert s.initial_period == 3
    with pytest.raises(ValueError, match="initial_period must be in"):
        Seasonality(period=12, initial_period=-3)
    with pytest.raises(ValueError, match="period must be >= 2"):
        Seasonality(period=1)


def test_seasonality_time_index(time_index):
    tt = Seasonality.from_index(time_index)
    assert tt.period == 5

    fcast = tt.out_of_sample(steps=12, index=time_index)
    new_idx = DeterministicTerm._extend_index(time_index, 12)
    pd.testing.assert_index_equal(fcast.index, new_idx)


def test_fourier(index):
    f = Fourier(period=12, order=3)
    terms = f.in_sample(index)
    assert f.order == 3
    assert terms.shape == (index.shape[0], 2 * f.order)
    loc = np.arange(index.shape[0]) / 12
    for i, col in enumerate(terms):
        j = i // 2 + 1
        fn = np.cos if (i % 2) else np.sin
        expected = fn(2 * np.pi * j * loc)
        np.testing.assert_allclose(terms[col], expected, atol=1e-8)
    cols = []
    for i in range(2 * f.order):
        fn = "cos" if (i % 2) else "sin"
        cols.append(f"{fn}({(i // 2) + 1},12)")
    assert list(terms.columns) == cols


def test_index_like():
    idx = np.empty((100, 2))
    with pytest.raises(TypeError, match="index must be a pandas"):
        DeterministicTerm._index_like(idx)


def test_calendar_fourier():
    rs = np.random.RandomState(43437243)
    inc = np.abs(rs.standard_normal(1000))
    inc = np.cumsum(inc)
    inc = 10 * inc / inc[-1]
    offset = (24 * 3600 * inc).astype(np.int64)
    base = pd.Timestamp("2000-1-1")
    index = [base + pd.Timedelta(val, unit="s") for val in offset]
    index = pd.Index(index)

    cf = CalendarFourier("D", 2)
    assert cf.order == 2
    terms = cf.in_sample(index)
    cols = []
    for i in range(2 * cf.order):
        fn = "cos" if (i % 2) else "sin"
        cols.append(f"{fn}({(i // 2) + 1},freq=D)")
    assert list(terms.columns) == cols

    inc = offset / (24 * 3600)
    loc = 2 * np.pi * (inc - np.floor(inc))
    expected = []
    for i in range(4):
        scale = i // 2 + 1
        fn = np.cos if (i % 2) else np.sin
        expected.append(fn(scale * loc))
    expected = np.column_stack(expected)
    np.testing.assert_allclose(expected, terms.values)


def test_calendar_time_trend():
    rs = np.random.RandomState(43437243)
    inc = np.abs(rs.standard_normal(1000))
    inc = np.cumsum(inc)
    inc = 10 * inc / inc[-1]
    offset = (24 * 3600 * inc).astype(np.int64)
    base = pd.Timestamp("2000-1-1")
    index = [base + pd.Timedelta(val, "s") for val in offset]
    index = pd.Index(index)

    ctt = CalendarTimeTrend("D", True, order=3, base_period=base)
    assert ctt.order == 3
    terms = ctt.in_sample(index)
    cols = ["const", "trend", "trend_squared", "trend_cubed"]
    assert list(terms.columns) == cols

    inc = 1 + offset / (24 * 3600)
    expected = [inc**i for i in range(4)]
    expected = np.column_stack(expected)
    np.testing.assert_allclose(expected, terms.values)

    ctt = CalendarTimeTrend("D", True, order=2, base_period=base)
    ctt2 = CalendarTimeTrend.from_string("D", trend="ctt", base_period=base)
    pd.testing.assert_frame_equal(ctt.in_sample(index), ctt2.in_sample(index))

    ct = CalendarTimeTrend("D", True, order=1, base_period=base)
    ct2 = CalendarTimeTrend.from_string("D", trend="ct", base_period=base)
    pd.testing.assert_frame_equal(ct.in_sample(index), ct2.in_sample(index))

    ctttt = CalendarTimeTrend("D", True, order=4, base_period=base)
    assert ctttt.order == 4
    terms = ctttt.in_sample(index)
    cols = ["const", "trend", "trend_squared", "trend_cubed", "trend**4"]
    assert list(terms.columns) == cols


def test_calendar_seasonal_period_w():
    period = "W"
    index = pd.date_range("2000-01-03", freq="h", periods=600)
    cs = CalendarSeasonality("h", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 168] == 1.0

    index = pd.date_range("2000-01-03", freq="B", periods=600)
    cs = CalendarSeasonality("B", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 5] == 1.0

    index = pd.date_range("2000-01-03", freq="D", periods=600)
    cs = CalendarSeasonality("D", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 7] == 1.0


def test_calendar_seasonal_period_d():
    period = "D"
    index = pd.date_range("2000-01-03", freq="h", periods=600)
    cs = CalendarSeasonality("h", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 24] == 1.0


def test_calendar_seasonal_period_q():
    period = "Q"
    index = pd.date_range("2000-01-01", freq=MONTH_END, periods=600)
    cs = CalendarSeasonality(MONTH_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 3] == 1.0


def test_calendar_seasonal_period_a():
    period = "Y"
    index = pd.date_range("2000-01-01", freq=MONTH_END, periods=600)
    cs = CalendarSeasonality(MONTH_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 12] == 1.0

    cs = CalendarSeasonality(QUARTER_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(axis=1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, (i % 12) // 3] == 1.0


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("seasonal", [True, False])
@pytest.mark.parametrize("fourier", [0, 1])
@pytest.mark.parametrize("period", [None, 10])
@pytest.mark.parametrize("drop", [True, False])
def test_deterministic_process(
    time_index, constant, order, seasonal, fourier, period, drop
):
    if seasonal and fourier:
        return
    dp = DeterministicProcess(
        time_index,
        constant=constant,
        order=order,
        seasonal=seasonal,
        fourier=fourier,
        period=period,
        drop=drop,
    )
    terms = dp.in_sample()
    pd.testing.assert_index_equal(terms.index, time_index)
    terms = dp.out_of_sample(23)
    assert isinstance(terms, pd.DataFrame)


def test_index_and_terms_properties(time_index):
    # .index must equal the index the process was constructed from, and
    # .terms must equal the list of DeterministicTerm objects that
    # determine it -- both for terms supplied explicitly via
    # additional_terms and for the constant/order/seasonal convenience
    # arguments that build the default term set internally.
    tt = TimeTrend(constant=True, order=2)
    seas = Seasonality(period=5)
    dp = DeterministicProcess(time_index, additional_terms=[tt, seas])
    pd.testing.assert_index_equal(dp.index, time_index)
    assert dp.terms == [tt, seas]

    idx2 = pd.RangeIndex(0, 50)
    dp2 = DeterministicProcess(idx2, constant=True, order=1, seasonal=True, period=4)
    pd.testing.assert_index_equal(dp2.index, idx2)
    assert dp2.terms == [TimeTrend(True, 1), Seasonality(4)]

    # no terms at all is a valid (if useless) configuration
    dp3 = DeterministicProcess(idx2)
    assert dp3.terms == []


@pytest.mark.parametrize("drop", [True, False])
def test_out_of_sample_without_in_sample(drop):
    # GH: out_of_sample()/range() on a fresh DeterministicProcess used to raise
    # a bare ``AssertionError`` when drop=False, because the lazy in_sample()
    # call that populates _retain_cols was gated on self._drop.
    idx = pd.RangeIndex(0, 10)
    fresh = DeterministicProcess(idx, constant=True, order=1, drop=drop)
    primed = DeterministicProcess(idx, constant=True, order=1, drop=drop)
    primed.in_sample()
    pd.testing.assert_frame_equal(fresh.out_of_sample(5), primed.out_of_sample(5))
    fresh2 = DeterministicProcess(idx, constant=True, order=1, drop=drop)
    pd.testing.assert_frame_equal(fresh2.range(10, 15), primed.range(10, 15))


@pytest.mark.parametrize(
    "index, expected",
    [
        (pd.RangeIndex(10), pd.RangeIndex(10, 13)),
        (pd.Index(np.arange(10)), pd.Index(np.arange(10, 13))),
        (
            pd.date_range("2000-01-01", periods=10, freq="MS"),
            pd.date_range("2000-11-01", periods=3, freq="MS"),
        ),
        (
            pd.period_range("2000-01", periods=10, freq="M"),
            pd.period_range("2000-11", periods=3, freq="M"),
        ),
    ],
)
def test_out_of_sample_without_terms(index, expected):
    # A process without terms returned nobs rows on the in-sample index
    dp = DeterministicProcess(index)
    fcast = dp.out_of_sample(3)
    assert fcast.shape == (3, 0)
    pd.testing.assert_index_equal(fcast.index, expected)
    # The forecast index of a process with terms
    with_terms = DeterministicProcess(index, constant=True).out_of_sample(3)
    pd.testing.assert_index_equal(fcast.index, with_terms.index)

    forecast_index = pd.RangeIndex(20, 23)
    fcast = dp.out_of_sample(3, forecast_index=forecast_index)
    assert fcast.shape == (3, 0)
    pd.testing.assert_index_equal(fcast.index, forecast_index)
    with pytest.raises(ValueError, match="The number of values in forecast_"):
        dp.out_of_sample(4, forecast_index=forecast_index)


def test_range_without_terms():
    # The out-of-sample part used to have the rows of the in-sample index
    dp = DeterministicProcess(pd.RangeIndex(10))
    terms = dp.range(8, 12)
    assert terms.shape == (5, 0)
    pd.testing.assert_index_equal(terms.index, pd.RangeIndex(8, 13), exact=False)


def test_deterministic_process_errors(time_index):
    with pytest.raises(ValueError, match="seasonal and fourier"):
        DeterministicProcess(time_index, seasonal=True, fourier=2, period=5)
    with pytest.raises(TypeError, match="All additional terms"):
        DeterministicProcess(time_index, seasonal=True, additional_terms=[1])


def test_range_error():
    idx = pd.Index([0, 1, 1, 2, 3, 5, 8, 13])
    dp = DeterministicProcess(idx, constant=True, order=2, seasonal=True, period=2)
    with pytest.raises(TypeError, match="The index in the deterministic"):
        dp.range(0, 12)


def test_range_index_basic():
    idx = pd.date_range("2000-1-1", freq=MONTH_END, periods=120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True)
    dp.range("2001-1-1", "2008-1-1")
    dp.range("2001-1-1", "2015-1-1")
    dp.range("2013-1-1", "2008-1-1")
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(130, 150)
    with pytest.raises(ValueError):
        dp.range("1990-1-1", "2010-1-1")

    idx = pd.period_range("2000-1-1", freq="M", periods=120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True)
    dp.range("2001-1-1", "2008-1-1")
    dp.range("2001-1-1", "2015-1-1")
    dp.range("2013-1-1", "2008-1-1")
    with pytest.raises(ValueError, match="start must be non-negative"):
        dp.range(-7, 200)

    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(130, 150)

    idx = pd.RangeIndex(0, 120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True, period=12)
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(120, 150)
    dp.range(130, 150)
    with pytest.raises(ValueError):
        dp.range(-10, 0)


def test_range_casting():
    idx = np.arange(120).astype(np.int64)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True, period=12)
    idx = pd.RangeIndex(0, 120)
    dp2 = DeterministicProcess(idx, constant=True, order=1, seasonal=True, period=12)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())
    pd.testing.assert_frame_equal(dp.range(100, 150), dp2.range(100, 150))


def test_non_unit_range():
    idx = pd.RangeIndex(0, 700, 7)
    dp = DeterministicProcess(idx, constant=True)
    with pytest.raises(ValueError, match="The step of the index is not 1"):
        dp.range(11, 900)


@pytest.mark.parametrize("index", [pd.RangeIndex(0, 20, 2), pd.RangeIndex(1, 31, 3)])
@pytest.mark.parametrize(
    "first, last",
    [(2, 5), (10, 13), (11, 13), (8, 12)],
    ids=["in-sample", "next-out-of-sample", "later-out-of-sample", "both"],
)
def test_range_non_unit_step(index, first, last):
    # The step of the index was ignored. The labels start, start + step, ...,
    # stop are at positions first, ..., last.
    start = index[0] + first * index.step
    stop = index[0] + last * index.step
    kwargs = {"constant": True, "order": 1, "seasonal": True, "period": 3}
    dp = DeterministicProcess(index, **kwargs)
    terms = dp.range(start, stop)
    expected_index = pd.RangeIndex(start, stop + 1, index.step)
    pd.testing.assert_index_equal(terms.index, expected_index, exact=False)
    full = pd.concat([dp.in_sample(), dp.out_of_sample(4)], axis=0)
    pd.testing.assert_frame_equal(terms, full.iloc[first : last + 1])
    trend = np.arange(first + 1.0, last + 2.0)
    np.testing.assert_allclose(terms["trend"], trend, rtol=1e-15)


# Integer indexes with a constant step. int64, as a RangeIndex, since np.arange
# gives int32 where the default integer has 32 bits (Pyodide).
INT_INDEXES = {
    "step-2": np.arange(0, 20, 2, dtype=np.int64),
    "step-3-from-1": np.arange(1, 31, 3, dtype=np.int64),
    "step-1-from-5": np.arange(5, 15, dtype=np.int64),
    "step-1": np.arange(0, 10, dtype=np.int64),
}


def _equal_range_index(values):
    step = int(values[1] - values[0])
    return pd.RangeIndex(values[0], values[-1] + step, step), step


@pytest.mark.parametrize("values", list(INT_INDEXES.values()), ids=list(INT_INDEXES))
@pytest.mark.parametrize(
    "first, last",
    [(2, 5), (10, 13), (11, 13), (8, 12)],
    ids=["in-sample", "next-out-of-sample", "later-out-of-sample", "both"],
)
def test_range_int_index_constant_step(values, first, last):
    # An integer Index with a constant step is extended like the RangeIndex
    # with the same values. Only one that starts at 0 with step 1 could be
    # extended, and range raised a TypeError for the others.
    range_index, step = _equal_range_index(values)
    start = int(values[0]) + first * step
    stop = int(values[0]) + last * step
    kwargs = {"constant": True, "order": 1, "seasonal": True, "period": 3}
    dp = DeterministicProcess(pd.Index(values), **kwargs)
    expected = DeterministicProcess(range_index, **kwargs)
    pd.testing.assert_frame_equal(dp.in_sample(), expected.in_sample())
    terms = dp.range(start, stop)
    expected_terms = expected.range(start, stop)
    pd.testing.assert_frame_equal(terms, expected_terms, check_index_type=False)
    np.testing.assert_array_equal(terms.index, np.arange(start, stop + 1, step))
    trend = np.arange(first + 1.0, last + 2.0)
    np.testing.assert_allclose(terms["trend"], trend, rtol=1e-15)


@pytest.mark.parametrize(
    "values",
    list(INT_INDEXES.values()) + [np.arange(20, 0, -2, dtype=np.int64)],
    ids=list(INT_INDEXES) + ["step-minus-2"],
)
@pytest.mark.parametrize(
    "term", [TimeTrend(True, 2), Seasonality(3), Fourier(3, 1)], ids=str
)
def test_term_out_of_sample_int_index_constant_step(term, values):
    # The labels of the forecasts continue the values with their step, as the
    # RangeIndex with the same values. The forecasts used to have labels
    # counted from the number of observations, with a warning, unless the
    # step was 1.
    range_index, step = _equal_range_index(values)
    with pytest_warns(None):
        fcast = term.out_of_sample(5, pd.Index(values))
    expected = term.out_of_sample(5, range_index)
    pd.testing.assert_frame_equal(fcast, expected, check_index_type=False)
    np.testing.assert_array_equal(fcast.index, values[-1] + step * np.arange(1, 6))
    # An integer Index, as before for step 1, and not a RangeIndex (an
    # Int64Index, a subclass of Index, on pandas < 2)
    assert isinstance(fcast.index, pd.Index)
    assert not isinstance(fcast.index, pd.RangeIndex)
    assert fcast.index.dtype == np.int64
    dp = DeterministicProcess(pd.Index(values), additional_terms=[term])
    expected = DeterministicProcess(range_index, additional_terms=[term])
    pd.testing.assert_frame_equal(
        dp.out_of_sample(5), expected.out_of_sample(5), check_index_type=False
    )


def test_int_index_one_value():
    # One value has step 1, as the RangeIndex with the same value
    index = pd.Index(np.array([5], dtype=np.int64))
    fcast = TimeTrend(True, 1).out_of_sample(3, index)
    np.testing.assert_array_equal(fcast.index, [6, 7, 8])
    np.testing.assert_allclose(fcast["trend"], [2.0, 3.0, 4.0], rtol=1e-15)
    dp = DeterministicProcess(index, constant=True, order=1)
    expected = DeterministicProcess(pd.RangeIndex(5, 6), constant=True, order=1)
    pd.testing.assert_frame_equal(
        dp.range(5, 7), expected.range(5, 7), check_index_type=False
    )


@pytest.mark.parametrize(
    "values",
    [[0, 1, 3, 4], [0, 2, 4, 5], [3, 3, 3, 3]],
    ids=["gap", "step-change", "step-0"],
)
def test_int_index_irregular_not_extended(values):
    # An integer Index without a constant step still cannot be extended
    index = pd.Index(np.array(values, dtype=np.int64))
    dp = DeterministicProcess(index, constant=True, order=1)
    np.testing.assert_allclose(dp.in_sample()["trend"], np.arange(1.0, 5.0))
    with pytest.raises(TypeError, match="does not support extension"):
        dp.range(0, 6)
    with pytest.warns(UserWarning, match="integer Indexes with a constant step"):
        fcast = TimeTrend(True, 1).out_of_sample(2, index)
    pd.testing.assert_index_equal(fcast.index, pd.RangeIndex(5, 7))


def test_additional_terms(time_index):
    add_terms = [TimeTrend(True, order=1)]
    dp = DeterministicProcess(time_index, additional_terms=add_terms)
    dp2 = DeterministicProcess(time_index, constant=True, order=1)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())
    with pytest.raises(ValueError, match="One or more terms in additional_terms"):
        DeterministicProcess(time_index, additional_terms=add_terms + add_terms)
    with pytest.raises(ValueError, match="One or more terms in additional_terms"):
        DeterministicProcess(
            time_index, constant=True, order=1, additional_terms=add_terms
        )


def test_drop_two_consants(time_index):
    tt = TimeTrend(constant=True, order=1)
    dp = DeterministicProcess(
        time_index, constant=True, additional_terms=[tt], drop=True
    )
    assert dp.in_sample().shape[1] == 2
    dp2 = DeterministicProcess(time_index, additional_terms=[tt], drop=True)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 200),
        pd.Index(np.arange(200)),
        pd.date_range("2000-1-1", freq="MS", periods=200),
        pd.period_range("2000-1-1", freq="M", periods=200),
    ],
)
def test_determintic_term_equiv(index):
    base = DeterministicProcess(pd.RangeIndex(0, 200), constant=True, order=2)
    dp = DeterministicProcess(index, constant=True, order=2)
    np.testing.assert_array_equal(base.in_sample(), dp.in_sample())
    np.testing.assert_array_equal(base.out_of_sample(37), dp.out_of_sample(37))
    np.testing.assert_array_equal(base.range(200, 237), dp.range(200, 237))
    np.testing.assert_array_equal(base.range(50, 150), dp.range(50, 150))
    np.testing.assert_array_equal(base.range(50, 250), dp.range(50, 250))


class DummyTerm(DeterministicTerm):
    @property
    def _eq_attr(self) -> tuple[Hashable, ...]:
        return ("Dummy",)

    def __str__(self) -> str:
        return "Dummy"

    columns = [
        "const1",
        "const2",
        "trend1",
        "trend2",
        "normal1",
        "normal2",
        "dummy1_1",
        "dummy1_2",
        "always_drop1",
        "always_drop2",
        "dummy2_1",
        "dummy2_2",
    ]

    def in_sample(self, index: pd.Index) -> pd.DataFrame:
        nobs = index.shape[0]
        terms = np.empty((index.shape[0], 12))
        rs = np.random.RandomState(43437243)
        for i in range(0, 12, 2):
            if i == 0:
                value = 1
            elif i == 2:
                value = np.arange(nobs)
            elif i == 4:
                value = rs.standard_normal(nobs)
            elif i == 6:
                value = np.zeros(nobs)
                value[::2] = 1
            elif i == 8:
                value = 0
            else:  # elif i == 8:
                value = np.zeros(nobs)
                value[1::2] = 1
            terms[:, i] = terms[:, i + 1] = value
        return pd.DataFrame(terms, columns=self.columns, index=index)

    def out_of_sample(
        self,
        steps: int,
        index: pd.Index,
        forecast_index: pd.Index = None,
    ) -> pd.DataFrame:
        fcast_index = self._extend_index(index, steps, forecast_index)
        rs = np.random.RandomState(4343)
        terms = rs.standard_normal((steps, 12))

        return pd.DataFrame(terms, columns=self.columns, index=fcast_index)


def test_drop():
    index = pd.RangeIndex(0, 200)
    dummy = DummyTerm()
    str(dummy)
    assert dummy != TimeTrend()
    dp = DeterministicProcess(index, additional_terms=[dummy], drop=True)
    in_samp = dp.in_sample()
    assert in_samp.shape == (200, 4)
    oos = dp.out_of_sample(37)
    assert oos.shape == (37, 4)
    assert list(oos.columns) == list(in_samp.columns)
    valid = ("const", "trend", "dummy", "normal")
    for valid_col in valid:
        assert sum([1 for col in oos if valid_col in col]) == 1


def test_calendar_time_trend_period_index():
    # For periods, the time trend counts the periods since base_period, which
    # is 1, and no part of a period has elapsed at its start
    index = pd.period_range("2000-01", periods=24, freq="M")
    ct = CalendarTimeTrend(MONTH_END, True, order=2, base_period="2000-01-01")
    terms = ct.in_sample(index)
    t = np.arange(1.0, 25.0)
    np.testing.assert_allclose(terms, np.column_stack([t**0, t, t**2]), rtol=1e-15)
    pd.testing.assert_index_equal(terms.index, index)
    fcast = ct.out_of_sample(3, index)
    t = np.arange(25.0, 28.0)
    np.testing.assert_allclose(fcast, np.column_stack([t**0, t, t**2]), rtol=1e-15)
    expected_index = pd.period_range("2002-01", periods=3, freq="M")
    pd.testing.assert_index_equal(fcast.index, expected_index)


def test_calendar_time_trend_datetime_index():
    # A DatetimeIndex was converted to a PeriodIndex, so the terms did not
    # align with the other terms of a DeterministicProcess
    index = pd.date_range("2000-01-31", periods=27, freq=MONTH_END)
    # The months since base_period plus the part of the month elapsed at the
    # time stamp, which is the start of the last day
    trend = np.arange(1.0, 28.0) + (index.day - 1) / index.days_in_month
    trend = np.asarray(trend)
    ct = CalendarTimeTrend(MONTH_END, True, order=1, base_period="2000-01-01")
    terms = ct.in_sample(index[:24])
    pd.testing.assert_index_equal(terms.index, index[:24])
    np.testing.assert_allclose(terms["trend"], trend[:24], rtol=1e-15)
    fcast = ct.out_of_sample(3, index[:24])
    pd.testing.assert_index_equal(fcast.index, index[24:])
    np.testing.assert_allclose(fcast["trend"], trend[24:], rtol=1e-15)

    dp = DeterministicProcess(index[:24], seasonal=True, additional_terms=[ct])
    seasonal = Seasonality(12).in_sample(index).iloc[:, 1:]
    expected = pd.concat(
        [seasonal, pd.DataFrame({"const": 1.0, "trend": trend}, index=index)],
        axis=1,
    )
    pd.testing.assert_frame_equal(dp.in_sample(), expected.iloc[:24])
    pd.testing.assert_frame_equal(dp.out_of_sample(3), expected.iloc[24:])


@pytest.mark.parametrize(
    "freq, term_freq, trend",
    [
        # Each quarter has three months
        ("Q", MONTH_END, [1, 4, 7, 10, 13, 16]),
        # The first two quarters of 2000 have 91 days each
        ("M", QUARTER_END, [1, 1 + 31 / 91, 1 + 60 / 91, 2, 2 + 30 / 91, 2 + 61 / 91]),
        # The days from Jan 29, 2000; January has 31 days, February 29
        (
            "D",
            MONTH_END,
            [1 + 28 / 31, 1 + 29 / 31, 1 + 30 / 31, 2, 2 + 1 / 29, 2 + 2 / 29],
        ),
    ],
)
def test_calendar_time_trend_period_index_other_freq(freq, term_freq, trend):
    # The periods were counted in units of the freq of the index, not of the
    # freq of the term
    start = "2000-01-29" if freq == "D" else "2000-01-01"
    index = pd.period_range(start, periods=6, freq=freq)
    ct = CalendarTimeTrend(term_freq, True, order=1, base_period="2000-01-01")
    terms = ct.in_sample(index[:4])
    pd.testing.assert_index_equal(terms.index, index[:4])
    np.testing.assert_allclose(terms["trend"], trend[:4], rtol=1e-14)
    fcast = ct.out_of_sample(2, index[:4])
    pd.testing.assert_index_equal(fcast.index, index[4:])
    np.testing.assert_allclose(fcast["trend"], trend[4:], rtol=1e-14)


def test_calendar_fourier_period_index():
    # tau is the fraction of the year that has elapsed at the start of each
    # month
    index = pd.period_range("2000-01", periods=24, freq="M")
    cf = CalendarFourier(YEAR_END, 2)
    terms = cf.in_sample(index)
    tau = []
    for ts in index.to_timestamp():
        year_start = pd.Timestamp(ts.year, 1, 1)
        year_length = pd.Timestamp(ts.year + 1, 1, 1) - year_start
        tau.append((ts - year_start) / year_length)
    tau = 2 * np.pi * np.array(tau)
    expected = np.column_stack(
        [np.sin(tau), np.cos(tau), np.sin(2 * tau), np.cos(2 * tau)]
    )
    # atol: some of the values are close to zero
    np.testing.assert_allclose(terms, expected, rtol=1e-12, atol=1e-14)
    pd.testing.assert_index_equal(terms.index, index)
    fcast = cf.out_of_sample(12, index[:12])
    np.testing.assert_allclose(fcast, expected[12:], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("trend, constant", [("n", False), ("c", True)])
def test_calendar_time_trend_from_string_without_trend(trend, constant):
    index = pd.date_range("2000-01-01", periods=10, freq="D")
    ct = CalendarTimeTrend.from_string("D", trend)
    assert ct == CalendarTimeTrend("D", constant, 0)
    terms = ct.in_sample(index)
    assert terms.shape == (10, int(constant))
    np.testing.assert_allclose(terms, np.ones((10, int(constant))), rtol=1e-15)


@pytest.mark.parametrize(
    "index", [pd.Index(list("abcdef")), pd.Index(np.arange(6) + 0.5)]
)
def test_deterministic_process_non_extendable_index(index):
    # The terms of any index can be computed in-sample, but the index cannot
    # be extended
    dp = DeterministicProcess(index, constant=True, order=1)
    terms = dp.in_sample()
    pd.testing.assert_index_equal(terms.index, index)
    expected = np.column_stack([np.ones(6), np.arange(1.0, 7.0)])
    np.testing.assert_allclose(terms, expected, rtol=1e-15)
    with pytest.raises(TypeError, match="does not support extension"):
        dp.range(0, 7)


def test_drop_full_rank_keeps_all_columns():
    # The terms have full column rank, so drop removes none of them
    index = pd.RangeIndex(50)
    kwargs = {"constant": True, "order": 3, "fourier": 2, "period": 12}
    full = DeterministicProcess(index, **kwargs).in_sample()
    assert full.shape == (50, 8)
    assert np.linalg.matrix_rank(full.to_numpy()) == 8
    dropped = DeterministicProcess(index, drop=True, **kwargs).in_sample()
    pd.testing.assert_frame_equal(dropped, full)


INDEX_FORMS = {
    "list": list,
    "tuple": tuple,
    "ndarray": lambda idx: idx.to_numpy(),
    "series": pd.Series,
}


@pytest.mark.parametrize("form", list(INDEX_FORMS))
@pytest.mark.parametrize(
    "term",
    [
        TimeTrend(True, 2),
        Seasonality(12),
        Fourier(12, 2),
        CalendarFourier(YEAR_END, 2),
        CalendarSeasonality(MONTH_END, "Y"),
    ],
    ids=str,
)
def test_term_index_forms(term, form):
    # An index_like is converted to an index
    index = pd.date_range("2000-01-31", periods=30, freq=MONTH_END)
    terms = term.in_sample(INDEX_FORMS[form](index))
    np.testing.assert_allclose(terms, term.in_sample(index), rtol=1e-15)
    assert terms.index.equals(index)


@pytest.mark.parametrize("form", list(INDEX_FORMS))
def test_deterministic_process_index_forms(form):
    index = pd.date_range("2000-01-31", periods=30, freq=MONTH_END)
    kwargs = {"constant": True, "order": 1, "seasonal": True}
    expected = DeterministicProcess(index, **kwargs)
    dp = DeterministicProcess(INDEX_FORMS[form](index), **kwargs)
    np.testing.assert_allclose(dp.in_sample(), expected.in_sample(), rtol=1e-15)
    assert dp.in_sample().index.equals(index)
    # The frequency is inferred, so the index can be extended
    pd.testing.assert_frame_equal(
        dp.range(25, 34), expected.range(25, 34), check_freq=False
    )


@pytest.mark.parametrize("form", list(INDEX_FORMS) + ["no-freq"])
def test_deterministic_process_out_of_sample_inferred_freq(form):
    # range used the inferred freq, but out_of_sample warned and gave the
    # forecasts integer labels, and calendar terms raised
    index = pd.date_range("2000-01-31", periods=30, freq=MONTH_END)
    if form == "no-freq":
        value = pd.DatetimeIndex(list(index))
    else:
        value = INDEX_FORMS[form](index)
    kwargs = {
        "constant": True,
        "order": 1,
        "seasonal": True,
        "additional_terms": [CalendarFourier(YEAR_END, 1)],
    }
    dp = DeterministicProcess(value, **kwargs)
    assert dp.index.freq is None
    with pytest_warns(None):
        fcast = dp.out_of_sample(3)
    fcast_index = pd.date_range("2002-07-31", periods=3, freq=MONTH_END)
    pd.testing.assert_index_equal(fcast.index, fcast_index)
    expected = DeterministicProcess(index, **kwargs).out_of_sample(3)
    pd.testing.assert_frame_equal(fcast, expected)
    np.testing.assert_allclose(fcast["trend"], [31.0, 32.0, 33.0], rtol=1e-15)
    pd.testing.assert_frame_equal(fcast, dp.range(30, 32), check_freq=False)
    # A process without terms
    with pytest_warns(None):
        empty = DeterministicProcess(value).out_of_sample(3)
    pd.testing.assert_index_equal(empty.index, fcast_index)


def test_deterministic_process_out_of_sample_irregular_dates():
    # No freq can be inferred, so the forecasts have integer labels
    index = pd.date_range("2000-01-31", periods=30, freq=MONTH_END).delete(5)
    dp = DeterministicProcess(index, constant=True, order=1)
    with pytest_warns(UserWarning):
        fcast = dp.out_of_sample(3)
    assert is_int_index(fcast.index)
    np.testing.assert_allclose(fcast["trend"], [30.0, 31.0, 32.0], rtol=1e-15)


NO_FREQ_FORMS = {
    "DatetimeIndex": lambda idx: pd.DatetimeIndex(list(idx)),
    "list": list,
}


@pytest.mark.parametrize("form", list(NO_FREQ_FORMS))
@pytest.mark.parametrize("freq", [MONTH_END, "D", "B", "W-WED"])
@pytest.mark.parametrize(
    "term",
    [
        TimeTrend(True, 2),
        Seasonality(12),
        Fourier(12, 2),
        CalendarTimeTrend(MONTH_END, True, 2),
        CalendarFourier(YEAR_END, 2),
        CalendarSeasonality(MONTH_END, "Y"),
    ],
    ids=str,
)
def test_term_out_of_sample_inferred_freq(term, freq, form):
    # A DatetimeIndex without a freq is extended with the inferred freq, as
    # the same index with the freq. The forecasts used to have integer
    # labels, with a warning, and calendar terms raised.
    index = pd.date_range("2000-01-31", periods=30, freq=freq)
    value = NO_FREQ_FORMS[form](index)
    assert pd.Index(value).freq is None
    with pytest_warns(None):
        fcast = term.out_of_sample(7, value)
    pd.testing.assert_frame_equal(fcast, term.out_of_sample(7, index))
    # The forecasts are those of the dates that continue the index
    fcast_index = pd.date_range(index[-1], periods=8, freq=freq)[1:]
    explicit = term.out_of_sample(7, value, forecast_index=fcast_index)
    pd.testing.assert_frame_equal(fcast, explicit)


@pytest.mark.parametrize(
    "term", [TimeTrend(True, 1), Seasonality(12), Fourier(12, 1)], ids=str
)
def test_term_out_of_sample_irregular_dates(term):
    # No freq can be inferred, so the forecasts have integer labels
    index = pd.date_range("2000-01-31", periods=30, freq=MONTH_END).delete(5)
    with pytest.warns(UserWarning, match="with a frequency set or inferred"):
        fcast = term.out_of_sample(3, index)
    pd.testing.assert_index_equal(fcast.index, pd.RangeIndex(30, 33))


def test_apply():
    # apply keeps the terms and uses the new index
    index = pd.RangeIndex(40)
    fourier = Fourier(period=7, order=1)
    kwargs = {"constant": True, "order": 1, "seasonal": True, "period": 4}
    dp = DeterministicProcess(index, additional_terms=[fourier], drop=True, **kwargs)
    new_index = pd.date_range("2000-01-01", periods=12, freq="D")
    applied = dp.apply(new_index)
    assert applied.terms == dp.terms
    terms = applied.in_sample()
    pd.testing.assert_index_equal(terms.index, new_index)
    np.testing.assert_allclose(terms, dp.in_sample().iloc[:12], rtol=1e-15)


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2000-01-01", periods=12, freq="MS"),
        pd.period_range("2000-01", periods=12, freq="M"),
    ],
    ids=["datetime", "period"],
)
@pytest.mark.parametrize(
    "to_loc", [int, np.int64, np.array], ids=["int", "int64", "0d-array"]
)
def test_range_date_index_int_like_locations(index, to_loc):
    # A 0-d integer array is the location of an observation, as an int is.
    # It raised a TypeError.
    dp = DeterministicProcess(index, constant=True, order=1)
    terms = dp.range(to_loc(10), to_loc(14))
    # The trend counts the observations from 1
    expected = np.column_stack([np.ones(5), np.arange(11.0, 16.0)])
    np.testing.assert_allclose(np.asarray(terms), expected, rtol=1e-15)


@pytest.mark.parametrize("value", [True, np.True_], ids=["True", "np.True_"])
@pytest.mark.parametrize("arg", ["start", "stop"])
def test_range_date_index_bool_location(value, arg):
    # A bool is not the location of an observation
    index = pd.date_range("2000-01-01", periods=12, freq="MS")
    dp = DeterministicProcess(index, constant=True, order=1)
    kwargs = {"start": 0, "stop": 5, arg: value}
    with pytest.raises(TypeError):
        dp.range(**kwargs)
