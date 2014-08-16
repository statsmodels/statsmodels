import numpy as np
from pandas import Series
from pandas import date_range
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
import numpy.testing as npt
from statsmodels.tools.testing import assert_equal

def test_pandas_nodates_index():
    from statsmodels.datasets import sunspots
    y = sunspots.load_pandas().data.SUNACTIVITY
    npt.assert_raises(ValueError, TimeSeriesModel, y)

def test_predict_freq():
    # test that predicted dates have same frequency
    x = np.arange(1,36.)

    # there's a bug in pandas up to 0.10.2 for YearBegin
    #dates = date_range("1972-4-1", "2007-4-1", freq="AS-APR")
    dates = date_range("1972-4-30", "2006-4-30", freq="A-APR")
    series = Series(x, index=dates)
    model = TimeSeriesModel(series)
    #npt.assert_(model.data.freq == "AS-APR")
    npt.assert_(model.data.freq == "A-APR")

    start = model._get_predict_start("2006-4-30")
    end = model._get_predict_end("2016-4-30")
    model._make_predict_dates()

    predict_dates = model.data.predict_dates

    #expected_dates = date_range("2006-12-31", "2016-12-31",
    #                            freq="AS-APR")
    expected_dates = date_range("2006-4-30", "2016-4-30", freq="A-APR")
    assert_equal(predict_dates, expected_dates)
    #ptesting.assert_series_equal(predict_dates, expected_dates)


def test_keyerror_start_date():
    x = np.arange(1,36.)

    from pandas import date_range

    # there's a bug in pandas up to 0.10.2 for YearBegin
    #dates = date_range("1972-4-1", "2007-4-1", freq="AS-APR")
    dates = date_range("1972-4-30", "2006-4-30", freq="A-APR")
    series = Series(x, index=dates)
    model = TimeSeriesModel(series)

    npt.assert_raises(ValueError, model._get_predict_start, "1970-4-30")

def test_period_index():
    # test 1285
    from pandas import PeriodIndex, TimeSeries
    dates = PeriodIndex(start="1/1/1990", periods=20, freq="M")
    x = np.arange(1, 21.)

    model = TimeSeriesModel(Series(x, index=dates))
    npt.assert_(model.data.freq == "M")
    model = TimeSeriesModel(TimeSeries(x, index=dates))
    npt.assert_(model.data.freq == "M")
