import numpy as np
from pandas import Series
from pandas.util import testing as ptesting
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.base.datetools import dates_from_range
import numpy.testing as npt

try:
    from pandas import DatetimeIndex
    _pandas_08x = True
except ImportError:
    _pandas_08x = False

def test_pandas_nodates_index():
    from statsmodels.datasets import sunspots
    y = sunspots.load_pandas().data.SUNACTIVITY
    npt.assert_raises(ValueError, TimeSeriesModel, y)

def test_predict_freq():
    # test that predicted dates have same frequency
    x = np.arange(1,36.)

    if _pandas_08x:
        from pandas import date_range

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
        npt.assert_equal(predict_dates, expected_dates)
        #ptesting.assert_series_equal(predict_dates, expected_dates)

    else:
        from pandas import DateRange, datetools
        dates = DateRange("1972-1-1", "2007-1-1", offset=datetools.yearEnd)
        series = Series(x, index=dates)
        model = TimeSeriesModel(series)
        npt.assert_(model.data.freq == "A")

        start = model._get_predict_start("2006-12-31")
        end = model._get_predict_end("2016-12-31")
        model._make_predict_dates()

        predict_dates = model.data.predict_dates

        expected_dates = DateRange("2006-12-31", "2016-12-31",
                                    offset=datetools.yearEnd)
        npt.assert_array_equal(predict_dates, expected_dates)

def test_keyerror_start_date():
    x = np.arange(1,36.)

    if _pandas_08x:
        from pandas import date_range

        # there's a bug in pandas up to 0.10.2 for YearBegin
        #dates = date_range("1972-4-1", "2007-4-1", freq="AS-APR")
        dates = date_range("1972-4-30", "2006-4-30", freq="A-APR")
        series = Series(x, index=dates)
        model = TimeSeriesModel(series)
    else:
        from pandas import DateRange, datetools
        dates = DateRange("1972-1-1", "2007-1-1", offset=datetools.yearEnd)
        series = Series(x, index=dates)
        model = TimeSeriesModel(series)

    npt.assert_raises(ValueError, model._get_predict_start, "1970-4-30")
