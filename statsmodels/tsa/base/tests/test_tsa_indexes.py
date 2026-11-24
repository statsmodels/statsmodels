"""
Test index support in time series models

1. Test support for passing / constructing the underlying index in __init__
2. Test wrapping of output using the underlying index
3. Test wrapping of prediction / forecasting using the underlying index or
   extensions of it.

Author: Chad Fulton
License: BSD-3
"""

from statsmodels.compat.pandas import (
    MONTH_END,
    PD_LT_2,
    PD_LT_2_2_0,
    QUARTER_END,
    YEAR_END,
    is_int_index,
)

import contextlib
import datetime as dt
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import warns_or_none
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

nobs = 5
base_dta = np.arange(nobs)
dta = [
    base_dta.tolist(),
    base_dta,
    pd.Series(base_dta),
    pd.DataFrame(base_dta),
]

TWO_QE_DEC = "2Q-DEC" if PD_LT_2_2_0 else "2QE-DEC"
base_date_indexes = [
    # (usual candidates)
    pd.date_range(start="1950-01-01", periods=nobs, freq="D"),
    pd.date_range(start="1950-01-01", periods=nobs, freq="W"),
    pd.date_range(start="1950-01-01", periods=nobs, freq="MS"),
    pd.date_range(start="1950-01-01", periods=nobs, freq="QS"),
    pd.date_range(start="1950-01-01", periods=nobs, freq=YEAR_END),
    # (some more complicated frequencies)
    pd.date_range(start="1950-01-01", periods=nobs, freq=TWO_QE_DEC),
    pd.date_range(start="1950-01-01", periods=nobs, freq="2QS"),
    pd.date_range(start="1950-01-01", periods=nobs, freq="5s"),
    pd.date_range(start="1950-01-01", periods=nobs, freq="1D10min"),
]

# Note: we separate datetime indexes and period indexes because the
# date coercion does not handle string versions of PeriodIndex objects
# most of the time.
base_period_indexes = [
    pd.period_range(start="1950-01-01", periods=nobs, freq="D"),
    pd.period_range(start="1950-01-01", periods=nobs, freq="W"),
    pd.period_range(start="1950-01-01", periods=nobs, freq="M"),
    pd.period_range(start="1950-01-01", periods=nobs, freq="Q"),
    pd.period_range(start="1950-01-01", periods=nobs, freq="Y"),
]
try:
    # Only later versions of pandas support these
    base_period_indexes += [
        pd.period_range(start="1950-01-01", periods=nobs, freq="2Q"),
        pd.period_range(start="1950-01-01", periods=nobs, freq="5s"),
        pd.period_range(start="1950-01-01", periods=nobs, freq="1D10min"),
    ]
except AttributeError:
    pass

date_indexes = [(x, None) for x in base_date_indexes]
period_indexes = [(x, None) for x in base_period_indexes]

numpy_datestr_indexes = [(x.map(str), x.freq) for x in base_date_indexes]
list_datestr_indexes = [(x.tolist(), y) for x, y in numpy_datestr_indexes]
series_datestr_indexes = [(pd.Series(x), y) for x, y in list_datestr_indexes]

numpy_datetime_indexes = [
    (pd.to_datetime(x).to_pydatetime(), x.freq) for x in base_date_indexes
]
list_datetime_indexes = [(x.tolist(), y) for x, y in numpy_datetime_indexes]
series_datetime_indexes = [
    (pd.Series(x, dtype=object), y) for x, y in list_datetime_indexes
]

series_timestamp_indexes = [(pd.Series(x), x.freq) for x in base_date_indexes]

# Supported increment indexes
supported_increment_indexes = [
    (pd.Index(np.arange(nobs)), None),
    (pd.RangeIndex(start=0, stop=nobs, step=1), None),
    (pd.RangeIndex(start=-5, stop=nobs - 5, step=1), None),
    (pd.RangeIndex(start=0, stop=nobs * 6, step=6), None),
]

# Supported date indexes
# Only the NumericIndex and the `date_indexes` are valid without
# frequency information
supported_date_indexes = (
    numpy_datestr_indexes
    + list_datestr_indexes
    + series_datestr_indexes
    + numpy_datetime_indexes
    + list_datetime_indexes
    + series_datetime_indexes
    + series_timestamp_indexes
)

# Unsupported (but still valid) indexes
unsupported_indexes = [
    # Non-incrementing-from-zero indexes
    (np.arange(1, nobs + 1), None),
    (np.arange(nobs)[::-1], None),
    # Float indexes, even if they increment from zero
    (np.arange(nobs) * 1.0, None),
    # Non-date-string indexes
    (list("abcde"), None),
    # Non-date-object indexes
    ([str, 1, "a", -30.1, {}], None),
]

# Unsupported date indexes (i.e., those without inferrable frequency)
unsupported_date_indexes = [
    (["1950", "1952", "1941", "1954", "1991"], None),
    (
        ["1950-01-01", "1950-01-02", "1950-01-03", "1950-01-04", "1950-01-06"],
        None,
    ),
]


def test_instantiation_valid():
    tsa_model.__warningregistry__ = {}

    # The primary goal of this test function is to make sure the
    # combinations that are supposed to be valid are actually valid, and
    # that valid but unsupported options give the appropriate warning
    # Secondarily, it also has some tests that invalid combinations raise
    # exceptions, although it's not intended to be comprehensive.
    #
    # Each of `endog`, `exog` can be in the following categories:
    # 0. None (only for exog)
    # 1. list
    # 2. numpy array
    # 3. pandas series
    # 4. pandas dataframe
    #
    # Each pandas index (of `endog`, `exog`, or passed to `dates`) can be:
    # 0. None
    # 1. RangeIndex (if applicable; i.e., if Pandas >= 0.18)
    # 2. Integral Indexes with values exactly equal to 0, 1, ..., nobs-1
    # 3. DatetimeIndex with frequency
    # 4. PeriodIndex with frequency
    # 5. Anything that does not fall into the above categories also should
    #    only raise an exception if it was passed to dates, and may trigger
    #    a warning otherwise.
    #
    # `date` can be one of the following:
    # 0. None
    # 2. Pandas index #2
    # 3. Pandas index #3
    # 4. List of date strings (requires freq)
    # 5. List of datetime objects (requires freq)
    # 6. Array of date strings (requires freq)
    # 7. Array of datetime objects (requires freq)
    # 8. Series of date strings (requires freq)
    # 9. Series of datetime objects (requires freq)
    # 10. Series of pandas timestamps (requires freq)
    # 11. Anything that does not fall into the above categories should raise
    #     an exception.
    #
    # `freq` can be:
    # 0. None
    # 1. Something that can be passed to `pd.to_offset`
    # 2. Anything that cannot should raise an Exception
    #
    # Each test will be denoted by:
    # endog.index:exog.index/date/freq where the corresponding
    # location is the integer from above; e.g., 1.0:0.0/9/1 corresponds to
    # - List endog (with no index)
    # - No exog
    # - Series of datetime objects
    # - Something valid for `pd.to_offset` (e.g., 'D', if that works with
    #   dates)
    #
    # Notice that the endog.index:exog.index really collapses to a single
    # element, which is the evaluated `row_label`. This is first the exog
    # index, if exists, then the endog index, if it exists, or None
    # otherwise. **Thus, we will not test `exog` here.**
    #
    # Example valid combinations of row_label/date/freq include:
    # - */0/0 (i.e., anything is valid if date and freq are not passed)
    # - */%/% where %/% denotes a valid date/freq combination (i.e., any
    #   row_label is valid if a valid date/freq combination is given)
    #
    # Example invalid combinations include:
    # - [1-2],[3-4].4/0/[1-2] (i.e., if have freq, then must have, or
    #   coerce, a date index)
    # - */[4-10]/0 (i.e., for some types of dates, freq must be passed)

    # Baseline: list, numpy endog with no dates, no freq
    for endog in dta[:2]:
        # No indexes, should not raise warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            mod = tsa_model.TimeSeriesModel(endog)
            assert isinstance(mod._index, pd.RangeIndex) or np.issubdtype(
                mod._index.dtype, np.integer
            )
            assert_equal(mod._index_none, True)
            assert_equal(mod._index_dates, False)
            assert_equal(mod._index_generated, True)
            assert_equal(mod.data.dates, None)
            assert_equal(mod.data.freq, None)

    # Test list, numpy endog, pandas w/o index; with dates / freq argument
    for endog in dta:
        # Supported date indexes, should not raise warnings, do not need freq
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            for ix, freq in date_indexes + period_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = ix.freqstr
                assert_equal(
                    isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)),
                    True,
                )
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)

        # Supported date indexes, should not raise warnings, can use valid freq
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            for ix, freq in date_indexes + period_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = ix.freqstr
                assert_equal(
                    isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)),
                    True,
                )
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)

        # Other supported indexes, with valid freq, should not raise warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            for ix, freq in supported_date_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(
                    isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)),
                    True,
                )
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)

        # Since only supported indexes are valid `dates` arguments, everything
        # else is invalid here
        for ix, _ in supported_increment_indexes + unsupported_indexes:
            # pandas >= 2 warns that it cannot infer the format of strings
            with pytest.raises(ValueError), warns_or_none(
                UserWarning, match="Could not infer format"
            ):
                tsa_model.TimeSeriesModel(endog, dates=ix)

    # Test pandas (Series, DataFrame); with index (no dates/freq argument)
    for base_endog in dta[2:4]:
        # DatetimeIndex and PeriodIndex, should not raise warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            for ix, freq in date_indexes + period_indexes:
                endog = base_endog.copy()
                endog.index = ix

                mod = tsa_model.TimeSeriesModel(endog)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = ix.freqstr
                assert_equal(
                    isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)),
                    True,
                )
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)

        # Increment index (this is a "supported" index in the sense that it
        # does not raise a warning, but obviously not a date index)
        endog = base_endog.copy()
        endog.index = supported_increment_indexes[0][0]

        mod = tsa_model.TimeSeriesModel(endog)
        assert is_int_index(mod._index)
        assert_equal(mod._index_none, False)
        assert_equal(mod._index_dates, False)
        assert_equal(mod._index_generated, False)
        assert_equal(mod._index_freq, None)
        assert_equal(mod.data.dates, None)
        assert_equal(mod.data.freq, None)

        # RangeIndex (start=0, end=nobs, so equivalent to increment index)
        endog = base_endog.copy()
        endog.index = supported_increment_indexes[1][0]

        mod = tsa_model.TimeSeriesModel(endog)
        assert type(mod._index) is pd.RangeIndex
        assert not mod._index_none
        assert not mod._index_dates
        assert not mod._index_generated
        assert mod._index_freq is None
        assert mod.data.dates is None
        assert mod.data.freq is None

        # Supported indexes *when a freq is given*, should not raise a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            for ix, freq in supported_date_indexes:
                # Avoid warnings due to Series with object dtype
                if isinstance(ix, pd.Series) and ix.dtype == object:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        endog = pd.DataFrame(base_endog, index=ix)
                else:
                    endog = pd.DataFrame(base_endog, index=ix)

                mod = tsa_model.TimeSeriesModel(endog, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(
                    isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)),
                    True,
                )
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)

        # Unsupported (or any) indexes to the given series, *when a supported
        # date and freq is given*, should not raise a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            for ix, freq in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = unsupported_indexes[0][0]

                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(
                    isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)),
                    True,
                )
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)

        # Date indexes with inferrable freq, but no given freq, should all give
        # warnings
        message = (
            "No frequency information was provided,"
            " so inferred frequency %s will be used."
        )
        last_len = 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for ix, freq in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert type(mod._index) is pd.DatetimeIndex
                assert not mod._index_none
                assert mod._index_dates
                assert not mod._index_generated
                assert_equal(mod._index.freq, mod._index_freq)
                assert mod.data.dates.equals(mod._index)

                # Note: here, we need to hedge the test a little bit because
                # inferred frequencies are not always the same as the original
                # frequency. From the examples above, when the actual freq is
                # 2QS-OCT, the inferred freq is 2QS-JAN. This is an issue with
                # inferred frequencies, but since we are warning the user, it's
                # not a failure of the code. Thus we only test the "major" part
                # of the freq, and just test that the right message is given
                # (even though it will not have the actual freq of the data in
                # it).
                if len(w) == last_len:
                    continue
                assert_equal(mod.data.freq.split("-")[0], freq.split("-")[0])
                assert_equal(str(w[-1].message), message % mod.data.freq)
                last_len = len(w)

        # Unsupported (but valid) indexes, should all give warnings
        message = (
            "An unsupported index was provided. As a result, forecasts "
            "cannot be generated. To use the model for forecasting, use "
            "on the supported classes of index."
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for ix, _ in unsupported_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                assert_equal(isinstance(mod._index, (pd.Index, pd.RangeIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, False)
                assert_equal(mod._index_generated, True)
                assert_equal(mod._index_freq, None)
                assert_equal(mod.data.dates, None)
                assert_equal(mod.data.freq, None)

                assert_equal(str(w[0].message), message)

        # Date indexes without inferrable freq, and with no given freq, should
        # all give warnings
        message = (
            "A date index has been provided, but it has no"
            " associated frequency information and so will be"
            " ignored when e.g., forecasting."
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for ix, _ in unsupported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                assert isinstance(mod._index, pd.RangeIndex) or is_int_index(mod._index)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, False)
                assert_equal(mod._index_generated, True)
                assert_equal(mod._index_freq, None)
                assert_equal(mod.data.dates, None)
                assert_equal(mod.data.freq, None)

                assert_equal(str(w[0].message), message)

    # Test (invalid) freq with no index
    endog = dta[0]
    with pytest.raises(ValueError):
        tsa_model.TimeSeriesModel(
            endog,
            freq=date_indexes[1][0].freq,
        )

    # Test conflicting index, freq specifications
    endog = dta[2].copy()
    endog.index = date_indexes[0][0]
    with pytest.raises(ValueError):
        tsa_model.TimeSeriesModel(
            endog,
            freq=date_indexes[1][0].freq,
        )

    # Test unsupported index, but a freq specification
    endog = dta[2].copy()
    endog.index = unsupported_indexes[0][0]
    with pytest.raises(ValueError):
        tsa_model.TimeSeriesModel(
            endog,
            freq=date_indexes[1][0].freq,
        )

    # Test index that can coerce to date time but incorrect freq
    endog = dta[2].copy()
    endog.index = numpy_datestr_indexes[0][0]
    with pytest.raises(ValueError):
        tsa_model.TimeSeriesModel(
            endog,
            freq=date_indexes[1][0].freq,
        )


def test_prediction_increment_unsupported():
    # a. Generated from unsupported index
    endog = dta[2].copy()
    endog.index = unsupported_indexes[-2][0]
    with pytest.warns(ValueWarning, match="An unsupported index was provided"):
        # pandas >= 2 warns that it cannot infer the format of strings
        with warns_or_none(UserWarning, match="Could not infer format"):
            mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]; notice that since this is an in-sample
    # prediction, the index returned is the (unsupported) original index
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod.data.row_labels), True)

    # Negative index: [-2, end]; notice that since this is an in-sample
    # prediction, the index returned is a piece of the (unsupported)
    # original index
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod.data.row_labels[3:]), True)

    # Forecasting: [1, 5], notice that since an unsupported index was given,
    # a warning will be issued
    start_key = 1
    end_key = nobs
    with pytest.raises(ValueError, match="No supported index is available"):
        mod._get_prediction_index(start_key, end_key)

    # Test getting a location that exists in the (internal) index
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the (internal) index
    # when using the function that alternatively falls back to the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the given (unsupported) index
    # Note that the returned index is now like the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc("c")
    assert_equal(loc, 2)
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)


def test_prediction_increment_nonpandas():
    endog = dta[0]
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]; since there was no index at all and the data
    # is not Pandas, the returned prediction_index is None
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index is None, True)

    # Negative index: [-2, end]; since there was no index at all and the data
    # is not Pandas, the returned prediction_index is None
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index is None, True)

    # Forecasting: [1, 5]; since there was no index at all and the data
    # is not Pandas, the returned prediction_index is None
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    assert_equal(prediction_index is None, True)

    # Test getting a location that exists in the (internal) index
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the (internal) index
    # when using the function that alternatively falls back to the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)


def test_prediction_increment_pandas_noindex():
    endog = dta[2].copy()
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]; since there was no index and the data is
    # Pandas, the index is the generated incrementing index, and no warning is
    # issued
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod._index), True)

    # Negative index: [-2, end]; since there was no index and the data is
    # Pandas, the index is the generated incrementing index, and no warning is
    # issued
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod._index[3:]), True)

    # Forecasting: [1, 5]; since there was no index and the data is
    # Pandas, the index is the generated incrementing index, and no warning is
    # issued
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    assert_equal(prediction_index.equals(pd.Index(np.arange(1, 6))), True)


def test_prediction_increment_pandas_dates_daily():
    # Date-based index
    endog = dta[2].copy()
    endog.index = date_indexes[0][0]  # Daily, 1950-01-01, 1950-01-02, ...
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]; the index is the date index
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)

    # In-sample prediction: [0, 3]; the index is a subset of the date index
    start_key = 0
    end_key = 3
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[:4]), True)

    # Negative index: [-2, end]
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)

    # Forecasting: [1, 5]; the index is an extended version of the date index
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start="1950-01-02", periods=5, freq="D")
    assert_equal(prediction_index.equals(desired_index), True)

    # Date-based keys

    # In-sample prediction (equivalent to [1, 3])
    start_key = "1950-01-02"
    end_key = "1950-01-04"
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[1:4]), True)

    # Out-of-sample forecasting (equivalent to [0, 5])
    start_key = "1950-01-01"
    end_key = "1950-01-08"
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start="1950-01-01", periods=8, freq="D")
    assert_equal(prediction_index.equals(desired_index), True)

    # Test getting a location that exists in the (internal) index
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start="1950-01-01", periods=3, freq="D")
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the (internal) index
    # when using the function that alternatively falls back to the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start="1950-01-01", periods=3, freq="D")
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the given (unsupported) index
    # Note that the returned index is now like the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc("1950-01-03")
    assert_equal(loc, 2)
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)


def test_prediction_increment_pandas_dates_monthly():
    # Date-based index
    endog = dta[2].copy()
    endog.index = date_indexes[2][0]  # Monthly, 1950-01, 1950-02, ...
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]; the index is the date index
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)

    # In-sample prediction: [0, 3]; the index is a subset of the date index
    start_key = 0
    end_key = 3
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[:4]), True)

    # Negative index: [-2, end]
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)

    # Forecasting: [1, 5]; the index is an extended version of the date index
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start="1950-02", periods=5, freq="MS")
    assert_equal(prediction_index.equals(desired_index), True)

    # Date-based keys

    # In-sample prediction (equivalent to [1, 3])
    start_key = "1950-02"
    end_key = "1950-04"
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[1:4]), True)

    # Out-of-sample forecasting (equivalent to [0, 5])
    start_key = "1950-01"
    end_key = "1950-08"
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start="1950-01", periods=8, freq="MS")
    assert_equal(prediction_index.equals(desired_index), True)

    # Test getting a location that exists in the (internal) index
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start="1950-01", periods=3, freq="MS")
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the (internal) index
    # when using the function that alternatively falls back to the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start="1950-01", periods=3, freq="MS")
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)

    # Test getting a location that exists in the given (unsupported) index
    # Note that the returned index is now like the row labels
    loc, index, index_was_expanded = mod._get_index_label_loc("1950-03")
    assert_equal(loc, slice(2, 3, None))
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)


def test_prediction_increment_pandas_dates_nanosecond():
    # Date-based index
    endog = dta[2].copy()
    endog.index = pd.date_range(start="1970-01-01", periods=len(endog), freq="ns")
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]; the index is the date index
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)

    # Negative index: [-2, end]
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)

    # Forecasting: [1, 5]; the index is an extended version of the date index
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start="1970-01-01", periods=6, freq="ns")[1:]
    assert_equal(prediction_index.equals(desired_index), True)

    # Date-based keys
    start_key = pd.Timestamp("1970-01-01")
    end_key = pd.Timestamp(start_key.value + 7)
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start="1970-01-01", periods=8, freq="ns")
    assert_equal(prediction_index.equals(desired_index), True)


def test_range_index():
    rs = np.random.RandomState(328389218)
    tsa_model.__warningregistry__ = {}

    endog = pd.Series(rs.normal(size=5))
    assert_equal(isinstance(endog.index, pd.RangeIndex), True)
    # Warning should not be given
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tsa_model.TimeSeriesModel(endog)
        assert_equal(len(w), 0)


def test_prediction_rangeindex():
    index = supported_increment_indexes[2][0]
    endog = pd.Series(dta[0], index=index)
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=-5, stop=0, step=1)
    assert_equal(prediction_index.equals(desired_index), True)

    # Negative index: [-2, end]
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=-2, stop=0, step=1)
    assert_equal(prediction_index.equals(desired_index), True)

    # Forecasting: [1, 5]
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.RangeIndex(start=-4, stop=1, step=1)
    assert_equal(prediction_index.equals(desired_index), True)


def test_prediction_rangeindex_withstep():
    index = supported_increment_indexes[3][0]
    endog = pd.Series(dta[0], index=index)
    mod = tsa_model.TimeSeriesModel(endog)

    # Tests three common use cases: basic prediction, negative indexes, and
    # out-of-sample indexes.

    # Basic prediction: [0, end]
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=0, stop=nobs * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)

    # Negative index: [-2, end]
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=3 * 6, stop=nobs * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)

    # Forecasting: [1, 5]
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.RangeIndex(start=1 * 6, stop=(nobs + 1) * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)

    # Test getting a location that exists in the index
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3 * 6, step=6)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)


def test_custom_index():
    rs = np.random.RandomState(328392810)
    tsa_model.__warningregistry__ = {}

    endog = pd.Series(rs.normal(size=5), index=["a", "b", "c", "d", "e"])
    message = (
        "An unsupported index was provided. As a result, forecasts cannot be "
        "generated. To use the model for forecasting, use on the "
        "supported classes of index."
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mod = tsa_model.TimeSeriesModel(endog)
        # Change due to pandas changes that produce a warning
        messages = [str(warn.message) for warn in w]
        assert message in messages
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key
    )

    # Test the default output index
    assert_equal(prediction_index.equals(pd.Index(["d", "e"])), True)

    # Test custom output index
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key, index=["f", "g"]
    )
    assert_equal(prediction_index.equals(pd.Index(["f", "g"])), True)

    # Test getting a location in the index w/o fallback to row labels
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    assert_equal(index.equals(pd.RangeIndex(0, 3)), True)
    assert_equal(index_was_expanded, False)
    assert_equal(index_was_expanded, False)

    # Test getting an invalid location in the index w/ fallback to row labels
    with pytest.raises(KeyError):
        mod._get_index_loc("c")

    # Test getting a location in the index w/ fallback to row labels
    loc, index, index_was_expanded = mod._get_index_label_loc("c")
    assert_equal(loc, 2)
    assert_equal(index.equals(pd.Index(["a", "b", "c"])), True)
    assert_equal(index_was_expanded, False)

    # Test getting an invalid location in the index w/ fallback to row labels
    with pytest.raises(KeyError):
        mod._get_index_label_loc("aa")

    # Test out-of-sample
    start_key = 4
    end_key = 5
    with pytest.raises(ValueError, match="No supported index is available"):
        mod._get_prediction_index(start_key, end_key)

    # Test out-of-sample custom index
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        start_key, end_key, index=["f", "g"]
    )
    assert_equal(prediction_index.equals(pd.Index(["f", "g"])), True)

    # Test invalid custom index
    with pytest.raises(ValueError):
        mod._get_prediction_index(
            start_key,
            end_key,
            index=["f", "g", "h"],
        )


# pandas 3.1 deprecates Period with a business day frequency
B_PERIOD_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:Period with BDay freq is deprecated:FutureWarning",
    r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning",
)
HOUR = "H" if PD_LT_2_2_0 else "h"
# Starts on a Monday, so that business day periods skip a weekend in the sample
PERIOD_START = "2000-01-03"
PERIOD_FREQS = [
    "Y",
    "Q",
    "M",
    "W",
    "D",
    HOUR,
    "2Q",
    "3D",
    "2W",
    pytest.param("B", marks=B_PERIOD_DEPRECATION),
    pytest.param("2B", marks=B_PERIOD_DEPRECATION),
]


def test_nonmonotonic_periodindex():
    # Create a nonmonotonic period index
    tmp = pd.period_range(start="2000", end="2002", freq="Y")
    index = tmp.tolist() + tmp.tolist()
    endog = pd.Series(np.zeros(len(index)), index=index)

    message = (
        "A date index has been provided, but it is not"
        " monotonic and so will be ignored when e.g."
        " forecasting."
    )
    with pytest.warns(ValueWarning, match=message):
        tsa_model.TimeSeriesModel(endog)


@pytest.mark.parametrize("order", ["descending", "unsorted"])
def test_nonmonotonic_periodindex_single_warning(order):
    # A non-monotonic index only gets the "not monotonic" warning, not the one
    # for non-consecutive periods
    index = pd.period_range(start="2000-01", periods=6, freq="M")
    index = index[::-1] if order == "descending" else index.take([1, 0, 2, 3, 5, 4])
    endog = pd.Series(np.zeros(len(index)), index=index)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mod = tsa_model.TimeSeriesModel(endog)
    messages = [str(w.message) for w in record if w.category is ValueWarning]
    assert len(messages) == 1
    assert "not monotonic" in messages[0]
    assert "consecutive" not in messages[0]
    assert mod._index_generated


@pytest.mark.parametrize("freq", PERIOD_FREQS)
@pytest.mark.parametrize("case", ["gap", "repeat", "gap_len2", "repeat_len2"])
def test_nonconsecutive_periodindex(freq, case):
    # See GH#6247, GH#8950
    base = pd.period_range(start=PERIOD_START, periods=8, freq=freq)
    index = {
        "gap": base.delete(3),
        "repeat": base[[0, 1, 2, 2, 3, 4]],
        "gap_len2": base[[0, 2]],
        "repeat_len2": base[[0, 0]],
    }[case]
    endog = pd.Series(np.zeros(len(index)), index=index)

    message = (
        "A Period index has been provided, but its periods are not"
        r" consecutive \(it has gaps or repeated periods\) and so will be"
        " ignored. Forecasting will fail unless an index is given."
    )
    with pytest.warns(ValueWarning, match=message):
        mod = tsa_model.TimeSeriesModel(endog)
    assert mod._index_generated
    assert not mod._index_dates
    assert mod._index.equals(pd.RangeIndex(len(index)))
    assert mod.data.dates is None

    # In-sample prediction keeps the original labels
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(
        1, len(index) - 1
    )
    assert prediction_index.equals(index[1:])

    # Forecasting must not silently produce labels from inside the sample
    with pytest.raises(ValueError, match="No supported index is available"):
        mod._get_prediction_index(len(index), len(index) + 2)


@pytest.mark.parametrize(
    "index",
    [
        # Daily periods that skip weekends (GH#8950)
        pd.bdate_range("2023-07-03", periods=10).to_period("D"),
        # Gap in a frequency with a multiplier
        pd.PeriodIndex(["2000Q1", "2000Q3", "2001Q3"], freq="2Q"),
    ],
)
def test_nonconsecutive_periodindex_literal(index):
    # See GH#6247, GH#8950
    endog = pd.Series(np.zeros(len(index)), index=index)

    with pytest.warns(ValueWarning, match="its periods are not consecutive"):
        mod = tsa_model.TimeSeriesModel(endog)
    assert mod._index_generated
    assert mod._index.equals(pd.RangeIndex(len(index)))


@pytest.mark.parametrize("nobs", [1, 2, 5])
@pytest.mark.parametrize("freq", PERIOD_FREQS)
def test_consecutive_periodindex(freq, nobs):
    # Consecutive periods, including those with a frequency multiplier and
    # those that skip weekends (B), are supported
    index = pd.period_range(start=PERIOD_START, periods=nobs, freq=freq)
    endog = pd.Series(np.zeros(len(index)), index=index)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ValueWarning)
        mod = tsa_model.TimeSeriesModel(endog)
    assert not mod._index_generated
    assert mod._index.equals(index)

    _, _, out_of_sample, prediction_index = mod._get_prediction_index(
        nobs, nobs + 1
    )
    assert out_of_sample == 2
    expected = pd.period_range(start=index[-1] + 1, periods=2, freq=index.freq)
    assert prediction_index.equals(expected)


@pytest.mark.parametrize(
    "freq, expected",
    [
        # Thursday 2023-07-06 and Friday 2023-07-07, then the weekend is skipped
        ("B", ["2023-07-10", "2023-07-11"]),
        # Additionally skip the 2023-07-10 holiday
        (
            pd.offsets.CustomBusinessDay(holidays=["2023-07-10"]),
            ["2023-07-11", "2023-07-12"],
        ),
    ],
)
def test_business_day_index(freq, expected):
    # Data with gaps that follow a calendar rule (e.g. trading days) are
    # supported through a DatetimeIndex with a matching frequency, see GH#8950
    index = pd.date_range(end="2023-07-07", periods=5, freq=freq)
    endog = pd.Series(np.zeros(len(index)), index=index)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ValueWarning)
        mod = tsa_model.TimeSeriesModel(endog)
    assert not mod._index_generated

    nobs = len(index)
    _, _, out_of_sample, prediction_index = mod._get_prediction_index(
        nobs, nobs + 1
    )
    assert out_of_sample == 2
    assert prediction_index.equals(pd.DatetimeIndex(expected, freq=freq))


def test_get_index_loc_quarterly():
    # See GH#6339
    ix = pd.period_range("2000Q1", periods=8, freq="Q").to_timestamp()
    endog = pd.Series(np.zeros(8), index=ix)

    mod = tsa_model.TimeSeriesModel(endog)
    loc, index, _ = mod._get_index_loc("2003Q2")
    assert_equal(index[loc], pd.Period("2003Q2").to_timestamp())


class ConstantMean(tsa_model.TimeSeriesModel):
    # A minimal model, so that the index handling of TimeSeriesModel is reached
    # through the public predict method of its results, as in the TSA models
    def predict(self, params, exog=None, start=None, end=None):
        start = 0 if start is None else start
        start, end, out_of_sample, _ = self._get_prediction_index(start, end)
        return np.full(end - start + 1 + out_of_sample, params[0])


def constant_mean_results(endog, **kwargs):
    mod = ConstantMean(endog, **kwargs)
    res = tsa_model.TimeSeriesModelResults(mod, np.array([2.5]), np.eye(1))
    return tsa_model.TimeSeriesResultsWrapper(res)


@contextlib.contextmanager
def value_warning(match):
    # Expect a ValueWarning that matches `match`, or no ValueWarning if None
    if match is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ValueWarning)
            yield
    else:
        with pytest.warns(ValueWarning, match=match):
            yield


MS_INDEX = pd.date_range("2000-01-01", periods=6, freq="MS")
INFERRED_MS = "No frequency information was provided, so inferred frequency MS "
# index of 6 observations, the expected ValueWarning, the expected labels of
# the locations 4 to 7 (two in-sample, two out-of-sample) and data.freq
SUPPORTED_INDEXES = {
    "datetime": (
        MS_INDEX,
        None,
        pd.date_range("2000-05-01", periods=4, freq="MS"),
        "MS",
    ),
    "datetime-no-freq": (
        pd.DatetimeIndex(MS_INDEX.tolist()),
        INFERRED_MS,
        pd.date_range("2000-05-01", periods=4, freq="MS"),
        "MS",
    ),
    "date-strings": (
        pd.Index(MS_INDEX.strftime("%Y-%m-%d").tolist()),
        INFERRED_MS,
        pd.date_range("2000-05-01", periods=4, freq="MS"),
        "MS",
    ),
    "period": (
        pd.period_range("2000-01", periods=6, freq="M"),
        None,
        pd.period_range("2000-05", periods=4, freq="M"),
        "M",
    ),
    "range-with-step": (
        pd.RangeIndex(start=10, stop=28, step=3),
        None,
        pd.RangeIndex(start=22, stop=34, step=3),
        None,
    ),
    "integer": (
        pd.Index(np.arange(6, dtype=np.int64)),
        None,
        pd.Index(np.arange(4, 8, dtype=np.int64)),
        None,
    ),
}
# index of 6 observations and the expected ValueWarning
UNSUPPORTED_INDEXES = {
    "datetime-irregular": (
        pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-04",
                "2000-01-08",
                "2000-01-09",
                "2000-01-10",
            ]
        ),
        "A date index has been provided, but it has no associated frequency",
    ),
    "integer-from-one": (
        pd.Index(np.arange(1, 7, dtype=np.int64)),
        "An unsupported index was provided",
    ),
    "integer-with-step": (
        pd.Index(np.arange(0, 12, 2, dtype=np.int64)),
        "An unsupported index was provided",
    ),
    "integer-irregular": (
        pd.Index(np.array([0, 1, 2, 4, 5, 7], dtype=np.int64)),
        "An unsupported index was provided",
    ),
    "float": (pd.Index(np.arange(6.0)), "An unsupported index was provided"),
    "string": (pd.Index(list("abcdef")), "An unsupported index was provided"),
    "object": (
        pd.Index(["a", 1, "b", 2, "c", 3], dtype=object),
        "An unsupported index was provided",
    ),
    "period-with-gap": (
        pd.period_range("2000-01", periods=7, freq="M").delete(3),
        "A Period index has been provided, but its periods are not consecutive",
    ),
}


@pytest.mark.parametrize("case", list(SUPPORTED_INDEXES))
def test_supported_index_prediction_labels(case):
    # Predictions are labelled with the index of the data, which is extended
    # for forecasts; a negative start counts from the end of the sample
    index, warning, expected, freq = SUPPORTED_INDEXES[case]
    endog = pd.Series(np.arange(6.0), index=index)
    with value_warning(warning):
        res = constant_mean_results(endog)
    assert res.model.data.freq == freq

    assert res.predict(start=-2, end=7).index.equals(expected)
    assert res.predict(start=6, end=7).index.equals(expected[2:])


@pytest.mark.parametrize("case", list(UNSUPPORTED_INDEXES))
def test_unsupported_index_prediction_labels(case):
    # An unsupported index labels in-sample predictions, but it cannot be
    # extended for forecasts
    index, warning = UNSUPPORTED_INDEXES[case]
    endog = pd.Series(np.arange(6.0), index=index)
    with pytest.warns(ValueWarning, match=warning):
        # pandas >= 2 warns that it cannot infer the format of strings
        with warns_or_none(UserWarning, match="Could not infer format"):
            res = constant_mean_results(endog)
    assert res.model.data.dates is None
    assert res.model.data.freq is None

    assert res.predict(start=-2, end=5).index.equals(index[-2:])
    with pytest.raises(ValueError, match="No supported index is available"):
        res.predict(start=4, end=6)


@pytest.mark.parametrize("case", list(SUPPORTED_INDEXES))
def test_arima_prediction_labels(case):
    index, warning, expected, _ = SUPPORTED_INDEXES[case]
    endog = pd.Series(np.arange(6.0), index=index)
    with value_warning(warning):
        res = ARIMA(endog, order=(1, 0, 0)).filter([0.0, 0.5, 1.0])
    pred = res.get_prediction(start=-2, end=7)
    assert pred.predicted_mean.index.equals(expected)
    assert res.forecast(2).index.equals(expected[2:])


@pytest.mark.parametrize("case", ["datetime", "datetime-no-freq", "period", "integer"])
def test_autoreg_prediction_labels(case):
    index, warning, expected, _ = SUPPORTED_INDEXES[case]
    rng = np.random.default_rng(20261121)
    endog = pd.Series(rng.standard_normal(6), index=index)
    with value_warning(warning):
        res = AutoReg(endog, 1).fit()
    pred = res.get_prediction(start=4, end=7)
    assert pred.predicted_mean.index.equals(expected)
    assert res.forecast(2).index.equals(expected[2:])


ME_INDEX = pd.date_range("2000-01-31", periods=6, freq=MONTH_END)


@pytest.mark.parametrize(
    "index, start, end, expected",
    [
        (
            pd.period_range("2000-01", periods=6, freq="M"),
            "2000-03",
            "2000-08",
            pd.period_range("2000-03", "2000-08", freq="M"),
        ),
        (
            pd.period_range("2000-01", periods=6, freq="M"),
            pd.Period("2000-02", freq="M"),
            pd.Period("2000-07", freq="M"),
            pd.period_range("2000-02", "2000-07", freq="M"),
        ),
        # Quarter strings for quarterly periods and dates at the quarter end
        (
            pd.period_range("2000Q1", periods=8, freq="Q"),
            "2001Q2",
            "2002Q2",
            pd.period_range("2001Q2", "2002Q2", freq="Q"),
        ),
        (
            pd.date_range("2000-03-31", periods=8, freq=QUARTER_END),
            "2001Q2",
            "2001Q4",
            pd.date_range("2001-06-30", periods=3, freq=QUARTER_END),
        ),
        (
            ME_INDEX,
            dt.datetime(2000, 2, 29),
            pd.Timestamp("2000-07-31"),
            pd.date_range("2000-02-29", "2000-07-31", freq=MONTH_END),
        ),
        # The index is extended until it reaches a date between two dates
        (
            ME_INDEX,
            "2000-02-29",
            "2000-08-15",
            pd.date_range("2000-02-29", "2000-08-31", freq=MONTH_END),
        ),
    ],
)
def test_date_key_prediction_labels(index, start, end, expected):
    endog = pd.Series(np.arange(len(index), dtype=float), index=index)
    res = constant_mean_results(endog)
    assert res.predict(start=start, end=end).index.equals(expected)


@pytest.mark.parametrize(
    "index, start, end, error, match",
    [
        (pd.RangeIndex(6), -7, None, KeyError, "The `start` argument"),
        (pd.RangeIndex(6), 0, -7, KeyError, "The `end` argument"),
        (MS_INDEX, "1999-12-01", None, KeyError, "The `start` argument"),
        (MS_INDEX, 0, "1999-12-01", KeyError, "The `end` argument"),
        (
            pd.Index(np.arange(6, dtype=np.int64)),
            "a",
            None,
            KeyError,
            "The `start` argument",
        ),
        (MS_INDEX, 3, 1, ValueError, "Prediction must have `end` after `start`"),
    ],
)
def test_prediction_key_errors(index, start, end, error, match):
    endog = pd.Series(np.arange(6.0), index=index)
    res = constant_mean_results(endog)
    with pytest.raises(error, match=match):
        res.predict(start=start, end=end)


@pytest.mark.parametrize(
    "index, key, loc, expected, expanded",
    [
        (
            pd.Index(np.arange(5, dtype=np.int64)),
            -2,
            3,
            pd.Index(np.arange(4, dtype=np.int64)),
            False,
        ),
        (
            pd.Index(np.arange(5, dtype=np.int64)),
            7,
            7,
            pd.Index(np.arange(8, dtype=np.int64)),
            True,
        ),
        (
            pd.RangeIndex(start=10, stop=25, step=3),
            -2,
            3,
            pd.RangeIndex(start=10, stop=22, step=3),
            False,
        ),
        (
            pd.RangeIndex(start=10, stop=25, step=3),
            7,
            7,
            pd.RangeIndex(start=10, stop=34, step=3),
            True,
        ),
        (pd.Index(list("abcde")), "c", 2, pd.Index(list("abc")), False),
        (
            pd.period_range("2000-01", periods=5, freq="M"),
            "2000-08",
            7,
            pd.period_range("2000-01", periods=8, freq="M"),
            True,
        ),
        (
            pd.date_range("2000-01-31", periods=5, freq=MONTH_END),
            "2000-08-15",
            7,
            pd.date_range("2000-01-31", periods=8, freq=MONTH_END),
            True,
        ),
        (
            pd.date_range("2000-03-31", periods=8, freq=QUARTER_END),
            "2001Q2",
            5,
            pd.date_range("2000-03-31", periods=6, freq=QUARTER_END),
            False,
        ),
    ],
)
def test_get_index_loc(index, key, loc, expected, expanded):
    result = tsa_model.get_index_loc(key, index)
    assert result[0] == loc
    assert result[1].equals(expected)
    assert result[2] is expanded


def test_get_index_loc_missing_label():
    with pytest.raises(KeyError):
        tsa_model.get_index_loc("z", pd.Index(list("abcde")))


@pytest.mark.parametrize(
    "row_labels, key, first, stop",
    [
        # pandas.Index.get_loc returns a slice, a boolean mask and integer
        # locations for these
        (pd.Index(list("abbcd")), "b", 1, 3),
        (pd.Index(list("abcbd")), "b", 1, None),
        (
            pd.DatetimeIndex(
                ["2001-02-01", "2001-01-01", "2002-01-01", "2001-03-01", "2003-01-01"]
            ),
            "2001",
            0,
            None,
        ),
    ],
)
def test_repeated_row_label_first_location(row_labels, key, first, stop):
    # A key that is found more than once in the row labels gives its first
    # location, or the slice of all of them if they are consecutive
    loc, index, expanded = tsa_model.get_index_label_loc(
        key, pd.RangeIndex(5), row_labels
    )
    if stop is None:
        assert loc == first
        assert index.equals(row_labels[: first + 1])
    else:
        assert loc == slice(first, stop)
        assert index.equals(row_labels[:stop])
    assert expanded is False

    endog = pd.Series(np.arange(5.0), index=row_labels)
    with pytest.warns(ValueWarning):
        with warns_or_none(UserWarning, match="Could not infer format"):
            res = constant_mean_results(endog)
    assert res.predict(start=key, end=4).index.equals(row_labels[first:])


def test_get_prediction_index_given_index():
    # A given index is converted to a pandas Index. For data that are not
    # pandas objects it is only used internally, which is warned about unless
    # silent is True.
    mod = tsa_model.TimeSeriesModel(np.arange(5.0))
    base_index = pd.RangeIndex(5)
    kwargs = {"index_none": True, "index_generated": True, "data": mod.data}
    match = "the given `index` argument will only be used internally"
    with pytest.warns(ValueWarning, match=match):
        out = tsa_model.get_prediction_index(
            3, 6, 5, base_index, index=list("abcd"), **kwargs
        )
    assert out[:3] == (3, 4, 2)
    assert out[3].equals(pd.Index(list("abcd")))

    with warnings.catch_warnings():
        warnings.simplefilter("error", ValueWarning)
        out = tsa_model.get_prediction_index(
            3, 6, 5, base_index, index=list("abcd"), silent=True, **kwargs
        )
    assert out[3].equals(pd.Index(list("abcd")))

    # Without a given index, predictions of numpy data have no labels
    assert tsa_model.get_prediction_index(3, 6, 5, base_index, **kwargs) == (
        3,
        4,
        2,
        None,
    )
    with pytest.raises(ValueError, match="Invalid `index` provided"):
        tsa_model.get_prediction_index(
            3, 6, 5, base_index, index=list("abc"), **kwargs
        )


@pytest.mark.parametrize(
    "dates, freq, match",
    [
        # No frequency given and none can be inferred
        (
            ["2000-01-01", "2000-01-02", "2000-01-04"],
            None,
            "No frequency information was provided with date index",
        ),
        (
            pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"]),
            None,
            "No frequency information was provided with date index",
        ),
        # As many dates as pd.date_range("2000-01-01", "2000-01-05", freq="2D"),
        # but not the same dates
        (
            ["2000-01-01", "2000-01-02", "2000-01-05"],
            "2D",
            "The given frequency argument could not be matched",
        ),
        (
            pd.period_range("2000-01", periods=3, freq="M"),
            "D",
            "The given frequency argument is incompatible",
        ),
        # pandas assembles a frame of date parts into a Series, not an index
        (
            pd.DataFrame(
                {"year": ["2000", "2001", "2002"], "month": ["1"] * 3, "day": ["1"] * 3}
            ),
            None,
            "Non-date index provided",
        ),
    ],
)
def test_dates_errors(dates, freq, match):
    with pytest.raises(ValueError, match=match):
        tsa_model.TimeSeriesModel(np.arange(3.0), dates=dates, freq=freq)


@pytest.mark.parametrize(
    "freq, dates, expected",
    [
        (
            "MS",
            [dt.datetime(2000, month, 1) for month in (1, 2, 3)],
            pd.date_range("2000-01-01", periods=5, freq="MS"),
        ),
        (
            pd.offsets.MonthBegin(),
            [dt.datetime(2000, month, 1) for month in (1, 2, 3)],
            pd.date_range("2000-01-01", periods=5, freq="MS"),
        ),
        (
            dt.timedelta(days=1),
            [dt.datetime(2000, 1, day) for day in (1, 2, 3)],
            pd.date_range("2000-01-01", periods=5, freq="D"),
        ),
    ],
)
def test_dates_list_with_freq(freq, dates, expected):
    # dates given as a list for numpy data, with freq as a string, an offset
    # or a timedelta
    with warnings.catch_warnings():
        warnings.simplefilter("error", ValueWarning)
        res = constant_mean_results(np.arange(3.0), dates=dates, freq=freq)
    assert res.model.data.dates.equals(expected[:3])

    pred = res.predict(start=0, end=4)
    assert isinstance(pred, np.ndarray)
    assert_allclose(pred, np.full(5, 2.5))
    assert res.model.data.predict_dates.equals(expected)


@pytest.mark.parametrize(
    "index_freq, freq",
    [
        # Period aliases, which pandas 3 does not accept as offset aliases
        ("M", "M"),
        ("Q", "Q"),
        ("Q", "Q-DEC"),
        ("2Q", "2Q"),
        ("Y", "Y"),
        # Offset aliases and offsets, which were already accepted. YEAR_END
        # is "Y" before pandas 2.2, which would repeat the ID of the case above
        pytest.param("Y", YEAR_END, id="Y-YEAR_END"),
        ("M", pd.offsets.MonthEnd()),
        ("D", "D"),
        ("W", "W"),
    ],
)
def test_period_index_with_freq(index_freq, freq):
    # The frequency of a PeriodIndex can be given as its period alias
    index = pd.period_range("2000-01-01", periods=5, freq=index_freq)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mod = tsa_model.TimeSeriesModel(np.arange(5.0), dates=index, freq=freq)
    assert mod.data.freq == index.freqstr
    assert mod._index.equals(index)

    _, _, out_of_sample, prediction_index = tsa_model.get_prediction_index(
        3, 6, 5, mod._index, data=mod.data
    )
    assert out_of_sample == 2
    expected = pd.period_range(index[3], periods=4, freq=index.freq)
    assert prediction_index.equals(expected)


@pytest.mark.parametrize("freq", ["Q", "D", pd.offsets.QuarterEnd()])
def test_period_index_with_other_freq(freq):
    index = pd.period_range("2000-01", periods=5, freq="M")
    with pytest.raises(ValueError, match="incompatible with the given index"):
        tsa_model.TimeSeriesModel(np.arange(5.0), dates=index, freq=freq)


@pytest.mark.parametrize("dtype", [np.uint64, np.uint32, np.int32, np.int64])
def test_integer_index_forecast_labels(dtype):
    # An integer index 0, ..., nobs - 1 is extended with integer labels
    index = pd.Index(np.arange(5, dtype=dtype))
    loc, labels, expanded = tsa_model.get_index_loc(7, index)
    assert loc == 7
    assert expanded
    np.testing.assert_array_equal(labels.to_numpy(), np.arange(8))
    if index.dtype == np.uint64:
        # uint64 labels instead of float labels
        assert labels.dtype == np.uint64
    else:
        # signed integer labels, as before
        assert labels.dtype.kind == "i"

    mod = tsa_model.TimeSeriesModel(pd.Series(np.zeros(5), index=index))
    _, _, out_of_sample, prediction_index = tsa_model.get_prediction_index(
        3, 7, 5, mod._index, data=mod.data
    )
    assert out_of_sample == 3
    np.testing.assert_array_equal(prediction_index.to_numpy(), np.arange(3, 8))
    assert prediction_index.dtype == labels.dtype


@pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo"])
@pytest.mark.parametrize(
    "freq, start, end, expected_start, expected_end",
    [
        # In the sample, also a month that covers several days
        ("MS", "2001-04", "2001-06", "2001-04-01", "2001-06-01"),
        ("D", "2001-04", "2001-04-03", "2001-04-01", "2001-04-03"),
        # The end after the sample
        ("MS", "2001-10-01", "2002-03", "2001-10-01", "2002-03-01"),
        ("D", "2001-05-10", "2001-05-20", "2001-05-10", "2001-05-20"),
    ],
)
def test_string_keys_tz_aware_index(tz, freq, start, end, expected_start, expected_end):
    # Date strings are local times of a tz-aware index; comparing them with
    # the index as naive timestamps raised a TypeError
    nobs = 24 if freq == "MS" else 500
    index = pd.date_range("2000-01-01", periods=nobs, freq=freq, tz=tz)
    endog = pd.Series(np.zeros(nobs), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start_loc, end_loc, out_of_sample, prediction_index = (
        tsa_model.get_prediction_index(start, end, nobs, mod._index, data=mod.data)
    )
    expected = pd.date_range(expected_start, expected_end, freq=freq, tz=tz)
    assert prediction_index.equals(expected)
    assert index[start_loc] == expected[0]
    assert end_loc == min(nobs - 1, start_loc + len(expected) - 1)
    assert end_loc + out_of_sample - start_loc + 1 == len(expected)


def test_date_keys_tz_aware_index_other_keys():
    # Only naive date strings of a DatetimeIndex are localized. The string
    # keys of a PeriodIndex are periods, which have no time zone.
    periods = pd.period_range("2000-01", periods=24, freq="M")
    assert tsa_model.get_index_loc("2001-04", periods)[0] == 15
    assert tsa_model.get_index_loc("2002-03", periods)[0] == 26
    index = pd.date_range("2000-01-01", periods=24, freq="MS", tz="US/Eastern")
    # A string with a UTC offset is already tz-aware: 2001-04-01 is in EST.
    # The seconds are given since pandas < 2.2 cannot look up such a string
    # without them (KeyError: 14)
    key = "2001-04-01T00:00:00-05:00"
    loc, new_index, expanded = tsa_model.get_index_loc(key, index)
    assert loc == 15
    assert not expanded
    assert new_index.equals(index[:16])
    # Naive datetime objects are not taken as local times, as pandas does not
    # do either
    for key in (pd.Timestamp("2001-04-01"), pd.Timestamp("2002-03-01")):
        with pytest.raises(TypeError, match="tz-naive and tz-aware"):
            tsa_model.get_index_loc(key, index)


@pytest.mark.parametrize(
    "start, end, expected_start, expected_end",
    [
        # No date in February 2000: prediction starts at the next date
        ("2000-02", "2000-10", "2000-04-01", "2000-10-01"),
        # No date in May 2000: prediction ends at the previous date
        (0, "2000-05", "2000-01-01", "2000-04-01"),
    ],
)
def test_empty_partial_date_key_deprecated(start, end, expected_start, expected_end):
    # pandas gives an empty slice for a partial date string between two dates
    # of the index, which was used silently
    index = pd.date_range("2000-01-01", periods=8, freq="QS")
    endog = pd.Series(np.zeros(8), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    with pytest.warns(FutureWarning, match="No date of the index matches") as rec:
        start_loc, end_loc, out_of_sample, prediction_index = (
            tsa_model.get_prediction_index(start, end, 8, mod._index, data=mod.data)
        )
    # The warning points at the caller outside statsmodels
    assert rec[0].filename == __file__
    expected = pd.date_range(expected_start, expected_end, freq="QS")
    assert prediction_index.equals(expected)
    assert index[start_loc] == expected[0]
    assert index[end_loc] == expected[-1]
    assert out_of_sample == 0


def test_partial_date_key_with_dates_not_deprecated():
    # A partial date string that matches dates gives a non-empty slice
    index = pd.date_range("2000-01-01", periods=8, freq="QS")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loc, new_index, _ = tsa_model.get_index_loc("2000-04", index)
    assert loc == slice(1, 2)
    assert new_index.equals(index[:2])


@pytest.mark.parametrize(
    "dtype, key",
    [("int8", 300), ("uint8", 300), ("int16", 70000), ("uint16", 70000)],
)
def test_small_int_index_extended_past_dtype_max(dtype, key):
    # Extending an index 0, ..., n - 1 of a small integer dtype past the
    # largest value of the dtype raised "Maximum allowed size exceeded"
    index = pd.Index(np.arange(5, dtype=dtype))
    # pandas < 2 has no index of a small integer dtype: it stores int8 and
    # int16 as int64 and uint8 and uint16 as uint64, and an index of dtype
    # uint64 is extended with uint64 labels
    expected_dtype = np.uint64 if index.dtype == np.uint64 else np.int64
    loc, new_index, expanded = tsa_model.get_index_loc(key, index)
    assert loc == key
    assert expanded
    assert new_index.equals(pd.Index(np.arange(key + 1, dtype=expected_dtype)))
    assert new_index.dtype == expected_dtype

    endog = pd.Series(np.zeros(5), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start, end, out_of_sample, prediction_index = tsa_model.get_prediction_index(
        3, key, 5, mod._index, data=mod.data
    )
    assert (start, end, out_of_sample) == (3, 4, key - 4)
    expected = pd.Index(np.arange(3, key + 1, dtype=expected_dtype))
    assert prediction_index.equals(expected)


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2000-01-01", periods=5, freq="YS"),
        pd.date_range("2000-01-01", periods=8, freq="2QS"),
        # Dates in the second quarter of 2002 and in 2001, but not in 2001Q2
        pd.date_range("2000-01-01", periods=8, freq="3QS"),
    ],
)
@pytest.mark.parametrize("which", ["start", "end"])
def test_quarter_key_without_date(index, which):
    # The row labels are the model's own dates, so a quarter without a date is
    # not looked up again with pandas' quarter parsing, which found a date of
    # another quarter, e.g. 2000-07-01 for 2001Q2 with semiannual dates
    assert not ((index >= "2001-04-01") & (index < "2001-07-01")).any()
    endog = pd.Series(np.zeros(len(index)), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start, end = ("2001Q2", None) if which == "start" else (0, "2001Q2")
    with pytest.raises(KeyError, match=f"The `{which}` argument could not"):
        tsa_model.get_prediction_index(
            start, end, len(index), mod._index, data=mod.data
        )


def test_quarter_key_row_label_retry_other_keys():
    # Row labels that differ from the dates are still searched for a quarter
    # key that has dates
    dates = pd.date_range("2000-01-01", periods=12, freq="QS")
    labels = pd.Index(
        [f"{year}Q{q}" for year in (2000, 2001, 2002) for q in range(1, 5)]
    )
    endog = pd.Series(np.zeros(12), index=labels)
    mod = tsa_model.TimeSeriesModel(endog, dates=dates)
    start, end, _, _ = tsa_model.get_prediction_index(
        "2001Q2", None, 12, mod._index, data=mod.data
    )
    assert dates[start] == pd.Timestamp("2001-04-01")
    assert end == 11
    # A key that is not a quarter string is looked up in the row labels even
    # if no date lies in its quarter
    years = pd.date_range("2000-01-01", periods=3, freq="YS")
    endog = pd.Series(np.zeros(3), index=["2000-06-30", "2001-06-30", "2002-06-30"])
    mod = tsa_model.TimeSeriesModel(endog, dates=years)
    loc, _, _ = tsa_model.get_index_label_loc(
        "2001-06-30", mod._index, mod.data.row_labels
    )
    assert loc == 1
    # Keys that are not quarter strings, or other indexes, raise as before
    with pytest.raises(KeyError):
        tsa_model.get_index_label_loc(pd.Timestamp("2000-01-15"), dates, dates)
    endog = pd.Series(np.zeros(12))
    mod = tsa_model.TimeSeriesModel(endog)
    with pytest.raises(KeyError, match="The `start` argument could not"):
        tsa_model.get_prediction_index("2001Q2", None, 12, mod._index, data=mod.data)


# An out-of-sample date string as `end` covers its period: the kind and the
# frequency of the index, its first date and its length, the key, and the
# last date of prediction, written out
OOS_END_KEYS = [
    # monthly data that end on 2001-12: a quarter or a year ends at its last
    # month, a month is a single date
    ("date", "MS", "2000-01-01", 24, "2002Q2", "2002-06-01"),
    ("date", "MS", "2000-01-01", 24, "2002", "2002-12-01"),
    ("date", "MS", "2000-01-01", 24, "2002-03", "2002-03-01"),
    ("date", MONTH_END, "2000-01-31", 24, "2002Q2", "2002-06-30"),
    ("date", MONTH_END, "2000-01-31", 24, "2002-03", "2002-03-31"),
    # daily data that end on 2002-09-26
    ("date", "D", "2000-01-01", 1000, "2003-06", "2003-06-30"),
    ("date", "D", "2000-01-01", 1000, "2002Q4", "2002-12-31"),
    ("date", "D", "2000-01-01", 1000, "2003-06-30", "2003-06-30"),
    # business days that end on 2003-01-24; 2003-05-31 is a Saturday
    ("date", "B", "2000-01-03", 800, "2003-05", "2003-05-30"),
    # Sundays that end on 2002-11-10
    ("date", "W", "2000-01-02", 150, "2003-06", "2003-06-29"),
    # quarterly data that end on 2002-10-01 or 2002-12-31: a quarter is a
    # single date, and a month without a date keeps the next date
    ("date", "QS", "2000-01-01", 12, "2003", "2003-10-01"),
    ("date", "QS", "2000-01-01", 12, "2003Q2", "2003-04-01"),
    ("date", "QS", "2000-01-01", 12, "2003-05", "2003-07-01"),
    ("date", QUARTER_END, "2000-03-31", 12, "2003", "2003-12-31"),
    ("date", QUARTER_END, "2000-03-31", 12, "2003Q2", "2003-06-30"),
    # periods: a finer key is the period that contains it
    ("period", "M", "2000-01", 24, "2002Q2", "2002-06"),
    ("period", "M", "2000-01", 24, "2002", "2002-12"),
    ("period", "Q", "2000Q1", 12, "2003", "2003Q4"),
    ("period", "Q", "2000Q1", 12, "2003-05", "2003Q2"),
    ("period", "D", "2000-01-01", 1000, "2003-06", "2003-06-30"),
]


@pytest.mark.parametrize("kind, freq, first, nobs, key, last", OOS_END_KEYS)
def test_oos_end_key_covers_period(kind, freq, first, nobs, key, last):
    # Before, prediction ended at the first date of the period
    if kind == "date":
        index = pd.date_range(first, periods=nobs, freq=freq)
        expected = pd.date_range(index[-2], last, freq=freq)
    else:
        index = pd.period_range(first, periods=nobs, freq=freq)
        expected = pd.period_range(index[-2], last, freq=freq)
    res = constant_mean_results(pd.Series(np.zeros(nobs), index=index))
    pred = res.predict(start=nobs - 2, end=key)
    assert pred.index.equals(expected)
    assert_allclose(pred, np.full(len(expected), 2.5))


@pytest.mark.parametrize("kind", ["date", "period"])
def test_oos_end_key_start(kind):
    # A start key keeps the first date of its period
    if kind == "date":
        index = pd.date_range("2000-01-01", periods=24, freq="MS")
        half = pd.date_range("2002-01-01", "2002-06-01", freq="MS")
        year = pd.date_range("2002-01-01", "2002-12-01", freq="MS")
    else:
        index = pd.period_range("2000-01", periods=24, freq="M")
        half = pd.period_range("2002-01", "2002-06", freq="M")
        year = pd.period_range("2002-01", "2002-12", freq="M")
    res = constant_mean_results(pd.Series(np.zeros(24), index=index))
    pred = res.predict(start="2002Q1", end="2002Q2")
    assert pred.index.equals(half)
    pred = res.predict(start="2002", end="2002")
    assert pred.index.equals(year)


@pytest.mark.parametrize("kind", ["date", "period"])
@pytest.mark.parametrize("nobs", [24, 23, 22])
@pytest.mark.parametrize("key", ["2001Q2", "2001Q4", "2001", "2001-03"])
def test_in_sample_end_key_unchanged(kind, nobs, key):
    # A key whose period starts in the sample, also one that ends after it
    # (e.g. "2001Q4" for 23 or 22 months), ends prediction at the last date
    # that the index lookup gives for it
    if kind == "date":
        index = pd.date_range("2000-01-01", periods=nobs, freq="MS")
    else:
        index = pd.period_range("2000-01", periods=nobs, freq="M")
    res = constant_mean_results(pd.Series(np.zeros(nobs), index=index))
    loc, _, _ = tsa_model.get_index_label_loc(key, index, index)
    last = loc.stop - 1 if isinstance(loc, slice) else loc
    pred = res.predict(start=0, end=key)
    assert pred.index.equals(index[: last + 1])


@pytest.mark.parametrize(
    "key, last", [("2002Q2", "2002-06-01"), ("2002-03", "2002-03-01")]
)
def test_oos_end_key_tz_aware(key, last):
    # The period of the key is a local time of the index; this raised a
    # TypeError (comparison of tz-naive and tz-aware dates)
    index = pd.date_range("2000-01-01", periods=24, freq="MS", tz="US/Eastern")
    res = constant_mean_results(pd.Series(np.zeros(24), index=index))
    pred = res.predict(start=22, end=key)
    expected = pd.date_range("2001-11-01", last, freq="MS", tz="US/Eastern")
    assert pred.index.equals(expected)


@pytest.mark.parametrize("kind", ["date", "period"])
@pytest.mark.parametrize("key", ["1999", "NaT", ""])
def test_oos_end_key_no_date(kind, key):
    # A key before the sample or a missing date raises as before
    if kind == "date":
        index = pd.date_range("2000-01-01", periods=24, freq="MS")
    else:
        index = pd.period_range("2000-01", periods=24, freq="M")
    res = constant_mean_results(pd.Series(np.zeros(24), index=index))
    error, match = KeyError, "The `end` argument could not be"
    if PD_LT_2 and kind == "period" and key == "":
        # pandas < 2 cannot parse "" as a period (pandas >= 2 gives NaT), so
        # its DateParseError, a ValueError, is raised there, as before
        error, match = ValueError, "Unable to parse datetime string"
    with pytest.raises(error, match=match):
        res.predict(start=0, end=key)


def test_oos_end_key_errors_and_no_freq():
    # A string that is not a date raises as before, "today", which pandas
    # parses as a date but not as a period, is the current date as before,
    # and without a frequency a string is its first date as before
    index = pd.date_range("2000-01-01", periods=24, freq="MS")
    res = constant_mean_results(pd.Series(np.zeros(24), index=index))
    # pandas' message depends on its version
    with pytest.raises(ValueError, match="string"):
        res.predict(start=0, end="abc")
    by_string = res.predict(start=0, end="today")
    by_date = res.predict(start=0, end=pd.Timestamp("today"))
    assert by_string.index.equals(by_date.index)

    no_freq = pd.DatetimeIndex(index.tolist())
    assert no_freq.freq is None
    data = res.model.data
    by_string = tsa_model.get_prediction_index(0, "2002Q2", 24, no_freq, data=data)
    by_date = tsa_model.get_prediction_index(
        0, pd.Timestamp("2002-04-01"), 24, no_freq, data=data
    )
    assert by_string[:3] == by_date[:3]
    assert by_string[3].equals(by_date[3])


def test_end_key_not_a_date_string():
    # An integer is a location, also if it reads as a year, and a string is
    # a row label if the dates have no frequency
    index = pd.date_range("2000-01-01", periods=1000, freq="D")
    res = constant_mean_results(pd.Series(np.zeros(1000), index=index))
    pred = res.predict(start=998, end=2005)
    assert pred.index.equals(pd.date_range(index[998], periods=1008, freq="D"))

    dates = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04", "2000-01-07"])
    with pytest.warns(ValueWarning, match="no associated frequency"):
        res = constant_mean_results(pd.Series(np.zeros(4), index=dates))
    pred = res.predict(start=0, end="2000-01-04")
    assert pred.index.equals(dates[:3])


@pytest.mark.parametrize("model", ["AutoReg", "ARIMA", "ETSModel"])
def test_oos_end_key_models(model):
    # Through the predict methods of the models, with the predictions at the
    # integer locations of the same dates
    rng = np.random.default_rng(20261122)
    index = pd.date_range("2000-01-01", periods=36, freq="MS")
    y = pd.Series(10 + rng.standard_normal(36), index=index)
    if model == "AutoReg":
        res = AutoReg(y, 2, trend="c").fit()
    elif model == "ARIMA":
        res = ARIMA(y, order=(1, 0, 0)).smooth([10.0, 0.5, 1.0])
    else:
        # ndarray data with dates; predictions are ndarrays
        res = ETSModel(y.to_numpy(), dates=index).smooth([0.3, 10.0])
    pred = res.predict(start="2002-11", end="2003Q2")
    expected_index = pd.date_range("2002-11-01", "2003-06-01", freq="MS")
    assert res.model.data.predict_dates.equals(expected_index)
    if model != "ETSModel":
        assert pred.index.equals(expected_index)
    assert_allclose(
        np.asarray(pred), np.asarray(res.predict(start=34, end=41)), rtol=1e-12
    )


ROW_LABEL_DATES = pd.DatetimeIndex(
    ["2000-01-03", "2000-01-10", "2000-01-24", "2000-03-06", "2000-03-27"]
)


@pytest.mark.parametrize(
    "start, end, expected_start, expected_end",
    [
        # All dates of January, not only the first one
        (0, "2000-01", 0, 2),
        (1, "2000-01", 1, 2),
        ("2000-01", "2000-03", 0, 4),
        ("2000-01", "2000", 0, 4),
        # Without an end, prediction runs from the first date of the start key
        ("2000-03", None, 3, 4),
    ],
)
def test_row_label_partial_date_keys(start, end, expected_start, expected_end):
    # Dates without a frequency are only row labels, and a partial date string
    # that matches several of them gives a slice
    endog = pd.Series(np.zeros(5), index=ROW_LABEL_DATES)
    with pytest.warns(ValueWarning, match="no associated frequency"):
        mod = tsa_model.TimeSeriesModel(endog)
    start_loc, end_loc, out_of_sample, prediction_index = (
        tsa_model.get_prediction_index(
            start,
            end,
            5,
            mod._index,
            index_generated=mod._index_generated,
            data=mod.data,
        )
    )
    assert (start_loc, end_loc, out_of_sample) == (expected_start, expected_end, 0)
    expected = ROW_LABEL_DATES[expected_start : expected_end + 1]
    assert prediction_index.equals(expected)


def test_row_label_repeated_key():
    # A key that matches consecutive row labels covers all of them as the end
    endog = pd.Series(np.zeros(4), index=pd.Index(["a", "b", "b", "c"]))
    with pytest.warns(ValueWarning, match="An unsupported index was provided"):
        # pandas >= 2 warns that it cannot infer the format of strings
        with warns_or_none(UserWarning, match="Could not infer format"):
            mod = tsa_model.TimeSeriesModel(endog)
    start, end, _, prediction_index = tsa_model.get_prediction_index(
        0, "b", 4, mod._index, index_generated=mod._index_generated, data=mod.data
    )
    assert (start, end) == (0, 2)
    assert prediction_index.equals(pd.Index(["a", "b", "b"]))
    start, end, _, prediction_index = tsa_model.get_prediction_index(
        "b", None, 4, mod._index, index_generated=mod._index_generated, data=mod.data
    )
    assert (start, end) == (1, 3)
    assert prediction_index.equals(pd.Index(["b", "b", "c"]))


def test_get_index_label_loc_row_label_slice():
    endog = pd.Series(np.zeros(5), index=ROW_LABEL_DATES)
    with pytest.warns(ValueWarning, match="no associated frequency"):
        mod = tsa_model.TimeSeriesModel(endog)
    loc, index, expanded = tsa_model.get_index_label_loc(
        "2000-01", mod._index, mod.data.row_labels
    )
    assert loc == slice(0, 3)
    assert index.equals(ROW_LABEL_DATES[:3])
    assert not expanded


@pytest.mark.parametrize(
    "index, start, expected",
    [
        # A month of daily dates and daily periods
        (
            pd.date_range("2000-01-01", periods=100, freq="D"),
            "2000-02",
            pd.date_range("2000-02-01", "2000-04-09", freq="D"),
        ),
        (
            pd.period_range("2000-01-01", periods=100, freq="D"),
            "2000-02",
            pd.period_range("2000-02-01", "2000-04-09", freq="D"),
        ),
        # pandas also gives a slice for a month of a month-start index
        (
            pd.date_range("2000-01-01", periods=12, freq="MS"),
            "2000-03",
            pd.date_range("2000-03-01", "2000-12-01", freq="MS"),
        ),
    ],
)
def test_partial_date_start_without_end(index, start, expected):
    # A start key that covers several dates, without an end, predicts from
    # the first of these dates to the end of the sample
    endog = pd.Series(np.zeros(len(index)), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    nobs = len(index)
    start, end, out_of_sample, prediction_index = tsa_model.get_prediction_index(
        start, None, nobs, mod._index, data=mod.data
    )
    assert start == nobs - len(expected)
    assert end == nobs - 1
    assert out_of_sample == 0
    assert prediction_index.equals(expected)


QUARTER_KEY_CASES = {
    "daily": (
        pd.date_range("2000-01-01", periods=800, freq="D"),
        pd.date_range("2001-04-01", "2001-09-30", freq="D"),
    ),
    "month-start": (
        pd.date_range("2000-01-01", periods=36, freq="MS"),
        pd.date_range("2001-04-01", "2001-09-01", freq="MS"),
    ),
    "quarter-start": (
        pd.date_range("2000-01-01", periods=12, freq="QS"),
        pd.date_range("2001-04-01", "2001-07-01", freq="QS"),
    ),
    "quarter-end": (
        pd.date_range("2000-03-31", periods=12, freq=pd.offsets.QuarterEnd()),
        pd.date_range("2001-06-30", "2001-09-30", freq=pd.offsets.QuarterEnd()),
    ),
    "monthly-periods": (
        pd.period_range("2000-01", periods=36, freq="M"),
        pd.period_range("2001-04", "2001-09", freq="M"),
    ),
    "quarterly-periods": (
        pd.period_range("2000Q1", periods=12, freq="Q"),
        pd.period_range("2001Q2", "2001Q3", freq="Q"),
    ),
}


@pytest.mark.parametrize("case", list(QUARTER_KEY_CASES))
def test_quarter_string_keys_in_sample(case):
    # A quarter string selects the dates or periods of the whole quarter, as
    # pandas' partial-string indexing does for e.g. "2001-04"
    index, expected = QUARTER_KEY_CASES[case]
    endog = pd.Series(np.zeros(len(index)), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    nobs = len(index)
    start, end, out_of_sample, prediction_index = tsa_model.get_prediction_index(
        "2001Q2", "2001Q3", nobs, mod._index, data=mod.data
    )
    assert prediction_index.equals(expected)
    assert out_of_sample == 0
    assert end - start + 1 == len(expected)
    assert index[start] == expected[0]

    # Without an end, prediction runs from the start of the quarter to the
    # end of the sample
    start, end, _, prediction_index = tsa_model.get_prediction_index(
        "2001Q2", None, nobs, mod._index, data=mod.data
    )
    assert index[start] == expected[0]
    assert end == nobs - 1


@pytest.mark.parametrize(
    "index, start, end, expected",
    [
        # A month as the start and a quarter as the end
        (
            pd.date_range("2000-01-01", periods=36, freq="MS"),
            "2001-01",
            "2001Q2",
            pd.date_range("2001-01-01", "2001-06-01", freq="MS"),
        ),
        # A quarter that is only partly in the sample
        (
            pd.date_range("2000-01-01", periods=17, freq="MS"),
            "2001Q1",
            "2001Q2",
            pd.date_range("2001-01-01", "2001-05-01", freq="MS"),
        ),
        (
            pd.date_range("2000-01-01", periods=500, freq="D"),
            "2001Q2",
            "2001-05-14",
            pd.date_range("2001-04-01", "2001-05-14", freq="D"),
        ),
    ],
)
def test_quarter_string_keys_mixed(index, start, end, expected):
    endog = pd.Series(np.zeros(len(index)), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    out = tsa_model.get_prediction_index(
        start, end, len(index), mod._index, data=mod.data
    )
    assert out[3].equals(expected)
    assert out[2] == 0


@pytest.mark.parametrize(
    "index, loc",
    [
        (pd.date_range("2000-01-01", periods=12, freq="QS"), 5),
        (pd.date_range("2000-03-31", periods=12, freq=pd.offsets.QuarterEnd()), 5),
        (pd.period_range("2000Q1", periods=12, freq="Q"), 5),
        # The fiscal quarter of quarters ending in November
        (pd.period_range("2000Q1", periods=12, freq="Q-NOV"), 5),
    ],
)
def test_quarter_string_key_single_date(index, loc):
    # A quarter with one observation gives its location, not a slice
    result, _, _ = tsa_model.get_index_loc("2001Q2", index)
    assert not isinstance(result, slice)
    assert result == loc
    assert (index[loc].year, index[loc].quarter) == (2001, 2)


@pytest.mark.parametrize(
    "index, key",
    [
        # Before the sample
        (pd.date_range("2000-01-01", periods=800, freq="D"), "1999Q4"),
        (pd.date_range("2000-01-01", periods=12, freq="QS"), "1999Q4"),
        # Year-end dates have no date in the second quarter
        (pd.date_range("1999-12-31", periods=6, freq=YEAR_END), "2001Q2"),
    ],
)
def test_quarter_string_key_without_dates(index, key):
    with pytest.raises(KeyError):
        tsa_model.get_index_loc(key, index)


def test_quarter_string_keys_autoreg():
    from statsmodels.tsa.ar_model import AutoReg

    rng = np.random.default_rng(20261121)
    index = pd.date_range("2000-01-01", periods=36, freq="MS")
    res = AutoReg(pd.Series(rng.standard_normal(36), index=index), 1).fit()
    pred = res.predict(start="2001-01", end="2001Q2")
    assert pred.index.equals(pd.date_range("2001-01-01", "2001-06-01", freq="MS"))
    np.testing.assert_allclose(pred, res.predict(start=12, end=17), rtol=1e-13)

    # Dynamic prediction from a quarter of quarterly data
    index = pd.date_range("2000-03-31", periods=12, freq=pd.offsets.QuarterEnd())
    res = AutoReg(pd.Series(rng.standard_normal(12), index=index), 1).fit()
    np.testing.assert_allclose(
        res.predict(start=2, dynamic="2001Q2"),
        res.predict(start=2, dynamic=3),
        rtol=1e-13,
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2000-01-01", periods=8, freq="MS"),
        pd.period_range("2000-01", periods=8, freq="M"),
        pd.RangeIndex(8),
        pd.Index(np.arange(8, dtype=np.int64)),
    ],
    ids=["datetime", "period", "range", "integer"],
)
def test_negative_location_before_start(index):
    # A negative location before the start raises a KeyError for every
    # index, while -nobs is the first location
    with pytest.raises(KeyError):
        tsa_model.get_index_loc(-9, index)
    loc, prediction_index, _ = tsa_model.get_index_loc(-8, index)
    assert loc == 0
    assert prediction_index.equals(index[:1])

    mod = tsa_model.TimeSeriesModel(pd.Series(np.zeros(8), index=index))
    with pytest.raises(KeyError, match="The `start` argument could not"):
        tsa_model.get_prediction_index(-20, None, 8, mod._index, data=mod.data)


@pytest.mark.parametrize("key", ["2000-01-01", "a"])
def test_label_key_without_row_labels(key):
    # A label key for numpy data, which have no row labels, raises a KeyError
    with pytest.raises(KeyError):
        tsa_model.get_index_label_loc(key, pd.RangeIndex(8), None)

    mod = tsa_model.TimeSeriesModel(np.arange(8.0))
    assert mod.data.row_labels is None
    with pytest.raises(KeyError, match="The `start` argument could not"):
        tsa_model.get_prediction_index(key, None, 8, mod._index, data=mod.data)
    with pytest.raises(KeyError, match="The `end` argument could not"):
        tsa_model.get_prediction_index(0, key, 8, mod._index, data=mod.data)


@pytest.mark.parametrize(
    "dates, freq",
    [
        # pd.date_range("2000-01-01", "2000-01-05", freq="D") has 5 dates
        (["2000-01-01", "2000-01-02", "2000-01-05"], "D"),
        (pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-05"]), "D"),
        # pd.date_range("2000-01-01", "2000-01-03", freq="2D") has 2 dates
        (["2000-01-01", "2000-01-02", "2000-01-03"], "2D"),
        # As many dates, but not the same ones
        (["2000-01-01", "2000-01-02", "2000-01-05"], "2D"),
    ],
)
def test_freq_not_matching_dates(dates, freq):
    with pytest.raises(ValueError, match="The given frequency argument could not"):
        tsa_model.TimeSeriesModel(np.arange(3.0), dates=dates, freq=freq)


FINER_KEY_CASES = [
    # Keys within one period, which raised a KeyError in the sample
    (pd.period_range("2000Q1", periods=8, freq="Q"), "2001-05-15", "2001Q2"),
    (pd.period_range("2000Q1", periods=8, freq="Q"), "2001-05", "2001Q2"),
    # The last month of a quarter
    (pd.period_range("2000Q1", periods=8, freq="Q"), "2001-06", "2001Q2"),
    (
        pd.period_range("2000Q1", periods=8, freq="Q"),
        pd.Period("2001-05", freq="M"),
        "2001Q2",
    ),
    (pd.period_range("2000-01", periods=24, freq="M"), "2001-05-15 10:00", "2001-05"),
    (pd.period_range("1998", periods=6, freq="Y"), "2001Q2", "2001"),
    (
        pd.period_range("2000-01-01", periods=60, freq="D"),
        "2000-02-09 10:00",
        "2000-02-09",
    ),
    (pd.period_range("2000-01-02", periods=60, freq="W"), "2000-02-09", "2000-02-09"),
    # Keys that were already located
    (pd.period_range("2000Q1", periods=8, freq="Q"), "2001Q2", "2001Q2"),
    (
        pd.period_range("2000Q1", periods=8, freq="Q"),
        pd.Timestamp("2001-05-15"),
        "2001Q2",
    ),
    (pd.period_range("2000-01", periods=24, freq="M"), "2001-05", "2001-05"),
]


@pytest.mark.parametrize("index, key, period", FINER_KEY_CASES)
def test_period_index_key_within_period(index, key, period):
    # A key within one period of a PeriodIndex is that period, in the sample
    # as after its end
    expected = pd.Period(period, freq=index.freq)
    loc, _, expanded = tsa_model.get_index_loc(key, index)
    assert not isinstance(loc, slice)
    assert index[loc] == expected
    assert not expanded

    # The same key one year after the end of the sample
    if isinstance(key, str):
        later = key.replace(str(expected.year), str(expected.year + 6))
        loc, extended, expanded = tsa_model.get_index_loc(later, index)
        assert expanded
        assert extended[loc] == pd.Period(
            period.replace(str(expected.year), str(expected.year + 6)),
            freq=index.freq,
        )


def test_period_index_finer_keys_prediction():
    index = pd.period_range("2000Q1", periods=8, freq="Q")
    mod = tsa_model.TimeSeriesModel(pd.Series(np.zeros(8), index=index))
    out = tsa_model.get_prediction_index(
        "2001-05-15", "2002-11-30", 8, mod._index, data=mod.data
    )
    assert out[3].equals(pd.period_range("2001Q2", "2002Q4", freq="Q"))
    assert out[2] == 4


def test_period_index_coarser_key():
    # A key that covers several periods is a slice of them
    index = pd.period_range("2000-01", periods=24, freq="M")
    loc, _, _ = tsa_model.get_index_loc("2001", index)
    assert index[loc].equals(pd.period_range("2001-01", "2001-12", freq="M"))


@pytest.mark.parametrize("key", ["", "NaT"])
@pytest.mark.parametrize("which", ["start", "end"])
def test_period_index_nat_key_raises_key_error(key, which):
    # A key that pandas parses as NaT on a PeriodIndex raises the KeyError
    # of an unmatched key, not an AttributeError from NaT
    from statsmodels.tsa.ar_model import AutoReg

    if key == "" and PD_LT_2:
        # pandas < 2 raises a parse error for "" instead of returning NaT
        pytest.skip('pandas < 2 does not parse "" as NaT')
    index = pd.period_range("2000-01", periods=24, freq="M")
    y = pd.Series(np.arange(24.0) + np.sin(np.arange(24.0)), index=index)
    res = AutoReg(y, 1).fit()
    kwargs = {"start": 1, "end": 30}
    kwargs[which] = key
    with pytest.raises(KeyError, match="could not be matched"):
        res.predict(**kwargs)


@pytest.mark.parametrize(
    "key, loc",
    [
        # The strings give the seconds since pandas < 2.2 cannot parse a date
        # string with a UTC offset without them (KeyError: 14)
        ("2002-03-01T00:00:00-05:00", 26),
        ("2002-03-01T05:00:00Z", 26),
        (pd.Timestamp("2002-03-01T05:00", tz="UTC"), 26),
        (pd.Timestamp("2002-03-01T14:00", tz="Asia/Tokyo"), 26),
        # Not a date of the index: extended through the next one
        (pd.Timestamp("2002-01-15", tz="UTC"), 25),
    ],
    ids=["str-offset", "str-utc", "utc", "other-tz", "utc-between-dates"],
)
def test_get_index_loc_tz_aware_key_after_sample(key, loc):
    # A key after the end of a tz-aware index in another time zone, or with
    # a UTC offset, is the same date as the key converted to the time zone of
    # the index; date_range raised a TypeError for it
    tz = "US/Eastern"
    index = pd.date_range("2000-01-01", periods=24, freq="MS", tz=tz)
    expected = pd.date_range("2000-01-01", periods=loc + 1, freq="MS", tz=tz)
    result = tsa_model.get_index_loc(key, index)
    assert result[0] == loc
    assert result[1].equals(expected)
    assert result[2] is True
    converted = tsa_model.get_index_loc(pd.Timestamp(key).tz_convert(tz), index)
    assert converted[0] == loc
    assert converted[1].equals(expected)

    endog = pd.Series(np.zeros(24), index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start, end, out_of_sample, prediction_index = tsa_model.get_prediction_index(
        20, key, 24, mod._index, data=mod.data
    )
    assert (start, end, out_of_sample) == (20, 23, loc - 23)
    assert prediction_index.equals(expected[20:])


@pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo"])
@pytest.mark.parametrize(
    "freq, nobs", [("D", 500), ("MS", 24), ("QS", 12), ("h", 24 * 200)]
)
def test_quarter_string_key_tz_aware_index(tz, freq, nobs):
    # The bounds of the quarter of a key such as "2000Q2" are local times of
    # a tz-aware index; comparing them with it as naive timestamps raised a
    # TypeError
    index = pd.date_range("2000-01-01", periods=nobs, freq=freq, tz=tz)
    loc, new_index, expanded = tsa_model.get_index_loc("2000Q2", index)
    # The dates of the second quarter of 2000 in local time (the hourly
    # US/Eastern dates skip the hour lost to daylight saving time on April 2)
    expected = index[(index.year == 2000) & (index.quarter == 2)]
    if len(expected) == 1:
        # A single date is an exact match
        assert loc == index.get_loc(expected[0])
    else:
        assert index[loc].equals(expected)
    assert not expanded
    assert new_index.equals(index[: index.get_loc(expected[-1]) + 1])


def test_quarter_string_keys_tz_aware_prediction():
    index = pd.date_range("2000-01-01", periods=500, freq="D", tz="US/Eastern")
    res = constant_mean_results(pd.Series(np.zeros(500), index=index))
    pred = res.predict(start="2000Q2", end="2000Q3")
    expected = pd.date_range("2000-04-01", "2000-09-30", freq="D", tz="US/Eastern")
    assert pred.index.equals(expected)


@pytest.mark.parametrize(
    "start, end, expected_start, expected_end",
    [
        # No date in February: the end is the last date before it
        (0, "2000-02", 0, 2),
        # and the start is the first date after it
        ("2000-02", None, 3, 4),
    ],
)
def test_row_label_empty_partial_date_key_deprecated(
    start, end, expected_start, expected_end
):
    # Dates without a frequency are only row labels. pandas gives an empty
    # slice for a partial date string between two of them, which was used
    # silently, as for dates with a frequency
    endog = pd.Series(np.zeros(5), index=ROW_LABEL_DATES)
    with pytest.warns(ValueWarning, match="no associated frequency"):
        mod = tsa_model.TimeSeriesModel(endog)
    with pytest.warns(FutureWarning, match="No date of the index matches") as rec:
        start_loc, end_loc, out_of_sample, prediction_index = (
            tsa_model.get_prediction_index(
                start,
                end,
                5,
                mod._index,
                index_generated=mod._index_generated,
                data=mod.data,
            )
        )
    # The warning points at the caller outside statsmodels
    assert rec[0].filename == __file__
    assert (start_loc, end_loc, out_of_sample) == (expected_start, expected_end, 0)
    expected = ROW_LABEL_DATES[expected_start : expected_end + 1]
    assert prediction_index.equals(expected)


def test_row_label_empty_partial_date_key_deprecated_prediction():
    y = pd.Series([0.3, 1.2, 1.9, 3.4, 3.8], index=ROW_LABEL_DATES)
    with pytest.warns(ValueWarning, match="no associated frequency"):
        res = AutoReg(y, 1).fit()
    with pytest.warns(FutureWarning, match="No date of the index matches"):
        pred = res.predict(start="2000-02")
    # The first date after February
    expected = res.predict(start=3)
    assert pred.index.equals(expected.index)
    assert_allclose(pred, expected, rtol=1e-12)


def test_row_label_partial_date_key_with_dates_not_deprecated():
    # A partial date string that matches row labels gives a non-empty slice
    endog = pd.Series(np.zeros(5), index=ROW_LABEL_DATES)
    with pytest.warns(ValueWarning, match="no associated frequency"):
        mod = tsa_model.TimeSeriesModel(endog)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loc, index, _ = tsa_model.get_index_label_loc(
            "2000-03", mod._index, mod.data.row_labels
        )
    assert loc == slice(3, 5)
    assert index.equals(ROW_LABEL_DATES)
