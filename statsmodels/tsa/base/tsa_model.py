from __future__ import annotations

from statsmodels.compat.pandas import (
    _infer_freq_returns_offset,
    is_float_index,
    is_int_index,
    is_numeric_dtype,
    to_period_offset,
)

import re
import warnings

import numpy as np
import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
)
from pandas.tseries.frequencies import to_offset

from statsmodels.base.covtype import _stacklevel_outside_statsmodels
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning

_tsa_doc = """
    %(model)s

    %(params)s
    dates : array_like, optional
        The dates of the observations, used instead of the index of `endog`
        or `exog`. A DatetimeIndex or a PeriodIndex, or anything that
        ``pandas.to_datetime`` converts to a DatetimeIndex, such as a list
        of datetime objects or date strings. A ValueError is raised if the
        dates cannot be converted.
    freq : str, datetime.timedelta, or DateOffset, optional
        The frequency of the time series, as a pandas offset, a timedelta or
        an offset alias such as 'B', 'D', 'W', 'MS' or 'QS'. If it is not
        given and the dates have no frequency, it is inferred from the
        dates, with a ValueWarning. If none can be inferred, `dates` raise a
        ValueError, while a date index of `endog` or `exog` is ignored for
        forecasting, with a ValueWarning. If the dates have a frequency,
        `freq` must match it. A frequency cannot be given without dates or a
        pandas index.
    %(extra_params)s
    **kwargs
        Extra arguments passed to the base model, for example ``hasconst``,
        or used to set model properties when using the formula interface.
    %(extra_sections)s"""

_model_doc = "Timeseries model base class"

_generic_params = base._model_params_doc
_missing_param_doc = base._missing_param_doc


def get_index_loc(key, index):
    """
    Get the location of a specific key in an index

    .. versionchanged:: 0.15.1

       A key within one period of a PeriodIndex, such as a day of quarterly
       periods, is the location of that period in the sample, as it already
       was after the end of the sample, instead of raising a KeyError.

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or a location if the underlying index is a RangeIndex
        or an Index with an integer dtype.

        .. deprecated:: 0.15.1

           A partial date string that matches no date of the index but lies
           between two of its dates, such as "2000-02" for quarterly dates
           starting on 2000-01-01, gives an empty slice, so that prediction
           starts at the next date or ends at the previous one. This is
           deprecated and will raise a KeyError after statsmodels 0.17 is
           released.

        .. versionchanged:: 0.15.1

           A quarter string such as "2001Q2" matches all dates of the
           quarter in a DatetimeIndex (in its time zone if it is tz-aware),
           and all periods of the quarter in a PeriodIndex with a finer
           frequency, as pandas' partial-string indexing does for e.g.
           "2001-03". Previously it was taken as the last day of the quarter,
           or did not match periods.
    index : pd.Index
        The index to search.

        .. versionchanged:: 0.15.1

           An Index with dtype uint64 is extended with uint64 labels
           instead of float labels.

    Returns
    -------
    loc : int or slice
        The location of the key. A slice if `key` is a partial date string,
        for example "2000-03" for a daily index.
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    If `key` is past the end of the given index, and the index is either
    an Index with an integral dtype or a date index, this function extends
    the index up to and including key, and then returns the location in the
    new index.

    An Index with an integer dtype (other than a RangeIndex) must have the
    values 0, 1, ..., nobs - 1 to be extended to an out-of-sample `key`.
    This is the only such index that a `TimeSeriesModel` uses.
    """
    base_index = index

    index = base_index
    date_index = isinstance(base_index, (PeriodIndex, DatetimeIndex))
    int_index = is_int_index(base_index)
    range_index = isinstance(base_index, RangeIndex)
    index_class = type(base_index)
    nobs = len(index)

    # Special handling for RangeIndex
    if range_index and isinstance(key, (int, np.integer)):
        # Negative indices (that lie in the Index)
        if key < 0 and -key <= nobs:
            key = nobs + key
        # Out-of-sample (note that we include key itself in the new index)
        elif key > nobs - 1:
            stop = base_index.start + (key + 1) * base_index.step
            index = RangeIndex(
                start=base_index.start, stop=stop, step=base_index.step
            )

    # Special handling for NumericIndex
    if (
        not range_index
        and int_index
        and not date_index
        and isinstance(key, (int, np.integer))
    ):
        # Negative indices (that lie in the Index)
        if key < 0 and -key <= nobs:
            key = nobs + key
        # Out-of-sample (note that we include key itself in the new index)
        elif key > base_index[-1]:
            # NumPy computes the range in the dtype of a small integer index,
            # such as int8, which cannot hold the new labels, and
            # np.arange(np.uint64(0), int) gives floats, so keep uint64 and
            # use int64 otherwise
            dtype = np.uint64 if base_index.dtype == np.uint64 else np.int64
            index = Index(np.arange(int(base_index[0]), int(key + 1), dtype=dtype))

    # Special handling for date indexes
    if date_index:
        # Use index type to choose creation function
        if index_class is DatetimeIndex:
            index_fn = date_range
        else:
            index_fn = period_range
        # Integer key (i.e., already given a location)
        if isinstance(key, (int, np.integer)):
            # Negative indices (that lie in the Index)
            if key < 0 and -key < nobs:
                key = index[nobs + key]
            # Out-of-sample (note that we include key itself in the new
            # index)
            elif key > len(base_index) - 1:
                index = index_fn(
                    start=base_index[0],
                    periods=int(key + 1),
                    freq=base_index.freq,
                )
                key = index[-1]
            else:
                try:
                    key = index[key]
                except IndexError as exc:
                    # A negative location before the start of the sample
                    raise KeyError(str(exc)) from exc
        # Other key types (i.e., string date or some datetime-like object)
        else:
            # Convert the key to the appropriate date-like object
            if index_class is PeriodIndex:
                date_key = Period(key, freq=base_index.freq)
                # A key within one period, such as a day of quarterly data,
                # is that period, as it is after the end of the sample
                # (date_key is the period that contains the start of key)
                if (
                    isinstance(key, (str, Period))
                    and pd.notna(date_key)
                    and Period(key).end_time <= date_key.end_time
                ):
                    key = date_key
            elif isinstance(key, str) and re.fullmatch("[0-9]*Q[1-4]", key):
                # Special case dates with a format like
                # 2026Q2
                date_key = pd.Period(key).to_timestamp()
            else:
                date_key = Timestamp(key)
            # A date string is a local time of a tz-aware index, as in
            # pandas' string indexing (a no-op for a tz-naive index)
            if (
                isinstance(key, str)
                and index_class is DatetimeIndex
                and date_key.tz is None
            ):
                date_key = date_key.tz_localize(base_index.tz)

            # Out-of-sample
            if date_key > base_index[-1]:
                # date_range needs a tz-aware key in the time zone of the
                # index, so convert a key with another time zone or offset
                if index_class is DatetimeIndex and date_key.tz is not None:
                    date_key = date_key.tz_convert(base_index.tz)
                # First create an index that may not always include `key`
                index = index_fn(
                    start=base_index[0], end=date_key, freq=base_index.freq
                )

                # Now make sure we include `key`
                if index[-1] != date_key:
                    index = index_fn(
                        start=base_index[0],
                        periods=len(index) + 1,
                        freq=base_index.freq,
                    )

                # To avoid possible inconsistencies with `get_loc` below,
                # set the key directly equal to the last index location
                key = index[-1]

    # Get the location
    if date_index:
        # (note that get_loc will throw a KeyError if key is invalid)
        if (
            isinstance(index, DatetimeIndex)
            and isinstance(key, str)
            and re.fullmatch("[0-9]*Q[1-4]", key)
        ):
            # pandas deprecates parsing quarter strings as dates, so select
            # the dates in the quarter as its partial-string indexing does,
            # e.g. for "2001-03" on a daily index (in local time if the index
            # is tz-aware)
            period = Period(key)
            loc = index.slice_indexer(
                period.start_time.tz_localize(index.tz),
                period.end_time.tz_localize(index.tz),
            )
            if loc.start == loc.stop:
                raise KeyError(key)
            if loc.stop - loc.start == 1:
                # A single date, e.g. of quarterly data, is an exact match
                loc = loc.start
        else:
            loc = index.get_loc(key)
    elif int_index or range_index:
        # For NumericIndex and RangeIndex, key is assumed to be the location
        # and not an index value (this assumption is required to support
        # RangeIndex)
        try:
            index[key]
        # We want to raise a KeyError in this case, to keep the exception
        # consistent across index types.
        # - Attempting to index with an out-of-bound location (e.g.
        #   index[10] on an index of length 9) will raise an IndexError
        #   (as of Pandas 0.22)
        # - Attemtping to index with a type that cannot be cast to integer
        #   (e.g., a non-numeric string) will raise a ValueError if the
        #   index is RangeIndex (otherwise will raise an IndexError)
        #   (as of Pandas 0.22)
        except (IndexError, ValueError) as exc:
            raise KeyError(str(exc)) from exc
        loc = key
    else:
        loc = index.get_loc(key)

    # Check if we now have a modified index
    index_was_expanded = index is not base_index

    # Return the index through the end of the loc / slice
    if isinstance(loc, slice):
        if loc.start == loc.stop:
            # pandas gives an empty slice for a partial date string that lies
            # between two dates of the index
            warnings.warn(
                f"No date of the index matches the key {key!r}. Keys that "
                "match no date are deprecated and will raise a KeyError "
                "after statsmodels 0.17 is released.",
                FutureWarning,
                stacklevel=_stacklevel_outside_statsmodels(),
            )
        end = loc.stop - 1
    else:
        end = loc

    return loc, index[: end + 1], index_was_expanded


def get_index_label_loc(key, index, row_labels):
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or is only being used as row labels, or a location if
        the underlying index is a RangeIndex or a NumericIndex.

        .. versionchanged:: 0.15.1

           A quarter string such as "2001Q2" for which a DatetimeIndex
           `index` has no date raises a KeyError. Previously it was looked
           up in `row_labels` with pandas' quarter parsing, which could
           match a date of another quarter, such as 2000-07-01 for
           semiannual dates starting on 2000-01-01.
    index : pd.Index
        The index to search.
    row_labels : pd.Index
        Row labels to search if key not found in index

        .. versionchanged:: 0.15.1

           A key that matches several consecutive row labels, such as
           "2000-01" for dates without a frequency, gives the slice of all of
           them, as in `index`. Previously it gave the location of the first
           one, so that prediction with this key as `end` stopped there.

    Returns
    -------
    loc : int or slice
        The location of the key. A slice if `key` is a partial date string
        that matches several dates, for example "2000-03" for daily dates.
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    This function expands on `get_index_loc` by first trying the given
    base index (or the model's index if the base index was not given) and
    then falling back to try again with the model row labels as the base
    index.
    """
    try:
        loc, index, index_was_expanded = get_index_loc(key, index)
    except KeyError:
        # A quarter string such as "2001Q2" for which a date index has no
        # date is not looked up in the row labels, where pandas' deprecated
        # quarter parsing can find a date of another quarter
        if (
            isinstance(index, DatetimeIndex)
            and isinstance(key, str)
            and re.fullmatch("[0-9]*Q[1-4]", key)
        ):
            period = Period(key)
            in_quarter = (index.year == period.year) & (
                index.quarter == period.quarter
            )
            if not in_quarter.any():
                raise
        # Without row labels (e.g. numpy data) there is nothing else to search
        if not isinstance(key, (int, np.integer)) and row_labels is not None:
            loc = row_labels.get_loc(key)
        else:
            raise
        # Pandas may return a slice if there are multiple matching
        # locations that are monotonic increasing (otherwise it may
        # return an array of integer locations, see below). Keep the slice,
        # as get_index_loc does, so that a key such as "2000-01" covers all
        # of its dates as the end of prediction.
        if isinstance(loc, slice):
            return loc, row_labels[: loc.stop], False
        if isinstance(loc, np.ndarray):
            # Pandas may return a mask (boolean array), for e.g.:
            # pd.Index(list('abcb')).get_loc('b')
            if loc.dtype == bool:
                # Return the first True value
                # (we know there is at least one True value if we're
                # here because otherwise the get_loc call would have
                # raised an exception)
                loc = np.argmax(loc)
            # Finally, Pandas may return an integer array of
            # locations that match the given value, for e.g.
            # pd.DatetimeIndex(['2001-02', '2001-01']).get_loc('2001')
            # (this appears to be slightly undocumented behavior, since
            # only int, slice, and mask are mentioned in docs for
            # pandas.Index.get_loc as of 0.23.4)
            else:
                loc = loc[0]

        index = row_labels[: loc + 1]
        index_was_expanded = False

    return loc, index, index_was_expanded


def get_prediction_index(
    start,
    end,
    nobs,
    base_index,
    index=None,
    silent=False,
    index_none=False,
    index_generated=None,
    data=None,
) -> tuple[int, int, int, Index | None]:
    """
    Get the location of the start and end of prediction, and the associated index

    Parameters
    ----------
    start : label
        The key at which to start prediction. Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    end : label
        The key at which to end prediction (note that this key will be
        *included* in prediction). Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels. A date string that covers several dates of a
        date index with a frequency, such as "2002Q2" or "2002" for monthly
        data, ends prediction at the last of them in the index, or, if its
        period is after the sample, at the last date of the period in the
        frequency of the index.

        .. versionchanged:: 0.15.1

           A date string whose period is after the sample, such as "2002Q2"
           for monthly data that end in 2001, ends prediction at the last
           date of the period in the frequency of the index (2002-06).
           Previously prediction ended at the first date of the period
           (2002-04).
    nobs : int
        The number of observations in the model.
    base_index : pd.Index
        The base index of the model, against which `start` and `end` are
        resolved.
    index : array_like, optional
        Optionally an index to associate the predicted results to, converted
        to a pandas Index. If None, an attempt is made to create an index for
        the predicted results from the model's index or model's row labels.
    silent : bool, optional
        Argument to silence warnings.
    index_none : bool, optional
        Whether the returned prediction index should be forced to None
        even if a supported index would otherwise be available. Default
        is False.
    index_generated : bool, optional
        Whether the model's underlying index was internally generated
        rather than taken directly from `endog` / `exog`. Default is None.
    data : statsmodels.base.data.ModelData
        The model's data object, used to obtain row labels and to set the
        `predict_start`, `predict_end`, and `predict_dates` attributes. It
        is required even though it has a default of None.

    Returns
    -------
    start : int
        The index / observation location at which to begin prediction.
    end : int
        The index / observation location at which to end in-sample
        prediction. The maximum value for this is nobs-1.
    out_of_sample : int
        The number of observations to forecast after the end of the sample.
    prediction_index : pd.Index or None
        The index associated with the prediction results. This index covers
        the range [start, end + out_of_sample]. If the model has no given
        index and no given row labels (i.e., endog/exog is not Pandas), then
        this will be None.

    Notes
    -----
    The arguments `start` and `end` behave differently, depending on if
    they are integer or not. If either is an integer, then it is assumed
    to refer to a *location* in the index, not to an index value. On the
    other hand, if it is a date string or some other type of object, then
    it is assumed to refer to an index *value*. In all cases, the returned
    `start` and `end` values refer to index *locations* (so in the former
    case, the given location is validated and returned whereas in the
    latter case a location is found that corresponds to the given index
    value).

    This difference in behavior is necessary to support `RangeIndex`. This
    is because integers for a RangeIndex could refer either to index values
    or to index locations in an ambiguous way (while for `NumericIndex`,
    since we have required them to be full indexes, there is no ambiguity).

    .. versionchanged:: 0.15.1

       A quarter string such as "2001Q2" as `start` or `end` covers the
       whole quarter of a date index, so that prediction starts at its first
       date or ends at its last date in the sample.
    """

    # Convert index keys (start, end) to index locations and get associated
    # indexes.
    try:
        start, _, start_oos = get_index_label_loc(
            start, base_index, data.row_labels
        )
    except KeyError as exc:
        raise KeyError(
            "The `start` argument could not be matched to a"
            " location related to the index of the data."
        ) from exc
    # A key that covers more than one date (e.g. "2000-02" for a daily index)
    # gives a slice, of which prediction starts at the first date
    if isinstance(start, slice):
        start = start.start
    if end is None:
        end = max(start, len(base_index) - 1)
    try:
        end, end_index, end_oos = get_index_label_loc(
            _period_end(end, base_index), base_index, data.row_labels
        )
    except KeyError as exc:
        raise KeyError(
            "The `end` argument could not be matched to a"
            " location related to the index of the data."
        ) from exc

    # Handle slices (if the given index keys cover more than one date)
    if isinstance(end, slice):
        end = end.stop - 1

    # Get the actual index for the prediction
    prediction_index = end_index[start:]

    # Validate prediction options
    if end < start:
        raise ValueError("Prediction must have `end` after `start`.")

    # Handle custom prediction index
    # First, if we were given an index, check that it's the right size and
    # use it if so
    if index is not None:
        if not len(prediction_index) == len(index):
            raise ValueError(
                "Invalid `index` provided in prediction."
                " Must have length consistent with `start`"
                " and `end` arguments."
            )
        # But if we weren't given Pandas input, this index will not be
        # used because the data will not be wrapped; in that case, issue
        # a warning
        if not isinstance(data, PandasData) and not silent:
            warnings.warn(
                "Because the model data (`endog`, `exog`) were"
                " not given as Pandas objects, the prediction"
                " output will be Numpy arrays, and the given"
                " `index` argument will only be used"
                " internally.",
                ValueWarning,
                stacklevel=2,
            )
        prediction_index = Index(index)
    # Now, if we *do not* have a supported index, but we were given some
    # kind of index...
    elif index_generated and not index_none:
        # If we are in sample, and have row labels, use them
        if data.row_labels is not None and not (start_oos or end_oos):
            prediction_index = data.row_labels[start : end + 1]
        # Otherwise, warn the user that they will get an NumericIndex
        else:
            raise ValueError("No supported index is available.")
    elif index_none:
        prediction_index = None

    # For backwards compatibility, set `predict_*` values
    if prediction_index is not None:
        data.predict_start = prediction_index[0]
        data.predict_end = prediction_index[-1]
        data.predict_dates = prediction_index
    else:
        data.predict_start = None
        data.predict_end = None
        data.predict_dates = None

    # Compute out-of-sample observations
    out_of_sample = max(end - (nobs - 1), 0)
    end -= out_of_sample

    return start, end, out_of_sample, prediction_index


def _period_end(key, index):
    """
    The last date of the period of a date string key that is after index

    A date string such as "2002Q2" covers several dates of a monthly index.
    In the sample, partial-string indexing selects all of them. If the
    period starts after the index, this returns its last date in the
    frequency of the index, so that prediction can end there. Other keys,
    keys whose period starts in the index, and keys of a period without a
    date of the frequency are returned unchanged.
    """
    if not (
        isinstance(key, str)
        and isinstance(index, (DatetimeIndex, PeriodIndex))
        and index.freq is not None
    ):
        return key
    try:
        # The period of the resolution of the string, e.g. a quarter
        period = Period(key)
    except ValueError:
        period = None
    if not isinstance(period, Period):
        # Not a date of pandas' parser, e.g. "now", or NaT, e.g. ""
        return key
    if isinstance(index, PeriodIndex):
        if period.asfreq(index.freq, how="start") <= index[-1]:
            return key
        return period.asfreq(index.freq, how="end")
    start = period.start_time.tz_localize(index.tz)
    if start <= index[-1]:
        return key
    end = period.end_time.tz_localize(index.tz)
    last = date_range(start=index[0], end=end, freq=index.freq)[-1]
    # A period without a date, e.g. "2002-02" of quarterly dates, is left to
    # get_index_loc, which uses the first date after it
    return last if last >= start else key


class TimeSeriesModel(base.LikelihoodModel):
    __doc__ = _tsa_doc % {
        "model": _model_doc,
        "params": _generic_params,
        "extra_params": _missing_param_doc,
        "extra_sections": "",
    }

    def __init__(
        self, endog, exog=None, dates=None, freq=None, missing="none", **kwargs
    ):
        super().__init__(endog, exog, missing=missing, **kwargs)

        # Date handling in indexes
        self._init_dates(dates, freq)

    def _init_dates(self, dates=None, freq=None):
        """
        Initialize dates

        Parameters
        ----------
        dates : array_like, optional
            An array like object containing dates.
        freq : str, datetime.timedelta, or DateOffset, optional
            A frequency specification for either `dates` or the row labels from
            the endog / exog data.

        Notes
        -----
        Creates `self._index` and related attributes. `self._index` is always
        a Pandas index, and it is always NumericIndex, DatetimeIndex, or
        PeriodIndex.

        If Pandas objects, endog / exog may have any type of index. If it is
        a NumericIndex with values 0, 1, ..., nobs-1 or if it is (coercible to)
        a monotonic DatetimeIndex or PeriodIndex *with an associated
        frequency*, then it is called a "supported" index. A PeriodIndex must
        additionally consist of consecutive periods, i.e. without gaps or
        repeated periods. Otherwise it is called an "unsupported" index.

        Passing a PeriodIndex that is not consecutive as `dates` issues a
        warning and the index is ignored, whereas passing a DatetimeIndex as
        `dates` raises an exception if it has no frequency and none can be
        inferred.

        Supported indexes are standardized (i.e., a list of date strings is
        converted to a DatetimeIndex) and the result is put in `self._index`.

        Unsupported indexes are ignored, and a supported NumericIndex is
        generated and put in `self._index`. Warnings are issued in this case
        to alert the user if the returned index from some operation (e.g.
        forecasting) is different from the original data's index. However,
        whenever possible (e.g., purely in-sample prediction), the original
        index is returned.

        The benefit of supported indexes is that they allow *forecasting*, i.e.
        it is possible to extend them in a reasonable way. Thus every model
        must have an underlying supported index, even if it is just a generated
        NumericIndex.
        """

        # Get our index from `dates` if available, otherwise from whatever
        # Pandas index we might have retrieved from endog, exog
        if dates is not None:
            index = dates
        else:
            index = self.data.row_labels

        # Sanity check that we do not have a `freq` without an index
        if index is None and freq is not None:
            raise ValueError("Frequency provided without associated index.")

        # If an index is available, see if it is a date-based index or if it
        # can be coerced to one. (If it cannot we'll fall back, below, to an
        # internal, 0, 1, ... nobs-1 integer index for modeling purposes)
        inferred_freq = False
        if index is not None:
            # Try to coerce to date-based index
            if not isinstance(index, (DatetimeIndex, PeriodIndex)):
                try:
                    # Only try to coerce non-numeric index types (string,
                    # list of date-times, etc.)
                    # Note that np.asarray(Float64Index([...])) yields an
                    # object dtype array in earlier versions of Pandas (and so
                    # will not have is_numeric_dtype == True), so explicitly
                    # check for it here. But note also that in very early
                    # Pandas (~0.12), Float64Index does not exist (and so the
                    # statsmodels compat makes it an empty tuple, so in that
                    # case also check if the first element is a float.
                    _index = np.asarray(index)
                    if (
                        is_numeric_dtype(_index)
                        or is_float_index(index)
                        or (isinstance(_index[0], float))
                    ):
                        raise ValueError("Numeric index given")
                    # If a non-index Pandas series was given, only keep its
                    # values (because we must have a pd.Index type, below, and
                    # pd.to_datetime will return a Series when passed
                    # non-list-like objects)
                    if isinstance(index, Series):
                        index = index.values
                    # All coercion is done via pd.to_datetime
                    # Note: date coercion via pd.to_datetime does not handle
                    # string versions of PeriodIndex objects most of the time.
                    _index = to_datetime(index)
                    # Older versions of Pandas can sometimes fail here and
                    # return a numpy array - check to make sure it's an index
                    if not isinstance(_index, Index):
                        raise ValueError("Could not coerce to date index")
                    index = _index
                except Exception as exc:
                    # Only want to actually raise an exception if `dates` was
                    # provided but cannot be coerced. If we got the index from
                    # the row_labels, we'll just ignore it and use the integer
                    # index below
                    if dates is not None:
                        raise ValueError(
                            "Non-date index provided to"
                            " `dates` argument."
                        ) from exc
            # Now, if we were given, or coerced, a date-based index, make sure
            # it has an associated frequency
            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                # If no frequency, try to get an inferred frequency
                if freq is None and index.freq is None:
                    with _infer_freq_returns_offset():
                        freq = index.inferred_freq
                    # If we got an inferred frequncy, alert the user
                    if freq is not None:
                        inferred_freq = True
                        if isinstance(freq, pd.tseries.offsets.BaseOffset):
                            freqstr = freq.freqstr
                        else:
                            freqstr = str(freq)
                        warnings.warn(
                            "No frequency information was provided, so inferred "
                            f"frequency {freqstr} will be used.",
                            ValueWarning,
                            stacklevel=2,
                        )

                # Convert the passed freq to a pandas offset object
                if freq is not None and isinstance(index, PeriodIndex):
                    # Accept the period alias of the index (e.g. "M"), which
                    # pandas 3 does not accept as an offset alias, as well as
                    # an offset alias (e.g. "ME")
                    try:
                        freq = to_period_offset(freq)
                    except ValueError:
                        freq = to_offset(freq)
                elif freq is not None:
                    freq = to_offset(freq)

                # Now, if no frequency information is available from the index
                # itself or from the `freq` argument, raise an exception
                if freq is None and index.freq is None:
                    # But again, only want to raise the exception if `dates`
                    # was provided.
                    if dates is not None:
                        raise ValueError(
                            "No frequency information was"
                            " provided with date index and no"
                            " frequency could be inferred."
                        )
                # However, if the index itself has no frequency information but
                # the `freq` argument is available (or was inferred), construct
                # a new index with an associated frequency
                elif freq is not None and index.freq is None:
                    resampled_index = date_range(
                        start=index[0], end=index[-1], freq=freq
                    )
                    if not inferred_freq and (
                        len(resampled_index) != len(index)
                        or not (resampled_index == index).all()
                    ):
                        raise ValueError(
                            "The given frequency argument could"
                            " not be matched to the given index."
                        )
                    index = resampled_index
                # Finally, if the index itself has a frequency and there was
                # also a given frequency, raise an exception if they are not
                # equal
                elif (
                    freq is not None
                    and not inferred_freq
                    and index.freq != freq
                ):
                    raise ValueError(
                        "The given frequency argument is"
                        " incompatible with the given index."
                    )
            # Finally, raise an exception if we could not coerce to date-based
            # but we were given a frequency argument
            elif freq is not None:
                raise ValueError(
                    "Given index could not be coerced to dates"
                    " but `freq` argument was provided."
                )

        # Get attributes of the index
        has_index = index is not None
        date_index = isinstance(index, (DatetimeIndex, PeriodIndex))
        int_index = is_int_index(index)
        range_index = isinstance(index, RangeIndex)
        has_freq = index.freq is not None if date_index else None
        increment = Index(range(self.endog.shape[0]))
        is_increment = index.equals(increment) if int_index else None
        if date_index:
            is_monotonic = index.is_monotonic_increasing
        else:
            is_monotonic = None
        # A PeriodIndex always has a frequency, even if it has gaps or
        # repeated periods, so check that its periods are consecutive.
        # A non-monotonic index is skipped here since it is already rejected,
        # with its own warning, through `is_monotonic`.
        has_nonconsecutive_periods = (
            isinstance(index, PeriodIndex)
            and is_monotonic
            and len(index) > 0
            and not np.all(np.diff(index.asi8) == index.freq.n)
        )

        # Issue warnings for unsupported indexes
        if has_index and not (date_index or range_index or is_increment):
            warnings.warn(
                "An unsupported index was provided. As a result, forecasts "
                "cannot be generated. To use the model for forecasting, use on the "
                "supported classes of index.",
                ValueWarning,
                stacklevel=2,
            )
        if date_index and not has_freq:
            warnings.warn(
                "A date index has been provided, but it has no"
                " associated frequency information and so will be"
                " ignored when e.g., forecasting.",
                ValueWarning,
                stacklevel=2,
            )
        if date_index and not is_monotonic:
            warnings.warn(
                "A date index has been provided, but it is not"
                " monotonic and so will be ignored when e.g."
                " forecasting.",
                ValueWarning,
                stacklevel=2,
            )
        if has_nonconsecutive_periods:
            warnings.warn(
                "A Period index has been provided, but its periods are not"
                " consecutive (it has gaps or repeated periods) and so will be"
                " ignored. Forecasting will fail unless an index is given. Use"
                " a DatetimeIndex with a frequency (e.g. 'B' or a"
                " CustomBusinessDay) if the gaps follow a calendar rule, or"
                " reindex the data to consecutive periods.",
                ValueWarning,
                stacklevel=2,
            )

        # Construct the internal index
        index_generated = False
        valid_index = (
            (
                date_index
                and has_freq
                and is_monotonic
                and not has_nonconsecutive_periods
            )
            or (int_index and is_increment)
            or range_index
        )

        if valid_index:
            _index = index
        else:
            _index = increment
            index_generated = True
        self._index = _index
        self._index_generated = index_generated
        self._index_none = index is None
        self._index_int64 = int_index and not range_index and not date_index
        self._index_dates = date_index and not index_generated
        self._index_freq = self._index.freq if self._index_dates else None
        self._index_inferred_freq = inferred_freq

        # For backwards compatibility, set data.dates, data.freq
        self.data.dates = self._index if self._index_dates else None
        self.data.freq = self._index.freqstr if self._index_dates else None

    def _get_index_loc(self, key):
        """
        Get the location of a specific key in an index

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or a location if the underlying index is a RangeIndex
            or a NumericIndex.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        If `key` is past the end of the given index, and the index is either
        a NumericIndex or a date index, this function extends the index up to
        and including key, and then returns the location in the new index.
        """
        return get_index_loc(key, self._index)

    def _get_index_label_loc(self, key):
        """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or is only being used as row labels, or a location if
            the underlying index is a RangeIndex or a NumericIndex.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        This method expands on `_get_index_loc` by first trying the model's
        index and then falling back to try again with the model row labels as
        the base index.
        """
        return get_index_label_loc(key, self._index, self.data.row_labels)

    def _get_prediction_index(
        self, start, end, index=None, silent=False
    ) -> tuple[int, int, int, Index | None]:
        """
        Get the location of the start and end of prediction, and the associated index

        Parameters
        ----------
        start : label
            The key at which to start prediction. Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        end : label
            The key at which to end prediction (note that this key will be
            *included* in prediction). Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        silent : bool, optional
            Argument to silence warnings.

        Returns
        -------
        start : int
            The index / observation location at which to begin prediction.
        end : int
            The index / observation location at which to end in-sample
            prediction. The maximum value for this is nobs-1.
        out_of_sample : int
            The number of observations to forecast after the end of the sample.
        prediction_index : pd.Index or None
            The index associated with the prediction results. This index covers
            the range [start, end + out_of_sample]. If the model has no given
            index and no given row labels (i.e., endog/exog is not Pandas), then
            this will be None.

        Notes
        -----
        The arguments `start` and `end` behave differently, depending on if
        they are integer or not. If either is an integer, then it is assumed
        to refer to a *location* in the index, not to an index value. On the
        other hand, if it is a date string or some other type of object, then
        it is assumed to refer to an index *value*. In all cases, the returned
        `start` and `end` values refer to index *locations* (so in the former
        case, the given location is validated and returned whereas in the
        latter case a location is found that corresponds to the given index
        value).

        This difference in behavior is necessary to support `RangeIndex`. This
        is because integers for a RangeIndex could refer either to index values
        or to index locations in an ambiguous way (while for `NumericIndex`,
        since we have required them to be full indexes, there is no ambiguity).
        """
        nobs = len(self.endog)
        return get_prediction_index(
            start,
            end,
            nobs,
            base_index=self._index,
            index=index,
            silent=silent,
            index_none=self._index_none,
            index_generated=self._index_generated,
            data=self.data,
        )

    def _get_exog_names(self):
        return self.data.xnames

    def _set_exog_names(self, vals):
        if not isinstance(vals, list):
            vals = [vals]
        self.data.xnames = vals

    # TODO: This is an antipattern, fix/remove with VAR
    # overwrite with writable property for (V)AR models
    exog_names = property(
        _get_exog_names,
        _set_exog_names,
        None,
        "The names of the exogenous variables.",
    )


class TimeSeriesModelResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale=1.0):
        self.data = model.data
        super().__init__(model, params, normalized_cov_params, scale)


class TimeSeriesResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        base.LikelihoodResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {"predict": "dates"}
    _wrap_methods = wrap.union_dicts(
        base.LikelihoodResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(
    TimeSeriesResultsWrapper, TimeSeriesModelResults
)
