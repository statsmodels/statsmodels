from statsmodels.compat.python import lrange, long
from statsmodels.compat.pandas import is_numeric_dtype, Float64Index

import datetime

import warnings
import numpy as np
from pandas import (to_datetime, Int64Index, DatetimeIndex, Period,
                    PeriodIndex, Timestamp, Series, Index)
from pandas.tseries.frequencies import to_offset

from statsmodels.base import data
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.base import datetools
from statsmodels.tools.sm_exceptions import ValueWarning

_tsa_doc = """
    %(model)s

    Parameters
    ----------
    %(params)s
    dates : array-like of datetime, optional
        An array-like object of datetime objects. If a pandas object is given
        for endog or exog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    %(extra_params)s
    %(extra_sections)s
"""

_model_doc = "Timeseries model base class"

_generic_params = base._model_params_doc
_missing_param_doc = base._missing_param_doc


class TimeSeriesModel(base.LikelihoodModel):

    __doc__ = _tsa_doc % {"model": _model_doc, "params": _generic_params,
                          "extra_params": _missing_param_doc,
                          "extra_sections": ""}

    def __init__(self, endog, exog=None, dates=None, freq=None,
                 missing='none', **kwargs):
        super(TimeSeriesModel, self).__init__(endog, exog, missing=missing,
                                              **kwargs)

        # Date handling in indexes
        self._init_dates(dates, freq)

    def _init_dates(self, dates=None, freq=None):
        """
        Initialize dates

        Parameters
        ----------
        dates : array_like, optional
            An array like object containing dates.
        freq : str, tuple, datetime.timedelta, DateOffset or None, optional
            A frequency specification for either `dates` or the row labels from
            the endog / exog data.

        Notes
        -----
        Creates `self._index` and related attributes. `self._index` is always
        a Pandas index, and it is always Int64Index, DatetimeIndex, or
        PeriodIndex.

        If Pandas objects, endog / exog may have any type of index. If it is
        an Int64Index with values 0, 1, ..., nobs-1 or if it is (coerceable to)
        a DatetimeIndex or PeriodIndex *with an associated frequency*, then it
        is called a "supported" index. Otherwise it is called an "unsupported"
        index.

        Supported indexes are standardized (i.e. a list of date strings is
        converted to a DatetimeIndex) and the result is put in `self._index`.

        Unsupported indexes are ignored, and a supported Int64Index is
        generated and put in `self._index`. Warnings are issued in this case
        to alert the user if the returned index from some operation (e.g.
        forecasting) is different from the original data's index. However,
        whenever possible (e.g. purely in-sample prediction), the original
        index is returned.

        The benefit of supported indexes is that they allow *forecasting*, i.e.
        it is possible to extend them in a reasonable way. Thus every model
        must have an underlying supported index, even if it is just a generated
        Int64Index.

        """

        # Get our index from `dates` if available, otherwise from whatever
        # Pandas index we might have retrieved from endog, exog
        if dates is not None:
            index = dates
        else:
            index = self.data.row_labels

        # Sanity check that we don't have a `freq` without an index
        if index is None and freq is not None:
            raise ValueError('Frequency provided without associated index.')

        # If an index is available, see if it is a date-based index or if it
        # can be coerced to one. (If it can't we'll fall back, below, to an
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
                    # Pandas (~0.12), Float64Index doesn't exist (and so the
                    # Statsmodels compat makes it an empty tuple, so in that
                    # case also check if the first element is a float.
                    _index = np.asarray(index)
                    if (is_numeric_dtype(_index) or
                            isinstance(index, Float64Index) or
                            (Float64Index == tuple() and
                             isinstance(_index[0], float))):
                        raise ValueError('Numeric index given')
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
                        raise ValueError('Could not coerce to date index')
                    index = _index
                except:
                    # Only want to actually raise an exception if `dates` was
                    # provided but can't be coerced. If we got the index from
                    # the row_labels, we'll just ignore it and use the integer
                    # index below
                    if dates is not None:
                        raise ValueError('Non-date index index provided to'
                                         ' `dates` argument.')
            # Now, if we were given, or coerced, a date-based index, make sure
            # it has an associated frequency
            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                # If no frequency, try to get an inferred frequency
                if freq is None and index.freq is None:
                    freq = index.inferred_freq
                    # If we got an inferred frequncy, alert the user
                    if freq is not None:
                        inferred_freq = True
                        if freq is not None:
                            warnings.warn('No frequency information was'
                                          ' provided, so inferred frequency %s'
                                          ' will be used.'
                                          % freq, ValueWarning)

                # Test for nanoseconds in early pandas versions
                if ((freq is not None and str(freq) == 'N') or
                        (index is not None and index.freq is not None and
                         index.freqstr == 'N')):
                    from distutils.version import LooseVersion
                    from pandas import __version__ as pd_version
                    if LooseVersion(pd_version) < '0.14':
                        raise NotImplementedError('Nanosecond index not'
                                                  ' available in Pandas'
                                                  ' < 0.14')

                # Convert the passed freq to a pandas offset object
                if freq is not None:
                    freq = to_offset(freq)

                # Now, if no frequency information is available from the index
                # itself or from the `freq` argument, raise an exception
                if freq is None and index.freq is None:
                    # But again, only want to raise the exception if `dates`
                    # was provided.
                    if dates is not None:
                        raise ValueError('No frequency information was'
                                         ' provided with date index and no'
                                         ' frequency could be inferred.')
                # However, if the index itself has no frequency information but
                # the `freq` argument is available (or was inferred), construct
                # a new index with an associated frequency
                elif freq is not None and index.freq is None:
                    resampled_index = type(index)(
                        start=index[0], end=index[-1], freq=freq)
                    if not inferred_freq and not resampled_index.equals(index):
                        raise ValueError('The given frequency argument could'
                                         ' not be matched to the given index.')
                    index = resampled_index
                # Finally, if the index itself has a frequency and there was
                # also a given frequency, raise an exception if they are not
                # equal
                elif (freq is not None and not inferred_freq and
                        not (index.freq == freq)):
                    raise ValueError('The given frequency argument is'
                                     ' incompatible with the given index.')
            # Finally, raise an exception if we could not coerce to date-based
            # but we were given a frequency argument
            elif freq is not None:
                raise ValueError('Given index could not be coerced to dates'
                                 ' but `freq` argument was provided.')

        # Get attributes of the index
        has_index = index is not None
        date_index = isinstance(index, (DatetimeIndex, PeriodIndex))
        int_index = isinstance(index, Int64Index)
        has_freq = index.freq is not None if date_index else None
        increment = Int64Index(np.arange(self.endog.shape[0]))
        is_increment = index.equals(increment) if int_index else None

        # Issue warnings for unsupported indexes
        if has_index and not (date_index or is_increment):
            warnings.warn('An unsupported index was provided and will be'
                          ' ignored when e.g. forecasting.', ValueWarning)
        if date_index and not has_freq:
            warnings.warn('A date index has been provided, but it has no'
                          ' associated frequency information and so will be'
                          ' ignored when e.g. forecasting.', ValueWarning)

        # Construct the internal index
        index_generated = False

        if (date_index and has_freq) or (int_index and is_increment):
            _index = index.copy()
        else:
            _index = increment
            index_generated = True
        self._index = _index
        self._index_generated = index_generated
        self._index_none = index is None
        self._index_dates = date_index and not index_generated
        self._index_freq = self._index.freq if self._index_dates else None
        self._index_inferred_freq = inferred_freq

        # For backwards compatibility, set data.dates, data.freq
        self.data.dates = self._index if self._index_dates else None
        self.data.freq = self._index.freqstr if self._index_dates else None

    def _get_index_loc(self, key, base_index=None):
        """
        Get the location of a specific key in an index

        Parameters
        ----------
        key : label
            The key for which to find the location
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accomodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accomodate `key`.

        Notes
        -----
        If `key` is past the end of of the given index, and the index is either
        an Int64Index or a date index, this function extends the index up to
        and including key, and then returns the location in the new index.

        """
        if base_index is None:
            base_index = self._index

        index = base_index
        date_index = isinstance(base_index, (PeriodIndex, DatetimeIndex))
        index_class = type(base_index)
        nobs = len(index)

        # Special handling for Int64Index
        if (isinstance(index, Int64Index) and not date_index and
                isinstance(key, (int, long, np.integer))):
            # Negative indices (that lie in the Index)
            if key < 0 and -key <= nobs:
                key = nobs + key
            # Out-of-sample (note that we include key itself in the new index)
            elif key > base_index[-1]:
                index = Int64Index(np.arange(base_index[0], int(key + 1)))

        # Special handling for date indexes
        if date_index:
            # Integer key (i.e. already given a location)
            if isinstance(key, (int, long, np.integer)):
                # Negative indices (that lie in the Index)
                if key < 0 and -key < nobs:
                    key = index[nobs + key]
                # Out-of-sample (note that we include key itself in the new
                # index)
                elif key > len(base_index) - 1:
                    index = index_class(start=base_index[0],
                                        periods=int(key + 1),
                                        freq=base_index.freq)
                    key = index[-1]
                else:
                    key = index[key]
            # Other key types (i.e. string date or some datetime-like object)
            else:
                # Covert the key to the appropriate date-like object
                if index_class is PeriodIndex:
                    date_key = Period(key, freq=base_index.freq)
                else:
                    date_key = Timestamp(key)

                # Out-of-sample
                if date_key > base_index[-1]:
                    # First create an index that may not always include `key`
                    index = index_class(start=base_index[0], end=date_key,
                                        freq=base_index.freq)

                    # Now make sure we include `key`
                    if not index[-1] == date_key:
                        index = index_class(start=base_index[0],
                                            periods=len(index) + 1,
                                            freq=base_index.freq)

        # Get the location (note that get_loc will throw a KeyError if key is
        # invalid)
        loc = index.get_loc(key)

        # Check if we now have a modified index
        index_was_expanded = index is not base_index

        # (Never return the actual index object)
        if not index_was_expanded:
            index = index.copy()

        # Return the index through the end of the loc / slice
        if isinstance(loc, slice):
            end = loc.stop
        else:
            end = loc

        return loc, index[:end + 1], index_was_expanded

    def _get_index_label_loc(self, key, base_index=None):
        """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        key : label
            The key for which to find the location
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accomodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accomodate `key`.

        Notes
        -----
        This method expands on `_get_index_loc` by first trying the given
        base index (or the model's index if the base index was not given) and
        then falling back to try again with the model row labels as the base
        index.

        """
        try:
            loc, index, index_was_expanded = (
                self._get_index_loc(key, base_index))
        except KeyError as e:
            try:
                if not isinstance(key, (int, long, np.integer)):
                    loc = self.data.row_labels.get_loc(key)
                else:
                    raise
                loc = loc[0]  # Require scalar
                index = self.data.row_labels[:loc + 1]
                index_was_expanded = False
            except:
                raise e
        return loc, index, index_was_expanded

    def _get_prediction_index(self, start, end, index=None, silent=False):
        """
        Get the location of a specific key in an index or model row labels

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
            index and no given row labels (i.e. endog/exog is not Pandas), then
            this will be None.

        Notes
        -----
        This method expands on `_get_index_loc` by first trying the given
        base index (or the model's index if the base index was not given) and
        then falling back to try again with the model row labels as the base
        index.

        """

        # Convert index keys (start, end) to index locations and get associated
        # indexes.
        try:
            start, start_index, start_oos = self._get_index_label_loc(start)
        except KeyError:
            raise KeyError('The `start` argument could not be matched to a'
                           ' location related to the index of the data.')
        if end is None:
            end = max(start, len(self._index) - 1)
        try:
            end, end_index, end_oos = self._get_index_label_loc(end)
        except KeyError:
            raise KeyError('The `end` argument could not be matched to a'
                           ' location related to the index of the data.')

        # Handle slices (if the given index keys cover more than one date)
        if isinstance(start, slice):
            start = start.start
        if isinstance(end, slice):
            end = end.stop - 1

        # Get the actual index for the prediction
        prediction_index = end_index[start:]

        # Validate prediction options
        if end < start:
            raise ValueError('Prediction must have `end` after `start`.')

        # Handle custom prediction index
        # First, if we were given an index, check that it's the right size and
        # use it if so
        if index is not None:
            if not len(prediction_index) == len(index):
                raise ValueError('Invalid `index` provided in prediction.'
                                 ' Must have length consistent with `start`'
                                 ' and `end` arguments.')
            # But if we weren't given Pandas input, this index will not be
            # used because the data will not be wrapped; in that case, issue
            # a warning
            if not isinstance(self.data, data.PandasData) and not silent:
                warnings.warn('Because the model data (`endog`, `exog`) were'
                              ' not given as Pandas objects, the prediction'
                              ' output will be Numpy arrays, and the given'
                              ' `index` argument will only be used'
                              ' internally.', ValueWarning)
            prediction_index = Index(index)
        # Now, if we *do not* have a supported index, but we were given some
        # kind of index...
        elif self._index_generated and not self._index_none:
            # If we are in sample, and have row labels, use them
            if self.data.row_labels is not None and not (start_oos or end_oos):
                prediction_index = self.data.row_labels[start:end + 1]
            # Otherwise, warn the user that they will get an Int64Index
            elif not silent:
                warnings.warn('No supported index is available.'
                              ' Prediction results will be given with'
                              ' an integer index beginning at `start`.',
                              ValueWarning)
        elif self._index_none:
            prediction_index = None

        # For backwards compatibility, set `predict_*` values
        if prediction_index is not None:
            self.data.predict_start = prediction_index[0]
            self.data.predict_end = prediction_index[-1]
            self.data.predict_dates = prediction_index
        else:
            self.data.predict_start = None
            self.data.predict_end = None
            self.data.predict_dates = None

        # Compute out-of-sample observations
        nobs = len(self.endog)
        out_of_sample = max(end - (nobs - 1), 0)
        end -= out_of_sample

        return start, end, out_of_sample, prediction_index

    def _get_exog_names(self):
        return self.data.xnames

    def _set_exog_names(self, vals):
        if not isinstance(vals, list):
            vals = [vals]
        self.data.xnames = vals

    # overwrite with writable property for (V)AR models
    exog_names = property(_get_exog_names, _set_exog_names)


class TimeSeriesModelResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale=1.):
        self.data = model.data
        super(TimeSeriesModelResults,
                self).__init__(model, params, normalized_cov_params, scale)

class TimeSeriesResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_attrs,
                                    _attrs)
    _methods = {'predict' : 'dates'}
    _wrap_methods = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(TimeSeriesResultsWrapper,
                      TimeSeriesModelResults)

if __name__ == "__main__":
    import statsmodels.api as sm
    import pandas

    data = sm.datasets.macrodata.load()

    #make a DataFrame
    #TODO: attach a DataFrame to some of the datasets, for quicker use
    dates = [str(int(x[0])) +':'+ str(int(x[1])) \
             for x in data.data[['year','quarter']]]

    df = pandas.DataFrame(data.data[['realgdp','realinv','realcons']], index=dates)
    ex_mod = TimeSeriesModel(df)
