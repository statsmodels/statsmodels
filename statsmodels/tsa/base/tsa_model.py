from statsmodels.compat.python import lrange, long
from statsmodels.compat.pandas import is_numeric_dtype

import datetime

import warnings
import numpy as np
from pandas import (to_datetime, Int64Index, DatetimeIndex, Period,
                    PeriodIndex, Timestamp)

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
        self.data.dates = None
        self.data.freq = None
        self._init_dates(dates, freq)

    def _init_dates(self, dates=None, freq=None):
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
        if index is not None:
            # Try to coerce to date-based index
            if not isinstance(index, (DatetimeIndex, PeriodIndex)):
                try:
                    # Only try to coerce non-numeric index types (string,
                    # list of date-times, etc.)
                    if is_numeric_dtype(index):
                        raise ValueError
                    # All coercion is done via pandas.to_datetime
                    index = to_datetime(index)
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
                # If no frequency information is available from the index
                # itself or from the `freq` argument, raise an exception
                if index.freq is None and freq is None:
                    # But again, only want to raise the exception if `dates`
                    # was provided. If the index came from row labels, just
                    # reset it to whatever it was before and ignore it below
                    if dates is not None:
                        raise ValueError('No frequency information provided'
                                         ' with date index.')
                    else:
                        index = self.data.row_labels
                # If the index itself has no frequency information but the
                # `freq` argument is available, construct a new index with an
                # associated frequency
                elif index.freq is None and freq is not None:
                    resampled_index = type(index)(
                        start=index[0], end=index[-1], freq=freq)
                    if not resampled_index.equals(index):
                        raise ValueError('Given index could not be matched to'
                                         ' given frequency.')
                    index = resampled_index
                # If the index itself has a frequency and there was a
                # given frequency raise an exception if they are not equal
                elif freq is not None and not (index.freq == freq):
                    raise ValueError('Given index frequency different from'
                                     ' given frequency argument.')
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

        # Validate
        if has_index and not (date_index or is_increment):
            warnings.warn('An unsupported index was provided and will be'
                          ' ignored when e.g. forecasting.', ValueWarning)
            # raise ValueError('If Pandas data is provided for time series'
            #                  ' models, a DatetimeIndex or PeriodIndex must'
            #                  ' be used.')
        if date_index and not has_freq:
            warnings.warn('A date index has been provided, but it has no'
                          ' associated frequency information and so will be'
                          ' ignored', ValueWarning)
            # raise ValueError('If Pandas data with a DatetimeIndex or'
            #                  ' PeriodIndex is provided for time series'
            #                  ' models, the index must have an associated'
            #                  ' frequency value.')

        # Construct the internal index
        index_increment = False

        if (date_index and has_freq) or (int_index and is_increment):
            _index = index.copy()
        else:
            _index = increment
            index_increment = True
        self._index = _index
        self._index_increment = index_increment
        self._index_none = index is None
        self._index_dates = date_index and not index_increment
        self._index_freq = index.freqstr if self._index_dates else None

        # For backwards compatibility, set data.dates, data.freq
        if self._index_dates:
            self.data.dates = self._index
            self.data.freq = self._index_freq

    def _get_index_loc(self, key, index=None):
        if index is None:
            index = self._index
        return datetools._get_index_loc(key, index)

    def _get_prediction_index(self, start, end, index=None):
        # Convert index keys (start, end) to index locations and get associated
        # indexes.
        try:
            start, start_index, start_oos = self._get_index_loc(start)
        except KeyError as e:
            try:
                if not isinstance(start, (int, long, np.integer)):
                    start = self.data.row_labels.get_loc(start)
                else:
                    raise
                start_index = self.data.row_labels[:start + 1]
                start_oos = False
            except:
                raise e

        if end is None:
            end = max(start, len(self._index) - 1)
        try:
            end, end_index, end_oos = self._get_index_loc(end)
        except KeyError as e:
            try:
                if not isinstance(end, (int, long, np.integer)):
                    end = self.data.row_labels.get_loc(end)
                else:
                    raise
                end_index = self.data.row_labels[:end + 1]
                end_oos = False
            except:
                raise e

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
        if (not (start_oos or end_oos or self._index_none or self._index_dates)
                and index is None and self.data.row_labels is not None):
            prediction_index = self.data.row_labels[start:end + 1]
        elif not self._index_none and self._index_increment:
            if index is None:
                warnings.warn('The model does not have an associated supported'
                              ' index, and `index` argument was not provided'
                              ' in prediction. Prediction results will be'
                              ' given with an integer index beginning at'
                              ' `start`.',
                              ValueWarning)
            elif not len(prediction_index) == len(index):
                raise ValueError('Invalid `index` provided in prediction.'
                                 ' Must have length consistent with `start`'
                                 ' and `end` arguments.')
            else:
                prediction_index = pd.Index(index)
        elif index is not None:
            warnings.warn('`index` argument provided in prediction'
                          ' but the model already has an associated supported'
                          ' index. The `index` argument will be'
                          ' ignored.', ValueWarning)
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
