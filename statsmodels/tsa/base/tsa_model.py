from statsmodels.compat.python import lrange, long
from statsmodels.compat.pandas import is_numeric_dtype

import datetime

from pandas import to_datetime, DatetimeIndex, Period, PeriodIndex, Timestamp

from statsmodels.base import data
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.base import datetools

_freq_to_pandas = datetools._freq_to_pandas

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

    __doc__ = _tsa_doc % {"model" : _model_doc, "params" : _generic_params,
                          "extra_params" : _missing_param_doc,
                          "extra_sections" : ""}

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none'):
        super(TimeSeriesModel, self).__init__(endog, exog, missing=missing)
        self._init_dates(dates, freq)

    def _init_dates(self, dates, freq):
        if dates is None:
            dates = self.data.row_labels

        if dates is not None:
            if (not datetools._is_datetime_index(dates) and
                    isinstance(self.data, data.PandasData)):
                try:
                    if is_numeric_dtype(dates):
                        raise ValueError
                    dates = to_datetime(dates)
                except ValueError:
                    raise ValueError("Given a pandas object and the index does "
                                     "not contain dates")
            if not freq:
                try:
                    freq = datetools._infer_freq(dates)
                except:
                    raise ValueError("Frequency inference failed. Use `freq` "
                                     "keyword.")

            if isinstance(dates[0], datetime.datetime):
                dates = DatetimeIndex(dates)
            else: # preserve PeriodIndex
                dates = PeriodIndex(dates)
        self.data.dates = dates
        self.data.freq = freq

        # Test for nanoseconds in early pandas versions
        if freq is not None and _freq_to_pandas[freq].freqstr == 'N':
            from distutils.version import LooseVersion
            from pandas import __version__ as pd_version
            if LooseVersion(pd_version) < '0.14':
                raise NotImplementedError('Nanosecond index not available in'
                                          ' Pandas < 0.14')


    def _get_exog_names(self):
        return self.data.xnames

    def _set_exog_names(self, vals):
        if not isinstance(vals, list):
            vals = [vals]
        self.data.xnames = vals

    #overwrite with writable property for (V)AR models
    exog_names = property(_get_exog_names, _set_exog_names)

    def _get_dates_loc(self, dates, date):
        date = dates.get_loc(date)
        return date

    def _str_to_date(self, date):
        """
        Takes a string and returns a datetime object
        """
        if isinstance(self.data.dates, PeriodIndex):
            return Period(date)
        else:
            return datetools.date_parser(date)

    def _set_predict_start_date(self, start):
        dates = self.data.dates
        if dates is None:
            return
        if start > len(dates):
            raise ValueError("Start must be <= len(endog)")
        if start == len(dates):
            self.data.predict_start = datetools._date_from_idx(dates[-1],
                                                    1, self.data.freq)
        elif start < len(dates):
            self.data.predict_start = dates[start]
        else:
            raise ValueError("Start must be <= len(dates)")

    def _get_predict_start(self, start):
        """
        Returns the index of the given start date. Subclasses should define
        default behavior for start = None. That isn't handled here.

        Start can be a string or an integer if self.data.dates is None.
        """
        dates = self.data.dates
        if not isinstance(start, (int, long)):
            start = str(start)
            if dates is None:
                raise ValueError("Got a string for start and dates is None")
            dtstart = self._str_to_date(start)
            self.data.predict_start = dtstart
            try:
                start = self._get_dates_loc(dates, dtstart)
            except KeyError:
                raise ValueError("Start must be in dates. Got %s | %s" %
                        (str(start), str(dtstart)))

        self._set_predict_start_date(start)
        return start


    def _get_predict_end(self, end):
        """
        See _get_predict_start for more information. Subclasses do not
        need to define anything for this.
        """

        out_of_sample = 0 # will be overwritten if needed
        if end is None: # use data for ARIMA - endog changes
            end = len(self.data.endog) - 1

        dates = self.data.dates
        freq = self.data.freq

        if isinstance(end, datetime.datetime):
            end = self._str_to_date(str(end))

        if isinstance(end, str) or (dates is not None
                                    and isinstance(end, type(dates[0]))):
            if dates is None:
                raise ValueError("Got a string or date for `end` and `dates` is None")

            if isinstance(end, str):
                dtend = self._str_to_date(end)
            else:
                dtend = end  # end could be a pandas TimeStamp not a datetime

            self.data.predict_end = dtend
            try:
                end = self._get_dates_loc(dates, dtend)
            except KeyError as err: # end is greater than dates[-1]...probably
                if dtend > self.data.dates[-1]:
                    end = len(self.data.endog) - 1
                    freq = self.data.freq
                    out_of_sample = datetools._idx_from_dates(dates[-1], dtend,
                                            freq)
                else:
                    if freq is None:
                        raise ValueError("There is no frequency for these "
                                         "dates and date %s is not in dates "
                                         "index. Try giving a date that is in "
                                         "the dates index or use an integer."
                                         % dtend)
                    else: #pragma: no cover
                        raise err # should never get here
            self._make_predict_dates() # attaches self.data.predict_dates

        elif isinstance(end, (int, long)) and dates is not None:
            try:
                self.data.predict_end = dates[end]
            except IndexError as err:
                nobs = len(self.data.endog) - 1 # as an index
                out_of_sample = end - nobs
                end = nobs
                if freq is not None:
                    self.data.predict_end = datetools._date_from_idx(dates[-1],
                            out_of_sample, freq)
                elif out_of_sample <= 0: # have no frequency but are in sample
                    #TODO: what error to catch here to make sure dates is
                    #on the index?
                    try:
                        self.data.predict_end = self._get_dates_loc(dates, end)
                    except KeyError:
                        raise
                else:
                    self.data.predict_end = end + out_of_sample
                    self.data.predict_start = self._get_dates_loc(dates,
                                                self.data.predict_start)

            self._make_predict_dates()

        elif isinstance(end, (int, long)):
            nobs = len(self.data.endog) - 1 # is an index
            if end > nobs:
                out_of_sample = end - nobs
                end = nobs

        elif freq is None: # should have a date with freq = None
            print('#'*80)
            print(freq)
            print(type(freq))
            print('#'*80)
            raise ValueError("When freq is None, you must give an integer "
                             "index for end.")

        else:
            print('#'*80)
            print(freq)
            print(type(freq))
            print('#'*80)
            raise ValueError("no rule for interpreting end")

        return end, out_of_sample

    def _make_predict_dates(self):
        data = self.data
        dtstart = data.predict_start
        dtend = data.predict_end
        freq = data.freq

        if freq is not None:
            pandas_freq = _freq_to_pandas[freq]
            # preserve PeriodIndex or DatetimeIndex
            dates = self.data.dates.__class__(start=dtstart,
                                              end=dtend,
                                              freq=pandas_freq)

            if pandas_freq.freqstr == 'N':
                _dtend = dtend
                if isinstance(dates[-1], Period):
                    _dtend = pd.to_datetime(_dtend).to_period(dates.freq)
                if not dates[-1] == _dtend:
                    # TODO: this is a hack because a DatetimeIndex with
                    # nanosecond frequency does not include "end"
                    dtend = Timestamp(dtend.value + 1)
                    dates = self.data.dates.__class__(start=dtstart,
                                                      end=dtend,
                                                      freq=pandas_freq)
        # handle
        elif freq is None and (isinstance(dtstart, (int, long)) and
                               isinstance(dtend, (int, long))):
            from pandas import Index
            dates = Index(lrange(dtstart, dtend+1))
        # if freq is None and dtstart and dtend aren't integers, we're
        # in sample
        else:
            dates = self.data.dates
            start = self._get_dates_loc(dates, dtstart)
            end = self._get_dates_loc(dates, dtend)
            dates = dates[start:end+1] # is this index inclusive?
        self.data.predict_dates = dates

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
