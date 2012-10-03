import statsmodels.base.model as base
from statsmodels.base import data
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.base import datetools
from numpy import arange, asarray
from pandas import Index
from pandas import datetools as pandas_datetools
import datetime

_freqs = ['B','D','W','M','A', 'Q']

_freq_to_pandas = datetools._freq_to_pandas

def _check_freq(freq):
    if freq and freq not in _freqs:
        raise ValueError("freq %s not understood" % freq)
    return freq

_tsa_doc = """
    %(model)s

    Parameters
    ----------
    %(params)s
    dates : array-like of datetime, optional
        An array-like object of datetime objects. If a pandas object is given
        for endog or exog, it is assumed to have a DateIndex.
    freq : str, {'B', 'D', 'W', 'M', 'A', 'Q'}, optional
        The frequency of the time-series. This is optional if dates are given.
    %(extra_params)s
"""

_model_doc = "Timeseries model base class"

_generic_params = base._model_params_doc
_missing_param_doc = base._missing_param_doc

#REPLACE frequencies with either timeseries or pandas conventions
class TimeSeriesModel(base.LikelihoodModel):

    __doc__ = _tsa_doc % {"model" : _model_doc, "params" : _generic_params,
                          "extra_params" : base._missing_param_doc}

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none'):
        super(TimeSeriesModel, self).__init__(endog, exog, missing=missing)
        self._init_dates(dates, freq)

    def _init_dates(self, dates, freq):
        if dates is None:
            dates = self.data.row_labels

        if dates is not None:
            if (not isinstance(dates[0], datetime.datetime) and
                    isinstance(self.data, data.PandasData)):
                raise ValueError("Given a pandas object and the index does "
                                 "not contain dates")
            if not freq:
                try:
                    freq = datetools._infer_freq(dates)
                except:
                    raise ValueError("Frequency inference failed. Use `freq` "
                            "keyword.")
            dates = Index(dates)
        self.data.dates = dates
        self.data.freq = _check_freq(freq) #TODO: drop if can get info from dates
        #TODO: still gonna need some more sophisticated handling here


    def _get_exog_names(self):
        return self.data.xnames

    def _set_exog_names(self, vals):
        if not isinstance(vals, list):
            vals = [vals]
        self.data.xnames = vals

    #overwrite with writable property for (V)AR models
    exog_names = property(_get_exog_names, _set_exog_names)

    def _str_to_date(self, date):
        """
        Takes a string and returns a datetime object
        """
        return datetools.date_parser(date)

    def _set_predict_start_date(self, start):
        dates = self.data.dates
        if dates is None:
            return
        if start > len(dates):
            raise ValueError("Start must be <= len(endog)")
        if start == len(dates):
            self.data.predict_start = datetools._date_from_idx(dates[-1],
                                                    start, self.data.freq)
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
        if isinstance(start, str):
            if dates is None:
                raise ValueError("Got a string for start and dates is None")
            try:
                dtstart = self._str_to_date(start)
                self.data.predict_start = dtstart
                # for pandas 0.7.x vs 0.8.x
                if hasattr(dates, 'indexMap'): # 0.7.x
                    start = dates.indexMap[dtstart]
                else:
                    start = dates.get_loc(dtstart)
            except: # this catches all errors in the above..
                    #FIXME to be less greedy
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
        if isinstance(end, str):
            if dates is None:
                raise ValueError("Got a string for end and dates is None")
            try:
                dtend = self._str_to_date(end)
                self.data.predict_end = dtend
                # for pandas 0.7.x vs 0.8.x
                if hasattr(dates, 'indexMap'): # 0.7.x
                    end = dates.indexMap[dtend]
                else:
                    end = dates.get_loc(dtend)
            except KeyError, err: # end is greater than dates[-1]...probably
                if dtend > self.data.dates[-1]:
                    end = len(self.data.endog) - 1
                    freq = self.data.freq
                    out_of_sample = datetools._idx_from_dates(dates[-1], dtend,
                                            freq)
                else:
                    raise err
            self._make_predict_dates() # attaches self.data.predict_dates

        elif isinstance(end, int) and dates is not None:
            try:
                self.data.predict_end = dates[end]
            except IndexError, err:
                nobs = len(self.data.endog) - 1 # as an index
                out_of_sample = end - nobs
                end = nobs
                freq = self.data.freq
                self.data.predict_end = datetools._date_from_idx(dates[-1],
                        out_of_sample, freq)
            self._make_predict_dates()

        elif isinstance(end, int):
            nobs = len(self.data.endog) - 1 # is an index
            if end > nobs:
                out_of_sample = end - nobs
                end = nobs

        return end, out_of_sample

    def _make_predict_dates(self):
        data = self.data
        dtstart = data.predict_start
        dtend = data.predict_end
        freq = data.freq
        pandas_freq = _freq_to_pandas[freq]
        try:
            from pandas import DatetimeIndex
            dates = DatetimeIndex(start=dtstart, end=dtend,
                                    freq=pandas_freq)
        except ImportError, err:
            from pandas import DateRange
            dates = DateRange(dtstart, dtend, offset = pandas_freq).values
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
    import datetime
    import pandas

    data = sm.datasets.macrodata.load()

    #make a DataFrame
    #TODO: attach a DataFrame to some of the datasets, for quicker use
    dates = [str(int(x[0])) +':'+ str(int(x[1])) \
             for x in data.data[['year','quarter']]]

    df = pandas.DataFrame(data.data[['realgdp','realinv','realcons']], index=dates)
    ex_mod = TimeSeriesModel(df)
    #ts_series = pandas.TimeSeries()


