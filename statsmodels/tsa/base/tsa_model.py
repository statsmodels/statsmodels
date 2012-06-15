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

#REPLACE frequencies with either timeseries or pandas conventions
class TimeSeriesModel(base.LikelihoodModel):
    """
    Timeseries model base class

    Parameters
    ----------
    endog
    exog
    dates
    freq : str {'B','D','W','M','A', 'Q'}
        'B' - business day, ie., Mon. - Fri.
        'D' - daily
        'W' - weekly
        'M' - monthly
        'A' - annual
        'Q' - quarterly

    """
    def __init__(self, endog, exog=None, dates=None, freq=None):
        super(TimeSeriesModel, self).__init__(endog, exog)
        self._init_dates(dates, freq)

    def _init_dates(self, dates, freq):
        if dates is None:
            dates = self._data.row_labels

        if dates is not None:
            try:
                from scikits.timeseries import Date
                if not isinstance(dates[0], (datetime.datetime,Date)):
                    raise ValueError("dates must be of type datetime or "
                                     "scikits.timeseries.Date")
            except ImportError:
                if not isinstance(dates[0], (datetime.datetime)):
                    raise ValueError("dates must be of type datetime")
            if not freq:
                #if isinstance(dates, DateRange):
                #    freq = datetools.inferTimeRule(dates)
                #elif isinstance(dates, TimeSeries):
                #    freq = dates.freqstr
                raise ValueError("Currently, you need to give freq if dates "
                        "are used.")
            dates = Index(dates)
        self._data.dates = dates
        self._data.freq = _check_freq(freq) #TODO: drop if can get info from dates
        #TODO: still gonna need some more sophisticated handling here


    def _get_exog_names(self):
        return self._data.xnames

    def _set_exog_names(self, vals):
        if not isinstance(vals, list):
            vals = [vals]
        self._data.xnames = vals

    #overwrite with writable property for (V)AR models
    exog_names = property(_get_exog_names, _set_exog_names)

    def _str_to_date(self, date):
        """
        Takes a string and returns a datetime object
        """
        return datetools.date_parser(date)

    def _get_predict_start(self, start):
        """
        Returns the index of the given start date. Subclasses should define
        default behavior for start = None. That isn't handled here.

        Start can be a string or an integer if self._data.dates is None.
        """
        dates = self._data.dates
        if isinstance(start, str):
            if dates is None:
                raise ValueError("Got a string for start and dates is None")
            try:
                dtstart = self._str_to_date(start)
                self._data.predict_start = dtstart
                # for pandas 0.7.x vs 0.8.x
                if hasattr(dates, 'indexMap'): # 0.7.x
                    start = dates.indexMap[dtstart]
                else:
                    start = dates.get_loc(dtstart)
            except: # this catches all errors in the above..
                    #FIXME to be less greedy
                raise ValueError("Start must be in dates. Got %s | %s" %
                        (str(start), str(dtstart)))

        if isinstance(start, int) and dates is not None:
            if start >= len(dates):
                raise ValueError("Start must be <= len(endog)")
            self._data.predict_start = dates[start]

        if start >= len(self.endog):
            raise ValueError("Start must be <= len(endog)")

        return start


    def _get_predict_end(self, end):
        """
        See _get_predict_start for more information. Subclasses do not
        need to define anything for this.
        """

        out_of_sample = 0 # will be overwritten if needed
        if end is None:
            end = len(self.endog) - 1

        dates = self._data.dates
        if isinstance(end, str):
            if dates is None:
                raise ValueError("Got a string for end and dates is None")
            try:
                dtend = self._str_to_date(end)
                self._data.predict_end = dtend
                # for pandas 0.7.x vs 0.8.x
                if hasattr(dates, 'indexMap'): # 0.7.x
                    end = dates.indexMap[dtend]
                else:
                    end = dates.get_loc(dtend)
            except KeyError, err: # end is greater than dates[-1]...probably
                if dtend > self._data.dates[-1]:
                    end = len(self.endog) - 1
                    freq = self._data.freq
                    out_of_sample = datetools._idx_from_dates(dates[-1], dtend,
                                            freq)
                else:
                    raise err
            self._make_predict_dates() # attaches self._data.predict_dates

        elif isinstance(end, int) and dates is not None:
            try:
                self._data.predict_end = dates[end]
            except IndexError, err:
                nobs = len(self.endog) - 1 # as an index
                out_of_sample = end - nobs
                end = nobs
                freq = self._data.freq
                self._data.predict_end = datetools._date_from_idx(dates[-1],
                        out_of_sample, freq)
            self._make_predict_dates()

        elif isinstance(end, int):
            nobs = len(self.endog) - 1 # is an index
            if end > nobs:
                out_of_sample = end - nobs
                end = nobs

        return end, out_of_sample

    def _make_predict_dates(self):
        from pandas import DateRange
        data = self._data
        dtstart = data.predict_start
        dtend = data.predict_end
        freq = data.freq
        pandas_freq = _freq_to_pandas[freq]
        dates = DateRange(dtstart, dtend, offset = pandas_freq).values
        self._data.predict_dates = dates

class TimeSeriesModelResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale=1.):
        self._data = model._data
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
    try:
        import scikits.timeseries as ts
        ts_dates = date_array(start_date = Date(year=1959,quarter=1,freq='Q'),
                             length=len(data.data))
    except:
        pass


    df = pandas.DataFrame(data.data[['realgdp','realinv','realcons']], index=dates)
    ex_mod = TimeSeriesModel(df)
    #ts_series = pandas.TimeSeries()


