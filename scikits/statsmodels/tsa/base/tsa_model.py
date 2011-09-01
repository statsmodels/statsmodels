import scikits.statsmodels.base.model as base
import scikits.statsmodels.base.wrapper as wrap
from numpy import arange
from pandas import Index
import datetime

_freq = ['B','D','W','M','A']

def _check_freq(freq):
    if freq and freq not in _freq:
        raise ValueError("freq %s not understood" % freq)
    return freq

#TODO: where should this live?
def _idx_from_dates(d1, d2, freq):
    """
    rd should be a dateutil.relativedelta instance
    freq should be in _freq

    Note that it rounds down the index.
    """
    if freq == 'A':
        idx = d2.year - d1.year
    if freq == 'D':
        idx = (d2 - d1).days
    if freq == 'W':
        idx = (d2 - d1).days // 7
    if freq == 'M':
        idx = (d2.year - d1.year) * 12 + d2.month - d1.month
    if freq == 'B':
        # business days left in first week
        wk1 = max(0, 4 - d1.weekday())

        # business days used in last week
        wk2 = min(4, d2.weekday()) + 1

        d1 += datetime.timedelta(days=wk1+2)
        d2 -= datetime.timedelta(days=wk2)
        idx = (d2 - d1).days // 7 + wk1 + wk2

    return idx


class TimeSeriesModel(base.LikelihoodModel):
    """
    Timeseries model base class

    Parameters
    ----------
    endog
    exog
    dates
    freq : str {'B','D','W','M','A'}
        'B' - business day, ie., Mon. - Fri.
        'D' - daily
        'W' - weekly
        'M' - monthly
        'A' - annual

    """
    def __init__(self, endog, exog=None, dates=None, freq=None):
        super(TimeSeriesModel, self).__init__(endog, exog)
        self._init_dates(dates, freq)

    def _init_dates(self, dates, freq):
        if dates is None:
            dates = self._data.row_labels
        if dates is not None:
            if not isinstance(dates[0], datetime.datetime):
                raise ValueError("dates must be of type datetime ")
            if not freq:
                raise ValueError("Currently, you need to give freq if dates "
                        "are used.")
            dates = Index(dates)
        self.dates = dates
        self.freq = _check_freq(freq) #TODO: drop if can get info from dates
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

        Uses scikits.timeseries.parsers.DateTimeFromString
        """
        #TODO: either copy over parsers, put it in pandas, or add
        #dependency
        try:
            from scikits.timeseries.parser import DateTimeFromString
        except:
            raise ImportError("Need scikits.timeseries.")
        return DateTimeFromString(date)

    def _get_predict_start(self, start):
        """
        Returns the index of the given start date. Subclasses should define
        default behavior for start = None. That isn't handled here.

        Start can be a string, see scikits.timeseries.parser.DateTimeFromString
        or an integer if dates is None.
        """
        dates = self.dates
        if dates is not None:
            if not isinstance(start, str):
                raise ValueError("start should be a string if dates is not "
                                 "None")
            try:
                dtstart = self._str_to_date(start)
                start = dates.indexMap[dtstart] # NOTE: are these consistent?
            except ImportError as err: # make sure timeseries isn't the prob
                raise ImportError(err)
            except:
                raise ValueError("Start must be in dates. Got %s" % str(start))

        else:
            if start > len(self.endog):
                raise ValueError("Start must be <= len(endog)")

        return start


    def _get_predict_end(self, end):
        """
        See _get_predict_start for more information. Subclasses do not
        need to define anything for this.
        """

        out_of_sample = 0 # will be overwritten if needed
        if end is None:
            end = len(self.endog)
            return end, out_of_sample

        dates = self.dates
        if dates is not None:
            if not isinstance(end, str):
                raise ValueError("end should be a string if dates is not "
                                 "None")
            try:
                dtend = self._str_to_date(end)
                end = dates.indexMap[dtend] # NOTE: are these consistent?
            except ImportError as err: # make sure timeseries isn't the prob
                raise ImportError(err)
            except KeyError as err: # end is greater than dates[-1]...probably
                if end > len(self.endog):
                    end = len(self.endog)
                    freq = self.freq
                    out_of_sample = _idx_from_dates(dates[-1], dtend,
                                            freq)
                else:
                    raise err

        else:
            nobs = len(self.endog)
            if end > nobs:
                out_of_sample = end - nobs
                end = nobs

        return end, out_of_sample



#NOTE: this is just a stub for now, overwrite what we need to higher up
# and bring methods from child classes up as well.
class TimeSeriesModelResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale=1.):
        self.dates = model.dates
        super(TimeSeriesModelResults,
                self).__init__(model, params, normalized_cov_params, scale)

class TimeSeriesResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_attrs,
                                    _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(TimeSeriesResultsWrapper,
                      TimeSeriesModelResults)

if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    import datetime
    import pandas

    data = sm.datasets.macrodata.load()

    #make a DataFrame
    #TODO: attach a DataFrame to some of the datasets, for quicker use
    dates = [str(int(x[0])) +':'+ str(int(x[1])) \
             for x in data.data[['year','quarter']]]
    try:
        import scikits.timeseries as ts
        ts_dates = ts.date_array(start_date = ts.Date(year=1959,quarter=1,freq='Q'),
                             length=len(data.data))
    except:
        pass


    df = pandas.DataFrame(data.data[['realgdp','realinv','realcons']], index=dates)
    ex_mod = TimeSeriesModel(df)
    #ts_series = pandas.TimeSeries()


