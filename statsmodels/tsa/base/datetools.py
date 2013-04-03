import re
import datetime
from pandas import datetools as pandas_datetools
import numpy as np
from statsmodels.compatnp.py3k import asstr

#NOTE: All of these frequencies assume end of period (except wrt time)
try:
    from pandas.tseries.frequencies import get_offset
    class _freq_to_pandas_class(object):
        # being lazy, don't want to replace dictionary below
        def __getitem__(self, key):
            return get_offset(key)
    _freq_to_pandas = _freq_to_pandas_class()
except ImportError, err:
    _freq_to_pandas = {'B' : pandas_datetools.BDay(1),
                       'D' : pandas_datetools.day,
                       'W' : pandas_datetools.Week(weekday=6),
                       'M' : pandas_datetools.monthEnd,
                       'A' : pandas_datetools.yearEnd,
                       'Q' : pandas_datetools.quarterEnd}

def _index_date(date, dates):
    """
    Gets the index number of a date in a date index.

    Works in-sample and will return one past the end of the dates since
    prediction can start one out.

    Currently used to validate prediction start dates.

    If there dates are not of a fixed-frequency and date is not on the
    existing dates, then a ValueError is raised.
    """
    if isinstance(date, basestring):
        date = date_parser(date)
    try:
        return dates.get_loc(date)
    except KeyError, err:
        freq = _infer_freq(dates)
        if freq is None:
            #TODO: try to intelligently roll forward onto a date in the
            # index. Waiting to drop pandas 0.7.x support so this is
            # cleaner to do.
            raise ValueError("There is no frequency for these dates and "
                             "date %s is not in dates index. Try giving a "
                             "date that is in the dates index or use "
                             "an integer" % date)

        # we can start prediction at the end of endog
        if _idx_from_dates(dates[-1], date, freq) == 1:
            return len(dates)

        raise ValueError("date %s not in date index. Try giving a "
                         "date that is in the dates index or use an integer"
                         % date)

def _date_from_idx(d1, idx, freq):
    """
    Returns the date from an index beyond the end of a date series.
    d1 is the datetime of the last date in the series. idx is the
    index distance of how far the next date should be from d1. Ie., 1 gives
    the next date from d1 at freq.

    Notes
    -----
    This does not do any rounding to make sure that d1 is actually on the
    offset. For now, this needs to be taken care of before you get here.
    """
    return d1 + idx * _freq_to_pandas[freq]

def _idx_from_dates(d1, d2, freq):
    """
    Returns an index offset from datetimes d1 and d2. d1 is expected to be the
    last date in a date series and d2 is the out of sample date.

    Notes
    -----
    Rounds down the index if the end date is before the next date at freq.
    Does not check the start date to see whether it is on the offest but
    assumes that it is.
    """
    try: # pandas 0.8.x
        from pandas import DatetimeIndex
        return len(DatetimeIndex(start=d1, end=d2,
                                 freq = _freq_to_pandas[freq])) - 1
    except ImportError, err:
        from pandas import DateRange
        return len(DateRange(d1, d2, offset = _freq_to_pandas[freq])) - 1

_quarter_to_day = {
        "1" : (3, 31),
        "2" : (6, 30),
        "3" : (9, 30),
        "4" : (12, 31),
        "I" : (3, 31),
        "II" : (6, 30),
        "III" : (9, 30),
        "IV" : (12, 31)
        }

_mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_months_with_days = zip(range(1,13), _mdays)
_month_to_day = dict(zip(map(str,range(1,13)), _months_with_days))
_month_to_day.update(dict(zip(["I", "II", "III", "IV", "V", "VI",
                               "VII", "VIII", "IX", "X", "XI", "XII"],
                               _months_with_days)))

# regex patterns
_y_pattern = '^\d?\d?\d?\d$'

_q_pattern = '''
^               # beginning of string
\d?\d?\d?\d     # match any number 1-9999, includes leading zeros

(:?q)           # use q or a : as a separator

([1-4]|(I{1,3}V?)) # match 1-4 or I-IV roman numerals

$               # end of string
'''

_m_pattern = '''
^               # beginning of string
\d?\d?\d?\d     # match any number 1-9999, includes leading zeros

(:?m)           # use m or a : as a separator

(([1-9][0-2]?)|(I?XI{0,2}|I?VI{0,3}|I{1,3}))  # match 1-12 or
                                              # I-XII roman numerals

$               # end of string
'''

#NOTE: see also ts.extras.isleapyear, which accepts a sequence
def _is_leap(year):
    year = int(year)
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def date_parser(timestr, parserinfo=None, **kwargs):
    """
    Uses dateutils.parser.parse, but also handles monthly dates of the form
    1999m4, 1999:m4, 1999:mIV, 1999mIV and the same for quarterly data
    with q instead of m. It is not case sensitive. The default for annual
    data is the end of the year, which also differs from dateutils.
    """
    flags = re.IGNORECASE | re.VERBOSE
    if re.search(_q_pattern, timestr, flags):
        y,q = timestr.replace(":","").lower().split('q')
        month, day = _quarter_to_day[q.upper()]
        year = int(y)
    elif re.search(_m_pattern, timestr, flags):
        y,m = timestr.replace(":","").lower().split('m')
        month, day = _month_to_day[m.upper()]
        year = int(y)
        if _is_leap(y) and month == 2:
            day += 1
    elif re.search(_y_pattern, timestr, flags):
        month, day = 12, 31
        year = int(timestr)
    else:
        return pandas_datetools.parser.parse(timestr, parserinfo, **kwargs)

    return datetime.datetime(year, month, day)

def date_range_str(start, end=None, length=None):
    """
    Returns a list of abbreviated date strings.

    Parameters
    ----------
    start : str
        The first abbreviated date, for instance, '1965q1' or '1965m1'
    end : str, optional
        The last abbreviated date if length is None.
    length : int, optional
        The length of the returned array of end is None.

    Returns
    -------
    date_range : list
        List of strings
    """
    flags = re.IGNORECASE | re.VERBOSE
    #_check_range_inputs(end, length, freq)
    start = start.lower()
    if re.search(_m_pattern, start, flags):
        annual_freq = 12
        split = 'm'
    elif re.search(_q_pattern, start, flags):
        annual_freq = 4
        split = 'q'
    elif re.search(_y_pattern, start, flags):
        annual_freq = 1
        start += 'a1' # hack
        if end:
            end += 'a1'
        split = 'a'
    else:
        raise ValueError("Date %s not understood" % start)
    yr1, offset1 = map(int, start.replace(":","").split(split))
    if end is not None:
        end = end.lower()
        yr2, offset2 = map(int, end.replace(":","").split(split))
        length = (yr2 - yr1) * annual_freq + offset2
    elif length:
        yr2 = yr1 + length // annual_freq
        offset2 = length % annual_freq
    years = np.repeat(range(yr1+1, yr2), annual_freq).tolist()
    years = np.r_[[str(yr1)]*(annual_freq+1-offset1), years] # tack on first year
    years = np.r_[years, [str(yr2)]*offset2] # tack on last year
    if split != 'a':
        offset = np.tile(np.arange(1, annual_freq+1), yr2-yr1-1)
        offset = np.r_[np.arange(offset1, annual_freq+1).astype('a2'), offset]
        offset = np.r_[offset, np.arange(1,offset2+1).astype('a2')]
        date_arr_range = [''.join([i, split, asstr(j)]) for i,j in
                                                        zip(years, offset)]
    else:
        date_arr_range = years.tolist()
    return date_arr_range

def dates_from_str(dates):
    """
    Turns a sequence of date strings and returns a list of datetime.

    Parameters
    ----------
    dates : array-like
        A sequence of abbreviated dates as string. For instance,
        '1996m1' or '1996Q1'. The datetime dates are at the end of the
        period.

    Returns
    -------
    date_list : array
        A list of datetime types.
    """
    return map(date_parser, dates)

def dates_from_range(start, end=None, length=None):
    """
    Turns a sequence of date strings and returns a list of datetime.

    Parameters
    ----------
    start : str
        The first abbreviated date, for instance, '1965q1' or '1965m1'
    end : str, optional
        The last abbreviated date if length is None.
    length : int, optional
        The length of the returned array of end is None.

    Example
    -------
    >>> import statsmodels.api as sm
    >>> dates = sm.tsa.datetools.date_range('1960m1', length=nobs)


    Returns
    -------
    date_list : array
        A list of datetime types.
    """
    dates = date_range_str(start, end, length)
    return dates_from_str(dates)

def _add_datetimes(dates):
    return reduce(lambda x, y: y+x, dates)

def _infer_freq(dates):
    try:
        from pandas.tseries.api import infer_freq
        freq = infer_freq(dates)
        return freq
    except ImportError:
        pass

    timedelta = datetime.timedelta
    nobs = min(len(dates), 6)
    if nobs == 1:
        raise ValueError("Cannot infer frequency from one date")
    if hasattr(dates, 'values'):
        dates = dates.values # can't do a diff on a DateIndex
    diff = np.diff(dates[:nobs])
    delta = _add_datetimes(diff)
    nobs -= 1 # after diff
    if delta == timedelta(nobs): #greedily assume 'D'
        return 'D'
    elif delta == timedelta(nobs + 2):
        return 'B'
    elif delta == timedelta(7*nobs):
        return 'W'
    elif delta >= timedelta(28*nobs) and delta <= timedelta(31*nobs):
        return 'M'
    elif delta >= timedelta(90*nobs) and delta <= timedelta(92*nobs):
        return 'Q'
    elif delta >= timedelta(365 * nobs) and delta <= timedelta(366 * nobs):
        return 'A'
    else:
        return
