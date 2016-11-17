"""
Tools for working with dates
"""
from statsmodels.compat.python import (lrange, lzip, lmap, string_types, long,
                                       callable, asstr, reduce, zip, map)
import re
import datetime

from pandas import datetools as pandas_datetools
from pandas import Int64Index, Period, PeriodIndex, Timestamp, DatetimeIndex
import numpy as np

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
_months_with_days = lzip(lrange(1,13), _mdays)
_month_to_day = dict(zip(map(str,lrange(1,13)), _months_with_days))
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
    Uses dateutil.parser.parse, but also handles monthly dates of the form
    1999m4, 1999:m4, 1999:mIV, 1999mIV and the same for quarterly data
    with q instead of m. It is not case sensitive. The default for annual
    data is the end of the year, which also differs from dateutil.
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
        return pandas_datetools.to_datetime(timestr, **kwargs)

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
    yr1, offset1 = lmap(int, start.replace(":","").split(split))
    if end is not None:
        end = end.lower()
        yr2, offset2 = lmap(int, end.replace(":","").split(split))
        length = (yr2 - yr1) * annual_freq + offset2
    elif length:
        yr2 = yr1 + length // annual_freq
        offset2 = length % annual_freq + (offset1 - 1)
    years = np.repeat(lrange(yr1+1, yr2), annual_freq).tolist()
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
    return lmap(date_parser, dates)

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

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> dates = sm.tsa.datetools.date_range('1960m1', length=nobs)


    Returns
    -------
    date_list : array
        A list of datetime types.
    """
    dates = date_range_str(start, end, length)
    return dates_from_str(dates)


def _get_index_loc(key, base_index):
    index = base_index
    date_index = isinstance(base_index, (PeriodIndex, DatetimeIndex))
    index_class = type(base_index)
    element_class = type(base_index[0])
    nobs = len(index)

    # Special handling for Int64Index
    if not date_index and isinstance(key, (int, long, np.integer)):
        # Negative indices (that lie in the Index)
        if key < 0 and -key < nobs:
            key = nobs + key
        # Out-of-sample (note that we include key itself in the new index)
        elif key > base_index[-1]:
            index = Int64Index(np.arange(base_index[0], key + 1))

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
                index = index_class(start=base_index[0], periods=key + 1,
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
                # First create an index that does *not* include `key`
                index = index_class(start=base_index[0], end=date_key,
                                    freq=base_index.freq)
                # Now we know the number of periods we need to make the new
                # index so that it does include `key`
                # (this method is to avoid relying on `TimeDelta` objects)
                index = index_class(start=base_index[0],
                                    periods=len(index) + 1,
                                    freq=base_index.freq)

    # Get the location (note that get_loc will throw a key error if key is
    # invalid)
    loc = index.get_loc(key)

    # Check if we now have a modified index
    modified_index = index is not base_index

    # (Never return the actual index object)
    if not modified_index:
        index = index.copy()

    # Return the index through the end of the loc / slice
    if isinstance(loc, slice):
        end = loc.stop
    else:
        end = loc

    return loc, index[:end + 1], modified_index
