import re
import datetime
from pandas import datetools as pandas_datetools
import scikits.timeseries as ts

#TODO: unify all the frequency information
def _date_from_idx(d1, idx, freq):
    """
    Returns the date from an index beyond the end of a date series.
    d1 is the datetime of the last date in the series. idx is the
    index distance of how far the next date should be from d1. Ie., 1 gives
    the next date from d1 at freq.
    """
    tsd1 = ts.Date(freq, datetime=d1)
    tsd2 = datetime.datetime.fromordinal((tsd1 + idx).toordinal())
    return tsd2


def _idx_from_dates(d1, d2, freq):
    """
    Returns an index offset from datetimes d1 and d2. d1 is expected to be the
    last date in a date series and d2 is the out of sample date.

    Note that it rounds down the index if the date is before the next date at
    freq.
    """
    d1 = ts.Date(freq, datetime=d1)
    d2 = ts.Date(freq, datetime=d2)
    return d2 - d1

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

mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_months_with_days = zip(range(1,13),mdays)
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

(([1-9][1-2]?)|(I?XI{0,2}|I?VI{0,3}|I{1,3}))  # match 1-12 or
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
    elif re.search(_m_pattern, timestr, flags):
        y,m = timestr.replace(":","").lower().split('m')
        month, day = _month_to_day[m.upper()]
        if _is_leap(y) and month == 2:
            day += 1
    elif re.search(_y_pattern, timestr, flags):
        month, day = 12, 31
        year = int(timestr)
    else:
        return pandas_datetools.parser.parse(timestr, parserinfo, kwargs)

    return datetime.datetime(year, month, day)
