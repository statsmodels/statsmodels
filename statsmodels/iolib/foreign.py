"""
Input/Output tools for working with binary data.

The Stata input tools were originally written by Joe Presbrey as part of PyDTA.

You can find more information here http://presbrey.mit.edu/PyDTA

See also
---------
numpy.lib.io
"""
from statsmodels.compat.python import (zip, lzip, lmap, lrange, string_types, long, lfilter,
                                       asbytes, asstr, range, PY3)
from struct import unpack, calcsize, pack
from struct import error as struct_error
import datetime
import sys
import numpy as np
from numpy.lib._iotools import _is_string_like, easy_dtype
import statsmodels.tools.data as data_util
from pandas import isnull
from statsmodels.iolib.openfile import get_file_obj

_date_formats = ["%tc", "%tC", "%td", "%tw", "%tm", "%tq", "%th", "%ty"]

def _datetime_to_stata_elapsed(date, fmt):
    """
    Convert from datetime to SIF. http://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    date : datetime.datetime
        The date to convert to the Stata Internal Format given by fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
    """
    if not isinstance(date, datetime.datetime):
        raise ValueError("date should be datetime.datetime format")
    stata_epoch = datetime.datetime(1960, 1, 1)
    if fmt in ["%tc", "tc"]:
        delta = date - stata_epoch
        return (delta.days * 86400000 + delta.seconds*1000 +
                delta.microseconds/1000)
    elif fmt in ["%tC", "tC"]:
        from warnings import warn
        warn("Stata Internal Format tC not supported.", UserWarning)
        return date
    elif fmt in ["%td", "td"]:
        return (date- stata_epoch).days
    elif fmt in ["%tw", "tw"]:
        return (52*(date.year-stata_epoch.year) +
                (date - datetime.datetime(date.year, 1, 1)).days / 7)
    elif fmt in ["%tm", "tm"]:
        return (12 * (date.year - stata_epoch.year) + date.month - 1)
    elif fmt in ["%tq", "tq"]:
        return 4*(date.year-stata_epoch.year) + int((date.month - 1)/3)
    elif fmt in ["%th", "th"]:
        return 2 * (date.year - stata_epoch.year) + int(date.month > 6)
    elif fmt in ["%ty", "ty"]:
        return date.year
    else:
        raise ValueError("fmt %s not understood" % fmt)

def _stata_elapsed_date_to_datetime(date, fmt):
    """
    Convert from SIF to datetime. http://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    date : int
        The Stata Internal Format date to convert to datetime according to fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty

    Examples
    --------
    >>> _stata_elapsed_date_to_datetime(52, "%tw")                                datetime.datetime(1961, 1, 1, 0, 0)

    Notes
    -----
    datetime/c - tc
        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day
    datetime/C - tC - NOT IMPLEMENTED
        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds
    date - td
        days since 01jan1960 (01jan1960 = 0)
    weekly date - tw
        weeks since 1960w1
        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.
        The datetime value is the start of the week in terms of days in the
        year, not ISO calendar weeks.
    monthly date - tm
        months since 1960m1
    quarterly date - tq
        quarters since 1960q1
    half-yearly date - th
        half-years since 1960h1 yearly
    date - ty
        years since 0000

    If you don't have pandas with datetime support, then you can't do
    milliseconds accurately.
    """
    #NOTE: we could run into overflow / loss of precision situations here
    # casting to int, but I'm not sure what to do. datetime won't deal with
    # numpy types and numpy datetime isn't mature enough / we can't rely on
    # pandas version > 0.7.1
    #TODO: IIRC relative delta doesn't play well with np.datetime?
    date = int(date)
    stata_epoch = datetime.datetime(1960, 1, 1)
    if fmt in ["%tc", "tc"]:
        from dateutil.relativedelta import relativedelta
        return stata_epoch + relativedelta(microseconds=date*1000)
    elif fmt in ["%tC", "tC"]:
        from warnings import warn
        warn("Encountered %tC format. Leaving in Stata Internal Format.",
             UserWarning)
        return date
    elif fmt in ["%td", "td"]:
        return stata_epoch + datetime.timedelta(int(date))
    elif fmt in ["%tw", "tw"]: # does not count leap days - 7 days is a week
        year = datetime.datetime(stata_epoch.year + date // 52, 1, 1)
        day_delta = (date  % 52 ) * 7
        return year + datetime.timedelta(int(day_delta))
    elif fmt in ["%tm", "tm"]:
        year = stata_epoch.year + date // 12
        month_delta = (date  % 12 ) + 1
        return datetime.datetime(year, month_delta, 1)
    elif fmt in ["%tq", "tq"]:
        year = stata_epoch.year + date // 4
        month_delta = (date % 4) * 3 + 1
        return datetime.datetime(year, month_delta, 1)
    elif fmt in ["%th", "th"]:
        year = stata_epoch.year + date // 2
        month_delta = (date % 2) * 6 + 1
        return datetime.datetime(year, month_delta, 1)
    elif fmt in ["%ty", "ty"]:
        if date > 0:
            return datetime.datetime(date, 1, 1)
        else: # don't do negative years bc can't mix dtypes in column
            raise ValueError("Year 0 and before not implemented")
    else:
        raise ValueError("Date fmt %s not understood" % fmt)

### Helper classes for StataReader ###

class _StataMissingValue(object):
    """
    An observation's missing value.

    Parameters
    -----------
    offset
    value

    Attributes
    ----------
    string
    value

    Notes
    -----
    More information: <http://www.stata.com/help.cgi?missing>
    """

    def __init__(self, offset, value):
        self._value = value
        if isinstance(value, (int, long)):
            self._str = value-offset is 1 and \
                    '.' or ('.' + chr(value-offset+96))
        else:
            self._str = '.'
    string = property(lambda self: self._str, doc="The Stata representation of \
the missing value: '.', '.a'..'.z'")
    value = property(lambda self: self._value, doc='The binary representation \
of the missing value.')
    def __str__(self): return self._str
    __str__.__doc__ = string.__doc__

class _StataVariable(object):
    """
    A dataset variable.  Not intended for public use.

    Parameters
    ----------
    variable_data

    Attributes
    -----------
    format : str
        Stata variable format.  See notes for more information.
    index : int
        Zero-index column index of variable.
    label : str
        Data Label
    name : str
        Variable name
    type : str
        Stata data type.  See notes for more information.
    value_format : str
        Value format.

    Notes
    -----
    More information: http://www.stata.com/help.cgi?format
    """
    def __init__(self, variable_data):
        self._data = variable_data

    def __int__(self):
        return self.index

    def __str__(self):
        return self.name
    index = property(lambda self: self._data[0], doc='the variable\'s index \
within an observation')
    type = property(lambda self: self._data[1], doc='the data type of \
variable\n\nPossible types are:\n{1..244:string, b:byte, h:int, l:long, \
f:float, d:double)')
    name = property(lambda self: self._data[2], doc='the name of the variable')
    format = property(lambda self: self._data[4], doc='the variable\'s Stata \
format')
    value_format = property(lambda self: self._data[5], doc='the variable\'s \
value format')
    label = property(lambda self: self._data[6], doc='the variable\'s label')
    __int__.__doc__ = index.__doc__
    __str__.__doc__ = name.__doc__

class StataReader(object):
    """
    Stata .dta file reader.

    Provides methods to return the metadata of a Stata .dta file and
    a generator for the data itself.

    Parameters
    ----------
    file : file-like
        A file-like object representing a Stata .dta file.
    missing_values : bool
        If missing_values is True, parse missing_values and return a
        Missing Values object instead of None.
    encoding : string, optional
        Used for Python 3 only. Encoding to use when reading the .dta file.
        Defaults to `locale.getpreferredencoding`

    See also
    --------
    statsmodels.iolib.foreign.genfromdta
    pandas.read_stata
    pandas.io.stata.StataReader

    Notes
    -----
    This is known only to work on file formats 113 (Stata 8/9), 114
    (Stata 10/11), and 115 (Stata 12).  Needs to be tested on older versions.
    Known not to work on format 104, 108. If you have the documentation for
    older formats, please contact the developers.

    For more information about the .dta format see
    http://www.stata.com/help.cgi?dta
    http://www.stata.com/help.cgi?dta_113
    """

    _header = {}
    _data_location = 0
    _col_sizes = ()
    _has_string_data = False
    _missing_values = False
    #type          code
    #--------------------
    #str1        1 = 0x01
    #str2        2 = 0x02
    #...
    #str244    244 = 0xf4
    #byte      251 = 0xfb  (sic)
    #int       252 = 0xfc
    #long      253 = 0xfd
    #float     254 = 0xfe
    #double    255 = 0xff
    #--------------------
    #NOTE: the byte type seems to be reserved for categorical variables
    # with a label, but the underlying variable is -127 to 100
    # we're going to drop the label and cast to int
    DTYPE_MAP = dict(lzip(lrange(1,245), ['a' + str(i) for i in range(1,245)]) + \
                    [(251, np.int16),(252, np.int32),(253, int),
                        (254, np.float32), (255, np.float64)])
    TYPE_MAP = lrange(251)+list('bhlfd')
    #NOTE: technically, some of these are wrong. there are more numbers
    # that can be represented. it's the 27 ABOVE and BELOW the max listed
    # numeric data type in [U] 12.2.2 of the 11.2 manual
    MISSING_VALUES = { 'b': (-127,100), 'h': (-32767, 32740), 'l':
            (-2147483647, 2147483620), 'f': (-1.701e+38, +1.701e+38), 'd':
            (-1.798e+308, +8.988e+307) }

    def __init__(self, fname, missing_values=False, encoding=None):
        if encoding == None:
            import locale
            self._encoding = locale.getpreferredencoding()
        else:
            self._encoding = encoding
        self._missing_values = missing_values
        self._parse_header(fname)

    def file_headers(self):
        """
        Returns all .dta file headers.

        out: dict
            Has keys typlist, data_label, lbllist, varlist, nvar, filetype,
            ds_format, nobs, fmtlist, vlblist, time_stamp, srtlist, byteorder
        """
        return self._header

    def file_format(self):
        """
        Returns the file format.

        Returns
        -------
        out : int

        Notes
        -----
        Format 113: Stata 8/9
        Format 114: Stata 10/11
        Format 115: Stata 12
        """
        return self._header['ds_format']

    def file_label(self):
        """
        Returns the dataset's label.

        Returns
        -------
        out: string
        """
        return self._header['data_label']

    def file_timestamp(self):
        """
        Returns the date and time Stata recorded on last file save.

        Returns
        -------
        out : str
        """
        return self._header['time_stamp']

    def variables(self):
        """
        Returns a list of the dataset's StataVariables objects.
        """
        return lmap(_StataVariable, zip(lrange(self._header['nvar']),
            self._header['typlist'], self._header['varlist'],
            self._header['srtlist'],
            self._header['fmtlist'], self._header['lbllist'],
            self._header['vlblist']))

    def dataset(self, as_dict=False):
        """
        Returns a Python generator object for iterating over the dataset.


        Parameters
        ----------
        as_dict : bool, optional
            If as_dict is True, yield each row of observations as a dict.
            If False, yields each row of observations as a list.

        Returns
        -------
        Generator object for iterating over the dataset.  Yields each row of
        observations as a list by default.

        Notes
        -----
        If missing_values is True during instantiation of StataReader then
        observations with _StataMissingValue(s) are not filtered and should
        be handled by your applcation.
        """

        try:
            self._file.seek(self._data_location)
        except Exception:
            pass

        if as_dict:
            vars = lmap(str, self.variables())
            for i in range(len(self)):
                yield dict(zip(vars, self._next()))
        else:
            for i in range(self._header['nobs']):
                yield self._next()

    ### Python special methods

    def __len__(self):
        """
        Return the number of observations in the dataset.

        This value is taken directly from the header and includes observations
        with missing values.
        """
        return self._header['nobs']

    def __getitem__(self, k):
        """
        Seek to an observation indexed k in the file and return it, ordered
        by Stata's output to the .dta file.

        k is zero-indexed.  Prefer using R.data() for performance.
        """
        if not (isinstance(k, (int, long))) or k < 0 or k > len(self)-1:
            raise IndexError(k)
        loc = self._data_location + sum(self._col_size()) * k
        if self._file.tell() != loc:
            self._file.seek(loc)
        return self._next()

    ### Private methods

    def _null_terminate(self, s, encoding):
        if PY3: # have bytes not strings, so must decode
            null_byte = asbytes('\x00')
            try:
                s = s.lstrip(null_byte)[:s.index(null_byte)]
            except:
                pass
            return s.decode(encoding)
        else:
            null_byte = asbytes('\x00')
            try:
                return s.lstrip(null_byte)[:s.index(null_byte)]
            except:
                return s

    def _parse_header(self, file_object):
        self._file = file_object
        encoding = self._encoding

        # parse headers
        self._header['ds_format'] = unpack('b', self._file.read(1))[0]

        if self._header['ds_format'] not in [113, 114, 115]:
            raise ValueError("Only file formats >= 113 (Stata >= 9)"
                             " are supported.  Got format %s.  Please report "
                             "if you think this error is incorrect." %
                             self._header['ds_format'])
        byteorder = self._header['byteorder'] = unpack('b',
                self._file.read(1))[0]==0x1 and '>' or '<'
        self._header['filetype'] = unpack('b', self._file.read(1))[0]
        self._file.read(1)
        nvar = self._header['nvar'] = unpack(byteorder+'h',
                self._file.read(2))[0]
        self._header['nobs'] = unpack(byteorder+'i', self._file.read(4))[0]
        self._header['data_label'] = self._null_terminate(self._file.read(81),
                                                            encoding)
        self._header['time_stamp'] = self._null_terminate(self._file.read(18),
                                                            encoding)

        # parse descriptors
        typlist =[ord(self._file.read(1)) for i in range(nvar)]
        self._header['typlist'] = [self.TYPE_MAP[typ] for typ in typlist]
        self._header['dtyplist'] = [self.DTYPE_MAP[typ] for typ in typlist]
        self._header['varlist'] = [self._null_terminate(self._file.read(33),
                                    encoding) for i in range(nvar)]
        self._header['srtlist'] = unpack(byteorder+('h'*(nvar+1)),
                self._file.read(2*(nvar+1)))[:-1]
        if self._header['ds_format'] <= 113:
            self._header['fmtlist'] = \
                    [self._null_terminate(self._file.read(12), encoding) \
                    for i in range(nvar)]
        else:
            self._header['fmtlist'] = \
                    [self._null_terminate(self._file.read(49), encoding) \
                    for i in range(nvar)]
        self._header['lbllist'] = [self._null_terminate(self._file.read(33),
                                encoding) for i in range(nvar)]
        self._header['vlblist'] = [self._null_terminate(self._file.read(81),
                                encoding) for i in range(nvar)]

        # ignore expansion fields
        # When reading, read five bytes; the last four bytes now tell you the
        # size of the next read, which you discard.  You then continue like
        # this until you read 5 bytes of zeros.

        while True:
            data_type = unpack(byteorder+'b', self._file.read(1))[0]
            data_len = unpack(byteorder+'i', self._file.read(4))[0]
            if data_type == 0:
                break
            self._file.read(data_len)

        # other state vars
        self._data_location = self._file.tell()
        self._has_string_data = len(lfilter(lambda x: isinstance(x, int),
            self._header['typlist'])) > 0
        self._col_size()

    def _calcsize(self, fmt):
        return isinstance(fmt, int) and fmt or \
                calcsize(self._header['byteorder']+fmt)

    def _col_size(self, k = None):
        """Calculate size of a data record."""
        if len(self._col_sizes) == 0:
            self._col_sizes = lmap(lambda x: self._calcsize(x),
                    self._header['typlist'])
        if k == None:
            return self._col_sizes
        else:
            return self._col_sizes[k]

    def _unpack(self, fmt, byt):
        d = unpack(self._header['byteorder']+fmt, byt)[0]
        if fmt[-1] in self.MISSING_VALUES:
            nmin, nmax = self.MISSING_VALUES[fmt[-1]]
            if d < nmin or d > nmax:
                if self._missing_values:
                    return _StataMissingValue(nmax, d)
                else:
                    return None
        return d

    def _next(self):
        typlist = self._header['typlist']
        if self._has_string_data:
            data = [None]*self._header['nvar']
            for i in range(len(data)):
                if isinstance(typlist[i], int):
                    data[i] = self._null_terminate(self._file.read(typlist[i]),
                                self._encoding)
                else:
                    data[i] = self._unpack(typlist[i],
                            self._file.read(self._col_size(i)))
            return data
        else:
            return lmap(lambda i: self._unpack(typlist[i],
                self._file.read(self._col_size(i))),
                lrange(self._header['nvar']))

def _set_endianness(endianness):
    if endianness.lower() in ["<", "little"]:
        return "<"
    elif endianness.lower() in [">", "big"]:
        return ">"
    else: # pragma : no cover
        raise ValueError("Endianness %s not understood" % endianness)

def _dtype_to_stata_type(dtype):
    """
    Converts dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 244 are strings of this length
    251 - chr(251) - for int8 and int16, byte
    252 - chr(252) - for int32, int
    253 - chr(253) - for int64, long
    254 - chr(254) - for float32, float
    255 - chr(255) - double, double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
    #TODO: expand to handle datetime to integer conversion
    if dtype.type == np.string_:
        return chr(dtype.itemsize)
    elif dtype.type == np.object_: # try to coerce it to the biggest string
                        # not memory efficient, what else could we do?
        return chr(244)
    elif dtype == np.float64:
        return chr(255)
    elif dtype == np.float32:
        return chr(254)
    elif dtype == np.int64:
        return chr(253)
    elif dtype == np.int32:
        return chr(252)
    elif dtype == np.int8 or dtype == np.int16: # ok to assume bytes?
        return chr(251)
    else: # pragma : no cover
        raise ValueError("Data type %s not currently understood. "
                         "Please report an error to the developers." % dtype)

def _dtype_to_default_stata_fmt(dtype):
    """
    Maps numpy dtype to stata's default format for this type. Not terribly
    important since users can change this in Stata. Semantics are

    string  -> "%DDs" where DD is the length of the string
    float64 -> "%10.0g"
    float32 -> "%9.0g"
    int64   -> "%9.0g"
    int32   -> "%9.0g"
    int16   -> "%9.0g"
    int8    -> "%8.0g"
    """
    #TODO: expand this to handle a default datetime format?
    if dtype.type == np.string_:
        return "%" + str(dtype.itemsize) + "s"
    elif dtype.type == np.object_:
        return "%244s"
    elif dtype == np.float64:
        return "%10.0g"
    elif dtype == np.float32:
        return "%9.0g"
    elif dtype == np.int64:
        return "%9.0g"
    elif dtype == np.int32:
        return "%8.0g"
    elif dtype == np.int8 or dtype == np.int16: # ok to assume bytes?
        return "%8.0g"
    else: # pragma : no cover
        raise ValueError("Data type %s not currently understood. "
                         "Please report an error to the developers." % dtype)

def _pad_bytes(name, length):
    """
    Takes a char string and pads it wih null bytes until it's length chars
    """
    return name + "\x00" * (length - len(name))

def _default_names(nvar):
    """
    Returns default Stata names v1, v2, ... vnvar
    """
    return ["v%d" % i for i in range(1,nvar+1)]

def _convert_datetime_to_stata_type(fmt):
    """
    Converts from one of the stata date formats to a type in TYPE_MAP
    """
    if fmt in ["tc", "%tc", "td", "%td", "tw", "%tw", "tm", "%tm", "tq",
               "%tq", "th", "%th", "ty", "%ty"]:
        return np.float64 # Stata expects doubles for SIFs
    else:
        raise ValueError("fmt %s not understood" % fmt)

def _maybe_convert_to_int_keys(convert_dates, varlist):
    new_dict = {}
    for key in convert_dates:
        if not convert_dates[key].startswith("%"): # make sure proper fmts
            convert_dates[key] = "%" + convert_dates[key]
        if key in varlist:
            new_dict.update({varlist.index(key) : convert_dates[key]})
        else:
            if not isinstance(key, int):
                raise ValueError("convery_dates key is not in varlist "
                        "and is not an int")
            new_dict.update({key : convert_dates[key]})
    return new_dict

_type_converters = {253 : np.long, 252 : int}

class StataWriter(object):
    """
    A class for writing Stata binary dta files from array-like objects

    Parameters
    ----------
    fname : file path or buffer
        Where to save the dta file.
    data : array-like
        Array-like input to save. Pandas objects are also accepted.
    convert_dates : dict
        Dictionary mapping column of datetime types to the stata internal
        format that you want to use for the dates. Options are
        'tc', 'td', 'tm', 'tw', 'th', 'tq', 'ty'. Column can be either a
        number or a name.
    encoding : str
        Default is latin-1. Note that Stata does not support unicode.
    byteorder : str
        Can be ">", "<", "little", or "big". The default is None which uses
        `sys.byteorder`

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Examples
    --------
    >>> writer = StataWriter('./data_file.dta', data)
    >>> writer.write_file()

    Or with dates

    >>> writer = StataWriter('./date_data_file.dta', date, {2 : 'tw'})
    >>> writer.write_file()
    """
    #type          code
    #--------------------
    #str1        1 = 0x01
    #str2        2 = 0x02
    #...
    #str244    244 = 0xf4
    #byte      251 = 0xfb  (sic)
    #int       252 = 0xfc
    #long      253 = 0xfd
    #float     254 = 0xfe
    #double    255 = 0xff
    #--------------------
    #NOTE: the byte type seems to be reserved for categorical variables
    # with a label, but the underlying variable is -127 to 100
    # we're going to drop the label and cast to int
    DTYPE_MAP = dict(lzip(lrange(1,245), ['a' + str(i) for i in range(1,245)]) + \
                    [(251, np.int16),(252, np.int32),(253, int),
                        (254, np.float32), (255, np.float64)])
    TYPE_MAP = lrange(251)+list('bhlfd')
    MISSING_VALUES = { 'b': 101,
                       'h': 32741,
                       'l' : 2147483621,
                       'f': 1.7014118346046923e+38,
                       'd': 8.98846567431158e+307}
    def __init__(self, fname, data, convert_dates=None, encoding="latin-1",
                 byteorder=None):

        self._convert_dates = convert_dates
        # attach nobs, nvars, data, varlist, typlist
        if data_util._is_using_pandas(data, None):
            self._prepare_pandas(data)

        elif data_util._is_array_like(data, None):
            data = np.asarray(data)
            if data_util._is_structured_ndarray(data):
                self._prepare_structured_array(data)
            else:
                if convert_dates is not None:
                    raise ValueError("Not able to convert dates in a plain"
                                     " ndarray.")
                self._prepare_ndarray(data)

        else: # pragma : no cover
            raise ValueError("Type %s for data not understood" % type(data))


        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        self._encoding = encoding
        self._file = get_file_obj(fname, 'wb', encoding)

    def _write(self, to_write):
        """
        Helper to call asbytes before writing to file for Python 3 compat.
        """
        self._file.write(asbytes(to_write))

    def _prepare_structured_array(self, data):
        self.nobs = len(data)
        self.nvar = len(data.dtype)
        self.data = data
        self.datarows = iter(data)
        dtype = data.dtype
        descr = dtype.descr
        if dtype.names is None:
            varlist = _default_names(nvar)
        else:
            varlist = dtype.names

        # check for datetime and change the type
        convert_dates = self._convert_dates
        if convert_dates is not None:
            convert_dates = _maybe_convert_to_int_keys(convert_dates,
                                                      varlist)
            self._convert_dates = convert_dates
            for key in convert_dates:
                descr[key] = (
                        descr[key][0],
                        _convert_datetime_to_stata_type(convert_dates[key])
                                )
            dtype = np.dtype(descr)

        self.varlist = varlist
        self.typlist = [_dtype_to_stata_type(dtype[i])
                        for i in range(self.nvar)]
        self.fmtlist = [_dtype_to_default_stata_fmt(dtype[i])
                        for i in range(self.nvar)]
        # set the given format for the datetime cols
        if convert_dates is not None:
            for key in convert_dates:
                self.fmtlist[key] = convert_dates[key]


    def _prepare_ndarray(self, data):
        if data.ndim == 1:
            data = data[:,None]
        self.nobs, self.nvar = data.shape
        self.data = data
        self.datarows = iter(data)
        #TODO: this should be user settable
        dtype = data.dtype
        self.varlist = _default_names(self.nvar)
        self.typlist = [_dtype_to_stata_type(dtype) for i in range(self.nvar)]
        self.fmtlist = [_dtype_to_default_stata_fmt(dtype)
                        for i in range(self.nvar)]

    def _prepare_pandas(self, data):
        #NOTE: we might need a different API / class for pandas objects so
        # we can set different semantics - handle this with a PR to pandas.io
        class DataFrameRowIter(object):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                for i, row in data.iterrows():
                    yield row

        data = data.reset_index()
        self.datarows = DataFrameRowIter(data)
        self.nobs, self.nvar = data.shape
        self.data = data
        self.varlist = data.columns.tolist()
        dtypes = data.dtypes
        convert_dates = self._convert_dates
        if convert_dates is not None:
            convert_dates = _maybe_convert_to_int_keys(convert_dates,
                                                      self.varlist)
            self._convert_dates = convert_dates
            for key in convert_dates:
                new_type = _convert_datetime_to_stata_type(convert_dates[key])
                dtypes[key] = np.dtype(new_type)
        self.typlist = [_dtype_to_stata_type(dt) for dt in dtypes]
        self.fmtlist = [_dtype_to_default_stata_fmt(dt) for dt in dtypes]
        # set the given format for the datetime cols
        if convert_dates is not None:
            for key in convert_dates:
                self.fmtlist[key] = convert_dates[key]

    def write_file(self):
        self._write_header()
        self._write_descriptors()
        self._write_variable_labels()
        # write 5 zeros for expansion fields
        self._write(_pad_bytes("", 5))
        if self._convert_dates is None:
            self._write_data_nodates()
        else:
            self._write_data_dates()
        #self._write_value_labels()

    def _write_header(self, data_label=None, time_stamp=None):
        byteorder = self._byteorder
        # ds_format - just use 114
        self._write(pack("b", 114))
        # byteorder
        self._write(byteorder == ">" and "\x01" or "\x02")
        # filetype
        self._write("\x01")
        # unused
        self._write("\x00")
        # number of vars, 2 bytes
        self._write(pack(byteorder+"h", self.nvar)[:2])
        # number of obs, 4 bytes
        self._write(pack(byteorder+"i", self.nobs)[:4])
        # data label 81 bytes, char, null terminated
        if data_label is None:
            self._write(self._null_terminate(_pad_bytes("", 80),
                             self._encoding))
        else:
            self._write(self._null_terminate(_pad_bytes(data_label[:80],
                                80), self._encoding))
        # time stamp, 18 bytes, char, null terminated
        # format dd Mon yyyy hh:mm
        if time_stamp is None:
            time_stamp = datetime.datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError("time_stamp should be datetime type")
        self._write(self._null_terminate(
                            time_stamp.strftime("%d %b %Y %H:%M"),
                            self._encoding))

    def _write_descriptors(self, typlist=None, varlist=None, srtlist=None,
                           fmtlist=None, lbllist=None):
        nvar = self.nvar
        # typlist, length nvar, format byte array
        for typ in self.typlist:
            self._write(typ)

        # varlist, length 33*nvar, char array, null terminated
        for name in self.varlist:
            name = self._null_terminate(name, self._encoding)
            name = _pad_bytes(asstr(name[:32]), 33)
            self._write(name)

        # srtlist, 2*(nvar+1), int array, encoded by byteorder
        srtlist = _pad_bytes("", (2*(nvar+1)))
        self._write(srtlist)

        # fmtlist, 49*nvar, char array
        for fmt in self.fmtlist:
            self._write(_pad_bytes(fmt, 49))

        # lbllist, 33*nvar, char array
        #NOTE: this is where you could get fancy with pandas categorical type
        for i in range(nvar):
            self._write(_pad_bytes("", 33))

    def _write_variable_labels(self, labels=None):
        nvar = self.nvar
        if labels is None:
            for i in range(nvar):
                self._write(_pad_bytes("", 81))

    def _write_data_nodates(self):
        data = self.datarows
        byteorder = self._byteorder
        TYPE_MAP = self.TYPE_MAP
        typlist = self.typlist
        for row in data:
            #row = row.squeeze().tolist() # needed for structured arrays
            for i,var in enumerate(row):
                typ = ord(typlist[i])
                if typ <= 244: # we've got a string
                    if len(var) < typ:
                        var = _pad_bytes(asstr(var), len(var) + 1)
                    self._write(var)
                else:
                    try:
                        self._write(pack(byteorder+TYPE_MAP[typ], var))
                    except struct_error:
                        # have to be strict about type pack won't do any
                        # kind of casting
                        self._write(pack(byteorder+TYPE_MAP[typ],
                                    _type_converters[typ](var)))

    def _write_data_dates(self):
        convert_dates = self._convert_dates
        data = self.datarows
        byteorder = self._byteorder
        TYPE_MAP = self.TYPE_MAP
        MISSING_VALUES = self.MISSING_VALUES
        typlist = self.typlist
        for row in data:
            #row = row.squeeze().tolist() # needed for structured arrays
            for i,var in enumerate(row):
                typ = ord(typlist[i])
                #NOTE: If anyone finds this terribly slow, there is
                # a vectorized way to convert dates, see genfromdta for going
                # from int to datetime and reverse it. will copy data though
                if i in convert_dates:
                    var = _datetime_to_stata_elapsed(var, self.fmtlist[i])
                if typ <= 244: # we've got a string
                    if isnull(var):
                        var = "" # missing string
                    if len(var) < typ:
                        var = _pad_bytes(var, len(var) + 1)
                    self._write(var)
                else:
                    if isnull(var): # this only matters for floats
                        var = MISSING_VALUES[typ]
                    self._write(pack(byteorder+TYPE_MAP[typ], var))


    def _null_terminate(self, s, encoding):
        null_byte = '\x00'
        if PY3:
            s += null_byte
            return s.encode(encoding)
        else:
            s += null_byte
            return s

def genfromdta(fname, missing_flt=-999., encoding=None, pandas=False,
                convert_dates=True):
    """
    Returns an ndarray or DataFrame from a Stata .dta file.

    Parameters
    ----------
    fname : str or filehandle
        Stata .dta file.
    missing_flt : numeric
        The numeric value to replace missing values with. Will be used for
        any numeric value.
    encoding : string, optional
        Used for Python 3 only. Encoding to use when reading the .dta file.
        Defaults to `locale.getpreferredencoding`
    pandas : bool
        Optionally return a DataFrame instead of an ndarray
    convert_dates : bool
        If convert_dates is True, then Stata formatted dates will be converted
        to datetime types according to the variable's format.
    """
    if isinstance(fname, string_types):
        fhd = StataReader(open(fname, 'rb'), missing_values=False,
                          encoding=encoding)
    elif not hasattr(fname, 'read'):
        raise TypeError("The input should be a string or a filehandle. "\
                        "(got %s instead)" % type(fname))
    else:
        fhd = StataReader(fname, missing_values=False, encoding=encoding)
#    validate_names = np.lib._iotools.NameValidator(excludelist=excludelist,
#                                    deletechars=deletechars,
#                                    case_sensitive=case_sensitive)

    #TODO: This needs to handle the byteorder?
    header = fhd.file_headers()
    types = header['dtyplist']
    nobs = header['nobs']
    numvars = header['nvar']
    varnames = header['varlist']
    fmtlist = header['fmtlist']
    dataname = header['data_label']
    labels = header['vlblist'] # labels are thrown away unless DataArray
                               # type is used
    data = np.zeros((nobs,numvars))
    stata_dta = fhd.dataset()

    dt = np.dtype(lzip(varnames, types))
    data = np.zeros((nobs), dtype=dt) # init final array

    for rownum,line in enumerate(stata_dta):
        # doesn't handle missing value objects, just casts
        # None will only work without missing value object.
        if None in line:
            for i,val in enumerate(line):
                #NOTE: This will only be scalar types because missing strings
                # are empty not None in Stata
                if val is None:
                    line[i] = missing_flt
        data[rownum] = tuple(line)

    if pandas:
        from pandas import DataFrame
        data = DataFrame.from_records(data)
        if convert_dates:
            cols = np.where(lmap(lambda x : x in _date_formats, fmtlist))[0]
            for col in cols:
                i = col
                col = data.columns[col]
                data[col] = data[col].apply(_stata_elapsed_date_to_datetime,
                        args=(fmtlist[i],))
    elif convert_dates:
        #date_cols = np.where(map(lambda x : x in _date_formats,
        #                                                    fmtlist))[0]
        # make the dtype for the datetime types
        cols = np.where(lmap(lambda x : x in _date_formats, fmtlist))[0]
        dtype = data.dtype.descr
        dtype = [(dt[0], object) if i in cols else dt for i,dt in
                 enumerate(dtype)]
        data = data.astype(dtype) # have to copy
        for col in cols:
            def convert(x):
                return _stata_elapsed_date_to_datetime(x, fmtlist[col])
            data[data.dtype.names[col]] = lmap(convert,
                                              data[data.dtype.names[col]])
    return data

def savetxt(fname, X, names=None, fmt='%.18e', delimiter=' '):
    """
    Save an array to a text file.

    This is just a copy of numpy.savetxt patched to support structured arrays
    or a header of names.  Does not include py3 support now in savetxt.

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : array_like
        Data to be saved to a text file.
    names : list, optional
        If given names will be the column header in the text file.  If None and
        X is a structured or recarray then the names are taken from
        X.dtype.names.
    fmt : str or sequence of strs
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored.
    delimiter : str
        Character separating columns.

    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into a ``.npz`` compressed archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to preceed result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.

    Examples
    --------
    >>> savetxt('test.out', x, delimiter=',')   # x is an array
    >>> savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    """

    with get_file_obj(fname, 'w') as fh:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        if X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(X.dtype.descr)
        else:
            ncol = X.shape[1]

        # `fmt` can be a string with multiple insertion points or a list of formats.
        # E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        if isinstance(fmt, (list, tuple)):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = delimiter.join(fmt)
        elif isinstance(fmt, string_types):
            if fmt.count('%') == 1:
                fmt = [fmt, ]*ncol
                format = delimiter.join(fmt)
            elif fmt.count('%') != ncol:
                raise AttributeError('fmt has wrong number of %% formats.  %s'
                                     % fmt)
            else:
                format = fmt

        # handle names
        if names is None and X.dtype.names:
            names = X.dtype.names
        if names is not None:
            fh.write(delimiter.join(names) + '\n')

        for row in X:
            fh.write(format % tuple(row) + '\n')

if __name__ == "__main__":
    import os
    curdir = os.path.dirname(os.path.abspath(__file__))
    res1 = genfromdta(curdir+'/../../datasets/macrodata/macrodata.dta')
