"""
Input/Output tools for working with binary data.

The Stata input tools were originally written by Joe Presbrey as part of PyDTA.

You can find more information here http://presbrey.mit.edu/PyDTA

See also
---------
numpy.lib.io
"""

from struct import unpack, calcsize
import sys
import numpy as np


### Helper classes for Stata .dta files ###

class _StataMissingValue(object):
    """
    An observation's missing value.

    More information: <http://www.stata.com/help.cgi?missing>
    """

    def __init__(self, offset, value):
        self._value = value
        if type(value) is int or type(value) is long:
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
    A dataset variable.
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

#TODO: StataReader needs a seek method?
class StataReader(object):
    """
    Stata .dta file reader

    Parameters
    ----------
    file : Stata .dta file

    missing_values : bool
        If missing_values is True, parse missing_values and return a
        Missing Values object instead of None.

    Returns
    -------
    File-like object of .dta binary data file.

    Attributes
    ----------
    file_headers
    file_format
    file_label
    file_timestamp
    variables
    dataset


    See also
    --------
    scikits.statsmodels.lib.io.genfromdta

    Notes
    -----
    This is known only to work on 113 (untested) and 114.
    Needs to be tested on older versions.  Known not to work on format 104, 108.

    For more information about the .dta format see
    http://www.stata.com/help.cgi?dta
    http://www.stata.com/help.cgi?dta_113
    """

    _header = {}
    _data_location = 0
    _col_sizes = ()
    _has_string_data = False
    _missing_values = False
    TYPE_MAP = range(251)+list('bhlfd')
    MISSING_VALUES = { 'b': (-127,100), 'h': (-32767, 32740), 'l':
            (-2147483647, 2147483620), 'f': (-1.701e+38, +1.701e+38), 'd':
            (-1.798e+308, +8.988e+307) }

    def __init__(self, fname, missing_values=False):
        """
        Creates a new parser from a file object.

        If missing_values, parse missing values and return as a MissingValue
        object (instead of None).
        """
        self._missing_values = missing_values
        self._parse_header(fname)

    def file_headers(self):
        """
        Returns all .dta file headers.
        """
        return self._header

    def file_format(self):
        """
        Returns the file format.

        Format 113: Stata 9
        Format 114: Stata 10
        """
        return self._header['ds_format']

    def file_label(self):
        """
        Returns the dataset's label.
        """
        return self._header['data_label']

    def file_timestamp(self):
        """
        Returns the date and time Stata recorded on last file save.
        """
        return self._header['time_stamp']

    def variables(self):
        """
        Returns a list of the dataset's PyDTA.Variables.
        """
        return map(_StataVariable, zip(range(self._header['nvar']),
            self._header['typlist'], self._header['varlist'],
            self._header['srtlist'],
            self._header['fmtlist'], self._header['lbllist'],
            self._header['vlblist']))

    def dataset(self, as_dict=False):
        """
        Returns a Python generator object for iterating over the dataset.

        Each observation is returned as a list unless as_dict is set.
        Observations with a MissingValue(s) are not filtered and should be
        handled by your applcation.
        """
        try:
            self._file.seek(self._data_location)
        except Exception:
            pass

        if as_dict:
            vars = map(str, self.variables())
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
        if not (type(k) is int or type(k) is long) or k < 0 or k > len(self)-1:
            raise IndexError(k)
        loc = self._data_location + sum(self._col_size()) * k
        if self._file.tell() != loc:
            self._file.seek(loc)
        return self._next()

    ### Private methods

    def _null_terminate(self, s):
        try:
            return s.lstrip('\x00')[:s.index('\x00')]
        except Exception:
            return s

    def _parse_header(self, file_object):
        self._file = file_object

        # parse headers
        self._header['ds_format'] = unpack('b', self._file.read(1))[0]

        if self._header['ds_format'] not in [113,114]:
            raise ValueError, "Only file formats 113 and 114 (Stata 9, 10, 11)\
 are supported.  Got format %s.  Please report if you think this error is \
incorrect." % self._header['dsformat']
        byteorder = self._header['byteorder'] = unpack('b',
                self._file.read(1))[0]==0x1 and '>' or '<'
        self._header['filetype'] = unpack('b', self._file.read(1))[0]
        self._file.read(1)
        nvar = self._header['nvar'] = unpack(byteorder+'h',
                self._file.read(2))[0]
        if self._header['ds_format'] < 114:
            self._header['nobs'] = unpack(byteorder+'i', self._file.read(4))[0]
        else:
            self._header['nobs'] = unpack(byteorder+'i', self._file.read(4))[0]
        self._header['data_label'] = self._null_terminate(self._file.read(81))
        self._header['time_stamp'] = self._null_terminate(self._file.read(18))

        # parse descriptors
        self._header['typlist'] = [self.TYPE_MAP[ord(self._file.read(1))] \
                for i in range(nvar)]
        self._header['varlist'] = [self._null_terminate(self._file.read(33)) \
                for i in range(nvar)]
        self._header['srtlist'] = unpack(byteorder+('h'*(nvar+1)),
                self._file.read(2*(nvar+1)))[:-1]
        if self._header['ds_format'] <= 113:
            self._header['fmtlist'] = \
                    [self._null_terminate(self._file.read(12)) \
                    for i in range(nvar)]
        else:
            self._header['fmtlist'] = \
                    [self._null_terminate(self._file.read(49)) \
                    for i in range(nvar)]
        self._header['lbllist'] = [self._null_terminate(self._file.read(33)) \
                for i in range(nvar)]
        self._header['vlblist'] = [self._null_terminate(self._file.read(81)) \
                for i in range(nvar)]

        # ignore expansion fields
#    When
#    reading, read five bytes; the last four bytes now tell you the size of
#    the next read, which you discard.  You then continue like this until you
#    read 5 bytes of zeros.
# the way I read this is that they both should be zero, but that's not what we
# get.  And you can't just keep reading until both are zero because the 2nd
# iteration gives a big length.  Maybe there is an error in the above and
# we aren't where we think we are in the file?
# The above pertains to an unsupported file format (104?)

        while True:
            data_type = unpack(byteorder+'b', self._file.read(1))[0]
            data_len = unpack(byteorder+'i', self._file.read(4))[0]
            if data_type == 0:
                break
            self._file.read(data_len)

        # other state vars
        self._data_location = self._file.tell()
        self._has_string_data = len(filter(lambda x: type(x) is int,
            self._header['typlist'])) > 0
        self._col_size()

    def _calcsize(self, fmt):
        return type(fmt) is int and fmt or \
                calcsize(self._header['byteorder']+fmt)

    def _col_size(self, k = None):
        """Calculate size of a data record."""
        if len(self._col_sizes) == 0:
            self._col_sizes = map(lambda x: self._calcsize(x),
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
                if type(typlist[i]) is int:
                    data[i] = self._null_terminate(self._file.read(typlist[i]))
                else:
                    data[i] = self._unpack(typlist[i],
                            self._file.read(self._col_size(i)))
            return data
        else:
            return map(lambda i: self._unpack(typlist[i],
                self._file.read(self._col_size(i))),
                range(self._header['nvar']))

#TODO: extend to get data from online
def genfromdta(fname, missing_values=False, excludelist=None, missing_flt=-999., missing_str=""):
    """
    Returns an ndarray from a Stata .dta file.

    Parameters
    ----------
    fname
    missing_values
    excludelist
    missing_flt
    missing_str

    """
#TODO: not sure if we care about missing values, if it returns
# none when missing_values is false then that's fine for our purposes
    if isinstance(fname, basestring):
        fhd = StataReader(open(fname, 'r'), missing_values=missing_values)
    elif not hasattr(fname, 'read'):
        raise TypeError("The input should be a string or a filehandle. "\
                "(got %s instead)" % type(fname))
    else:
        fhd = StataReader(fname, missing_values)
    # validate_names = np.lib._iotools.NameValidator(excludelist=excludelist,
#                                    deletechars=deletechars,
#                                    case_sensitive=case_sensitive)


#TODO: does this need to handle the byteorder?
    header = fhd.file_headers()
#    types = header['typlist'] # maybe change this in StataReader
    nobs = header['nobs']
    numvars = header['nvar']
    varnames = header['varlist']
    dataname = header['data_label']
    labels = header['vlblist'] # labels are thrown away unless my DataArray
                               # type is used
    data = np.zeros((nobs,numvars))
    stata_dta = fhd.dataset()

    # build dtype from stata formats
    # see http://www.stata.com/help.cgi?format
    # This converts all of these to float64
    # all time and strings are converted to strings
    #TODO: need to write a time parser
    to_flt = ['g','e','f','h','gc','fc', 'x', 'l'] # how to deal with x
                                                   # and double-precision
    to_str = ['s']
    if 1:#    if not convert_time: #time parser not written
        to_str.append('t')
    flt_or_str = lambda x: (x.lower()[-1] in to_str and 's') or \
            (x.lower()[-1] in to_flt and 'f8')
        #TODO: this is surely not the best way to handle data types
    convert_missing = {'f8' : missing_flt, 's' : missing_str}
    #TODO: needs to be made more flexible when change types
    fmt = [_.split('.')[-1] for _ in header['fmtlist']]
    remove_comma = [fmt.index(_) for _ in fmt if 'c' in _]
    for i in range(len(fmt)): # remove commas and convert any time types to 't'
        if 't' in fmt[i]:
            fmt[i] = 't'
        if i in remove_comma:
            fmt[i] = fmt[i][:-1] # needs to be changed if time doesn't req.
                                 # loop
    dt = zip(varnames, map(flt_or_str, fmt)) # make dtype
    data = np.zeros((nobs), dtype=dt)
    for rownum,line in enumerate(stata_dta):
        # doesn't handle missing value objects
        # Untested for commas and string missing
        # None will only work without missing value object.
        if None in line and not remove_comma:
            for val in line:
                if val is None:
                    line[line.index(val)] = convert_missing[\
                            dt[line.index(val)][1]]
        if None in line and remove_comma:
            for i,val in enumerate(line):
                if val is None:
                    line[i] = convert_missing[dt[i][1]]
                elif i in remove_comma:
                    line[i] = ''.join(line[i].split(','))
                    if dt[i][1] == 'f8':
                        line[i] = float(line[i])
        if remove_comma and not None in line:
            for j in remove_comma:
                line[j] = ''.join(line[j].split(','))
                if dt[j][1] == 'f8': # change when change f8
                    line[j] = float(line[j])

        data[rownum] = tuple(line) # tuples the only solution?
#TODO: add informative error message similar to genfromtxt
#TODO: make it possible to return plain array if all 'f8' for example
    return data

if __name__=="__main__":
    try:
        data = genfromdta('./fullauto.dta')
    except:
        raise ImportError, "You don't have the Stata test file downloaded into\
 this directory.  It's not distributed but you can download it here \
http://www.stata-press.com/data/r11/fullauto.dta."
