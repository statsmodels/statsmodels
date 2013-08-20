from struct import unpack, unpack_from, calcsize
import sys
from statsmodels.compatnp.py3k import asbytes

import numpy as np
cimport numpy as np

def is_py3():
    import sys
    if sys.version_info[0] == 3:
        return True
    return False
PY3 = is_py3()
        
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


### StataReader ###

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
    statsmodels.lib.io.genfromdta

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
    range1 = zip(range(1,245), ['a' + str(i) for i in range(1,245)])
    range2 = [(251, np.int16),(252, np.int32),(253, int),
                        (254, np.float32), (255, np.float64)]

    DTYPE_MAP = dict(list(range1) + range2)
    TYPE_MAP = range(251)+list('bhlfd')
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
        return map(_StataVariable, zip(range(self._header['nvar']),
            self._header['typlist'], self._header['varlist'],
            self._header['srtlist'],
            self._header['fmtlist'], self._header['lbllist'],
            self._header['vlblist']))

    ### Python special methods

    def __len__(self):
        """
        Return the number of observations in the dataset.

        This value is taken directly from the header and includes observations
        with missing values.
        """
        return self._header['nobs']
    
    # have to modify to read an individual value now _next has been deleted 
    """
    def __getitem__(self, k):
        \"""
        Seek to an observation indexed k in the file and return it, ordered
        by Stata's output to the .dta file.

        k is zero-indexed.  Prefer using R.data() for performance.
        \"""
        if not (type(k) is int or type(k) is long) or k < 0 or k > len(self)-1:
            raise IndexError(k)
        loc = self._data_location + sum(self._col_size()) * k
        if self._file.tell() != loc:
            self._file.seek(loc)
        return self._next()
    
    """

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
            
    def _unpack_from(self, fmt, byt, offset, missing_flt):
        typlist = self._header['typlist']
        d = map(None, unpack_from(self._header['byteorder']+fmt, byt, offset))
        d = [self._null_terminate(d[i], self._encoding) \
                if type(typlist[i]) is int \
                else self._missing_values_test(d[i], typlist[i]) \
                for i in xrange(len(d))]
                
        if None in d:
            d = map(lambda x: missing_flt if x is None else x, d)
        
        return tuple(d)

    def _missing_values_test(self, num, typ):
        nmin, nmax = self.MISSING_VALUES[typ]
        if num < nmin or num > nmax:
            if self._missing_values:
                return _StataMissingValue(nmax, num)
            else:
                return None
        return num
        
def reader_inner_loop(file_obj, size, missing_flt):
    """
    Returns an ndarray of observations from a Stata .dta file.
    
    Parameters
    ----------
    file_obj : file-like
        file handle of Stata .dta file
    size : numeric
        upper limit on data chunk sizes in bytes
    missing_flt : numeric
        The numeric value to replace missing values with. Will be used for
        any numeric value.
    """
    
    try:
        file_obj._file.seek(file_obj._data_location)
    except Exception:
        pass
    
    header = file_obj.file_headers()
    types = header['dtyplist']
    typlist = header['typlist']
    nobs = header['nobs']
    nvar = header['nvar']
    varnames = header['varlist']
    fmtlist = header['fmtlist']
    dataname = header['data_label']
    labels = header['vlblist'] # labels are thrown away unless DataArray
                               # type is used
    target = np.zeros((nobs,nvar))
    dt = np.dtype(zip(varnames, types))
    
    target = np.zeros((nobs), dtype=dt) # init final array
    t_index = 0
    
    fmt = ''.join(map(lambda x: str(x)+'s' if type(x) is int else x, typlist))
    record_size = sum(file_obj._col_sizes)
    chunk_info = parse_chunk_sizes(size, record_size, nobs)
    
    for chunk in getchunks(file_obj, chunk_info):
        t_index = processchunk(file_obj, chunk, record_size, 
                                fmt, missing_flt, target, t_index)
                                
    return target

def parse_chunk_sizes(size, record_size, nobs):
    """
    Returns the following tuple:
        0 : number of chunks to parse out of Stata .dta file
        1 : chunk size
        2 : leftover chunk size
        
    
    size : numeric
        The numeric value that is the upper limit on the chunk size (in bytes)
    record_size : numeric
        Size of a complete record stored in the file (in bytes).
    nobs : numeric
        How many records are stored in the file
    """
    
    numrecs = size / record_size # max number of records per size bytes
    if numrecs > nobs: # if the file is smaller than size
        numrecs = nobs # read in entire file
    
    chunksize = numrecs * record_size # size in bytes of each chunk
    
    numrecs_lftovr, chunksize_lftovr = 0, 0
    if nobs % numrecs != 0: # if there are leftover records
        numrecs_lftovr = nobs % numrecs
        chunksize_lftovr = numrecs_lftovr * record_size # size of leftover
                                                        # chunk
    numchunks = nobs / numrecs # number of chunks
    
    return numchunks, chunksize, chunksize_lftovr

def getchunks(file_obj, chunk_info):
    """
    Yields chunks of data from the Stata .dta file for processing.
    
    file_obj : file-like
        file to read in data from
    chunk_info : array-like
        0 : number of chunks to parse
        1 : chunk size
        2 : leftover chunk size
    """
    
    for i in xrange(chunk_info[0]):
        yield file_obj._file.read(chunk_info[1])
    
    if chunk_info[2] != 0:
        yield file_obj._file.read(chunk_info[2])

def processchunk(file_obj, chunk, record_size, fmt, 
                        missing_flt, target, t_index):
    """
    Unpacks each records from the a chunk. Appends parsed record to target.
    
    file_obj : file-like
        Stata .dta where data is from.
    chunk : array-like
        Array-like object where unparsed data is stored
    record_size : numeric
        The size of each read from chunk.
    fmt : string
        Format of each record
    missing_flt : numeric
        The numeric value to replace missing values with. Will be used for
        any numeric value.
    target : array-like
        Target array where parsed records will be stored
    t_index : numeric
        target index to keep track of where to insert next record
    """
    numofrecs = len(chunk)/record_size # number of records in chunk
    for i in xrange(numofrecs):
        target[t_index] = file_obj._unpack_from(fmt, chunk, 
                                                i*record_size, missing_flt)
        t_index+=1
    return t_index
