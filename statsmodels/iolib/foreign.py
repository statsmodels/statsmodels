"""
Input/Output tools for working with binary data.

The Stata input tools were originally written by Joe Presbrey as part of PyDTA.

You can find more information here http://presbrey.mit.edu/PyDTA

See also
---------
numpy.lib.io
"""

from statsmodels.compat.python import zip, lzip, lmap, string_types

import datetime

import numpy as np
from pandas.io.stata import StataReader, StataWriter

from statsmodels.iolib.openfile import get_file_obj

_date_formats = ["%tc", "%tC", "%td", "%tw", "%tm", "%tq", "%th", "%ty"]


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
    labels = header['vlblist']  # labels are thrown away unless DataArray
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
