from statsmodels.tools._testing import PytestTester

from .python import (
    PY3, PY37,
    bytes, str, unicode, string_types,
    asunicode, asbytes, asstr, asstr2,
    range, zip, filter, map,
    lrange, lzip, lmap, lfilter,
    cStringIO, StringIO, BytesIO,
    cPickle, pickle,
    iteritems, iterkeys, itervalues,
    urlopen, urljoin, urlencode, HTTPError, URLError,
    reduce, long, unichr, zip_longest,
    builtins,
    getargspec,
    next, get_class
)

__all__ = ['PY3', 'PY37', 'bytes', 'str', 'unicode', 'string_types',
           'asunicode', 'asbytes', 'asstr', 'asstr2', 'range', 'zip',
           'filter', 'map', 'lrange', 'lzip', 'lmap', 'lfilter', 'cStringIO',
           'StringIO', 'BytesIO', 'cPickle', 'pickle', 'iteritems',
           'iterkeys', 'itervalues', 'urlopen', 'urljoin', 'urlencode',
           'HTTPError', 'URLError', 'reduce', 'long', 'unichr', 'zip_longest',
           'builtins', 'getargspec', 'next', 'get_class', 'test']

test = PytestTester()
