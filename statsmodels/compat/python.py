"""
Compatibility tools for differences between Python 2 and 3
"""
import functools
import itertools
import sys
import urllib

PY3 = (sys.version_info[0] >= 3)
PY37 = (sys.version_info[:2] == (3, 7))

if PY3:
    from collections import namedtuple
    from io import StringIO, BytesIO
    import inspect

    cStringIO = StringIO
    import pickle as cPickle
    import urllib.request
    import urllib.parse
    from urllib.request import HTTPError, URLError
    bytes = bytes
    str = str
    unicode = str

    def asstr(s):
        if isinstance(s, str):
            return s
        return s.decode('latin1')

    def asstr2(s):  # added JP, not in numpy version
        if isinstance(s, str):
            return s
        elif isinstance(s, bytes):
            return s.decode('latin1')
        else:
            return str(s)

    # have to explicitly put builtins into the namespace
    range = range
    map = map
    zip = zip
    filter = filter
    reduce = functools.reduce
    long = int
    zip_longest = itertools.zip_longest

    # list-producing versions of the major Python iterating functions
    def lrange(*args, **kwargs):
        return list(range(*args, **kwargs))

    def lzip(*args, **kwargs):
        return list(zip(*args, **kwargs))

    def lmap(*args, **kwargs):
        return list(map(*args, **kwargs))

    def lfilter(*args, **kwargs):
        return list(filter(*args, **kwargs))

    urlopen = urllib.request.urlopen
    urljoin = urllib.parse.urljoin
    urlretrieve = urllib.request.urlretrieve
    urlencode = urllib.parse.urlencode
    string_types = str
    input = input

    ArgSpec = namedtuple('ArgSpec',
                         ['args', 'varargs', 'keywords', 'defaults'])

    def getargspec(func):
        """
        Simple workaroung for getargspec deprecation that returns
        an ArgSpec-like object
        """
        sig = inspect.signature(func)
        parameters = sig.parameters
        args, defaults = [], []
        varargs, keywords = None, None

        for key in parameters:
            parameter = parameters[key]

            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                varargs = key
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                keywords = key
            else:
                args.append(key)
            if parameter.default is not parameter.empty:
                defaults.append(parameter.default)
        defaults = None if len(defaults) == 0 else defaults

        return ArgSpec(args, varargs, keywords, defaults)

else:
    import __builtin__ as builtins
    # not writeable when instantiated with string, doesn't handle unicode well
    from cStringIO import StringIO as cStringIO  # noqa:F401
    # always writeable
    from StringIO import StringIO
    from inspect import getargspec  # noqa:F401

    BytesIO = StringIO
    import cPickle  # noqa:F401
    import urllib2
    import urlparse

    bytes = str
    str = str
    unicode = unicode
    asstr = str
    asstr2 = str

    # import iterator versions of these functions
    range = xrange  # noqa:F401
    zip = itertools.izip
    filter = itertools.ifilter
    map = itertools.imap
    reduce = reduce
    long = long
    zip_longest = itertools.izip_longest

    # Python 2-builtin ranges produce lists
    lrange = builtins.range
    lzip = builtins.zip
    lmap = builtins.map
    lfilter = builtins.filter

    urlopen = urllib2.urlopen
    urljoin = urlparse.urljoin
    urlencode = urllib.urlencode
    HTTPError = urllib2.HTTPError
    URLError = urllib2.URLError
    string_types = basestring  # noqa:F401

    input = raw_input  # noqa:F401


try:
    next = next
except NameError:
    def next(it):
        return it.next()


def iteritems(obj, **kwargs):
    """replacement for six's iteritems for Python2/3 compat
       uses 'iteritems' if available and otherwise uses 'items'.

       Passes kwargs to method.
    """
    func = getattr(obj, "iteritems", None)
    if not func:
        func = obj.items
    return func(**kwargs)


def iterkeys(obj, **kwargs):
    func = getattr(obj, "iterkeys", None)
    if not func:
        func = obj.keys
    return func(**kwargs)


def itervalues(obj, **kwargs):
    func = getattr(obj, "itervalues", None)
    if not func:
        func = obj.values
    return func(**kwargs)


def get_function_name(func):
    try:
        return func.im_func.func_name
    except AttributeError:
        # Python 3
        return func.__name__
