"""
Compatibility tools for differences between Python 2 and 3
"""
import functools
import itertools
import sys
import urllib

PY3 = (sys.version_info[0] >= 3)
PY3_2 = sys.version_info[:2] == (3, 2)

if PY3:
    import builtins
    from collections import namedtuple
    from io import StringIO, BytesIO
    import inspect

    cStringIO = StringIO
    import pickle as cPickle
    pickle = cPickle
    import urllib.request
    import urllib.parse
    from urllib.request import HTTPError, urlretrieve, URLError
    import io
    bytes = bytes
    str = str
    asunicode = lambda x, _ : str(x)

    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')

    def asstr(s):
        if isinstance(s, str):
            return s
        return s.decode('latin1')

    def asstr2(s):  #added JP, not in numpy version
        if isinstance(s, str):
            return s
        elif isinstance(s, bytes):
            return s.decode('latin1')
        else:
            return str(s)

    def isfileobj(f):
        return isinstance(f, io.FileIO)

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode, encoding='iso-8859-1')

    strchar = 'U'

    # have to explicitly put builtins into the namespace
    range = range
    map = map
    zip = zip
    filter = filter
    reduce = functools.reduce
    long = int
    unichr = chr
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

    ArgSpec= namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
    def getargspec(func):
        """
        Simple workaroung for getargspec deprecation that returns
        an ArgSpec-like object
        """
        sig = inspect.signature(func)
        parameters = sig.parameters
        args, defaults  = [], []
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
    from cStringIO import StringIO as cStringIO
    # always writeable
    from StringIO import StringIO
    from inspect import getargspec

    BytesIO = StringIO
    import cPickle
    pickle = cPickle
    import urllib2
    import urlparse

    bytes = str
    str = str
    asbytes = str
    asstr = str
    asstr2 = str
    strchar = 'S'

    def isfileobj(f):
        return isinstance(f, file)

    def asunicode(s, encoding='ascii'):
        if isinstance(s, unicode):
            return s
        return s.decode(encoding)

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode)

    # import iterator versions of these functions
    range = xrange
    zip = itertools.izip
    filter = itertools.ifilter
    map = itertools.imap
    reduce = reduce
    long = long
    unichr = unichr
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
    string_types = basestring

    input = raw_input


def getexception():
    return sys.exc_info()[1]


def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, str)):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)


def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, str)):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)


try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()
next = advance_iterator


try:
    callable = callable
except NameError:
    def callable(obj):
        return any("__call__" in klass.__dict__ for klass in type(obj).__mro__)

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
        #Python 3
        return func.__name__

def get_class(func):
    try:
        return func.im_class
    except AttributeError:
        #Python 3
        return func.__self__.__class__

try:
    combinations = itertools.combinations
except:
    # Python 2.6 only
    def combinations(iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(lrange(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = lrange(r)
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(lrange(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)


