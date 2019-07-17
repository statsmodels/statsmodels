"""
Compatibility tools for differences between Python 2 and 3
"""
__all__ = ['HTTPError', 'URLError', 'BytesIO']

import functools
import itertools
import sys
import urllib
import inspect
from collections import namedtuple
from io import StringIO, BytesIO
import pickle as cPickle
import urllib.request
import urllib.parse
from urllib.error import HTTPError, URLError

PY37 = (sys.version_info[:2] == (3, 7))

cStringIO = StringIO
pickle = cPickle
bytes = bytes
str = str
unicode = str
asunicode = lambda x, _: str(x)  # noqa:E731


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode('latin1')


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


def get_class(func):
    try:
        return func.im_class
    except AttributeError:
        # Python 3
        return func.__self__.__class__


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})
