"""
Compatibility tools for differences between Python 2 and 3
"""
import sys

PY37 = (sys.version_info[:2] == (3, 7))

asunicode = lambda x, _: str(x)  # noqa:E731


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode('latin1')


def asstr(s):
    if isinstance(s, str):
        return s
    return s.decode('latin1')


# list-producing versions of the major Python iterating functions
def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))


def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs))


def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def lfilter(*args, **kwargs):
    return list(filter(*args, **kwargs))


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


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})
