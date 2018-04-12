from __future__ import print_function
from statsmodels.tools.sm_exceptions import CacheWriteWarning
import warnings

__all__ = ['resettable_cache', 'cache_readonly', 'cache_writable']


class ResettableCache(dict):
    """
    Dictionary whose elements mey depend one from another.

    If entry `B` depends on entry `A`, changing the values of entry `A` will
    reset the value of entry `B` to a default (None); deleteing entry `A` will
    delete entry `B`.  The connections between entries are stored in a
    `_resetdict` private attribute.

    Parameters
    ----------
    reset : dictionary, optional
        An optional dictionary, associated a sequence of entries to any key
        of the object.
    items : var, optional
        An optional dictionary used to initialize the dictionary

    Examples
    --------
    >>> reset = dict(a=('b',), b=('c',))
    >>> cache = resettable_cache(a=0, b=1, c=2, reset=reset)
    >>> assert_equal(cache, dict(a=0, b=1, c=2))

    >>> print("Try resetting a")
    >>> cache['a'] = 1
    >>> assert_equal(cache, dict(a=1, b=None, c=None))
    >>> cache['c'] = 2
    >>> assert_equal(cache, dict(a=1, b=None, c=2))
    >>> cache['b'] = 0
    >>> assert_equal(cache, dict(a=1, b=0, c=None))

    >>> print("Try deleting b")
    >>> del(cache['a'])
    >>> assert_equal(cache, {})
    """

    def __init__(self, reset=None, **items):
        dict.__init__(self, **items)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)


resettable_cache = ResettableCache


class CachedAttribute(object):

    def __init__(self, func, cachename=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'

    def __get__(self, obj, type=None):
        if obj is None:
            return self.fget
        # Get the cache or set a default one if needed
        _cachename = self.cachename
        _cache = getattr(obj, _cachename, None)
        if _cache is None:
            setattr(obj, _cachename, resettable_cache())
            _cache = getattr(obj, _cachename)
        # Get the name of the attribute to set and cache
        name = self.name
        _cachedval = _cache.get(name, None)
        # print("[_cachedval=%s]" % _cachedval)
        if _cachedval is None:
            # Call the "fget" function
            _cachedval = self.fget(obj)
            # Set the attribute in obj
            # print("Setting %s in cache to %s" % (name, _cachedval))
            try:
                _cache[name] = _cachedval
            except KeyError:
                setattr(_cache, name, _cachedval)
        # else:
        # print("Reading %s from cache (%s)" % (name, _cachedval))
        return _cachedval

    def __set__(self, obj, value):
        errmsg = "The attribute '%s' cannot be overwritten" % self.name
        warnings.warn(errmsg, CacheWriteWarning)


class CachedWritableAttribute(CachedAttribute):
    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        try:
            _cache[name] = value
        except KeyError:
            setattr(_cache, name, value)


class _cache_readonly(object):
    """
    Decorator for CachedAttribute
    """

    def __init__(self, cachename=None):
        self.func = None
        self.cachename = cachename

    def __call__(self, func):
        return CachedAttribute(func,
                               cachename=self.cachename)


cache_readonly = _cache_readonly()


class cache_writable(_cache_readonly):
    """
    Decorator for CachedWritableAttribute
    """
    def __call__(self, func):
        return CachedWritableAttribute(func,
                                       cachename=self.cachename)


def nottest(fn):
    fn.__test__ = False
    return fn
