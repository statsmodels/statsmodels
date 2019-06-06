from __future__ import print_function
from statsmodels.tools.sm_exceptions import CacheWriteWarning
import warnings

__all__ = ['cache_readonly', 'cache_writable']


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
            setattr(obj, _cachename, {})
            _cache = getattr(obj, _cachename)
        # Get the name of the attribute to set and cache
        name = self.name
        _cachedval = _cache.get(name, None)
        if _cachedval is None:
            _cachedval = self.fget(obj)
            _cache[name] = _cachedval

        return _cachedval

    def __set__(self, obj, value):
        errmsg = "The attribute '%s' cannot be overwritten" % self.name
        warnings.warn(errmsg, CacheWriteWarning)


class CachedWritableAttribute(CachedAttribute):
    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        _cache[name] = value


class _cache_readonly(property):
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
