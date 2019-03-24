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
        self._resetdict = reset or {}
        dict.__init__(self, **items)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        # if hasattr needed for unpickling with protocol=2
        if hasattr(self, '_resetdict'):
            for mustreset in self._resetdict.get(key, []):
                self[mustreset] = None

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        for mustreset in self._resetdict.get(key, []):
            del(self[mustreset])


resettable_cache = ResettableCache


class CachedAttribute(object):

    def __init__(self, func, cachename=None, resetlist=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'
        self.resetlist = resetlist or ()

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
            # Update the reset list if needed (and possible)
            resetlist = self.resetlist
            if resetlist is not ():
                try:
                    _cache._resetdict[name] = self.resetlist
                except AttributeError:
                    pass
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

    def __init__(self, cachename=None, resetlist=None):
        self.func = None
        self.cachename = cachename
        self.resetlist = resetlist or None

    def __call__(self, func):
        return CachedAttribute(func,
                               cachename=self.cachename,
                               resetlist=self.resetlist)


cache_readonly = _cache_readonly()


class cache_writable(_cache_readonly):
    """
    Decorator for CachedWritableAttribute
    """
    def __call__(self, func):
        return CachedWritableAttribute(func,
                                       cachename=self.cachename,
                                       resetlist=self.resetlist)


# this has been copied from nitime a long time ago
# TODO: ceck whether class has change in nitime
class OneTimeProperty(object):
    """
    A descriptor to make special properties that become normal attributes.

    This is meant to be used mostly by the auto_attr decorator in this module.
    Author: Fernando Perez, copied from nitime
    """
    def __init__(self, func):

        """Create a OneTimeProperty instance.

         Parameters
         ----------
           func : method

             The method that will be called the first time to compute a value.
             Afterwards, the method's name will be a standard attribute holding
             the value of this computation.
             """
        self.getter = func
        self.name = func.__name__

    def __get__(self, obj, type=None):
        """
        This will be called on attribute access on the class or instance.
        """

        if obj is None:
            # Being called on the class, return the original function.
            # This way, introspection works on the class.
            # return func
            # print('class access')
            return self.getter

        val = self.getter(obj)
        # print("** auto_attr - loading '%s'" % self.name  # dbg)
        setattr(obj, self.name, val)
        return val


def nottest(fn):
    fn.__test__ = False
    return fn
