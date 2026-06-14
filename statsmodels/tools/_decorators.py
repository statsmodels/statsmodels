"""Decorators and descriptors used for cached attributes."""

import warnings

from statsmodels.tools.sm_exceptions import CacheWriteWarning

__all__ = [
    "ResettableCache",
    "cache_readonly",
    "cache_writable",
    "cached_data",
    "cached_value",
    "deprecated_alias",
]


class ResettableCache(dict):
    """Dictionary cache retained for backward compatibility."""

    def __init__(self, *args, **kwargs):
        """Initialize the cache."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def deprecated_alias(
    old_name, new_name, remove_version=None, msg=None, warning=FutureWarning
):
    """
    Deprecate attribute in favor of alternative name.

    Parameters
    ----------
    old_name : str
        Old, deprecated name
    new_name : str
        New name
    remove_version : str, optional
        Version that the alias will be removed
    msg : str, optional
        Message to show.  Default is
        `old_name` is a deprecated alias for `new_name`
    warning : Warning, optional
        Warning class to give.  Default is FutureWarning.

    Notes
    -----
    Older or less-used classes may not conform to statsmodels naming
    conventions.  `deprecated_alias` lets us bring them into conformance
    without breaking backward-compatibility.

    Example
    -------
    Instances of the `Foo` class have a `nvars` attribute, but it _should_
    be called `neqs`:

    class Foo:
        nvars = deprecated_alias('nvars', 'neqs')
        def __init__(self, neqs):
            self.neqs = neqs

    >>> foo = Foo(3)
    >>> foo.nvars
    __main__:1: FutureWarning: nvars is a deprecated alias for neqs
    3

    """
    if msg is None:
        msg = f"{old_name} is a deprecated alias for {new_name}"
        if remove_version is not None:
            msg += ", will be removed in version %s" % remove_version

    def fget(self):
        warnings.warn(msg, warning, stacklevel=2)
        return getattr(self, new_name)

    def fset(self, value):
        warnings.warn(msg, warning, stacklevel=2)
        setattr(self, new_name, value)

    res = property(fget=fget, fset=fset)
    return res


class CachedAttribute:
    """Descriptor that caches a read-only attribute on first access."""

    def __init__(self, func, cachename=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or "_cache"

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
        warnings.warn(errmsg, CacheWriteWarning, stacklevel=2)


class CachedWritableAttribute(CachedAttribute):
    """Descriptor that caches an attribute and permits reassignment."""

    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        _cache[name] = value


class _cache_readonly(property):
    """Decorate a method as a CachedAttribute."""

    def __init__(self, cachename=None):
        self.func = None
        self.cachename = cachename

    def __call__(self, func):
        return CachedAttribute(func, cachename=self.cachename)


class cache_writable(_cache_readonly):
    """Decorate a method as a CachedWritableAttribute."""

    def __call__(self, func):
        """Return a writable cached descriptor for func."""
        return CachedWritableAttribute(func, cachename=self.cachename)


class cache_readonly:
    """
    Decorator for read-only, cached properties.

    Acts as a property that is computed once and then stored in the
    instance's ``_cache`` dictionary. This allows ``remove_data``
    to identify and clear cached values.

    Vendored to replace pandas._libs.properties.cache_readonly
    which is a private C extension.

    Parameters
    ----------
    func : callable
        The function to cache.

    Examples
    --------
    >>> class Foo:
    ...     @cache_readonly
    ...     def bar(self):
    ...         return 42
    >>> f = Foo()
    >>> f.bar
    42
    """

    def __init__(self, func):
        self.func = func
        self.fget = func
        self.__doc__ = getattr(func, "__doc__", None)
        self.__name__ = func.__name__
        self.__module__ = func.__module__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Use _cache dict — same as pandas implementation
        # This is required for remove_data() to work correctly
        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}
        val = cache.get(self.__name__, None)
        if val is None:
            val = self.func(obj)
            cache[self.__name__] = val
        return val

    def __set__(self, obj, values):
        raise AttributeError(f"The attribute '{self.__name__}' cannot be set.")


# cached_value and cached_data behave identically to cache_readonly, but
# are used by `remove_data` to
#   a) identify array-like attributes to remove (cached_data)
#   b) make sure certain values are evaluated before caching (cached_value)
cached_data = cache_readonly
cached_value = cache_readonly
