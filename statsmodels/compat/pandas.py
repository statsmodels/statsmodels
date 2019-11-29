
from distutils.version import LooseVersion

import pandas
from pandas.util._decorators import deprecate_kwarg, Appender, Substitution

__all__ = ['assert_frame_equal', 'assert_index_equal', 'assert_series_equal',
           'data_klasses', 'frequencies', 'is_numeric_dtype', 'testing',
           'cache_readonly', 'deprecate_kwarg', 'Appender', 'Substitution']

version = LooseVersion(pandas.__version__)
pandas_lt_25_0 = version < LooseVersion('0.25.0')
pandas_gte_23_0 = version >= LooseVersion('0.23.0')

try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype

try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies

data_klasses = (pandas.Series, pandas.DataFrame)
if pandas_lt_25_0:
    data_klasses += (pandas.Panel,)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal


if pandas_gte_23_0:
    from pandas.util._decorators import cache_readonly
else:
    class CachedProperty(object):

        def __init__(self, func):
            self.func = func
            self.name = func.__name__
            self.__doc__ = getattr(func, '__doc__', None)

        def __get__(self, obj, typ):
            if obj is None:
                # accessed on the class, not the instance
                return self

            # Get the cache or set a default one if needed
            cache = getattr(obj, '_cache', None)
            if cache is None:
                try:
                    cache = obj._cache = {}
                except (AttributeError):
                    return self

            if self.name in cache:
                # not necessary to Py_INCREF
                val = cache[self.name]
            else:
                val = self.func(obj)
                cache[self.name] = val
            return val

        def __set__(self, obj, value):
            raise AttributeError("Can't set attribute")

    cache_readonly = CachedProperty
