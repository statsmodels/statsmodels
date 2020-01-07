
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from pandas.util._decorators import deprecate_kwarg, Appender, Substitution

__all__ = ['assert_frame_equal', 'assert_index_equal', 'assert_series_equal',
           'data_klasses', 'frequencies', 'is_numeric_dtype', 'testing',
           'cache_readonly', 'deprecate_kwarg', 'Appender', 'Substitution',
           'make_dataframe', 'assert_equal']

version = LooseVersion(pd.__version__)
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

data_klasses = (pd.Series, pd.DataFrame)
if pandas_lt_25_0:
    data_klasses += (pd.Panel,)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal

try:
    from pandas.testing import assert_equal
except ImportError:
    def assert_equal(left, right):
        """
        pandas >= 0.24.0 has `pandas.testing.assert_equal` that works for any
        of Index, Series, and DataFrame inputs.  Until statsmodels requirements
        catch up to that, we implement a version of that here.

        Parameters
        ----------
        left : pd.Index, pd.Series, or pd.DataFrame
        right : object

        Raises
        ------
        AssertionError
        """
        if isinstance(left, pd.Index):
            assert_index_equal(left, right)
        elif isinstance(left, pd.Series):
            assert_series_equal(left, right)
        elif isinstance(left, pd.DataFrame):
            assert_frame_equal(left, right)
        else:
            raise TypeError(type(left))

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

try:
    from pandas._testing import makeDataFrame as make_dataframe
except ImportError:
    import string

    def rands_array(nchars, size, dtype="O"):
        """
        Generate an array of byte strings.
        """
        rands_chars = np.array(list(string.ascii_letters + string.digits),
                               dtype=(np.str_, 1))
        retval = (np.random.choice(rands_chars, size=nchars * np.prod(size))
                  .view((np.str_, nchars))
                  .reshape(size))
        if dtype is None:
            return retval
        else:
            return retval.astype(dtype)

    def make_dataframe():
        """
        Simple verion of pandas._testing.makeDataFrame
        """
        n = 30
        k = 4
        index = pd.Index(rands_array(nchars=10, size=n), name=None)
        data = {c: pd.Series(np.random.randn(n), index=index)
                for c in string.ascii_uppercase[:k]}

        return pd.DataFrame(data)
