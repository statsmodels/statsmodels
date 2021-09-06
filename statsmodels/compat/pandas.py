from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
    deprecate_kwarg,
)

__all__ = [
    "assert_frame_equal",
    "assert_index_equal",
    "assert_series_equal",
    "data_klasses",
    "frequencies",
    "is_numeric_dtype",
    "testing",
    "cache_readonly",
    "deprecate_kwarg",
    "Appender",
    "Substitution",
    "NumericIndex",
    "is_int_index",
    "make_dataframe",
    "to_numpy",
    "pandas_lt_1_0_0",
]

version = LooseVersion(pd.__version__)

pandas_lt_1_0_0 = version < LooseVersion("1.0.0")

try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype

try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies

data_klasses = (pd.Series, pd.DataFrame)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal

try:
    from pandas import NumericIndex

    has_numeric_index = True
except ImportError:
    from pandas import Int64Index as NumericIndex

    has_numeric_index = False


def is_int_index(index: pd.Index) -> bool:
    """
    Check if an index is integral

    Parameters
    ----------
    index : pd.NumericIndex
        Any numeric index

    Returns
    -------
    bool
        True if Int64Index, UInt64Index or NumericIndex with integral dtype
    """
    if type(index) is NumericIndex and np.issubdtype(index.dtype, np.integer):
        return True
    # Safe legacy path
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from pandas import Int64Index, UInt64Index

            if type(index) in (Int64Index, UInt64Index):
                return True
    except ImportError:
        pass
    return False


def is_float_index(index):
    """
    Check if an index is floating

    Parameters
    ----------
    index : pd.NumericIndex
        Any numeric index

    Returns
    -------
    bool
        True if Float64Index or NumericIndex with a floating dtype
    """
    if type(index) is NumericIndex and np.issubdtype(index.dtype, np.floating):
        return True
    # Safe legacy path
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from pandas import Float64Index

            if type(index) is Float64Index:
                return True
    except ImportError:
        pass
    return False


try:
    from pandas._testing import makeDataFrame as make_dataframe
except ImportError:
    import string

    def rands_array(nchars, size, dtype="O"):
        """
        Generate an array of byte strings.
        """
        rands_chars = np.array(
            list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
        )
        retval = (
            np.random.choice(rands_chars, size=nchars * np.prod(size))
            .view((np.str_, nchars))
            .reshape(size)
        )
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
        data = {
            c: pd.Series(np.random.randn(n), index=index)
            for c in string.ascii_uppercase[:k]
        }

        return pd.DataFrame(data)


def to_numpy(po: pd.DataFrame) -> np.ndarray:
    """
    Workaround legacy pandas lacking to_numpy

    Parameters
    ----------
    po : Pandas obkect

    Returns
    -------
    ndarray
        A numpy array
    """
    try:
        return po.to_numpy()
    except AttributeError:
        return po.values
