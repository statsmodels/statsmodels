from __future__ import annotations

from contextlib import contextmanager
import string
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, TypeVar

import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
    cache_readonly,
    deprecate_kwarg as pd_deprecate_kwarg,
)

from statsmodels.tools.docstring_helpers import Appender, Substitution

if TYPE_CHECKING:
    try:
        from typing import TypeAlias
    except ImportError:
        from typing_extensions import TypeAlias


FuncType: TypeAlias = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
__all__ = [
    "FUTURE_STACK",
    "MONTH_END",
    "PD_LT_1_0_0",
    "PD_LT_1_4",
    "PD_LT_2",
    "PD_LT_3",
    "PD_LT_3_1_0",
    "QUARTER_END",
    "YEAR_END",
    "Appender",
    "Substitution",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_series_equal",
    "cache_readonly",
    "call_cached_func",
    "data_klasses",
    "deprecate_kwarg",
    "frequencies",
    "get_cached_doc",
    "get_cached_func",
    "is_float_index",
    "is_int_index",
    "is_numeric_dtype",
    "make_dataframe",
    "testing",
    "to_numpy",
]

version = parse(pd.__version__)

PD_LT_3_1_0 = version < Version("3.0.99")
PD_LT_2_2_0 = version < Version("2.1.99")
PD_LT_2_1_0 = version < Version("2.0.99")
PD_LT_1_0_0 = version < Version("0.99.0")
PD_LT_1_4 = version < Version("1.3.99")
PD_LT_2 = version < Version("1.99.99")
PD_LT_3 = version < Version("2.99.99")

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
    from pandas import testing
except ImportError:
    from pandas.util import testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal


@contextmanager
def _infer_freq_returns_offset():

    # pandas 3.1 changes the return value of infer_freq
    try:
        with pd.option_context("future.infer_freq_returns_offset", True):
            yield
    except pd.errors.OptionError:
        # in older versions the option is not available and a str is returned
        yield


def infer_freq(index) -> str | None:

    # pandas 3.1 changes the return value of infer_freq
    with _infer_freq_returns_offset():
        freq = pd.infer_freq(index)

    # new pandas versions returns BaseOffset
    if not isinstance(freq, str) and freq is not None:
        return freq.freqstr
    return freq


def is_int_index(index: pd.Index) -> bool:
    """
    Check if an index is integral

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if is an index with a standard integral type
    """
    return (
        isinstance(index, pd.Index)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.integer)
    )


def is_float_index(index: pd.Index) -> bool:
    """
    Check if an index is floating

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if an index with a standard numpy floating dtype
    """
    return (
        isinstance(index, pd.Index)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.floating)
    )


def rands_array(generator=None, nchars=10, size=10, dtype="O"):
    """
    Generate an array of byte strings.
    """
    if generator is None:
        generator = np.random.default_rng()
    rands_chars = np.array(
        list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
    )
    retval = (
        generator.choice(rands_chars, size=nchars * np.prod(size))
        .view((np.str_, nchars))
        .reshape(size)
    )
    if dtype is None:
        return retval
    else:
        return retval.astype(dtype)


def make_dataframe(generator=None):
    """
    Simple version of pandas._testing.makeDataFrame
    """
    if generator is None:
        generator = np.random.default_rng()
    n = 30
    k = 4
    index = pd.Index(rands_array(generator=generator, nchars=10, size=n), name=None)
    data = {
        c: pd.Series(generator.standard_normal(n), index=index)
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


def get_cached_func(cached_prop):
    try:
        return cached_prop.fget
    except AttributeError:
        return cached_prop.func


def call_cached_func(cached_prop, *args, **kwargs):
    f = get_cached_func(cached_prop)
    return f(*args, **kwargs)


def get_cached_doc(cached_prop) -> Optional[str]:
    return get_cached_func(cached_prop).__doc__


MONTH_END = "M" if PD_LT_2_2_0 else "ME"
QUARTER_END = "Q" if PD_LT_2_2_0 else "QE"
YEAR_END = "Y" if PD_LT_2_2_0 else "YE"
FUTURE_STACK = {} if PD_LT_2_1_0 else {"future_stack": True}


def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    if PD_LT_3:
        return pd_deprecate_kwarg(
            old_arg_name=old_arg_name,
            new_arg_name=new_arg_name,
            mapping=mapping,
            stacklevel=stacklevel,
        )
    else:
        return pd_deprecate_kwarg(
            klass=FutureWarning,
            old_arg_name=old_arg_name,
            new_arg_name=new_arg_name,
            mapping=mapping,
            stacklevel=stacklevel,
        )
