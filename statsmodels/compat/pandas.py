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
    """Context manager that requests offset return values from infer_freq"""
    # pandas 3.1 changes the return value of infer_freq
    try:
        with pd.option_context("future.infer_freq_returns_offset", True):
            yield
    except pd.errors.OptionError:
        # in older versions the option is not available and a str is returned
        yield


def infer_freq(index) -> str | None:
    """
    Infer the frequency string of a datetime index

    Parameters
    ----------
    index : pd.Index
        An index to infer the frequency of.

    Returns
    -------
    str or None
        The inferred frequency string, or None if it cannot be inferred.
    """
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
    Generate an array of byte strings

    Parameters
    ----------
    generator : Generator, optional
        A NumPy random generator. If None, ``np.random.default_rng()`` is
        used.
    nchars : int, optional
        The number of characters in each string.
    size : int or tuple[int, ...], optional
        The shape of the output array.
    dtype : str or None, optional
        The dtype to cast the output to. If None, the raw string array is
        returned.

    Returns
    -------
    ndarray
        An array of random strings.
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

    Parameters
    ----------
    generator : Generator, optional
        A NumPy random generator. If None, ``np.random.default_rng()`` is
        used.

    Returns
    -------
    DataFrame
        A DataFrame with 30 rows and 4 columns of standard normal data.
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
    po : DataFrame
        A pandas object.

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
    """
    Get the underlying function of a cached property

    Parameters
    ----------
    cached_prop : cache_readonly
        A cached property instance.

    Returns
    -------
    callable
        The wrapped function.
    """
    try:
        return cached_prop.fget
    except AttributeError:
        return cached_prop.func


def call_cached_func(cached_prop, *args, **kwargs):
    """
    Call the underlying function of a cached property

    Parameters
    ----------
    cached_prop : cache_readonly
        A cached property instance.
    *args
        Positional arguments passed to the underlying function.
    **kwargs
        Keyword arguments passed to the underlying function.

    Returns
    -------
    object
        The result of calling the underlying function.
    """
    f = get_cached_func(cached_prop)
    return f(*args, **kwargs)


def get_cached_doc(cached_prop) -> Optional[str]:
    """
    Get the docstring of the underlying function of a cached property

    Parameters
    ----------
    cached_prop : cache_readonly
        A cached property instance.

    Returns
    -------
    str or None
        The docstring of the underlying function.
    """
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
    """
    Decorator to deprecate a keyword argument of a function

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate.
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning
        that ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable, optional
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be raised as-is.
    stacklevel : int, optional
        How far up the stack the warning is to be logged.

    Returns
    -------
    callable
        A decorator that wraps the decorated function to warn on use of
        the deprecated keyword.
    """
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
