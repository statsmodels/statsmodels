"""Validation helpers for array-like and scalar inputs"""

from __future__ import annotations

from collections.abc import Mapping
import math
import numbers
import operator
from typing import Any
import warnings

import numpy as np
import pandas as pd


def _right_squeeze(arr, stop_dim=0):
    """
    Remove trailing singleton dimensions

    Parameters
    ----------
    arr : ndarray
        Input array
    stop_dim : int
        Dimension where checking should stop so that shape[i] is not checked
        for i < stop_dim

    Returns
    -------
    squeezed : ndarray
        Array with all trailing singleton dimensions (0 or 1) removed.
        Singleton dimensions for dimension < stop_dim are retained.

    """
    last = arr.ndim
    for s in reversed(arr.shape):
        if s > 1:
            break
        last -= 1
    last = max(last, stop_dim)

    return arr.reshape(arr.shape[:last])


def array_like(
    obj,
    name,
    *,
    dtype=np.double,
    ndim=None,
    maxdim=None,
    mindim: int = 1,
    shape=None,
    order=None,
    contiguous=False,
    optional=False,
    writeable=True,
):
    """
    Convert array-like to a ndarray and check conditions

    Parameters
    ----------
    obj : array_like
         An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    name : str
        Name of the variable to use in exceptions
    dtype : numpy.dtype or str, optional
        Required dtype. Default is double. If None, does not change the dtype
        of obj (if present) or uses NumPy to automatically detect the dtype
    ndim : int, optional
        Required number of dimensions of obj. If None, no check is performed.
        If the number of dimensions of obj is less than ndim, additional axes
        are inserted on the right. See examples.
    maxdim : int, optional
        Maximum allowed dimension.  Use ``maxdim`` instead of ``ndim`` when
        inputs are allowed to have ndim 1, 2, ..., or maxdim.
    mindim : int, optional
        Minimum allowed dimension.  Arrays with ndim < mindim will be reshaped
        to have ndim = mindim by adding singleton dimensions on the right.
    shape : tuple of int, optional
        Required shape obj.  If None, no check is performed. Partially
        restricted shapes can be checked using None. See examples.
    order : {'C', 'F'}, optional
        Order of the array
    contiguous : bool, optional
        Ensure that the array's data is contiguous with order ``order``
    optional : bool, optional
        Flag indicating whether None is allowed
    writeable : bool, optional
        Whether to ensure the returned array is writeable

    Returns
    -------
    ndarray
        The converted input, always a base-class ndarray: a subclass such as
        ``np.matrix`` or a masked array is converted, as ``np.asarray`` does,
        and the mask of a masked array is dropped.

    Notes
    -----
    .. versionchanged:: 0.15.1

       Subclasses of ndarray are converted to ndarray. They were returned
       unchanged, so ``np.matrix`` input kept matrix semantics and the mask
       of a masked array was kept.

    Examples
    --------
    Convert a list or pandas series to an array
    >>> import pandas as pd
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    >>> a = array_like(pd.Series(x), 'x', ndim=1)
    >>> a.shape
    (4,)

    Squeezes singleton dimensions when required
    >>> x = np.array(x).reshape((4, 1))
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    Right-appends when required size is larger than actual
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=2)
    >>> a.shape
    (4, 1)

    Check only the first and last dimension of the input
    >>> x = np.arange(4*10*4).reshape((4, 10, 4))
    >>> y = array_like(x, 'x', ndim=3, shape=(4, None, 4))

    Check only the first two dimensions
    >>> z = array_like(x, 'x', ndim=3, shape=(4, 10, None))

    Raises ValueError if constraints are not satisfied
    >>> z = array_like(x, 'x', ndim=2)
    Traceback (most recent call last):
     ...
    ValueError: x is required to have ndim 2 but has ndim 3

    >>> z = array_like(x, 'x', ndim=3, shape=(10, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (10, 4, 4) but has shape (4, 10, 4)

    >>> z = array_like(x, 'x', ndim=3, shape=(None, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (*, 4, 4) but has shape (4, 10, 4)

    """
    if obj is None:
        if optional:
            return None
        raise TypeError(f"{name} must be array_like, not None")
    # "E" makes np.require return a base-class ndarray, as np.asarray does
    reqs = ["E", "W"] if writeable else ["E"]
    if order == "C" or contiguous:
        reqs += ["C"]
    elif order == "F":
        reqs += ["F"]
    arr = np.require(obj, dtype=dtype, requirements=reqs)
    input_ndim = arr.ndim
    if maxdim is not None:
        if arr.ndim > maxdim:
            msg = f"{name} must have ndim <= {maxdim}"
            raise ValueError(msg)
    if mindim is not None and arr.ndim < mindim:
        arr = np.reshape(arr, arr.shape + (1,) * (mindim - arr.ndim))
    if ndim is not None:
        if arr.ndim > ndim:
            arr = _right_squeeze(arr, stop_dim=ndim)
        elif arr.ndim < ndim:
            arr = np.reshape(arr, arr.shape + (1,) * (ndim - arr.ndim))
        if arr.ndim != ndim:
            msg = "{0} is required to have ndim {1} but has ndim {2}"
            raise ValueError(msg.format(name, ndim, arr.ndim))
    if shape is not None:
        if len(shape) != arr.ndim:
            unit = "dimension" if len(shape) == 1 else "dimensions"
            msg = f"{name} must have {len(shape)} {unit}, but has {input_ndim}"
            raise ValueError(msg)
        for actual, req in zip(arr.shape, shape, strict=True):
            if req is not None and actual != req:
                req_shape = str(shape).replace("None, ", "*, ")
                msg = "{0} is required to have shape {1} but has shape {2}"
                raise ValueError(msg.format(name, req_shape, arr.shape))
    return arr


class PandasWrapper:
    """
    Wrap array_like using the index from the original input, if pandas

    Parameters
    ----------
    pandas_obj : Series or DataFrame
        Object to extract the index from for wrapping

    Notes
    -----
    Raises if ``orig`` is a pandas type but obj and ``orig`` have
    different numbers of elements in axis 0. Also raises if the ndim of obj
    is larger than 2.

    """

    def __init__(self, pandas_obj):
        self._pandas_obj = pandas_obj
        self._is_pandas = isinstance(pandas_obj, (pd.Series, pd.DataFrame))

    def wrap(self, obj, columns=None, append=None, trim_start=0, trim_end=0):
        """
        Wrap array_like using the index from the original input

        Parameters
        ----------
        obj : array_like
            The value to wrap like to a pandas Series or DataFrame.
        columns : str or list of str, optional
            Column names or series name, if obj is 1d.
        append : str, optional
            String to append to the columns to create a new column name.
        trim_start : int, optional
            The number of observations to drop from the start of the index, so
            that the index applied is index[trim_start:].
        trim_end : int, optional
            The number of observations to drop from the end of the index , so
            that the index applied is index[:nobs - trim_end].

        Returns
        -------
        Series, DataFrame or ndarray
            A pandas Series or DataFrame, depending on the shape of obj, if
            the object used to create the wrapper was pandas. Otherwise,
            returns `obj` converted to an ndarray, unchanged.

        """
        obj = np.asarray(obj)
        if not self._is_pandas:
            return obj

        if obj.shape[0] + trim_start + trim_end != self._pandas_obj.shape[0]:
            raise ValueError(
                "obj must have the same number of elements in axis 0 as orig"
            )
        index = self._pandas_obj.index
        index = index[trim_start : index.shape[0] - trim_end]
        if obj.ndim == 1:
            if columns is None:
                name = getattr(self._pandas_obj, "name", None)
            elif isinstance(columns, str):
                name = columns
            else:
                name = columns[0]
            if append is not None:
                name = append if name is None else f"{name}_{append}"

            return pd.Series(obj, name=name, index=index)
        elif obj.ndim == 2:
            if columns is None:
                columns = getattr(self._pandas_obj, "columns", None)
            if append is not None:
                columns = [append if c is None else f"{c}_{append}" for c in columns]
            return pd.DataFrame(obj, columns=columns, index=index)
        else:
            raise ValueError("Can only wrap 1 or 2-d array_like")


def bool_like(value, name, optional=False, strict=False):
    """
    Convert to bool or raise if not bool_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool, optional
        Flag indicating whether None is allowed
    strict : bool, optional
        If True, then only allow bool. If False, allow types that support
        casting to bool.

    Returns
    -------
    converted : bool
        value converted to a bool

    """
    if optional and value is None:
        return value
    extra_text = " or None" if optional else ""
    if strict:
        if isinstance(value, bool):
            return value
        else:
            raise TypeError(f"{name} must be a bool{extra_text}")

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()
    try:
        return bool(value)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a bool (or bool-compatible){extra_text}"
        ) from exc


def is_int_like(value: Any, strict: bool = False) -> bool:
    """
    Check whether a value is a scalar integer

    Parameters
    ----------
    value : object
        Value to check.
    strict : bool, optional
        If True, only integer types are integers. If False, a finite
        ``float`` or ``np.floating`` with an integral value, such as ``3.0``,
        is also an integer.

    Returns
    -------
    bool
        True if `value` is a scalar integer.

    See Also
    --------
    int_like
        Convert a value to an int, raising if it is not integer-like.

    Notes
    -----
    An integer is any object that supports ``__index__`` (``int``, NumPy
    integers, ``IntEnum`` members and the integer scalars of other array
    libraries), except ``bool``, ``np.bool_`` and ``np.timedelta64``, which
    are never integers. A 0-d array is treated as the scalar it holds; any
    other array, a Series and a list are not integers, even if they hold a
    single value.

    Use this function to dispatch on a parameter that can be an integer or
    something else, such as ``lags : int or str``. Test for the other types
    first when they are unambiguous, for example ``isinstance(lags, str)``.

    .. versionadded:: 0.15.1

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tools.validation import is_int_like
    >>> is_int_like(np.int64(3)), is_int_like(3.0), is_int_like(3.0, strict=True)
    (True, True, False)
    >>> is_int_like(True), is_int_like(np.array([3])), is_int_like("3")
    (False, False, False)
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bool, np.bool_, np.timedelta64)):
        return False
    if isinstance(value, (float, np.floating)):
        return not strict and math.isfinite(value) and float(value).is_integer()
    if isinstance(value, (int, np.integer)):
        return True
    try:
        operator.index(value)
    except TypeError:
        return False
    return True


def _legacy_int_like(value: Any, strict: bool) -> int | None:
    """
    Apply the rules int_like used before 0.15.1 to a value is_int_like rejects

    They accepted a container holding a single value (a size-1 ndarray,
    Series or DataFrame) and, unless strict, any number that equals its floor,
    such as ``Decimal("3")``. Bools are rejected here too: accepting
    ``np.bool_`` and boolean containers was a bug, not a deprecated feature.
    Returns None if the old rules reject the value as well.
    """
    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if isinstance(value, bool) or kind in ("b", "m", "M"):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if not strict:
        try:
            if value == (value // 1):
                return int(value)
        except Exception:
            pass
    return None


def int_like(
    value: Any, name: str, optional: bool = False, strict: bool = False
) -> int | None:
    """
    Convert to int or raise if not int_like

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool, optional
        Flag indicating whether None is allowed.
    strict : bool, optional
        If True, only integer types are accepted. If False, a finite float
        with an integral value, such as ``3.0``, is also accepted.

    Returns
    -------
    converted : int
        Value converted to an int.

    Raises
    ------
    TypeError
        If `value` is not integer-like according to :func:`is_int_like`.

    Warns
    -----
    FutureWarning
        If `value` is a container holding a single integer-like value (a
        size-1 ndarray, Series or DataFrame) or, unless `strict`, a number
        such as ``Decimal("3")`` that :func:`is_int_like` rejects. These are
        accepted for now and will raise a ``TypeError`` in statsmodels 0.17.

    See Also
    --------
    is_int_like
        The rule that decides what is integer-like.

    Notes
    -----
    .. versionchanged:: 0.15.1

       ``np.bool_`` and boolean arrays and Series are rejected, as ``bool``
       always was. Objects that support ``__index__`` are accepted.
       Containers that hold a single value are deprecated.
    """
    if optional and value is None:
        return None
    if is_int_like(value, strict=strict):
        return int(value)
    legacy = _legacy_int_like(value, strict)
    if legacy is not None:
        warnings.warn(
            f"Passing {name} of type {type(value).__name__} is deprecated and "
            f"will raise a TypeError in statsmodels 0.17. Pass {name}={legacy} "
            "as an int instead.",
            FutureWarning,
            stacklevel=2,
        )
        return legacy
    extra_text = " or None" if optional else ""
    if strict:
        accepted = "an int or a NumPy integer, but not a bool or a float"
    else:
        accepted = (
            "an int, a NumPy integer or a float with an integral value, "
            "but not a bool"
        )
    raise TypeError(f"{name} must be integer_like ({accepted}){extra_text}")


def required_int_like(value: Any, name: str, strict: bool = False) -> int:
    """
    Convert to int or raise if not int_like, requiring a value (not optional)

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    strict : bool, optional
        If True, only integer types are accepted. If False, a finite float
        with an integral value, such as ``3.0``, is also accepted.

    Returns
    -------
    converted : int
        Value converted to an int.

    See Also
    --------
    int_like
        The same conversion, with None allowed if ``optional=True``.
    """
    _int = int_like(value, name, optional=False, strict=strict)
    assert _int is not None
    return _int


def is_float_like(value: Any, strict: bool = False) -> bool:
    """
    Check whether a value is a scalar real number

    Parameters
    ----------
    value : object
        Value to check.
    strict : bool, optional
        If True, only ``float``, NumPy floating types and the integers
        accepted by ``is_int_like(value, strict=True)`` are real numbers. If
        False, any other instance of :class:`numbers.Real`, such as a
        :class:`fractions.Fraction`, is also a real number.

    Returns
    -------
    bool
        True if `value` is a scalar real number.

    See Also
    --------
    float_like
        Convert a value to a float, raising if it is not float-like.
    is_int_like
        Check whether a value is a scalar integer.

    Notes
    -----
    ``bool``, ``np.bool_`` and ``np.timedelta64`` are never real numbers,
    and neither is a complex number, even with a zero imaginary part. NaN
    and infinity are real numbers. A 0-d array is treated as the scalar it
    holds; any other array, a Series and a list are not real numbers, even
    if they hold a single value.

    Use this function to dispatch on a parameter that can be a number or
    something else, such as ``scale : float or str``.

    .. versionadded:: 0.15.1

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tools.validation import is_float_like
    >>> is_float_like(0.5), is_float_like(np.int64(3)), is_float_like(np.nan)
    (True, True, True)
    >>> is_float_like(True), is_float_like(1 + 0j), is_float_like(np.array([0.5]))
    (False, False, False)
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bool, np.bool_, np.timedelta64)):
        return False
    if isinstance(value, (float, np.floating)) or is_int_like(value, strict=True):
        return True
    return not strict and isinstance(value, numbers.Real)


def _legacy_float_like(value: Any, strict: bool) -> float | None:
    """
    Apply the rules float_like used before 0.15.1 to a value is_float_like rejects

    They accepted a container holding a single value (a size-1 ndarray,
    Series or DataFrame) and, unless strict, a complex number with a zero
    imaginary part. Bools are rejected here too: accepting ``np.bool_`` and
    boolean containers was a bug, not a deprecated feature. Returns None if
    the old rules reject the value as well.
    """
    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if isinstance(value, bool) or kind in ("b", "m", "M"):
        return None
    is_complex = isinstance(value, (complex, np.complexfloating)) or kind == "c"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return float(value)
    if strict:
        return None
    if is_complex:
        return float(np.real(value)) if np.imag(value) == 0 else None
    try:
        return float(value / 1.0)
    except Exception:
        return None


def float_like(
    value: Any, name: str, optional: bool = False, strict: bool = False
) -> float | None:
    """
    Convert to float or raise if not float_like

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool, optional
        Flag indicating whether None is allowed.
    strict : bool, optional
        If True, only ``float``, ``int`` and NumPy real types are accepted.
        If False, any other instance of :class:`numbers.Real`, such as a
        :class:`fractions.Fraction`, is also accepted.

    Returns
    -------
    converted : float
        Value converted to a float.

    Raises
    ------
    TypeError
        If `value` is not float-like according to :func:`is_float_like`.

    Warns
    -----
    FutureWarning
        If `value` is a container holding a single real number (a size-1
        ndarray, Series or DataFrame) or, unless `strict`, a complex number
        with a zero imaginary part. These are accepted for now and will raise
        a ``TypeError`` in statsmodels 0.17.

    See Also
    --------
    is_float_like
        The rule that decides what is float-like.

    Notes
    -----
    .. versionchanged:: 0.15.1

       ``np.bool_``, ``np.timedelta64`` and boolean arrays and Series are
       rejected, as ``bool`` always was. Containers that hold a single value
       and complex numbers with a zero imaginary part are deprecated.
    """
    if optional and value is None:
        return None
    if is_float_like(value, strict=strict):
        return float(value)
    legacy = _legacy_float_like(value, strict)
    if legacy is not None:
        warnings.warn(
            f"Passing {name} of type {type(value).__name__} is deprecated and "
            f"will raise a TypeError in statsmodels 0.17. Pass {name}={legacy} "
            "as a float instead.",
            FutureWarning,
            stacklevel=2,
        )
        return legacy
    extra_text = " or None" if optional else ""
    if strict:
        accepted = "a float, an int or a NumPy real number"
    else:
        accepted = "a real number such as a float, an int or a NumPy real number"
    raise TypeError(
        f"{name} must be float_like ({accepted}, but not a bool or a complex "
        f"number){extra_text}"
    )


def string_like(
    value,
    name,
    optional=False,
    options=None,
    lower=True,
    deprecated=None,
    removed_after=None,
):
    """
    Check if object is string-like and raise if not

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool, optional
        Flag indicating whether None is allowed.
    options : tuple[str], optional
        Allowed values for input parameter `value`.
    lower : bool, optional
        Convert all case-based characters in `value` into lowercase.
    deprecated : dict[str, str], optional
        Mapping from a deprecated, undocumented spelling of an option to
        the documented value in `options` it stands for. A `value`
        matching one of these deprecated spellings is still accepted and
        normalized to its replacement, but raises a ``FutureWarning``.
        Ignored if `options` is None.
    removed_after : str, optional
        statsmodels version after which the spellings in `deprecated` will
        stop being accepted, e.g. ``"0.16"``. Included in the
        ``FutureWarning`` message when `deprecated` triggers. Ignored if
        `deprecated` is None; if `deprecated` is set but `removed_after` is
        not, the message says only that the alias will be removed "in a
        future version".

    Returns
    -------
    str
        The validated input, normalized to its replacement in `options`
        if it matched a key of `deprecated`.

    Raises
    ------
    TypeError
        If the value is not a string or None when optional is True.
    ValueError
        If the input is not in ``options`` (or a key of ``deprecated``,
        if provided) when ``options`` is set.

    """
    if optional and value is None:
        return None
    if not isinstance(value, str):
        extra_text = " or None" if optional else ""
        raise TypeError(f"{name} must be a string{extra_text}")
    if lower:
        value = value.lower()
    all_options = options
    if options is not None and deprecated is not None:
        all_options = tuple(options) + tuple(deprecated)
    if all_options is not None and value not in all_options:
        extra_text = "If not None, " if optional else ""
        options_text = "'" + "', '".join(all_options) + "'"
        msg = f"{extra_text}{name} must be one of: {options_text}"
        raise ValueError(msg)
    if deprecated is not None and value in deprecated:
        replacement = deprecated[value]
        if removed_after is not None:
            removal_text = (
                f"and will be removed after statsmodels {removed_after} "
                "is released."
            )
        else:
            removal_text = "and will be removed in a future version."
        warnings.warn(
            f"{name}={value!r} is a deprecated alias for {name}={replacement!r} "
            f"{removal_text}",
            FutureWarning,
            stacklevel=2,
        )
        value = replacement
    return value


def dict_like(value, name, optional=False, strict=True):
    """
    Check if dict_like (dict, Mapping) or raise if not

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool, optional
        Flag indicating whether None is allowed
    strict : bool, optional
        If True, then only allow dict. If False, allow any Mapping-like object.

    Returns
    -------
    converted : dict or Mapping
        value

    """
    if optional and value is None:
        return None
    if not isinstance(value, Mapping) or (strict and not (isinstance(value, dict))):
        extra_text = "If not None, " if optional else ""
        strict_text = "" if strict else " or dict_like (i.e., a Mapping)"
        msg = f"{extra_text}{name} must be a dict{strict_text}"
        raise TypeError(msg)
    return value
