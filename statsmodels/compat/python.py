"""
Compatibility tools for differences between Python 2 and 3
"""

import platform
import sys

PYTHON_IMPL_WASM = sys.platform == "emscripten" or platform.machine() in [
    "wasm32",
    "wasm64",
]


def asunicode(x, _):
    """
    Convert an object to a unicode string

    Parameters
    ----------
    x : object
        The object to convert.
    _ : str
        Unused, retained for signature compatibility.

    Returns
    -------
    str
        The unicode string representation of `x`.
    """
    return str(x)


__all__ = [
    "PYTHON_IMPL_WASM",
    "asbytes",
    "asstr",
    "asunicode",
    "lfilter",
    "lmap",
    "lrange",
    "lzip",
    "with_metaclass",
]


def asbytes(s):
    """
    Convert a string to bytes using latin1 encoding

    Parameters
    ----------
    s : bytes or str
        The value to convert.

    Returns
    -------
    bytes
        The bytes representation of `s`.
    """
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")


def asstr(s):
    """
    Convert bytes to a string using latin1 encoding

    Parameters
    ----------
    s : bytes or str
        The value to convert.

    Returns
    -------
    str
        The string representation of `s`.
    """
    if isinstance(s, str):
        return s
    return s.decode("latin1")


# list-producing versions of the major Python iterating functions
def lrange(*args, **kwargs):
    """
    A list-producing version of range

    Parameters
    ----------
    *args
        Positional arguments passed to ``range``.
    **kwargs
        Keyword arguments passed to ``range``.

    Returns
    -------
    list
        The values produced by ``range``.
    """
    return list(range(*args, **kwargs))


def lzip(*args, **kwargs):
    """
    A list-producing version of zip

    Parameters
    ----------
    *args
        Positional arguments passed to ``zip``.
    **kwargs
        Keyword arguments passed to ``zip``.

    Returns
    -------
    list
        The values produced by ``zip``.
    """
    return list(zip(*args, **kwargs))


def lmap(*args, **kwargs):
    """
    A list-producing version of map

    Parameters
    ----------
    *args
        Positional arguments passed to ``map``.
    **kwargs
        Keyword arguments passed to ``map``.

    Returns
    -------
    list
        The values produced by ``map``.
    """
    return list(map(*args, **kwargs))


def lfilter(*args, **kwargs):
    """
    A list-producing version of filter

    Parameters
    ----------
    *args
        Positional arguments passed to ``filter``.
    **kwargs
        Keyword arguments passed to ``filter``.

    Returns
    -------
    list
        The values produced by ``filter``.
    """
    return list(filter(*args, **kwargs))


def with_metaclass(meta, *bases):
    """
    Create a base class with a metaclass

    Parameters
    ----------
    meta : type
        The metaclass to use.
    *bases : type
        The base classes for the new class.

    Returns
    -------
    type
        A temporary base class using the given metaclass.
    """

    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

    return type.__new__(metaclass, "temporary_class", (), {})
