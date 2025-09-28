"""
Compatibility tools for differences between Python 2 and 3
"""

import platform
import sys

PYTHON_IMPL_WASM = (
    sys.platform == "emscripten" or platform.machine() in ["wasm32", "wasm64"]
)


def asunicode(x, _):
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
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")


def asstr(s):
    if isinstance(s, str):
        return s
    return s.decode("latin1")


# list-producing versions of the major Python iterating functions
def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))


def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs, strict=False))


def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def lfilter(*args, **kwargs):
    return list(filter(*args, **kwargs))


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

    return type.__new__(metaclass, "temporary_class", (), {})
