"""Decorators for validating function arguments."""

from functools import wraps

import numpy as np

import statsmodels.tools.validation.validation as v


def array_like(
    pos,
    name,
    dtype=np.double,
    ndim=None,
    maxdim=None,
    shape=None,
    order="C",
    contiguous=False,
):
    """
    Decorate a function argument with array_like validation.

    Parameters
    ----------
    pos : int
        Positional argument index to validate.
    name : str
        Argument name to use in exceptions and keyword lookup.
    dtype : dtype
        Required dtype.
    ndim : int or None
        Required number of dimensions.
    maxdim : int or None
        Maximum allowed number of dimensions.
    shape : tuple or None
        Required shape.
    order : {"C", "F", None}
        Required memory order.
    contiguous : bool
        Whether to require contiguous memory.

    """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if pos < len(args):
                arg = args[pos]
                arg = v.array_like(
                    arg, name, dtype, ndim, maxdim, shape, order, contiguous
                )
                if pos == 0:
                    args = (arg,) + args[1:]
                else:
                    args = args[:pos] + (arg,) + args[pos + 1:]
            else:
                arg = kwargs[name]
                arg = v.array_like(
                    arg, name, dtype, ndim, maxdim, shape, order, contiguous
                )
                kwargs[name] = arg

            return func(*args, **kwargs)

        return wrapper

    return inner
