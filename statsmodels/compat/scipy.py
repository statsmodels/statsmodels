import numpy as np
from packaging.version import Version, parse
import scipy

SP_VERSION = parse(scipy.__version__)
SP_LT_15 = SP_VERSION < Version("1.4.99")
SCIPY_GT_14 = not SP_LT_15
SP_LT_16 = SP_VERSION < Version("1.5.99")
SP_LT_17 = SP_VERSION < Version("1.6.99")
SP_LT_19 = SP_VERSION < Version("1.8.99")
SP_LT_116 = SP_VERSION < Version("1.15.99")


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            p2 = 2 ** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _valarray(shape, value=np.nan, typecode=None):
    """Return an array of all value."""

    out = np.ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


if SP_LT_16:
    # copied from scipy, added to scipy in 1.6.0
    from ._scipy_multivariate_t import multivariate_t  # noqa: F401
else:
    from scipy.stats import multivariate_t  # noqa: F401


def apply_where(  # type: ignore[explicit-any] # numpydoc ignore=PR01,PR02
    cond, args, f1, f2=None, /, *, fill_value=None
):
    """
    Run one of two elementwise functions depending on a condition.

    Equivalent to ``f1(*args) if cond else fill_value`` performed elementwise
    when `fill_value` is defined, otherwise to ``f1(*args) if cond else f2(*args)``.

    Parameters
    ----------
    cond : array
        The condition, expressed as a boolean array.
    args : Array or tuple of Arrays
        Argument(s) to `f1` (and `f2`). Must be broadcastable with `cond`.
    f1 : callable
        Elementwise function of `args`, returning a single array.
        Where `cond` is True, output will be ``f1(arg0[cond], arg1[cond], ...)``.
    f2 : callable, optional
        Elementwise function of `args`, returning a single array.
        Where `cond` is False, output will be ``f2(arg0[cond], arg1[cond], ...)``.
        Mutually exclusive with `fill_value`.
    fill_value : Array or scalar, optional
        If provided, value with which to fill output array where `cond` is False.
        It does not need to be scalar; it needs however to be broadcastable with
        `cond` and `args`.
        Mutually exclusive with `f2`. You must provide one or the other.
    xp : array_namespace, optional
        The standard-compatible namespace for `cond` and `args`. Default: infer.

    Returns
    -------
    Array
        An array with elements from the output of `f1` where `cond` is True and either
        the output of `f2` or `fill_value` where `cond` is False. The returned array has
        data type determined by type promotion rules between the output of `f1` and
        either `fill_value` or the output of `f2`.

    Notes
    -----
    Falls back to _lazywhere if xpx.apply_where is not available.

    ``xp.where(cond, f1(*args), f2(*args))`` requires explicitly evaluating `f1` even
    when `cond` is False, and `f2` when cond is True. This function evaluates each
    function only for their matching condition, if the backend allows for it.

    On Dask, `f1` and `f2` are applied to the individual chunks and should use functions
    from the namespace of the chunks.

    """
    try:
        import scipy._lib.array_api_extra as xpx

        return xpx.apply_where(cond, args, f1, f2, fill_value=fill_value)
    except (ImportError, AttributeError):
        from scipy._lib._util import _lazywhere

        return _lazywhere(cond, args, f1, fill_value, f2)
