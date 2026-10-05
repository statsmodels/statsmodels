from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from packaging.version import Version, parse
import scipy

if TYPE_CHECKING:
    from statsmodels.tools.typing import ArrayLike, NDArray

SP_VERSION = parse(scipy.__version__)
SP_LT_19 = SP_VERSION < Version("1.8.99")
SP_LT_110 = SP_VERSION < Version("1.10.99")
SP_LT_112 = SP_VERSION < Version("1.12.99")
SP_LT_114 = SP_VERSION < Version("1.13.99")
SP_LT_115 = SP_VERSION < Version("1.14.99")
SP_LT_116 = SP_VERSION < Version("1.15.99")
SP_LT_118 = SP_VERSION < Version("1.17.99")
SP_LT_2 = SP_VERSION < Version("1.99.99")
BASINHOPPING_RNG = "seed" if SP_LT_115 else "rng"

if SP_LT_2:
    from scipy.stats.mstats import mquantiles as _sp_mquantiles


def _mquantiles_numpy(
    x: ArrayLike,
    p: ArrayLike,
    alphap: float = 0.4,
    betap: float = 0.4,
    axis: int | None = None,
) -> NDArray | float:
    """
    Compute empirical quantiles using plotting positions with NumPy

    Implementation of :func:`_mquantiles` used when SciPy >= 2. See
    :func:`_mquantiles` for the description of the parameters and the
    return value.

    Parameters
    ----------
    x : array_like
        Input data.
    p : array_like
        Probabilities at which to compute the quantiles.
    alphap : float, optional
        Plotting positions parameter.
    betap : float, optional
        Plotting positions parameter.
    axis : int, optional
        Axis along which to compute the quantiles.

    Returns
    -------
    float or ndarray
        The quantiles.
    """
    x = np.asarray(x)
    if axis is None:
        x = x.ravel()
        axis = 0
    n = x.shape[axis]
    p = np.asarray(p, dtype=float)
    # Position of the quantile in the sorted data, zero-indexed
    pos = (n + 1 - alphap - betap) * p + alphap - 1
    # Probability that np.quantile maps to pos using linear interpolation.
    # When n == 1, every probability in [0, 1] returns the only observation.
    p_adj = np.clip(pos / max(n - 1, 1), 0, 1)
    return np.quantile(x, p_adj, method="linear", axis=axis)


def _mquantiles(
    x: ArrayLike,
    p: ArrayLike,
    alphap: float = 0.4,
    betap: float = 0.4,
    axis: int | None = None,
) -> NDArray | float:
    """
    Compute empirical quantiles using plotting positions

    Replacement for ``scipy.stats.mstats.mquantiles``, which is deprecated
    as of SciPy 2, for data without masked values. Uses
    ``scipy.stats.mstats.mquantiles`` when SciPy < 2 and an equivalent
    implementation based on ``numpy.quantile`` otherwise. The shape of the
    output is the same in both cases.

    Parameters
    ----------
    x : array_like
        Input data. Must contain at least one observation along `axis`.
    p : array_like
        Probabilities at which to compute the quantiles. Values must be in
        [0, 1].
    alphap : float, optional
        Plotting positions parameter. The default is 0.4.
    betap : float, optional
        Plotting positions parameter. The default is 0.4.
    axis : int, optional
        Axis along which to compute the quantiles. If None (default), `x`
        is flattened.

    Returns
    -------
    float or ndarray
        The quantiles. A float if `p` is a scalar and `axis` is None or `x`
        is 1-dimensional. Otherwise an ndarray. If `p` is not a scalar, the
        first dimension of the result corresponds to the elements of `p`
        and the remaining dimensions are the dimensions of `x` excluding
        `axis`.

    Notes
    -----
    The sample quantile is located at the one-based position
    ``h = n * p + alphap + p * (1 - alphap - betap)`` in the sorted data,
    where ``n`` is the number of observations, and is computed by linear
    interpolation between the order statistics adjacent to ``h``. Positions
    outside of ``[1, n]`` are clipped so that the smallest and largest
    observations are returned.

    Common choices of (alphap, betap) are

    * (0, 1) : Hyndman and Fan type 4, linear interpolation of the ECDF
    * (0.5, 0.5) : Hyndman and Fan type 5, piecewise linear
    * (0, 0) : Hyndman and Fan type 6, ``p(k) = k / (n + 1)``
    * (1, 1) : Hyndman and Fan type 7, the default of ``numpy.quantile``
    * (1/3, 1/3) : Hyndman and Fan type 8, approximately median-unbiased
    * (3/8, 3/8) : Hyndman and Fan type 9, approximately unbiased if `x` is
      normally distributed
    * (0.4, 0.4) : approximately quantile unbiased (Cunnane)

    Unlike ``scipy.stats.mstats.mquantiles``, this function does not
    support masked arrays, does not accept a ``limit`` argument and always
    returns an ndarray or a float, never a masked array. When `axis` is not
    the first axis of a multidimensional `x` and `p` is not a scalar, the
    quantiles are in the first dimension of the result, while
    ``scipy.stats.mstats.mquantiles`` places them at the position of `axis`.

    References
    ----------
    .. [1] Hyndman, R. J. and Fan, Y. (1996). "Sample quantiles in
       statistical packages." The American Statistician, 50(4), 361-365.
    """
    if not SP_LT_2:
        return _mquantiles_numpy(x, p, alphap=alphap, betap=betap, axis=axis)
    x = np.asarray(x)
    scalar_p = np.ndim(p) == 0
    res = np.ma.getdata(_sp_mquantiles(x, p, alphap=alphap, betap=betap, axis=axis))
    if axis is not None and x.ndim > 1:
        # mquantiles places the quantiles at the position of axis
        res = np.moveaxis(res, axis, 0)
    return res[0] if scalar_p else res


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
        # if p35 < match: match = p35
        match = min(p35, match)
        p5 *= 5
        if p5 == target:
            return p5
    #  if p5 < match: match = p5
    match = min(match, p5)
    return match


def _valarray(shape, value=np.nan, typecode=None):
    """Return an array of all value."""

    out = np.ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


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
        try:
            # From scipy >= 1.18.0
            import scipy._external.array_api_extra as xpx
        except (ImportError, AttributeError):
            import scipy._lib.array_api_extra as xpx

        return xpx.apply_where(cond, args, f1, f2, fill_value=fill_value)
    except (ImportError, AttributeError):
        from scipy._lib._util import _lazywhere

        return _lazywhere(cond, args, f1, fill_value, f2)


__all__ = [
    "BASINHOPPING_RNG",
    "SP_LT_2",
    "SP_LT_19",
    "SP_LT_110",
    "SP_LT_112",
    "SP_LT_114",
    "SP_LT_115",
    "SP_LT_116",
    "SP_LT_118",
    "SP_VERSION",
    "apply_where",
]
