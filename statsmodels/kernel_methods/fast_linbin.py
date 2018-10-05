"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module defines functions to deal with binning data on regular grids.
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from . import _cy_fast_linbin
from .kde_utils import Grid


def _fast_bin(fct, X, bounds, M, weights, bin_type, out):
    r"""

    Parameters
    ----------
    X: ndarray
        Input data
    bounds: (float, float)
        Bounds of the grid
    M: int
        Number of bins. This is ignored for discrete axes, in which case the
        number of bins if computed as (upper bound - lower_bound + 1).
    weights: ndarray
        Array of same size as X with weights for each point, or scalar if all weights are 1
    bin_type: str
        Type of bin as a string (see Notes)

    Returns
    -------
    mesh: Grid
        Grid of the points
    values: ndarray
        Values for each bin in the grid

    Notes
    -----

    The bin can be:
        - Bounded ('b')
        - Reflected ('r')
        - Cyclic ('c')
        - Discrete ('d')
    """
    X = np.atleast_1d(X).astype(float)
    if X.ndim != 1:
        raise ValueError("Error, X must be a 1D array")
    if bin_type not in ('c', 'r', 'b', 'd'):
        raise ValueError("Error, bin_type must be one of 'c', 'r', 'b' or 'd'")
    try:
        float(weights)
    except TypeError:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
            raise ValueError("Weights must be a single value or an array of the same shape as X")
    else:
        weights = np.empty((0,), dtype=float)
    if bin_type == 'd':
        M = int(bounds[1] - bounds[0]) + 1
    else:
        M = int(M)
    if out is None:
        out = np.zeros((M,), dtype=float)
    elif out.shape != (M,) or out.dtype != np.dtype(np.float):
        raise ValueError("Error, out array must be a 1D float array with length M")
    try:
        a, b = bounds
        a = float(a)
        b = float(b)
    except:
        raise ValueError("Bounds must be a pair of floating point values")
    mesh, bounds = fct(X, a, b, out, weights, bin_type)

    return Grid(mesh, bounds, bin_type), out


def fast_linbin(X, bounds, M, weights=1., bin_type='b', out=None):
    return _fast_bin(_cy_fast_linbin.fast_linbin, X, bounds, M, weights, bin_type, out)

fast_linbin.__doc__ = """
    Linear Binning as described in Fan and Marron (1994)
    """ + _fast_bin.__doc__ + r"""

    Unless the bin is discrete, for a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions
    :math:`p_i` and :math:`p_{i+1}`, the bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    The placement of the bin depends on the bin type:
    - For continuous bins, the bins are placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots
    b-\delta/2\}` with :math:`delta = \frac{M-1}{b-a}`.
    - For discrete bins, the bins are placed on the integer numbers


    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.
    """


def fast_bin(X, bounds, M, weights=1., bin_type='b', out=None):
    return _fast_bin(_cy_fast_linbin.fast_bin, X, bounds, M, weights, bin_type, out)

fast_bin.__doc__ = """
    Fast Binning on regular grid
    """ + _fast_bin.__doc__


def _fast_bin_nd(fct, X, bounds, M, weights, bin_type, out):
    r"""

    Parameters
    ----------
    X: ndarray
        Input data
    bounds: ndarray
        2D array (D,2) describing for each dimension the bounds
    M: ndarray
        Number of bins for each dimension. If this is only one value, all axis
        will have the same number of bins.
        This is ignore for discrete bins, in which case the number of bins if
        the (upper bound  - lower bound + 1).
    weights: ndarray
        Array of same size as X with weights for each point, or None if all weights are 1
    bin_type: str
        Type of bin as a string (see Notes)

    Returns
    -------
    mesh: Grid
        Grid of the points
    values: ndarray
        Values for each bin in the grid

    Notes
    -----

    The bin can be:
        - Bounded ('b')
        - Reflected ('r')
        - Cyclic ('c')
        - Discrete ('d')
    """
    M = np.asarray(M).astype(int)
    X = np.atleast_2d(X).astype(np.float)
    bounds = np.atleast_2d(bounds).astype(float)

    if X.ndim != 2:
        raise ValueError("Error, the data must have at most 2 dimension.")

    D = X.shape[1]
    max_d = _cy_fast_linbin.MAX_DIMENSION
    if D > max_d:
        raise ValueError("Error, you cannot have more than {0} dimensions, and you have {1}".format(max_d, D))

    if len(bin_type) == 1:
        bin_type = bin_type * D
    elif len(bin_type) != D:
        raise ValueError("Error, bin_type must be a string of length 1 or D (e.g. the number of dimensions)")

    n = X.shape[0]
    if M.ndim == 0:
        M = M * np.ones((D,), dtype=int)
    if np.any(M < 2):
        raise ValueError("You need to specify at least 2 elements per dimension")
    if (bounds.ndim != 2 or bounds.shape[0] != D or bounds.shape[1] != 2 or M.ndim != 1 or M.shape[0] != D):
        raise ValueError("Error, incompatible dimensions for bounds, M and X")

    try:
        float(weights)
    except TypeError:
        weights = np.asarray(weights).astype(float)
        if weights.shape != (n,):
            raise ValueError("weights must be a floating point or a 1D array of same length as X")
    else:
        weights = np.empty((0,), dtype=float)

    for d in range(D):
        if bin_type[d] == 'd':
            M[d] = int(bounds[d, 1] - bounds[d, 0]) + 1
    tM = tuple(M)

    if out is None:
        out = np.zeros(tM, dtype=np.float)
    elif out.shape != tM or out.dtype != np.dtype(np.float):
        raise ValueError("Output array is either of the wrong size or the wrong type")

    mesh, bounds = fct(X, bounds[:, 0], bounds[:, 1], out, weights, bin_type)
    return Grid(mesh, bounds, bin_type), out


def fast_linbin_nd(X, bounds, M, weights=1., bin_type='b', out=None):
    X = np.atleast_2d(X).astype(np.float)
    if X.shape[1] == 2:
        return _fast_bin_nd(_cy_fast_linbin.fast_linbin_2d, X, bounds, M, weights, bin_type, out)
    if X.shape[1] == 3:
        return _fast_bin_nd(_cy_fast_linbin.fast_linbin_3d, X, bounds, M, weights, bin_type, out)
    return _fast_bin_nd(_cy_fast_linbin.fast_linbin_nd, X, bounds, M, weights, bin_type, out)

fast_linbin_nd.__doc__ = r"""
    Linear Binning in nD as described in Fan and Marron (1994)
    """ + _fast_bin_nd.__doc__ + r"""

    Unless the bin is non-continuous, for a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions
    :math:`p_i` and :math:`p_{i+1}`, the bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    The placement of the bin depends on the bin type:
    - For reflected or un-bounded bins, the bins are placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1,
    \ldots b-\delta/2\}` with :math:`delta = \frac{M-1}{b-a}`.
    - For cyclic bins, the bins are placed at :math:`\{a,\ldots,\}


    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.
    """


def fast_bin_nd(X, bounds, M, weights=1., bin_type='b', out=None):
    return _fast_bin_nd(_cy_fast_linbin.fast_bin_nd, X, bounds, M, weights, bin_type, out)

fast_bin_nd.__doc__ = r"""
    Simple Binning in nD
    """ + _fast_bin_nd.__doc__
