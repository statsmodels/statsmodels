# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:19:30 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats, interpolate


# helper functions to work on a grid of cdf and pdf, histogram

class _Grid(object):
    """Create Grid values and indices, grid in [0, 1]^d

    This class creates a regular grid in a d dimensional hyper cube.

    Intended for internal use, implementation might change without warning.


    Parameters
    ----------
    k_grid : tuple or array_like
        number of elements for axes, this defines k_grid - 1 equal sized
        intervals of [0, 1] for each axis.
    eps : float
        If eps is not zero, then x values will be clipped to [eps, 1 - eps],
        i.e. to the interior of the unit interval or hyper cube.


    Attributes
    ----------
    k_grid : list of number of grid points
    x_marginal: list of 1-dimensional marginal values
    idx_flat: integer array with indices
    x_flat: flattened grid values,
        rows are grid points, columns represent variables or axis.
        ``x_flat`` is currently also 2-dim in the univariate 1-dim grid case.

    """

    def __init__(self, k_grid, eps=0):
        self.k_grid = k_grid

        x_marginal = [np.arange(ki) / (ki - 1) for ki in k_grid]

        idx_flat = np.column_stack(
                np.unravel_index(np.arange(np.product(k_grid)), k_grid)
                ).astype(float)
        x_flat = idx_flat / idx_flat.max(0)
        if eps != 0:
            x_marginal = [np.clip(xi, eps, 1 - eps) for xi in x_marginal]
            x_flat = np.clip(x_flat, eps, 1 - eps)

        self.x_marginal = x_marginal
        self.idx_flat = idx_flat
        self.x_flat = x_flat


def prob2cdf_grid(x):
    """cumulative counts from cell provabilites on a grid
    """
    cdf = np.asarray(x).copy()
    k = cdf.ndim
    for i in range(k):
        cdf = cdf.cumsum(axis=i)

    return cdf


def cdf2prob_grid(x, prepend=0):
    """cell provabilites from cumulative counts on a grid
    """
    if prepend is None:
        prepend = np._NoValue
    pdf = np.asarray(x).copy()
    k = pdf.ndim
    for i in range(k):
        pdf = np.diff(pdf, prepend=prepend, axis=i)

    return pdf


def average_grid(values, coords=None, _method="slicing"):
    """compute average for each cell in grid using endpoints

    Parameters
    ----------
    values : array_like
        Values on a grid that will average over corner points of each cell.
    coords : None or list of array_like
        Grid coordinates for each axis use to compute volumne of cell.
        If None, then averaged values are not rescaled.
    _method : {"slicing", "convolve"}
        Grid averaging is implemented using numpy "slicing" or using
        scipy.signal "convolve".


    """

    k_dim = values.ndim
    if _method == "slicing":
        p = values.copy()

        for d in range(k_dim):
            # average (p[:-1] + p[1:]) / 2 over each axis
            sl1 = [slice(None, None, None)] * k_dim
            sl2 = [slice(None, None, None)] * k_dim
            sl1[d] = slice(None, -1, None)
            sl2[d] = slice(1, None, None)
            sl1 = tuple(sl1)
            sl2 = tuple(sl2)

            p = (p[sl1] + p[sl2]) / 2

    elif _method == "convolve":
        from scipy import signal
        p = signal.convolve(values, 0.5**k_dim * np.ones([2] * k_dim),
                            mode="valid")

    if coords is not None:
        dx = np.array(1)
        for d in range(k_dim):
            dx = dx[..., None] * np.diff(coords[d])

        p = p * dx

    return p


# functions to evaluate bernstein polynomials

def _eval_bernstein_1d(x, fvals, method="binom"):
    """evaluate 1-dimensional bernstein polynomial given grid of values

    experimental, comparing methods

    """
    k_terms = fvals.shape[-1]
    xx = np.asarray(x)
    k = np.arange(k_terms).astype(float)
    n = k_terms - 1.

    if method.lower() == "binom":
        poly_base = stats.binom.pmf(k, n, xx[..., None])
        bp_values = (fvals * poly_base).sum(-1)
    elif method.lower() == "bpoly":
        bpb = interpolate.BPoly(fvals[:, None], [0., 1])
        bp_values = bpb(x)
    elif method.lower() == "beta":
        poly_base = stats.beta.pdf(xx[..., None], k + 1, n - k + 1) / (n + 1)
        bp_values = (fvals * poly_base).sum(-1)
    else:
        raise ValueError("method not recogized")

    return bp_values


def _eval_bernstein_2d(x, fvals):
    """evaluate 2-dimensional bernstein polynomial given grid of values

    experimental

    """
    k_terms = fvals.shape
    k_dim = fvals.ndim
    if k_dim != 2:
        raise ValueError("`fval` needs to be 2-dimensional")
    xx = np.atleast_2d(x)
    if xx.shape[1] != 2:
        raise ValueError("x needs to be bivariate and have 2 columns")

    x1, x2 = xx.T
    n1, n2 = k_terms[0] - 1, k_terms[1] - 1
    k1 = np.arange(k_terms[0]).astype(float)
    k2 = np.arange(k_terms[1]).astype(float)

    # we are building a nobs x n1 x n2 array
    poly_base = np.zeros(x.shape[0])
    poly_base = (stats.binom.pmf(k1[None, :, None], n1, x1[:, None, None]) *
                 stats.binom.pmf(k2[None, None, :], n2, x2[:, None, None]))
    bp_values = (fvals * poly_base).sum(-1).sum(-1)

    return bp_values


def _eval_bernstein_dd(x, fvals):
    """evaluate d-dimensional bernstein polynomial given grid of values

    experimental, currently requires square grid.

    """
    k_terms = fvals.shape
    k_dim = fvals.ndim
    xx = np.atleast_2d(x)
    # assuming square grid, same ki in all dimensions
    ki = np.arange(k_terms[0]).astype(float)

    # The following loop is a tricky
    # we add terms for each x and expand dimension of poly base in each
    # iteration using broadcasting

    poly_base = np.zeros(x.shape[0])
    for i in range(k_dim):
        ki = np.arange(k_terms[i]).astype(float)
        for j in range(i+1):
            ki = ki[..., None]
        ni = k_terms[i] - 1
        xi = xx[:, i]
        poly_base = poly_base[None, ...] + stats.binom._logpmf(ki, ni, xi)

    poly_base = np.exp(poly_base)
    bp_values = fvals.T[..., None] * poly_base

    for i in range(k_dim):
        bp_values = bp_values.sum(0)

    return bp_values
