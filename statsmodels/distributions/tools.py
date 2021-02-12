# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:19:30 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats, interpolate


# helper functions to work on a grid of cdf and pdf, histogram

def prob2cdf_grid(x):
    """cumulative counts from cell provabilites on a grid
    """
    cdf = np.asarray(x).copy()
    k = cdf.ndim
    for i in range(k):
        cdf = cdf.cumsum(axis=i)

    return cdf


def cdf2prob_grid(x):
    """cell provabilites from cumulative counts on a grid
    """
    pdf = np.asarray(x).copy()
    k = pdf.ndim
    for i in range(k):
        pdf = np.diff(pdf, prepend=0, axis=i)

    return pdf


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
        # ki = np.arange(k_terms[i]).astype(float)
        ki = ki[..., None]
        ni = k_terms[i] - 1
        xi = xx[:, i]
        poly_base = poly_base[None, ...] + stats.binom._logpmf(ki, ni, xi)

    poly_base = np.exp(poly_base)
    bp_values = fvals[..., None] * poly_base

    for i in range(k_dim):
        bp_values = bp_values.sum(0)

    return bp_values
