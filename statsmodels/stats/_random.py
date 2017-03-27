# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:40:25 2017

Author: Josef Perktold

"""
from __future__ import division

import numbers
import numpy as np
from scipy.special import gammaln


# copy-pasted from scipy which copy-pasted from scikit-learn utils/validation.py
# temporary location, will need to be more central in utils.
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _logfactsum(x, axis=None):
    """compute sum of log factorial of array x

    This is just a convenience function using scipy.special.gammaln
    """
    x = np.atleast_1d(x)
    return gammaln(x + 1).sum(axis)


def simulate_table_permutation_gen(n_row, n_col, seed=None):
    """generator for random contingency table by permutation

    This is a generator, that yields the next random table.
    The contingency table is generated under the assumption of independence
    with fixed margins.

    Parameters
    ----------
    n_row : array_like, int
        row margin, count of observations for each row
    n_row : array_like, int
        row margin, count of observations for each row

    Returns
    -------
    gen : generator
        The generator is of infinite length, use `next(gen)` to obtain
        the next random contingency table with given row and column margins

    Notes
    -----
    This implements a simple permutation and np.bincount. Time will largely
    depend on the number of observations and less on the number of cells.
    This could be extended to more than two dimensional contingency tables.

    """
    rng = check_random_state(seed)
    n_row = np.asarray(n_row, np.int64)
    n_col = np.asarray(n_col, np.int64)
    k_rows = len(n_row)
    k_cols = len(n_col)

    g1 = np.repeat(np.arange(k_rows), n_row)
    g2 = np.repeat(np.arange(k_cols), n_col)
    while True:
        rng.shuffle(g2)
        table = np.bincount(g1 * k_cols + g2, minlength=k_rows * k_cols
                            ).reshape(k_rows, k_cols)
        yield table


def simulate_table_conditional(n_row, n_col, seed=None):
    """simulate a contingency table with given margins

    This implementation cannot be vectorized and is not a very
    fast algorithm

    Parameters
    ----------
    n_row : array_like, int
        row margin, count of observations for each row
    n_row : array_like, int
        row margin, count of observations for each row

    Returns
    -------
    table : ndarray, 2-D
        random contingency table with given row and column margins

    Notes
    -----
    This implements the algorithm in Boyett 1979 using searchsorted and
    bincount.

    Boyett 1979
    """
    rng = check_random_state(seed)
    n_row = np.asarray(n_row, np.int64)
    n_col = np.asarray(n_col, np.int64)
    transpose = False
    if len(n_row) > len(n_col):
        # swap to loop over shorter dimension
        n_row, n_col = n_col, n_row
        transpose = True
    k_rows = len(n_row)
    k_cols = len(n_col)
    nobs = sum(n_row)
    assert nobs == sum(n_col)

    colcumsum = np.cumsum(np.concatenate(([0], n_col)))
    rowcumsum = np.cumsum(np.concatenate(([0], n_row)))

    x = np.arange(1, nobs + 1, dtype=np.int64)
    rng.shuffle(x)
    rvs_table = np.zeros((k_rows, k_cols), dtype=np.int64)
    for i_row in range(k_rows):
        row_int = x[rowcumsum[i_row]:rowcumsum[i_row+1]]
        # process one group/row
        # this should be same as histogram, for int
        ix = np.searchsorted(colcumsum, row_int)
        x_row = np.bincount(ix, minlength=k_cols+1)
        rvs_table[i_row] = x_row[1:]

    if transpose:
        rvs_table = rvs_table.T
    return rvs_table


def p_table(table):
    """conditional probability of a contingency table under independence

    The probability of table i

    Parameters
    ----------
    table : array_like, 2-D
        contingency table with counts in each cell

    Returns
    -------
    prob : float
        probability that table is

    Notes
    -----
    This could be extended to more than two dimensional contingency tables.


    """
    table = np.asarray(table)
    if table.ndim != 2:
        raise ValueError('table needs to be 2 dimensional')
    n_row = table.sum(1)
    n_col = table.sum(0)
    nobs = n_col.sum()
    prob = np.exp(_logfactsum(n_row) + _logfactsum(n_col) -
                  _logfactsum(nobs) - _logfactsum(table))
    return prob
