# -*- coding: utf-8 -*-
""" Measures and Diagnostics for Multicollinearity in data


Created on Tue Apr 28 15:09:45 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import statsmodels.stats.outliers_influence as smio
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly


class MultiCollinearity(object):
    """class for multicollinearity measures for an array of variables

    This treats all variables in a symmetric way, analysing each variable compared to all
    others.

    see class MultiCollinearSequential where all variables are analyzed in given sequence.

    """

    def __init__(self, data, moment_matrix=None, standardize=True):
        x = np.asarray(data)
        if hasattr(data, 'columns'):
            self.columns = data.columns
        else:
            self.columns = None

        self.k_vars = x.shape[1]
        # TODO: use pandas corrcoef below to have nan handling ?

        if moment_matrix is not None:
            xm = np.asarray(moment_matrix, copy=True)
        else:
            if standardize:
                xm = np.corrcoef(x, rowvar=0)
                if np.any(np.ptp(x, axis=0) == 0):
                    # TODO: change this? detect from xcorr, drop constant ?
                    raise ValueError('If standardize is true, then data should ' +
                                     'not include a constant')
            else:
                xm = np.dot(x.T, x)
        self.mom = xm

    @cache_readonly
    def vif(self):
        # np.diag returns read only array, need copy
        vif_ = np.diag(np.linalg.inv(self.mom)).copy()
        # It is possible that singular matrix has slightly negative eigenvalues,
        # and large negative instead of positive vif
        mask = vif_ < 1e-14
        vif_[mask] = np.inf
        return vif_

    @property
    def partial_corr(self):
        """
        this assumes standardize is true

        not cached
        """
        return 1. - 1. / self.vif


    @cache_readonly
    def eigenvalues(self):
        evals = np.sort(np.linalg.eigvalsh(self.mom))[::-1]
        # set small negative eigenvalues to zero
        mask = evals > 1e-14 & evals < 0
        evals[mask] = 0
        return evals


    @cache_readonly
    def condition_number(self):
        """Normalized condition number of data matrix.

        Calculated as ratio of largest to smallest eigenvalue of
        correlation matrix.
        """
        eigvals = self.eigenvalues
        return np.sqrt(eigvals[0] / eigvals[-1])


class MultiCollinearitySequential(MultiCollinearity):
    """class for multicollinearity measures for sequence of variables

    This takes the order of columns as relevant and calculates statistics for
    a column based on information in all previous columns.

    see class MultiCollinear where all variables are treated symmetrically.

    """

    def __init__(self, data, moment_matrix=None, standardize=True):
        x = np.asarray(data)
        if hasattr(data, 'columns'):
            self.columns = data.columns
        else:
            self.columns = None

        self.k_vars = x.shape[1]

        if moment_matrix is not None:
            triu = np.linalg.cholesky(moment_matrix).T
        else:
            if standardize:
                x = (x - x.mean(0)) / x.std(0)
                if np.any(np.ptp(x, axis=0) == 0):
                    # TODO: change this? detect from xcorr, drop constant ?
                    raise ValueError('If standardize is true, then data should ' +
                                     'not include a constant')
            triu = np.linalg.qr(x, mode='r')
        # Note: we only need elementwise squares, signs in qr are irrelevant
        self.triu2 = triu**2

    @cache_readonly
    def vif(self):
        return self.tss / self.rss

    @property
    def tss(self):
        return self.triu2.sum(0)

    @property
    def rss(self):
        return np.diag(self.triu2)

    @property
    def partial_corr(self):
        """
        this assumes standardize is true

        not cached
        """
        return 1 - self.rss / self.tss


def vif(data, standardize=True, moment_matrix=None):
    """Recursively drop variables if VIF is above threshold

    default threshold is just a number, maybe don't use a default

    Notes
    -----
    This avoids a loop to calculate the vif at a given set of variables, but
    has one matrix inverse at each set.

    TODO: replace inv by sweep algorithm,
    question: how do we start sweep in backwards selection?


    """
    # most parts are duplicate code, copied from vif_selection
    x = np.asarray(data)
    # TODO: use pandas corrcoef below to have nan handling ?

    if moment_matrix is not None:
        xm = np.asarray(moment_matrix)
        if standardize:
            # check if we have correlation matrix,
            # we cannot demean but we can scale
            # (We could demean if const is included in uncentred moment matrix)
            if (np.diag(xm) != 1).any():
                xstd = np.sqrt(np.diag(xm))
                xm /= np.outer(xstd, xstd)
    else:
        if standardize:
            xm = np.corrcoef(x, rowvar=0)
            if np.any(np.ptp(x, axis=0) == 0):
                # TODO: change this? detect from xcorr, drop constant ?
                raise ValueError('If standardize is true, then data should ' +
                                 'not include a constant')
        else:
            xm = np.dot(x.T, x)

    vif_ = np.abs(np.diag(np.linalg.inv(xm)))

    if hasattr(data, 'columns'):
        import pandas
        return pandas.Series(vif_, index=data.columns)
    else:
        return vif_


def vif_selection(data, threshold=10, standardize=True, moment_matrix=None):
    """Recursively drop variables if VIF is above threshold

    default threshold is just a number, maybe don't use a default

    Notes
    -----
    This avoids a loop to calculate the vif at a given set of variables, but
    has one matrix inverse at each set.

    TODO: replace inv by sweep algorithm,
    question: how do we start sweep in backwards selection?


    Note: This function tries to remove the last variable in case of ties in
    vif. However, which variable of those that have theoretically equal vif
    may be random because of floating point imprecision.


    """
    x = np.asarray(data)
    k_vars = x.shape[1]
    # TODO: use pandas corrcoef below to have nan handling ?

    if moment_matrix is not None:
        xm = np.asarray(moment_matrix, copy=True)
    else:
        if standardize:
            xm = np.corrcoef(x, rowvar=0)
            if np.any(np.ptp(x, axis=0) == 0):
                # TODO: change this? detect from xcorr, drop constant ?
                raise ValueError('If standardize is true, then data should ' +
                                 'not include a constant')
        else:
            xm = np.dot(x.T, x)

    # set up for recursive, itearative dropping of variables
    #we reverse data sequence to drop later variables in case of ties for max
    xidx = list(range(k_vars))[::-1]
    xc_remain = xm[::-1, ::-1]
    dropped = []
    for i in range(k_vars):
        vif = np.abs(np.diag(np.linalg.inv(xc_remain)))
        max_idx = np.argmax(vif)
        print('dropping', max_idx, vif[max_idx], xidx[max_idx])
        if vif[max_idx] <= threshold:
            break
        else:
            #keep = np.ones(len(xidx), bool)
            #keep[max_idx] = False
            keep = list(range(len(xidx)))
            keep.pop(max_idx)
            keep = np.asarray(keep)
            dropped.append(xidx.pop(max_idx))
            xc_remain = xc_remain[keep[:, None], keep]

    # None in return is placeholder for more results
    if hasattr(data, 'columns'):
        return data.columns[xidx[::-1]], None   # TODO: do we need iloc ?
    else:
        return xidx[::-1], None
