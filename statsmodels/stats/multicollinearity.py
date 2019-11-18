# -*- coding: utf-8 -*-
""" Measures and Diagnostics for Multicollinearity in data

Created on Tue Apr 28 15:09:45 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.tools.decorators import cache_readonly


class MultiCollinearity(object):
    """class for multicollinearity measures for an array of variables

    This treats all variables in a symmetric way, analysing each variable
    compared to all others.

    see class MultiCollinearSequential where all variables are analyzed in
    given sequence.

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns. This is
        ignored if moment_matrix is not None.
    moment_matrix : None or ndarray
        Optional moment or correlation matrix to be used for the analysis.
        If it is provided then data is ignored.
    standardize : bool, default True
        If true and data are provided, then the data will be standardized to
        mean zero and standard deviation equal to one, which is equivalent to
        using the correlation matrix. TODO: This might be replaced by separate
        demean option. Variance inflation factor can be calculated for data
        that is scaled but not demeaned.
        See Notes

    Notes
    -----

    The default treatment assumes that there is no constant in the data array
    or in the moment matrix. However, VIF is defined under the assumption of a
    constant in the corrsponding regression. Use `standardize=False` if
    demeaning is not desired.

    THe only case I know so far is if we have a categorical factor in the
    design matrix but no constant, that is the constant is implicit in the full
    dummy set. Using VIF with demeaned data will show that the dummy set is
    perfectly collinear (up to floating point precision).


    """

    def __init__(self, data, moment_matrix=None, standardize=True,
                 ridge_factor=1e-14):

        if hasattr(data, 'columns'):
            self.columns = data.columns
        else:
            self.columns = None

        if data is not None:
            x = np.asarray(data)

        # TODO: use pandas corrcoef below to have nan handling ?

        if moment_matrix is not None:
            xm = np.asarray(moment_matrix)
            if standardize:
                # check if we have correlation matrix,
                # we cannot demean but we can scale
                # (We could demean if const is included in uncentred moment
                #  matrix)
                if (np.diag(xm) != 1).any():
                    xstd = np.sqrt(np.diag(xm))
                    xm = xm / np.outer(xstd, xstd)
        else:
            if standardize:
                xm = np.corrcoef(x, rowvar=0)
                if np.any(np.ptp(x, axis=0) == 0):
                    # TODO: change this? detect from xcorr, drop constant ?
                    raise ValueError('If standardize is true, then data should'
                                     ' not include a constant')
            else:
                xm = np.dot(x.T, x) / float(x.shape[0])

        self.k_vars = xm.shape[1]
        self.mom = xm
        self.ridge_factor = ridge_factor

    @cache_readonly
    def mom_inv(self):
        return np.linalg.inv(self.mom)

    @cache_readonly
    def vif(self):
        """Variance inflation factor based on moment or correlation matrix.
        """
        ridge = self.ridge_factor * np.eye(self.k_vars)
        # np.diag returns read only array, need copy
        vif_ = np.diag(np.linalg.inv(self.mom + ridge)).copy()
        # It is possible that singular matrix has slightly negative eigenvalues
        # and large negative instead of positive vif
        mask = vif_ < 1e-13
        vif_[mask] = np.inf
        return vif_

    @property
    def rsquared_partial(self):
        """Partial correlation of one variable with all others

        This corresponds to R^2 in a linear regression of one variable on all
        others, including a constant if standarize is true.

        The interpretation as correlation assumes that standardize is true.

        not cached

        TODO: rsquared should be scale invariant
        TODO: we might want to use name `partial_corr` for partial correlation
        of pairs given all others.
        """
        return 1. - 1. / self.vif

    @property
    def corr_partial(self):
        """partial correlation matrix

        Pairwise partial correlation of two variables conditional on remaining
        variables.
        """
        return cov2corr(self.mom_inv)

    @cache_readonly
    def eigenvalues(self):
        """eigenvalue of the correlation matrix

        Note: Small negative eigenvalues indicating a singular matrix are set
        to zero.
        """
        evals = np.sort(np.linalg.eigvalsh(self.mom))[::-1]
        # set small negative eigenvalues to zero
        mask = (evals > -1e-14) & (evals < 0)
        evals[mask] = 0
        return evals

    @cache_readonly
    def condition_number(self):
        """Normalized condition number of data matrix.

        Calculated as ratio of largest to smallest eigenvalue of the
        correlation matrix.
        """
        eigvals = self.eigenvalues
        return np.sqrt(eigvals[0] / eigvals[-1])


class MultiCollinearitySequential(MultiCollinearity):
    """class for multicollinearity measures for sequence of variables

    This takes the order of columns as relevant and calculates statistics for
    a column based on information in all previous columns.

    see class MultiCollinear where all variables are treated symmetrically.

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns. This is
        ignored if moment_matrix is not None. Data can be of smaller than full
        rank.
    moment_matrix : None or ndarray
        Optional moment or correlation matrix to be used for the analysis.
        If it is provided then data is ignored.
        The moment_matrix needs to be nonsingular because Cholesky
        decomposition is used.
    standardize : bool, default True
        If true and data are provided, then the data will be standardized to
        mean zero and standard deviation equal to one, which is equivalent to
        using the correlation matrix.
        TODO: This might be replaced by separate demean option. Variance
        inflation factor can be calculated for data that is scaled but not
        demeaned.
        See Notes in class MultiCollinear

    """

    def __init__(self, data, moment_matrix=None, standardize=True):

        if hasattr(data, 'columns'):
            self.columns = data.columns
        else:
            self.columns = None

        if data is not None:
            x = np.asarray(data)

        if moment_matrix is not None:
            xm = np.asarray(moment_matrix)
            if standardize:
                # check if we have correlation matrix,
                # we cannot demean but we can scale
                # (We could demean if const is included in uncentred moment
                #  matrix)
                if (np.diag(xm) != 1).any():
                    xstd = np.sqrt(np.diag(xm))
                    xm = xm / np.outer(xstd, xstd)

            triu = np.linalg.cholesky(xm).T

        else:
            if standardize:
                x = (x - x.mean(0)) / x.std(0)
                if np.any(np.ptp(x, axis=0) == 0):
                    # TODO: change this? detect from xcorr, drop constant ?
                    raise ValueError('If standardize is true, then data should'
                                     ' not include a constant')
            triu = np.linalg.qr(x, mode='r')
        # Note: we only need elementwise squares, signs in qr are irrelevant
        self.k_vars = triu.shape[1]
        self.triu2 = triu**2

    @cache_readonly
    def vif(self):
        """Variance inflation factor.
        """
        return self.tss / self.rss

    @property
    def tss(self):
        """Total sum of squares of squares in sequence of regression"""
        return self.triu2.sum(0)

    @property
    def rss(self):
        """Residual sum of squares of squares in sequence of regression"""
        return np.diag(self.triu2)

    @property
    def rsquared_partial(self):
        """Partial correlation of one variable with all previous variables

        This corresponds to R^2 in a linear regression of one variable on all
        other variables that are before in the sequence, including a constant
        if standarize is true.

        The interpretation as correlation assumes that standardize is true.
        TODO: rsquared should be scale invariant

        not cached
        """
        return 1 - self.rss / self.tss


def vif(data, standardize=True, moment_matrix=None, ridge_factor=1e-14):
    """Variance inflation factor

    The standard interpretation requires standardize is true, or that the data
    is already standardized or that a given moment_matrix is a correlation
    matrix.

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns. This is
        ignored if moment_matrix is not None. Data can be of smaller than full
        rank.
    moment_matrix : None or ndarray
        Optional moment or correlation matrix to be used for the analysis.
        If it is providedthen data is ignored.
        The moment_matrix needs to be nonsingular for the matrix inverse. A
        small nonzero ridge factor can be used to get vif for singular
        matrices. vif for collinear variables will be very large instead of
        infinite.
    standardize : bool, default True
        If true and data are provided, then the data will be standardized to
        mean zero and standard deviation equal to one, which is equivalent to
        using the correlation matrix. TODO: This might be replaced by separate
        demean option. Variance inflation factor can be calculated for data
        that is scaled but not demeaned. See Notes in class MultiCollinear

    Return
    ------
    vif : ndarray or pandas.Series
        variance inflation factor. This will be a pandas Series if the data has
        a `columns` attribute as provided by a pandas DataFrame.

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
            if np.any(np.ptp(x, axis=0) == 0):
                # TODO: change this? detect from xcorr, drop constant ?
                raise ValueError('If standardize is true, then data should ' +
                                 'not include a constant')
            xm = np.corrcoef(x, rowvar=0)
        else:
            xm = np.dot(x.T, x) / float(x.shape[0])

    k_vars = xm.shape[1]
    ridge = ridge_factor * np.eye(k_vars)
    vif_ = np.abs(np.diag(np.linalg.inv(xm + ridge)))

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

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns. This is
        ignored if moment_matrix is not None. Data can be of smaller than full
        rank.
    moment_matrix : None or ndarray
        Optional moment or correlation matrix to be used for the analysis.
        If it is provided then data is ignored.
    standardize : bool, default True
        If true and data are provided, then the data will be standardized to
        mean zero and standard deviation equal to one, which is equivalent to
        using the correlation matrix.

        TODO: This might be replaced by separate demean option. Variance
        inflation factor can be calculated for data that is scaled but not
        demeaned.
        See Notes in class MultiCollinear

    Return
    ------
    keep : ndarray or DataFrame columns
        Indices or column names that are still included after dropping
        variables
    results : None
        currently not used, place holder for additional results

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
    # we reverse data sequence to drop later variables in case of ties for max
    xidx = list(range(k_vars))[::-1]
    xc_remain = xm[::-1, ::-1]
    dropped = []
    for _ in range(k_vars):
        vif = np.abs(np.diag(np.linalg.inv(xc_remain)))
        max_idx = np.argmax(vif)
        # print('dropping', max_idx, vif[max_idx], xidx[max_idx])
        if vif[max_idx] <= threshold:
            break
        else:
            # keep = np.ones(len(xidx), bool)
            # keep[max_idx] = False
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


def vif_ridge(corr_x, pen_factors, is_corr=True):
    """variance inflation factor for Ridge regression

    assumes penalization is on standardized variables
    data should not include a constant

    Parameters
    ----------
    corr_x : array_like
        correlation matrix if is_corr=True or original data if is_corr is False
    pen_factors : iterable
        iterable of Ridge penalization factors
    is_corr : bool
        Boolean to indicate how corr_x is interpreted, see corr_x

    Returns
    -------
    vif : ndarray
        variance inflation factors for parameters in columns and ridge
        penalization factors in rows

    could be optimized for repeated calculations for different pen_factors.
    """
    corr_x = np.asarray(corr_x)
    if not is_corr:
        corr = np.corrcoef(corr_x, rowvar=0, bias=True)
    else:
        corr = corr_x

    eye = np.eye(corr.shape[1])
    res = []
    for k in pen_factors:
        minv = np.linalg.inv(corr + k * eye)
        vif = minv.dot(corr).dot(minv)
        res.append(np.diag(vif))
    return np.asarray(res)


def collinear_index(data, atol=1e-14, rtol=1e-13):
    """find sequential index of perfectly collinear columns

    This function uses QR decomposition to detect columns that are perfectly
    collinear with earlier columns.

    Warning: This function does not include a constant, data is treated as a
    design matrix. If a constant is part of the design, then it is recommended
    to put it in the first column with for example `add_constant`.

    Parameters
    ----------
    data : array_like, 2-D
        data is assumed to have observations in rows and variables in columns.
    atol, rtol : float
        Absolute and relative tolerance for the residual sum of squares of the
        sequential regression. `rtol` is relative to the variance of the
        variable.

    Returns
    -------
    idx_collinear : array of int or string
        Index of columns that are collinear with preceding columns.
        If data has a `columns` attribute, then the names of columns are
        returned.
    idx_keep : array of int or string
        Index of columns that are not collinear with preceding columns.
        If data has a `columns` attribute, then the names of columns are
        returned.

    See Also
    --------
    `MutliCollinearitySequential` : class with more sequential results
    """
    x = np.asarray(data)
    tol = atol + rtol * x.var(0)
    r = np.linalg.qr(x, mode='r')
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    idx_collinear = np.where(mask)[0]
    idx_keep = np.where(~mask)[0]

    if not hasattr(data, 'columns'):
        return idx_collinear, idx_keep
    else:
        names = data.columns.values
        return names[idx_collinear], names[idx_keep]
