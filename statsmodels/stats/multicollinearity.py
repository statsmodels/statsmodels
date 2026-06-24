# -*- coding: utf-8 -*-
""" Measures and Diagnostics for Multicollinearity in data

Created on Tue Apr 28 15:09:45 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import pandas as pd

from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.tools.decorators import cache_readonly


class MultiCollinearityBase:
    """
    Base class for multicollinearity measures
    """

    def __init__(
        self, data, moment_matrix=None, standardize=True, ridge_factor=1e-14
    ):

        if isinstance(data, pd.DataFrame):
            self.columns = data.columns
        else:
            self.columns = None
        self.ridge_factor = ridge_factor

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

            self._init_moment(xm)
        else:
            x = np.asarray(data)
            if standardize:
                if np.any(np.ptp(x, axis=0) == 0):
                    # TODO: change this? detect from xcorr, drop constant ?
                    raise ValueError(
                        "If standardize is true, then data should"
                        " not include a constant"
                    )
                x = (x - x.mean(0)) / x.std(0)

            self._init_data(x)

    def _init_moment(self, data):
        raise NotImplementedError("Subclasses must implement")

    def _init_data(self, data):
        raise NotImplementedError("Subclasses must implement")


class MultiCollinearity(MultiCollinearityBase):
    """
    Multicollinearity measures for an array of variables

    This treats all variables in a symmetric way, analysing each variable
    compared to all others.

    Parameters
    ----------
    data : {array_like, None}
        data with observations in rows and variables in columns. This is
        ignored if moment_matrix is not None.
    moment_matrix : {None, ndarray}
        Optional moment or correlation matrix to be used for the analysis.
        If it is provided then data is ignored. Default is None.
    standardize : bool, default True
        If true and data are provided, then the data will be standardized to
        mean zero and standard deviation equal to one, which is equivalent to
        using the correlation matrix.
        Inflation factor can be calculated for data that is scaled but not
        demeaned. See Notes.

    Notes
    -----
    The default treatment assumes that there is no constant in the data array
    or in the moment matrix. However, VIF is defined under the assumption of a
    constant in the corresponding regression. Use `standardize=False` if
    demeaning is not desired.

    The only case I know so far is if we have a categorical factor in the
    design matrix but no constant, that is the constant is implicit in the
    full dummy set. Using VIF with demeaned data will show that the dummy set
    is perfectly collinear (up to floating point precision).

    This class does not return pandas object if data is a pandas object.

    **Singular and near-singular cases**

    The computations are based on the matrix inverse of the moment matrix.
    This might fail or be numerically unstable in singular or near singular
    cases. `vif` uses a ridge_factor to make the moment matrix invertible
    which converts infinite vif to numerically very large but finite vif.

    Partial correlation, `corr_partial` is not well defined for some singular
    cases and might return nans or raise an exception in the matrix inverse.

    **Status**
    Good test coverage, maybe small future changes in API.
    The explicit handling of singular cases and of cases with implicit
    constant is incomplete and the behavior in those cases will likely change.

    See Also
    --------
    statsmodels.stats.multicollinearity.MultiCollinearitySequential
        Sequential treatment of multicollinearity diagnostics

    .. todo:: Consider separate demean option form standardize
    """

    def __init__(
        self, data, moment_matrix=None, standardize=True, ridge_factor=1e-14
    ):
        super(MultiCollinearity, self).__init__(
            data,
            moment_matrix=moment_matrix,
            standardize=standardize,
            ridge_factor=ridge_factor,
        )

        self.k_vars = self.mom.shape[1]

    def _init_moment(self, xm):
        self.mom = xm

    def _init_data(self, x):
        xm = x.T @ x / x.shape[0]
        self.mom = xm

    @cache_readonly
    def mom_inv(self):
        """
        Inverse of the moment matrix

        Cached, behavior in (near) singular cases comes from numpy.linalg.inv
        """
        return np.linalg.inv(self.mom)

    def get_vif(self, ridge_factor=None):
        """
        Variance inflation factor based on moment or correlation matrix.

        Parameters
        ----------
        ridge_factor : float
            A ridge factor is added to the moment matrix before inverting it.
            This can be used to avoid problems with singular moment matrices.
            By the default the ridge factor of the instance is used, but can
            be overridden here.

        Returns
        -------
        ndarray
            Variance inflation factor
        """
        if ridge_factor is None:
            ridge_factor = self.ridge_factor

        if ridge_factor == 0:
            mmi = self.mom_inv
        else:
            ridge = self.ridge_factor * np.eye(self.k_vars)
            mmi = np.linalg.inv(self.mom + ridge)

        # np.diag returns read only array, need copy
        vif_ = np.diag(mmi).copy()
        return vif_

    @cache_readonly
    def vif(self):
        """
        Variance inflation factor based on moment or correlation matrix.

        cached attribute, uses the ridge_factor in the attribute
        """
        vif_ = self.get_vif()
        # It is possible that singular matrix has slightly negative eigenvalues
        # and large negative instead of positive vif
        mask = vif_ < 1e-13
        vif_[mask] = np.inf
        return vif_

    @property
    def tss(self):
        """
        Total mean sum of squares in partial regression

        Notes
        -----
        Ones for standardized data
        """
        return np.diag(self.mom)

    @property
    def rss(self):
        """
        Residual mean sum of squares in partial regression

        This is the residual mean squared error without degrees of freedom
        correction for the partial regression when standardized data are used.
        """
        return 1 / np.diag(self.mom_inv)

    @property
    def rsquared_partial(self):
        """
        Partial correlation of one variable with all others

        This corresponds to :math:`R^2` in a linear regression of one
        variable on all others, including a constant if standardize is
        true, i.e., data was demeaned.
        """
        # direct computation, but vif is cached so we use that
        # r2p = 1 - self.rss / self.tss
        r2p = 1.0 - 1.0 / self.vif
        return r2p

    @property
    def corr_partial(self):
        """
        Partial correlation matrix

        Pairwise partial correlation of two variables conditional on remaining
        variables.

        Warning: This uses the matrix inverse of the moment matrix, which can
        produce nans or raise exceptions in singular or near-singular cases.
        """
        c = cov2corr(self.mom_inv)
        # only off diagonal elements should be negated
        np.fill_diagonal(c, -np.diag(c))
        return -c

    @cache_readonly
    def eigenvalues(self):
        """
        Eigenvalue of the correlation matrix

        Notes
        -----
        Small negative eigenvalues indicating a singular matrix are set
        to zero.
        """
        # TODO: Determine appropriate scale to use based on data
        evals = np.sort(np.linalg.eigvalsh(self.mom))[::-1]
        # set small negative eigenvalues to zero
        mask = (evals > -1e-14) & (evals < 0)
        evals[mask] = 0
        return evals

    @cache_readonly
    def condition_number(self):
        """
        Normalized condition number of data matrix.

        Calculated as ratio of largest to smallest eigenvalue of the
        correlation matrix.
        """
        eigvals = self.eigenvalues
        return np.sqrt(eigvals[0] / eigvals[-1])


class MultiCollinearitySequential(MultiCollinearity):
    """
    Multicollinearity measures for ordered sequence of variables

    This takes the order of columns as relevant and calculates statistics for
    a column based on information in all previous columns.

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
        using the correlation matrix. Inflation factor can be calculated
        for data that is scaled but not demeaned.

    Notes
    -----\n%(notes)

    See Also
    --------
    statsmodels.stats.multicollinearity.MultiCollinear
        Symmetric treatment of multicollinearity diagnostics
    """

    def __init__(self, data, moment_matrix=None, standardize=True):
        super(MultiCollinearitySequential, self).__init__(
            data, moment_matrix=moment_matrix, standardize=standardize
        )

        self.k_vars = self.triu.shape[1]  # innocent override of super
        self.triu2 = self.triu ** 2

    def _init_moment(self, xm):
        self.mom = xm
        self.triu = np.linalg.cholesky(xm).T

    def _init_data(self, x):
        self.triu = triu = np.linalg.qr(x, mode="r")
        self.mom = triu.T.dot(triu)

    @cache_readonly
    def vif(self):
        """Variance inflation factor."""
        return self.tss / self.rss

    @property
    def tss(self):
        """Total mean sum of squares in sequence of regression"""
        return self.triu2.sum(0)

    @property
    def rss(self):
        """Residual mean sum of squares in sequence of regression"""
        return np.diag(self.triu2)

    @property
    def rsquared_partial(self):
        """
        Partial correlation of one variable with all previous variables

        This corresponds to R^2 in a linear regression of one variable on all
        other variables that are before in the sequence, including a constant
        if standarize is true.
        """
        return 1 - self.rss / self.tss


def vif(data, standardize=True, moment_matrix=None, ridge_factor=1e-14):
    """
    Variance inflation factor

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
    vif : {ndarray, Series}
        Variance inflation factor. This will be a pandas Series if the data
        is a pandas DataFrame.
    """

    mc = MultiCollinearity(
        data,
        moment_matrix=moment_matrix,
        standardize=standardize,
        ridge_factor=ridge_factor,
    )
    vif_ = mc.vif
    if isinstance(data, pd.DataFrame):
        vif_ = pd.Series(vif_, index=data.columns)
    return vif_


def vif_selection(data, threshold=10, standardize=True, moment_matrix=None):
    """
    Recursively drop variables if VIF is above threshold

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

    # TODO: use pandas corrcoef below to have nan handling ?
    if moment_matrix is not None:
        xm = np.array(moment_matrix, copy=True)
    else:
        x = np.asarray(data)
        if standardize:
            if np.any(np.ptp(x, axis=0) == 0):
                # TODO: change this? detect from xcorr, drop constant ?
                raise ValueError(
                    "If standardize is true, then data should "
                    + "not include a constant"
                )
            xm = np.corrcoef(x, rowvar=0)
        else:
            xm = np.dot(x.T, x)

    k_vars = xm.shape[1]
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
    if isinstance(data, pd.DataFrame):
        return list(data.columns[xidx[::-1]]), None
    return xidx[::-1], None


def vif_ridge(corr_x, pen_factors, is_corr=True):
    """
    Variance inflation factor for Ridge regression

    Assumes penalization is on standardized variables
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

    Notes
    -----
    Could be optimized for repeated calculations for different pen_factors.
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
    """
    Find sequential index of perfectly collinear columns

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
    MutliCollinearitySequential : class with more sequential results
    """
    x = np.asarray(data)
    tol = atol + rtol * x.var(0)
    r = np.linalg.qr(x, mode="r")
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    idx_collinear = np.where(mask)[0]
    idx_keep = np.where(~mask)[0]

    if isinstance(data, pd.DataFrame):
        names = data.columns
        collinear = [names[idx] for idx in idx_collinear]
        keep = [names[idx] for idx in idx_keep]
        return collinear, keep
    return idx_collinear, idx_keep
