# -*- coding: utf-8 -*-
"""Nonparametric estimation of variance or residual variance

Created on Wed Jan 18 11:54:31 2017

Author: Josef Perktold
Licence: BSD-3
"""

import numpy as np
from scipy import sparse
from scipy.misc import comb


def _diff_kernel(order):
    """
    compute a differencing kernel of given order

    This is used for nonparametric variance computation for an equal spaced
    series. The kernel is normalized to square norm equal to 1.

    Parameters
    ----------
    order : int
        order of the differencing kernel

    Returns
    -------
    d : ndarray
        differencing kernel or window of length equal to order + 1
    """
    r = order
    ii = np.arange(r + 1)
    d = (-1)**ii * comb(r, ii) / np.sqrt(comb(2 * r, r))
    return d


def var_differencing(y, x=None, order=2, method=None):
    """
    nonparametric estimate of variance for data with smooth mean function

    Parameters
    ----------
    y : array_like
        series of data, currently 1-d only
    x : array_like or None
        If None, then the observations are assumed to be equal spaced in the
        underlying order.
        x is not yet available
    method : ignored
        only one available

    Returns
    -------
    var : float
        estimate of variance
    resid : ndarray
        filtered series
    todo: ??? switch to Bunch


    """
    if x is not None:
        raise NotImplementedError("unequal spacing with x is not yet available")

    if order < 1:
        raise ValueError("order must be one or larger")

    y = np.asarray(y)
    d = _diff_kernel(order)
    if method is None:
        resid = np.convolve(y, d, mode='valid')
        var = resid.dot(resid) / resid.shape[0]
    else:
        nobs = y.shape[0]
        offsets = np.arange(order + 1)
        data = np.repeat(d[:, None], nobs, axis=1)
        k_mat = sparse.dia_matrix((data, offsets), shape=(nobs-order, nobs))
        resid = k_mat.dot(y)
        var = resid.dot(resid) / resid.shape[0]

    return var, resid



############

def _create_d1_mat(nobs):
    offsets = np.array([0, 1])
    data1 = np.repeat([[-1], [1.]], nobs, axis=1)
    diff1_mat = sparse.dia_matrix((data1, offsets), shape=(nobs-1,nobs))
    return diff1_mat


class VarianceDiffProjector(object):
    """
    nonparametric estimation of variance with unequal spaced regressor.

    This uses sparse matrices for the computation.

    This is a projection or filter class and not a model class.
    The explanatory variables, `exog` are processed in init, but `endog` can be
    provided in the methods. Those endog are assumed to correspond to the
    same exog.

    Parameters
    ----------
    endog : array_like, 1-D (or None)
        dependent or response variable
    exog : array_like, 1-D
        independent variable or regressor. Only 1-D, univariate exog is
        supported
    order : int, default is 2
        differencing order, this corresponds to the choice of a bandwidth.
        This is the maximum order computed during initialization. Because of
        the recursive computation all orders up to and including the order
        given by the argument will be available

    Notes
    -----
    Status: limited testing, only equal-spaced example checked


    TODO: I'm attaching more matrices during development than we need

    """
    def __init__(self, endog, exog, order=2):
        # TODO: endog is not needed in __init__
        self.endog = np.asarray(endog)
        if self.endog.ndim != 1:
            raise ValueError("endog has to be 1-D")

        self.exog = np.asarray(exog)
        if self.exog.ndim != 1:
            raise ValueError("exog has to be 1-D")

        if order < 1:
            raise ValueError("order has to be 1 or larger")

        self.nobs = exog.shape[0]
        self.order = order
        self.dexog_mat_all = []   # (1 - F^k) x_i sparse matrix (forward)
        self.diff1_mat_all = []   # (1 - F) sparse matrix (forward)
        self.dhalf_mat_all = []   # sqrt of kernel/window matrix
        self._initialize()


    def _initialize(self):
        """Compute the required sparse projection matrices
        """
        for _ in range(self.order):
            self.increase_order()


    def increase_order(self):
        """increase order of differencing window

        this should be a public method and change internal state

        """
        #  define local alias as shortcuts
        x = self.exog
        nobs = self.nobs
        i = len(self.dhalf_mat_all) + 1   # next order

        d1_mat = _create_d1_mat(nobs - i + 1)
        self.diff1_mat_all.append(d1_mat)

        # construct k-th lead matrix  (lags are forward)
        xd_diff = x[i:] - x[:-i]
        assert len(xd_diff) == nobs - i
        xd_mat = sparse.dia_matrix((xd_diff, [0]), shape=(nobs-i,nobs-i))
        self.dexog_mat_all.append(xd_mat)

        d_mat = xd_mat.dot(d1_mat)

        if i > 1:
            d_mat = d_mat.dot(self.dhalf_mat_all[-1])

        # normalize, (squared row)sum = 1
        # Note: It's more difficult with sparse than arrays
        # can it be made simpler with elementwise multiplication?
        norm = np.sqrt(d_mat.power(2).sum(1).A.ravel())
        n = d_mat.shape[0]
        assert norm.shape == (n, )
        Dn = (sparse.dia_matrix((1. / norm, [0]), shape=[n, n]))
        d_mat = Dn.dot(d_mat)

        self.dhalf_mat_all.append(d_mat)


    def get_resid(self, endog=None, order=None):
        """Compute the residual using the window of givenorder.
        """
        if endog is None:
            endog = self.endog

        if order is None:
            D = self.dhalf_mat_all[-1]
        else:
            if order < 1 or order > len(self.dhalf_mat_all):
                raise ValueError("order need to be between 1 and %d)" %
                                 len(self.dhalf_mat_all))

            D = self.dhalf_mat_all[order - 1]

        return D.dot(endog)


    def var(self, endog=None, order=None):
        """non-parameter estimate of variance

        Paremeters
        ----------
        endog : None or array_like
            If endog is None, then the endog provided in init will be used.
            endog needs to correspond to the exog in init.
        order : None or int
            differencing order used in filtering or differencing. order needs
            to be at least 1 and at most the maximal order with pre-computed
            projection matrix.

        Returns
        -------
        var : float
            nonparametric estimate of the variance

        """
        #  TODO: should be add ddof, not in articles
        resid = self.get_resid(endog=endog, order=order)

        return resid.dot(resid) / resid.shape[0]
