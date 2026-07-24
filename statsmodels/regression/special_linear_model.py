# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:31:14 2016

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import sparse
#import scipy.sparse.linalg as sparsela
import pandas as pd


from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools._sparse import PartialingSparse, dummy_sparse

def cat2dummy_sparse(xcat, use_pandas=False):
    """categorical to sparse dummy, use pandas for quick implementation

    xcat needs to be ndarray for now

    use_pandas is currently only for testing purposes, it uses dense intermediat dummy
    """
    # prepare np array for column iteration
    if xcat.ndim == 1:
        xcat_t = [xcat]
    else:
        xcat_t = xcat.T

    if use_pandas:
        dfs = [pd.get_dummies(xc) for xc in xcat_t]
        x = np.column_stack(dfs)[:, 1:]   # full rank
        xsp = sparse.csc_matrix(x)  #.astype(int))  # int breaks in LU-solve
    else:
        ds = [dummy_sparse(xc) for xc in xcat_t]
        xsp = sparse.hstack(ds, format='csr')[:, 1:]   # full rank

    return xsp


def _group_demean_iterative(exog_dense, groups, add_mean=True, max_iter=10,
                            atol=1e-8):
    """iteratively demean an array for two-way fixed effects

    This is intended for almost balanced panels. The data is converted
    to a 3-dimensional array with nans for missing cells.

    currently works only for two-way effects
    groups have to be integers corresponding to range(k_cati)

    no input error checking

    This function will change as more options and special cases are
    included.

    Parameters
    ----------
    exog_dense : 2d ndarray
        data with observations in rows and variables in columns.
        This array will currently not be modified.
    groups : 2d ndarray, int
        groups labels specified as consecutive integers starting at zero
    add_mean : bool
        If true (default), then the total variable means are added back into
        the group demeand exog_dense
    max_iter : int
        maximum number of iterations
    atol : float
        tolerance for convergence. Convergence is achieved if the
        maximum absolute change (np.ptp) is smaller than atol.

    Returns
    -------
    ex_dm_w : ndarray
        group demeaned exog_dense array in wide format
    ex_dm : ndarray
        group demeaned exog_dense array in long format
    it : int
        number of iterations used. If convergence has not been
        achieved then it will be equal to max_iter - 1

    """

    # with unbalanced panel

    k_cat = tuple((groups.max(0) + 1).tolist())
    xm = np.empty(exog_dense.shape[1:] + k_cat)
    xm.fill(np.nan)
    xm[:, groups[:, 0], groups[:, 1]] = exog_dense.T
    keep = ~np.isnan(xm[0]).ravel()
    finished = False
    for it in range(max_iter):
        for axis in range(1, xm.ndim):
            group_mean = np.nanmean(xm, axis=axis, keepdims=True)
            xm -= group_mean
            if np.ptp(group_mean) < atol:
                finished = True
                break
        if finished:
            break

    xd = xm.reshape(exog_dense.shape[-1], -1).T[keep]
    if add_mean:
        xmean = exog_dense.mean(0)
        xd += xmean
        xm += xmean[:, None, None]
    return xm, xd, it


class OLSAbsorb(WLS):
    """OLS model that absorbs categorical explanatory variables

    see docstring for OLS, the following only has the extra parameters for now

    Parameters
    ----------
    exog_absorb : ndarray, 1D or 2D
        categorical, factor variables that will be absorbed
    absorb_method : string, 'lu' or 'lmgres'
        method used in projection for absorbing the factor variables.
        Currently the options use either sparse LU decomposition or sparse
        `lmgres`

    Notes
    -----
    constant: the current parameterization produces a constant when the mean
    categorical effect is set to zero

    Warning: currently not all inherited methods for OLS are correct.
    Parameters and inference are correct and correspond to the full model for
    OLS that includes all factor variables as dummies with zero mean factor
    effects.
    """

    def __init__(self, endog, exog, exog_absorb, absorb_method='lu', **kwds):
        # TODO: does exog_absorb need to be a keyword for from_formula ?
        # TODO: missing handling and shape, type check for exog_absorb in super
        absorb = cat2dummy_sparse(exog_absorb)
        self.projector = PartialingSparse(absorb, method=absorb_method)
        super(OLSAbsorb, self).__init__(endog, exog, **kwds)
        # projection is in whiten

        self.k_absorb = absorb.shape[1]
        #self.df_resid -= self.k_absorb
        # inline doesn't work, df_resid is property
        self.df_resid = self.df_resid - self.k_absorb + 1
        #check this, why + 1
        self.df_model = self.df_model + self.k_absorb - 1


        self.absorb = absorb   # mainly for checking

    def whiten(self, y):
        # add the mean back in to get a constant
        # this does not reproduce the constant if fixed effect use reference encoding
        # It produces the constant if the mean fixed effect is zero
        y_mean = y.mean(0)
        return self.projector.partial_sparse(y)[1] + y_mean


    def _get_fixed_effects(self, resid):
        """temporary: recover fixed effects from regression using residuals

        Warning: This uses dense fixed effects dummies at the moment

        We add a constant to correct for constant in absorbed regression.
        This will depend on the encoding of the absorb fixed effects.

        Parameter
        ---------
        resid : ndarray
            residuals of absorbed OLS regression

        """

        exog_dummies = np.column_stack((np.ones(len(self.endog)),
                                        self.absorb.toarray()[:, :-1]))

        # [:, :-1] is just because I used it in the example, drop again

        return OLS(resid, exog_dummies).fit()
