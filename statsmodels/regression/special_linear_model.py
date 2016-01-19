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


from statsmodels.regression.linear_model import OLS
from statsmodels.tools._sparse import PartialingSparse

def cat2dummy_sparse(xcat):
    """categorical to sparse dummy, use pandas for quick implementation

    xcat needs to be ndarray for now
    """
    # prepare np array for column iteration
    if xcat.ndim == 1:
        xcat_t = [xcat]
    else:
        xcat_t = xcat.T

    dfs = [pd.get_dummies(xc) for xc in xcat_t]
    x = np.column_stack(dfs)[:, 1:]   # full rank
    xsp = sparse.csc_matrix(x)  #.astype(int))  # int breaks in LU-solve
    return xsp


class OLSAbsorb(OLS):
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
        # projection is moved to whiten
        #self.wendog = projector.partial_sparse(self.endog)[1]
        #self.wexog = projector.partial_sparse(self.exog)[1]
        self.k_absorb = absorb.shape[1]
        #self.df_resid -= self.k_absorb
        # inline doesn't work, df_resid is property
        self.df_resid = self.df_resid - self.k_absorb + 1
        #check this, why + 1


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
