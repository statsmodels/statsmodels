# -*- coding: utf-8 -*-
"""Sandwich covariance estimators


Created on Sun Nov 27 14:10:57 2011

Author: Josef Perktold
Author: Skipper Seabold for HCxxx in linear_model.RegressionResults
License: BSD-3


"""

import numpy as np
from numpy.testing import assert_almost_equal

from scikits.statsmodels.tools.tools import (chain_dot)

#----------- from linear_model.RegressionResults
'''
    HC0_se
        White's (1980) heteroskedasticity robust standard errors.
        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        where e_i = resid[i]
        HC0_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC0, which is the full heteroskedasticity
        consistent covariance matrix and also `het_scale`, which is in
        this case just resid**2.  HCCM matrices are only appropriate for OLS.
    HC1_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as sqrt(diag(n/(n-p)*HC_0)
        HC1_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC1, which is the full HCCM and also `het_scale`,
        which is in this case n/(n-p)*resid**2.  HCCM matrices are only
        appropriate for OLS.
    HC2_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC2_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC2, which is the full HCCM and also `het_scale`,
        which is in this case is resid^(2)/(1-h_ii).  HCCM matrices are only
        appropriate for OLS.
    HC3_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC3_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC3, which is the full HCCM and also `het_scale`,
        which is in this case is resid^(2)/(1-h_ii)^(2).  HCCM matrices are
        only appropriate for OLS.

'''

def _HCCM(self, scale):
    H = np.dot(self.model.pinv_wexog,
        scale[:,None]*self.model.pinv_wexog.T)
    return H

def HC0_se(self):
    """
    See statsmodels.RegressionResults
    """
    if self._HC0_se is None:
        self.het_scale = self.resid**2 # or whitened residuals? only OLS?
        self.cov_HC0 = self._HCCM(self.het_scale)
        self._HC0_se = np.sqrt(np.diag(self.cov_HC0))
    return self._HC0_se

def HC1_se(self):
    """
    See statsmodels.RegressionResults
    """
    if self._HC1_se is None:
        self.het_scale = self.nobs/(self.df_resid)*(self.resid**2)
        self.cov_HC1 = self._HCCM(self.het_scale)
        self._HC1_se = np.sqrt(np.diag(self.cov_HC1))
    return self._HC1_se

def HC2_se(self):
    """
    See statsmodels.RegressionResults
    """
    if self._HC2_se is None:
        # probably could be optimized
        h = np.diag(chain_dot(self.model.exog,
                              self.normalized_cov_params,
                              self.model.exog.T))
        self.het_scale = self.resid**2/(1-h)
        self.cov_HC2 = self._HCCM(self.het_scale)
        self._HC2_se = np.sqrt(np.diag(self.cov_HC2))
    return self._HC2_se

def HC3_se(self):
    """
    See statsmodels.RegressionResults
    """
    if self._HC3_se is None:
        # above probably could be optimized to only calc the diag
        h = np.diag(chain_dot(self.model.exog,
                              self.normalized_cov_params,
                              self.model.exog.T))
        self.het_scale=(self.resid/(1-h))**2
        self.cov_HC3 = self._HCCM(self.het_scale)
        self._HC3_se = np.sqrt(np.diag(self.cov_HC3))
    return self._HC3_se

#---------------------------------------


def _HCCM1(self, scale):
    if scale.ndim == 1:
        H = np.dot(self.model.pinv_wexog,
                   scale[:,None]*self.model.pinv_wexog.T)
    else:
        H = np.dot(self.model.pinv_wexog,
                   np.dot(scale, self.model.pinv_wexog.T))
    return H

def _HCCM2(self, scale):
    if scale.ndim == 1:
        scale = scale[:,None]

    xxi = self.normalized_cov_params
    H = np.dot(xxi, scale).dot(xxi.T)
    return H


def weights_bartlett(nlags):
    #with lag zero
    return 1 - np.arange(nlags+1)/(nlags+1.)

def S_hac_simple(x, nlags=1, weights_func=weights_bartlett):
    '''HAC (Newey, West) with first axis consecutive time periods,

    uses Bartlett weights

    '''

    if x.ndim == 1:
        x = x[:,None]

    weights = weights_func(nlags)

    S = weights[0] * np.dot(x.T, x)  #weights[0] just for completeness, is 1

    for lag in range(1, nlags):
        s = np.dot(x[lag:].T, x[:-lag])
        S += weights[lag] * (s + s.T)

    return S

def S_white_simple(x):
    if x.ndim == 1:
        x = x[:,None]

    return np.dot(x.T, x)
