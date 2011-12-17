# -*- coding: utf-8 -*-
"""Sandwich covariance estimators


Created on Sun Nov 27 14:10:57 2011

Author: Josef Perktold
Author: Skipper Seabold for HCxxx in linear_model.RegressionResults
License: BSD-3

Notes
-----

for calculating it we have two versions

version 1: use pinv
pinv(x) scale pinv(x)   used currently in linear_model, with scale is
1d (or diagonal matrix)
(x'x)^(-1) x' scale x (x'x)^(-1),  scale in general is (nobs, nobs) so
pretty large
general formulas for scale in cluster case are in
http://pubs.amstat.org/doi/abstract/10.1198/jbes.2010.07136 which also
has the second version

version 2:
(x'x)^(-1) S (x'x)^(-1)    with S = x' scale x,    S is (kvar,kvars),
(x'x)^(-1) is available as normalized_covparams.



S = sum (x*u) dot (x*u)' = sum x*u*u'*x'  where sum here can aggregate
over observations or groups. u is regression residual.

x is (nobs, k_var)
u is (nobs, 1)
x*u is (nobs, k_var)


For cluster robust standard errors, we first sum (x*w) over other groups
(including time) and then take the dot product (sum of outer products)

S = sum_g(x*u)' dot sum_g(x*u)
For HAC by clusters, we first sum over groups for each time period, and then
use HAC on the group sums of (x*w).
If we have several groups, we have to sum first over all relevant groups, and
then take the outer product sum. This can be done by summing using indicator
functions or matrices or with explicit loops. Alternatively we calculate
separate covariance matrices for each group, sum them and subtract the
duplicate counted intersection.

Not checked in details yet: degrees of freedom or small sample correction
factors, see (two) references (?)


This is the general case for MLE and GMM also

in MLE     hessian H, outerproduct of jacobian S,   cov_hjjh = HJJH,
which reduces to the above in the linear case, but can be used
generally, e.g. in discrete, and is misnomed in GenericLikelihoodModel

in GMM it's similar but I would have to look up the details, (it comes
out in sandwich form by default, it's in the sandbox), standard Newey
West or similar are on the covariance matrix of the moment conditions

quasi-MLE: MLE with mis-specified model where parameter estimates are
fine (consistent ?) but cov_params needs to be adjusted similar or
same as in sandwiches. (I didn't go through any details yet.)

TODO
----
* small sample correction factors
* automatic lag-length selection for Newey-West HAC
* get consistent notation, varies by paper, S, scale, sigma?

"""

import numpy as np
from numpy.testing import assert_almost_equal

from scikits.statsmodels.tools.tools import (chain_dot)
from scikits.statsmodels.tools.grouputils import Group


def se_cov(cov):
    '''get standard deviation from covariance matrix

    just a shorthand function np.sqrt(np.diag(cov))

    '''
    return np.sqrt(np.diag(cov))

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
    '''
    sandwich with pinv(x) * diag(scale) * pinv(x).T

    where pinv(x) = (X'X)^(-1) X
    and scale is (nobs,)
    '''
    H = np.dot(self.model.pinv_wexog,
        scale[:,None]*self.model.pinv_wexog.T)
    return H

def cov_HC0(self):
    """
    See statsmodels.RegressionResults
    """

    het_scale = self.resid**2 # or whitened residuals? only OLS?
    cov_HC0_ = _HCCM(self, het_scale)

    return cov_HC0_

def cov_HC1(self):
    """
    See statsmodels.RegressionResults
    """

    het_scale = self.nobs/(self.df_resid)*(self.resid**2)
    cov_HC1_ = _HCCM(self, het_scale)
    return cov_HC1_

def cov_HC2(self):
    """
    See statsmodels.RegressionResults
    """

    # probably could be optimized
    h = np.diag(chain_dot(self.model.exog,
                          self.normalized_cov_params,
                          self.model.exog.T))
    het_scale = self.resid**2/(1-h)
    cov_HC2_ = _HCCM(self, het_scale)
    return cov_HC2_

def cov_HC3(self):
    """
    See statsmodels.RegressionResults
    """

    # above probably could be optimized to only calc the diag
    h = np.diag(chain_dot(self.model.exog,
                          self.normalized_cov_params,
                          self.model.exog.T))
    het_scale=(self.resid/(1-h))**2
    cov_HC3_ = _HCCM(self, het_scale)
    return cov_HC3_

#---------------------------------------


def _HCCM1(self, scale):
    '''
    sandwich with pinv(x) * scale * pinv(x).T

    where pinv(x) = (X'X)^(-1) X
    and scale is (nobs, nobs), or (nobs,) with diagonal matrix diag(scale)
    '''
    if scale.ndim == 1:
        H = np.dot(self.model.pinv_wexog,
                   scale[:,None]*self.model.pinv_wexog.T)
    else:
        H = np.dot(self.model.pinv_wexog,
                   np.dot(scale, self.model.pinv_wexog.T))
    return H

def _HCCM2(self, scale):
    '''
    sandwich with (X'X)^(-1) * scale * (X'X)^(-1)

    scale is (kvars, kvars)
    this uses self.normalized_cov_params for (X'X)^(-1)

    '''
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


def group_sums(x, group):
    '''simple bincount version, again

    group : array, integer
        assumed to be consecutive integers

    no dtype checking because I want to raise in that case

    uses loop over columns of x
    '''

    return np.array([np.bincount(group, weights=x[:,col])
                            for col in range(x.shape[1])])


def S_hac_groupsum(x, time, nlags=1, weights_func=weights_bartlett):
    '''HAC over group sums, where group is non-time dimension

    assumes we have complete equal spaced time periods.
    number of time periods per group need not be the same, but we need
    at least one observation for each time period

    I guess for a single categorical group only
    '''
    #needs groupsums

    x_group_sums = group_sums(x, time)

    return S_hac_simple(x_group_sums, nlags=nlags, weights_func=weights_func)


def S_crosssection(x, group):
    ''' check this, aggregated over groups, White on group sums

    I guess for a single categorical group only,
    categorical group, can also be the product/intersection of groups

    '''
    x_group_sums = group_sums(x, group).T  #TODO: why transposed

    return S_white_simple(x_group_sums)


def cov_crosssection_0(self, group):
    '''this one is wrong'''

    #TODO: currently used version of groupsums requires 2d resid
    scale = S_crosssection(self.resid[:,None], group)
    scale = np.squeeze(scale)
    c = _HCCM1(self, scale)
    bse = np.sqrt(np.diag(c))
    return c, bse

def cov_cluster(self, group, use_correction=True):
    '''cluster robust covariance matrix

    calculates sandwich covariance matrix for a single cluster, grouped
    variables.

    same result as Stata in UCLA example
    slightly different than Peterson

    '''
    #TODO: currently used version of groupsums requires 2d resid
    xu = self.model.exog * self.resid[:, None]
    scale = S_crosssection(xu, group)

    nobs, k_vars = self.model.exog.shape
    n_groups = len(np.unique(group)) #replace with stored group attributes if available
    if use_correction:
        corr_fact = (n_groups / (n_groups - 1.)) * ((nobs-1.) / float(nobs - k_vars))
    else:
        corr_fact = 1.
    c = corr_fact * _HCCM2(self, scale)
    bse = np.sqrt(np.diag(c))
    return c, bse

def cov_cluster_2groups(self, group, group2=None):
    '''cluster robust covariance matrix for two groups/clusters

    verified against Peterson's table, (4 decimal print precision)
    '''

    if group2 is None:
        if group.ndim !=2 or group.shape[1] != 2:
            raise ValueError('if group2 is not given, then groups needs to be ' +
                             'an array with two columns')
        group0 = group[:, 0]
        group1 = group[:, 1]
    else:
        group0 = group
        group1 = group2
        group = (group0, group1)


    cov0 = cov_cluster(self, group0)[0]  #get still bse returned also
    cov1 = cov_cluster(self, group1)[0]
    group_intersection = Group(group)
    cov01 = cov_cluster(self, group_intersection.group_int)[0]

    #robust cov matrix for union of groups
    cov_both = cov0 + cov1 - cov01

    #return all three (for now?)
    return cov_both, cov0, cov1


def cov_white_simple(self):
    xu = self.model.exog * self.resid[:, None]
    sigma = S_white_simple(xu)

    c = _HCCM2(self, sigma)  #add bread to sandwich
    bse = np.sqrt(np.diag(c))
    return c, bse

def cov_hac_simple(self, nlags=1, weights_func=weights_bartlett):
    xu = self.model.exog * self.resid[:, None]
    sigma = S_hac_simple(xu, nlags=nlags, weights_func=weights_func)

    c = _HCCM2(self, sigma)
    bse = np.sqrt(np.diag(c))
    return c, bse
