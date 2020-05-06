# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:48:19 2017

Author: Josef Perktold
License: BSD-3
"""


import numpy as np
from scipy import stats

from statsmodels.stats.base import HolderTuple


def hotelling_1samp(data, mean_null=0, return_results=True):
    """Hotellings test for multivariate mean in one sample

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns
    mean_null : array_like
        mean of the multivariate data under the null hypothesis
    return_results : bool
        If true, then a results instance is returned. If False, then only
        the test statistic and pvalue are returned.

    Returns
    -------
    results : instance of a results class with attributes
        statistic, pvalue, t2 and df
    (statistic, pvalue) : tuple
        If return_results is false, then only the test statistic and the
        pvalue are returned.

    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=1)
    diff = mean - mean_null
    t2 = nobs * diff.dot(np.linalg.solve(cov, diff))
    factor = (nobs - 1) * k_vars / (nobs - k_vars)
    statistic = t2 / factor
    df = (k_vars, nobs - k_vars)
    pvalue = stats.f.sf(statistic, df[0], df[1])
    if return_results:
        res = HolderTuple(statistic=statistic,
                          pvalue=pvalue,
                          df=df,
                          t2=t2,
                          distr="F")
        return res
    else:
        return statistic, pvalue


def mv_mean_conf_int_simult(data, lin_transf=None, alpha=0.5):
    """simultaneous confidency interval for linear transformation

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns
    lin_transf : array_like or None
        The linear transformation or contrast matrix for transforming the
        vector of means. If this is None, then the identity matrix is used
        which specifies the means themselves.
    alpha : float in (0, 1)
        confidence level for the confidence interval, commonly used is
        alpha=0.05.

    Returns
    -------
    low : ndarray
        lower confidence bound on the linear transformed
    upp : ndarray
        upper confidence bound on the linear transformed
    values : ndarray
        mean or their linear transformation, center of the confidence region


    Result 5.3 page 225
    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    if lin_transf is None:
        lin_transf = np.eye(k_vars)
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=1)

    ci = mv_mean_conf_int_simult_stat(mean, cov, nobs,
                                      lin_transf=lin_transf, alpha=alpha)
    return ci


def mv_mean_conf_int_simult_stat(mean, cov, nobs, lin_transf=None, alpha=0.05):
    """simultaneous confidence interval for linear transformation

    based on summary statistic

    Parameters
    ----------
    mean
    cov
    nobs

    Result 5.3 page 225
    This looks like Sheffe simultaneous confidence intervals
    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    c = np.atleast_2d(lin_transf)
    k_vars = len(mean)

    values = c.dot(mean)
    quad_form = (c * cov.dot(c.T).T).sum(1)
    factor = (nobs - 1) * k_vars / (nobs - k_vars) / nobs
    df = (k_vars, nobs - k_vars)
    f_critval = stats.f.isf(alpha, df[0], df[1])
    ci_diff = np.sqrt(factor * quad_form * f_critval)
    low = values - ci_diff
    upp = values + ci_diff
    return low, upp, values  # , (f_critval, factor, quad_form, df)


def mv_mean_conf_int_pointwise_stat(mean, cov, nobs, lin_transf=None,
                                    alpha=0.05):
    """pointwise confidence interval for linear transformation

    based on summary statistic

    Parameters
    ----------
    mean
    cov
    nobs

    Result 5.3 page 224

    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    c = np.atleast_2d(lin_transf)

    values = c.dot(mean)
    quad_form = (c * cov.dot(c.T).T).sum(1)
    df = nobs - 1  # k_vars
    t_critval = stats.t.isf(alpha / 2, df)
    ci_diff = np.sqrt(quad_form / df) * t_critval
    low = values - ci_diff
    upp = values + ci_diff
    return low, upp, values
