# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:48:19 2017

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats

from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.base import HolderTuple


# shortcut function
logdet = lambda x: np.linalg.slogdet(x)[1]  # noqa: E731


def test_mvmean(data, mean_null=0, return_results=True):
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


def confint_mvmean(data, lin_transf=None, alpha=0.5, simult=False):
    """Confidence interval for linear transformation of a multivariate mean

    Either pointwise or simultaneous confidence intervals are returned.

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
    simult: bool
        If ``simult`` is False (default), then the pointwise confidence
        interval is returned.
        Otherwise, a simultaneous confidence interval is returned.
        Warning: additional simultaneous confidence intervals might be added
        and the default for those might change.

    Returns
    -------
    low : ndarray
        lower confidence bound on the linear transformed
    upp : ndarray
        upper confidence bound on the linear transformed
    values : ndarray
        mean or their linear transformation, center of the confidence region

    Notes
    -----
    Pointwise confidence interval is based on Johnson and Wichern
    equation (5-21) page 224.

    Simultaneous confidence interval is based on Johnson and Wichern
    Result 5.3 page 225.
    This looks like Sheffe simultaneous confidence intervals.

    Bonferroni corrected simultaneous confidence interval might be added in
    future

    References
    ----------
    Johnson, Richard A., and Dean W. Wichern. 2007. Applied Multivariate
    Statistical Analysis. 6th ed. Upper Saddle River, N.J: Pearson Prentice
    Hall.
    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    if lin_transf is None:
        lin_transf = np.eye(k_vars)
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=0)

    ci = confint_mvmean_fromstats(mean, cov, nobs, lin_transf=lin_transf,
                                  alpha=alpha, simult=simult)
    return ci


def confint_mvmean_fromstats(mean, cov, nobs, lin_transf=None, alpha=0.05,
                             simult=False):
    """Confidence interval for linear transformation of a multivariate mean

    Either pointwise or simultaneous conficence intervals are returned.
    Data is provided in the form of summary statistics, mean, cov, nobs.

    Parameters
    ----------
    mean : ndarray
    cov : ndarray
    nobs : int
    lin_transf : array_like or None
        The linear transformation or contrast matrix for transforming the
        vector of means. If this is None, then the identity matrix is used
        which specifies the means themselves.
    alpha : float in (0, 1)
        confidence level for the confidence interval, commonly used is
        alpha=0.05.
    simult: bool
        If simult is False (default), then pointwise confidence interval is
        returned.
        Otherwise, a simultaneous confidence interval is returned.
        Warning: additional simultaneous confidence intervals might be added
        and the default for those might change.

    Notes
    -----
    Pointwise confidence interval is based on Johnson and Wichern
    equation (5-21) page 224.

    Simultaneous confidence interval is based on Johnson and Wichern
    Result 5.3 page 225.
    This looks like Sheffe simultaneous confidence intervals.

    Bonferroni corrected simultaneous confidence interval might be added in
    future

    References
    ----------
    Johnson, Richard A., and Dean W. Wichern. 2007. Applied Multivariate
    Statistical Analysis. 6th ed. Upper Saddle River, N.J: Pearson Prentice
    Hall.

    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    c = np.atleast_2d(lin_transf)
    k_vars = len(mean)

    if simult is False:
        values = c.dot(mean)
        quad_form = (c * cov.dot(c.T).T).sum(1)
        df = nobs - 1
        t_critval = stats.t.isf(alpha / 2, df)
        ci_diff = np.sqrt(quad_form / df) * t_critval
        low = values - ci_diff
        upp = values + ci_diff
    else:
        values = c.dot(mean)
        quad_form = (c * cov.dot(c.T).T).sum(1)
        factor = (nobs - 1) * k_vars / (nobs - k_vars) / nobs
        df = (k_vars, nobs - k_vars)
        f_critval = stats.f.isf(alpha, df[0], df[1])
        ci_diff = np.sqrt(factor * quad_form * f_critval)
        low = values - ci_diff
        upp = values + ci_diff

    return low, upp, values  # , (f_critval, factor, quad_form, df)


"""
Created on Tue Nov  7 13:22:44 2017

Author: Josef Perktold


References
----------
Stata manual for mvtest covariances
Rencher and Christensen 2012
Bartlett 1954

Stata refers to Rencher and Christensen for the formulas. Those correspond
to the formula collection in Bartlett 1954 for several of them.


"""  # pylint: disable=W0105


def cov_test(cov, cov_null, nobs):
    # using Stata formulas where cov_sample use nobs in denominator
    # Bartlett 1954 has fewer terms

    S = np.asarray(cov) * (nobs - 1) / nobs
    S0 = np.asarray(cov_null)
    k = cov.shape[0]
    n = nobs

    fact = nobs - 1.
    fact *= 1 - (2 * k + 1 - 2 / (k + 1)) / (6 * (n - 1) - 1)
    fact2 = logdet(S0) - logdet(n / (n - 1) * S)
    fact2 += np.trace(n / (n - 1) * np.linalg.solve(S0, S)) - k
    statistic = fact * fact2
    df = k * (k + 1) / 2
    pvalue = stats.chi2.sf(statistic, df)
    return statistic, pvalue


def cov_test_spherical(cov, nobs):
    # unchanged Stata formula, but denom is cov cancels, AFAICS
    # Bartlett 1954 correction factor in IIIc
    cov = np.asarray(cov)
    k = cov.shape[0]

    statistic = nobs - 1 - (2 * k**2 + k + 2) / (6 * k)
    statistic *= k * np.log(np.trace(cov)) - logdet(cov) - k * np.log(k)
    df = k * (k + 1) / 2 - 1
    pvalue = stats.chi2.sf(statistic, df)
    return statistic, pvalue


def cov_test_diagonal(cov, nobs):
    cov = np.asarray(cov)
    k = cov.shape[0]
    R = cov2corr(cov)

    statistic = -(nobs - 1 - (2 * k + 5) / 6) * logdet(R)
    df = k * (k - 1) / 2
    pvalue = stats.chi2.sf(statistic, df)
    return statistic, pvalue


def cov_test_blockdiagonal(cov, cov_blocks, nobs):
    cov = np.asarray(cov)
    cov_blocks = list(map(np.asarray, cov_blocks))
    k = cov.shape[0]
    k_blocks = [c.shape[0] for c in cov_blocks]
    if k != sum(k_blocks):
        msg = "sample covariances and blocks do not have matching shape"
        raise ValueError(msg)
    logdet_blocks = sum(logdet(c) for c in cov_blocks)
    a2 = k**2 - sum(ki**2 for ki in k_blocks)
    a3 = k**3 - sum(ki**3 for ki in k_blocks)

    statistic = (nobs - 1 - (2 * a3 + 3 * a2) / (6. * a2))
    statistic *= logdet_blocks - logdet(cov)

    df = a2 / 2
    pvalue = stats.chi2.sf(statistic, df)
    return statistic, pvalue


def cov_test_indep(cov_list, nobs_list):
    """

    approximations to distribution of test statistic is by Box

    """
    # TODO: name is misleading with independent cov structure
    # Note stata uses nobs in cov, this uses nobs - 1
    cov_list = list(map(np.asarray, cov_list))
    m = len(cov_list)
    nobs = sum(nobs_list)  # total number of observations
    k = cov_list[0].shape[0]

    cov_pooled = sum((n - 1) * c for (n, c) in zip(nobs_list, cov_list))
    cov_pooled /= (nobs - m)
    stat0 = (nobs - m) * logdet(cov_pooled)
    stat0 -= sum((n - 1) * logdet(c) for (n, c) in zip(nobs_list, cov_list))

    # Box's chi2
    c1 = sum(1 / (n - 1) for n in nobs_list) - 1 / (nobs - m)
    c1 *= (2 * k*k + 3 * k - 1) / (6 * (k + 1) * (m - 1))
    df_chi2 = (m - 1) * k * (k + 1) / 2
    statistic_chi2 = (1 - c1) * stat0
    pvalue_chi2 = stats.chi2.sf(statistic_chi2, df_chi2)

    c2 = sum(1 / (n - 1)**2 for n in nobs_list) - 1 / (nobs - m)**2
    c2 *= (k - 1) * (k + 2) / (6 * (m - 1))
    a1 = df_chi2
    a2 = (a1 + 2) / abs(c2 - c1**2)
    b1 = (1 - c1 - a1 / a2) / a1
    b2 = (1 - c1 + 2 / a2) / a2
    if c2 > c1**2:
        statistic_f = b1 * stat0
    else:
        tmp = b2 * stat0
        statistic_f = a2 / a1 * tmp / (1 + tmp)
    df_f = (a1, a2)
    pvalue_f = stats.f.sf(statistic_f, *df_f)
    return HolderTuple(statistic=statistic_f,  # name convention, using F here
                       pvalue=pvalue_f,   # name convention, using F here
                       statistic_base=stat0,
                       statistic_chi2=statistic_chi2,
                       pvalue_chi2=pvalue_chi2,
                       df_chi2=df_chi2,
                       distr_chi2='chi2',
                       statistic_f=statistic_f,
                       pvalue_f=pvalue_f,
                       df_f=df_f,
                       distr_f='F')
