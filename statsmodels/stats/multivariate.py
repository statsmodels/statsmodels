"""
Created on Sun Nov  5 14:48:19 2017

Author: Josef Perktold
License: BSD-3
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import stats

from statsmodels.stats.base import LimitedIterationMixin
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.tools.validation import array_like, int_like


# shortcut function
def _logdet(x):
    return np.linalg.slogdet(x)[1]


@dataclass(frozen=True, slots=True)
class HotellingResult(LimitedIterationMixin[float]):
    """
    Result of :func:`test_mvmean` and :func:`test_mvmean_2indep`.

    Parameters
    ----------
    statistic : float
        Hotelling's T-squared test statistic, rescaled to be F-distributed.
    pvalue : float
        p-value based on the F distribution.
    df : tuple
        Numerator and denominator degrees of freedom of the F distribution.
    t2 : float
        Hotelling's T-squared statistic.
    distribution : str
        Name of the reference distribution used for `pvalue`, ``"F"``.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    statistic: float
    pvalue: float
    df: tuple
    t2: float
    distribution: str


def test_mvmean(data, mean_null=0, return_results=True):
    """
    Hotelling's test for multivariate mean in one sample

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns
    mean_null : array_like, optional
        mean of the multivariate data under the null hypothesis
    return_results : bool, optional
        If true, then a results instance is returned. If False, then only
        the test statistic and pvalue are returned.

    Returns
    -------
    HotellingResult
        Result object with attributes statistic, pvalue, t2 and df.
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
        res = HotellingResult(
            statistic=statistic, pvalue=pvalue, df=df, t2=t2, distribution="F"
        )
        return res
    else:
        return statistic, pvalue


def test_mvmean_2indep(data1, data2):
    """
    Hotelling's test for multivariate mean in two independent samples

    The null hypothesis is that both samples have the same mean.
    The alternative hypothesis is that means differ.

    Parameters
    ----------
    data1 : array_like
        first sample data with observations in rows and variables in columns
    data2 : array_like
        second sample data with observations in rows and variables in columns

    Returns
    -------
    HotellingResult
        Result object with attributes statistic, pvalue, t2 and df.
    """
    x1 = array_like(data1, "x1", ndim=2)
    x2 = array_like(data2, "x2", ndim=2)
    nobs1, k_vars = x1.shape
    nobs2, k_vars2 = x2.shape
    if k_vars2 != k_vars:
        msg = "both samples need to have the same number of columns"
        raise ValueError(msg)
    mean1 = x1.mean(0)
    mean2 = x2.mean(0)
    cov1 = np.cov(x1, rowvar=False, ddof=1)
    cov2 = np.cov(x2, rowvar=False, ddof=1)
    nobs_t = nobs1 + nobs2
    combined_cov = ((nobs1 - 1) * cov1 + (nobs2 - 1) * cov2) / (nobs_t - 2)
    diff = mean1 - mean2
    t2 = (nobs1 * nobs2) / nobs_t * diff @ np.linalg.solve(combined_cov, diff)
    factor = ((nobs_t - 2) * k_vars) / (nobs_t - k_vars - 1)
    statistic = t2 / factor
    df = (k_vars, nobs_t - 1 - k_vars)
    pvalue = stats.f.sf(statistic, df[0], df[1])
    return HotellingResult(
        statistic=statistic, pvalue=pvalue, df=df, t2=t2, distribution="F"
    )


def confint_mvmean(data, lin_transf=None, alpha=0.05, simult=False):
    """
    Confidence interval for linear transformation of a multivariate mean

    Either pointwise or simultaneous confidence intervals are returned.

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns
    lin_transf : array_like or None, optional
        The linear transformation or contrast matrix for transforming the
        vector of means. If this is None, then the identity matrix is used
        which specifies the means themselves.
    alpha : float in (0, 1), optional
        confidence level for the confidence interval, commonly used is
        alpha=0.05.
    simult : bool, optional
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

    ci = confint_mvmean_fromstats(
        mean, cov, nobs, lin_transf=lin_transf, alpha=alpha, simult=simult
    )
    return ci


def confint_mvmean_fromstats(
    mean, cov, nobs, lin_transf=None, alpha=0.05, simult=False
):
    """
    Confidence interval for linear transformation of a multivariate mean

    Either pointwise or simultaneous confidence intervals are returned.
    Data is provided in the form of summary statistics, mean, cov, nobs.

    Parameters
    ----------
    mean : array_like
        Mean of the multivariate data.
    cov : array_like
        Covariance matrix of the multivariate data.
    nobs : int
        Number of observations used in the estimation of mean and cov.
    lin_transf : array_like or None, optional
        The linear transformation or contrast matrix for transforming the
        vector of means. If this is None, then the identity matrix is used
        which specifies the means themselves.
    alpha : float in (0, 1), optional
        confidence level for the confidence interval, commonly used is
        alpha=0.05.
    simult : bool, optional
        If simult is False (default), then pointwise confidence interval is
        returned.
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


@dataclass(frozen=True, slots=True)
class CovTestResult(LimitedIterationMixin[float]):
    """
    Result of :func:`test_cov`, :func:`test_cov_spherical`,
    :func:`test_cov_diagonal` and :func:`test_cov_blockdiagonal`.

    Parameters
    ----------
    statistic : float
        Chi-square test statistic.
    pvalue : float
        p-value based on the chi-square distribution.
    df : float
        Degrees of freedom of the chi-square distribution.
    distribution : str
        Name of the reference distribution used for `pvalue`, ``"chi2"``.
    null : str
        Short description of the null hypothesis being tested.
    cov_null : ndarray or None, optional
        Covariance matrix under the null hypothesis. Only set by
        :func:`test_cov`, otherwise None.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    statistic: float
    pvalue: float
    df: float
    distribution: str
    null: str
    cov_null: np.ndarray | None = None


def test_cov(cov, nobs, cov_null):
    """
    One sample hypothesis test for covariance equal to null covariance

    The Null hypothesis is that cov = cov_null, against the alternative that
    it is not equal to cov_null

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the data, estimated with denominator ``(N - 1)``,
        i.e., `ddof=1`.
    nobs : int
        number of observations used in the estimation of the covariance
    cov_null : array_like
        covariance under the null hypothesis

    Returns
    -------
    CovTestResult
        Result object with ``statistic, pvalue`` and other attributes like
        ``df``.

    References
    ----------
    Bartlett, M. S. 1954. “A Note on the Multiplying Factors for Various Χ2
    Approximations.” Journal of the Royal Statistical Society. Series B
    (Methodological) 16 (2): 296-98.

    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.

    """
    # using Stata formulas where cov_sample use nobs in denominator
    # Bartlett 1954 has fewer terms
    cov = array_like(cov, "cov", ndim=2)
    nobs = int_like(nobs, "nobs")
    cov_null = array_like(cov_null, "cov_null", ndim=2)

    S = cov * (nobs - 1) / nobs
    S0 = cov_null
    k = cov.shape[0]
    n = nobs

    fact = nobs - 1.0
    fact *= 1 - (2 * k + 1 - 2 / (k + 1)) / (6 * (n - 1) - 1)
    fact2 = _logdet(S0) - _logdet(n / (n - 1) * S)
    fact2 += np.trace(n / (n - 1) * np.linalg.solve(S0, S)) - k
    statistic = fact * fact2
    df = k * (k + 1) / 2
    pvalue = stats.chi2.sf(statistic, df)
    return CovTestResult(
        statistic=statistic,
        pvalue=pvalue,
        df=df,
        distribution="chi2",
        null="equal value",
        cov_null=cov_null,
    )


def test_cov_spherical(cov, nobs):
    r"""
    One sample hypothesis test that covariance matrix is spherical

    The Null and alternative hypotheses are

    .. math::

       H0 &: \Sigma = \sigma I \\
       H1 &: \Sigma \neq \sigma I

    where :math:`\sigma_i` is the common variance with unspecified value.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the data, estimated with denominator ``(N - 1)``,
        i.e., `ddof=1`.
    nobs : int
        number of observations used in the estimation of the covariance

    Returns
    -------
    CovTestResult
        Result object with ``statistic, pvalue`` and other attributes like
        ``df``.

    References
    ----------
    Bartlett, M. S. 1954. “A Note on the Multiplying Factors for Various Χ2
    Approximations.” Journal of the Royal Statistical Society. Series B
    (Methodological) 16 (2): 296-98.

    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.
    """

    # unchanged Stata formula, but denom is cov cancels, AFAICS
    # Bartlett 1954 correction factor in IIIc
    cov = np.asarray(cov)
    k = cov.shape[0]

    statistic = nobs - 1 - (2 * k**2 + k + 2) / (6 * k)
    statistic *= k * np.log(np.trace(cov)) - _logdet(cov) - k * np.log(k)
    df = k * (k + 1) / 2 - 1
    pvalue = stats.chi2.sf(statistic, df)
    return CovTestResult(
        statistic=statistic, pvalue=pvalue, df=df, distribution="chi2", null="spherical"
    )


def test_cov_diagonal(cov, nobs):
    r"""
    One sample hypothesis test that covariance matrix is diagonal matrix

    The Null and alternative hypotheses are

    .. math::

       H0 &: \Sigma = diag(\sigma_i) \\
       H1 &: \Sigma \neq diag(\sigma_i)

    where :math:`\sigma_i` are the variances with unspecified values.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the data, estimated with denominator ``(N - 1)``,
        i.e., `ddof=1`.
    nobs : int
        number of observations used in the estimation of the covariance

    Returns
    -------
    CovTestResult
        Result object with ``statistic, pvalue`` and other attributes like
        ``df``.

    References
    ----------
    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.
    """
    cov = np.asarray(cov)
    k = cov.shape[0]
    R = cov2corr(cov)

    statistic = -(nobs - 1 - (2 * k + 5) / 6) * _logdet(R)
    df = k * (k - 1) / 2
    pvalue = stats.chi2.sf(statistic, df)
    return CovTestResult(
        statistic=statistic, pvalue=pvalue, df=df, distribution="chi2", null="diagonal"
    )


def _get_blocks(mat, block_len):
    """
    Get diagonal blocks from matrix

    Parameters
    ----------
    mat : ndarray
        Square matrix.
    block_len : list of int
        List of length of each square diagonal block.

    Returns
    -------
    blocks : list of ndarray
        List of diagonal blocks extracted from mat.
    idx_blocks : list of ndarray
        List of arrays containing the indices for each block.
    """
    k = len(mat)
    idx = np.cumsum(block_len)
    if idx[-1] == k:
        idx = idx[:-1]
    elif idx[-1] > k:
        raise ValueError("sum of block_len larger than shape of mat")
    else:
        # allow one missing block that is the remainder
        pass
    idx_blocks = np.split(np.arange(k), idx)
    blocks = [mat[ii[:, None], ii] for ii in idx_blocks]
    return blocks, idx_blocks


def test_cov_blockdiagonal(cov, nobs, block_len):
    r"""
    One sample hypothesis test that covariance is block diagonal

    The Null and alternative hypotheses are

    .. math::

       H0 &: \Sigma = diag(\Sigma_i) \\
       H1 &: \Sigma \neq diag(\Sigma_i)

    where :math:`\Sigma_i` are covariance blocks with unspecified values.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the data, estimated with denominator ``(N - 1)``,
        i.e., `ddof=1`.
    nobs : int
        number of observations used in the estimation of the covariance
    block_len : list of int
        list of length of each square block

    Returns
    -------
    CovTestResult
        Result object with ``statistic, pvalue`` and other attributes like
        ``df``.

    References
    ----------
    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.
    """
    cov = np.asarray(cov)
    cov_blocks = _get_blocks(cov, block_len)[0]
    k = cov.shape[0]
    k_blocks = [c.shape[0] for c in cov_blocks]
    if k != sum(k_blocks):
        msg = "sample covariances and blocks do not have matching shape"
        raise ValueError(msg)
    logdet_blocks = sum(_logdet(c) for c in cov_blocks)
    a2 = k**2 - sum(ki**2 for ki in k_blocks)
    a3 = k**3 - sum(ki**3 for ki in k_blocks)

    statistic = nobs - 1 - (2 * a3 + 3 * a2) / (6.0 * a2)
    statistic *= logdet_blocks - _logdet(cov)

    df = a2 / 2
    pvalue = stats.chi2.sf(statistic, df)
    return CovTestResult(
        statistic=statistic,
        pvalue=pvalue,
        df=df,
        distribution="chi2",
        null="block-diagonal",
    )


@dataclass(frozen=True, slots=True)
class CovOnewayResult(LimitedIterationMixin[float]):
    """
    Result of :func:`test_cov_oneway`.

    Parameters
    ----------
    statistic : float
        Test statistic of the F-test version, same value as `statistic_f`.
    pvalue : float
        p-value of the F-test version, same value as `pvalue_f`.
    statistic_base : float
        Box's M statistic before the chi-square or F approximation
        correction factors are applied.
    statistic_chi2 : float
        Test statistic of the chi-square approximation.
    pvalue_chi2 : float
        p-value based on the chi-square approximation.
    df_chi2 : float
        Degrees of freedom of the chi-square approximation.
    distribution_chi2 : str
        Name of the chi-square reference distribution, ``"chi2"``.
    statistic_f : float
        Test statistic of the F approximation, same value as `statistic`.
    pvalue_f : float
        p-value based on the F approximation, same value as `pvalue`.
    df_f : tuple
        Numerator and denominator degrees of freedom of the F
        approximation.
    distribution_f : str
        Name of the F reference distribution, ``"F"``.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    statistic: float
    pvalue: float
    statistic_base: float
    statistic_chi2: float
    pvalue_chi2: float
    df_chi2: float
    distribution_chi2: str
    statistic_f: float
    pvalue_f: float
    df_f: tuple
    distribution_f: str


def test_cov_oneway(cov_list, nobs_list):
    r"""
    Multiple sample hypothesis test that covariance matrices are equal

    This is commonly known as Box-M test.

    The Null and alternative hypotheses are

    .. math::

       H0 &: \Sigma_i = \Sigma_j  \text{ for all i and j} \\
       H1 &: \Sigma_i \neq \Sigma_j \text{ for at least one i and j}

    where :math:`\Sigma_i` is the covariance of sample `i`.

    Parameters
    ----------
    cov_list : list of array_like
        Covariance matrices of the sample, estimated with denominator
        ``(N - 1)``, i.e., `ddof=1`.
    nobs_list : list of int
        List of the number of observations used in the estimation of the
        covariance for each sample.

    Returns
    -------
    CovOnewayResult
        Result object containing test statistic and pvalues for both chisquare
        and F distribution based tests, identified by the name ending
        "_chi2" and "_f". Attributes ``statistic, pvalue`` refer to the
        F-test version.

    Notes
    -----
    approximations to distribution of test statistic is by Box

    References
    ----------
    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.
    """
    # Note stata uses nobs in cov, this uses nobs - 1
    cov_list = [np.asarray(cov) for cov in cov_list]
    m = len(cov_list)
    nobs = sum(nobs_list)  # total number of observations
    k = cov_list[0].shape[0]

    cov_pooled = sum((n - 1) * c for (n, c) in zip(nobs_list, cov_list, strict=True))
    cov_pooled /= nobs - m
    stat0 = (nobs - m) * _logdet(cov_pooled)
    stat0 -= sum((n - 1) * _logdet(c) for (n, c) in zip(nobs_list, cov_list, strict=True))

    # Box's chi2
    c1 = sum(1 / (n - 1) for n in nobs_list) - 1 / (nobs - m)
    c1 *= (2 * k * k + 3 * k - 1) / (6 * (k + 1) * (m - 1))
    df_chi2 = (m - 1) * k * (k + 1) / 2
    statistic_chi2 = (1 - c1) * stat0
    pvalue_chi2 = stats.chi2.sf(statistic_chi2, df_chi2)

    c2 = sum(1 / (n - 1) ** 2 for n in nobs_list) - 1 / (nobs - m) ** 2
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
    return CovOnewayResult(
        statistic=statistic_f,  # name convention, using F here
        pvalue=pvalue_f,  # name convention, using F here
        statistic_base=stat0,
        statistic_chi2=statistic_chi2,
        pvalue_chi2=pvalue_chi2,
        df_chi2=df_chi2,
        distribution_chi2="chi2",
        statistic_f=statistic_f,
        pvalue_f=pvalue_f,
        df_f=df_f,
        distribution_f="F",
    )
