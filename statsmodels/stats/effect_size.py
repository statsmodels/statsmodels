"""
Created on Mon Oct  5 12:36:54 2020

Author: Josef Perktold
License: BSD-3

"""
from typing import NamedTuple

import numpy as np
from scipy import special


class NoncentralityChisquareResult(NamedTuple):
    """
    Result of :func:`_noncentrality_chisquare`.

    Parameters
    ----------
    nc : float
        Zero-truncated UMVUE estimate of the noncentrality parameter.
    confint : ndarray
        Lower and upper bound of the confidence interval for `nc`.
    nc_umvue : float
        Unbiased (UMVUE), possibly negative, estimate of `nc`.
    nc_lzd : float
        Estimate of `nc` following Li, Zhang and Dai (2009).
    nc_krs : float
        Estimate of `nc` following Kubokawa, Robert and Saleh (1993).
    nc_median : float
        Estimate of `nc` as the median-unbiased estimator.
    name : str
        Description of the random variable that `nc` applies to.
    """

    nc: float
    confint: np.ndarray
    nc_umvue: float
    nc_lzd: float
    nc_krs: float
    nc_median: float
    name: str


def _noncentrality_chisquare(chi2_stat, df, alpha=0.05):
    """
    Noncentrality parameter for chi-square statistic

    `nc` is zero-truncated umvue

    Parameters
    ----------
    chi2_stat : float
        Chisquare-statistic, for example from a hypothesis test.
    df : int or float
        Degrees of freedom.
    alpha : float in (0, 1), optional
        Significance level for the confidence interval, coverage is 1 - alpha.

    Returns
    -------
    NoncentralityChisquareResult
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for ``nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of
        Noncentrality Parameters.”
        Canadian Journal of Statistics 21 (1): 45-57.
        https://doi.org/10.2307/3315657.

    .. [2] Li, Qizhai, Junjian Zhang, and Shuai Dai. 2009. “On Estimating the
        Non-Centrality Parameter of a Chi-Squared Distribution.”
        Statistics & Probability Letters 79 (1): 98-104.
        https://doi.org/10.1016/j.spl.2008.07.025.

    """
    alpha_half = alpha / 2

    nc_umvue = chi2_stat - df
    nc = np.maximum(nc_umvue, 0)
    nc_lzd = np.maximum(nc_umvue, chi2_stat / (df + 1))
    nc_krs = np.maximum(nc_umvue, chi2_stat * 2 / (df + 2))
    nc_median = special.chndtrinc(chi2_stat, df, 0.5)
    ci = special.chndtrinc(chi2_stat, df, [1 - alpha_half, alpha_half])

    res = NoncentralityChisquareResult(
        nc=nc,
        confint=ci,
        nc_umvue=nc_umvue,
        nc_lzd=nc_lzd,
        nc_krs=nc_krs,
        nc_median=nc_median,
        name="Noncentrality for chisquare-distributed random variable",
    )
    return res


class NoncentralityFResult(NamedTuple):
    """
    Result of :func:`_noncentrality_f`.

    Parameters
    ----------
    nc : float
        Zero-truncated UMVUE estimate of the noncentrality parameter.
    confint : ndarray
        Lower and upper bound of the confidence interval for `nc`.
    nc_umvue : float
        Unbiased (UMVUE), possibly negative, estimate of `nc`.
    nc_krs : float
        Estimate of `nc` following Kubokawa, Robert and Saleh (1993).
    nc_median : float
        Estimate of `nc` as the median-unbiased estimator.
    name : str
        Description of the random variable that `nc` applies to.
    """

    nc: float
    confint: np.ndarray
    nc_umvue: float
    nc_krs: float
    nc_median: float
    name: str


def _noncentrality_f(f_stat, df1, df2, alpha=0.05):
    """
    Noncentrality parameter for f statistic

    `nc` is zero-truncated umvue

    Parameters
    ----------
    f_stat : float
        f-statistic, for example from a hypothesis test.
    df1 : int or float
        Numerator degrees of freedom.
    df2 : int or float
        Denominator degrees of freedom.
    alpha : float in (0, 1), optional
        Significance level for the confidence interval, coverage is 1 - alpha.

    Returns
    -------
    NoncentralityFResult
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for ``nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of
       Noncentrality Parameters.” Canadian Journal of Statistics 21 (1): 45-57.
       https://doi.org/10.2307/3315657.
    """
    alpha_half = alpha / 2

    x_s = f_stat * df1 / df2
    nc_umvue = (df2 - 2) * x_s - df1
    nc = np.maximum(nc_umvue, 0)
    nc_krs = np.maximum(nc_umvue, x_s * 2 * (df2 - 1) / (df1 + 2))
    nc_median = special.ncfdtrinc(df1, df2, 0.5, f_stat)
    ci = special.ncfdtrinc(df1, df2, [1 - alpha_half, alpha_half], f_stat)

    res = NoncentralityFResult(
        nc=nc,
        confint=ci,
        nc_umvue=nc_umvue,
        nc_krs=nc_krs,
        nc_median=nc_median,
        name="Noncentrality for F-distributed random variable",
    )
    return res


class NoncentralityTResult(NamedTuple):
    """
    Result of :func:`_noncentrality_t`.

    Parameters
    ----------
    nc : float
        Estimate of the noncentrality parameter.
    confint : ndarray
        Lower and upper bound of the confidence interval for `nc`.
    nc_median : float
        Estimate of `nc` as the median-unbiased estimator.
    name : str
        Description of the random variable that `nc` applies to.
    """

    nc: float
    confint: np.ndarray
    nc_median: float
    name: str


def _noncentrality_t(t_stat, df, alpha=0.05):
    """
    Noncentrality parameter for t statistic

    Parameters
    ----------
    t_stat : float
        t-statistic, for example from a hypothesis test.
    df : int or float
        Degrees of freedom.
    alpha : float in (0, 1), optional
        Significance level for the confidence interval, coverage is 1 - alpha.

    Returns
    -------
    NoncentralityTResult
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for ``nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Hedges, Larry V. 2016. “Distribution Theory for Glass's Estimator of
       Effect Size and Related Estimators.”
       Journal of Educational Statistics, November.
       https://doi.org/10.3102/10769986006002107.

    """
    alpha_half = alpha / 2

    gfac = np.exp(special.gammaln(df/2.-0.5) - special.gammaln(df/2.))
    c11 = np.sqrt(df/2.) * gfac
    nc = t_stat / c11
    nc_median = special.nctdtrinc(df, 0.5, t_stat)
    ci = special.nctdtrinc(df, [1 - alpha_half, alpha_half], t_stat)

    res = NoncentralityTResult(
        nc=nc,
        confint=ci,
        nc_median=nc_median,
        name="Noncentrality for t-distributed random variable",
    )
    return res
