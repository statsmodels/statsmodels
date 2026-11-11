'''
Tools for multivariate analysis

Author : Josef Perktold
License : BSD-3

TODO:

- names of functions, currently just "working titles"

'''
from dataclasses import dataclass

import numpy as np

from statsmodels.tools.validation import array_like


@dataclass(frozen=True, slots=True)
class PartialProjectResult:
    """
    Result of :func:`partial_project`.

    Parameters
    ----------
    params : ndarray
        OLS parameter estimates from the projection of endog on exog.
    fittedvalues : ndarray
        Predicted values of endog given exog.
    resid : ndarray
        Residuals of the regression, i.e. the values of endog with the
        effect of exog partialled out.
    """

    params: np.ndarray
    fittedvalues: np.ndarray
    resid: np.ndarray


def partial_project(endog, exog):
    """
    Helper function to get linear projection or partialling out of variables

    endog variables are projected on exog variables

    Parameters
    ----------
    endog : array_like
        Array of variables where the effect of exog is partialled out.
    exog : array_like
        Array of variables on which the endog variables are projected.

    Returns
    -------
    PartialProjectResult
        Result instance with attributes ``params``, ``fittedvalues`` and
        ``resid``. See :class:`PartialProjectResult` for details.

    Notes
    -----
    This is no-frills mainly for internal calculations. ``endog`` and
    ``exog`` are converted to ndarrays, but no other error
    checking, such as verifying that both arrays have the same number of
    observations, is performed.
    """
    x1 = array_like(endog, "endog", ndim=2)
    x2 = array_like(exog, "exog", ndim=2)
    params = np.linalg.pinv(x2).dot(x1)
    predicted = x2.dot(params)
    residual = x1 - predicted

    return PartialProjectResult(params=params, fittedvalues=predicted, resid=residual)


def cancorr(x1, x2, demean=True, standardize=False):
    """
    Canonical correlation coefficient between 2 arrays

    Parameters
    ----------
    x1, x2 : array_like, 2-D
        Two 2-dimensional data arrays, observations in rows, variables in
        columns.
    demean : bool, optional
        If demean is true, then the mean is subtracted from each variable.
    standardize : bool, optional
        If standardize is true, then each variable is demeaned and divided by
        its standard deviation. Rescaling does not change the canonical
        correlation coefficients.

    Returns
    -------
    ccorr : ndarray, 1-D
        Canonical correlation coefficients, sorted from largest to smallest.
        Note, that these are the square root of the eigenvalues.

    See Also
    --------
    cc_ranktest
        Rank tests based on the smallest canonical correlations.
    cc_stats
        MANOVA statistics based on the canonical correlations.
    CCA
        Not yet implemented.

    Notes
    -----
    This is a helper function for other statistical functions. It only
    calculates the canonical correlation coefficients and does not do a full
    canonical correlation analysis.

    The canonical correlation coefficient is calculated with the generalized
    matrix inverse and does not raise an exception if one of the data arrays
    have less than full column rank.

    The eigenvalues underlying the canonical correlations are mathematically
    guaranteed to be real and non-negative, but the generalized eigenvalue
    problem is solved numerically and can return eigenvalues with a small
    spurious complex part or a small negative real part. Such numerical noise
    is clipped to zero before taking the square root, so ``ccorr`` is always
    real-valued.
    """
    x1 = array_like(x1, "x1", ndim=2)
    x2 = array_like(x2, "x2", ndim=2)

    if demean or standardize:
        x1 = x1 - x1.mean(0)
        x2 = x2 - x2.mean(0)

    if standardize:
        # std does not make a difference to canonical correlation coefficients
        x1 = x1 / x1.std(0)
        x2 = x2 / x2.std(0)

    t1 = np.linalg.pinv(x1).dot(x2)
    t2 = np.linalg.pinv(x2).dot(x1)
    m = t1.dot(t2)
    # eigvals of m are theoretically real and non-negative (they equal the
    # squared canonical correlations), but the generalized eigenvalue
    # solver can return a complex dtype with a negligible imaginary part,
    # or a tiny negative real part from floating point noise. ``.real``
    # is a no-op when the dtype is already real, so this is safe either way.
    eigval = np.clip(np.linalg.eigvals(m).real, 0, None)
    cc = np.sqrt(eigval)
    cc = np.sort(cc)[::-1]
    cc = cc[:min(x1.shape[1], x2.shape[1])]

    return cc


@dataclass(frozen=True, slots=True)
class CCRankTestResult:
    """
    Result of :func:`cc_ranktest`.

    Only returned if ``return_object=True`` is used, otherwise
    :func:`cc_ranktest` returns the equivalent plain tuple
    ``(statistic, pvalue, df, ccorr, wald_statistic, wald_pvalue)``.

    Parameters
    ----------
    statistic : float or ndarray
        Value of the LM (Anderson canonical correlations) test statistic.
    pvalue : float or ndarray
        P-value for the LM test statistic, based on the chi-square
        distribution.
    df : int or ndarray
        Degrees of freedom for the chi-square distribution in the
        hypothesis test.
    ccorr : ndarray, 1-D
        All canonical correlation coefficients sorted from largest to
        smallest.
    wald_statistic : float or ndarray
        Value of the Wald (Cragg-Donald) test statistic.
    wald_pvalue : float or ndarray
        P-value for the Wald test statistic, based on the chi-square
        distribution.
    """

    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    df: int | np.ndarray
    ccorr: np.ndarray
    wald_statistic: float | np.ndarray
    wald_pvalue: float | np.ndarray


def cc_ranktest(x1, x2, demean=True, fullrank=False, return_object=False):
    """
    Rank tests based on smallest canonical correlation coefficients

    Anderson canonical correlations test (LM test) and
    Cragg-Donald test (Wald test)
    Assumes homoskedasticity and independent observations, overrejects if
    there is heteroscedasticity or autocorrelation.

    The Null Hypothesis is that the rank is k - 1, the alternative hypothesis
    is that the rank is at least k.

    Parameters
    ----------
    x1, x2 : array_like, 2-D
        Two 2-dimensional data arrays, observations in rows, variables in
        columns.
    demean : bool, optional
        If demean is true, then the mean is subtracted from each variable.
    fullrank : bool, optional
        If true, then only the test that the matrix has full rank is returned.
        If false, the test for all possible ranks are returned. However,
        the p-values are not corrected for the multiplicity of tests.
    return_object : bool, optional
        If False (default), the results are returned as a plain tuple
        ``(statistic, pvalue, df, ccorr, wald_statistic, wald_pvalue)``, the
        legacy return format of this function. If True, a
        :class:`CCRankTestResult` instance is returned instead, which
        exposes the same values as named attributes and still unpacks as the
        same 6-tuple.

    Returns
    -------
    tuple or CCRankTestResult
        By default, the tuple ``(statistic, pvalue, df, ccorr,
        wald_statistic, wald_pvalue)``. If ``return_object`` is True, a
        :class:`CCRankTestResult` with the same values as attributes.

    See Also
    --------
    cancorr
        Canonical correlation coefficients used by this test.
    cc_stats
        MANOVA statistics based on the canonical correlations.

    Notes
    -----
    Degrees of freedom for the distribution of the test statistic are based on
    number of columns of x1 and x2 and not on their matrix rank.
    (I'm not sure yet what the interpretation of the test is if x1 or x2 are of
    reduced rank.)

    References
    ----------
    Anderson, T. W. 1951. "Estimating Linear Restrictions on Regression
    Coefficients for Multivariate Normal Distributions." The Annals of
    Mathematical Statistics 22 (3): 327-51.

    Cragg, John G., and Stephen G. Donald. 1993. "Testing Identifiability
    and Specification in Instrumental Variable Models." Econometric Theory
    9 (2): 222-40.
    """

    from scipy import stats

    x1 = array_like(x1, "x1", ndim=2)
    x2 = array_like(x2, "x2", ndim=2)
    nobs1, k1 = x1.shape
    nobs2, k2 = x2.shape

    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc * cc
    if fullrank:
        df = np.abs(k1 - k2) + 1
        value = nobs1 * cc2[-1]
        w_value = nobs1 * (cc2[-1] / (1. - cc2[-1]))
        pvalue = stats.chi2.sf(value, df)
        w_pvalue = stats.chi2.sf(w_value, df)
    else:
        r = np.arange(min(k1, k2))[::-1]
        df = (k1 - r) * (k2 - r)
        value = nobs1 * cc2[::-1].cumsum()
        w_value = nobs1 * (cc2 / (1. - cc2))[::-1].cumsum()
        pvalue = stats.chi2.sf(value, df)
        w_pvalue = stats.chi2.sf(w_value, df)

    if return_object:
        return CCRankTestResult(
            statistic=value,
            pvalue=pvalue,
            df=df,
            ccorr=cc,
            wald_statistic=w_value,
            wald_pvalue=w_pvalue,
        )
    return value, pvalue, df, cc, w_value, w_pvalue


@dataclass(frozen=True, slots=True)
class CCStatsResult:
    """
    Result of :func:`cc_stats`.

    Only returned if ``return_object=True`` is used, otherwise
    :func:`cc_stats` returns the equivalent legacy dict.

    Parameters
    ----------
    ccorr : ndarray, 1-D
        Canonical correlation coefficients, sorted from largest to smallest.
    eigenvalues : ndarray, 1-D
        Eigenvalues corresponding to ``ccorr ** 2 / (1 - ccorr ** 2)``.
    pillai_trace : float
        Pillai's Trace statistic.
    wilks_lambda : float
        Wilk's Lambda statistic.
    hotelling_trace : float
        Hotelling's Trace statistic.
    roys_largest_root : float
        Roy's Largest Root statistic.
    df_resid : float
        Residual degrees of freedom.
    df_model : float
        Model (hypothesis) degrees of freedom.
    """

    ccorr: np.ndarray
    eigenvalues: np.ndarray
    pillai_trace: float
    wilks_lambda: float
    hotelling_trace: float
    roys_largest_root: float
    df_resid: float
    df_model: float


def cc_stats(x1, x2, demean=True, return_object=False):
    """
    MANOVA statistics based on canonical correlation coefficient

    Calculates Pillai's Trace, Wilk's Lambda, Hotelling's Trace and
    Roy's Largest Root.

    Parameters
    ----------
    x1, x2 : array_like, 2-D
        Two 2-dimensional data arrays, observations in rows, variables in
        columns.
    demean : bool, optional
        If demean is true, then the mean is subtracted from each variable.
    return_object : bool, optional
        If False (default), the results are returned as a dict with the
        legacy keys used by this function ("canonical correlation
        coefficient", "eigenvalues", "Pillai's Trace", "Wilk's Lambda",
        "Hotelling's Trace", "Roy's Largest Root", "df_resid", "df_m"). If
        True, a :class:`CCStatsResult` instance is returned instead, which
        exposes the same values as named attributes.

    Returns
    -------
    dict or CCStatsResult
        By default, a dict with the legacy keys described above. If
        ``return_object`` is True, a :class:`CCStatsResult` with the same
        values as attributes.

    See Also
    --------
    cancorr : Canonical correlation coefficients used by these statistics.
    cc_ranktest : Rank tests based on the smallest canonical correlations.

    Notes
    -----
    Same as `canon` in Stata.

    Missing: F-statistics and p-values.

    Can produce nans, for example, if x1 and x2 are (numerically) singular or
    perfectly correlated, in which case a canonical correlation equals one
    and ``Hotelling's Trace`` and the underlying ``eigenvalues`` are
    infinite or not a number.
    """
    x1 = array_like(x1, "x1", ndim=2)
    x2 = array_like(x2, "x2", ndim=2)
    nobs1, k1 = x1.shape  # endogenous ?
    nobs2, k2 = x2.shape
    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc**2
    lam = (cc2 / (1 - cc2))  # what if max cc2 is 1 ?
    # Problem: ccr might not care if x1 or x2 are reduced rank,
    #          but df will depend on rank
    df_model = k1 * k2  # df_hypothesis (we do not include mean in x1, x2)
    df_resid = k1 * (nobs1 - k2 - demean)
    m = 0.5 * (df_model - k1)

    pt_value = cc2.sum()    # Pillai's trace
    wl_value = np.prod(1 / (1 + lam))   # Wilk's Lambda
    ht_value = lam.sum()    # Hotelling's Trace
    rm_value = lam.max()    # Roy's largest root
    # from scipy import stats
    # what's the distribution, the test statistic ?
    if return_object:
        return CCStatsResult(
            ccorr=cc,
            eigenvalues=lam,
            pillai_trace=pt_value,
            wilks_lambda=wl_value,
            hotelling_trace=ht_value,
            roys_largest_root=rm_value,
            df_resid=df_resid,
            df_model=m,
        )
    res = {}
    res["canonical correlation coefficient"] = cc
    res["eigenvalues"] = lam
    res["Pillai's Trace"] = pt_value
    res["Wilk's Lambda"] = wl_value
    res["Hotelling's Trace"] = ht_value
    res["Roy's Largest Root"] = rm_value
    res["df_resid"] = df_resid
    res["df_m"] = m
    return res
