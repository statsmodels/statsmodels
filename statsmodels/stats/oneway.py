"""
Created on Wed Mar 18 10:33:38 2020

Author: Josef Perktold
License: BSD-3

"""

from dataclasses import dataclass, fields
from typing import ClassVar, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ncfdtrinc

from statsmodels.stats.base import LimitedIterationMixin

# functions that use scipy.special instead of boost based function in stats
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.tools.validation import array_like, float_like, string_like


def effectsize_oneway(means, vars_, nobs, use_var="unequal", ddof_between=0):
    """
    Effect size corresponding to Cohen's f = nc / nobs for oneway anova

    This contains adjustment for Welch and Brown-Forsythe Anova so that
    effect size can be used with FTestAnovaPower.

    Parameters
    ----------
    means : array_like
        Mean of samples to be compared
    vars_ : float or array_like
        Residual (within) variance of each sample or pooled
        If ``vars_`` is scalar, then it is interpreted as pooled variance that
        is the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : int or array_like
        Number of observations for the samples.
        If nobs is scalar, then it is assumed that all samples have the same
        number ``nobs`` of observation, i.e., a balanced sample case.
        Otherwise, statistics will be weighted corresponding to nobs.
        Only relative sizes are relevant, any proportional change to nobs does
        not change the effect size.
    use_var : {"unequal", "equal", "bf"}, optional
        If ``use_var`` is "unequal", then the variances can differ across
        samples and the effect size for Welch anova will be computed.
    ddof_between : int, optional
        Degrees of freedom correction for the weighted between sum of squares.
        The denominator is ``nobs_total - ddof_between``
        This can be used to match differences across reference literature.

    Returns
    -------
    f2 : float
        Effect size corresponding to squared Cohen's f, which is also equal
        to the noncentrality divided by total number of observations.

    Notes
    -----
    This currently handles the following cases for oneway anova

    - balanced sample with homoscedastic variances
    - samples with different number of observations and with homoscedastic
      variances
    - samples with different number of observations and with heteroskedastic
      variances. This corresponds to Welch anova

    In the case of "unequal" and "bf" methods for unequal variances, the
    effect sizes do not directly correspond to the test statistic in Anova.
    Both have correction terms dropped or added, so the effect sizes match up
    with using FTestAnovaPower.
    If all variances are equal, then all three methods result in the same
    effect size. If variances are unequal, then the three methods produce
    small differences in effect size.

    Note, the effect size and power computation for BF Anova was not found in
    the literature. The correction terms were added so that FTestAnovaPower
    provides a good approximation to the power.

    Status: experimental
    We might add additional returns, if those are needed to support power
    and sample size applications.

    Examples
    --------
    The following shows how to compute effect size and power for each of the
    three anova methods. The null hypothesis is that the means are equal which
    corresponds to a zero effect size. Under the alternative, means differ
    with two sample means at a distance delta from the mean. We assume the
    variance is the same under the null and alternative hypothesis.

    ``nobs`` for the samples defines the fraction of observations in the
    samples. ``nobs`` in the power method defines the total sample size.

    In simulations, the computed power for standard anova,
    i.e.``use_var="equal"`` overestimates the simulated power by a few percent.
    The equal variance assumption does not hold in this example.

    >>> from statsmodels.stats.oneway import effectsize_oneway
    >>> from statsmodels.stats.power import FTestAnovaPower
    >>>
    >>> nobs = np.array([10, 12, 13, 15])
    >>> delta = 0.5
    >>> means_alt = np.array([-1, 0, 0, 1]) * delta
    >>> vars_ = np.arange(1, len(means_alt) + 1)
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="equal")
    >>> f2_alt
    0.04581300813008131
    >>>
    >>> kwds = {'effect_size': np.sqrt(f2_alt), 'nobs': 100, 'alpha': 0.05,
    ...         'k_groups': 4}
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.39165892158983273
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="unequal")
    >>> f2_alt
    0.060640138408304504
    >>>
    >>> kwds['effect_size'] = np.sqrt(f2_alt)
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.5047366512800622
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="bf")
    >>> f2_alt
    0.04391324307956788
    >>>
    >>> kwds['effect_size'] = np.sqrt(f2_alt)
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.3765792117047725

    """
    # the code here is largely a copy of onway_generic with adjustments
    use_var = string_like(
        use_var, "use_var", options=("unequal", "equal", "bf"), lower=True
    )
    means = np.asarray(means)
    n_groups = means.shape[0]

    vars_ = np.asarray(vars_)
    nobs = np.asarray(nobs)
    if np.any(vars_ < 0):
        raise ValueError("vars_ must be non-negative")
    if np.any(nobs <= 0):
        raise ValueError("nobs must be positive")

    if np.size(nobs) == 1:
        nobs = np.ones(n_groups) * nobs

    nobs_t = nobs.sum()

    if use_var == "equal":
        if np.size(vars_) == 1:
            var_resid = vars_
        else:
            var_resid = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)

        vars_ = var_resid  # scalar, if broadcasting works

    weights = nobs / vars_

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    f2 = np.dot(weights, (means - meanw_t) ** 2) / (nobs_t - ddof_between)

    if use_var.lower() == "bf":
        if np.size(vars_) == 1:
            # a scalar is the variance of all samples, the formulas below sum
            # over the samples
            vars_ = np.ones(n_groups) * vars_
        weights = nobs
        w_total = weights.sum()
        w_rel = weights / w_total
        meanw_t = w_rel @ means
        # TODO: reuse general case with weights
        tmp = ((1.0 - nobs / nobs_t) * vars_).sum()
        statistic = 1.0 * (nobs * (means - meanw_t) ** 2).sum()
        statistic /= tmp
        f2 = statistic * (1.0 - nobs / nobs_t).sum() / nobs_t
        # correction factor for df_num in BFM
        df_num2 = n_groups - 1
        df_num = tmp**2 / (
            (vars_**2).sum()
            + (nobs / nobs_t * vars_).sum() ** 2
            - 2 * (nobs / nobs_t * vars_**2).sum()
        )
        f2 *= df_num / df_num2

    return f2


class EffectSizeFsquResult(NamedTuple):
    """
    Result of :func:`convert_effectsize_fsqu`.

    Parameters
    ----------
    f2 : float or ndarray
        Squared Cohen's f effect size, the signal to noise ratio
        ``var_explained / var_residual``.
    eta2 : float or ndarray
        Squared eta effect size, the proportion of explained variance
        ``var_explained / var_total``.
    """

    f2: float
    eta2: float


def convert_effectsize_fsqu(f2=None, eta2=None):
    """
    Convert squared effect sizes in f family

    f2 is signal to noise ratio, var_explained / var_residual

    eta2 is proportion of explained variance, var_explained / var_total

    uses the relationship:
    f2 = eta2 / (1 - eta2)

    Parameters
    ----------
    f2 : None or array_like, optional
       Squared Cohen's F effect size. If f2 is not None, then eta2 will be
       computed.
    eta2 : None or array_like, optional
       Squared eta effect size. If f2 is None and eta2 is not None, then f2 is
       computed.

    Returns
    -------
    EffectSizeFsquResult
        A namedtuple with f2 and eta2 as attributes.

    """
    if f2 is not None:
        eta2 = 1 / (1 + 1 / f2)

    elif eta2 is not None:
        f2 = eta2 / (1 - eta2)

    res = EffectSizeFsquResult(f2=f2, eta2=eta2)
    return res


class FstatEffectSizeResult(NamedTuple):
    """
    Result of :func:`_fstat2effectsize`.

    Parameters
    ----------
    f2 : float or ndarray
        Squared Cohen's f effect size, ``f_stat * df1 / df2``.
    eta2 : float or ndarray
        Squared eta effect size, ``f2 / (f2 + 1)``.
    omega2 : float or ndarray
        Squared omega effect size, computed as
        ``(f2 - df1 / df2) / (f2 + 1 + 1 / df2)``.
    eps2 : float or ndarray
        Squared epsilon effect size, computed as
        ``(f2 - df1 / df2) / (f2 + 1)``.
    eps2_alt : float or ndarray
        Squared epsilon effect size, computed with the alternative
        expression ``(f_stat - 1) / (f_stat + df2 / df1)``.
    omega2_alt : float or ndarray
        Squared omega effect size, computed with the alternative
        expression ``(f_stat - 1) / (f_stat + (df2 + 1) / df1)``.
    """

    f2: np.ndarray
    eta2: np.ndarray
    omega2: np.ndarray
    eps2: np.ndarray
    eps2_alt: np.ndarray
    omega2_alt: np.ndarray


def _fstat2effectsize(f_stat, df):
    """
    Compute anova effect size from F-statistic

    This might be combined with convert_effectsize_fsqu

    Parameters
    ----------
    f_stat : array_like
        Test statistic of an F-test
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where
         - df1 : numerator degrees of freedom, number of constraints
         - df2 : denominator degrees of freedom, df_resid

    Returns
    -------
    FstatEffectSizeResult
        This namedtuple contains effect size measures f2, eta2, omega2 and
        eps2 as attributes.

    Notes
    -----
    This uses the following definitions:

    - f2 = f_stat * df1 / df2
    - eta2 = f2 / (f2 + 1)
    - omega2 = (f2 - df1 / df2) / (f2 + 2)
    - eps2 = (f2 - df1 / df2) / (f2 + 1)

    This differs from effect size measures in other function which define
    ``f2 = f_stat * df1 / nobs``
    or an equivalent expression for power computation. The noncentrality
    index for the hypothesis test is in those cases given by
    ``nc = f_stat * df1``.

    Currently omega2 and eps2 are computed in two different ways. Those
    values agree for regular cases but can show different behavior in corner
    cases (e.g., zero division).

    """
    df1, df2 = df
    f2 = f_stat * df1 / df2
    eta2 = f2 / (f2 + 1)
    omega2_alt = (f_stat - 1) / (f_stat + (df2 + 1) / df1)
    omega2 = (f2 - df1 / df2) / (f2 + 1 + 1 / df2)  # rewrite
    eps2_alt = (f_stat - 1) / (f_stat + df2 / df1)
    eps2 = (f2 - df1 / df2) / (f2 + 1)  # rewrite
    return FstatEffectSizeResult(
        f2=f2,
        eta2=eta2,
        omega2=omega2,
        eps2=eps2,
        eps2_alt=eps2_alt,
        omega2_alt=omega2_alt,
    )


# conversion functions for Wellek's equivalence effect size
# these are mainly to compare with literature


def wellek_to_f2(eps, n_groups):
    """
    Convert Wellek's effect size (sqrt) to Cohen's f-squared

    This computes the following effect size :

       f2 = 1 / n_groups * eps**2

    Parameters
    ----------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test
    n_groups : int
        Number of groups in oneway comparison

    Returns
    -------
    f2 : float or ndarray
        Effect size Cohen's f-squared.

    """
    f2 = 1 / n_groups * eps**2
    return f2


def f2_to_wellek(f2, n_groups):
    """
    Convert Cohen's f-squared to Wellek's effect size (sqrt)

    This computes the following effect size :

       eps = sqrt(n_groups * f2)

    Parameters
    ----------
    f2 : float or ndarray
        Effect size Cohen's f-squared
    n_groups : int
        Number of groups in oneway comparison

    Returns
    -------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test
    """
    eps = np.sqrt(n_groups * f2)
    return eps


def fstat_to_wellek(f_stat, n_groups, nobs_mean):
    """
    Convert F statistic to wellek's effect size eps squared

    This computes the following effect size :

       es = f_stat * (n_groups - 1) / nobs_mean

    Parameters
    ----------
    f_stat : float or ndarray
        Test statistic of an F-test.
    n_groups : int
        Number of groups in oneway comparison
    nobs_mean : float or ndarray
        Average number of observations across groups.

    Returns
    -------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test

    """
    es = f_stat * (n_groups - 1) / nobs_mean
    return es


def confint_noncentrality(f_stat, df, alpha=0.05, alternative="two-sided"):
    """
    Confidence interval for noncentrality parameter in F-test

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Parameters
    ----------
    f_stat : float
        F-statistic for which the noncentrality parameter is inverted.
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float, optional
        Significance level for the confidence interval.
    alternative : {"two-sided"}, optional
        Other alternatives have not been implemented.

    Returns
    -------
    ndarray
        Lower and upper confidence limits for the noncentrality parameter.

    See Also
    --------
    confint_effectsize_oneway

    Notes
    -----
    The algorithm inverts the cdf of the noncentral F distribution with
    respect to the noncentrality parameters.
    See Steiger 2004 and references cited in it.

    References
    ----------
    .. [1] Steiger, James H. 2004. “Beyond the F Test: Effect Size Confidence
       Intervals and Tests of Close Fit in the Analysis of Variance and
       Contrast Analysis.” Psychological Methods 9 (2): 164-82.
       https://doi.org/10.1037/1082-989X.9.2.164.
    """

    df1, df2 = df
    alternative = string_like(
        alternative,
        "alternative",
        options=("two-sided",),
        lower=False,
        deprecated={"2s": "two-sided", "ts": "two-sided"},
        removed_after="0.16",
    )
    alpha = float_like(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in the range (0, 1), got {alpha}")
    alpha1s = alpha / 2
    ci = ncfdtrinc(df1, df2, [1 - alpha1s, alpha1s], f_stat)

    return ci


class ConfintEffectSizeResult(NamedTuple):
    """
    Result of :func:`confint_effectsize_oneway`.

    Parameters
    ----------
    f2 : ndarray
        Squared Cohen's f effect size at the confidence limits of the
        noncentrality parameter.
    eta2 : ndarray
        Squared eta effect size at the confidence limits of the
        noncentrality parameter.
    ci_omega2 : ndarray
        Confidence interval for the omega-squared effect size.
    ci_nc : ndarray
        Confidence interval for the noncentrality parameter.
    ci_f : ndarray
        Confidence interval for Cohen's f effect size, ``sqrt(f2)``.
    ci_eta : ndarray
        Confidence interval for the eta effect size, ``sqrt(eta2)``.
    ci_f_corrected : ndarray
        Confidence interval for Cohen's f effect size with the small-sample
        correction used by Steiger (2004).
    """

    f2: float
    eta2: float
    ci_omega2: np.ndarray
    ci_nc: np.ndarray
    ci_f: np.ndarray
    ci_eta: np.ndarray
    ci_f_corrected: np.ndarray


def confint_effectsize_oneway(f_stat, df, alpha=0.05, nobs=None):
    """
    Confidence interval for effect size in oneway anova for F distribution

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Parameters
    ----------
    f_stat : float
        F-statistic for which the effect size confidence interval is
        computed.
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float, optional
        Significance level for the confidence interval.
    nobs : int, optional
        Total number of observations. If None, then it is set to
        ``df1 + df2 + 1``.

    Returns
    -------
    ConfintEffectSizeResult
        Namedtuple with effect size and confidence attributes.

    See Also
    --------
    confint_noncentrality

    Notes
    -----
    The confidence interval for the noncentrality parameter is obtained by
    inverting the cdf of the noncentral F distribution. Confidence intervals
    for other effect sizes are computed by endpoint transformation.

    R package ``effectsize`` does not compute the confidence intervals in the
    same way. Their confidence intervals can be replicated with

    >>> ci_nc = confint_noncentrality(f_stat, (df1, df2), alpha=0.1)
    >>> ci_es = smo._fstat2effectsize(ci_nc / df1, (df1, df2))
    """

    df1, df2 = df
    if nobs is None:
        nobs = df1 + df2 + 1
    nobs = float_like(nobs, "nobs")
    if not nobs > 0:
        raise ValueError("nobs must be positive")
    ci_nc = confint_noncentrality(f_stat, df, alpha=alpha)

    ci_f2 = ci_nc / nobs
    es = convert_effectsize_fsqu(f2=ci_f2)
    ci_res = ConfintEffectSizeResult(
        f2=es.f2,
        eta2=es.eta2,
        ci_omega2=(ci_f2 - df1 / df2) / (ci_f2 + 1 + 1 / df2),
        ci_nc=ci_nc,
        ci_f=np.sqrt(es.f2),
        ci_eta=np.sqrt(es.eta2),
        ci_f_corrected=np.sqrt(es.f2 * (df1 + 1) / df1),
    )

    return ci_res


@dataclass(frozen=True, slots=True, kw_only=True)
class AnovaResult(LimitedIterationMixin[float]):
    """
    Result of :func:`anova_generic` and :func:`anova_oneway`.

    Parameters
    ----------
    statistic : float
        Test statistic for k-sample mean comparison which is approximately
        F-distributed.
    pvalue : float
        If ``use_var="bf"``, then the p-value is based on corrected degrees
        of freedom following Mehrotra 1997.
    df : tuple
        Degrees of freedom ``(df_num, df_denom)`` for the F-distribution
        used for `pvalue`.
    df_num : float
        Numerator degrees of freedom.
    df_denom : float
        Denominator degrees of freedom used for `pvalue`.
    nobs_total : float
        Total number of observations across all samples.
    n_groups : int
        Number of samples being compared.
    means : ndarray
        Mean of each sample.
    nobs : ndarray
        Number of observations in each sample.
    vars_ : ndarray
        Residual (within) variance of each sample.
    use_var : {"unequal", "equal", "bf"}
        The `use_var` option that was used to compute the test.
    welch_correction : bool
        Whether the Welch correction was included in the test statistic.
    df2 : tuple or None
        Degrees of freedom ``(df_num2, df_denom)`` for the Brown-Forsythe
        1974 p-value. Only set if ``use_var="bf"``, otherwise None.
    df_num2 : float or None
        Numerator degrees of freedom for the Brown-Forsythe 1974 p-value.
        Only set if ``use_var="bf"``, otherwise None.
    pvalue2 : float or None
        p-value based on degrees of freedom as in Brown-Forsythe 1974.
        Only set if ``use_var="bf"``, otherwise None.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    statistic: float
    pvalue: float
    df: tuple
    df_num: float
    df_denom: float
    nobs_total: float
    n_groups: int
    means: np.ndarray
    nobs: np.ndarray
    vars_: np.ndarray
    use_var: str
    welch_correction: bool
    df2: tuple | None = None
    df_num2: float | None = None
    pvalue2: float | None = None


def anova_generic(
    means, variances, nobs, use_var="unequal", welch_correction=True, info=None
):
    """
    Oneway Anova based on summary statistics

    Parameters
    ----------
    means : ndarray
        Mean of samples to be compared
    variances : float or array_like
        Residual (within) variance of each sample or pooled.
        If ``variances`` is scalar, then it is interpreted as pooled variance
        that is the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : ndarray
        Number of observations for each sample. Statistics are weighted
        corresponding to nobs. Only relative sizes are relevant, any
        proportional change to nobs does not change the effect size.
    use_var : {"unequal", "equal", "bf"}, optional
        If ``use_var`` is "unequal", then the variances can differ across
        samples and the effect size for Welch anova will be computed.
    welch_correction : bool, optional
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    info : optional
        Not used yet.

    Returns
    -------
    AnovaResult
        This includes `statistic` and `pvalue`.

    """
    use_var = string_like(
        use_var, "use_var", options=("unequal", "equal", "bf"), lower=True
    )
    options = {"use_var": use_var, "welch_correction": welch_correction}
    if means.ndim != 1:
        raise ValueError("data (means, ...) has to be one-dimensional")
    nobs_t = nobs.sum()
    n_groups = len(means)
    # mean_t = (nobs * means).sum() / nobs_t
    if use_var == "unequal":
        weights = nobs / variances
    else:
        weights = nobs

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    statistic = np.dot(weights, (means - meanw_t) ** 2) / (n_groups - 1.0)
    df_num = n_groups - 1.0

    if use_var == "unequal":
        tmp = ((1 - w_rel) ** 2 / (nobs - 1)).sum() / (n_groups**2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1.0 / (3.0 * tmp)

    elif use_var == "equal":
        # variance of group demeaned total sample, pooled var_resid
        tmp = ((nobs - 1) * variances).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = nobs_t - n_groups

    elif use_var == "bf":
        tmp = ((1.0 - nobs / nobs_t) * variances).sum()
        statistic = 1.0 * (nobs * (means - meanw_t) ** 2).sum()
        statistic /= tmp

        df_num2 = n_groups - 1
        df_denom = (
            tmp**2 / ((1.0 - nobs / nobs_t) ** 2 * variances**2 / (nobs - 1)).sum()
        )
        df_num = tmp**2 / (
            (variances**2).sum()
            + (nobs / nobs_t * variances).sum() ** 2
            - 2 * (nobs / nobs_t * variances**2).sum()
        )
        pval2 = stats.f.sf(statistic, df_num2, df_denom)
        options["df2"] = (df_num2, df_denom)
        options["df_num2"] = df_num2
        options["pvalue2"] = pval2

    else:
        raise ValueError('use_var is to be one of "unequal", "equal" or "bf"')

    pval = stats.f.sf(statistic, df_num, df_denom)
    res = AnovaResult(
        statistic=statistic,
        pvalue=pval,
        df=(df_num, df_denom),
        df_num=df_num,
        df_denom=df_denom,
        nobs_total=nobs_t,
        n_groups=n_groups,
        means=means,
        nobs=nobs,
        vars_=variances,
        **options,
    )
    return res


def anova_oneway(
    data, groups=None, use_var="unequal", welch_correction=True, trim_frac=0
):
    """
    Oneway Anova

    This implements standard anova, Welch and Brown-Forsythe, and trimmed
    (Yuen) variants of those.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2.
        The data can be provided as a tuple or list of arrays or in long
        format with outcome observations in ``data`` and group membership in
        ``groups``.
    groups : ndarray or Series, optional
        If data is in long format, then groups is needed as indicator to which
        group or sample and observations belongs.
    use_var : {"unequal", "equal", "bf"}, optional
        `use_var` specified how to treat heteroscedasticity, unequal variance,
        across samples. Three approaches are available

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    welch_correction : bool, optional
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    trim_frac : float in [0, 0.5), optional
        Optional trimming for Anova with trimmed mean and winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        The number of trimmed observations is the fraction of number of
        observations in the sample truncated to the next lower integer.
        `trim_frac` has to be smaller than 0.5, however, if the fraction is
        so large that there are not enough observations left over, then `nan`
        will be returned.

    Returns
    -------
    AnovaResult
        See :class:`AnovaResult` for a description of the attributes. If
        ``use_var="bf"``, then `pvalue` is based on corrected degrees of
        freedom following Mehrotra 1997, and `pvalue2`, `df2` and `df_num2`
        hold the Brown-Forsythe 1974 alternative; otherwise those three are
        None.

    See Also
    --------
    anova_generic

    Notes
    -----
    Welch's anova is correctly sized (not liberal or conservative) in smaller
    samples if the distribution of the samples is not very far away from the
    normal distribution. The test can become liberal if the data is strongly
    skewed. Welch's Anova can also be correctly sized for discrete
    distributions with finite support, like Likert scale data.
    The trimmed version is robust to many non-normal distributions, it stays
    correctly sized in many cases, and is more powerful in some cases with
    skewness or heavy tails.

    Trimming is currently based on the integer part of ``nobs * trim_frac``.
    The default might change to including fractional observations as in the
    original articles by Yuen.

    References
    ----------
    Brown, Morton B., and Alan B. Forsythe. 1974. “The Small Sample Behavior
    of Some Statistics Which Test the Equality of Several Means.”
    Technometrics 16 (1) (February 1): 129-132. doi:10.2307/1267501.

    Mehrotra, Devan V. 1997. “Improving the Brown-Forsythe Solution to the
    Generalized Behrens-Fisher Problem.” Communications in Statistics -
    Simulation and Computation 26 (3): 1139-1145.
    doi:10.1080/03610919708813431.
    """
    if groups is not None:
        uniques = np.unique(groups)
        data = [data[groups == uni] for uni in uniques]
    else:
        # uniques = None  # not used yet, add to info?
        pass
    args = [np.asarray(x) for x in data]
    if any(x.ndim != 1 for x in args):
        raise ValueError("data arrays have to be one-dimensional")

    nobs = np.array([len(x) for x in args], float)
    # n_groups = len(args)  # not used
    # means = np.array([np.mean(x, axis=0) for x in args], float)
    # vars_ = np.array([np.var(x, ddof=1, axis=0) for x in args], float)

    if trim_frac == 0:
        means = np.array([x.mean() for x in args])
        vars_ = np.array([x.var(ddof=1) for x in args])
    else:
        tms = [TrimmedMean(x, trim_frac) for x in args]
        means = np.array([tm.mean_trimmed for tm in tms])
        # R doesn't use uncorrected var_winsorized
        # vars_ = np.array([tm.var_winsorized for tm in tms])
        vars_ = np.array(
            [tm.var_winsorized * (tm.nobs - 1) / (tm.nobs_reduced - 1) for tm in tms]
        )
        # nobs_original = nobs  # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    res = anova_generic(
        means, vars_, nobs, use_var=use_var, welch_correction=welch_correction
    )

    return res


@dataclass(frozen=True, slots=True)
class IccOnewayResult:
    """
    Result of :func:`icc_oneway`.

    .. versionadded:: 0.15.1

    Parameters
    ----------
    icc : float
        Estimate of the intraclass correlation coefficient.
    conf_int : tuple
        Lower and upper limit of the ``1 - alpha`` confidence interval for
        the intraclass correlation coefficient.
    statistic : float
        F statistic of the oneway Anova, the ratio of the mean squares
        between and within groups.
    pvalue : float
        The p-value of the F test of the null hypothesis that the intraclass
        correlation is zero against the alternative that it is positive.
    df : tuple
        Degrees of freedom ``(df_num, df_denom)`` of the F distribution.
    var_between : float
        Estimate of the variance of the random group effects.
    var_within : float
        Estimate of the variance within groups, the mean square within
        groups.
    nobs_mean : float
        Group size ``k0`` that is used in the estimate of the variance
        between groups. It is the common group size if all groups have the
        same size.
    n_groups : int
        Number of groups.
    nobs_total : int
        Total number of observations.
    alpha : float
        Significance level of the confidence interval.
    """

    icc: float
    conf_int: tuple
    statistic: float
    pvalue: float
    df: tuple
    var_between: float
    var_within: float
    nobs_mean: float
    n_groups: int
    nobs_total: int
    alpha: float


def icc_oneway(data, groups=None, alpha=0.05):
    r"""
    Compute the intraclass correlation of the oneway random effects model.

    .. versionadded:: 0.15.1

    Parameters
    ----------
    data : tuple of array_like or array_like
        Data for two or more groups, either as a tuple or list of
        one-dimensional arrays, one for each group, or in long format with
        all observations in ``data`` and the group of each observation in
        ``groups``. If ``groups`` is None, then a two-dimensional array or
        DataFrame has one row for each group, for example the ratings of a
        subject in each row.
    groups : array_like, optional
        Group of each observation if ``data`` is in long format.
    alpha : float, optional
        Significance level for the confidence interval, default is 0.05.

    Returns
    -------
    IccOnewayResult
        See :class:`IccOnewayResult` for a description of the attributes.

    Raises
    ------
    ValueError
        If there are fewer than two groups, if a group has no observation,
        if no group has two or more observations, if the data are not finite,
        if ``groups`` has missing values, if the data of a group are not
        one-dimensional, if ``data`` and ``groups`` do not have the same
        length, or if ``alpha`` is not in (0, 1).

    See Also
    --------
    anova_oneway : Oneway Anova with the same F statistic if
        ``use_var="equal"``.

    Notes
    -----
    In the oneway random effects model, the observation ``j`` of group ``i``
    is :math:`y_{ij} = \mu + a_i + e_{ij}`, where the group effects
    :math:`a_i` and the errors :math:`e_{ij}` are independent with variances
    :math:`\sigma_a^2` and :math:`\sigma_e^2`. The intraclass correlation
    is the correlation of two observations of the same group,
    :math:`\sigma_a^2 / (\sigma_a^2 + \sigma_e^2)`.

    The variances are estimated with the mean squares between groups,
    :math:`MSB`, and within groups, :math:`MSW`, of the oneway Anova
    (Donner 1986),

    .. math::

        \hat\sigma_e^2 = MSW, \quad
        \hat\sigma_a^2 = (MSB - MSW) / k_0, \quad
        k_0 = \frac{1}{g - 1} \left(N - \sum_i n_i^2 / N\right)

    where :math:`g` is the number of groups, :math:`n_i` the size of group
    :math:`i` and :math:`N` the total number of observations. The estimate of
    the variance between groups and of the intraclass correlation is negative
    if :math:`MSB < MSW`. It is not truncated at zero.

    The confidence interval is based on the F distribution of
    :math:`F = MSB / MSW`. With :math:`F_L = F / f_{1 - \alpha / 2}(g - 1,
    N - g)` and :math:`F_U = F \, f_{1 - \alpha / 2}(N - g, g - 1)`, where
    :math:`f_q` is the :math:`q` quantile of the F distribution, the limits
    are :math:`(F_L - 1) / (F_L + k_0 - 1)` and
    :math:`(F_U - 1) / (F_U + k_0 - 1)`. If all groups have the same size,
    then these are the exact limits under normality of Searle (1971). If the
    groups have different sizes, then :math:`k_0` replaces the common group
    size and the interval is approximate.

    If there is no variation within groups but some between groups, then
    the intraclass correlation and both limits are 1, the F statistic is
    ``inf`` and the p-value is 0. If all observations are equal, then the
    intraclass correlation, the F statistic, the p-value and the limits are
    ``nan``.

    References
    ----------
    .. [*] Donner, Allan. 1986. "A Review of Inference Procedures for the
       Intraclass Correlation Coefficient in the One-Way Random Effects
       Model." International Statistical Review 54 (1): 67-82.
       https://doi.org/10.2307/1403259
    .. [*] Searle, S. R. 1971. Linear Models. New York: Wiley.

    Examples
    --------
    Six subjects are rated four times each.

    >>> import numpy as np
    >>> from statsmodels.stats.oneway import icc_oneway
    >>> ratings = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
    ...                     [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    >>> res = icc_oneway(ratings)
    >>> round(res.icc, 4)
    0.1657
    >>> [round(v, 4) for v in res.conf_int]
    [-0.1329, 0.7226]
    """
    alpha = float_like(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in the range (0, 1), got {alpha}")
    if groups is not None:
        data = array_like(data, "data", ndim=1)
        groups = array_like(groups, "groups", ndim=1, dtype=None)
        if groups.shape != data.shape:
            raise ValueError("data and groups must have the same length")
        if pd.isna(groups).any():
            raise ValueError("groups must not have missing values")
        args = [data[groups == uni] for uni in np.unique(groups)]
    else:
        if getattr(data, "ndim", None) == 2:
            # DataFrame or 2-d array, one row for each group
            data = np.asarray(data, dtype=float)
        args = [array_like(x, "data of a group", ndim=1) for x in data]
    nobs = np.array([len(x) for x in args])
    n_groups = len(args)
    nobs_total = int(nobs.sum())
    if n_groups < 2:
        raise ValueError("at least two groups are required")
    if np.any(nobs == 0):
        raise ValueError("each group needs at least one observation")
    if nobs_total == n_groups:
        raise ValueError("at least one group needs two or more observations")
    if not all(np.all(np.isfinite(x)) for x in args):
        raise ValueError("data must be finite")

    df_num = n_groups - 1
    df_denom = nobs_total - n_groups
    nobs_mean = (nobs_total - (nobs**2).sum() / nobs_total) / df_num
    means = np.array([x.mean() for x in args])
    mean_total = np.concatenate(args).mean()
    ms_between = (nobs * (means - mean_total) ** 2).sum() / df_num

    # The cases without variation are detected from the data, because the
    # sums of squares can have rounding errors if all values are equal.
    if np.ptp(np.concatenate(args)) == 0:
        # all observations are equal, the correlation is 0 / 0
        var_between = var_within = 0.0
        icc = f_stat = pvalue = np.nan
        conf_int = (np.nan, np.nan)
    elif all(np.ptp(x) == 0 for x in args):
        # no variation within groups
        var_within = 0.0
        var_between = ms_between / nobs_mean
        icc = 1.0
        f_stat = np.inf
        pvalue = 0.0
        conf_int = (1.0, 1.0)
    else:
        ss_within = sum(((x - m) ** 2).sum() for x, m in zip(args, means, strict=True))
        var_within = ss_within / df_denom
        var_between = (ms_between - var_within) / nobs_mean
        icc = var_between / (var_between + var_within)
        f_stat = ms_between / var_within
        pvalue = stats.f.sf(f_stat, df_num, df_denom)
        f_low = f_stat / stats.f.isf(alpha / 2, df_num, df_denom)
        f_upp = f_stat * stats.f.isf(alpha / 2, df_denom, df_num)
        conf_int = (
            (f_low - 1) / (f_low + nobs_mean - 1),
            (f_upp - 1) / (f_upp + nobs_mean - 1),
        )

    return IccOnewayResult(
        icc=float(icc),
        conf_int=tuple(float(v) for v in conf_int),
        statistic=float(f_stat),
        pvalue=float(pvalue),
        df=(df_num, df_denom),
        var_between=float(var_between),
        var_within=float(var_within),
        nobs_mean=float(nobs_mean),
        n_groups=n_groups,
        nobs_total=nobs_total,
        alpha=alpha,
    )


@dataclass(frozen=True, slots=True)
class EquivalenceOnewayResult(LimitedIterationMixin[float]):
    """
    Result of :func:`equivalence_oneway_generic` and :func:`equivalence_oneway`.

    Parameters
    ----------
    statistic : float
        F-statistic used for the equivalence test, same value as `f_stat`.
    pvalue : float
        p-value of the equivalence test.
    effectsize : float
        Estimated effect size, in the units given by `type_effectsize`.
    crit_f : float
        Critical value for the F-statistic at the boundary of the
        equivalence margin.
    crit_es : float
        Critical value for the effect size at the boundary of the
        equivalence margin, in the same units as `effectsize`.
    reject : bool
        True if the null hypothesis of non-equivalence is rejected, i.e.
        the data supports equivalence within `equiv_margin`.
    power_zero : float
        Power of the equivalence test if the true effect size were zero.
    df : tuple
        Degrees of freedom ``(df1, df2)`` of the F-distribution used for
        the test.
    f_stat : float
        F-statistic used for the equivalence test, same value as
        `statistic`.
    type_effectsize : str
        Description of the effect size type used for `effectsize` and
        `crit_es`, determined by `margin_type`.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    statistic: float
    pvalue: float
    effectsize: float
    crit_f: float
    crit_es: float
    reject: bool
    power_zero: float
    df: tuple
    f_stat: float
    type_effectsize: str


def equivalence_oneway_generic(
    f_stat, n_groups, nobs, equiv_margin, df, alpha=0.05, margin_type="f2"
):
    """
    Equivalence test for oneway anova (Wellek and extensions)

    This is an helper function when summary statistics are available.
    Use `equivalence_oneway` instead.

    The null hypothesis is that the means differ by more than `equiv_margin`
    in the anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e., within a given distance.

    Parameters
    ----------
    f_stat : float
        F-statistic
    n_groups : int
        Number of groups in oneway comparison.
    nobs : ndarray
        Array of number of observations in groups.
    equiv_margin : float or array_like
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float in (0, 1), optional
        Significance level for the hypothesis test.
    margin_type : {"f2", "wellek"}, optional
        Type of effect size used for equivalence margin.

    Returns
    -------
    EquivalenceOnewayResult
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`. See :class:`EquivalenceOnewayResult` for the full list.

    Notes
    -----
    Equivalence in this function is defined in terms of a squared distance
    measure similar to Mahalanobis distance.
    Alternative definitions for the oneway case are based on maximum difference
    between pairs of means or similar pairwise distances.

    The equivalence margin is used for the noncentrality parameter in the
    noncentral F distribution for the test statistic. In samples with unequal
    variances estimated using Welch or Brown-Forsythe Anova, the f-statistic
    depends on the unequal variances and corrections to the test statistic.
    This means that the equivalence margins are not fully comparable across
    methods for treating unequal variances.

    References
    ----------
    Wellek, Stefan. 2010. Testing Statistical Hypotheses of Equivalence and
    Noninferiority. 2nd ed. Boca Raton: CRC Press.

    Cribbie, Robert A., Chantal A. Arpin-Cribbie, and Jamie A. Gruman. 2009.
    “Tests of Equivalence for One-Way Independent Groups Designs.” The Journal
    of Experimental Education 78 (1): 1-13.
    https://doi.org/10.1080/00220970903224552.

    Jan, Show-Li, and Gwowen Shieh. 2019. “On the Extended Welch Test for
    Assessing Equivalence of Standardized Means.” Statistics in
    Biopharmaceutical Research 0 (0): 1-8.
    https://doi.org/10.1080/19466315.2019.1654915.

    """
    if not np.all(np.greater(equiv_margin, 0)):
        raise ValueError(
            f"equiv_margin must be positive, got {equiv_margin}"
        )
    nobs_t = nobs.sum()
    nobs_mean = nobs_t / n_groups

    if margin_type == "wellek":
        nc_null = nobs_mean * equiv_margin**2
        es = f_stat * (n_groups - 1) / nobs_mean
        type_effectsize = "Wellek's psi_squared"
    elif margin_type in ["f2", "fsqu", "fsquared"]:
        nc_null = nobs_t * equiv_margin
        es = f_stat / nobs_t
        type_effectsize = "Cohen's f_squared"
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')
    crit_f = ncf_ppf(alpha, df[0], df[1], nc_null)

    if margin_type == "wellek":
        # TODO: do we need a sqrt
        crit_es = crit_f * (n_groups - 1) / nobs_mean
    elif margin_type in ["f2", "fsqu", "fsquared"]:
        crit_es = crit_f / nobs_t

    reject = es < crit_es

    pv = ncf_cdf(f_stat, df[0], df[1], nc_null)
    pwr = ncf_cdf(crit_f, df[0], df[1], 1e-13)  # scipy, cannot be 0
    res = EquivalenceOnewayResult(
        statistic=f_stat,
        pvalue=pv,
        effectsize=es,  # match es type to margin_type
        crit_f=crit_f,
        crit_es=crit_es,
        reject=reject,
        power_zero=pwr,
        df=df,
        f_stat=f_stat,
        type_effectsize=type_effectsize,
    )
    return res


def equivalence_oneway(
    data,
    equiv_margin,
    groups=None,
    use_var="unequal",
    welch_correction=True,
    trim_frac=0,
    margin_type="f2",
):
    """
    Equivalence test for oneway anova (Wellek's Anova)

    The null hypothesis is that the means differ by more than `equiv_margin`
    in the anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e., within a given distance.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2.
        The data can be provided as a tuple or list of arrays or in long
        format with outcome observations in ``data`` and group membership in
        ``groups``.
    equiv_margin : float or array_like
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    groups : ndarray or Series, optional
        If data is in long format, then groups is needed as indicator to which
        group or sample and observations belongs.
    use_var : {"unequal", "equal", "bf"}, optional
        `use_var` specified how to treat heteroscedasticity, unequal variance,
        across samples. Three approaches are available

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    welch_correction : bool, optional
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    trim_frac : float in [0, 0.5), optional
        Optional trimming for Anova with trimmed mean and winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        The number of trimmed observations is the fraction of number of
        observations in the sample truncated to the next lower integer.
        `trim_frac` has to be smaller than 0.5, however, if the fraction is
        so large that there are not enough observations left over, then `nan`
        will be returned.
    margin_type : {"f2", "wellek"}, optional
        Type of effect size used for equivalence margin, either squared
        Cohen's f or Wellek's psi. Default is "f2".

    Returns
    -------
    EquivalenceOnewayResult
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`. See :class:`EquivalenceOnewayResult` for the full list.

    See Also
    --------
    anova_oneway
    equivalence_scale_oneway
    """

    # use anova to compute summary statistics and f-statistic
    res0 = anova_oneway(
        data,
        groups=groups,
        use_var=use_var,
        welch_correction=welch_correction,
        trim_frac=trim_frac,
    )
    f_stat = res0.statistic
    res = equivalence_oneway_generic(
        f_stat,
        res0.n_groups,
        res0.nobs_total,
        equiv_margin,
        res0.df,
        alpha=0.05,
        margin_type=margin_type,
    )

    return res


def _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, df, alpha=0.05):
    """
    Empirical power of oneway equivalence test

    This only returns post-hoc, empirical power.

    Warning: eps is currently effect size margin as defined as in Wellek, and
    not the signal to noise ratio (Cohen's f family).

    Parameters
    ----------
    f_stat : float
        F-statistic from oneway anova, used to compute empirical effect size
    n_groups : int
        Number of groups in oneway comparison.
    nobs : ndarray
        Array of number of observations in groups.
    eps : float
        Equivalence margin in terms of effect size given by Wellek's psi.
    df : tuple
        Degrees of freedom for F distribution.
    alpha : float in (0, 1), optional
        Significance level for the hypothesis test.

    Returns
    -------
    pow : float
        Ex-post, post-hoc or empirical power at f-statistic of the equivalence
        test.
    """

    res = equivalence_oneway_generic(
        f_stat, n_groups, nobs, eps, df, alpha=alpha, margin_type="wellek"
    )

    nobs_mean = nobs.sum() / n_groups
    fn = f_stat  # post-hoc power, empirical power at estimate
    esn = fn * (n_groups - 1) / nobs_mean  # Wellek psi
    pow_ = ncf_cdf(res.crit_f, df[0], df[1], nobs_mean * esn)

    return pow_


def power_equivalence_oneway(
    f2_alt, equiv_margin, nobs_t, n_groups=None, df=None, alpha=0.05, margin_type="f2"
):
    """
    Power of oneway equivalence test

    Parameters
    ----------
    f2_alt : float
        Effect size, squared Cohen's f, under the alternative.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    nobs_t : ndarray
        Total number of observations summed over all groups.
    n_groups : int, optional
        Number of groups in oneway comparison. If margin_type is "wellek",
        then either ``n_groups`` or ``df`` has to be given.
    df : tuple, optional
        Degrees of freedom for F distribution,
        ``df = (n_groups - 1, nobs_t - n_groups)``
    alpha : float in (0, 1), optional
        Significance level for the hypothesis test.
    margin_type : {"f2", "wellek"}, optional
        Type of effect size used for equivalence margin, either squared
        Cohen's f or Wellek's psi. Default is "f2".

    Returns
    -------
    pow_alt : float
        Power of the equivalence test at given equivalence effect size under
        the alternative.
    """

    # one of n_groups or df has to be specified
    if df is None:
        if n_groups is None:
            raise ValueError("either df or n_groups has to be provided")
        df = (n_groups - 1, nobs_t - n_groups)

    # esn = fn * (n_groups - 1) / nobs_mean  # Wellek psi

    # fix for scipy, ncf does not allow nc == 0, fixed in scipy master
    if f2_alt == 0:
        f2_alt = 1e-13
    # effect size, critical value at margin
    # f2_null = equiv_margin
    if margin_type in ["f2", "fsqu", "fsquared"]:
        f2_null = equiv_margin
    elif margin_type == "wellek":
        if n_groups is None:
            raise ValueError(
                "If margin_type is wellek, then n_groups has to be provided"
            )
        #  f2_null = (n_groups - 1) * n_groups / nobs_t * equiv_margin**2
        nobs_mean = nobs_t / n_groups
        f2_null = nobs_mean * equiv_margin**2 / nobs_t
        f2_alt = nobs_mean * f2_alt**2 / nobs_t
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')

    crit_f_margin = ncf_ppf(alpha, df[0], df[1], nobs_t * f2_null)
    pwr_alt = ncf_cdf(crit_f_margin, df[0], df[1], nobs_t * f2_alt)
    return pwr_alt


class SimulatePowerEquivalenceResult(NamedTuple):
    """
    Result of :func:`simulate_power_equivalence_oneway`.

    Parameters
    ----------
    f_stat : ndarray
        Simulated F-statistics, or Wellek's effect size estimate depending
        on `margin_type`, with one row per Monte Carlo replication and one
        column per entry in `options_var`.
    other : ndarray
        Simulated `crit_f`, `crit_es` and `power_zero` from
        :class:`EquivalenceOnewayResult`, stacked column-wise for each
        Monte Carlo replication and entry in `options_var`.
    pvalue : ndarray
        Simulated p-values, with one row per Monte Carlo replication and
        one column per entry in `options_var`.
    reject : ndarray
        Simulated rejection decisions, with one row per Monte Carlo
        replication and one column per entry in `options_var`.
    """

    f_stat: np.ndarray
    other: np.ndarray
    pvalue: np.ndarray
    reject: np.ndarray


def simulate_power_equivalence_oneway(
    means,
    nobs,
    equiv_margin,
    vars_=None,
    k_mc=1000,
    trim_frac=0,
    options_var=None,
    margin_type="f2",
    rng=None,
):  # , anova_options=None):  # TODO
    """
    Simulate Power for oneway equivalence test (Wellek's Anova)

    This function is experimental and written to evaluate asymptotic power
    function. This function will change without backwards compatibility
    constraints. The only part that is stable is `pvalue` attribute in results.

    Effect size for equivalence margin.

    Parameters
    ----------
    means : array_like
        Mean of samples to be compared. Currently only four groups are
        supported.
    nobs : ndarray
        Number of observations for the samples.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be
        chosen with `margin_type`. default is squared Cohen's f.
    vars_ : array_like, optional
        Variances of the samples used to simulate the data. If None, then
        unit variance, i.e., standard deviation equal to 1, is used for all
        samples.
    k_mc : int, optional
        Number of Monte Carlo replications.
    trim_frac : float in [0, 0.5), optional
        Optional trimming for Anova with trimmed mean and winsorized
        variances, see `trim_frac` in `anova_oneway`.
    options_var : list of str, optional
        List of `use_var` options that are used in the loop over Monte
        Carlo replications. If None, then
        ``["unequal", "equal", "bf"]`` is used.
    margin_type : {"f2", "wellek"}, optional
        Type of effect size used for equivalence margin.
    rng : int, array_like of int, numpy.random.Generator, numpy.random.RandomState, optional
        If `rng` is None, a new ``Generator`` is created using fresh
        entropy from the operating system. If `rng` is an int or array
        of ints, a new ``Generator`` is created, seeded with `rng`. If
        `rng` is already a ``Generator`` or ``RandomState`` instance,
        that instance is used.

    Returns
    -------
    SimulatePowerEquivalenceResult
        Namedtuple with Monte Carlo results in the attributes `f_stat`,
        `other`, `pvalue` and `reject`, each an ndarray with one row per
        Monte Carlo replication and one column per entry in `options_var`.
    """
    if options_var is None:
        options_var = ["unequal", "equal", "bf"]
    if vars_ is not None:
        stds = np.sqrt(vars_)
    else:
        stds = np.ones(len(means))

    nobs_mean = nobs.mean()
    n_groups = len(nobs)
    res_mc = []
    f_mc = []
    reject_mc = []
    other_mc = []
    rng = check_random_state(rng)
    for _ in range(k_mc):
        y0, y1, y2, y3 = (
            m + std * rng.standard_normal(n) for (n, m, std) in zip(nobs, means, stds, strict=True)
        )

        res_i = []
        f_i = []
        reject_i = []
        other_i = []
        for uv in options_var:
            # for welch in options_welch:
            # res1 = sma.anova_generic(means, vars_, nobs, use_var=uv,
            #                          welch_correction=welch)
            res0 = anova_oneway([y0, y1, y2, y3], use_var=uv, trim_frac=trim_frac)
            f_stat = res0.statistic
            res1 = equivalence_oneway_generic(
                f_stat,
                n_groups,
                nobs.sum(),
                equiv_margin,
                res0.df,
                alpha=0.05,
                margin_type=margin_type,
            )
            res_i.append(res1.pvalue)
            es_wellek = f_stat * (n_groups - 1) / nobs_mean
            f_i.append(es_wellek)
            reject_i.append(res1.reject)
            other_i.extend([res1.crit_f, res1.crit_es, res1.power_zero])
        res_mc.append(res_i)
        f_mc.append(f_i)
        reject_mc.append(reject_i)
        other_mc.append(other_i)

    f_mc = np.asarray(f_mc)
    other_mc = np.asarray(other_mc)
    res_mc = np.asarray(res_mc)
    reject_mc = np.asarray(reject_mc)
    res = SimulatePowerEquivalenceResult(
        f_stat=f_mc, other=other_mc, pvalue=res_mc, reject=reject_mc
    )
    return res


@dataclass(frozen=True, slots=True, kw_only=True)
class ScaleAnovaResult(AnovaResult):
    """
    Result of :func:`test_scale_oneway`.

    Extends :class:`AnovaResult`, computed on the transformed data, with
    one additional field.

    Parameters
    ----------
    data_transformed : list of ndarray
        The centered and transformed data used to compute the test.

    See Also
    --------
    AnovaResult

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    data_transformed: list


def test_scale_oneway(
    data,
    method="bf",
    center="median",
    transform="abs",
    trim_frac_mean=0.1,
    trim_frac_anova=0.0,
):
    """
    Oneway Anova test for equal scale, variance or dispersion

    This hypothesis test performs a oneway anova test on transformed data and
    includes Levene and Brown-Forsythe tests for equal variances as special
    cases.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2. The data can be provided
        as a tuple or list of arrays or in long format with outcome
        observations in ``data`` and group membership in ``groups``.
    method : {"unequal", "equal", "bf"}, optional
        How to treat heteroscedasticity across samples. This is used as
        `use_var` option in `anova_oneway` and refers to the variance of the
        transformed data, i.e., assumption is on 4th moment if squares are used
        as transform.
        Three approaches are available:

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.
            This is the default.

    center : {"median", "mean", "trimmed"} or float, optional
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : {"abs", "square", "identity"} or callable, optional
        Transformation for the centered observations. If a callable, then this
        function is called on the centered data.
        Default is absolute value.
    trim_frac_mean : float in [0, 0.5), optional
        Trim fraction for the trimmed mean when `center` is "trimmed"
    trim_frac_anova : float in [0, 0.5), optional
        Optional trimming for Anova with trimmed mean and Winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        see ``trim_frac`` option in `anova_oneway`

    Returns
    -------
    ScaleAnovaResult
        See :class:`ScaleAnovaResult` for a description of the attributes.
        If ``method="bf"``, then `pvalue` is based on corrected degrees of
        freedom following Mehrotra 1997, and `pvalue2`, `df2` and `df_num2`
        hold the Brown-Forsythe 1974 alternative; otherwise those three are
        None.

    See Also
    --------
    anova_oneway
    scale_transform
    """

    xxd = [
        scale_transform(
            np.asarray(x), center=center, transform=transform, trim_frac=trim_frac_mean
        )
        for x in data
    ]

    res0 = anova_oneway(
        xxd,
        groups=None,
        use_var=method,
        welch_correction=True,
        trim_frac=trim_frac_anova,
    )
    res = ScaleAnovaResult(
        data_transformed=xxd,
        **{f.name: getattr(res0, f.name) for f in fields(res0)},
    )
    return res


@dataclass(frozen=True, slots=True)
class ScaleEquivalenceResult(LimitedIterationMixin[float]):
    """
    Result of :func:`equivalence_scale_oneway`.

    Has the same attributes as :class:`EquivalenceOnewayResult`, computed
    on the transformed data, plus `x_transformed`.

    Parameters
    ----------
    statistic : float
        F-statistic used for the equivalence test, same value as `f_stat`.
    pvalue : float
        p-value of the equivalence test.
    effectsize : float
        Estimated effect size, in the units given by `type_effectsize`.
    crit_f : float
        Critical value for the F-statistic at the boundary of the
        equivalence margin.
    crit_es : float
        Critical value for the effect size at the boundary of the
        equivalence margin, in the same units as `effectsize`.
    reject : bool
        True if the null hypothesis of non-equivalence is rejected, i.e.
        the data supports equivalence within `equiv_margin`.
    power_zero : float
        Power of the equivalence test if the true effect size were zero.
    df : tuple
        Degrees of freedom ``(df1, df2)`` of the F-distribution used for
        the test.
    f_stat : float
        F-statistic used for the equivalence test, same value as
        `statistic`.
    type_effectsize : str
        Description of the effect size type used for `effectsize` and
        `crit_es`, determined by `margin_type`.
    x_transformed : list of ndarray
        The centered and transformed data used to compute the test.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    statistic: float
    pvalue: float
    effectsize: float
    crit_f: float
    crit_es: float
    reject: bool
    power_zero: float
    df: tuple
    f_stat: float
    type_effectsize: str
    x_transformed: list


def equivalence_scale_oneway(
    data,
    equiv_margin,
    method="bf",
    center="median",
    transform="abs",
    trim_frac_mean=0.0,
    trim_frac_anova=0.0,
):
    """
    Oneway Anova test for equivalence of scale, variance or dispersion

    This hypothesis test performs a oneway equivalence anova test on
    transformed data.

    Note, the interpretation of the equivalence margin `equiv_margin` will
    depend on the transformation of the data. Transformations like
    absolute deviation are not scaled to correspond to the variance under
    normal distribution.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2. The data can be provided
        as a tuple or list of arrays or in long format with outcome
        observations in ``data`` and group membership in ``groups``.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    method : {"unequal", "equal", "bf"}, optional
        How to treat heteroscedasticity across samples. This is used as
        `use_var` option in `anova_oneway` and refers to the variance of the
        transformed data, i.e., assumption is on 4th moment if squares are used
        as transform.
        Three approaches are available:

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.
            This is the default.
    center : {"median", "mean", "trimmed"} or float, optional
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : {"abs", "square", "identity"} or callable, optional
        Transformation for the centered observations. If a callable, then this
        function is called on the centered data.
        Default is absolute value.
    trim_frac_mean : float in [0, 0.5), optional
        Trim fraction for the trimmed mean when `center` is "trimmed"
    trim_frac_anova : float in [0, 0.5), optional
        Optional trimming for Anova with trimmed mean and Winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        see ``trim_frac`` option in `anova_oneway`

    Returns
    -------
    ScaleEquivalenceResult
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`. See :class:`ScaleEquivalenceResult` for the full list.

    See Also
    --------
    anova_oneway
    scale_transform
    equivalence_oneway
    """
    xxd = [
        scale_transform(
            np.asarray(x), center=center, transform=transform, trim_frac=trim_frac_mean
        )
        for x in data
    ]

    res0 = equivalence_oneway(
        xxd,
        equiv_margin,
        use_var=method,
        welch_correction=True,
        trim_frac=trim_frac_anova,
    )
    res = ScaleEquivalenceResult(
        x_transformed=xxd,
        **{f.name: getattr(res0, f.name) for f in fields(res0)},
    )
    return res
