"""

Created on Fri Mar 30 18:27:25 2012
Author: Josef Perktold
"""

import numpy as np
from scipy import special, stats

from statsmodels.sandbox.stats.multicomp import (
    MultiComparison,
    tukeyhsd,
    varcorrection_pairs_unequal,
)

__all__ = ["MultiComparison", "tukeyhsd"]


def pairwise_tukeyhsd(endog, groups, alpha=0.05, use_var="equal"):
    """
    Calculate all pairwise comparisons with TukeyHSD or Games-Howell

    Parameters
    ----------
    endog : array_like, 1d
        response variable
    groups : array_like, 1d
        array with groups, can be string or integers
    alpha : float, optional
        significance level for the test
    use_var : {"unequal", "equal"}, optional
        If ``use_var`` is "equal", then the Tukey-hsd pvalues are returned.
        Tukey-hsd assumes that (within) variances are the same across groups.
        If ``use_var`` is "unequal", then the Games-Howell pvalues are
        returned. This uses Welch's t-test for unequal variances with
        Satterthwaite's corrected degrees of freedom for each pairwise
        comparison.

    Returns
    -------
    results : TukeyHSDResults instance
        A results class containing relevant data and some post-hoc
        calculations, including adjusted p-value.

    See Also
    --------
    MultiComparison
        Class for pairwise comparisons of multiple groups.
    tukeyhsd
        Compute simultaneous Tukey HSD comparisons from summary data.
    statsmodels.sandbox.stats.multicomp.TukeyHSDResults
        Results from a Tukey HSD comparison.

    Notes
    -----
    The results include the following attributes and methods:

    * ``reject`` is a boolean array indicating whether each comparison is
      statistically significant.
    * ``pvalues`` contains the adjusted p-values for each comparison.
    * ``summary()`` returns a printable table that includes the reject column.
    * ``summary_frame()`` returns a DataFrame with the comparison results.

    This is just a wrapper around tukeyhsd method of MultiComparison.
    Tukey-hsd is not robust to heteroscedasticity, i.e., variance differ across
    groups, especially if group sizes also vary. In those cases, the actual
    size (rejection rate under the Null hypothesis) might be far from the
    nominal size of the test.
    The Games-Howell method uses pairwise t-tests that are robust to differences
    in variances and approximately maintains size unless samples are very
    small.

    .. versionadded:: 0.15

        The `use_var` keyword and option for Games-Howell test.

    Examples
    --------
    The reject decisions and adjusted p-values can be accessed directly from
    the results instance.

    >>> import numpy as np
    >>> endog = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array(["a", "a", "b", "b", "c", "c"])
    >>> res = pairwise_tukeyhsd(endog, groups)
    >>> res.reject
    array([False,  True, False])
    >>> res.pvalues.round(3)
    array([0.129, 0.022, 0.129])

    """

    return MultiComparison(endog, groups).tukeyhsd(alpha=alpha,
                                                   use_var=use_var)


# Gauss-Legendre rule used on each panel in _smm_integrate
_SMM_NODES, _SMM_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _smm_tail(z, k):
    # P(max |Z_i| > z) for k independent standard normal random variables
    with np.errstate(divide="ignore"):
        return -np.expm1(k * np.log1p(-2 * stats.norm.sf(z)))


def _smm_tail_pdf(z, k):
    # derivative of 1 - _smm_tail(z, k) with respect to z, for k > 1
    with np.errstate(divide="ignore"):
        power = np.exp((k - 1) * np.log1p(-2 * stats.norm.sf(z)))
    return 2 * k * stats.norm.pdf(z) * power


def _smm_integrate(q, k, df, pdf=False):
    """
    Survival function and density of the studentized maximum modulus

    q and df are 1-d arrays with q > 0 and finite df. The integral over the
    distribution of s = sqrt(chi2 / df) is computed in x = log(s) with a
    Gauss-Legendre rule on panels that separate the mode of the density at
    x = 0 and the region around x = -log(q) where the tail function changes
    from 1 to 0.
    """
    a = df / 2
    # P(|t| > q) is a lower bound of the survival function, the probability
    # below the lower limit is negligible relative to it
    lower_bound = 2 * stats.t.sf(q, df)
    y_lo = stats.gamma.ppf(np.maximum(1e-12 * lower_bound, 1e-300), a)
    y_lo = np.maximum(y_lo, np.finfo(float).tiny)
    x_lo = 0.5 * np.log(y_lo / a)
    x_t = -np.log(q)
    x_hi = 0.5 * np.log(stats.gamma.isf(1e-16, a) / a)
    x_hi = np.minimum(x_hi, x_t + np.log(40.0))
    width = 1 / np.sqrt(df)
    breaks = np.column_stack(
        [x_lo, x_hi]
        + [x_t + d for d in (-3.0, -1.0, 0.0, 1.0, 3.0)]
        + [d * width for d in (-4.0, -1.0, 0.0, 1.0, 4.0)]
    )
    breaks = np.sort(np.clip(breaks, x_lo[:, None], x_hi[:, None]), axis=1)
    half = np.diff(breaks, axis=1)[..., None] / 2
    x = breaks[:, :-1, None] + half * (1 + _SMM_NODES)
    weights = half * _SMM_WEIGHTS
    # density of x, y = a * exp(2 * x) is gamma distributed with shape a
    a3 = a[:, None, None]
    logy = np.log(a3) + 2 * x
    dens = np.exp(np.log(2) - special.gammaln(a3) + a3 * logy - np.exp(logy))
    s = np.exp(x)
    qs = q[:, None, None] * s
    sf = (weights * dens * _smm_tail(qs, k)).sum(axis=(1, 2))
    if not pdf:
        return sf
    density = (weights * dens * s * _smm_tail_pdf(qs, k)).sum(axis=(1, 2))
    return sf, density


def _smm_sf(q, k, df):
    r"""
    Survival function of the studentized maximum modulus distribution

    Parameters
    ----------
    q : array_like
        Quantiles.
    k : int
        Number of variables.
    df : array_like
        Degrees of freedom of the denominator. Can be non-integer or inf.

    Returns
    -------
    ndarray
        The probability that the studentized maximum modulus is larger
        than q.

    Notes
    -----
    The studentized maximum modulus is the maximum of the absolute values of
    k independent standard normal random variables divided by
    :math:`s = \sqrt{\chi^2_{df} / df}`, where the chi-square random variable
    is independent of the normal random variables. This is the maximum
    absolute value of a multivariate t random variable with uncorrelated
    components.

    The survival function is computed by integrating over the distribution
    of s,

    .. math::

        P(M > q) = \int_0^\infty 1 - (1 - 2 \Phi(-q s))^k f(s) ds

    where :math:`f` is the density of s. The integral is computed for all
    values at once with a fixed Gauss-Legendre rule. The relative error is
    smaller than 1e-10 for probabilities larger than 1e-20.

    References
    ----------
    .. [*] Stoline, M. R., and Ury, H. K. (1979). Tables of the Studentized
       Maximum Modulus Distribution and an Application to Multiple
       Comparisons among Means. Technometrics, 21(1), 87-93.
    """
    q, df = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(df, dtype=float))
    shape = q.shape
    q, df = q.ravel(), df.ravel()
    out = np.full(q.shape, np.nan)
    valid = ~(np.isnan(q) | np.isnan(df))
    out[valid & (q <= 0)] = 1.0
    positive = valid & (q > 0)
    inf_df = positive & np.isinf(df)
    out[inf_df] = _smm_tail(q[inf_df], k)
    finite = positive & np.isfinite(df)
    if finite.any():
        out[finite] = _smm_integrate(q[finite], k, df[finite])
    return out.reshape(shape)


def _smm_ppf(prob, k, df):
    """
    Quantile function of the studentized maximum modulus distribution

    Parameters
    ----------
    prob : float
        Probability, 1 - alpha.
    k : int
        Number of variables.
    df : array_like
        Degrees of freedom of the denominator. Can be non-integer or inf.

    Returns
    -------
    ndarray
        The quantiles of the distribution.

    See Also
    --------
    _smm_sf
        Survival function of the studentized maximum modulus distribution.

    Notes
    -----
    The quantiles are computed for all df at once with Newton steps that
    stay inside the bracket given by k = 1 and by the Bonferroni inequality,
    and bisection otherwise.
    """
    alpha = 1 - prob
    df = np.asarray(df, dtype=float)
    shape = df.shape
    df = df.ravel()
    lower = stats.t.isf(alpha / 2, df)
    if k == 1:
        return lower.reshape(shape)
    upper = stats.t.isf(alpha / (2 * k), df)
    out = np.full(df.shape, np.nan)
    active = ~np.isnan(df)
    lower, upper, df = lower[active], upper[active], df[active]
    finite = np.isfinite(df)
    q = upper.copy()
    for _ in range(100):
        sf = _smm_tail(q, k)
        density = _smm_tail_pdf(q, k)
        if finite.any():
            sf[finite], density[finite] = _smm_integrate(
                q[finite], k, df[finite], pdf=True
            )
        diff = sf - alpha
        lower = np.where(diff > 0, q, lower)
        upper = np.where(diff > 0, upper, q)
        q_new = q + diff / density
        outside = ~((q_new > lower) & (q_new < upper))
        q_new[outside] = (lower[outside] + upper[outside]) / 2
        converged = np.abs(q_new - q) <= 1e-12 * q
        q = q_new
        if converged.all():
            break
    out[active] = q
    return out.reshape(shape)


def _dunnett_t3(mean_all, nobs_all, var_all, alpha=0.05):
    """
    Dunnett's T3 simultaneous comparison of all pairs of means

    Parameters
    ----------
    mean_all : array_like
        The sample mean of each group.
    nobs_all : array_like
        The number of observations of each group.
    var_all : array_like
        The sample variance of each group, using ddof=1.
    alpha : float, optional
        The family-wise error rate.

    Returns
    -------
    dict
        Dictionary with the pair indices, the mean differences, their
        standard errors, the Welch degrees of freedom, the t statistics,
        the adjusted p-values, the critical values, the simultaneous
        confidence intervals and the reject decisions.
    """
    mean_all = np.asarray(mean_all, dtype=float)
    nobs_all = np.asarray(nobs_all, dtype=float)
    var_all = np.asarray(var_all, dtype=float)
    n_means = len(mean_all)

    # pairs where both variances are zero have undefined statistics, nan
    with np.errstate(invalid="ignore", divide="ignore"):
        var_pairs, df_pairs = varcorrection_pairs_unequal(
            var_all, nobs_all, nobs_all - 1
        )
        idx1, idx2 = np.triu_indices(n_means, 1)
        meandiffs = (mean_all - mean_all[:, None])[idx1, idx2]
        std_pairs = np.sqrt(var_pairs[idx1, idx2])
        df_pairs = df_pairs[idx1, idx2]
        tvalues = meandiffs / std_pairs
    n_pairs = len(idx1)

    pvalues = _smm_sf(np.abs(tvalues), n_pairs, df_pairs)
    q_crit = _smm_ppf(1 - alpha, n_pairs, df_pairs)
    crit_int = std_pairs * q_crit
    confint = np.column_stack((meandiffs - crit_int, meandiffs + crit_int))
    reject = np.abs(tvalues) > q_crit

    return {
        "pair_indices": (idx1, idx2),
        "meandiffs": meandiffs,
        "std_pairs": std_pairs,
        "df_pairs": df_pairs,
        "tvalues": tvalues,
        "pvalues": pvalues,
        "q_crit": q_crit,
        "confint": confint,
        "reject": reject,
    }
