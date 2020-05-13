# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:33:38 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc

from statsmodels.stats.robust_compare import TrimmedMean
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple


def effectsize_oneway(means, vars_, nobs, use_var="unequal", ddof_between=0):
    """effect size corresponding to Cohen's f^2 = nc / nobs for oneway anova

    Parameters
    ----------
    means: array_like
        Mean of samples to be compared
    vars_ : float or array_like
        Residual (within) variance of each sample or pooled
        If var_ is scalar, then it is interpreted as pooled variance that is
        the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : int or array_like
        Number of observations for the samples.
        If nobs is scalar, then it is assumed that all samples have the same
        number ``nobs`` of observation, i.e. a balanced sample case.
        Otherwise, statistics will be weighted corresponding to nobs.
    use_var : {"unequal", "equal"}
        If ``use_var`` is "unequal", then the variances can differe across
        samples and the effect size for Welch anova will be computed.
    ddof_between : int
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
    This currently handles the following 2 cases for oneway anova

    - balanced sample with homoscedastic variances
    - samples with different number of observations and  with homoscedastic
      variances
    - samples with different number of observations and  with heteroscedastic
      variances. This corresponds to Welch anova

    Status: experimental
    This function will be changed to support additional cases like
    Brown-Forsythe anova.
    We might add additional returns, if those are needed to support power
    and sample size applications.


    """
    # the code here is largely a copy of onway_generic

    means = np.asarray(means)
    n_groups = means.shape[0]

    if np.size(nobs) == 1:
        nobs = np.ones(n_groups) * nobs

    nobs_t = nobs.sum()

    if use_var == "equal":
        if np.size(vars_) == 1:
            var_resid = vars_
        else:
            vars_ = np.asarray(vars_)
            var_resid = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)

        vars_ = var_resid  # scalar, if broadcasting works

    weights = nobs / vars_

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    # f2 = np.dot(weights, (means - meanw_t)**2) / (n_groups - ddof_between)
    f2 = np.dot(weights, (means - meanw_t)**2) / (nobs_t - ddof_between)

    return np.sqrt(f2)


def convert_effectsize_fsqu(f2=None, eta2=None):
    """convert squared effect sizes in f family

    f2 is signal to noise ratio, var_explained / var_residual
    eta2 is proportion of explained variance, var_explained / var_total
    omega2 is ...

    uses the relationship:
    f2 = eta2 / (1 - eta2)

    """
    if f2 is not None:
        eta2 = 1 / (1 + 1 / f2)

    elif eta2 is not None:
        f2 = eta2 / (1 - eta2)

    res = Holder(f2=f2, eta2=eta2)
    return res


def _fstat2effectsize(f_stat, df1, df2):
    """Compute anova effect size from F-statistic

    This might be combined with convert_effectsize_fsqu

    Parameters
    ----------
    f_stat : array_like
        F-statistic corresponding to an F-test
    df1 : int or float
        numerator degrees of freedom, number of constraints
    df2 : int or float
        denominator degrees of freedom, df_resid

    Returns
    -------
    res : Holder instance
        This instance contains effect size measures f2, eta2, omega2 and eps2
        as attributes.
    """
    f2 = f_stat * df1 / df2
    eta2 = f2 / (f2 + 1)
    omega2_ = (f_stat - 1) / (f_stat + (df2 + 1) / df1)
    omega2 = (f2 - df1 / df2) / (f2 + 2)  # rewrite
    eps2_ = (f_stat - 1) / (f_stat + df2 / df1)
    eps2 = (f2 - df1 / df2) / (f2 + 1)  # rewrite
    return Holder(f2=f2, eta2=eta2, omega2=omega2, eps2=eps2, eps2_=eps2_,
                  omega2_=omega2_)


def confint_noncentrality(f_stat, df1, df2, alpha=0.05,
                          alternative="two-sided"):
    """confidence interval for noncentality parameter in F-test

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Notes
    -----
    The algorithm inverts the cdf of the noncentral F distribution with
    respect to the noncentrality parameters.
    See Steiger 2004 and references cited in it.

    References
    ----------
    Steiger, James H. 2004. “Beyond the F Test: Effect Size Confidence
    Intervals and Tests of Close Fit in the Analysis of Variance and Contrast
    Analysis.” Psychological Methods 9 (2): 164–82.
    https://doi.org/10.1037/1082-989X.9.2.164.

    See Also
    --------
    `confint_effectsize_oneway`
    """

    if alternative in ["two-sided", "2s", "ts"]:
        alpha1s = alpha / 2
        ci = ncfdtrinc(df1, df2, [1 - alpha1s, alpha1s], f_stat)
    else:
        raise NotImplementedError

    return ci


def confint_effectsize_oneway(f_stat, df1, df2, alpha=0.05, nobs=None,
                              alternative="two-sided"):
    """confidence interval for effect size in oneway anova for F distribution

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    returns an instance of a Holder class with effect size confidence
    intervals as attributes.

    Notes
    -----
    The confidence interval for the noncentrality parameter is obtained by
    inverting the cdf of the noncentral F distribution. Confidence intervals
    for other effect sizes are computed by endpoint transformation.

    See Also
    --------
    `confint_noncentrality`

    """
    if nobs is None:
        nobs = df1 + df2 + 1
    ci_nc = confint_noncentrality(f_stat, df1, df2, alpha=alpha,
                                  alternative="two-sided")

    ci_f2 = ci_nc / nobs
    ci_res = convert_effectsize_fsqu(f2=ci_f2)
    ci_res.ci_nc = ci_nc
    ci_res.ci_f = np.sqrt(ci_res.f2)
    ci_res.ci_eta = np.sqrt(ci_res.eta2)
    ci_res.ci_f_corrected = np.sqrt(ci_res.f2 * (df1 + 1) / df1)

    return ci_res


def anova_generic(means, vars_, nobs, use_var="unequal",
                  welch_correction=True, info=None):
    """oneway anova based on summary statistics

    incompletely verified

    """
    options = {"use_var": use_var,
               "welch_correction": welch_correction
               }
    nobs_t = nobs.sum()
    n_groups = len(means)
    # mean_t = (nobs * means).sum() / nobs_t
    if use_var == "unequal":
        weights = nobs / vars_
    else:
        weights = nobs

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    statistic = np.dot(weights, (means - meanw_t)**2) / (n_groups - 1.)
    use_satt = False
    df_num = n_groups - 1.

    if use_var == "unequal":
        use_satt = True
        tmp = ((1 - w_rel)**2 / (nobs - 1)).sum() / (n_groups**2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1. / (3. * tmp)

    elif use_var == "equal":
        # variance of group demeaned total sample, pooled var_resid
        tmp = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = nobs_t - n_groups
    elif use_var == "bf":
        tmp = ((1. - nobs / nobs_t) * vars_).sum()
        statistic = 1. * (nobs * (means - meanw_t)**2).sum()
        statistic /= tmp

        df_num2 = n_groups - 1
        df_denom = tmp**2 / ((1. - nobs / nobs_t)**2 *
                             vars_**2 / (nobs - 1)).sum()
        df_num = tmp**2 / ((vars_**2).sum() +
                           (nobs / nobs_t * vars_).sum()**2 -
                           2 * (nobs / nobs_t * vars_**2).sum())
        pval2 = stats.f.sf(statistic, df_num2, df_denom)
        options["df2"] = (df_num2, df_denom)
        options["df_num2"] = df_num2
        options["pvalue2"] = pval2
    else:
        raise ValueError('use_var is to be one of "unequal", "equal" or "bf"')

#     if use_satt:  # Satterthwaite/Welch degrees of freedom
#         df_denom = 1. / (3. * tmp)
#     else:
#         df_denom = nobs_t - n_groups

    pval = stats.f.sf(statistic, df_num, df_denom)
    res = HolderTuple(statistic=statistic,
                      pvalue=pval,
                      df=(df_num, df_denom),
                      df_num=df_num,
                      df_denom=df_denom,
                      **options
                      )
    return res


def anova_oneway(data, groups=None, use_var="unequal", welch_correction=True,
                 trim_frac=0):
    """one-way anova

    This implements standard anova, Welch and Brown-Forsythe and trimmed
    (Yuen) variants of them.

    Parameters
    ----------

    use_var : {"unequal", "equal" or "bf"}
        `use_var` specified how to treat heteroscedasticity, uneqau variance,
        across samples. Three approaches are available

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : variances are assumed to be equal across samples. This is
            the standard Anova.
        "bf: Variances are not assumed to be equal across samples. The method
            is Browne-Forsythe (1971) with the corrected degrees of freedom
            by Merothra


    """
    if groups is not None:
        uniques = np.unique(groups)
        data = [data[groups == uni] for uni in uniques]
        raise NotImplementedError('groups is not available yet')
    else:
        uniques = None
    args = list(map(np.asarray, data))
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')

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
        vars_ = np.array([tm.var_winsorized * (tm.nobs - 1) /
                          (tm.nobs_reduced - 1) for tm in tms])
        # nobs_original = nobs  # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    res = anova_generic(means, vars_, nobs, use_var=use_var,
                        welch_correction=welch_correction)

    return res


def oneway_equivalence_generic(f, n_groups, nobs, eps, df, alpha=0.05):
    """Equivalence test for oneway anova (Wellek and extensions)

    Warning: eps is currently defined as in Wellek, but will change to
    signal to noise ration (Cohen's f family)

    The null hypothesis is that the means differ by more than `eps` in the
    anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e. within a given distance.

    Parameters
    ----------
    f, n_groups, nobs, eps, df, alpha

    Returns
    -------
    results : instance of a Holder class



    Notes
    -----
    Equivalence in this function is defined in terms of a squared distance
    measure similar to Mahalanobis distance.
    Alternative definitions for the oneway case are based on maximum difference
    between pairs of means or similar pairwise distances.

    References
    ----------
    Wellek book

    Cribbie, Robert A., Chantal A. Arpin-Cribbie, and Jamie A. Gruman. 2009.
    “Tests of Equivalence for One-Way Independent Groups Designs.” The Journal
    of Experimental Education 78 (1): 1–13.
    https://doi.org/10.1080/00220970903224552.

    Jan, Show-Li, and Gwowen Shieh. 2019. “On the Extended Welch Test for
    Assessing Equivalence of Standardized Means.” Statistics in
    Biopharmaceutical Research 0 (0): 1–8.
    https://doi.org/10.1080/19466315.2019.1654915.

    """
    nobs_mean = nobs.sum() / n_groups

    es = f * (n_groups - 1) / nobs_mean
    crit_f = stats.ncf.ppf(alpha, df[0], df[1], nobs_mean * eps**2)
    crit_es = (n_groups - 1) / nobs_mean * crit_f
    reject = (es < crit_es)

    pv = stats.ncf.cdf(f, df[0], df[1], nobs_mean * eps**2)
    pwr = stats.ncf.cdf(crit_f, df[0], df[1], nobs_mean * 1e-13)
    res = HolderTuple(statistic=es,
                      pvalue=pv,
                      es=es,
                      crit_f=crit_f,
                      crit_es=crit_es,
                      reject=reject,
                      power_zero=pwr,
                      df=df,
                      # es is currently hard coded,
                      #     not correct for Welch anova `f`
                      type_effectsize="Welleks psi_squared"
                      )
    return res


def power_oneway_equivalence(f, n_groups, nobs, eps, df, alpha=0.05):
    """power for oneway equivalence test

    This is incomplete and currently only returns post-hoc, empirical power.

    Warning: eps is currently defined as in Wellek, but will change to
    signal to noise ration (Cohen's f family)

    draft version, need specification of alternative
    """

    res = oneway_equivalence_generic(f, n_groups, nobs, eps, df, alpha=0.05)
    # at effect size at alternative
    # fn, pvn, dfn = oneway_equivalence_generic(f, n_groups, nobs, eps, df,
    #                                          alpha=0.05)
    # f, pv, df0, df1 = anova_generic(means, stds**2, nobs,
    #                                use_var="equal")
    nobs_mean = nobs.sum() / n_groups
    fn = f  # post-hoc power, empirical power at estimate
    esn = fn * (n_groups - 1) / nobs_mean  # Wellek psi
    pow_ = stats.ncf.cdf(res.crit_f, df[0], df[1], nobs_mean * esn)

    return pow_
