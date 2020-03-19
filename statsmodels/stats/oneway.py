# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:33:38 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc

from statsmodels.tools.testing import Holder


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

    # Not sure if I need it in this function
    compute_df = False
    welch_correction = False
    if compute_df:
        statistic = f2
        if use_var == "unequal":
            use_satt = True
            tmp = ((1 - w_rel)**2 / (nobs - 1)).sum() / (n_groups**2 - 1)
            if welch_correction:
                statistic /= 1 + 2 * (n_groups - 2) * tmp
            df_denom = 1. / (3. * tmp)

        else:
            use_satt = False
            # variance of group demeaned total sample, pooled var_resid
            tmp = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
            statistic /= tmp
            df_denom = 1. / (3. * tmp)

        df_num = n_groups - 1.
        if use_satt:  # Satterthwaite/Welch degrees of freedom
            df_denom = 1. / (3. * tmp)
        else:
            df_denom = nobs_t - n_groups

    return np.sqrt(f2)


def convert_effectsize_fsqu(f2=None, eta2=None):
    """convert squared effect sizes in f family

    f2 is signal to noise ration, var_explained / var_residual
    eta2 is proportion of explained variance, var_explained / var_total
    omega2 is ...

    uses the relationship:
    f2 = eta2 / (1 - eta2)

    """
    if f2 is not None:
        # f2 = f**2
        eta2 = 1 / (1 + 1 / f2)

    elif eta2 is not None:
        f2 = eta2 / (1 - eta2)

    res = Holder(f2=f2, eta2=eta2)
    return res


def confint_noncentrality(f_stat, df1, df2, alpha=0.05,
                          alternative="two-sided"):
    """confidence interval for noncentality parameter in F-test

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.


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
                  welch_correction=True):
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

    if use_var == "unequal":
        use_satt = True
        tmp = ((1 - w_rel)**2 / (nobs - 1)).sum() / (n_groups**2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1. / (3. * tmp)

    else:
        use_satt = False
        # variance of group demeaned total sample, pooled var_resid
        tmp = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = 1. / (3. * tmp)

    df_num = n_groups - 1.
    if use_satt:  # Satterthwaite/Welch degrees of freedom
        df_denom = 1. / (3. * tmp)
    else:
        df_denom = nobs_t - n_groups

    pval = stats.f.sf(statistic, df_num, df_denom)
    return statistic, pval, df_num, df_denom
