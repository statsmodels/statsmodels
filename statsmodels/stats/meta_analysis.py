# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:34:25 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
import pandas as pd
from scipy import stats


class CombineResults(object):

    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self._ini_keys = list(kwds.keys())

        # explained variance measures
        self.h2 = self.q / (self.k - 1)
        self.i2 = 1 - 1 / self.h2

    def test_homogeneity(self):
        pvalue = stats.chi2.sf(self.q, self.k - 1)
        return self.q, pvalue, self.k - 1

    def summary_array(self):
        res = np.column_stack([self.eff, self.sd_eff,
                               self.ci_low, self.ci_upp,
                               self.weights_rel_fe, self.weights_rel_re])

        res_fe = [[self.mean_effect_fe, self.sd_eff_w_fe,
                  self.ci_low_eff_fe, self.ci_upp_eff_fe, 1, np.nan]]
        res_re = [[self.mean_effect_re, self.sd_eff_w_re,
                  self.ci_low_eff_re, self.ci_upp_eff_re, np.nan, 1]]

        res = np.concatenate([res, res_fe, res_re], axis=0)
        column_names = ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe", "w_re"]
        return res, column_names

    def summary_frame(self):
        labels = list(self.row_names) + ["fixed effect", "random effect"]
        res, col_names = self.summary_array()
        results = pd.DataFrame(res, index=labels, columns=col_names)
        return results


def effectsize_smd(mean2, sd2, nobs2, mean1, sd1, nobs1, row_names=None,
                   alpha=0.05):
    """effect sizes for mean difference for use in meta-analysis

    mean2, sd2, nobs2 are for treatment
    mean1, sd1, nobs1 are for control

    Effect sizes are computed for the mean difference ``mean2 - mean1``

    This does not have option yet.
    It uses standardized mean difference with bias correction as effect size.

    This currently does not use np.asarray, all computations are possible in
    pandas.

    References
    ----------
    Borenstein, Michael. 2009. Introduction to Meta-Analysis.
        Chichester: Wiley.

    Chen, Ding-Geng, and Karl E. Peace. 2013. Applied Meta-Analysis with R.
        Chapman & Hall/CRC Biostatistics Series.
        Boca Raton: CRC Press/Taylor & Francis Group.


    """
    k = len(mean2)
    if row_names is None:
        row_names = list(range(k))
    # crit = stats.norm.isf(alpha / 2)
    # TODO: not used yet, design and options
    # var_diff_uneq = sd2**2 / nobs2 + sd1**2 / nobs1
    var_diff = (sd2**2 * (nobs2 - 1) +
                sd1**2 * (nobs2 - 1)) /  (nobs2 + nobs1 - 2)
    sd_diff = np.sqrt(var_diff)
    nobs = nobs2 + nobs1
    bias_correction = 1 - 3 / (4 * nobs - 9)
    smd = (mean2 - mean1) / sd_diff
    smd_bc = bias_correction * smd
    var_smdbc = nobs / nobs2 / nobs1  + smd_bc**2 / 2 / (nobs - 3.94)
    return smd_bc, var_smdbc


def combine_effects(effect, variance, row_names=None, alpha=0.05):
    """combining effect sizes for effect sizes using meta-analysis



    This does not have option yet.
    It uses standardized mean difference with bias correction as effect size.

    This currently does not use np.asarray, all computations are possible in
    pandas.

    References
    ----------
    Borenstein, Michael. 2009. Introduction to Meta-Analysis.
        Chichester: Wiley.

    Chen, Ding-Geng, and Karl E. Peace. 2013. Applied Meta-Analysis with R.
        Chapman & Hall/CRC Biostatistics Series.
        Boca Raton: CRC Press/Taylor & Francis Group.


    """
    k = len(effect)
    if row_names is None:
        row_names = list(range(k))
    crit = stats.norm.isf(alpha / 2)

    # alias for initial version
    eff = effect
    var_eff = variance
    sd_eff = np.sqrt(var_eff)

    ci_low = eff - 1.96 * sd_eff
    ci_upp = eff + 1.96 * sd_eff

    # fixed effects computation

    weights_fe = 1 / var_eff # no bias correction ?
    w_total_fe = weights_fe.sum(0)
    weights_rel_fe = weights_fe / w_total_fe

    eff_w_fe = weights_rel_fe * eff
    mean_effect_fe = eff_w_fe.sum()
    var_eff_w_fe = 1 / w_total_fe
    sd_eff_w_fe = np.sqrt(var_eff_w_fe)

    ci_low_eff_fe = mean_effect_fe - crit * sd_eff_w_fe
    ci_upp_eff_fe = mean_effect_fe + crit * sd_eff_w_fe

    # random effects computation

    q = (weights_fe * eff**2).sum(0)
    q -= (weights_fe * eff).sum()**2 / w_total_fe
    df = k - 1
    c = w_total_fe - (weights_fe**2).sum() / w_total_fe
    tau2 = (q - df) / c

    weights_re = 1 / (var_eff + tau2) # no  bias_correction ?
    w_total_re = weights_re.sum(0)
    weights_rel_re = weights_re / weights_re.sum(0)

    eff_w_re = weights_rel_re * eff
    mean_effect_re = eff_w_re.sum()
    var_eff_w_re = 1 / w_total_re
    sd_eff_w_re = np.sqrt(var_eff_w_re)
    ci_low_eff_re = mean_effect_re - crit * sd_eff_w_re
    ci_upp_eff_re = mean_effect_re + crit * sd_eff_w_re

    scale_hksj_re = (weights_re * (eff - mean_effect_re)**2).sum() / df
    scale_hksj_fe = (weights_fe * (eff - mean_effect_fe)**2).sum() / df
    var_hksj_re = (weights_rel_re * (eff - mean_effect_re)**2).sum() / df
    var_hksj_fe = (weights_rel_fe * (eff - mean_effect_fe)**2).sum() / df

    res = CombineResults(**locals())
    return res


def _fit_tau_iterative(eff, var_eff, tau2_start=0, atol=1e-5, maxiter=50):
    """Paule-Mandel iterative estimate of between random effect variance


    implementation follows DerSimonian and Kacker 2007 Appendix 8
    see also Kacker 2004
    """
    tau2 = tau2_start
    k = eff.shape[0]
    converged = False
    for i in range(maxiter):
        w = 1 / (var_eff + tau2)
        m = w.dot(eff) / w.sum(0)
        resid_sq = (eff - m)**2
        q_w = w.dot(resid_sq)
        # estimating equation
        ee = q_w - (k - 1)
        if ee < 0:
            tau2 = 0
            converged = 0
            break
        if np.allclose(ee, 0, atol=atol):
            converged = True
            break
        # update tau2
        delta = ee / (w**2).dot(resid_sq)
        tau2 += delta

    return tau2, converged


def _fit_tau_mm(eff, var_eff, weights):
    """method of moment estimate of between random effect variance


    implementation follows Kacker 2004 and DerSimonian and Kacker 2007 eq. 6

    """
    w = weights

    m = w.dot(eff) / w.sum(0)
    resid_sq = (eff - m)**2
    q_w = w.dot(resid_sq)
    w_t = w.sum()
    expect = w.dot(var_eff) - (w**2).dot(var_eff) / w_t
    denom = w_t - (w**2).sum() / w_t
    # estimating equation
    tau2 = (q_w - expect) / denom

    return tau2


def _fit_tau_iter_mm(eff, var_eff, tau2_start=0, atol=1e-5, maxiter=50):
    """iterated method of moment estimate of between random effect variance

    This repeatedly estimates tau, updating weights in each iteration
    see two-step estimators in DerSimonian and Kacker 2007
    """
    tau2 = tau2_start
    converged = False
    for i in range(maxiter):
        w = 1 / (var_eff + tau2)

        tau2_new = _fit_tau_mm(eff, var_eff, w)
        tau2_new = max(0, tau2_new)

        delta = tau2_new - tau2
        if np.allclose(delta, 0, atol=atol):
            converged = True
            break

        tau2 = tau2_new

    return tau2, converged
