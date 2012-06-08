# -*- coding: utf-8 -*-
"""

Created on Fri Jun 08 15:48:23 2012

Author: Josef Perktold
License : BSD-3

"""



#TODO: split into modification and pvalue functions separately ?
#      for separate testing and combining different pieces

def dplus_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = np.exp(-2 * stat_modified**2)
    digits = np.sum(stat > np.array([0.82, 0.82, 1.00]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

dminus_st70_upp = dplus_st70_upp


def d_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = 2 * np.exp(-2 * stat_modified**2)
    digits = np.sum(stat > np.array([0.91, 0.91, 1.08]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

def v_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.155 + 0.24 / np.sqrt(nobs)
    #repeat low to get {0,2,3}
    stat_modified = stat * mod_factor
    zsqu = stat_modified**2
    pval = (8 * zsqu - 2) * np.exp(-2 * zsqu)
    digits = np.sum(stat > np.array([1.06, 1.06, 1.26]))
    return stat_modified, pval, digits

def wsqu_st70_upp(stat, nobs):
    nobsinv = 1. / nobs
    stat_modified = (stat - 0.4 * nobsinv + 0.6 * nobsinv**2) * (1 + nobsinv)
    pval = 0.05 * np.exp(2.79 - 6 * stat_modified)
    digits = np.nan  # some explanation in txt
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

def usqu_st70_upp(stat, nobs):
    nobsinv = 1. / nobs
    stat_modified = (stat - 0.1 * nobsinv + 0.1 * nobsinv**2)
    stat_modified *= (1 + 0.8 * nobsinv)
    pval = 2 * np.exp(- 2 * stat_modified * np.pi**2)
    digits = np.sum(stat > np.array([0.29, 0.29, 0.34]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

def a_st70_upp(stat, nobs):
    nobsinv = 1. / nobs
    stat_modified = (stat - 0.7 * nobsinv + 0.9 * nobsinv**2)
    stat_modified *= (1 + 1.23 * nobsinv)
    pval = 1.273 * np.exp(- 2 * stat_modified / 2. * np.pi**2)
    digits = np.sum(stat > np.array([0.11, 0.11, 0.452]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits



gof_pvals = {}

gof_pvals['stephens70upp'] = {
    'd_plus' : dplus_st70_upp,
    'd_minus' : dplus_st70_upp,
    'd' : d_st70_upp,
    'v' : v_st70_upp,
    'wsqu' : wsqu_st70_upp,
    'usqu' : usqu_st70_upp,
    'a' : a_st70_upp }

def pval_kstest_approx(D, N):
    pval_two = distributions.kstwobign.sf(D*np.sqrt(N))
    if N > 2666 or pval_two > 0.80 - N*0.3/1000.0 :
        return D, distributions.kstwobign.sf(D*np.sqrt(N)), np.nan
    else:
        return D, distributions.ksone.sf(D,N)*2, np.nan

gof_pvals['scipy'] = {
    'd_plus' : lambda Dplus, N: (Dplus, distributions.ksone.sf(Dplus, N), np.nan),
    'd_minus' : lambda Dmin, N: (Dmin, distributions.ksone.sf(Dmin,N), np.nan),
    'd' : lambda D, N: (D, distributions.kstwobign.sf(D*np.sqrt(N)), np.nan)
    }

gof_pvals['scipy_approx'] = {
    'd' : pval_kstest_approx }

