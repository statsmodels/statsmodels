# -*- coding: utf-8 -*-
"""

Created on Fri Mar 30 18:27:25 2012
Author: Josef Perktold
"""

from statsmodels.sandbox.stats.multicomp import tukeyhsd, MultiComparison

def pairwise_tukeyhsd(endog, groups, alpha=0.05):
    '''calculate all pairwise comparisons with TukeyHSD confidence intervals

    this is just a wrapper around tukeyhsd method of MultiComparison

    Parameters
    ----------
    endog : ndarray, float, 1d
        response variable
    groups : ndarray, 1d
        array with groups, can be string or integers
    alpha : float
        significance level for the test

    Returns
    -------
    table : SimpleTable instance
        table for printing
    tukeyhsd_res : list
        contains detailed results from tukeyhsd function
        [(idx1, idx2), reject, meandiffs, std_pairs, confint, q_crit,
           df_total, reject2]

    See Also
    --------
    MultiComparison
    tukeyhsd

    '''

    return MultiComparison(endog, groups).tukeyhsd(alpha=alpha)
