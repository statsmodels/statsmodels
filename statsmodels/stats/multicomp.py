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
    results : TukeyHSDResults instance
        A results class containing relevant data and some post-hoc
        calculations

    See Also
    --------
    MultiComparison
    tukeyhsd
    statsmodels.sandbox.stats.multicomp.TukeyHSDResults

    '''

    return MultiComparison(endog, groups).tukeyhsd(alpha=alpha)
