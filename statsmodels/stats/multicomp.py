# -*- coding: utf-8 -*-
"""

Created on Fri Mar 30 18:27:25 2012
Author: Josef Perktold
"""

from statsmodels.sandbox.stats.multicomp import (  # noqa:F401
    tukeyhsd, games_howell, MultiComparison)

__all__ = ["tukeyhsd", "games_howell", "MultiComparison"]


def pairwise_tukeyhsd(endog, groups, alpha=0.05):
    """
    Calculate all pairwise comparisons with TukeyHSD confidence intervals

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
        calculations, including adjusted p-value

    Notes
    -----
    This is just a wrapper around tukeyhsd method of MultiComparison

    See Also
    --------
    MultiComparison
    tukeyhsd
    statsmodels.sandbox.stats.multicomp.TukeyHSDResults
    """

    return MultiComparison(endog, groups).tukeyhsd(alpha=alpha)


def pairwise_games_howell(endog, groups, alpha=0.05):
    """
    Calculate all pairwise comparisons using Games-Howell.

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
    results : GamesHowellResults instance
        A results class containing relevant data and some post-hoc
        calculations, including adjusted p-value

    Notes
    -----
    This is just a wrapper around games_howell method of MultiComparison

    See Also
    --------
    MultiComparison
    games_howell
    statsmodels.sandbox.stats.multicomp.GamesHowellResults
    """

    return MultiComparison(endog, groups).games_howell(alpha=alpha)
