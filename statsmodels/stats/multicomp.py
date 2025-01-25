"""

Created on Fri Mar 30 18:27:25 2012
Author: Josef Perktold
"""

from statsmodels.sandbox.stats.multicomp import (  # noqa:F401
    tukeyhsd, MultiComparison)

__all__ = ['tukeyhsd', 'MultiComparison']


def pairwise_tukeyhsd(endog, groups, alpha=0.05, use_var='equal'):
    """
    Calculate all pairwise comparisons with TukeyHSD or Games-Howell.

    Parameters
    ----------
    endog : ndarray, float, 1d
        response variable
    groups : ndarray, 1d
        array with groups, can be string or integers
    alpha : float
        significance level for the test
    use_var : {"unequal", "equal"}
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
        calculations, including adjusted p-value

    Notes
    -----
    This is just a wrapper around tukeyhsd method of MultiComparison.
    Tukey-hsd is not robust to heteroscedasticity, i.e. variance differ across
    groups, especially if group sizes also vary. In those cases, the actual
    size (rejection rate under the Null hypothesis) might be far from the
    nominal size of the test.
    The Games-Howell method uses pairwise t-tests that are robust to differences
    in variances and approximately maintains size unless samples are very
    small.

    .. versionadded:: 0.15
   `   The `use_var` keyword and option for Games-Howell test.

    See Also
    --------
    MultiComparison
    tukeyhsd
    statsmodels.sandbox.stats.multicomp.TukeyHSDResults
    """

    return MultiComparison(endog, groups).tukeyhsd(alpha=alpha,
                                                   use_var=use_var)
