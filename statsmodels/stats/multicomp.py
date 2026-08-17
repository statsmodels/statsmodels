"""

Created on Fri Mar 30 18:27:25 2012
Author: Josef Perktold
"""

from statsmodels.sandbox.stats.multicomp import MultiComparison, tukeyhsd

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
