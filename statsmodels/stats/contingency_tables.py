"""
Methods for analyzing contingency tables.
"""

import numpy as np
from scipy import stats
import pandas as pd


def _handle_pandas_square(table):
    """
    Reindex a pandas table so that it becomes square (extending the
    row and column index as needed).
    """

    if not isinstance(table, pd.DataFrame):
        return table

    # If the table is not square, make it square
    if table.shape[0] != table.shape[1]:
        ix = list(set(table.index) | set(table.columns))
        table = table.reindex(ix, axis=0)
        table = table.reindex(ix, axis=1)

    return table

class _bunch(object):
    pass


def symmetry(table, method="bowker", return_object=True):
    """
    Test for symmetry of a joint distribution.

    This procedure tests the null hypothesis that the joint
    distribution is symmetric around the main diagonal, that is

    ..math::

    p_{i, j} = p_{j, i}  for all i, j

    Parameters
    ----------
    table : array_like, 2d, (k, k)
        A square contingency table that contains the count for k
        categories in rows and columns.

    Returns
    -------
    statistic : float
        chisquare test statistic
    p-value : float
        p-value of the test statistic based on chisquare distribution
    df : int
        degrees of freedom of the chisquare distribution

    Notes
    -----
    The implementation is based on the SAS documentation. R includes
    it in `mcnemar.test` if the table is not 2 by 2.  However a more
    direct generalization of the McNemar test to large tables is
    provided by the homogeneity test.

    The p-value is based on the chi-square distribution which requires
    that the sample size is not very small to be a good approximation
    of the true distribution. For 2x2 contingency tables the exact
    distribution can be obtained with `mcnemar`

    See Also
    --------
    mcnemar
    homogeneity
    """

    if method.lower() != "bowker":
        raise ValueError("method for symmetry testing must be 'bowker'")
    table = _handle_pandas_square(table)
    table = np.asarray(table, dtype=np.float64)
    k, k2 = table.shape
    if k != k2:
        raise ValueError('table must be square')

    upp_idx = np.triu_indices(k, 1)

    tril = table.T[upp_idx]   # lower triangle in column order
    triu = table[upp_idx]     # upper triangle in row order

    stat = ((tril - triu)**2 / (tril + triu + 1e-20)).sum()
    df = k * (k-1) / 2.
    pval = stats.chi2.sf(stat, df)

    if return_object:
        b = _bunch()
        b.stat = stat
        b.df = df
        b.pvalue = pvalue
        return b

    return stat, pval, df


def ordinal_association(table, row_scores=None, col_scores=None, method="lbl",
                        return_object=True):
    """
    Assess row/column association in a table with ordinal rows/columns.

    Parameters
    ----------
    table : array-like
        A contingency table.
    row_scores : array-like
        Scores used to weight the rows, defaults to 0, 1, ...
    col_scores : array-like
        Scores used to weight the columns, defaults to 0, 1, ...
    method : string
        Only 'lbl' (Agresti's 'linear by linear' method) is implemented.
    return_object : bool
        If True, return an object containing test results.

    Returns
    -------
    If `return_object` is False, returns the z-score and the p-value
    obtained from the normal distribution.  Otherwise returns a bunch
    containing the test statistic, its null mean and standard
    deviation, the corresponding Z-score, degrees of freedom, and
    p-value.

    Notes
    -----
    Using the default row and column scores gives the Cochran-Armitage
    trend test.

    To assess association in a table with nominal row and column
    factors, a Pearson chi^2 test can be used.
    """

    table = np.asarray(table, dtype=np.float64)

    method = method.lower()
    if method != "lbl":
        raise ValueError("method for asociation must be 'lbl'")

    # Set defaults if needed.
    if row_scores is None:
        row_scores = np.arange(table.shape[0])
    if col_scores is None:
        col_scores = np.arange(table.shape[1])

    if len(row_scores) != table.shape[0]:
        raise ValueError("The length of `row_scores` must match the first dimension of `table`.")

    if len(col_scores) != table.shape[1]:
        raise ValueError("The length of `col_scores` must match the second dimension of `table`.")

    # The test statistic
    stat = np.dot(row_scores, np.dot(table, col_scores))

    # Some needed quantities
    n_obs = table.sum()
    rtot = table.sum(1)
    um = np.dot(row_scores, rtot)
    u2m = np.dot(row_scores**2, rtot)
    ctot = table.sum(0)
    vn = np.dot(col_scores, ctot)
    v2n = np.dot(col_scores**2, ctot)

    # The null mean and variance of the test statistic
    e_stat = um * vn / n_obs
    v_stat = (u2m - um**2 / n_obs) * (v2n - vn**2 / n_obs) / (n_obs - 1)
    sd_stat = np.sqrt(v_stat)

    zscore = (stat - e_stat) / sd_stat
    pvalue = 2 * stats.norm.cdf(-np.abs(zscore))

    if return_object:
        b = _bunch()
        b.stat = stat
        b.stat_e0 = e_stat
        b.stat_sd0 = sd_stat
        b.zscore = zscore
        b.pvalue = pvalue
        return b

    return zscore, pvalue


def stratified_association(table, method='cmh', correction=True, alpha=0.05,
                           return_object=True):
    """
    Assess the common association in a family of contingency tables.

    This type of analysis is usually known as a 'Mantel-Haenszel' or
    'Cochran-Mantel-Haenszel' test, or corresponding estimate of a
    common odds ratio.

    Parameters
    ----------
    table : list
        A list containing 2x2 contingency tables.
    method : string
        Only 'cmh', the Cochran-Mantel-Haenzsel approach is
        currently available.
    correction : bool
        Use a continuity correction.
    alpha : float
        `1 - alpha` is the nominal coverage probability of the confidence
        intervals for the common odds ratio and its log.
    return_object : bool
        Return an object rather than selected quantities.

    Returns
    -------
    If `return_object` is False, returns the chi^2 test statistic and
    p-value for the test of a common association parameter.  Otherwise
    returns a bunch with attributes for the test statistic and
    p-value, estimates of the common odds and risk ratios, and
    standard errors and confidence intervals for the odds ratio.
    """

    table = [x[:, :, None] for x in table]
    table = np.concatenate(table, axis=2).astype(np.float64)

    apb = table[0, 0, :] + table[0, 1, :]
    apc = table[0, 0, :] + table[1, 0, :]
    bpd = table[0, 1, :] + table[1, 1, :]
    cpd = table[1, 0, :] + table[1, 1, :]
    n = table.sum(0).sum(0).astype(np.float64)

    # chi^2 test statistic for assessing that the common odds ratio is
    # zero
    stat = np.sum(table[0, 0, :]  - apb * apc / n)
    stat = np.abs(stat)
    if correction:
        stat -= 0.5
    stat = stat**2
    denom = apb * apc * bpd * cpd / (n**2 * (n - 1))
    denom = np.sum(denom)
    stat /= denom

    # df is always 1
    pvalue = 1 - stats.chi2.cdf(stat, 1)

    if return_object:

        # Estimate the common odds and risk ratios
        ad = table[0, 0, :] * table[1, 1, :]
        bc = table[0, 1, :] * table[1, 0, :]
        acd = table[0, 0, :] * cpd
        cab = table[1, 0, :] * apb
        apd = table[0, 0, :] + table[1, 1, :]
        risk_ratio = np.sum(acd / n) / np.sum(cab / n)
        odds_ratio = np.sum(ad / n) / np.sum(bc / n)

        # Standard error of the common log odds ratio
        adns = np.sum(ad / n)
        bcns = np.sum(bc / n)
        lor_va = np.sum(apd * ad / n**2) / adns**2
        lor_va += np.sum(apd * bc / n**2 + (1 - apd / n) * ad / n) / (adns * bcns)
        lor_va += np.sum((1 - apd / n) * bc / n) / bcns**2
        lor_va /= 2
        lor_se = np.sqrt(lor_va)

        f = -stats.norm.ppf(alpha / 2)

        b = _bunch()
        b.risk_ratio = risk_ratio
        b.odds_ratio = odds_ratio
        b.log_odds_ratio = np.log(odds_ratio)
        b.log_odds_ratio_se = lor_se

        # Confidence intervals for the odds ratio and log odds ratio.
        b.log_odds_ratio_lcb = b.log_odds_ratio - f * lor_se
        b.log_odds_ratio_ucb = b.log_odds_ratio + f * lor_se
        b.odds_ratio_lcb = np.exp(b.log_odds_ratio_lcb)
        b.odds_ratio_ucb = np.exp(b.log_odds_ratio_ucb)
        b.stat = stat
        b.pvalue = pvalue
        return b

    return stat, pvalue


def homogeneity(table, method="stuart_maxwell", return_object=True):
    """
    Compare row and column marginal distributions.

    Parameters
    ----------
    table : array-like
        A square contingency table.
    method : string
        Either 'stuart_maxwell' or 'bhapkar', leading to two different
        estimates of the covariance matrix for the estimated
        difference between the row margins and the column margins.
    return_object : bool
       If True, returns a bunch containing the test statistic,
       p-value, and degrees of freedom as attributes.  Otherwise these
       are returned individually.

    Returns
    -------
    The following attributes, returned as a bunch if return_object is
    True:

    stat : float
        The chi^2 test statistic
    pvalue : float
        The p-value of the test statistic
    df : integer
        The degrees of freedom of the reference distribution

    Notes
    -----
    For a 2x2 table this is equivalent to McNemar's test.  More
    generally the procedure tests the null hypothesis that the
    marginal distribution of the row factor is equal to the marginal
    distribution of the column factor.  For this to be meaningful, the
    two factors must have the same sample space.

    See also
    --------
    mcnemar
    homogeneity
    symmetry
    """
    table = _handle_pandas_square(table)
    table = np.asarray(table, dtype=np.float64)

    if table.shape[0] != table.shape[1]:
        raise ValueError('table must be square')

    if table.shape[0] < 1:
        raise ValueError('table is empty')
    elif table.shape[0] == 1:
        return 0., 1., 0

    method = method.lower()
    if method not in ["bhapkar", "stuart_maxwell"]:
        raise ValueError("method '%s' for homogeneity not known" % method)

    n_obs = table.sum()
    pr = table.astype(np.float64) / n_obs

    # Compute margins, eliminate last row/column so there is no
    # degeneracy
    row = pr.sum(1)[0:-1]
    col = pr.sum(0)[0:-1]
    pr = pr[0:-1, 0:-1]

    # The estimated difference between row and column margins.
    d = col - row

    # The degrees of freedom of the chi^2 reference distribution.
    df = pr.shape[0]

    if method == "bhapkar":
        vmat = -(pr + pr.T) - np.outer(d, d)
        dv = col + row - 2*np.diag(pr) - d**2
        np.fill_diagonal(vmat, dv)
    elif method == "stuart_maxwell":
        vmat = -(pr + pr.T)
        dv = row + col - 2*np.diag(pr)
        np.fill_diagonal(vmat, dv)

    try:
        stat = n_obs * np.dot(d, np.linalg.solve(vmat, d))
    except np.linalg.LinAlgError:
        warnings.warn("Unable to invert covariance matrix")
        return np.nan, np.nan, df

    pvalue = 1 - stats.chi2.cdf(stat, df)

    if return_object:
        b = _bunch()
        b.stat = stat
        b.df = df
        b.pvalue = pvalue
        return b

    return stat, pvalue, df


def mcnemar(table, exact=True, correction=True):
    """
    McNemar test of homogeneity.

    Parameters
    ----------
    table : array-like
        A square contingency table.
    exact : bool
        If exact is true, then the binomial distribution will be used.
        If exact is false, then the chisquare distribution will be
        used, which is the approximation to the distribution of the
        test statistic for large sample sizes.
    correction : bool
        If true, then a continuity correction is used for the chisquare
        distribution (if exact is false.)

    Returns
    -------
    stat : float or int, array
        The test statistic is the chisquare statistic if exact is
        false. If the exact binomial distribution is used, then this
        contains the min(n1, n2), where n1, n2 are cases that are zero
        in one sample but one in the other sample.
    pvalue : float or array
        p-value of the null hypothesis of equal marginal distributions.

    Notes
    -----
    This is a special case of Cochran's Q test, and of the homogeneity
    test. The results when the chisquare distribution is used are
    identical, except for continuity correction.
    """

    table = _handle_pandas_square(table)
    table = np.asarray(table, dtype=np.float64)
    n1, n2 = table[0, 1], table[1, 0]

    if exact:
        stat = np.minimum(n1, n2)
        # binom is symmetric with p=0.5
        pval = stats.binom.cdf(stat, n1 + n2, 0.5) * 2
        pval = np.minimum(pval, 1)  # limit to 1 if n1==n2
    else:
        corr = int(correction) # convert bool to 0 or 1
        stat = (np.abs(n1 - n2) - corr)**2 / (1. * (n1 + n2))
        df = 1
        pval = stats.chi2.sf(stat, df)
    return stat, pval


def cochrans_q(x, return_object=True):
    """
    Cochran's Q test for identical binomial proportions.

    Parameters
    ----------
    x : array_like, 2d (N, k)
        data with N cases and k variables
    return_object : boolean
        Return values as bunch instead of as individual values.

    Returns
    -------
    Returns a bunch containing the following attributes, or the
    individual values according to the value of `return_object`.

    q_stat : float
       test statistic
    pvalue : float
       pvalue from the chisquare distribution

    Notes
    -----
    Cochran's Q is a k-sample extension of the McNemar test. If there
    are only two groups, then Cochran's Q test and the McNemar test
    are equivalent.

    The procedure tests that the probability of success is the same
    for every group.  The alternative hypothesis is that at least two
    groups have a different probability of success.

    In Wikipedia terminology, rows are blocks and columns are
    treatments.  The number of rows N, should be large for the
    chisquare distribution to be a good approximation.

    The Null hypothesis of the test is that all treatments have the
    same effect.

    References
    ----------
    http://en.wikipedia.org/wiki/Cochran_test
    SAS Manual for NPAR TESTS
    """

    x = np.asarray(x, dtype=np.float64)
    gruni = np.unique(x)
    N, k = x.shape
    count_row_success = (x == gruni[-1]).sum(1, float)
    count_col_success = (x == gruni[-1]).sum(0, float)
    count_row_ss = count_row_success.sum()
    count_col_ss = count_col_success.sum()
    assert count_row_ss == count_col_ss  #just a calculation check

    # From the SAS manual
    q_stat = (k-1) * (k *  np.sum(count_col_success**2) - count_col_ss**2) \
             / (k * count_row_ss - np.sum(count_row_success**2))

    # Note: the denominator looks just like k times the variance of
    # the columns

    # Wikipedia uses a different, but equivalent expression
    #q_stat = (k-1) * (k *  np.sum(count_row_success**2) - count_row_ss**2) \
    #         / (k * count_col_ss - np.sum(count_col_success**2))

    df = k - 1
    pvalue = stats.chi2.sf(q_stat, df)

    if return_object:
        b = _bunch()
        b.stat = q_stat
        b.df = df
        b.pvalue = pvalue
        return b

    return q_stat, pvalue, df
