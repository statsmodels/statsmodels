"""
Methods for analyzing two-way contingency tables (i.e. frequency
tables for observations that are cross-classified with respect to two
categorical variables).

The main classes are:

  * Table : implements methods that can be applied to any two-way
  contingency table.

  * SquareTable : implements methods that can be applied to a square
  two-way contingency table.

  * Table2x2 : implements methods that can be applied to a 2x2
  contingency table.

  * StratifiedTable : implements methods that can be applied to a
  collection of 2x2 contingency tables.

  * MultipleResponseTable : implements methods that can be applied to contingency
  tables that contain "multiple response" variables,
  i.e. factors where a single observation may have more than one level.
  For example, a "select all that apply" question on a survey.

  * Factor : A data container class that implements methods for
  processing and reshaping data into a format that is amenable
  for use in the MultipleResponseTable class.

Also contains functions for conducting McNemar's test
and Cochran's q test.

Note that the inference procedures may depend on how the data were
sampled.  In general the observed units are assumed to be
independent and identically distributed.
"""

from __future__ import division

import itertools
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from numpy import linalg
from scipy import stats
from scipy.stats import chi2_contingency, chi2
from statsmodels.compat import asunicode

from statsmodels import iolib
from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

DEFAULT_DEDUPLICATION_PADDING = "'"


def _make_df_square(table):
    """
    Reindex a pandas DataFrame so that it becomes square, meaning that
    the row and column indices contain the same values, in the same
    order.  The row and column index are extended to achieve this.
    """

    if not isinstance(table, pd.DataFrame):
        return table

    # If the table is not square, make it square
    if not table.index.equals(table.columns):
        ix = list(set(table.index) | set(table.columns))
        ix.sort()
        table = table.reindex(index=ix, columns=ix, fill_value=0)

    # Ensures that the rows and columns are in the same order.
    table = table.reindex(table.columns)

    return table


class _Bunch(object):
    """
    A general-purpose container class for analysis results.

    Notes
    -----
    See (Alex Martelli's article explaining the "Bunch" concept
    for more background)[http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/]
    """

    def __repr__(self):
        return "<bunch containing results, print to see contents>"

    def __str__(self):
        ky = [k for k, _ in self.__dict__.items()]
        ky.sort()
        m = max([len(k) for k in ky])
        tab = []
        f = "{:" + str(m) + "}   {}"
        for k in ky:
            tab.append(f.format(k, self.__dict__[k]))
        return "\n".join(tab)

class ContingencyTableNominalIndependenceResult(_Bunch):
    """
    A container class specifically intended to hold the results of a nominal
    independence test on a traditional contingency table.
    """
    def __repr__(self):
        template = ("<bunch object containing contingency table "
                    "independence results at {}>")
        return template.format(id(self))

    def __str__(self):
        template = ("Contingency Table Independence Result:\n"
                    "chi-squared statistic: {statistic}\n"
                    "degrees of freedom: {df}\n"
                    "p value: {pvalue}\n")
        output = template.format(statistic=self.statistic,
                                df=self.df,
                                pvalue=self.pvalue)
        return output


class MRCVTableNominalIndependenceResult(_Bunch):
    """
    A container class specifically intended to hold the results of
    independence tests run on an MRCV contingency table,
    i.e. a contingency table containing "multiple response" variables

    Attributes
    ----------
    p_value_overall : float
        Probability that row factor is independent of column factor.
    p_values_cellwise : pd.DataFrame of floats
        Probabilities that each specific combination of levels in the
        the row variable is independent of each specific level in
        the column variables

    Notes
    -----
    If repeated samples were drawn from the underlying population
    and the row and column variables were in fact independent,
    these p-values would represent the fraction of those samples
    in which the observed relationship between the row and column
    variables would be at least as extreme as it is
    in the provided data.
    """
    def __repr__(self):
        return ("<bunch object containing multiple-response contingency"
                " table independence results at {}>".format(id(self)))

    def __str__(self):
        template = ("Multiple Response Contingency Table "
                    "Independence Result:\n"
                    "table p value: {p_value_overall}\n"
                    "cellwise p values: {p_values_cellwise}")
        output = template.format(p_value_overall=self.p_value_overall,
                                 p_values_cellwise=self.p_values_cellwise)
        return output


class Table(object):
    """
    A two-way contingency table.

    Parameters
    ----------
    table : array-like
        A contingency table.
    shift_zeros : boolean
        If True and any cell count is zero, add 0.5 to all values
        in the table.

    Attributes
    ----------
    table_orig : array-like
        The original table is cached as `table_orig`.
    marginal_probabilities : tuple of two ndarrays
        The estimated row and column marginal distributions.
    independence_probabilities : ndarray
        Estimated cell probabilities under row/column independence.
    fittedvalues : ndarray
        Fitted values under independence.
    resid_pearson : ndarray
        The Pearson residuals under row/column independence.
    standardized_resids : ndarray
        Residuals for the independent row/column model with approximate
        unit variance.
    chi2_contribs : ndarray
        The contribution of each cell to the chi^2 statistic.
    local_logodds_ratios : ndarray
        The local log odds ratios are calculated for each 2x2 subtable
        formed from adjacent rows and columns.
    local_oddsratios : ndarray
        The local odds ratios are calculated from each 2x2 subtable
        formed from adjacent rows and columns.
    cumulative_log_oddsratios : ndarray
        The cumulative log odds ratio at a given pair of thresholds is
        calculated by reducing the table to a 2x2 table based on
        dichotomizing the rows and columns at the given thresholds.
        The table of cumulative log odds ratios presents all possible
        cumulative log odds ratios that can be formed from a given
        table.
    cumulative_oddsratios : ndarray
        The cumulative odds ratios are calculated by reducing the
        table to a 2x2 table based on cutting the rows and columns at
        a given point.  The table of cumulative odds ratios presents
        all possible cumulative odds ratios that can be formed from a
        given table.

    See also
    --------
    statsmodels.graphics.mosaicplot.mosaic
    scipy.stats.chi2_contingency

    Notes
    -----
    The inference procedures used here are all based on a sampling
    model in which the units are independent and identically
    distributed, with each unit being classified with respect to two
    categorical variables.

    References
    ----------
    Definitions of residuals:
        https://onlinecourses.science.psu.edu/stat504/node/86
    """

    def __init__(self, table, shift_zeros=True):

        self.table_orig = table
        self.table = np.asarray(table, dtype=np.float64)

        if shift_zeros and (self.table.min() == 0):
            self.table = self.table + 0.5

    def __str__(self):
        s = "A %dx%d contingency table with counts:\n" % tuple(self.table.shape)
        s += np.array_str(self.table)
        return s

    @classmethod
    def from_data(cls, data, shift_zeros=True):
        """
        Construct a Table object from data.

        Parameters
        ----------
        data : array-like
            The raw data, from which a contingency table is constructed
            using the first two columns.
        shift_zeros : boolean
            If True and any cell count is zero, add 0.5 to all values
            in the table.

        Returns
        -------
        A Table instance.
        """

        if isinstance(data, pd.DataFrame):
            table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        else:
            table = pd.crosstab(data[:, 0], data[:, 1])
        return cls(table, shift_zeros)

    def test_nominal_association(self):
        """
        Assess independence for nominal factors.

        Assessment of independence between rows and columns using
        chi^2 testing.  The rows and columns are treated as nominal
        (unordered) categorical variables.

        Returns
        -------
        A bunch containing the following attributes:

        statistic : float
            The chi^2 test statistic.
        df : integer
            The degrees of freedom of the reference distribution
        pvalue : float
            The p-value for the test.
        """

        statistic = np.asarray(self.chi2_contribs).sum()
        df = np.prod(np.asarray(self.table.shape) - 1)
        pvalue = 1 - stats.chi2.cdf(statistic, df)
        b = ContingencyTableNominalIndependenceResult()
        b.statistic = statistic
        b.df = df
        b.pvalue = pvalue
        return b

    def test_ordinal_association(self, row_scores=None, col_scores=None):
        """
        Assess independence between two ordinal variables.

        This is the 'linear by linear' association test, which uses
        weights or scores to target the test to have more power
        against ordered alternatives.

        Parameters
        ----------
        row_scores : array-like
            An array of numeric row scores
        col_scores : array-like
            An array of numeric column scores

        Returns
        -------
        A bunch with the following attributes:

        statistic : float
            The test statistic.
        null_mean : float
            The expected value of the test statistic under the null
            hypothesis.
        null_sd : float
            The standard deviation of the test statistic under the
            null hypothesis.
        zscore : float
            The Z-score for the test statistic.
        pvalue : float
            The p-value for the test.

        Notes
        -----
        The scores define the trend to which the test is most sensitive.

        Using the default row and column scores gives the
        Cochran-Armitage trend test.
        """

        if row_scores is None:
            row_scores = np.arange(self.table.shape[0])

        if col_scores is None:
            col_scores = np.arange(self.table.shape[1])

        if len(row_scores) != self.table.shape[0]:
            msg = ("The length of `row_scores` must match the first " +
                   "dimension of `table`.")
            raise ValueError(msg)

        if len(col_scores) != self.table.shape[1]:
            msg = ("The length of `col_scores` must match the second " +
                   "dimension of `table`.")
            raise ValueError(msg)

        # The test statistic
        statistic = np.dot(row_scores, np.dot(self.table, col_scores))

        # Some needed quantities
        n_obs = self.table.sum()
        rtot = self.table.sum(1)
        um = np.dot(row_scores, rtot)
        u2m = np.dot(row_scores**2, rtot)
        ctot = self.table.sum(0)
        vn = np.dot(col_scores, ctot)
        v2n = np.dot(col_scores**2, ctot)

        # The null mean and variance of the test statistic
        e_stat = um * vn / n_obs
        v_stat = (u2m - um**2 / n_obs) * (v2n - vn**2 / n_obs) / (n_obs - 1)
        sd_stat = np.sqrt(v_stat)

        zscore = (statistic - e_stat) / sd_stat
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))

        b = _Bunch()
        b.statistic = statistic
        b.null_mean = e_stat
        b.null_sd = sd_stat
        b.zscore = zscore
        b.pvalue = pvalue
        return b

    @cache_readonly
    def marginal_probabilities(self):
        # docstring for cached attributes in init above

        n = self.table.sum()
        row = self.table.sum(1) / n
        col = self.table.sum(0) / n

        if isinstance(self.table_orig, pd.DataFrame):
            row = pd.Series(row, self.table_orig.index)
            col = pd.Series(col, self.table_orig.columns)

        return row, col

    @cache_readonly
    def independence_probabilities(self):
        # docstring for cached attributes in init above

        row, col = self.marginal_probabilities
        itab = np.outer(row, col)

        if isinstance(self.table_orig, pd.DataFrame):
            itab = pd.DataFrame(itab, self.table_orig.index,
                                self.table_orig.columns)

        return itab

    @cache_readonly
    def fittedvalues(self):
        # docstring for cached attributes in init above

        probs = self.independence_probabilities
        fit = self.table.sum() * probs
        return fit

    @cache_readonly
    def resid_pearson(self):
        # docstring for cached attributes in init above

        fit = self.fittedvalues
        resids = (self.table - fit) / np.sqrt(fit)
        return resids

    @cache_readonly
    def standardized_resids(self):
        # docstring for cached attributes in init above

        row, col = self.marginal_probabilities
        sresids = self.resid_pearson / np.sqrt(np.outer(1 - row, 1 - col))
        return sresids

    @cache_readonly
    def chi2_contribs(self):
        # docstring for cached attributes in init above

        return self.resid_pearson**2

    @cache_readonly
    def local_log_oddsratios(self):
        # docstring for cached attributes in init above

        ta = self.table.copy()
        a = ta[0:-1, 0:-1]
        b = ta[0:-1, 1:]
        c = ta[1:, 0:-1]
        d = ta[1:, 1:]
        tab = np.log(a) + np.log(d) - np.log(b) - np.log(c)
        rslt = np.empty(self.table.shape, np.float64)
        rslt *= np.nan
        rslt[0:-1, 0:-1] = tab

        if isinstance(self.table_orig, pd.DataFrame):
            rslt = pd.DataFrame(rslt, index=self.table_orig.index,
                                columns=self.table_orig.columns)

        return rslt

    @cache_readonly
    def local_oddsratios(self):
        # docstring for cached attributes in init above

        return np.exp(self.local_log_oddsratios)

    @cache_readonly
    def cumulative_log_oddsratios(self):
        # docstring for cached attributes in init above

        ta = self.table.cumsum(0).cumsum(1)

        a = ta[0:-1, 0:-1]
        b = ta[0:-1, -1:] - a
        c = ta[-1:, 0:-1] - a
        d = ta[-1, -1] - (a + b + c)

        tab = np.log(a) + np.log(d) - np.log(b) - np.log(c)
        rslt = np.empty(self.table.shape, np.float64)
        rslt *= np.nan
        rslt[0:-1, 0:-1] = tab

        if isinstance(self.table_orig, pd.DataFrame):
            rslt = pd.DataFrame(rslt, index=self.table_orig.index,
                                columns=self.table_orig.columns)

        return rslt

    @cache_readonly
    def cumulative_oddsratios(self):
        # docstring for cached attributes in init above

        return np.exp(self.cumulative_log_oddsratios)


class SquareTable(Table):
    """
    Methods for analyzing a square contingency table.

    Parameters
    ----------
    table : array-like
        A square contingency table, or DataFrame that is converted
        to a square form.
    shift_zeros : boolean
        If True and any cell count is zero, add 0.5 to all values
        in the table.

    These methods should only be used when the rows and columns of the
    table have the same categories.  If `table` is provided as a
    Pandas DataFrame, the row and column indices will be extended to
    create a square table, inserting zeros where a row or column is
    missing.  Otherwise the table should be provided in a square form,
    with the (implicit) row and column categories appearing in the
    same order.
    """

    def __init__(self, table, shift_zeros=True):
        table = _make_df_square(table)  # Non-pandas passes through
        k1, k2 = table.shape
        if k1 != k2:
            raise ValueError('table must be square')

        super(SquareTable, self).__init__(table, shift_zeros)

    def symmetry(self, method="bowker"):
        """
        Test for symmetry of a joint distribution.

        This procedure tests the null hypothesis that the joint
        distribution is symmetric around the main diagonal, that is

        .. math::

        p_{i, j} = p_{j, i}  for all i, j

        Returns
        -------
        A bunch with attributes:

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
        direct generalization of the McNemar test to larger tables is
        provided by the homogeneity test (TableSymmetry.homogeneity).

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

        k = self.table.shape[0]
        upp_idx = np.triu_indices(k, 1)

        tril = self.table.T[upp_idx]   # lower triangle in column order
        triu = self.table[upp_idx]     # upper triangle in row order

        statistic = ((tril - triu)**2 / (tril + triu + 1e-20)).sum()
        df = k * (k-1) / 2.
        pvalue = stats.chi2.sf(statistic, df)

        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        b.df = df

        return b

    def homogeneity(self, method="stuart_maxwell"):
        """
        Compare row and column marginal distributions.

        Parameters
        ----------
        method : string
            Either 'stuart_maxwell' or 'bhapkar', leading to two different
            estimates of the covariance matrix for the estimated
            difference between the row margins and the column margins.

        Returns a bunch with attributes:

        statistic : float
            The chi^2 test statistic
        pvalue : float
            The p-value of the test statistic
        df : integer
            The degrees of freedom of the reference distribution

        Notes
        -----
        For a 2x2 table this is equivalent to McNemar's test.  More
        generally the procedure tests the null hypothesis that the
        marginal distribution of the row factor is equal to the
        marginal distribution of the column factor.  For this to be
        meaningful, the two factors must have the same sample space
        (i.e. the same categories).
        """

        if self.table.shape[0] < 1:
            raise ValueError('table is empty')
        elif self.table.shape[0] == 1:
            b = _Bunch()
            b.statistic = 0
            b.pvalue = 1
            b.df = 0
            return b

        method = method.lower()
        if method not in ["bhapkar", "stuart_maxwell"]:
            raise ValueError("method '%s' for homogeneity not known" % method)

        n_obs = self.table.sum()
        pr = self.table.astype(np.float64) / n_obs

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
            statistic = n_obs * np.dot(d, np.linalg.solve(vmat, d))
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn("Unable to invert covariance matrix",
                          SingularMatrixWarning)
            b = _Bunch()
            b.statistic = np.nan
            b.pvalue = np.nan
            b.df = df
            return b

        pvalue = 1 - stats.chi2.cdf(statistic, df)

        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        b.df = df

        return b

    def summary(self, alpha=0.05, float_format="%.3f"):
        """
        Produce a summary of the analysis.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the interval.
        float_format : string
            Used to format numeric values in the table.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """

        fmt = float_format

        headers = ["Statistic", "P-value", "DF"]
        stubs = ["Symmetry", "Homogeneity"]
        sy = self.symmetry()
        hm = self.homogeneity()
        data = [[fmt % sy.statistic, fmt % sy.pvalue, '%d' % sy.df],
                [fmt % hm.statistic, fmt % hm.pvalue, '%d' % hm.df]]
        tab = iolib.SimpleTable(data, headers, stubs, data_aligns="r",
                                table_dec_above='')

        return tab


class Table2x2(SquareTable):
    """
    Analyses that can be performed on a 2x2 contingency table.

    Parameters
    ----------
    table : array-like
        A 2x2 contingency table
    shift_zeros : boolean
        If true, 0.5 is added to all cells of the table if any cell is
        equal to zero.

    Attributes
    ----------
    log_oddsratio : float
        The log odds ratio of the table.
    log_oddsratio_se : float
        The asymptotic standard error of the estimated log odds ratio.
    oddsratio : float
        The odds ratio of the table.
    riskratio : float
        The ratio between the risk in the first row and the risk in
        the second row.  Column 0 is interpreted as containing the
        number of occurences of the event of interest.
    log_riskratio : float
        The estimated log risk ratio for the table.
    log_riskratio_se : float
        The standard error of the estimated log risk ratio for the
        table.

    Notes
    -----
    The inference procedures used here are all based on a sampling
    model in which the units are independent and identically
    distributed, with each unit being classified with respect to two
    categorical variables.

    Note that for the risk ratio, the analysis is not symmetric with
    respect to the rows and columns of the contingency table.  The two
    rows define population subgroups, column 0 is the number of
    'events', and column 1 is the number of 'non-events'.
    """

    def __init__(self, table, shift_zeros=True):

        if type(table) is list:
            table = np.asarray(table)

        if (table.ndim != 2) or (table.shape[0] != 2) or (table.shape[1] != 2):
            raise ValueError("Table2x2 takes a 2x2 table as input.")

        super(Table2x2, self).__init__(table, shift_zeros)

    @classmethod
    def from_data(cls, data, shift_zeros=True):
        """
        Construct a Table object from data.

        Parameters
        ----------
        data : array-like
            The raw data, the first column defines the rows and the
            second column defines the columns.
        shift_zeros : boolean
            If True, and if there are any zeros in the contingency
            table, add 0.5 to all four cells of the table.
        """

        if isinstance(data, pd.DataFrame):
            table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        else:
            table = pd.crosstab(data[:, 0], data[:, 1])
        return cls(table, shift_zeros)

    @cache_readonly
    def log_oddsratio(self):
        # docstring for cached attributes in init above

        f = self.table.flatten()
        return np.dot(np.log(f), np.r_[1, -1, -1, 1])

    @cache_readonly
    def oddsratio(self):
        # docstring for cached attributes in init above

        return (self.table[0, 0] * self.table[1, 1] /
                (self.table[0, 1] * self.table[1, 0]))

    @cache_readonly
    def log_oddsratio_se(self):
        # docstring for cached attributes in init above

        return np.sqrt(np.sum(1 / self.table))

    def oddsratio_pvalue(self, null=1):
        """
        P-value for a hypothesis test about the odds ratio.

        Parameters
        ----------
        null : float
            The null value of the odds ratio.
        """

        return self.log_oddsratio_pvalue(np.log(null))

    def log_oddsratio_pvalue(self, null=0):
        """
        P-value for a hypothesis test about the log odds ratio.

        Parameters
        ----------
        null : float
            The null value of the log odds ratio.
        """

        zscore = (self.log_oddsratio - null) / self.log_oddsratio_se
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        return pvalue

    def log_oddsratio_confint(self, alpha=0.05, method="normal"):
        """
        A confidence level for the log odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """

        f = -stats.norm.ppf(alpha / 2)
        lor = self.log_oddsratio
        se = self.log_oddsratio_se
        lcb = lor - f * se
        ucb = lor + f * se
        return lcb, ucb

    def oddsratio_confint(self, alpha=0.05, method="normal"):
        """
        A confidence interval for the odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        lcb, ucb = self.log_oddsratio_confint(alpha, method=method)
        return np.exp(lcb), np.exp(ucb)

    @cache_readonly
    def riskratio(self):
        # docstring for cached attributes in init above

        p = self.table[:, 0] / self.table.sum(1)
        return p[0] / p[1]

    @cache_readonly
    def log_riskratio(self):
        # docstring for cached attributes in init above

        return np.log(self.riskratio)

    @cache_readonly
    def log_riskratio_se(self):
        # docstring for cached attributes in init above

        n = self.table.sum(1)
        p = self.table[:, 0] / n
        va = np.sum((1 - p) / (n*p))
        return np.sqrt(va)

    def riskratio_pvalue(self, null=1):
        """
        p-value for a hypothesis test about the risk ratio.

        Parameters
        ----------
        null : float
            The null value of the risk ratio.
        """

        return self.log_riskratio_pvalue(np.log(null))

    def log_riskratio_pvalue(self, null=0):
        """
        p-value for a hypothesis test about the log risk ratio.

        Parameters
        ----------
        null : float
            The null value of the log risk ratio.
        """

        zscore = (self.log_riskratio - null) / self.log_riskratio_se
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        return pvalue

    def log_riskratio_confint(self, alpha=0.05, method="normal"):
        """
        A confidence interval for the log risk ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        f = -stats.norm.ppf(alpha / 2)
        lrr = self.log_riskratio
        se = self.log_riskratio_se
        lcb = lrr - f * se
        ucb = lrr + f * se
        return lcb, ucb

    def riskratio_confint(self, alpha=0.05, method="normal"):
        """
        A confidence interval for the risk ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        lcb, ucb = self.log_riskratio_confint(alpha, method=method)
        return np.exp(lcb), np.exp(ucb)

    def summary(self, alpha=0.05, float_format="%.3f", method="normal"):
        """
        Summarizes results for a 2x2 table analysis.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the confidence
            intervals.
        float_format : string
            Used to format the numeric values in the table.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """

        def fmt(x):
            if type(x) is str:
                return x
            return float_format % x

        headers = ["Estimate", "SE", "LCB", "UCB", "p-value"]
        stubs = ["Odds ratio", "Log odds ratio", "Risk ratio",
                 "Log risk ratio"]

        lcb1, ucb1 = self.oddsratio_confint(alpha, method)
        lcb2, ucb2 = self.log_oddsratio_confint(alpha, method)
        lcb3, ucb3 = self.riskratio_confint(alpha, method)
        lcb4, ucb4 = self.log_riskratio_confint(alpha, method)
        data = [[fmt(x) for x in [self.oddsratio, "", lcb1, ucb1,
                                  self.oddsratio_pvalue()]],
                [fmt(x) for x in [self.log_oddsratio, self.log_oddsratio_se,
                                  lcb2, ucb2, self.oddsratio_pvalue()]],
                [fmt(x) for x in [self.riskratio, "", lcb3, ucb3,
                                  self.riskratio_pvalue()]],
                [fmt(x) for x in [self.log_riskratio, self.log_riskratio_se,
                                  lcb4, ucb4, self.riskratio_pvalue()]]]
        tab = iolib.SimpleTable(data, headers, stubs, data_aligns="r",
                                table_dec_above='')
        return tab


class StratifiedTable(object):
    """
    Analyses for a collection of 2x2 contingency tables.

    Such a collection may arise by stratifying a single 2x2 table with
    respect to another factor.  This class implements the
    'Cochran-Mantel-Haenszel' and 'Breslow-Day' procedures for
    analyzing collections of 2x2 contingency tables.

    Parameters
    ----------
    tables : list or ndarray
        Either a list containing several 2x2 contingency tables, or
        a 2x2xk ndarray in which each slice along the third axis is a
        2x2 contingency table.

    Attributes
    ----------
    logodds_pooled : float
        An estimate of the pooled log odds ratio.  This is the
        Mantel-Haenszel estimate of an odds ratio that is common to
        all the tables.
    log_oddsratio_se : float
        The estimated standard error of the pooled log odds ratio,
        following Robins, Breslow and Greenland (Biometrics
        42:311-323).
    oddsratio_pooled : float
        An estimate of the pooled odds ratio.  This is the
        Mantel-Haenszel estimate of an odds ratio that is common to
        all tables.
    risk_pooled : float
        An estimate of the pooled risk ratio.  This is an estimate of
        a risk ratio that is common to all the tables.

    Notes
    -----
    This results are based on a sampling model in which the units are
    independent both within and between strata.
    """

    def __init__(self, tables, shift_zeros=False):

        if isinstance(tables, np.ndarray):
            sp = tables.shape
            if (len(sp) != 3) or (sp[0] != 2) or (sp[1] != 2):
                raise ValueError("If an ndarray, argument must be 2x2xn")
            table = tables
        else:
            # Create a data cube
            table = np.dstack(tables).astype(np.float64)

        if shift_zeros:
            zx = (table == 0).sum(0).sum(0)
            ix = np.flatnonzero(zx > 0)
            if len(ix) > 0:
                table = table.copy()
                table[:, :, ix] += 0.5

        self.table = table

        self._cache = resettable_cache()

        # Quantities to precompute.  Table entries are [[a, b], [c,
        # d]], 'ad' is 'a * d', 'apb' is 'a + b', 'dma' is 'd - a',
        # etc.
        self._apb = table[0, 0, :] + table[0, 1, :]
        self._apc = table[0, 0, :] + table[1, 0, :]
        self._bpd = table[0, 1, :] + table[1, 1, :]
        self._cpd = table[1, 0, :] + table[1, 1, :]
        self._ad = table[0, 0, :] * table[1, 1, :]
        self._bc = table[0, 1, :] * table[1, 0, :]
        self._apd = table[0, 0, :] + table[1, 1, :]
        self._dma = table[1, 1, :] - table[0, 0, :]
        self._n = table.sum(0).sum(0)

    @classmethod
    def from_data(cls, var1, var2, strata, data):
        """
        Construct a StratifiedTable object from data.

        Parameters
        ----------
        var1 : int or string
            The column index or name of `data` specifying the variable
            defining the rows of the contingency table.  The variable
            must have only two distinct values.
        var2 : int or string
            The column index or name of `data` specifying the variable
            defining the columns of the contingency table.  The variable
            must have only two distinct values.
        strata : int or string
            The column index or name of `data` specifying the variable
            defining the strata.
        data : array-like
            The raw data.  A cross-table for analysis is constructed
            from the first two columns.

        Returns
        -------
        A StratifiedTable instance.
        """

        if not isinstance(data, pd.DataFrame):
            data1 = pd.DataFrame(index=np.arange(data.shape[0]),
                                 columns=[var1, var2, strata])
            data1.loc[:, var1] = data[:, var1]
            data1.loc[:, var2] = data[:, var2]
            data1.loc[:, strata] = data[:, strata]
        else:
            data1 = data[[var1, var2, strata]]

        gb = data1.groupby(strata).groups
        tables = []
        for g in gb:
            ii = gb[g]
            tab = pd.crosstab(data1.loc[ii, var1], data1.loc[ii, var2])
            if (tab.shape != np.r_[2, 2]).any():
                msg = "Invalid table dimensions"
                raise ValueError(msg)
            tables.append(np.asarray(tab))

        return cls(tables)

    def test_null_odds(self, correction=False):
        """
        Test that all tables have odds ratio equal to 1.

        This is the 'Mantel-Haenszel' test.

        Parameters
        ----------
        correction : boolean
            If True, use the continuity correction when calculating the
            test statistic.

        Returns
        -------
        A bunch containing the chi^2 test statistic and p-value.
        """

        statistic = np.sum(self.table[0, 0, :] -
                           self._apb * self._apc / self._n)
        statistic = np.abs(statistic)
        if correction:
            statistic -= 0.5
        statistic = statistic**2
        denom = self._apb * self._apc * self._bpd * self._cpd
        denom /= (self._n**2 * (self._n - 1))
        denom = np.sum(denom)
        statistic /= denom

        # df is always 1
        pvalue = 1 - stats.chi2.cdf(statistic, 1)

        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue

        return b

    @cache_readonly
    def oddsratio_pooled(self):
        # doc for cached attributes in init above

        odds_ratio = np.sum(self._ad / self._n) / np.sum(self._bc / self._n)
        return odds_ratio

    @cache_readonly
    def logodds_pooled(self):
        # doc for cached attributes in init above

        return np.log(self.oddsratio_pooled)

    @cache_readonly
    def risk_pooled(self):
        # doc for cached attributes in init above

        acd = self.table[0, 0, :] * self._cpd
        cab = self.table[1, 0, :] * self._apb

        rr = np.sum(acd / self._n) / np.sum(cab / self._n)
        return rr

    @cache_readonly
    def logodds_pooled_se(self):
        # doc for cached attributes in init above

        adns = np.sum(self._ad / self._n)
        bcns = np.sum(self._bc / self._n)
        lor_va = np.sum(self._apd * self._ad / self._n**2) / adns**2
        mid = self._apd * self._bc / self._n**2
        mid += (1 - self._apd / self._n) * self._ad / self._n
        mid = np.sum(mid)
        mid /= (adns * bcns)
        lor_va += mid
        lor_va += np.sum((1 - self._apd / self._n) *
                         self._bc / self._n) / bcns**2
        lor_va /= 2
        lor_se = np.sqrt(lor_va)
        return lor_se

    def logodds_pooled_confint(self, alpha=0.05, method="normal"):
        """
        A confidence interval for the pooled log odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            interval.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.

        Returns
        -------
        lcb : float
            The lower confidence limit.
        ucb : float
            The upper confidence limit.
        """

        lor = np.log(self.oddsratio_pooled)
        lor_se = self.logodds_pooled_se

        f = -stats.norm.ppf(alpha / 2)

        lcb = lor - f * lor_se
        ucb = lor + f * lor_se

        return lcb, ucb

    def oddsratio_pooled_confint(self, alpha=0.05, method="normal"):
        """
        A confidence interval for the pooled odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            interval.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.

        Returns
        -------
        lcb : float
            The lower confidence limit.
        ucb : float
            The upper confidence limit.
        """

        lcb, ucb = self.logodds_pooled_confint(alpha, method=method)
        lcb = np.exp(lcb)
        ucb = np.exp(ucb)
        return lcb, ucb

    def test_equal_odds(self, adjust=False):
        """
        Test that all odds ratios are identical.

        This is the 'Breslow-Day' testing procedure.

        Parameters
        ----------
        adjust : boolean
            Use the 'Tarone' adjustment to achieve the chi^2
            asymptotic distribution.

        Returns
        -------
        A bunch containing the following attributes:

        statistic : float
            The chi^2 test statistic.
        p-value : float
            The p-value for the test.
        """

        table = self.table

        r = self.oddsratio_pooled
        a = 1 - r
        b = r * (self._apb + self._apc) + self._dma
        c = -r * self._apb * self._apc

        # Expected value of first cell
        e11 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

        # Variance of the first cell
        v11 = (1 / e11 + 1 / (self._apc - e11) + 1 / (self._apb - e11) +
               1 / (self._dma + e11))
        v11 = 1 / v11

        statistic = np.sum((table[0, 0, :] - e11)**2 / v11)

        if adjust:
            adj = table[0, 0, :].sum() - e11.sum()
            adj = adj**2
            adj /= np.sum(v11)
            statistic -= adj

        pvalue = 1 - stats.chi2.cdf(statistic, table.shape[2] - 1)

        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue

        return b

    def summary(self, alpha=0.05, float_format="%.3f", method="normal"):
        """
        A summary of all the main results.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence intervals.
        float_format : string
            Used for formatting numeric values in the summary.
        method : string
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """

        def fmt(x):
            if type(x) is str:
                return x
            return float_format % x

        co_lcb, co_ucb = self.oddsratio_pooled_confint(
            alpha=alpha, method=method)
        clo_lcb, clo_ucb = self.logodds_pooled_confint(
            alpha=alpha, method=method)
        headers = ["Estimate", "LCB", "UCB"]
        stubs = ["Pooled odds", "Pooled log odds", "Pooled risk ratio", ""]
        data = [[fmt(x) for x in [self.oddsratio_pooled, co_lcb, co_ucb]],
                [fmt(x) for x in [self.logodds_pooled, clo_lcb, clo_ucb]],
                [fmt(x) for x in [self.risk_pooled, "", ""]],
                ['', '', '']]
        tab1 = iolib.SimpleTable(data, headers, stubs, data_aligns="r",
                                 table_dec_above='')

        headers = ["Statistic", "P-value", ""]
        stubs = ["Test of OR=1", "Test constant OR"]
        rslt1 = self.test_null_odds()
        rslt2 = self.test_equal_odds()
        data = [[fmt(x) for x in [rslt1.statistic, rslt1.pvalue, ""]],
                [fmt(x) for x in [rslt2.statistic, rslt2.pvalue, ""]]]
        tab2 = iolib.SimpleTable(data, headers, stubs, data_aligns="r")
        tab1.extend(tab2)

        headers = ["", "", ""]
        stubs = ["Number of tables", "Min n", "Max n", "Avg n", "Total n"]
        ss = self.table.sum(0).sum(0)
        data = [["%d" % self.table.shape[2], '', ''],
                ["%d" % min(ss), '', ''],
                ["%d" % max(ss), '', ''],
                ["%.0f" % np.mean(ss), '', ''],
                ["%d" % sum(ss), '', '', '']]
        tab3 = iolib.SimpleTable(data, headers, stubs, data_aligns="r")
        tab1.extend(tab3)

        return tab1


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
    A bunch with attributes:

    statistic : float or int, array
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

    table = _make_df_square(table)
    table = np.asarray(table, dtype=np.float64)
    n1, n2 = table[0, 1], table[1, 0]

    if exact:
        statistic = np.minimum(n1, n2)
        # binom is symmetric with p=0.5
        pvalue = stats.binom.cdf(statistic, n1 + n2, 0.5) * 2
        pvalue = np.minimum(pvalue, 1)  # limit to 1 if n1==n2
    else:
        corr = int(correction)  # convert bool to 0 or 1
        statistic = (np.abs(n1 - n2) - corr)**2 / (1. * (n1 + n2))
        df = 1
        pvalue = stats.chi2.sf(statistic, df)

    b = _Bunch()
    b.statistic = statistic
    b.pvalue = pvalue
    return b


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

    statistic : float
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
    assert count_row_ss == count_col_ss  # just a calculation check

    # From the SAS manual
    q_stat = ((k-1) * (k * np.sum(count_col_success**2) - count_col_ss**2)
              / (k * count_row_ss - np.sum(count_row_success**2)))

    # Note: the denominator looks just like k times the variance of
    # the columns
    # Wikipedia uses a different, but equivalent expression
    # q_stat = (k-1) * (k *  np.sum(count_row_success**2) - count_row_ss**2)
    #         / (k * count_col_ss - np.sum(count_col_success**2))

    df = k - 1
    pvalue = stats.chi2.sf(q_stat, df)

    if return_object:
        b = _Bunch()
        b.statistic = q_stat
        b.df = df
        b.pvalue = pvalue
        return b

    return q_stat, pvalue, df


def _shift_zeros(dataframe):
    """
    If table contains a cell with zero observations, jitter it up by 0.5

    Chi-squared tests can't handle having expected counts of zero.

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe of data to shift

    Return
    ------
    pd.DataFrame
        shifted data
    """
    if dataframe.min().min() == 0:
        return dataframe + 0.5
    else:
        return dataframe


def _count_level_combinations(data):
    """
    Count every combination of co-occurence of variables in `data`

    Parameters
    ---------
    data : pd.DataFrame
        The data to count up. Should be "wide" oriented, i.e. one column
        per variable

    Return
    ------
    pd.DataFrame
        A dataframe with each combination and its count on a row
    """
    variables = data.columns
    number_of_variables = len(variables)
    data = data.copy()  # don't modify original dataframe
    level_arguments = [[0, 1]] * number_of_variables
    level_combinations = list(itertools.product(*level_arguments))
    full_combinations = pd.DataFrame(level_combinations,
                                     columns=variables)
    full_combinations["_dummy"] = 0
    data['_dummy'] = 1
    to_concat = [data, full_combinations]
    data = pd.concat(to_concat).reset_index(drop=True)
    grouped = data.groupby(list(variables))
    return grouped.sum().reset_index()


class MultipleResponseTable(object):
    """
    Contingency tables for multiple response categorical variables.

    This class implements analyses that can be performed on a two-way
    contingency table that includes 'multiple response' categorical
    variables (e.g. 'choose all that apply' questions on surveys).

    Parameters
    ----------
    row_factors : a Factor instance or list of Factors
        The factor or factors containing data that you intend to have on the
        rows (i.e. the x axis) of the contingency table.
    column_factors : a Factor instance or list of Factors
        The factor or factors containing data that you intend to have on the
        columns (i.e. the y axis) of the contingency table.
    deduplication_padding : str
        Our tables don't deal well with duplicated index / column labels so
        we automatically add a padding character / string to duplicated
        names to make them unique. Defaults to the ' (i.e. "prime") character
        but you can pass any character to use instead.
    shift_zeros : bool
        If shift_zeros is set to true, as we build item-response sub-tables,
        we'll check if any cells in each item-response table is zero, and if so
        add 0.5 to each cell. This can prevent numerical problems
        with the chi-squared tests.

    Attributes
    ----------
    table: pd.DataFrame
        A rectangular table containing the tabulated totals for each
        combination of column factor level and row factor level.

    See also
    --------
    statsmodels.stats.contingency_tables.Table
    scipy.stats.chi2_contingency

    Notes
    -----
    At the moment, this class only supports one Factor on the rows and
    one Factor in the columns (i.e. making a 2x2x2 table is not allowed).

    This class is a close re-implementation of certain functions
    from the MRCV R Library [1]_. The code is used by permission of the authors.

    The MRCV library is itself an implementation of ideas presented
    in [2]_, [3]_, [4]_, and [5]_.

    At the moment, this class mostly restricts itself to ideas from [2]_.

    For more details about use, please see a supplemental Jupyter notebook
    [available here](https://github.com/rogueleaderr/statsmodels_supplementary_docs/blob/master/MRCV%20Table%20Documentation.ipynb)

    References
    ----------
    .. [1] Natalie A. Koziol and Christopher R. Bilder.
           MRCV: A package for analyzing categorical variables with multiple
           response options.
           The R Journal, 6(1):144–150, June 2014.
           CODEN ???? ISSN 2073-4859.
           URL https://cran.r-project.org/web/packages/MRCV/MRCV.pdf
    .. [2] C. Bilder and T. Loughin.
           Testing for marginal independence between two categorical
           variables with multiple responses.
           Biometrics, 60(1):241–248, 2004. [p144, 146]
           The R Journal Vol. 6/1, June ISSN 2073-4859
           CONTRIBUTED RESEARCH ARTICLE 150
    .. [3] C. Bilder and T. Loughin.
           Modeling association between two or more categorical variables
           that allow for multiple category choices.
           Communications in Statistics–Theory and Methods, 36(2):433–451,
           2007. [p144, 146, 149]
    .. [4] C. Bilder and T. Loughin.
           Modeling multiple-response categorical data from complex surveys.
           The Canadian Journal of Statistics, 37(4):553–570, 2009. [p149]
    .. [5] C. Bilder, T. Loughin, and D. Nettleton.
           Multiple marginal independence testing for pick any/c variables.
           Communications in Statistics–Simulation and Computation,
           29(4):1285–1316, 2000. [p149]
    """

    def __init__(self,
                 row_factors, column_factors,
                 deduplication_padding=DEFAULT_DEDUPLICATION_PADDING,
                 shift_zeros=False):
        validate = self._extract_and_validate_factors
        padding = deduplication_padding
        validated_factors = validate(column_factors, row_factors,
                                     deduplication_padding=padding)
        column_factor, row_factor = validated_factors
        self.row_factors = [row_factor,]
        self.column_factors = [column_factor,]
        self.table = self.table_from_factors(row_factors, column_factors)
        self.shift_zeros = shift_zeros

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        template = ("Contingency Table With Multi-Response Categorical "
                    "Variables (MRCV's).\nData:\n{table}")
        return template.format(table=asunicode(self.table, "utf8"))

    def __repr__(self):
        return "At {id} :: {_str}".format(id=id(self), _str=self.__str__())

    @classmethod
    def from_data(cls, data, num_cols_1st_var, num_cols_2nd_var,
                  rows_factor_name="factor_0",
                  columns_factor_name="factor_1"):
        """
        Construct a MultipleResponseTable object directly from data

        (As opposed to constructing from Factor instances).

        Parameters
        ----------
        data : a 2-dimensional numpy array or DataFrame
            The raw data from which a contingency table is
            to be constructed. Include only data that you intend to include
            in the contingency table.
        num_cols_1st_var : int
            The number of columns in the dataframe corresponding
            to the row factor. This # of columns in `data` will be
            used as the row factor
        num_cols_2nd_var : int
            The number of columns in the dataframe corresponding
            to the column factor. Columns #`num_cols_1st_var`
            through `num_cols_2nd_var` from `data` will be used as
            the column factor
        rows_factor_name : str, optional
            If you would like to provide a name for the row variable
        columns_factor_name : str, optional
            If you would like to provide a name for the column variable

        Returns
        -------
        table : MultipleResponseTable
        """
        I = num_cols_1st_var
        J = num_cols_2nd_var
        if isinstance(data, pd.DataFrame):
            rows_data = data.iloc[:, 0:I]
            columns_data = data.iloc[:, I:I + J]
            rows_factor = Factor(rows_data,
                                 name=rows_factor_name, orientation="wide")
            columns_factor = Factor(columns_data,
                                    name=columns_factor_name,
                                    orientation="wide")
        else:
            rows_data = data[:, 0:I]
            columns_data = data[:, I:I + J]
            rows_labels = ["level_{}".format(i) for i in range(0, I)]
            columns_labels = ["level_{}".format(i) for i in range(I, I + J)]
            rows_factor = Factor.from_array(rows_data, labels=rows_labels,
                                            name=rows_factor_name,
                                            orientation="wide")
            columns_factor = Factor.from_array(columns_data,
                                               labels=columns_labels,
                                               name=columns_factor_name,
                                               orientation="wide")

        return cls([rows_factor], [columns_factor])

    @classmethod
    def table_from_factors(cls, row_factors, column_factors):
        """
        Tabulate all combinations of levels of the row and column factors.

        Parameters
        ----------
        row_factors : list of Factor instances
            Factors to use on the rows
        column_factors : list of Factor
            Factors to use in the columns

        Returns
        -------
        table : pd.DataFrame

        Notes
        -----
        Many of the independence test methods for multiple response
        contingency tables require retaining the original data
        so it is not sufficient to simply accept a contingency
        table upfront.
        """
        row_reshaped = row_factors[0].reshape_for_contingency_table()
        col_reshaped = column_factors[0].reshape_for_contingency_table()
        joint_dataframe = _build_joint_dataframe(row_reshaped,
                                                 col_reshaped,
                                                 "_row", "_col")
        table = pd.pivot_table(joint_dataframe,
                               values='_joint_response',
                               fill_value=0,
                               index=['factor_level_row'],
                               columns=['factor_level_col'],
                               aggfunc=np.sum,)
        return table

    def test_for_independence(self, method="bonferroni"):
        """
        Test for independence between row and column factors.

        Test whether the variables on the rows and columns of this
        contingency table are statistically independent. Tries to
        automatically deterime the right test to use given the data you
        provide.

        Parameters
        ----------
        method : {'bonferroni', 'rao-scott-2'}
            The method to use in testing for independence.

        Returns
        -------
        MRCVTableNominalIndependenceResult
            Summary class containing results of the independence test,
            including "p_value_overall" and
            (where available) "p_values_cellwise"

        Notes
        -----
        Different testing procedures are required depending on how
        many multiple response variables the table includes:

        1) Two multiple response variables: we test for "single
           pairwise mutual independence"
        2) One multiple response variable and one single response variable:
           we test for "mutual marginal independence"
        3) Zero multiple response variables
           (i.e. only single response variables):
           we use traditional chi-square tests (from the `Table` class)

        Re: the `method` parameter:

        The "Bonferroni" method is simpler and more transparent and can
            provide p-values on a cell by cell basis. But it can be overly
            conservative, i.e. require an inefficient amount of evidence
            to reject the null hypothesis of independence.

            The "Rao-Scott Second Order Correction" method is
            less conservative but cannot provide cell by cell
            p-values and may be more difficult
            to understand or explain conceptually.
        """
        rows_are_multiple_response = any([f.multiple_response
                                          for f in self.row_factors])
        cols_are_multiple_response = any([f.multiple_response
                                             for f in self.column_factors])
        method_lower = method.lower()
        if method_lower in ("bonferroni", "bon", "b"):
            test = self._test_independence_bonferroni
            return test(cols_are_multiple_response,
                        rows_are_multiple_response)
        elif method_lower in ("rao-scott-2", "rao", "rs2", "rs"):
            test = self._test_independence_rao_scott
            return test(cols_are_multiple_response,
                        rows_are_multiple_response)
        else:
            msg = ("The {method} method is not currently supported. "
                   "Please choose \"bonferroni\" or \"rao-scott-2\""
                   .format(method=method))
            raise NotImplementedError(msg)

    def _build_MRCV_result(self, p_value_overall, p_values_cellwise,
                           method, independence_type):
        """
        Build and initialize a MRCVTableNominalIndependenceResult instance.

        Parameters
        ----------
        p_value_overall : float
        p_values_cellwise : pd.DataFrame of floats
        method : str
        independence_type : {'MMI', 'SPMI'}

        Return
        ------
        MRCVTableNominalIndependenceResult
        """
        result = MRCVTableNominalIndependenceResult()
        result.p_value_overall = p_value_overall
        result.p_values_cellwise = p_values_cellwise
        result.method = method
        if independence_type == "MMI":
            result.independence_type = "Marginal Mutual Independence"
        elif independence_type == "SPMI":
            result.independence_type = "Simultaneous Pairwise Mutual Independence"
        else:
            raise NotImplementedError("Independence Type Required")
        return result

    def _test_independence_bonferroni(self,
                                      columns_are_multiple_response,
                                      rows_are_multiple_response):
        """
        Select and execute the correct Bonferroni-based independence test.

        We need to use different tests depending on how many single vs.
        multiple response variables we're analyzing.

        Parameters
        ----------
        columns_are_multiple_response : bool
            Are any factors on the columns multiple response?
        rows_are_multiple_response : bool
            Are any factors on the rows multiple response?

        Return
        ------
        MRCVTableNominalIndependenceResult
        """
        mmi_test = self._test_MMI_using_bonferroni
        row_factor = self.row_factors[0]
        column_factor = self.column_factors[0]
        if rows_are_multiple_response and columns_are_multiple_response:
            spmi_test = self._test_SPMI_using_bonferroni
            p_value_overall, p_values_cellwise = spmi_test(row_factor,
                                                         column_factor)
            result = self._build_MRCV_result(p_value_overall,
                                             p_values_cellwise,
                                             "Bonferroni", "SPMI")
            return result
        elif columns_are_multiple_response:
            p_value_overall, p_values_cellwise = mmi_test(row_factor,
                                                        column_factor)
            result = self._build_MRCV_result(p_value_overall,
                                             p_values_cellwise,
                                             "Bonferroni", "MMI")
            return result
        elif rows_are_multiple_response:
            p_value_overall, p_values_cellwise = mmi_test(column_factor,
                                                        row_factor)
            result = self._build_MRCV_result(p_value_overall,
                                             p_values_cellwise,
                                             "Bonferroni", "MMI")
            return result
        else:
            single_response_table = Table(self.table)
            return single_response_table.test_nominal_association()

    def _test_independence_rao_scott(self,
                                     columns_are_multiple_response,
                                     rows_are_multiple_response):
        """
        Select and execute the correct Rao-Scott-2-based independence test

        We need to use different tests depending on how many single vs.
        multiple response variables we're analyzing.

        Parameters
        ----------
        columns_are_multiple_response : bool
            Are any factors on the columns multiple response?
        rows_are_multiple_response : bool
            Are any factors on the rows multiple response?

        Return
        ------
        MRCVTableNominalIndependenceResult
        """
        mmi_test = self._test_MMI_using_rao_scott_2
        NOT_AVAILABLE = "Not Available From Rao-Scott Method"
        row_factor = self.row_factors[0]
        column_factor = self.column_factors[0]
        if rows_are_multiple_response and columns_are_multiple_response:
            spmi_test = self._test_SPMI_using_rao_scott_2
            p_value_overall = spmi_test(row_factor, column_factor)
            result = self._build_MRCV_result(p_value_overall,
                                             NOT_AVAILABLE,
                                             "Rao-Scott", "SPMI")
            return result
        elif columns_are_multiple_response:
            p_value_overall = mmi_test(row_factor, column_factor)
            result = self._build_MRCV_result(p_value_overall,
                                             NOT_AVAILABLE,
                                             "Rao-Scott", "MMI")
            return result
        elif rows_are_multiple_response:
            p_value_overall = mmi_test(column_factor, row_factor)
            result = self._build_MRCV_result(p_value_overall,
                                             NOT_AVAILABLE,
                                             "Rao-Scott", "MMI")
            return result
        else:
            single_response_table = Table(self.table)
            return single_response_table.test_nominal_association()

    @classmethod
    def _extract_and_validate_factors(cls,
                                      column_factors, row_factors,
                                      deduplication_padding):
        """
        Validate provided Factors and pluck instances out of lists.

        Make sure that the factors that the user passed into
        the initial MultipleResponseTable initializer are valid and
        that the combination of single and multiple response variables is
        currently supported. Also, since we don't currently
        support multiple factors on a single axis we enforce that only one
        has been passed.

        Also de-duplicate factor level names and pivot factors to wide
        orientation (if necessary)

        Parameters
        ----------
        row_factors : list of Factor instances
            Factors to use on the rows
        column_factors : list of Factor instances
            Factors to use in the columns
        Return
        ------
        (Factor, Factor)
            A single factor instance to use on the columns and a
            single factor instance to use on the rows.
        """
        try:
            extracted = []
            for axis, factors in (("row", row_factors),
                                  ("column", column_factors)):
                if isinstance(factors, list):
                    if len(factors) > 1:
                        msg = ("we don't currently support "
                               "tables with more than one"
                               " factor on the {}".format(axis))
                        raise NotImplementedError(msg)
                    factor = factors[0]
                elif isinstance(factors, Factor):
                    factor = factors
                else:
                    msg = ("{} factors must be either a list of Factors "
                           "or a Factor instance".format(axis))
                    raise NotImplementedError(msg)
                extracted.append(factor)

            # deduplication needs to happen before pivoting
            # for a corner case where a narrow factor with duplicate
            # names won't get pivoted correctly to wide
            cls._deduplicate_level_names(extracted, deduplication_padding)
            for i, factor in enumerate(extracted):
                if (factor.orientation == "narrow" and
                        factor.multiple_response):
                    extracted[i] = factor.cast_narrow_to_wide()
            row_factor, column_factor = extracted
            return column_factor, row_factor
        except IndexError:
            explanation = ("Please be sure to pass at "
                           "least 1 factor on both the rows and columns")
            raise IndexError(explanation)


    @classmethod
    def _deduplicate_level_names(cls, factors, deduplication_padding):
        """
        Make sure that all of the factor level names are unique.

        pandas does not deal well with duplicates in indexes. So if
        duplicates are found, append characters as needed to
        make them unique.

        Parameters
        ----------
        factors : list of Factor instance
            row and column factors to deduplicate together
        """
        taken_names = set()
        for factor in factors:
            if factor.orientation == "wide":
                deduplicated_levels = []
                for level in factor.labels:
                    while level in taken_names:
                        level += deduplication_padding
                    taken_names.add(level)
                    deduplicated_levels.append(level)
                factor_levels = factor.data.columns.values
                if np.any(deduplicated_levels != factor_levels):
                    old_name = factor.data.columns.name
                    factor.data.columns = deduplicated_levels
                    factor.data.columns.name = old_name
            else:
                renames = {}
                for name in taken_names:
                    old_name = name
                    while name in taken_names:
                        name += deduplication_padding
                    renames[old_name] = name
                if renames:
                    factor.data.replace({'factor_level': renames},
                                        inplace=True)
                # this may be too magical. but with narrow data, we don't
                # have any good way to know that two identical names
                # correspond to two different variables, except the
                # assumption that the data wouldn't record two observations
                # of the same level for the same observation id
                # so we assume any duplicates are actually different
                # variables with the same name and tag on a ' marker
                subset = ["observation_id", "factor_level"]
                duplicated = factor.data.duplicated(subset=subset)
                while duplicated.any():
                    duplicates = factor.data[duplicated].copy()
                    replacements = duplicates.factor_level.astype(str) + deduplication_padding
                    duplicates.loc[:, 'factor_level'] = replacements
                    factor.data.update(duplicates)
                    duplicated = factor.data.duplicated(subset=subset)
                taken_names |= set(factor.data.factor_level.unique())

    @staticmethod
    def _item_response_table_for_MMI(single_response_factor,
                                     multiple_response_factor,
                                     shift_zeros=False):
        """
        Build item-response table between single and multiple response vars

        Parameters
        ----------
        single_response_factor : Factor instance
            Factor to use on the rows. Must be a single response variable,
            i.e. "choose 1-of-N"
        multiple_response_factor : Factor instance
            Factor to use in the columns. Must be a multiple
            response variable, i.e. "choose all that apply".
        shift_zeros : bool
            If shift_zeros is set to true, we'll check
            if any cells in each item-response table is zero, and if so
            add 0.5 to each cell. This can prevent numerical problems
            with the chi-squared tests.

        Return
        ------
        pd.DataFrame

        Notes
        -----
        The item response table builds "sub-tabulations" that compare
        the single response variable vs each level of the multiple
        response variable, e.g:

             |   A   |   B   |  C    |
             | 1 | 0 | 1 | 0 | 1 | 0 |
             |------------------------
         yes | 5   3 | 3   2 | 1   8
             |
         no  | 0   3 | 0   5 | 3   1

        """
        srcv_data = single_response_factor.data
        if single_response_factor.orientation == "narrow":
            renamings = {"value": "selected",
                         "factor_level": "single_response_level"}
            single_response_melted = srcv_data.rename(columns=renamings)
        else:
            srcv_data = srcv_data
            id_var = srcv_data.index.name
            renamings = {id_var: "observation_id", "value": "selected",
                         "factor_level": "single_response_level"}
            single_response_melted = (pd.melt(srcv_data.reset_index(),
                                              id_vars=id_var)
                                      .rename(columns=renamings))
            selected = single_response_melted.selected == 1
            single_response_melted = single_response_melted[selected]

        # don't modify original
        multiple_response_dataframe = multiple_response_factor.data.copy()
        multiple_response_dataframe.index.name = "observation_id"
        multiple_response_dataframe.reset_index(inplace=True)

        joint_dataframe = pd.merge(single_response_melted,
                                   multiple_response_dataframe,
                                   how="inner", on="observation_id")

        single_response_column = joint_dataframe.single_response_level
        item_response_pieces = {}
        for c in multiple_response_factor.labels:
            column_position = joint_dataframe.columns.get_loc(c)
            # if factors have been combined, c can be a tuple.
            # pandas assumes tuples passed into .loc
            # are looking for a multi-index pair.
            multiple_response_column = joint_dataframe.iloc[:, column_position]
            crosstab = pd.crosstab(single_response_column,
                                   multiple_response_column)
            if shift_zeros:
                # reindex in case one option is never selected
                srcv_labels = single_response_factor.labels
                crosstab = (crosstab.reindex(columns=[0, 1],
                                             index=srcv_labels).fillna(0))
                crosstab = _shift_zeros(crosstab)
            item_response_pieces[c] = crosstab

        names = ["multiple_response_level", "selected?"]
        item_response_table = pd.concat(item_response_pieces, axis=1,
                                        names=names)
        return item_response_table

    @classmethod
    def _chi2s_for_MMI_item_response_table(cls, srcv, mrcv, shift_zeros=False):
        """
        Calc chi-squared stat for pairings in the MMI item-response table.

        Parameters
        ----------
        srcv : Factor instance
            A single response categorical Factor
        mrcv : Factor instance
            A multiple response categorical Factor
        shift_zeros : bool
            If shift_zeros is set to true, we'll check
            if any cells in each item-response table is zero, and if so
            add 0.5 to each cell. This can prevent numerical problems
            with the chi-squared tests.

        Return
        ------
        pd.DataFrame

        Notes
        -----
        The item response table builds "sub-tabulations" that compare the
        single response variable vs each level of the multiple response
        variable, e.g:

             |   A   |   B   |  C    |
             | 1 | 0 | 1 | 0 | 1 | 0 |
             |------------------------
         yes | 5   3 | 3   2 | 1   8
             |
         no  | 0   3 | 0   5 | 3   1

        We can consider each 2 x 2 grid to be its own pairing of factors,
        e.g. "is whether a respondent chose 'A' independent of whether or
        not she said 'yes'?".

        We can then calculate a chi-squared statistic showing how strong
        the deviance is from expectation in that particular sub-table.
        Then by appropriately combining those individual chi-squared
        statistics we can investigate mutual marginal independence
        for the overall table.
        """
        item_response_table = cls._item_response_table_for_MMI(srcv, mrcv,
                                                        shift_zeros=shift_zeros)
        mmi_chi_squared_by_cell = pd.Series(index=mrcv.labels)
        rows_levels = item_response_table.index
        columns_levels = item_response_table.columns.levels[0]
        valid_shape = cls._item_response_tbl_shape_valid(
            item_response_table, columns_levels, srcv.labels, "MMI"
        )
        if not valid_shape:
            import warnings
            warnings.warn("The MMI item-response table had degenerate shape, "
                          "probably because certain factor levels lacked "
                          "variance, i.e. were 1 for all observations or 0 for "
                          "all observations. Statsmodels declines "
                          "to calculate a test for independence in this case "
                          "because doing so would require "
                          "substantial assumptions about the unobserved cases.")
            chis = pd.DataFrame(np.nan, index=rows_levels,
                                columns=columns_levels)
            return chis
        for factor_level in columns_levels:
            crosstab = item_response_table.loc[:, factor_level]
            chi2_results = chi2_contingency(crosstab, correction=False)
            chi_squared_stat, _, _, _ = chi2_results
            mmi_chi_squared_by_cell.loc[factor_level] = chi_squared_stat
        return mmi_chi_squared_by_cell

    @staticmethod
    def _item_response_table_for_SPMI(rows_factor, columns_factor,
                                      shift_zeros=False):
        """
        Build full item-response table between two multiple response vars

        Parameters
        ----------
        rows_factor : Factor instance
            Multiple response factor instance to use on the rows
        columns_factor : Factor instance
            Multiple fesponse factor instance to use in the columns
        shift_zeros : bool
            If shift_zeros is set to true, we'll check
            if any cells in each item-response table is zero, and if so
            add 0.5 to each cell. This can prevent numerical problems
            with the chi-squared tests.

        Return
        ------
        pd.DataFrame

        Notes
        -----
        The item response table builds "sub-tabulations" that compare
        whether each individual level of the first multiple response
        variable was selected to whether each individual level of
        the second multiple response variable was selected e.g:

                 |   A   |   B   |  C    |
                 | 1 | 0 | 1 | 0 | 1 | 0 |
                 -------------------------
         Eggs  1 | 5   3 | 3   2 | 1   8
               0 | 12  9 | 9   0 | 2   4
              --------------------------
         Pizza 1 | 0   3 | 10  5 | 13  1
               0 | 11  3 | 6   5 | 3   19

        """
        rows_data = rows_factor.data
        columns_data = columns_factor.data
        rows_levels = rows_factor.labels
        columns_levels = columns_factor.labels
        row_crosstabs = OrderedDict()
        for i, row_name in enumerate(rows_levels):
            column_crosstabs = OrderedDict()
            for j, col_name in enumerate(columns_levels):
                rows = rows_data.iloc[:, i]
                columns = columns_data.iloc[:, j]
                crosstab = pd.crosstab(index=rows, columns=columns,
                                       rownames=[row_name],
                                       colnames=[col_name])
                if shift_zeros:
                    crosstab = crosstab.reindex(columns=[0, 1],
                                                index=[0, 1]).fillna(0)
                    crosstab = _shift_zeros(crosstab)
                column_crosstabs[col_name] = crosstab
            row_crosstab = pd.concat(column_crosstabs, axis=1,
                                     names=["column_levels", "selected?"])
            ordered_column_keys = column_crosstabs.keys()
            # preserve column ordering
            row_crosstab = row_crosstab.reindex(columns=ordered_column_keys,
                                                level=0)
            row_crosstabs[row_name] = row_crosstab
        item_response_table = pd.concat(row_crosstabs, axis=0,
                                        names=["row_levels", "selected?"])
        # undo any name mangling
        item_response_table.columns.set_levels(columns_levels,
                                               level=0, inplace=True)
        keys = row_crosstabs.keys()
        item_response_table = item_response_table.reindex(index=keys,
                                                          level=0)
        return item_response_table

    @classmethod
    def _chi2s_for_SPMI_item_response_table(cls,
                                            rows_factor,
                                            columns_factor,
                                            shift_zeros=False):
        """
        Calc chi-squared stat for each pairing in SPMI item response table.

        Parameters
        ----------
        rows_factor : Factor instance
            A multiple response factor to use on the rows
        columns_factor : Factor instance
            A multiple response factor to use in the columns
        shift_zeros : bool
            If shift_zeros is set to true, we'll check
            if any cells in each item-response table is zero, and if so
            add 0.5 to each cell. This can prevent numerical problems
            with the chi-squared tests.

        Return
        ------
        pd.DataFrame

        Notes
        -----
        The item response table builds "sub-tabulations" that compare
        whether each individual level of the first multiple response
        variable was selected to whether each individual level of
        the second multiple response variable was selected e.g:

                 |   A   |   B   |  C    |
                 | 1 | 0 | 1 | 0 | 1 | 0 |
                 -------------------------
         Eggs  1 | 5   3 | 3   2 | 1   8
               0 | 12  9 | 9   0 | 2   4
              --------------------------
         Pizza 1 | 0   3 | 10  5 | 13  1
               0 | 11  3 | 6   5 | 3   19

        We can consider each 2 x 2 grid to be its own pairing of factors,
        e.g. "is whether a respondent chose 'A' independent of
        whether or not she choose 'Eggs'".

        We can then calculate a chi-squared statistic showing
        how strong the deviance is from expectation in that particular
        sub-table. Then by appropriately combining those individual
        chi-squared statistics we can investigate simultaneous pairwise
        marginal independence for the overall table
        """
        build_table = cls._item_response_table_for_SPMI
        item_response_table = build_table(rows_factor, columns_factor,
                                          shift_zeros=shift_zeros)
        rows_levels = item_response_table.index.levels[0]
        columns_levels = item_response_table.columns.levels[0]
        valid_shape = cls._item_response_tbl_shape_valid(
            item_response_table, columns_levels, rows_levels, "SPMI"
        )
        if not valid_shape:
            import warnings
            warnings.warn("The SPMI item-response table had degenerate shape, "
                          "probably because certain factor levels lacked "
                          "variance, i.e. were 1 for all observations or 0 for "
                          "all observations. Statsmodels declines "
                          "to calculate a test for independence in this case "
                          "because doing so would require "
                          "substantial assumptions about the unobserved cases.")
            chis = pd.DataFrame(np.nan, index=rows_levels,
                                columns=columns_levels)
            return chis
        chis_spmi = pd.DataFrame(index=rows_levels, columns=columns_levels)
        for row_level in rows_levels:
            for column_level in columns_levels:
                location = row_level, column_level
                crosstab = item_response_table.loc[location]
                crosstab = _shift_zeros(crosstab)
                chi2_results = chi2_contingency(crosstab, correction=False)
                chi_squared_statistic, _, _, _ = chi2_results
                chis_spmi.loc[location] = chi_squared_statistic
        return chis_spmi

    @classmethod
    def _item_response_tbl_shape_valid(cls,
                                       item_response_table,
                                       columns_levels,
                                       rows_levels,
                                       type_):
        """
        Ensure item response table shape makes sense to analyze.

        Crosstab will have degenerate shape (i.e. dimension != r*c)
        if one level had no observations, i.e. was all 0 or all 1.
        We could pad those out with 0.5 on the unobserved levels.
        But instead of making the assumption that that's desired,
        we'll just decline to calculate.

        Parameters
        ----------
        item_response_table : pd.DataFrame
            Table to validate
        columns_levels : pd.Index
            levels of column factor
        rows_levels: pd.Index
            levels of row factor

        Return
        ------
        bool
            Whether shape is valid
        """

        num_col_levels = len(columns_levels)
        num_row_levels = len(rows_levels)
        if type_ == 'SPMI':
            valid = item_response_table.shape == (num_row_levels * 2,
                                                  num_col_levels * 2)
            return valid
        elif type_ == "MMI":
            valid = item_response_table.shape == (num_row_levels,
                                                  num_col_levels * 2)
            return valid
        else:
            raise NotImplementedError('Invalid MRCV table type.')

    def _test_SPMI_using_bonferroni(self, row_factor, column_factor):
        """
        Test for SPMI between two multiple response vars using Bonferroni

        SPMI stands for "simultaneous pairwise mutual independence".

        To test, first calculate a full item response table comparing each
        pairing of levels from both variables, then calculate a
        chi-square statistic for each pairing, then adjust that table
        of pairwise statistics using Bonferroni correction to account
        for multiple comparisons within the single overall test.

        Parameters
        ----------
        row_factor : Factor instance
            A multiple response factor to use on the rows
        column_factor : Factor instance
            A multiple response factor to use in the columns

        Return
        ------
        float, pd.DataFrame
            Tuple containing a p value for independence for the
            overall table, plus a dataframe including cellwise p values
            assessing independence between each pairing of factor levels.
        """
        observed = self._chi2s_for_SPMI_item_response_table(row_factor,
                                                            column_factor,
                                                shift_zeros=self.shift_zeros)
        chi2_survival_with_1_dof = partial(chi2.sf, df=1)
        p_value_ij = observed.applymap(chi2_survival_with_1_dof)
        p_value_min = p_value_ij.min().min()
        bonferroni_correction_factor = (row_factor.factor_level_count *
                                        column_factor.factor_level_count)
        cap = lambda x: min(x, 1)
        capped_p_value = cap(bonferroni_correction_factor * p_value_min)
        p_value_overall_bonferroni = capped_p_value
        pairwise_bonferroni_p_values = ((p_value_ij *
                                        bonferroni_correction_factor)
                                        .applymap(cap))
        return p_value_overall_bonferroni, pairwise_bonferroni_p_values

    def _test_for_SPMI_using_bootstrap(self, row_factor, column_factor,
                                       verbose=False, b_max=1000):
        """
        Test for SPMI between two multiple response vars using Bootstrapping

        SPMI stands for "simultaneous pairwise mutual independence".

        First calculate a full item response table comparing each pairing
        of levels from both variables, then calculate a chi-square statistic
        for each pairing, then use a bootstrap process to estimate
        the underlying distribution of chi-squared's for a table with this
        structure and use that estimated distribution to assess the overall
        probability of table independence and cellwise independence.

        Parameters
        ----------
        row_factor : Factor instance
            A multiple response factor to use on the rows
        column_factor : Factor instance
            A multiple response factor to use in the columns

        Return
        ------
        float, pd.DataFrame
            Tuple containing a p value for independence for the
            overall table, plus a dataframe including cellwise p values
            assessing independence between each pairing of factor levels.
        """

        W = row_factor.data
        Y = column_factor.data
        I = row_factor.factor_level_count
        J = column_factor.factor_level_count
        spmi_df = pd.concat([W, Y], axis=1)  # type: pd.DataFrame
        chi2_survival_with_1_dof = partial(chi2.sf, df=1)

        n = len(spmi_df)
        q1 = spmi_df.iloc[:, :I]
        q2 = spmi_df.iloc[:, I:I + J]
        X_sq_S_star = []
        X_sq_S_ij_star = pd.DataFrame(index=range(0, I * J),
                                      columns=range(0, b_max))
        p_value_b_min = []
        p_value_b_prod = []
        rows_factor_name = row_factor.name
        columns_factor_name = column_factor.name
        calc_chis = self._chi2s_for_SPMI_item_response_table
        for i in range(0, b_max):
            if verbose and i % 50 == 0:
                print("sample {}".format(i))
            # pd.concat requires unique indexes
            # sampling with replacement produces duplicates
            q1_sample = q1.sample(n, replace=True).reset_index(drop=True)
            q2_sample = q2.sample(n, replace=True).reset_index(drop=True)

            sample_rows_factor = Factor(q1_sample, rows_factor_name,
                                        orientation="wide",
                                        multiple_response=True)
            sample_columns_factor = Factor(q2_sample, columns_factor_name,
                                           orientation="wide",
                                           multiple_response=True)
            stat_star = calc_chis(sample_rows_factor, sample_columns_factor,
                                  shift_zeros=self.shift_zeros)
            X_sq_S = stat_star.sum().sum()
            X_sq_S_star.append(X_sq_S)
            X_sq_S_ij_star.append(stat_star)
            p_value_ij = stat_star.applymap(chi2_survival_with_1_dof)
            p_value_min = p_value_ij.min().min()
            p_value_prod = p_value_ij.prod().prod()
            p_value_b_min.append(p_value_min)
            p_value_b_prod.append(p_value_prod)

        observed_rows_factor = Factor(q1, rows_factor_name,
                                    orientation="wide",
                                    multiple_response=True)
        observed_columns_factor = Factor(q2, columns_factor_name,
                                       orientation="wide",
                                       multiple_response=True)
        observed = calc_chis(observed_rows_factor,
                                observed_columns_factor,
                             shift_zeros=self.shift_zeros)
        observed_X_sq_S = observed.sum().sum()
        p_value_ij = observed.applymap(chi2_survival_with_1_dof)
        p_value_min = p_value_ij.min().min()

        p_value_boot = np.mean(X_sq_S_star >= observed_X_sq_S)
        print(p_value_boot)

        p_value_boot_min_overall = np.mean(p_value_b_min <= p_value_min)
        print(p_value_boot_min_overall)
        start = pd.DataFrame(0,
                             index=p_value_ij.index,
                             columns=p_value_ij.columns)
        for frame in X_sq_S_ij_star:
            start += frame
        average_p_ij = start / len(X_sq_S_ij_star)
        print(average_p_ij)

    def _test_SPMI_using_rao_scott_2(self, row_factor,
                                     column_factor):
        """
        Test for SPMI between two multiple response vars using Rao Scott

        SPMI stands for "simultaneous pairwise mutual independence".

        See [1]_ for more about the second-order Rao Scott Correction.

        First calculate a full item response table comparing each pairing
        of levels from both variables, then calculate a chi-square statistic
        for each pairing, then sum all the statistics in the table
        and make a second order Rao Scott correction to allow using a
        standard chi-squared distribution to estimate the overall
        probability of SPMI between the 2 factors from the adjusted
        sum of pairwise chi-squareds.

        Parameters
        ----------
        row_factor : Factor instance
            A multiple response factor to use on the rows
        column_factor : Factor instance
            A multiple response factor to use in the columns

        Return
        ------
        float
            A p-value for independence for the overall table.

        References
        ----------
        ..    [1] Rao, J. N. K. - Scott, A. J.
                  The analysis of categorical data from complex surveys:
                  Chi-squared tests for goodness of fit and independence
                  in two-way tables.
                  Journal of the American Statistical Association
                  76, 221-230, 1981.
        """
        observed = self._chi2s_for_SPMI_item_response_table(row_factor,
                                                            column_factor,
                                                shift_zeros=self.shift_zeros)
        W = row_factor.data
        Y = column_factor.data
        I = row_factor.factor_level_count
        J = column_factor.factor_level_count
        spmi_df = pd.concat([W, Y], axis=1)  # type: pd.DataFrame

        W_count_ordered = _count_level_combinations(W)
        Y_count_ordered = _count_level_combinations(Y)
        n_count_ordered = _count_level_combinations(spmi_df)

        n = len(spmi_df)
        G = (W_count_ordered.iloc[:, :-1]).T
        H = (Y_count_ordered.iloc[:, :-1]).T
        combined_counts = n_count_ordered.iloc[:, -1]
        tau = combined_counts / n
        m_row = G.dot(W_count_ordered.iloc[:, -1])
        m_col = H.dot(Y_count_ordered.iloc[:, -1])
        GH = np.kron(G, H)
        m = GH.dot(combined_counts)

        pi_row = m_row / n
        pi_col = m_col / n
        j_2r = np.ones((2 ** I, 1))
        i_2r = np.eye(2 ** I)
        j_2c = np.ones((2 ** J, 1))
        i_2c = np.eye(2 ** J)

        G_ij = G.dot(np.kron(i_2r, j_2c.T))
        H_ji = H.dot(np.kron(j_2r.T, i_2c))

        # extra .T's b/c Python handles
        # vector/matrix kronecker differently than R
        H_kron = np.kron(pi_row, H_ji.T).T
        G_kron = np.kron(G_ij.T, pi_col).T
        F = GH - H_kron - G_kron

        mult_cov = np.diag(tau) - np.outer(tau, tau.T)
        sigma = F.dot(mult_cov.dot(F.T))

        D = (np.diag(np.kron(pi_row, pi_col) *
                     np.kron(1 - pi_row, 1 - pi_col)))
        Di_sigma = np.diag(1 / np.diag(D)).dot(sigma)
        eigenvalues, eigenvectors = linalg.eig(Di_sigma)
        Di_sigma_eigen = np.real(eigenvalues)
        sum_Di_sigma_eigen_sq = (Di_sigma_eigen ** 2).sum()

        observed_X_sq_S = observed.sum().sum()
        X_sq_S_rs2 = I * J * observed_X_sq_S / sum_Di_sigma_eigen_sq
        df_rs2 = (I ** 2) * (J ** 2) / sum_Di_sigma_eigen_sq
        X_sq_S_p_value_rs2 = 1 - chi2.cdf(X_sq_S_rs2, df=df_rs2)
        return X_sq_S_p_value_rs2

    def _test_MMI_using_bonferroni(self,
                                   single_response_factor,
                                   multiple_response_factor):
        """
        Test for MMI between single and multiple response vars w/ Bonferroni

        MMI stands for "multiple marginal independence".

        First calculate a full item response table comparing the single
        response variable versus each level of the multiple response
        variable, then calculate a chi-square statistic for each pairing,
        then adjust that table of pairwise statistics using Bonferroni
        correction to account for multiple comparisons within the
        single overall test.

        Parameters
        ----------
        single_response_factor : Factor instance
            Single response factor to use on the rows
        multiple_response_factor : Factor instance
            Multiple response factor to use in the columns

        Return
        ------
        float, pd.DataFrame
            Tuple containing a p value for independence for the
            overall table, plus a dataframe including cellwise p values
            assessing independence between each pairing of factor levels.
        """
        calc_chis = self._chi2s_for_MMI_item_response_table
        mmi_pairwise_chis = calc_chis(single_response_factor,
                                      multiple_response_factor,
                                      shift_zeros=self.shift_zeros)
        c = len(multiple_response_factor.labels)
        r = len(single_response_factor.labels)

        chi2_survival = partial(chi2.sf, df=(r - 1))

        p_value_ij = mmi_pairwise_chis.apply(chi2_survival)
        p_value_min = p_value_ij.min()

        bonferroni_correction_factor = c
        cap = lambda x: min(x, 1)
        try:
            p_value_overall_bonferroni = cap(bonferroni_correction_factor *
                                           p_value_min)
            pairwise_bonferroni_p_values = ((p_value_ij *
                                             bonferroni_correction_factor)
                                            .apply(cap))
        except ValueError:
            # if p_value_ij are all nan, p_value min will come back as a
            # series and the min() function will fail. But the min
            # of a series of nans should be nan
            p_value_overall_bonferroni = np.nan
            pairwise_bonferroni_p_values = pd.Series(np.nan,
                                                     index=p_value_ij.index)

        return p_value_overall_bonferroni, pairwise_bonferroni_p_values

    def _test_MMI_using_rao_scott_2(self,
                                    single_response_factor,
                                    multiple_response_factor):
        """
        Test for MMI between single and multiple response vars w/ Rao Scott

        MMI stands for "multiple marginal independence".

        See [1]_ for information about the second order
        Rao Scott correction.

        First calculate a full item response table comparing the single
        response variable versus each level of the multiple response
        variable, then calculate a chi-square statistic for each pairing,
        then sum those statistics and make a second order Rao Scott
        correction to allow using a standard chi-squared distribution to
        estimate the overall probability of SPMI between the 2 factors from
        the adjusted sum of pairwise chi-squareds.

        Parameters
        ----------
        single_response_factor : Factor instance
            Single response factor to use on the rows
        multiple_response_factor : Factor instance
            Multiple response factor to use in the columns

        Return
        ------
        float
            A p value for independence for the overall table.

        References
        ----------
        ..    [1] Rao, J. N. K. - Scott, A. J.
                  The analysis of categorical data from complex surveys:
                  Chi-squared tests for goodness of fit and independence
                  in two-way tables.
                  Journal of the American Statistical Association
                  76, 221-230, 1981.
         """
        if single_response_factor.orientation == "wide":
            W = single_response_factor.cast_wide_to_narrow().data.copy()
            W = W[W['value'] == 1]  # only consider actually selected option
            W.set_index("observation_id", inplace=True)
        else:
            W = single_response_factor.data.iloc[:, 1:]
        if not isinstance(W, pd.Series):
            W = W.iloc[:, 0]
        Y = multiple_response_factor.data
        n = len(W)
        I = 1  # single response variable must have exactly one column
        J = len(Y.columns)
        c = J
        r = len(W.unique())

        def conjoint_combinations(srcv, mrcv):
            number_of_variables = 1 + len(mrcv.columns)
            srcv = srcv.copy()  # don't modify original dataframe
            mrcv = mrcv.copy()  # don't modify original dataframe
            srcv.name = "srcv"
            srcv_level_arguments = srcv.unique()
            mrcv_level_arguments = [[0, 1]] * (number_of_variables - 1)
            level_arguments = ([list(srcv_level_arguments), ] +
                               mrcv_level_arguments)
            variables = ['srcv', ] + list(mrcv.columns)
            level_combinations = list(itertools.product(*level_arguments))
            full_combinations = pd.DataFrame(level_combinations,
                                             columns=variables)
            full_combinations["_dummy"] = 0
            data = pd.concat([srcv, mrcv], axis=1)  # type: pd.DataFrame
            data['_dummy'] = 1
            data = pd.concat([data, full_combinations]) # type: pd.DataFrame
            data.reset_index(drop=True, inplace=True)
            grouped = data.groupby(list(variables))
            result = grouped.sum().reset_index()
            return result

        Y_count_ordered = _count_level_combinations(Y)
        n_count_ordered = conjoint_combinations(W, Y)
        n_counts_grouped = n_count_ordered.groupby('srcv')
        srcv_table_order = n_counts_grouped.first().index.values
        n_iplus = W.value_counts().reindex(srcv_table_order)
        tau = (n_count_ordered.iloc[:, -1] /
               np.repeat(n_iplus, repeats=(2 ** c)).reset_index(drop=True))
        # the R version subtracts 1 from G_tilde because
        # data.matrix converts 0->1 and 1->2
        # (probably because it thinks they're factors
        # and it's internally coding them)
        G_tilde = Y_count_ordered.iloc[:, :-1].T
        I_r = np.eye(r)
        G = np.kron(I_r, G_tilde)
        pi = G.dot(tau)
        m = pi * np.repeat(n_iplus, c)
        a_i = n_iplus / n
        pi_not_j = (1 / n) * np.kron(np.ones(r), np.eye(c)).dot(m)
        j_r = np.ones(r)
        I_rc = np.eye(r * c)
        I_c = np.eye(c)
        J_rr = np.ones((r, r))
        A = np.diag(a_i)
        H = I_rc - np.kron(J_rr.dot(A), I_c)
        D = (np.kron(np.diag(n / n_iplus), np.diag(pi_not_j) *
                     (1 - pi_not_j)))
        v_dim = r * (2 ** c)
        V = np.zeros((v_dim, v_dim))

        for i in range(1, r + 1):
            a = ((i - 1) * (2 ** c) + 1) - 1
            b = ((i - 1) * (2 ** c) + (2 ** c)) - 1
            tau_range = tau[a:b]
            a_v = (1 / a_i[i - 1])
            tau_diag = np.diag(tau_range)
            tau_tcrossproduct = np.outer(tau_range, tau_range.T)
            v = a_v * (tau_diag - tau_tcrossproduct)
            V[a:b, a:b] = v

        D_diag = np.diag(1 / np.diag(D))
        tcrossprod_VG = V.dot(G.T)
        tcrossprod_VGH = tcrossprod_VG.dot(H.T)
        Di_HG = D_diag.dot(H).dot(G)
        Di_HGVGH = np.matmul(Di_HG, tcrossprod_VGH)
        eigenvalues, eigenvectors = np.linalg.eig(Di_HGVGH)
        Di_HGVGH_eigen = np.real(eigenvalues)
        sum_Di_HGVGH_eigen_sq = (Di_HGVGH_eigen ** 2).sum()
        calculate_chis = self._chi2s_for_MMI_item_response_table
        observed = calculate_chis(single_response_factor,
                                  multiple_response_factor,
                                  shift_zeros=self.shift_zeros)
        observed_X_sq = observed.sum()
        rows_by_columns = ((r - 1) * c)
        X_sq_S_rs2 = rows_by_columns * observed_X_sq / sum_Di_HGVGH_eigen_sq
        df_rs2 = ((r - 1) ** 2) * (c ** 2) / sum_Di_HGVGH_eigen_sq
        X_sq_S_p_value_rs2 = 1 - chi2.cdf(X_sq_S_rs2, df=df_rs2)
        return X_sq_S_p_value_rs2


def _build_joint_dataframe(left_data, right_data, l_suffix, r_suffix):
    """
    Take two dataframes and combine them into a dataframe that
    can be pivoted using pandas .pivot_table() functionality

    Parameters
    ----------
    left_data : pd.DataFrame
        dataframe to concatenate on the left
    right_data : pd.DataFrame
        dataframe to concatenate on the right
    l_suffix : str
        if the data frames have overlapping column names apply
        this suffix to columns from the left dataframe
    r_suffix : str
        if the data frames have overlapping column names apply
         this suffix to columns from the right dataframe

    """
    joint_dataframe = pd.merge(left_data, right_data,
                               how="inner",
                               on='observation_id',
                               suffixes=(l_suffix, r_suffix))
    # without bool cast, '&' sometimes doesn't know how to compare types
    l_value_col = 'value{}'.format(l_suffix)
    r_value_col = 'value{}'.format(r_suffix)
    joint_response = ((joint_dataframe[l_value_col].astype(bool) &
                      joint_dataframe[r_value_col].astype(bool))
                      .astype(int))
    joint_dataframe['_joint_response'] = joint_response
    return joint_dataframe


class Factor(object):
    """
    Container class for a single variable in a contingency table analysis.

    Primarily used as an input to the MultipleResponseTable class.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A dataframe containing the data for the variable
    name : str
        The name of the variable
    orientation : {"wide", "narrow" }
        To specify whether the data is laid out in a
        wide (i.e. with one column per level)
        versus narrow (i.e. with one row per observation/level pairing).
    multiple_response : boolean
        Can the variable contain more than 1 level per observation, e.g.
        "subject #1 selected both eggs and pizza".

    Attributes
    ----------
    data : pd.DataFrame
        The underlying dataframe
    name : str
        The name of the variable
    orientation : { "wide", "narrow" }
        Whether the data is laid out in a
        wide orientation (i.e. with one column per level)
        versus narrow orientation (i.e. with one row per
        observation/level pairing)
    multiple_response : boolean
        Can the variable contain more than 1 level per observation, e.g.
        "subject #1 selected both eggs and pizza".
    labels : [str]
        Strings that designate each level that an observation can take, e.g.
        ["bicycle", "motorcycle", "car"]
    factor_level_counts : int
        The number of distinct levels that an observation can take.

    Notes
    -----
    Factors can have one of two *orientations* (narrow or wide) depending on the shape of the data they contain:

    Here's an example of a wide oriented factor:

    +----------------+------+-------+-------+
    | observation_id | eggs | pizza | candy |
    +================+======+=======+=======+
    |        1       |  1   |   0   |   0   |
    +----------------+------+-------+-------+
    |        2       |  0   |   1   |   1   |
    +----------------+------+-------+-------+
    |        3       |  0   |   0   |   0   |
    +----------------+------+-------+-------+
    |        4       |  1   |   1   |   1   |
    +----------------+------+-------+-------+
    |        5       |  1   |   0   |   0   |
    +----------------+------+-------+-------+
    |        6       |  1   |   0   |   1   |
    +----------------+------+-------+-------+

    Here's an example of a narrow oriented factor:

    +----------------+----------+----------+
    | observation_id | variable | selected |
    +================+==========+==========+
    |         1      |  eggs    |    1     |
    +----------------+----------+----------+
    |         1      |  pizza   |    1     |
    +----------------+----------+----------+
    |         1      |  candy   |    0     |
    +----------------+----------+----------+
    |         2      |  eggs    |    1     |
    +----------------+----------+----------+
    |         3      |  eggs    |    1     |
    +----------------+----------+----------+
    |         4      |  eggs    |    0     |
    +----------------+----------+----------+

    When you create a factor you should tell it the orientation of your data.

    If your data does not already conform to one of these two shapes, please reshape it
    so that it does before using the :class:`Factor` class.
    """

    def __init__(self, dataframe, name,
                 orientation="wide", multiple_response=None):
        self.orientation = orientation
        self.name = name
        if orientation == "narrow":
            if list(dataframe.columns.values) != ['observation_id',
                                                  'factor_level',
                                                  'value']:
                msg = ("If you provide data directly to a narrow-oriented "
                       "factor, the provided labels must be "
                       "['observation_id', 'factor_level', 'value'].")
                raise NotImplementedError(msg)

        if (dataframe.index.name is None and
                    "observation_id" not in dataframe.columns):
            dataframe.index.name = "observation_id"
        elif dataframe.index.name == "observation_id":
            if dataframe.index.nunique() < len(dataframe.index):
                import warnings
                warnings.warn("You have duplicate observations id's in your"
                              "index. That may cause strange behavior.")
        if (dataframe.columns.name is None and
            "factor_level" not in dataframe.columns):
            dataframe.columns.name = "factor_level"
        # don't modify original in subsequent operations
        self.data = dataframe.copy()

        if multiple_response is None:
            if orientation == "wide":
                responses_per_observation = dataframe.sum(axis=1)
                if np.max(responses_per_observation) > 1:
                    self.multiple_response = True
                else:
                    self.multiple_response = False
            else:
                self.multiple_response = False
        else:
            self.multiple_response = multiple_response

    def __unicode__(self):
        template = ("{mr_slug}Factor: {name}\n"
                    "Columns:{columns}\nData:\n{data}")
        if self.multiple_response:
            mr_slug = "Multiple Response "
        else:
            mr_slug = ""
        return template.format(mr_slug=mr_slug,
                               name=self.name, columns=self.labels,
                               data=self.data)

    def __repr__(self):
        template = "Factor at {id} :: {output}"
        output = template.format(id=id(self),
                                output=self.__unicode__())
        return output

    def __str__(self):
        return self.__unicode__()

    @classmethod
    def from_array(cls, data, labels, name,
                   orientation="wide", multiple_response=None):
        """
        Construct a factor from a numpy array

        Parameters
        ----------
        data : pd.DataFrame
            A numpy array containing your data
        labels: list of str
            a list of labels for each column header
        name : str
            The name of the variable
        orientation : { "wide", "narrow" }
            To specify whether the data is laid out in a
            wide (i.e. with one column per level)
            versus narrow (i.e. with one row per observation/level pairing,
        multiple_response : boolean
            Can the variable contain more than 1 level per observation, e.g.
            "subject #1 selected both eggs and pizza".
        """

        if len(labels) != data.shape[1]:
            raise ValueError("all columns must have labels")
        data = np.asarray(data, dtype=np.float64)
        dataframe = pd.DataFrame(data, columns=labels)
        factor = cls(dataframe, orientation=orientation,
                     multiple_response=multiple_response, name=name)
        return factor

    def reshape_for_contingency_table(self):
        """
        Orient factor data for tabulating into a contingency table

        We want to be able to use the pandas pivot_table function but
        it requires a specific format.
        """
        if self.orientation == "wide":
            return self.cast_wide_to_narrow().data
        else:
            return self.data

    @property
    def labels(self):
        """
        Strings that designate each level that an observation can take, e.g.
         ["bicycle", "motorcycle", "car"]
        """
        if self.orientation == "wide":
            return self.data.columns
        else:
            return self.data['factor_level'].unique()

    @property
    def factor_level_count(self):
        """
        The number of distinct levels that an observation can take.
        """
        return len(self.labels)

    def cast_wide_to_narrow(self):
        """
        Pivot a wide factor into a new factor in a narrow orientation
        """
        if self.orientation != "wide":
            raise TypeError("Factor is already narrow")
        solid_df = self.data
        index_name = solid_df.index.name
        melted = pd.melt(solid_df.reset_index(), id_vars=index_name)
        melted = melted.rename(columns={index_name: "observation_id"})
        narrowed = melted.sort_values("observation_id")
        narrow_data = narrowed.reset_index(drop=True)
        narrow_factor = Factor(narrow_data, self.name, orientation="narrow",
                               multiple_response=self.multiple_response)
        return narrow_factor

    def cast_narrow_to_wide(self):
        """
        Pivot a narrow factor into a new factor with a wide orientation
        """
        if self.orientation != "narrow":
            raise TypeError("Factor is already wide")
        narrow_df = self.data
        wide_df = pd.pivot_table(narrow_df,
                                 values='value',
                                 fill_value=0,
                                 index=['observation_id'],
                                 columns=['factor_level'],
                                 aggfunc=np.sum).sort_index()
        wide_factor = Factor(wide_df, self.name, orientation="wide",
                             multiple_response=self.multiple_response)
        return wide_factor

    def combine_with(self, subordinate):
        """
        Allow combining factors to put multiple vars on an table axis

        Parameters
        ----------
        subordinate : Factor
            Factor instance to merge in as the "lower rung" of the combined
            factor levels

        Returns
        -------
        Factor
            The two factors merged into a new Factor having a column for
            each combination of levels from the original factors

        Notes
        -----
        This method provides a backdoor way to create contingency tables
        involving three or more variables. Although the basic methodology
        in MultipleResponseTable can be extended to accommodate more than
        two variables, doing so substantially complicates the algorithms.

        Instead, you can combine two variables and then investigate whether
        the *combination* of two factors is independent of another factor.

        This method will take two factors and build a 'wide' oriented
        factor that has a column for each combination of levels in the
        original factors. Column names are constructed with this convention:
        "(superior_factor_level, subordinate_factor_level)."

        Please think carefully about the implication of merging factors
        for the particular statistical analysis you're trying to perform.
        The statements "variables X, Y, and Z are independent" is not
        equivalent to "variable X is independent of variables Y and Z
        together".

        """
        if self.orientation == "wide":
            superior = self.cast_wide_to_narrow()
        else:
            superior = self
        if subordinate.orientation == "wide":
            subordinate = subordinate.cast_wide_to_narrow()
        subordinate_data = subordinate.data
        superior_data = superior.data
        l_suffix = "_superior"
        r_suffix = "_subordinate"
        joint_dataframe = _build_joint_dataframe(superior_data,
                                                 subordinate_data,
                                                 l_suffix, r_suffix)
        l_factor_level_col = 'factor_level{}'.format(l_suffix)
        r_factor_level_col = 'factor_level{}'.format(r_suffix)
        combined_data = pd.pivot_table(joint_dataframe,
                                       values='_joint_response',
                                       fill_value=0,
                                       index=['observation_id'],
                                       columns=[l_factor_level_col,
                                                r_factor_level_col],
                                       aggfunc=np.sum)
        flat_columns = []
        top_levels, bottom_levels = combined_data.columns.levels
        top_labels, bottom_labels = combined_data.columns.labels
        for top_label, bottom_label in zip(top_labels, bottom_labels):
            top_level = top_levels[top_label]
            bottom_level = bottom_levels[bottom_label]
            flat_columns.append((top_level, bottom_level))
        # columns come out as tuples...that causes problems for some of
        # pandas indexing methods like .loc
        combined_data.columns = [asunicode(c, 'utf8') for c in flat_columns]
        template = "Combination of ({superior}) and ({subordinate})"
        merged_factor_name = template.format(superior=superior.name,
                                             subordinate=subordinate.name)
        is_multiple_response = (superior.multiple_response or
                                subordinate.multiple_response)
        combined_factor = Factor(combined_data, merged_factor_name,
                                 orientation="wide",
                                 multiple_response=is_multiple_response)
        return combined_factor
