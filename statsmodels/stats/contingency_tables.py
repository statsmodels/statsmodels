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
  collection of contingency tables.

Also contains functions for conducting Mcnemar's test and Cochran's q
test.

Note that the inference procedures may depend on how the data were
sampled.  In general the observed units are independent and
identically distributed.
"""

from __future__ import division

from collections import OrderedDict
from functools import partial
import itertools
import sys

from six import reraise

from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels import iolib
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

import numpy as np
import pandas as pd

from numpy import linalg
from scipy.stats import chi2_contingency, chi2
from scipy import stats


def _make_df_square(table):
    """
    Reindex a pandas DataFrame so that it becomes square, meaning that
    the row and column indices contain the same values, in the same
    order.  The row and column index are extended to achieve this.
    """

    if not isinstance(table, pd.DataFrame):
        return table

    # If the table is not square, make it square
    if table.shape[0] != table.shape[1]:
        ix = list(set(table.index) | set(table.columns))
        table = table.reindex(ix, axis=0)
        table = table.reindex(ix, axis=1)

    # Ensures that the rows and columns are in the same order.
    table = table.reindex(table.columns)

    return table


class _Bunch(object):

    def __repr__(self):
        return "<bunch object containing statsmodels results at {}>".format(id(self))


class ContingencyTableNominalIndependenceResult(_Bunch):

    def __repr__(self):
        return "<bunch object containing contingency table independence results at {}>".format(id(self))

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

    def __repr__(self):
        return "<bunch object containing contingency table independence results at {}>".format(id(self))

    def __str__(self):
        template = ("Multiple Response Contingency Table Independence Result:\n"
                    "table p value: {table_p_value}\n"
                    "cellwise p values: {cellwise_p_values}")
        output = template.format(table_p_value=self.table_p_value,
                                 cellwise_p_values=self.cellwise_p_values)
        return output


class Table(object):
    """
    Analyses that can be performed on a two-way contingency table.

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
        try:
            str_function = unicode
        except NameError:  # Python 3 no longer has the unicode() function
            str_function = str
        return "Contingency Table: \n{table}".format(table=str_function(self.table_orig))

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
            pd.melt

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
            raise ValueError("The length of `row_scores` must match the first dimension of `table`.")

        if len(col_scores) != self.table.shape[1]:
            raise ValueError("The length of `col_scores` must match the second dimension of `table`.")

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
    create a square table.  Otherwise the table should be provided in
    a square form, with the (implicit) row and column categories
    appearing in the same order.
    """

    def __init__(self, table, shift_zeros=True):
        table = _make_df_square(table) # Non-pandas passes through
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

        return self.table[0, 0] * self.table[1, 1] / (self.table[0, 1] * self.table[1, 0])


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
        stubs = ["Odds ratio", "Log odds ratio", "Risk ratio", "Log risk ratio"]

        lcb1, ucb1 = self.oddsratio_confint(alpha, method)
        lcb2, ucb2 = self.log_oddsratio_confint(alpha, method)
        lcb3, ucb3 = self.riskratio_confint(alpha, method)
        lcb4, ucb4 = self.log_riskratio_confint(alpha, method)
        data = [[fmt(x) for x in [self.oddsratio, "", lcb1, ucb1, self.oddsratio_pvalue()]],
                [fmt(x) for x in [self.log_oddsratio, self.log_oddsratio_se, lcb2, ucb2,
                                  self.oddsratio_pvalue()]],
                [fmt(x) for x in [self.riskratio, "", lcb2, ucb2, self.riskratio_pvalue()]],
                [fmt(x) for x in [self.log_riskratio, self.log_riskratio_se, lcb4, ucb4,
                                  self.riskratio_pvalue()]]]
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
            The column index or name of `data` containing the variable
            defining the rows of the contingency table.  The variable
            must have only two distinct values.
        var2 : int or string
            The column index or name of `data` containing the variable
            defining the columns of the contingency table.  The variable
            must have only two distinct values.
        strata : int or string
            The column index of name of `data` containing the variable
            defining the strata.
        data : array-like
            The raw data.  A cross-table for analysis is constructed
            from the first two columns.

        Returns
        -------
        A StratifiedTable instance.
        """

        if not isinstance(data, pd.DataFrame):
            data1 = pd.DataFrame(index=data.index, column=[var1, var2, strata])
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
            tables.append(tab)
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

        statistic = np.sum(self.table[0, 0, :] - self._apb * self._apc / self._n)
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
        lor_va += np.sum((1 - self._apd / self._n) * self._bc / self._n) / bcns**2
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
        v11 = 1 / e11 + 1 / (self._apc - e11) + 1 / (self._apb - e11) + 1 / (self._dma + e11)
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

        co_lcb, co_ucb = self.oddsratio_pooled_confint(alpha=alpha, method=method)
        clo_lcb, clo_ucb = self.logodds_pooled_confint(alpha=alpha, method=method)
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
        corr = int(correction) # convert bool to 0 or 1
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
        b = _Bunch()
        b.statistic = q_stat
        b.df = df
        b.pvalue = pvalue
        return b

    return q_stat, pvalue, df


def shift_zeros(dataframe):
    """
    chi-squared tests can't handle having expected counts of zero. so if the table contains
    a cell with zero observations, jitter it up by 0.5
    :param dataframe:
    :return:
    """
    if dataframe.min().min() == 0:
        return dataframe + 0.5
    else:
        return dataframe


class MRCVTable(object):
    """
    Analyses that can be performed on a two-way contingency table that contains
    'multiple response' categorical variables (e.g. 'choose all that apply' questions).

    Parameters
    ----------

    Attributes
    ----------


    See also
    --------
    scipy.stats.chi2_contingency

    Notes
    -----


    References
    ----------
    Bilder and Loughlin (2004)
    MRCV R Package, Bilder and Koziol
    """

    def __init__(self, row_factors, column_factors):
        self.row_factors = row_factors
        self.column_factors = column_factors
        self.table = self.table_from_factors(row_factors, column_factors)

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        template = "Contingency Table With Multi-Response Categorical Variables (MRCV's).\nData:\n{table}"
        return template.format(table=self.table)

    def __repr__(self):
        return self.table.__repr__()

    @classmethod
    def from_data(cls, data, I, J, rows_factor_name="factor_0", columns_factor_name="factor_1"):
        """
        Construct a Table object from data.

        Parameters
        ----------
        data : array-like
            The raw data, from which a contingency table is constructed
            using the first two columns.
        I: The number of columns in the dataframe corresponding to the first factor
        J: The number of columns in the dataframe corresponding to the second factor

        Returns
        -------
        An MRCVTable instance.
        """

        if isinstance(data, pd.DataFrame):
            rows_data = data.iloc[:, 0:I]
            columns_data = data.iloc[:, I:I + J]
            rows_factor = Factor(rows_data, name=rows_factor_name, orientation="wide")
            columns_factor = Factor(columns_data, name=columns_factor_name, orientation="wide")
        else:
            rows_data = data[:, 0:I]
            columns_data = data[:, I:I + J]
            rows_labels = ["level_{}".format(i) for i in range(0, I)]
            columns_labels = ["level_{}".format(i) for i in range(I, I + J)]
            rows_factor = Factor.from_array(rows_data, labels=rows_labels,
                                            name=rows_factor_name, orientation="wide")
            columns_factor = Factor.from_array(columns_data, labels=columns_labels,
                                               name=columns_factor_name, orientation="wide")

        return cls(rows_factor, columns_factor)

    @classmethod
    def table_from_factors(cls, row_factors, column_factors):
        column_factor, row_factor = cls._extract_and_validate_factors(column_factors, row_factors)
        row_reshaped = row_factor.reshape_for_contingency_table()
        col_reshaped = column_factor.reshape_for_contingency_table()
        joint_dataframe = pd.merge(row_reshaped, col_reshaped, how="inner",
                                   on='observation_id', suffixes=("_row", "_col"))
        # without bool cast, '&' sometimes doesn't know how to compare types
        joint_response = joint_dataframe['value_row'].astype(bool) & joint_dataframe['value_col'].astype(bool)
        joint_dataframe['_joint_response'] = joint_response
        table = pd.pivot_table(joint_dataframe,
                               values='_joint_response',
                               fill_value=0,
                               index=['factor_level_row'],
                               columns=['factor_level_col'],
                               aggfunc=np.sum,)
        return table

    def test_for_independence(self, method="bonferroni"):
        multiple_response_row_factor = any([f.multiple_response for f in self.row_factors])
        multiple_response_column_factor = any([f.multiple_response for f in self.column_factors])
        if method in ("bonferroni", "bon", "b"):
            return self._test_for_independence_using_bonferroni(multiple_response_column_factor,
                                                                multiple_response_row_factor)
        elif method in ("rao-scott-2", "rao", "rs2", "rs"):
            return self._test_for_independence_using_rao_scott(multiple_response_column_factor,
                                                               multiple_response_row_factor)

    def _test_for_independence_using_bonferroni(self,
                                                multiple_response_column_factor,
                                                multiple_response_row_factor):
        mmi_test = self._test_for_marginal_mutual_independence_using_bonferroni_correction
        if multiple_response_row_factor and multiple_response_column_factor:
            spmi_test = self._test_for_single_pairwise_mutual_independence_using_bonferroni
            table_p_value, cellwise_p_values = spmi_test(self.row_factors[0], self.column_factors[0])
            result = self._build_MRCV_result(table_p_value, cellwise_p_values, "Bonferroni", "SPMI")
            return result
        elif multiple_response_column_factor:
            table_p_value, cellwise_p_values = mmi_test(self.row_factors[0], self.column_factors[0])
            result = self._build_MRCV_result(table_p_value, cellwise_p_values, "Bonferroni", "MMI")
            return result
        elif multiple_response_row_factor:
            table_p_value, cellwise_p_values = mmi_test(self.column_factors[0], self.row_factors[0])
            result = self._build_MRCV_result(table_p_value, cellwise_p_values, "Bonferroni", "MMI")
            return result
        else:
            single_response_table = Table(self.table)
            return single_response_table.test_nominal_association()

    def _build_MRCV_result(self, table_p_value, cellwise_p_values, method, independence_type):
        result = MRCVTableNominalIndependenceResult()
        result.table_p_value = table_p_value
        result.cellwise_p_values = cellwise_p_values
        result.method = method
        if independence_type == "MMI":
            result.independence_type = "Marginal Mutual Independence"
        elif independence_type == "SPMI":
            result.independence_type = "Single Pairwise Mutual Independence"
        else:
            raise NotImplementedError("Independence Type Required")
        return result

    def _test_for_independence_using_rao_scott(self,
                                               multiple_response_column_factor,
                                               multiple_response_row_factor):
        mmi_test = self._test_for_marginal_mutual_independence_using_rao_scott_2
        NOT_AVAILABLE = "Not Available From Rao-Scott Method"
        if multiple_response_row_factor and multiple_response_column_factor:
            spmi_test = self._test_for_single_pairwise_mutual_independence_using_rao_scott_2
            table_p_value = spmi_test(self.row_factors[0], self.column_factors[0])
            result = self._build_MRCV_result(table_p_value, NOT_AVAILABLE, "Rao-Scott", "SPMI")
            return result
        elif multiple_response_column_factor:
            table_p_value = mmi_test(self.row_factors[0], self.column_factors[0])
            result = self._build_MRCV_result(table_p_value, NOT_AVAILABLE, "Rao-Scott", "MMI")
            return result
        elif multiple_response_row_factor:
            table_p_value = mmi_test(self.column_factors[0], self.row_factors[0])
            result = self._build_MRCV_result(table_p_value, NOT_AVAILABLE, "Rao-Scott", "MMI")
            return result
        else:
            single_response_table = Table(self.table)
            return single_response_table.test_nominal_association()

    @classmethod
    def _extract_and_validate_factors(cls, column_factors, row_factors):
        try:
            try:
                if len(row_factors) > 1:
                    msg = "we don't currently support tables with more than one factor on the rows"
                    raise NotImplementedError(msg)
                row_factor = row_factors[0]
            except TypeError:
                if isinstance(row_factors, Factor):
                    row_factor = row_factors
                else:
                    msg = "row_factors must be either a list of Factors or a Factor instance"
                    raise NotImplementedError(msg)
            try:
                if len(column_factors) > 2:
                    msg = "we don't currently support tables with more than one factor on the columns"
                    raise NotImplementedError(msg)
                column_factor = column_factors[0]
            except TypeError:
                if isinstance(column_factors, Factor):
                    column_factor = column_factors
                else:
                    msg = "column_factors must be either a list of Factors or a Factor instance"
                    raise NotImplementedError(msg)
            return column_factor, row_factor
        except IndexError:
            explanation = ("Please be sure to pass at "
                           "least 1 factor on both the rows and columns")
            raise IndexError(explanation)

    @staticmethod
    def _build_item_response_table_for_MMI(single_response_factor, multiple_response_factor):
        """
        :param single_response_factor:
        :param multiple_response_factor:
        """
        single_response_dataframe = single_response_factor.data
        multiple_response_dataframe = multiple_response_factor.data

        id_var = single_response_dataframe.index.name
        single_response_melted = pd.melt(single_response_dataframe.reset_index(), id_vars=id_var) \
            .rename(columns={id_var: "observation_id",
                             "value": "selected",
                             "factor_level": "single_response_level"})
        single_response_melted = single_response_melted[single_response_melted.selected == 1]

        multiple_response_dataframe.index.name = "observation_id"
        multiple_response_dataframe = multiple_response_dataframe.reset_index()
        joint_dataframe = pd.merge(single_response_melted.iloc[:, :2], multiple_response_dataframe, how="inner",
                                   on="observation_id")

        single_response_column = joint_dataframe.single_response_level
        item_response_pieces = {}
        for c in multiple_response_factor.labels:
            multiple_response_column = joint_dataframe.loc[:, c]
            crosstab = pd.crosstab(single_response_column, multiple_response_column)
            item_response_pieces[c] = crosstab

        item_response_table = pd.concat(item_response_pieces, axis=1,
                                        names=["multiple_response_level", "selected?"])
        return item_response_table

    @classmethod
    def _calculate_pairwise_chi2s_for_MMI_item_response_table(cls,
                                                              single_response_factor,
                                                              multiple_response_factor):
        item_response_table = cls._build_item_response_table_for_MMI(single_response_factor,
                                                                     multiple_response_factor)
        mmi_chi_squared_by_cell = pd.Series(index=multiple_response_factor.labels)
        for factor_level in item_response_table.columns.levels[0]:
            crosstab = item_response_table.loc[:, factor_level]
            chi2_results = chi2_contingency(crosstab, correction=False)
            chi_squared_statistic, _, _, _ = chi2_results
            mmi_chi_squared_by_cell.loc[factor_level] = chi_squared_statistic
        return mmi_chi_squared_by_cell

    @staticmethod
    def _build_item_response_table_for_SPMI(rows_factor, columns_factor):
        rows_data = rows_factor.data
        columns_data = columns_factor.data
        rows_levels = rows_factor.labels
        row_level_set = set(rows_levels)
        columns_levels = columns_factor.labels
        row_crosstabs = OrderedDict()
        for i, row_name in enumerate(rows_levels):
            column_crosstabs = OrderedDict()
            for j, col_name in enumerate(columns_levels):
                rows = rows_data.iloc[:, i]
                columns = columns_data.iloc[:, j]
                if col_name in row_level_set:
                    col_name = col_name + " (columns)"
                crosstab = pd.crosstab(index=rows, columns=columns, rownames=[row_name], colnames=[col_name])
                column_crosstabs[col_name] = crosstab
            row_crosstab = pd.concat(column_crosstabs, axis=1, names=["column_levels", "selected?"])
            ordered_column_keys = tuple(column_crosstabs.keys())
            row_crosstab = row_crosstab.reindex(columns=ordered_column_keys, level=0)  # preserve column ordering
            row_crosstabs[row_name] = row_crosstab
        item_response_table = pd.concat(row_crosstabs, axis=0, names=["row_levels", "selected?"])
        item_response_table.columns.set_levels(columns_levels, level=0, inplace=True) # undo any name mangling
        item_response_table = item_response_table.reindex(index=row_crosstabs.keys(), level=0)
        return item_response_table

    @classmethod
    def _calculate_pairwise_chi2s_for_SPMI_item_response_table(cls, rows_factor, columns_factor):
        item_response_table = cls._build_item_response_table_for_SPMI(rows_factor, columns_factor)
        rows_levels = item_response_table.index.levels[0]
        columns_levels = item_response_table.columns.levels[0]
        num_col_levels = len(columns_levels)
        num_row_levels = len(rows_levels)
        if item_response_table.shape != (num_row_levels * 2, num_col_levels * 2):
            # crosstab will have degenerate shape (i.e. dimension != r*c)
            # if one level had no observations, i.e. was all 0 or all 1
            # we could pad those out with 0.5 on the unobserved levels, but that can produce
            # extreme chi-square values, e.g. [[1000.5, 0.5], [0.5, 0.5]]
            # has a chi-square of around 250, which sort of makes sense if the top left pairing
            # always co-occurs. But instead of making that extreme assumption, we'll just decline
            # to calculate.
            return pd.DataFrame(np.nan, index=rows_levels, columns=columns_levels)
        chis_spmi = pd.DataFrame(index=rows_levels, columns=columns_levels)
        for i in range(0, num_row_levels * 2, 2):
            for j in range(0, num_col_levels * 2, 2):
                # use integer indexers because level labels are not necessarily unique
                # the "stride by 2" is because pandas does not support integer based indexing with multi-indexes
                # to capture a whole level of the time (i.e. we can't say "give me the first column-group")
                # so we need to manually select both the 0 and 1 column of each column group
                # by providing an explicit couple of index positions
                crosstab = item_response_table.iloc[(i, i+1), (j, j+1)]
                crosstab = shift_zeros(crosstab)
                chi2_results = chi2_contingency(crosstab, correction=False)
                chi_squared_statistic, _, _, _ = chi2_results
                row_level = rows_levels[int(i / 2)]
                column_level = columns_levels[int(j / 2)]
                chis_spmi.loc[row_level, column_level] = chi_squared_statistic
        return chis_spmi

    def _test_for_single_pairwise_mutual_independence_using_bonferroni(self, row_factor, column_factor):
        observed = self._calculate_pairwise_chi2s_for_SPMI_item_response_table(row_factor,
                                                                               column_factor)
        chi2_survival_with_1_dof = partial(chi2.sf, df=1)
        p_value_ij = observed.applymap(chi2_survival_with_1_dof)
        p_value_min = p_value_ij.min().min()
        bonferroni_correction_factor = row_factor.factor_level_count * column_factor.factor_level_count
        cap = lambda x: min(x, 1)
        table_p_value_bonferroni_corrected = cap(bonferroni_correction_factor * p_value_min)
        pairwise_bonferroni_corrected_p_values = (p_value_ij * bonferroni_correction_factor).applymap(cap)
        return table_p_value_bonferroni_corrected, pairwise_bonferroni_corrected_p_values

    def _test_for_single_pairwise_mutual_independence_using_bootstrap(self,
                                                                      row_factor,
                                                                      column_factor,
                                                                      verbose=False):
        W = row_factor.data
        Y = column_factor.data
        I = row_factor.factor_level_count
        J = column_factor.factor_level_count
        spmi_df = pd.concat([W, Y], axis=1)  # type: pd.DataFrame
        chi2_survival_with_1_dof = partial(chi2.sf, df=1)

        b_max = 1000
        n = len(spmi_df)
        q1 = spmi_df.iloc[:, :I]
        q2 = spmi_df.iloc[:, I:I + J]
        X_sq_S_star = []
        X_sq_S_ij_star = pd.DataFrame(index=range(0, I * J), columns=range(0, b_max))
        p_value_b_min = []
        p_value_b_prod = []
        rows_factor_name = row_factor.name
        rows_factor_labels = row_factor.labels
        columns_factor_name = column_factor.name
        columns_factor_labels = column_factor.labels
        for i in range(0, b_max):
            if verbose and i % 50 == 0:
                print("sample {}".format(i))
            # pd.concat requires unique indexes...sampling with replacement produces duplicates
            q1_sample = q1.sample(n, replace=True).reset_index(drop=True)
            q2_sample = q2.sample(n, replace=True).reset_index(drop=True)

            sample_rows_factor = Factor(q1_sample, rows_factor_name,
                                        orientation="wide", multiple_response=True)
            sample_columns_factor = Factor(q2_sample, columns_factor_name,
                                           orientation="wide", multiple_response=True)
            stat_star = self._calculate_pairwise_chi2s_for_SPMI_item_response_table(sample_rows_factor,
                                                                                    sample_columns_factor)
            X_sq_S = stat_star.sum().sum()
            X_sq_S_star.append(X_sq_S)
            X_sq_S_ij_star.append(stat_star)
            p_value_ij = stat_star.applymap(chi2_survival_with_1_dof)
            p_value_min = p_value_ij.min().min()
            p_value_prod = p_value_ij.prod().prod()
            p_value_b_min.append(p_value_min)
            p_value_b_prod.append(p_value_prod)

        observed = self._calculate_pairwise_chi2s_for_SPMI_item_response_table()
        observed_X_sq_S = observed.sum().sum()
        p_value_ij = observed.applymap(chi2_survival_with_1_dof)
        p_value_min = p_value_ij.min().min()

        p_value_boot = np.mean(X_sq_S_star >= observed_X_sq_S)
        print(p_value_boot)

        p_value_boot_min_overall = np.mean(p_value_b_min <= p_value_min)
        print(p_value_boot_min_overall)

    def _test_for_single_pairwise_mutual_independence_using_rao_scott_2(self, row_factor,
                                                                        column_factor):
        observed = self._calculate_pairwise_chi2s_for_SPMI_item_response_table(row_factor,
                                                                               column_factor)
        W = row_factor.data
        Y = column_factor.data
        I = row_factor.factor_level_count
        J = column_factor.factor_level_count
        spmi_df = pd.concat([W, Y], axis=1)  # type: pd.DataFrame

        def count_level_combinations(data, number_of_variables):
            data = data.copy()  # don't modify original dataframe
            level_arguments = [[0, 1]] * number_of_variables
            # the groupby statment requires the variables to be uniquely named
            # but the names aren't used after this
            # so mangling them is fine
            variables = list(range(0, len(data.columns)))
            data.columns = variables
            level_combinations = list(itertools.product(*level_arguments))
            full_combinations = pd.DataFrame(level_combinations, columns=variables)
            full_combinations["_dummy"] = 0
            data['_dummy'] = 1
            data = pd.concat([data, full_combinations]).reset_index(drop=True)
            grouped = data.groupby(list(variables))
            return grouped.sum().reset_index()

        W_count_ordered = count_level_combinations(W, I)
        Y_count_ordered = count_level_combinations(Y, J)
        n_count_ordered = count_level_combinations(spmi_df, I+J)

        n = len(spmi_df)
        G = (W_count_ordered.iloc[:, :-1]).T
        H = (Y_count_ordered.iloc[:, :-1]).T
        combined_counts = n_count_ordered.iloc[:, -1]
        tau = combined_counts / n
        m_row = G.dot(W_count_ordered.iloc[:, -1])
        m_col = H.dot(Y_count_ordered.iloc[:, -1])
        GH = np.kron(G, H)
        m = GH.dot(combined_counts)

        pi = m / n
        pi_row = m_row / n
        pi_col = m_col / n
        j_2r = np.ones((2 ** I, 1))
        i_2r = np.eye(2 ** I)
        j_2c = np.ones((2 ** J, 1))
        i_2c = np.eye(2 ** J)

        G_ij = G.dot(np.kron(i_2r, j_2c.T))
        H_ji = H.dot(np.kron(j_2r.T, i_2c))

        # extra .T's b/c Python handles vector/matrix kronecker differently than R
        H_kron = np.kron(pi_row, H_ji.T).T
        G_kron = np.kron(G_ij.T, pi_col).T
        F = GH - H_kron - G_kron

        mult_cov = np.diag(tau) - np.outer(tau, tau.T)
        sigma = F.dot(mult_cov.dot(F.T))

        D = np.diag(np.kron(pi_row, pi_col) * np.kron(1 - pi_row, 1 - pi_col))
        Di_sigma = np.diag(1 / np.diag(D)).dot(sigma)
        eigenvalues, eigenvectors = linalg.eig(Di_sigma)
        Di_sigma_eigen = np.real(eigenvalues)
        sum_Di_sigma_eigen_sq = (Di_sigma_eigen ** 2).sum()

        observed_X_sq_S = observed.sum().sum()
        X_sq_S_rs2 = I * J * observed_X_sq_S / sum_Di_sigma_eigen_sq
        df_rs2 = (I ** 2) * (J ** 2) / sum_Di_sigma_eigen_sq
        X_sq_S_p_value_rs2 = 1 - chi2.cdf(X_sq_S_rs2, df=df_rs2)
        return X_sq_S_p_value_rs2

    def _test_for_marginal_mutual_independence_using_bonferroni_correction(self,
                                                                           single_response_factor,
                                                                           multiple_response_factor):
        mmi_pairwise_chis = self._calculate_pairwise_chi2s_for_MMI_item_response_table(single_response_factor,
                                                                                       multiple_response_factor)
        c = len(multiple_response_factor.labels)
        r = len(single_response_factor.labels)

        chi2_survival_with_1_dof = partial(chi2.sf, df=(r - 1))

        p_value_ij = mmi_pairwise_chis.apply(chi2_survival_with_1_dof)
        p_value_min = p_value_ij.min()

        bonferroni_correction_factor = c
        cap = lambda x: min(x, 1)
        table_p_value_bonferroni_corrected = cap(bonferroni_correction_factor * p_value_min)
        pairwise_bonferroni_corrected_p_values = (p_value_ij * bonferroni_correction_factor).apply(cap)
        return table_p_value_bonferroni_corrected, pairwise_bonferroni_corrected_p_values

    def _test_for_marginal_mutual_independence_using_rao_scott_2(self,
                                                                 single_response_factor,
                                                                 multiple_response_factor):
        if single_response_factor.orientation == "wide":
            W = single_response_factor.cast_wide_to_narrow().data
            W = W[W['value'] == 1]  # only consider actually selected option
            W.set_index("observation_id", inplace=True)
        else:
            W = single_response_factor.data
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
            mrcv_level_arguments = [[0, 1] for i in range(0, number_of_variables - 1)]
            level_arguments = [list(srcv_level_arguments), ] + mrcv_level_arguments
            variables = ['srcv', ] + list(mrcv.columns)
            level_combinations = list(itertools.product(*level_arguments))
            full_combinations = pd.DataFrame(level_combinations, columns=variables)
            full_combinations["_dummy"] = 0
            data = pd.concat([srcv, mrcv], axis=1)
            data.srcv.value_counts()
            data['_dummy'] = 1
            data = pd.concat([data, full_combinations]).reset_index(drop=True)
            grouped = data.groupby(list(variables))
            result = grouped.sum().reset_index()
            return result

        def count_level_combinations(data, number_of_variables):
            data = data.copy()  # don't modify original dataframe
            level_arguments = [[0, 1] for i in range(0, number_of_variables)]
            variables = data.columns
            level_combinations = list(itertools.product(*level_arguments))
            full_combinations = pd.DataFrame(level_combinations, columns=variables)
            full_combinations["_dummy"] = 0
            data['_dummy'] = 1
            data = pd.concat([data, full_combinations]).reset_index(drop=True)
            grouped = data.groupby(list(variables))
            return grouped.sum().reset_index()

        Y_count_ordered = count_level_combinations(Y, J)
        n_count_ordered = conjoint_combinations(W, Y)
        # n_count_ordered.sort_values("_dummy", inplace=True)

        # need make n_iplus be in same order as SRCV options in the n_counts_ordered_table
        srcv_table_order = n_count_ordered.groupby('srcv').first().index.values
        n_iplus = W.value_counts().reindex(srcv_table_order)
        tau = n_count_ordered.iloc[:, -1].astype(int) / np.repeat(n_iplus, repeats=(2 ** c)).reset_index(drop=True)
        # the R version subtracts 1 from G_tilde because data.matrix converts 0->1 and 1->2
        # (probably because it thinks they're factors and it's internally coding them)
        G_tilde = Y_count_ordered.iloc[:, :-1].T
        I_r = np.eye(r)
        G = np.kron(I_r, G_tilde)
        pi = G.dot(tau)
        m = pi * np.repeat(n_iplus, c)
        a_i = n_iplus / n
        pi_not_j = (1 / n) * np.kron(np.ones(r), np.eye(c)).dot(m)
        j_r = np.ones(r)
        pi_not = np.kron(j_r, pi_not_j)
        I_rc = np.eye(r * c)
        I_c = np.eye(c)
        J_rr = np.ones((r, r))
        A = np.diag(a_i)
        H = I_rc - np.kron(J_rr.dot(A), I_c)
        D = np.kron(np.diag(n / n_iplus), np.diag(pi_not_j) * (1 - pi_not_j))
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
        observed = self._calculate_pairwise_chi2s_for_MMI_item_response_table(single_response_factor,
                                                                              multiple_response_factor)
        observed_X_sq = observed.sum()
        rows_by_columns = ((r - 1) * c)
        X_sq_S_rs2 = rows_by_columns * observed_X_sq / sum_Di_HGVGH_eigen_sq
        df_rs2 = ((r - 1) ** 2) * (c ** 2) / sum_Di_HGVGH_eigen_sq
        X_sq_S_p_value_rs2 = 1 - chi2.cdf(X_sq_S_rs2, df=df_rs2)
        return X_sq_S_p_value_rs2


class Factor(object):
    def __init__(self, dataframe, name, orientation="wide", multiple_response=None):
        self.orientation = orientation
        self.name = name
        if dataframe.index.name is None:
            dataframe.index.name = "observation_id"
        if dataframe.columns.name is None:
            dataframe.columns.name = "factor_level"
        self.data = dataframe
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
        template = "{multiple_response_slug}Factor: {name}\nColumns:{columns}\nData:\n{data}"
        if self.multiple_response:
            multiple_response_slug = "Multiple Response "
        else:
            multiple_response_slug = ""
        return template.format(multiple_response_slug=multiple_response_slug, name=self.name,
                               columns=self.labels, data=self.data)

    def __repr__(self):
        return "Factor at {id} :: {output}".format(id=id(self), output=self.__unicode__())

    def __str__(self):
        return self.__unicode__()

    @classmethod
    def from_array(cls, data, labels, name, orientation="wide", multiple_response=None):
        if len(labels) != data.shape[1]:
            raise ValueError("all columns must have labels")
        data = np.asarray(data, dtype=np.float64)
        dataframe = pd.DataFrame(data, columns=labels)
        factor = cls(dataframe, orientation=orientation,
                     multiple_response=multiple_response, name=name)
        return factor

    def reshape_for_contingency_table(self):
        if self.orientation == "wide":
            return self.cast_wide_to_narrow().data
        else:
            return self.data

    @property
    def labels(self):
        if self.orientation == "wide":
            return self.data.columns
        else:
            return self.data['factor_level'].unique()

    @property
    def factor_level_count(self):
        return len(self.labels)

    def cast_wide_to_narrow(self):
        if self.orientation != "wide":
            raise NotImplementedError("Factor is already narrow")
        solid_df = self.data
        index_name = solid_df.index.name
        melted = pd.melt(solid_df.reset_index(), id_vars=index_name)
        melted = melted.rename(columns={index_name: "observation_id"})
        narrow_data = melted.sort_values("observation_id").reset_index(drop=True)
        narrow_factor = Factor(narrow_data, self.name, orientation="narrow",
                               multiple_response=self.multiple_response)
        return narrow_factor

    def cast_narrow_to_wide(self):
        if self.orientation != "narrow":
            raise NotImplementedError("Factor is already wide")
        narrow_df = self.data
        wide_df = pd.pivot_table(narrow_df, values='value', fill_value=0,
                                 index=['observation_id'], columns=['factor_level'],
                                 aggfunc=np.sum).sort_index()
        wide_factor = Factor(wide_df, self.name, orientation="wide",
                             multiple_response=self.multiple_response)
        return wide_factor

