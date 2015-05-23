"""
Tests for contingency table analyses.
"""

import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
fname = "contingency_table_r_results.csv"
fpath = os.path.join(cur_dir, 'results', fname)
r_results = pd.read_csv(fpath)


tables = [None, None, None]

tables[0] = np.asarray([[23, 15], [19, 31]])

tables[1] = np.asarray([[144, 33, 84, 126],
                        [2, 4, 14, 29],
                        [0, 2, 6, 25],
                        [0, 0, 1, 5]])

tables[2] = np.asarray([[20, 10, 5],
                        [3, 30, 15],
                        [0, 5, 40]])


def test_homogeneity():

    for k,table in enumerate(tables):
        stat, pvalue, df = ctab.homogeneity(table, return_object=False)
        assert_allclose(stat, r_results.loc[k, "homog_stat"])
        assert_allclose(df, r_results.loc[k, "homog_df"])

        # Test Bhapkar via its relationship to Stuart_Maxwell.
        stat1, pvalue, df = ctab.homogeneity(table, method="bhapkar",
                                             return_object=False)
        assert_allclose(stat1, stat / (1 - stat / table.sum()))


def test_ordinal_association():

    for k,table in enumerate(tables):

        row_scores = 1 + np.arange(table.shape[0])
        col_scores = 1 + np.arange(table.shape[1])

        # First set of scores
        rslt = ctab.ordinal_association(table, row_scores,
                                        col_scores, return_object=True)
        assert_allclose(rslt.stat, r_results.loc[k, "lbl_stat"])
        assert_allclose(rslt.stat_e0, r_results.loc[k, "lbl_expval"])
        assert_allclose(rslt.stat_sd0**2, r_results.loc[k, "lbl_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl_chi2"], rtol=1e-5, atol=1e-5)
        assert_allclose(rslt.pvalue, r_results.loc[k, "lbl_pvalue"], rtol=1e-5, atol=1e-5)

        # Second set of scores
        rslt = ctab.ordinal_association(table, row_scores,
                                        col_scores**2, return_object=True)
        assert_allclose(rslt.stat, r_results.loc[k, "lbl2_stat"])
        assert_allclose(rslt.stat_e0, r_results.loc[k, "lbl2_expval"])
        assert_allclose(rslt.stat_sd0**2, r_results.loc[k, "lbl2_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl2_chi2"])
        assert_allclose(rslt.pvalue, r_results.loc[k, "lbl2_pvalue"], rtol=1e-5, atol=1e-5)


def test_symmetry():

    for k,table in enumerate(tables):

        stat, pvalue, df = ctab.symmetry(table, return_object=False)
        assert_allclose(stat, r_results.loc[k, "bowker_stat"])
        assert_equal(df, r_results.loc[k, "bowker_df"])
        assert_allclose(pvalue, r_results.loc[k, "bowker_pvalue"])


def test_mcnemar():

    # Use chi^2 without continuity correction
    stat1, pvalue1 = ctab.mcnemar(tables[0], exact=False,
                                  correction=False)
    stat2, pvalue2, df = ctab.homogeneity(tables[0], return_object=False)
    assert_allclose(stat1, stat2)
    assert_equal(df, 1)

    # Use chi^2 with continuity correction
    stat, pvalue = ctab.mcnemar(tables[0], exact=False,
                                correction=True)
    assert_allclose(pvalue, r_results.loc[0, "homog_cont_p"])

    # Use binomial reference distribution
    stat3, pvalue3 = ctab.mcnemar(tables[0], exact=True)
    assert_allclose(pvalue3, r_results.loc[0, "homog_binom_p"])


def test_cochranq():
    """
    library(CVST)
    table1 = matrix(c(1, 0, 1, 1,
                      0, 1, 1, 1,
                      1, 1, 1, 0,
                      0, 1, 0, 0,
                      0, 1, 0, 0,
                      1, 0, 1, 0,
                      0, 1, 0, 0,
                      1, 1, 1, 1,
                      0, 1, 0, 0), ncol=4, byrow=TRUE)
    rslt1 = cochranq.test(table1)
    table2 = matrix(c(0, 0, 1, 1, 0,
                      0, 1, 0, 1, 0,
                      0, 1, 1, 0, 1,
                      1, 0, 0, 0, 1,
                      1, 1, 0, 0, 0,
                      1, 0, 1, 0, 0,
                      0, 1, 0, 0, 0,
                      0, 0, 1, 1, 0,
                      0, 0, 0, 0, 0), ncol=5, byrow=TRUE)
    rslt2 = cochranq.test(table2)
    """

    table = [[1, 0, 1, 1],
             [0, 1, 1, 1],
             [1, 1, 1, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [1, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 1, 1, 1],
             [0, 1, 0, 0]]
    table = np.asarray(table)

    stat, pvalue, df = ctab.cochrans_q(table, return_object=False)
    assert_allclose(stat, 4.2)
    assert_allclose(df, 3)

    table = [[0, 0, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0]]
    table = np.asarray(table)

    stat, pvalue, df = ctab.cochrans_q(table, return_object=False)
    assert_allclose(stat, 1.2174, rtol=1e-4)
    assert_allclose(df, 4)

    # Cochrane q and Mcnemar are equivalent for 2x2 tables
    data = table[:, 0:2]
    xtab = np.asarray(pd.crosstab(data[:, 0], data[:, 1]))
    stat1, pvalue1, df1 = ctab.cochrans_q(data, return_object=False)
    stat2, pvalue2 = ctab.mcnemar(xtab, exact=False, correction=False)
    assert_allclose(stat1, stat2)
    assert_allclose(pvalue1, pvalue2)



class CheckStratifiedMixin(object):

    def test_common_odds(self):
        assert_allclose(self.rslt.common_odds, self.common_odds,
                        rtol=1e-4, atol=1e-4)


    def test_common_logodds(self):
        assert_allclose(self.rslt.common_logodds, self.common_logodds,
                        rtol=1e-4, atol=1e-4)


    def test_null_odds(self):
        stat, pvalue = self.rslt.test_null_odds(correction=True)
        assert_allclose(stat, self.mh_stat, rtol=1e-4, atol=1e-5)
        assert_allclose(pvalue, self.mh_pvalue, rtol=1e-4, atol=1e-4)


    def test_odds_ratio_confint(self):
        lcb, ucb = self.rslt.odds_ratio_confint()
        assert_allclose(lcb, self.or_lcb, rtol=1e-4, atol=1e-4)
        assert_allclose(ucb, self.or_ucb, rtol=1e-4, atol=1e-4)


    def test_logodds_ratio_confint(self):
        lcb, ucb = self.rslt.logodds_ratio_confint()
        assert_allclose(lcb, np.log(self.or_lcb), rtol=1e-4,
                        atol=1e-4)
        assert_allclose(ucb, np.log(self.or_ucb), rtol=1e-4,
                        atol=1e-4)


class TestStratified1(CheckStratifiedMixin):
    """
    data = array(c(0, 0, 6, 5,
                   3, 0, 3, 6,
                   6, 2, 0, 4,
                   5, 6, 1, 0,
                   2, 5, 0, 0),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    """

    def __init__(self):

        tables = [None] * 5
        tables[0] = np.array([[0, 0], [6, 5]])
        tables[1] = np.array([[3, 0], [3, 6]])
        tables[2] = np.array([[6, 2], [0, 4]])
        tables[3] = np.array([[5, 6], [1, 0]])
        tables[4] = np.array([[2, 5], [0, 0]])

        self.rslt = ctab.StratifiedTables(tables)

        self.common_odds = 7
        self.common_logodds = np.log(7)
        self.mh_stat = 3.9286
        self.mh_pvalue = 0.04747
        self.or_lcb = 1.026713
        self.or_ucb = 47.725133


class TestStratified2(CheckStratifiedMixin):
    """
    data = array(c(20, 14, 10, 24,
                   15, 12, 3, 15,
                   3, 2, 3, 2,
                   12, 3, 7, 5,
                   1, 0, 3, 2),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    """

    def __init__(self):
        tables = [None] * 5
        tables[0] = np.array([[20, 14], [10, 24]])
        tables[1] = np.array([[15, 12], [3, 15]])
        tables[2] = np.array([[3, 2], [3, 2]])
        tables[3] = np.array([[12, 3], [7, 5]])
        tables[4] = np.array([[1, 0], [3, 2]])

        self.rslt = ctab.StratifiedTables(tables)

        self.common_odds = 3.5912
        self.common_logodds = np.log(3.5912)

        self.mh_stat = 11.8852
        self.mh_pvalue = 0.0005658

        self.or_lcb = 1.781135
        self.or_ucb = 7.240633
