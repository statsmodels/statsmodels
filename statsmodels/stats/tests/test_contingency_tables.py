"""
Tests for contingency table analyses.
"""

import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal, assert_raises
import os
import statsmodels.api as sm

from statsmodels.datasets import presidential2016

cur_dir = os.path.dirname(os.path.abspath(__file__))
fname = "contingency_table_r_results.csv"
results_dirpath = os.path.join(cur_dir, 'results')
fpath = os.path.join(results_dirpath, fname)
r_results = pd.read_csv(fpath)
presidential_data = sm.datasets.presidential2016.load_pandas().data


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
        st = sm.stats.SquareTable(table, shift_zeros=False)
        hm = st.homogeneity()
        assert_allclose(hm.statistic, r_results.loc[k, "homog_stat"])
        assert_allclose(hm.df, r_results.loc[k, "homog_df"])

        # Test Bhapkar via its relationship to Stuart_Maxwell.
        hmb = st.homogeneity(method="bhapkar")
        assert_allclose(hmb.statistic, hm.statistic / (1 - hm.statistic / table.sum()))


def test_SquareTable_from_data():

    np.random.seed(434)
    df = pd.DataFrame(index=range(100), columns=["v1", "v2"])
    df["v1"] = np.random.randint(0, 5, 100)
    df["v2"] = np.random.randint(0, 5, 100)
    table = pd.crosstab(df["v1"], df["v2"])

    rslt1 = ctab.SquareTable(table)
    rslt2 = ctab.SquareTable.from_data(df)
    rslt3 = ctab.SquareTable(np.asarray(table))

    assert_equal(rslt1.summary().as_text(),
                 rslt2.summary().as_text())

    assert_equal(rslt2.summary().as_text(),
                 rslt3.summary().as_text())

    s = str(rslt1)
    assert_equal(s.startswith('A 5x5 contingency table with counts:'), True)
    assert_equal(rslt1.table[0, 0], 8.)


def test_SquareTable_nonsquare():

    tab = [[1, 0, 3], [2, 1, 4], [3, 0, 5]]
    df = pd.DataFrame(tab, index=[0, 1, 3], columns=[0, 2, 3])

    df2 = ctab.SquareTable(df, shift_zeros=False)

    e = np.asarray([[1, 0, 0, 3], [2, 0, 1, 4], [0, 0, 0, 0], [3, 0, 0, 5]],
                   dtype=np.float64)

    assert_equal(e, df2.table)


def test_cumulative_odds():

    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)

    cum_odds = tbl_obj.cumulative_oddsratios
    assert_allclose(cum_odds[0, 0], 28 / float(5 * 11))
    assert_allclose(cum_odds[0, 1], (3 * 15) / float(3 * 24), atol=1e-5,
                    rtol=1e-5)
    assert_allclose(np.log(cum_odds), tbl_obj.cumulative_log_oddsratios,
                    atol=1e-5, rtol=1e-5)


def test_local_odds():

    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)

    loc_odds = tbl_obj.local_oddsratios
    assert_allclose(loc_odds[0, 0], 5 / 8.)
    assert_allclose(loc_odds[0, 1], 12 / float(15), atol=1e-5,
                    rtol=1e-5)
    assert_allclose(np.log(loc_odds), tbl_obj.local_log_oddsratios,
                    atol=1e-5, rtol=1e-5)


def test_stratified_table_cube():
    # Test that we can pass a rank 3 ndarray or a list of rank 2
    # ndarrays to StratifiedTable and get the same results.

    tab1 = [[[8, 9], [6, 7]], [[4, 9], [5, 5]], [[8, 8], [9, 11]]]
    tab2 = np.asarray(tab1).T

    ct1 = ctab.StratifiedTable(tab1)
    ct2 = ctab.StratifiedTable(tab2)

    assert_allclose(ct1.oddsratio_pooled, ct2.oddsratio_pooled)
    assert_allclose(ct1.logodds_pooled, ct2.logodds_pooled)


def test_resids():

    # CHD x serum data
    table = [[12, 8, 31, 41], [307, 246, 439, 245]]

    # These results come from SAS
    fit = [[22.083, 17.583, 32.536, 19.798],
           [296.92, 236.42, 437.46, 266.2]]
    c2 = [[4.6037, 5.223, 0.0725, 22.704],
          [0.3424, 0.3885, 0.0054, 1.6886]]

    # These are regression tests
    pr = np.array([[-2.14562121, -2.28538719, -0.26923882,  4.7649169 ],
                   [ 0.58514314,  0.62325942,  0.07342547, -1.29946443]])
    sr = np.array([[-2.55112945, -2.6338782 , -0.34712127,  5.5751083 ],
                   [ 2.55112945,  2.6338782 ,  0.34712127, -5.5751083 ]])

    tab = ctab.Table(table)
    assert_allclose(tab.fittedvalues, fit, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.chi2_contribs, c2, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.resid_pearson, pr, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.standardized_resids, sr, atol=1e-4, rtol=1e-4)


def test_ordinal_association():

    for k,table in enumerate(tables):

        row_scores = 1 + np.arange(table.shape[0])
        col_scores = 1 + np.arange(table.shape[1])

        # First set of scores
        rslt = ctab.Table(table, shift_zeros=False).test_ordinal_association(row_scores, col_scores)
        assert_allclose(rslt.statistic, r_results.loc[k, "lbl_stat"])
        assert_allclose(rslt.null_mean, r_results.loc[k, "lbl_expval"])
        assert_allclose(rslt.null_sd**2, r_results.loc[k, "lbl_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl_chi2"], rtol=1e-5, atol=1e-5)
        assert_allclose(rslt.pvalue, r_results.loc[k, "lbl_pvalue"], rtol=1e-5, atol=1e-5)

        # Second set of scores
        rslt = ctab.Table(table, shift_zeros=False).test_ordinal_association(row_scores, col_scores**2)
        assert_allclose(rslt.statistic, r_results.loc[k, "lbl2_stat"])
        assert_allclose(rslt.null_mean, r_results.loc[k, "lbl2_expval"])
        assert_allclose(rslt.null_sd**2, r_results.loc[k, "lbl2_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl2_chi2"])
        assert_allclose(rslt.pvalue, r_results.loc[k, "lbl2_pvalue"], rtol=1e-5, atol=1e-5)


def test_chi2_association():

    np.random.seed(8743)

    table = np.random.randint(10, 30, size=(4, 4))

    from scipy.stats import chi2_contingency
    rslt_scipy = chi2_contingency(table)

    b = ctab.Table(table).test_nominal_association()

    assert_allclose(b.statistic, rslt_scipy[0])
    assert_allclose(b.pvalue, rslt_scipy[1])


def test_symmetry():

    for k,table in enumerate(tables):
        st = sm.stats.SquareTable(table, shift_zeros=False)
        b = st.symmetry()
        assert_allclose(b.statistic, r_results.loc[k, "bowker_stat"])
        assert_equal(b.df, r_results.loc[k, "bowker_df"])
        assert_allclose(b.pvalue, r_results.loc[k, "bowker_pvalue"])


def test_mcnemar():

    # Use chi^2 without continuity correction
    b1 = ctab.mcnemar(tables[0], exact=False, correction=False)

    st = sm.stats.SquareTable(tables[0])
    b2 = st.homogeneity()
    assert_allclose(b1.statistic, b2.statistic)
    assert_equal(b2.df, 1)

    # Use chi^2 with continuity correction
    b3 = ctab.mcnemar(tables[0], exact=False, correction=True)
    assert_allclose(b3.pvalue, r_results.loc[0, "homog_cont_p"])

    # Use binomial reference distribution
    b4 = ctab.mcnemar(tables[0], exact=True)
    assert_allclose(b4.pvalue, r_results.loc[0, "homog_binom_p"])

def test_from_data_stratified():

    df = pd.DataFrame([[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1]]).T
    e = np.asarray([[[0, 1], [1, 1]], [[2, 2], [1, 0]]])

    # Test pandas
    tab1 = ctab.StratifiedTable.from_data(0, 1, 2, df)
    assert_equal(tab1.table, e)

    # Test ndarray
    tab1 = ctab.StratifiedTable.from_data(0, 1, 2, np.asarray(df))
    assert_equal(tab1.table, e)

def test_from_data_2x2():

    df = pd.DataFrame([[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0]]).T
    e = np.asarray([[1, 2], [4, 1]])

    # Test pandas
    tab1 = ctab.Table2x2.from_data(df, shift_zeros=False)
    assert_equal(tab1.table, e)

    # Test ndarray
    tab1 = ctab.Table2x2.from_data(np.asarray(df), shift_zeros=False)
    assert_equal(tab1.table, e)


def test_cochranq():
    # library(CVST)
    # table1 = matrix(c(1, 0, 1, 1,
    #                   0, 1, 1, 1,
    #                   1, 1, 1, 0,
    #                   0, 1, 0, 0,
    #                   0, 1, 0, 0,
    #                   1, 0, 1, 0,
    #                   0, 1, 0, 0,
    #                   1, 1, 1, 1,
    #                   0, 1, 0, 0), ncol=4, byrow=TRUE)
    # rslt1 = cochranq.test(table1)
    # table2 = matrix(c(0, 0, 1, 1, 0,
    #                   0, 1, 0, 1, 0,
    #                   0, 1, 1, 0, 1,
    #                   1, 0, 0, 0, 1,
    #                   1, 1, 0, 0, 0,
    #                   1, 0, 1, 0, 0,
    #                   0, 1, 0, 0, 0,
    #                   0, 0, 1, 1, 0,
    #                   0, 0, 0, 0, 0), ncol=5, byrow=TRUE)
    # rslt2 = cochranq.test(table2)

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

    # Cochran's q and Mcnemar are equivalent for 2x2 tables
    data = table[:, 0:2]
    xtab = np.asarray(pd.crosstab(data[:, 0], data[:, 1]))
    b1 = ctab.cochrans_q(data, return_object=True)
    b2 = ctab.mcnemar(xtab, exact=False, correction=False)
    assert_allclose(b1.statistic, b2.statistic)
    assert_allclose(b1.pvalue, b2.pvalue)

    # Test for printing bunch
    assert_equal(str(b1).startswith("df          1\npvalue      0.65"), True)


class CheckStratifiedMixin(object):

    @classmethod
    def initialize(cls, tables):
        cls.rslt = ctab.StratifiedTable(tables)
        cls.rslt_0 = ctab.StratifiedTable(tables, shift_zeros=True)
        tables_pandas = [pd.DataFrame(x) for x in tables]
        cls.rslt_pandas = ctab.StratifiedTable(tables_pandas)


    def test_oddsratio_pooled(self):
        assert_allclose(self.rslt.oddsratio_pooled, self.oddsratio_pooled,
                        rtol=1e-4, atol=1e-4)


    def test_logodds_pooled(self):
        assert_allclose(self.rslt.logodds_pooled, self.logodds_pooled,
                        rtol=1e-4, atol=1e-4)


    def test_null_odds(self):
        rslt = self.rslt.test_null_odds(correction=True)
        assert_allclose(rslt.statistic, self.mh_stat, rtol=1e-4, atol=1e-5)
        assert_allclose(rslt.pvalue, self.mh_pvalue, rtol=1e-4, atol=1e-4)


    def test_oddsratio_pooled_confint(self):
        lcb, ucb = self.rslt.oddsratio_pooled_confint()
        assert_allclose(lcb, self.or_lcb, rtol=1e-4, atol=1e-4)
        assert_allclose(ucb, self.or_ucb, rtol=1e-4, atol=1e-4)


    def test_logodds_pooled_confint(self):
        lcb, ucb = self.rslt.logodds_pooled_confint()
        assert_allclose(lcb, np.log(self.or_lcb), rtol=1e-4,
                        atol=1e-4)
        assert_allclose(ucb, np.log(self.or_ucb), rtol=1e-4,
                        atol=1e-4)


    def test_equal_odds(self):

        if not hasattr(self, "or_homog"):
            return

        rslt = self.rslt_0.test_equal_odds()
        assert_allclose(rslt.statistic, self.or_homog, rtol=1e-4, atol=1e-4)
        assert_allclose(rslt.pvalue, self.or_homog_p, rtol=1e-4, atol=1e-4)


    def test_pandas(self):

        assert_equal(self.rslt.summary().as_text(),
                     self.rslt_pandas.summary().as_text())


    def test_from_data(self):

        np.random.seed(241)
        df = pd.DataFrame(index=range(100), columns=("v1", "v2", "strat"))
        df["v1"] = np.random.randint(0, 2, 100)
        df["v2"] = np.random.randint(0, 2, 100)
        df["strat"] = np.kron(np.arange(10), np.ones(10))

        tables = []
        for k in range(10):
            ii = np.arange(10*k, 10*(k+1))
            tables.append(pd.crosstab(df.loc[ii, "v1"], df.loc[ii, "v2"]))

        rslt1 = ctab.StratifiedTable(tables)
        rslt2 = ctab.StratifiedTable.from_data("v1", "v2", "strat", df)

        assert_equal(rslt1.summary().as_text(), rslt2.summary().as_text())


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

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[0, 0], [6, 5]])
        tables[1] = np.array([[3, 0], [3, 6]])
        tables[2] = np.array([[6, 2], [0, 4]])
        tables[3] = np.array([[5, 6], [1, 0]])
        tables[4] = np.array([[2, 5], [0, 0]])

        cls.initialize(tables)

        cls.oddsratio_pooled = 7
        cls.logodds_pooled = np.log(7)
        cls.mh_stat = 3.9286
        cls.mh_pvalue = 0.04747
        cls.or_lcb = 1.026713
        cls.or_ucb = 47.725133


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

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[20, 14], [10, 24]])
        tables[1] = np.array([[15, 12], [3, 15]])
        tables[2] = np.array([[3, 2], [3, 2]])
        tables[3] = np.array([[12, 3], [7, 5]])
        tables[4] = np.array([[1, 0], [3, 2]])

        cls.initialize(tables)

        cls.oddsratio_pooled = 3.5912
        cls.logodds_pooled = np.log(3.5912)

        cls.mh_stat = 11.8852
        cls.mh_pvalue = 0.0005658

        cls.or_lcb = 1.781135
        cls.or_ucb = 7.240633


class TestStratified3(CheckStratifiedMixin):
    """
    data = array(c(313, 512, 19, 89,
                   207, 353, 8, 17,
                   205, 120, 391, 202,
                   278, 139, 244, 131,
                   138, 53, 299, 94,
                   351, 22, 317, 24),
                   dim=c(2, 2, 6))
    rslt = mantelhaen.test(data)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 6
        tables[0] = np.array([[313, 512], [19, 89]])
        tables[1] = np.array([[207, 353], [8, 17]])
        tables[2] = np.array([[205, 120], [391, 202]])
        tables[3] = np.array([[278, 139], [244, 131]])
        tables[4] = np.array([[138, 53], [299, 94]])
        tables[5] = np.array([[351, 22], [317, 24]])

        cls.initialize(tables)

        cls.oddsratio_pooled = 1.101879
        cls.logodds_pooled = np.log(1.101879)

        cls.mh_stat = 1.3368
        cls.mh_pvalue = 0.2476

        cls.or_lcb = 0.9402012
        cls.or_ucb = 1.2913602

        cls.or_homog = 18.83297
        cls.or_homog_p = 0.002064786


class Check2x2Mixin(object):
    @classmethod
    def initialize(cls):
        cls.tbl_obj = ctab.Table2x2(cls.table)
        cls.tbl_data_obj = ctab.Table2x2.from_data(cls.data)

    def test_oddsratio(self):
        assert_allclose(self.tbl_obj.oddsratio, self.oddsratio)


    def test_log_oddsratio(self):
        assert_allclose(self.tbl_obj.log_oddsratio, self.log_oddsratio)


    def test_log_oddsratio_se(self):
        assert_allclose(self.tbl_obj.log_oddsratio_se, self.log_oddsratio_se)


    def test_oddsratio_pvalue(self):
        assert_allclose(self.tbl_obj.oddsratio_pvalue(), self.oddsratio_pvalue)


    def test_oddsratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.oddsratio_confint(0.05)
        lcb2, ucb2 = self.oddsratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)


    def test_riskratio(self):
        assert_allclose(self.tbl_obj.riskratio, self.riskratio)


    def test_log_riskratio(self):
        assert_allclose(self.tbl_obj.log_riskratio, self.log_riskratio)


    def test_log_riskratio_se(self):
        assert_allclose(self.tbl_obj.log_riskratio_se, self.log_riskratio_se)


    def test_riskratio_pvalue(self):
        assert_allclose(self.tbl_obj.riskratio_pvalue(), self.riskratio_pvalue)


    def test_riskratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.riskratio_confint(0.05)
        lcb2, ucb2 = self.riskratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)


    def test_log_riskratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.log_riskratio_confint(0.05)
        lcb2, ucb2 = self.log_riskratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)


    def test_from_data(self):
        assert_equal(self.tbl_obj.summary().as_text(),
                     self.tbl_data_obj.summary().as_text())

    def test_summary(self):

        assert_equal(self.tbl_obj.summary().as_text(),
                     self.summary_string)

class Test2x2_1(Check2x2Mixin):

    @classmethod
    def setup_class(cls):
        data = np.zeros((8, 2))
        data[:, 0] = [0, 0, 1, 1, 0, 0, 1, 1]
        data[:, 1] = [0, 1, 0, 1, 0, 1, 0, 1]
        cls.data = np.asarray(data)
        cls.table = np.asarray([[2, 2], [2, 2]])

        cls.oddsratio = 1.
        cls.log_oddsratio = 0.
        cls.log_oddsratio_se = np.sqrt(2)
        cls.oddsratio_confint = [0.062548836166112329, 15.987507702689751]
        cls.oddsratio_pvalue = 1.
        cls.riskratio = 1.
        cls.log_riskratio = 0.
        cls.log_riskratio_se = 1 / np.sqrt(2)
        cls.riskratio_pvalue = 1.
        cls.riskratio_confint = [0.25009765325990629,
                                  3.9984381579173824]
        cls.log_riskratio_confint = [-1.3859038243496782,
                                      1.3859038243496782]
        ss = [  '               Estimate   SE   LCB    UCB   p-value',
                '---------------------------------------------------',
                'Odds ratio        1.000        0.063 15.988   1.000',
                'Log odds ratio    0.000 1.414 -2.772  2.772   1.000',
                'Risk ratio        1.000        0.250  3.998   1.000',
                'Log risk ratio    0.000 0.707 -1.386  1.386   1.000',
                '---------------------------------------------------']
        cls.summary_string = '\n'.join(ss)
        cls.initialize()



# MRCV R values calculated by hand in this notebook:
# https://github.com/rogueleaderr/statsmodels_supplementary_docs/blob/master/MRCV%20R%20Reference%20Version.ipynb
def test_MMI_item_response_table():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, :6],
                                         presidential_data.columns[:6],
                              "expected_choice", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                            presidential_data.columns[6:11],
                                 "believe_true", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    build_table = multiple_response_table._item_response_table_for_MMI
    srcv_item_response_table_python = build_table(rows_factor,
                                          columns_factor)
    result_path = "srcv_r_item_response_table_result.csv"
    fpath = os.path.join(results_dirpath, result_path)
    srcv_item_response_table_r = pd.DataFrame.from_csv(fpath)
    # R writes out the csv in a weird flattened table with the column labels
    #  as "term", "term", "term"... so indexing sensibly is hard. also we
    # can't reindex either dataframe to match the
    # column order of the other b/c the column orders are lost
    # also the python table has nested columns while the R csv is flattened
    # so the striding by 2 matches columns appropriately
    for i in range(0, len(columns_factor.labels) * 2, 2):
        c = columns_factor.labels[i // 2]
        r_left_offset = i
        r_right_offset = i + 2
        py_group = srcv_item_response_table_python.loc[:, c]
        r_group = srcv_item_response_table_r.iloc[:,
                  r_left_offset:r_right_offset]
        assert_allclose(py_group.values, r_group)


def test_SPMI_item_response_table():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                         presidential_data.columns[6:11],
                              "believe_true", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 11:],
                                            presidential_data.columns[11:],
                                 "why_uncertain", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    build = multiple_response_table._item_response_table_for_SPMI
    spmi_item_response_table_python = build(rows_factor, columns_factor)
    result_path = "spmi_r_item_response_table_result.csv"
    fpath = os.path.join(results_dirpath, result_path)
    spmi_item_response_table_r = pd.DataFrame.from_csv(fpath)
    assert_allclose(spmi_item_response_table_r.values,
                    spmi_item_response_table_python.values)


def test_calculate_pairwise_chi2s_for_MMI_item_response_table():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, :6],
                                         presidential_data.columns[:6],
                              "expected_choice", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                            presidential_data.columns[6:11],
                                 "believe_true", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    calculate = multiple_response_table._chi2s_for_MMI_item_response_table
    pairwise_chis = calculate(rows_factor, columns_factor)
    r_results_fname = "srcv_r_all_chis_result.csv"
    r_results_fpath = os.path.join(results_dirpath, r_results_fname)
    results_from_r = pd.Series.from_csv(r_results_fpath)
    assert_allclose(pairwise_chis, results_from_r)


def test_multiple_mutual_independence_false_using_bonferroni():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, :6],
                                         presidential_data.columns[:6],
                              "expected_choice", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                            presidential_data.columns[6:11],
                                 "believe_true", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    bonferroni_test = multiple_response_table._test_MMI_using_bonferroni
    p_value_overall, p_values_cellwise = bonferroni_test(rows_factor,
                                                       columns_factor)
    fpath = os.path.join(results_dirpath, "srcv_r_bonferroni.csv")
    r_result = pd.DataFrame.from_csv(fpath)
    p_value_overall_r = r_result["p.value.bon"]
    cell_p_values_r = r_result.iloc[:, 1:]
    reshaped_python_values = p_values_cellwise.values.reshape(5, 1)
    assert_allclose(reshaped_python_values, cell_p_values_r.T)
    assert_allclose(p_value_overall_r, p_value_overall)


def test_multiple_mutual_independence_false_using_rao_scott_2():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, :6],
                                         presidential_data.columns[:6],
                              "expected_choice", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                            presidential_data.columns[6:11],
                                 "believe_true", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    rao_scott_2_test = multiple_response_table._test_MMI_using_rao_scott_2
    p_value_overall = rao_scott_2_test(rows_factor, columns_factor)
    fpath = os.path.join(results_dirpath, "srcv_r_rao_scott.csv")
    r_result = pd.DataFrame.from_csv(fpath)
    p_value_overall_r = r_result["p.value.rs2"]
    assert_allclose(p_value_overall_r, p_value_overall)


def test_calculate_pairwise_chi2s_for_SPMI_item_response_table():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                         presidential_data.columns[6:11],
                             "believe_true", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 11:],
                                            presidential_data.columns[11:],
                                 "why_uncertain", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    calculate = multiple_response_table._chi2s_for_SPMI_item_response_table
    spmi_pairwise_chis_python = calculate(rows_factor, columns_factor)
    r_results_fname = "spmi_r_pairwise_chis_result.csv"
    r_results_fpath = os.path.join(results_dirpath, r_results_fname)
    spmi_pairwise_chis_r = pd.DataFrame.from_csv(r_results_fpath)
    assert_allclose(spmi_pairwise_chis_r.values.astype(float),
                    spmi_pairwise_chis_python.values.astype(float))


def test_SPMI_false_using_bonferroni():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                         presidential_data.columns[6:11],
                              "believe_true", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 11:],
                                            presidential_data.columns[11:],
                                 "why_uncertain", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    test = multiple_response_table._test_SPMI_using_bonferroni
    result = test(rows_factor, columns_factor)
    p_value_overall_bonferroni, cellwise_p_bonferroni_python = result
    fpath = os.path.join(results_dirpath, "spmi_r_bonferroni.csv")
    spmi_bonferroni_r = pd.DataFrame.from_csv(fpath)

    p_value_overall_r = spmi_bonferroni_r["p.value.bon"]
    cell_p_values_r = spmi_bonferroni_r.iloc[:, 1:]

    assert_allclose(cellwise_p_bonferroni_python, cell_p_values_r)
    assert_allclose(p_value_overall_r, p_value_overall_bonferroni)


def test_SPMI_false_using_rao_scott_2():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                         presidential_data.columns[6:11],
                              "believe_true", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 11:],
                                            presidential_data.columns[11:],
                                 "why_uncertain", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    rao_scott_2_test = multiple_response_table._test_SPMI_using_rao_scott_2
    p_value_overall = rao_scott_2_test(rows_factor, columns_factor)
    fpath = os.path.join(results_dirpath, "spmi_r_rao_scott.csv")
    r_result = pd.DataFrame.from_csv(fpath)
    p_value_overall_r = r_result["p.value.rs2"]
    assert_allclose(p_value_overall_r, p_value_overall)


def build_random_single_select(n=10000, choices=None):
    if choices:
        k = len(choices)
    else:
        k = 3
        choices = ["sedan", "truck", "motorcycle"]
    car_type = np.random.randint(k, size=(n)) + 1
    base_pop = pd.DataFrame(car_type).reset_index()
    base_pop.columns = ['person', 'choice']
    base_pop['_response'] = 1
    dataframe = pd.pivot_table(base_pop,
                               values='_response',
                               fill_value=0,
                               index='person',
                               columns='choice',
                               aggfunc=np.sum,
                               margins=False)
    car_choice = dataframe.copy()
    car_choice.columns = choices
    return car_choice


def test_multiple_mutual_independence_true():
    np.random.seed(100)
    food_choices = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                                columns=["eggs", "cheese", "candy",
                                         "sushi", "none"])
    car_choice = build_random_single_select()
    srcv = ctab.Factor.from_array(car_choice, car_choice.columns,
                                  "car_choice", orientation="wide")
    mrcv = ctab.Factor.from_array(food_choices, food_choices.columns,
                                  "food_choices", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([srcv, ], [mrcv, ])
    rao_scott_2_test = multiple_response_table._test_MMI_using_rao_scott_2
    rao_p_value = rao_scott_2_test(srcv, mrcv)
    np.testing.assert_(rao_p_value >= 0.05)
    bonferroni_test = multiple_response_table._test_MMI_using_bonferroni
    bonferroni_p_value_overall, \
    bonferroni_cell_p_values = bonferroni_test(srcv, mrcv)
    np.testing.assert_(bonferroni_p_value_overall >= 0.05)
    np.testing.assert_(np.all(bonferroni_cell_p_values >= 0.05))


def test_simultaneous_pairwise_mutual_independence_true():
    np.random.seed(100)
    food_choices = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                                columns=["eggs", "cheese", "candy",
                                         "sushi", "none"])
    language = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                                           columns=["English", "French",
                                                    "Mandarin", "Hungarian",
                                                    "none"])
    mrcv_1 = ctab.Factor.from_array(language, language.columns,
                                    "car_choice", orientation="wide")
    mrcv_2 = ctab.Factor.from_array(food_choices, food_choices.columns,
                                    "food_choices", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    rao_scott_2_test = multiple_response_table._test_SPMI_using_rao_scott_2
    rao_p_value = rao_scott_2_test(mrcv_1, mrcv_2)
    np.testing.assert_(rao_p_value >= 0.05)
    bonferroni_test = multiple_response_table._test_SPMI_using_bonferroni
    result = bonferroni_test(mrcv_1, mrcv_2)
    bonferroni_p_value_overall, bonferroni_cell_p_values = result
    np.testing.assert_(bonferroni_p_value_overall >= 0.05)
    np.testing.assert_(np.all(bonferroni_cell_p_values >= 0.05))


def test_overlapping_names_allowed():
    # Hit a bug in development if two factors
    # shared levels with the same name
    np.random.seed(100)
    food_choices = ["eggs", "cheese", "candy", "sushi", "none"]
    best_food = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                             columns=food_choices)
    worst_food = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                              columns=food_choices)
    mrcv_1 = ctab.Factor.from_array(worst_food, worst_food.columns,
                                    "car_choice", orientation="wide")
    mrcv_2 = ctab.Factor.from_array(best_food, best_food.columns,
                                    "best_food", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    rao_scott_2_test = multiple_response_table._test_SPMI_using_rao_scott_2
    rao_p_value = rao_scott_2_test(mrcv_1, mrcv_2)
    np.testing.assert_(rao_p_value >= 0.05)

    car_choice = build_random_single_select(n=1000, choices=food_choices)
    srcv = ctab.Factor.from_array(car_choice, food_choices,
                                  "srcv", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([srcv, ],
                                                         [mrcv_2, ])
    rao_scott_2_test = multiple_response_table._test_SPMI_using_rao_scott_2
    rao_p_value = rao_scott_2_test(mrcv_1, mrcv_2)
    np.testing.assert_(rao_p_value >= 0.05)


def test_duplicate_names_allowed():
    np.random.seed(100)
    # Hit a bug in development if two levels had the same name
    food_choices = ["eggs", "eggs", "candy", "eggs", "none"]
    best_food = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                             columns=food_choices)
    worst_food = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                              columns=food_choices)
    mrcv_1 = ctab.Factor.from_array(worst_food, worst_food.columns,
                                    "", orientation="wide")
    mrcv_2 = ctab.Factor.from_array(best_food, best_food.columns,
                                    "", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    result = multiple_response_table.test_for_independence(method="rao")
    np.testing.assert_(result.p_value_overall >= 0.05)

    car_choice = build_random_single_select(n=1000, choices=food_choices)
    srcv = ctab.Factor.from_array(car_choice, food_choices, "srcv",
                                  orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([srcv, ],
                                                         [mrcv_2, ])
    result = multiple_response_table.test_for_independence(method="rao")
    np.testing.assert_(result.p_value_overall >= 0.05)

    # deduplicator modifies in-place so need to recreate data
    car_choice = build_random_single_select(n=1000, choices=food_choices)
    srcv = ctab.Factor.from_array(car_choice, food_choices, "srcv",
                                  orientation="wide")
    best_food = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                              columns=food_choices)
    mrcv_2 = ctab.Factor.from_array(best_food, best_food.columns,
                                    "best_food", orientation="wide")
    narrow_srcv = srcv.cast_wide_to_narrow()
    narrow_mrcv = mrcv_2.cast_wide_to_narrow()
    multiple_response_table = ctab.MultipleResponseTable([narrow_srcv, ],
                                                         [narrow_mrcv, ])
    result = multiple_response_table.test_for_independence(method="rao")
    np.testing.assert_(result.p_value_overall >= 0.05)


def test_MRCV_table_from_data():
    multiple_response_questions = presidential_data.iloc[:, 6:]
    construct = ctab.MultipleResponseTable.from_data
    table = construct(multiple_response_questions, 5, 5)
    expected = np.array([44, 49, 15, 22, 12])  # from a manual run
    np.testing.assert_equal(table.table.iloc[0, :], expected)


def test_MRCV_table_from_factors():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                         presidential_data.columns[6:11],
                              "believe_true", orientation="wide")
    columns_factor = ctab.Factor.from_array(presidential_data.iloc[:, 11:],
                                            presidential_data.columns[11:],
                                 "why_uncertain", orientation="wide")
    multiple_response_table = ctab.MultipleResponseTable([rows_factor, ],
                                                         [columns_factor])
    expected = np.array([44, 49, 15, 22, 12])  # from a manual run
    np.testing.assert_equal(multiple_response_table.table.iloc[0, :].values,
                            expected)


def test_Factor_from_wide_data():
    single_response_data = presidential_data.iloc[:, :6]
    single_response_factor = ctab.Factor.from_array(single_response_data,
                                                    range(0, 6), "")
    narrow_dataframe = single_response_factor.cast_wide_to_narrow().data
    # actually selected options
    narrow_dataframe = narrow_dataframe[narrow_dataframe.value == 1]
    columns = ['observation_id', 'factor_level', 'value']
    top_row = narrow_dataframe[columns].iloc[0]
    expected = [0, 4, 1]  # from a manual run
    np.testing.assert_array_equal(top_row, expected)


def test_Factor_wide_to_narrow():
    # had a bug with index names
    n = 100
    car_choice = build_random_single_select(n)
    srcv = ctab.Factor(car_choice, "car_choice", orientation="wide")
    wide = srcv.cast_wide_to_narrow()
    np.testing.assert_equal(wide.data.shape, (300, 3))


def test_Factor_from_narrow_data():
    rows_factor = ctab.Factor.from_array(presidential_data.iloc[:, 6:11],
                                         presidential_data.columns[6:11],
                              "believe_true", orientation="wide")
    narrow_factor = rows_factor.cast_wide_to_narrow()
    wide_factor = narrow_factor.cast_narrow_to_wide()
    # pivoting the dataframe sorts the columns lexographically
    # (because the original column order is not preserved when the
    # dataframe is cast to narrow) so to compare, we need to sort
    # the columns of the original dataframe
    rows_dataframe = rows_factor.data.sort_index(axis=1)
    matches = rows_dataframe == wide_factor.data
    np.testing.assert_(matches.all().all())


def test_Factor_autodetect_multiple_response():
    single_response_data = presidential_data.iloc[:, :6]
    fake_labels_srcv = list(range(0, 6))
    fake_labels_mrcv = list(range(0, 5))
    build = ctab.Factor.from_array
    single_response_factor = build(single_response_data,
                                   fake_labels_srcv, "")
    np.testing.assert_(not single_response_factor.multiple_response)
    multiple_response_data = presidential_data.iloc[:, 6:11]
    multiple_response_factor = build(multiple_response_data,
                                     fake_labels_mrcv, "")
    np.testing.assert_(multiple_response_factor.multiple_response)


def test_Factor_columns_must_have_labels():
    single_response_data = presidential_data.iloc[:, :6]
    with assert_raises(ValueError):
        build = ctab.Factor.from_array
        single_response_factor = build(single_response_data, [], "")


def test_MRCV_table_with_ones():
    a = np.ones((1000, 2))
    b = np.ones((1000, 2))
    labels = ["Yes", "No"]
    mrcv_1 = ctab.Factor.from_array(a, labels, "alive",
                                    orientation="wide",
                                    multiple_response=True)
    mrcv_2 = ctab.Factor.from_array(b, labels, "cool",
                                    orientation="wide",
                                    multiple_response=True)
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    results = multiple_response_table.test_for_independence()
    np.testing.assert_(np.all(np.isnan(results.p_values_cellwise)))


def test_MRCV_table_with_zeros():
    a = np.zeros((1000, 2))
    b = np.zeros((1000, 2))
    labels = ["Yes", "No"]
    mrcv_1 = ctab.Factor.from_array(a, labels, "alive",
                                    orientation="wide",
                                    multiple_response=True)
    mrcv_2 = ctab.Factor.from_array(b, labels, "cool",
                                    orientation="wide",
                                    multiple_response=True)
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    results = multiple_response_table.test_for_independence()
    np.testing.assert_(np.all(np.isnan(results.p_values_cellwise)))


def test_MMI_table_with_no_variance():
    # if the single response factor has every observation on the
    # same level, decline to calculate
    a = np.zeros((1000, 1))
    b = np.ones((1000, 1))
    food_choices = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                                columns=["eggs", "cheese",
                                         "candy", "sushi", "none"])
    labels = ["Yes", "No"]
    ab = np.concatenate((a, b), axis=1)
    srcv = ctab.Factor.from_array(ab, labels, "alive",
                                  orientation="wide",
                                  multiple_response=False)
    mrcv_2 = ctab.Factor(food_choices, "cool", orientation="wide",
                         multiple_response=True)
    multiple_response_table = ctab.MultipleResponseTable([srcv, ],
                                                         [mrcv_2, ])
    results = multiple_response_table.test_for_independence()
    np.testing.assert_(np.all(np.isnan(results.p_values_cellwise)))


def test_SPMI_table_with_no_variance():
    # if a factor has every observation on the
    # same level, decline to calculate
    a = np.zeros((1000, 1))
    b = np.ones((1000, 1))
    food_choices = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                                columns=["eggs", "cheese",
                                         "candy", "sushi", "none"])
    labels = ["Yes", "No"]
    ab = np.concatenate((a, b), axis=1)
    mrcv_1 = ctab.Factor.from_array(ab, labels, "alive",
                                  orientation="wide",
                                  multiple_response=True)
    mrcv_2 = ctab.Factor(food_choices, "cool", orientation="wide",
                         multiple_response=True)
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    results = multiple_response_table.test_for_independence()
    np.testing.assert_(np.all(np.isnan(results.p_values_cellwise)))


def test_MRCV_2x2_table():
    # hit a bug with 2x2 tables not working
    np.random.seed(100)
    a = pd.DataFrame(np.random.randint(2, size=(1000, 2)),
                             columns=["good", "bad"])
    b = pd.DataFrame(np.random.randint(2, size=(1000, 2)),
                             columns=["eggs", "cheese"])
    mrcv_1 = ctab.Factor(a, "alive", orientation="wide",
                         multiple_response=True)
    mrcv_2 = ctab.Factor(b, "cool", orientation="wide",
                         multiple_response=True)
    multiple_response_table = ctab.MultipleResponseTable([mrcv_1, ],
                                                         [mrcv_2, ])
    results = multiple_response_table.test_for_independence()
    np.testing.assert_(results.p_value_overall > 0.05)


def test_for_MRCV_independence():
    # test all the different combinations the top-level
    #  test_for_independence should be
    # able to dispatch automatically
    food_choices = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                                columns=["eggs", "cheese",
                                         "candy", "sushi", "none"])
    language = pd.DataFrame(np.random.randint(2, size=(10000, 5)),
                            columns=["English", "French",
                                     "Mandarin", "Hungarian", "none"])
    car_choice = build_random_single_select()
    second_car_choice = build_random_single_select()
    srcv_1 = ctab.Factor(car_choice, "", orientation="wide")
    srcv_2 = ctab.Factor(second_car_choice, "", orientation="wide")
    mrcv_1 = ctab.Factor(food_choices, "", orientation="wide",
                         multiple_response=True)
    mrcv_2 = ctab.Factor(language, "", orientation="wide",
                         multiple_response=True)

    table = ctab.MultipleResponseTable([srcv_1, ], [mrcv_1, ])
    results = table.test_for_independence()
    np.testing.assert_equal(results.independence_type,
                            'Marginal Mutual Independence')
    np.testing.assert_equal(results.method, 'Bonferroni')

    table = ctab.MultipleResponseTable([srcv_1, ], [mrcv_1, ])
    results = table.test_for_independence(method="bon")
    np.testing.assert_equal(results.independence_type,
                            'Marginal Mutual Independence')
    np.testing.assert_equal(results.method, 'Bonferroni')

    table = ctab.MultipleResponseTable([mrcv_1, ], [srcv_1, ])
    results = table.test_for_independence(method="bon")
    np.testing.assert_equal(results.independence_type,
                            'Marginal Mutual Independence')
    np.testing.assert_equal(results.method, 'Bonferroni')

    table = ctab.MultipleResponseTable([srcv_1, ], [mrcv_1, ])
    results = table.test_for_independence(method="rao")
    np.testing.assert_equal(results.independence_type,
                            'Marginal Mutual Independence')
    np.testing.assert_equal(results.method, 'Rao-Scott')

    table = ctab.MultipleResponseTable([mrcv_1, ], [srcv_1, ])
    results = table.test_for_independence(method="rao")
    np.testing.assert_equal(results.independence_type,
                            'Marginal Mutual Independence')
    np.testing.assert_equal(results.method, 'Rao-Scott')

    table = ctab.MultipleResponseTable([srcv_1, ], [srcv_2, ])
    results = table.test_for_independence()
    np.testing.assert_equal(type(results),
                            ctab.ContingencyTableNominalIndependenceResult)

    table = ctab.MultipleResponseTable([mrcv_2, ], [mrcv_1, ])
    results = table.test_for_independence()
    expected = 'Simultaneous Pairwise Mutual Independence'
    np.testing.assert_equal(results.independence_type, expected)
    np.testing.assert_equal(results.method, 'Bonferroni')

    table = ctab.MultipleResponseTable([mrcv_1, ], [mrcv_2, ])
    results = table.test_for_independence(method="bon")
    np.testing.assert_equal(results.independence_type, expected)
    np.testing.assert_equal(results.method, 'Bonferroni')

    table = ctab.MultipleResponseTable([mrcv_1, ], [mrcv_2, ])
    results = table.test_for_independence(method="rao")
    np.testing.assert_equal(results.independence_type, expected)
    np.testing.assert_equal(results.method, 'Rao-Scott')

    # test accepting narrows
    food_choices = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                                columns=["eggs", "cheese",
                                         "candy", "sushi", "none"])
    language = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                                           columns=["English", "French",
                                                    "Mandarin",
                                                    "Hungarian", "none"])
    mrcv_1 = ctab.Factor(food_choices, "car_choice", orientation="wide")
    mrcv_2 = ctab.Factor(language, "language", orientation="wide")
    narrow_mrcv_1 = mrcv_1.cast_wide_to_narrow()
    narrow_mrcv_2 = mrcv_2.cast_wide_to_narrow()
    multiple_response_table = ctab.MultipleResponseTable([narrow_mrcv_1, ],
                                                         [narrow_mrcv_2, ])
    result = multiple_response_table.test_for_independence(method="rao")
    np.testing.assert_(result.p_value_overall >= 0.05)
    result = multiple_response_table.test_for_independence(method="bon")
    np.testing.assert_(result.p_value_overall >= 0.05)

def test_combining_factors():
    food_choices = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                                columns=["eggs", "cheese",
                                         "candy", "sushi", "none"])
    language = pd.DataFrame(np.random.randint(2, size=(1000, 5)),
                            columns=["English", "French",
                                     "Mandarin", "Hungarian", "none"])
    car_choice = build_random_single_select(n=1000)
    second_car_choice = build_random_single_select(n=1000)
    srcv_1 = ctab.Factor(car_choice, "", orientation="wide")
    srcv_2 = ctab.Factor(second_car_choice, "", orientation="wide")
    mrcv_1 = ctab.Factor(food_choices, "", orientation="wide",
                         multiple_response=True)
    mrcv_2 = ctab.Factor(language, "", orientation="wide",
                         multiple_response=True)

    srcv_srcv = srcv_1.combine_with(srcv_2)

    np.testing.assert_(srcv_srcv.labels[0] ==
                       "('motorcycle', 'motorcycle')")
    np.testing.assert_(srcv_srcv.labels[-1] == "('truck', 'truck')")
    np.testing.assert_(srcv_srcv.data.shape == (1000, 9))

    srcv_mrcv = srcv_1.combine_with(mrcv_1)
    np.testing.assert_(srcv_mrcv.labels[0] == "('motorcycle', 'candy')")
    np.testing.assert_(srcv_mrcv.labels[-1] == "('truck', 'sushi')")
    np.testing.assert_(srcv_mrcv.data.shape == (1000, 15))

    mrcv_mrcv = mrcv_1.combine_with(mrcv_2)
    np.testing.assert_(mrcv_mrcv.labels[0] == "('candy', 'English')")
    np.testing.assert_(mrcv_mrcv.labels[-1] == "('sushi', 'none')")
    np.testing.assert_(mrcv_mrcv.data.shape == (1000, 25))

    narrow = mrcv_2.cast_wide_to_narrow()
    wide_narrow = mrcv_1.combine_with(narrow)
    np.testing.assert_(wide_narrow.labels[0] == "('candy', 'English')")
    np.testing.assert_(wide_narrow.labels[-1] == "('sushi', 'none')")
    np.testing.assert_(wide_narrow.data.shape == (1000, 25))

    narrow_wide = narrow.combine_with(mrcv_1)
    np.testing.assert_(narrow_wide.labels[0] == "('English', 'candy')")
    np.testing.assert_(narrow_wide.labels[-1] == "('none', 'sushi')")
    np.testing.assert_(narrow_wide.data.shape == (1000, 25))

    narrow_2 = mrcv_2.cast_wide_to_narrow()
    narrow_narrow = narrow.combine_with(narrow_2)
    np.testing.assert_(narrow_narrow.labels[0] ==
                       "('English', 'English')")
    np.testing.assert_(narrow_narrow.labels[-1] == "('none', 'none')")
    np.testing.assert_(narrow_narrow.data.shape == (1000, 25))


def test_creating_narrow_factor_from_data():
    car_choice = build_random_single_select(n=1000)
    index_name = car_choice.index.name
    melted = pd.melt(car_choice.reset_index(), id_vars=index_name)
    melted = melted.rename(columns={index_name: "observation_id"})
    narrowed = melted.sort_values("observation_id")
    narrow_data = narrowed.reset_index(drop=True)
    with assert_raises(NotImplementedError):
        ctab.Factor(narrow_data, "", orientation="narrow")
    narrow_data.columns = ['observation_id', 'factor_level', 'value']
    srcv = ctab.Factor(narrow_data, "", orientation="narrow")
    np.testing.assert_equal(srcv.data.shape, (3000, 3))
    levels = srcv.data['factor_level'].unique()
    expected = np.array(['sedan', 'truck', 'motorcycle'], dtype=object)
    np.testing.assert_equal(levels, expected)

if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)