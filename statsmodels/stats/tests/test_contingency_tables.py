"""
Tests for contingency table analyses.
"""
from statsmodels.compat.python import PYTHON_IMPL_WASM

from contextlib import nullcontext
from math import erfc, sqrt
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

import statsmodels.api as sm
import statsmodels.stats.contingency_tables as ctab

cur_dir = Path(__file__).resolve().parent
fname = "contingency_table_r_results.csv"
fpath = Path(cur_dir).joinpath("results", fname)
r_results = pd.read_csv(fpath)


tables = [None, None, None]

tables[0] = np.asarray([[23, 15], [19, 31]])

tables[1] = np.asarray(
    [[144, 33, 84, 126], [2, 4, 14, 29], [0, 2, 6, 25], [0, 0, 1, 5]]
)

tables[2] = np.asarray([[20, 10, 5], [3, 30, 15], [0, 5, 40]])


def test_homogeneity():

    for k, table in enumerate(tables):
        st = sm.stats.SquareTable(table, shift_zeros=False)
        hm = st.homogeneity()
        assert_allclose(hm.statistic, r_results.loc[k, "homog_stat"])
        assert_allclose(hm.df, r_results.loc[k, "homog_df"])

        # Test Bhapkar via its relationship to Stuart_Maxwell.
        hmb = st.homogeneity(method="bhapkar")
        assert_allclose(hmb.statistic, hm.statistic / (1 - hm.statistic / table.sum()))


def test_SquareTable_from_data():

    rs = np.random.RandomState(434)
    df = pd.DataFrame(index=range(100), columns=["v1", "v2"])
    df["v1"] = rs.randint(0, 5, 100)
    df["v2"] = rs.randint(0, 5, 100)
    table = pd.crosstab(df["v1"], df["v2"])

    rslt1 = ctab.SquareTable(table)
    rslt2 = ctab.SquareTable.from_data(df)
    rslt3 = ctab.SquareTable(np.asarray(table))

    assert_equal(rslt1.summary().as_text(), rslt2.summary().as_text())

    assert_equal(rslt2.summary().as_text(), rslt3.summary().as_text())

    s = str(rslt1)
    assert_equal(s.startswith("A 5x5 contingency table with counts:"), True)
    assert_equal(rslt1.table[0, 0], 8.0)


def test_SquareTable_nonsquare():

    tab = [[1, 0, 3], [2, 1, 4], [3, 0, 5]]
    df = pd.DataFrame(tab, index=[0, 1, 3], columns=[0, 2, 3])

    df2 = ctab.SquareTable(df, shift_zeros=False)

    e = np.asarray(
        [[1, 0, 0, 3], [2, 0, 1, 4], [0, 0, 0, 0], [3, 0, 0, 5]], dtype=np.float64
    )

    assert_equal(e, df2.table)


def test_cumulative_odds():

    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)

    cum_odds = tbl_obj.cumulative_oddsratios
    assert_allclose(cum_odds[0, 0], 28 / float(5 * 11))
    assert_allclose(cum_odds[0, 1], (3 * 15) / float(3 * 24), atol=1e-5, rtol=1e-5)
    assert_allclose(
        np.log(cum_odds), tbl_obj.cumulative_log_oddsratios, atol=1e-5, rtol=1e-5
    )


def test_local_odds():

    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)

    loc_odds = tbl_obj.local_oddsratios
    assert_allclose(loc_odds[0, 0], 5 / 8.0)
    assert_allclose(loc_odds[0, 1], 12 / float(15), atol=1e-5, rtol=1e-5)
    assert_allclose(
        np.log(loc_odds), tbl_obj.local_log_oddsratios, atol=1e-5, rtol=1e-5
    )


def test_shifting():

    t = np.zeros((3, 4), dtype=np.float64)
    result = np.full((3, 4), 0.5)
    assert_equal(ctab.Table(t, shift_zeros=False).table, t)
    assert_equal(ctab.Table(t, shift_zeros=True).table, result)

    t = np.asarray([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=np.float64)
    r = np.asarray([[0.5, 1, 2], [3, 0.5, 4], [5, 6, 0.5]], dtype=np.float64)
    assert_equal(ctab.Table(t).table, r)
    assert_equal(ctab.Table(t, shift_zeros=True).table, r)


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
    fit = [[22.083, 17.583, 32.536, 19.798], [296.92, 236.42, 437.46, 266.2]]
    c2 = [[4.6037, 5.223, 0.0725, 22.704], [0.3424, 0.3885, 0.0054, 1.6886]]

    # These are regression tests
    pr = np.array(
        [
            [-2.14562121, -2.28538719, -0.26923882, 4.76491690],
            [+0.58514314, 0.62325942, 0.07342547, -1.29946443],
        ]
    )
    sr = np.array(
        [
            [-2.55112945, -2.6338782, -0.34712127, 5.57510830],
            [+2.55112945, 2.6338782, 0.34712127, -5.57510830],
        ]
    )

    tab = ctab.Table(table)
    assert_allclose(tab.fittedvalues, fit, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.chi2_contribs, c2, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.resid_pearson, pr, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.standardized_resids, sr, atol=1e-4, rtol=1e-4)


def test_ordinal_association():

    for k, table in enumerate(tables):

        row_scores = 1 + np.arange(table.shape[0])
        col_scores = 1 + np.arange(table.shape[1])

        # First set of scores
        rslt = ctab.Table(table, shift_zeros=False).test_ordinal_association(
            row_scores, col_scores
        )
        assert_allclose(rslt.statistic, r_results.loc[k, "lbl_stat"])
        assert_allclose(rslt.null_mean, r_results.loc[k, "lbl_expval"])
        assert_allclose(rslt.null_sd**2, r_results.loc[k, "lbl_var"])
        assert_allclose(
            rslt.zscore**2, r_results.loc[k, "lbl_chi2"], rtol=1e-5, atol=1e-5
        )
        assert_allclose(
            rslt.pvalue, r_results.loc[k, "lbl_pvalue"], rtol=1e-5, atol=1e-5
        )

        # Second set of scores
        rslt = ctab.Table(table, shift_zeros=False).test_ordinal_association(
            row_scores, col_scores**2
        )
        assert_allclose(rslt.statistic, r_results.loc[k, "lbl2_stat"])
        assert_allclose(rslt.null_mean, r_results.loc[k, "lbl2_expval"])
        assert_allclose(rslt.null_sd**2, r_results.loc[k, "lbl2_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl2_chi2"])
        assert_allclose(
            rslt.pvalue, r_results.loc[k, "lbl2_pvalue"], rtol=1e-5, atol=1e-5
        )


def test_chi2_association():

    rs = np.random.RandomState(8743)

    table = rs.randint(10, 30, size=(4, 4))

    from scipy.stats import chi2_contingency

    rslt_scipy = chi2_contingency(table)

    b = ctab.Table(table).test_nominal_association()

    assert_allclose(b.statistic, rslt_scipy[0])
    assert_allclose(b.pvalue, rslt_scipy[1])


def test_symmetry():

    for k, table in enumerate(tables):
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


def test_mcnemar_non_2x2():
    # GH#9485: mcnemar only uses the [0, 1] and [1, 0] cells, so a table
    # that is not 2x2 must raise instead of silently ignoring the rest.
    table = np.asarray([[10, 5, 1], [3, 12, 2], [4, 6, 20]])
    with pytest.raises(ValueError, match="2x2"):
        ctab.mcnemar(table)
    with pytest.raises(ValueError, match="2x2"):
        ctab.mcnemar(table, exact=False)

    table_3d = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="two-dimensional"):
        ctab.mcnemar(table_3d)

    # A genuine 2x2 table must still work.
    b = ctab.mcnemar(tables[0], exact=False, correction=False)
    assert np.isfinite(b.statistic)


def test_from_data_stratified():

    df = pd.DataFrame(
        [[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]]
    ).T
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

    table = [
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
    ]
    table = np.asarray(table)

    stat, pvalue, df = ctab.cochrans_q(table)
    assert_allclose(stat, 4.2)
    assert_allclose(df, 3)

    table = [
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    table = np.asarray(table)

    stat, pvalue, df = ctab.cochrans_q(table)
    assert_allclose(stat, 1.2174, rtol=1e-4)
    assert_allclose(df, 4)

    # Cochran's q and Mcnemar are equivalent for 2x2 tables
    data = table[:, 0:2]
    xtab = np.asarray(pd.crosstab(data[:, 0], data[:, 1]))
    b1 = ctab.cochrans_q(data)
    b2 = ctab.mcnemar(xtab, exact=False, correction=False)
    assert_allclose(b1.statistic, b2.statistic)
    assert_allclose(b1.pvalue, b2.pvalue)

    # The single result supports both attribute access and unpacking
    assert isinstance(b1, ctab.CochransQResult)
    assert_equal(len(b1), 3)
    assert_allclose(tuple(b1), (b1.statistic, b1.pvalue, b1.df))
    assert_equal(b1.df, 1)


def test_cochrans_q_return_object_deprecated():
    # return_object no longer changes the result, but still warns
    rs = np.random.RandomState(12345)
    data = rs.randint(0, 2, size=(30, 4))

    expected = ctab.cochrans_q(data)
    for value in (True, False):
        with pytest.warns(FutureWarning, match="return_object"):
            res = ctab.cochrans_q(data, return_object=value)
        assert_equal(res, expected)


class CheckStratifiedMixin:

    @classmethod
    def initialize(cls, tables, use_arr=False):
        tables1 = tables if not use_arr else np.dstack(tables)
        cls.rslt = ctab.StratifiedTable(tables1)
        cls.rslt_0 = ctab.StratifiedTable(tables, shift_zeros=True)
        tables_pandas = [pd.DataFrame(x) for x in tables]
        cls.rslt_pandas = ctab.StratifiedTable(tables_pandas)

    def test_oddsratio_pooled(self):
        assert_allclose(
            self.rslt.oddsratio_pooled, self.oddsratio_pooled, rtol=1e-4, atol=1e-4
        )

    def test_logodds_pooled(self):
        assert_allclose(
            self.rslt.logodds_pooled, self.logodds_pooled, rtol=1e-4, atol=1e-4
        )

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
        assert_allclose(lcb, np.log(self.or_lcb), rtol=1e-4, atol=1e-4)
        assert_allclose(ucb, np.log(self.or_ucb), rtol=1e-4, atol=1e-4)

    def test_equal_odds(self):

        if not hasattr(self, "or_homog"):
            return

        rslt = self.rslt.test_equal_odds(adjust=False)
        assert_allclose(rslt.statistic, self.or_homog, rtol=1e-4, atol=1e-4)
        assert_allclose(rslt.pvalue, self.or_homog_p, rtol=1e-4, atol=1e-4)

        rslt = self.rslt.test_equal_odds(adjust=True)
        assert_allclose(rslt.statistic, self.or_homog_adj, rtol=1e-4, atol=1e-4)
        assert_allclose(rslt.pvalue, self.or_homog_adj_p, rtol=1e-4, atol=1e-4)

    def test_pandas(self):
        self.rslt_pandas.summary().as_text()
        self.rslt.summary().as_text()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_equal(
                self.rslt.summary().as_text(), self.rslt_pandas.summary().as_text()
            )

    def test_from_data(self):

        rs = np.random.RandomState(241)
        df = pd.DataFrame(index=range(100), columns=("v1", "v2", "strat"))
        df["v1"] = rs.randint(0, 2, 100)
        df["v2"] = rs.randint(0, 2, 100)
        df["strat"] = np.kron(np.arange(10), np.ones(10))

        tables = []
        for k in range(10):
            ii = np.arange(10 * k, 10 * (k + 1))
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
    library(DescTools)
    data = array(c(20, 14, 10, 24,
                   15, 12, 3, 15,
                   3, 2, 3, 2,
                   12, 3, 7, 5,
                   1, 0, 3, 2),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[20, 14], [10, 24]])
        tables[1] = np.array([[15, 12], [3, 15]])
        tables[2] = np.array([[3, 2], [3, 2]])
        tables[3] = np.array([[12, 3], [7, 5]])
        tables[4] = np.array([[1, 0], [3, 2]])

        # check array of int
        cls.initialize(tables, use_arr=True)

        cls.oddsratio_pooled = 3.5912
        cls.logodds_pooled = np.log(3.5912)

        cls.mh_stat = 11.8852
        cls.mh_pvalue = 0.0005658

        cls.or_lcb = 1.781135
        cls.or_ucb = 7.240633

        # Breslow Day test without Tarone adjustment
        cls.or_homog = 1.8438
        cls.or_homog_p = 0.7645

        # Breslow Day test with Tarone adjustment
        cls.or_homog_adj = 1.8436
        cls.or_homog_adj_p = 0.7645


class TestStratified3(CheckStratifiedMixin):
    """
    library(DescTools)
    data = array(c(313, 512, 19, 89,
                   207, 353, 8, 17,
                   205, 120, 391, 202,
                   278, 139, 244, 131,
                   138, 53, 299, 94,
                   351, 22, 317, 24),
                   dim=c(2, 2, 6))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
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

        # Breslow Day test without Tarone adjustment
        cls.or_homog = 18.83297
        cls.or_homog_p = 0.002064786

        # Breslow Day test with Tarone adjustment
        cls.or_homog_adj = 18.83297
        cls.or_homog_adj_p = 0.002064786


# R reference for the tests with a pooled (Mantel-Haenszel) odds ratio of 1.
# DescTools 0.99.60 (R 4.6.1), a, b, c, d being the cells of each 2x2 table
# [[a, b], [c, d]] and K the number of tables:
#     x <- array(0, c(2, 2, K))
#     for (k in 1:K) x[, , k] <- matrix(c(a[k], c[k], b[k], d[k]), 2, 2)
#     BreslowDayTest(x, correct=FALSE); BreslowDayTest(x, correct=TRUE)
# mantelhaen.test(x)$estimate is exactly 1 in all cases and the statistics
# with and without correct (Tarone) are identical:
#     (10, 20, 20, 10), (20, 10, 10, 20):
#         13.333333333333334, p 0.000260729632855283
#     (20, 10, 5, 15), (10, 20, 15, 5):
#         16.666666666666668, p 4.45570906040249e-05
#     (20, 10, 5, 15), (10, 20, 15, 5), (30, 12, 8, 25), (12, 30, 25, 8):
#         49.589575905365379, p 9.7701846613063e-11 (3 df)
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("as_array", [False, True])
def test_equal_odds_unit_pooled_or(adjust, as_array):
    # With common OR=1, all fitted counts are 15. The Breslow-Day
    # asymptotic variance is 1 / (4 / 15), so chi2 = 2 * 25 / (15 / 4).
    # The Tarone correction is zero because the deviations sum to zero.
    tables = [[[10, 20], [20, 10]], [[20, 10], [10, 20]]]
    if as_array:
        tables = np.dstack(tables)
    result = ctab.StratifiedTable(tables).test_equal_odds(adjust=adjust)
    assert_allclose(result.statistic, 40 / 3, rtol=1e-13)
    # For one degree of freedom, chi2 survival is erfc(sqrt(x / 2)).
    assert_allclose(result.pvalue, erfc(sqrt(20 / 3)), rtol=1e-13)


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize(
    "tables, expected, pvalue",
    [
        (
            [[[20, 10], [5, 15]], [[10, 20], [15, 5]]],
            16.666666666666668,
            4.45570906040249e-05,
        ),
        (
            [
                [[20, 10], [5, 15]],
                [[10, 20], [15, 5]],
                [[30, 12], [8, 25]],
                [[12, 30], [25, 8]],
            ],
            49.589575905365379,
            9.7701846613063e-11,
        ),
    ],
)
def test_equal_odds_unit_pooled_or_r(adjust, tables, expected, pvalue):
    # DescTools::BreslowDayTest, see the R reference above
    result = ctab.StratifiedTable(tables).test_equal_odds(adjust=adjust)
    assert_allclose(result.statistic, expected, rtol=1e-13)
    # R computes the p-value as 1 - pchisq, which loses digits for small values
    assert_allclose(result.pvalue, pvalue, rtol=1e-6)


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("delta", [-1, 1])
def test_equal_odds_near_unit_pooled_or(adjust, delta):
    # Identical strata have fitted counts equal to the observed counts,
    # giving statistic=0 and pvalue=1, even with an OR very close to 1.
    n = 10**8
    table = [[n, n], [n, n + delta]]
    result = ctab.StratifiedTable([table, table]).test_equal_odds(adjust=adjust)
    assert_allclose(result.statistic, 0, atol=1e-18)
    assert_allclose(result.pvalue, 1, rtol=1e-10)


@pytest.mark.parametrize("adjust", [False, True])
def test_equal_odds_mixed_margins(adjust):
    # Independent 80-digit Decimal bisection of the monotone cell-odds
    # equation e * (N-R-C+e) / ((R-e)*(C-e)) = OR_MH gives these values.
    # This case exercises both signs of the quadratic's linear term.
    tables = [[[100, 100], [100, 1]], [[1, 40], [50, 100]]]
    expected = 1.3064617383935995 if adjust else 1.3503676201849462
    result = ctab.StratifiedTable(tables).test_equal_odds(adjust=adjust)
    assert_allclose(result.statistic, expected, rtol=1e-12)
    assert_allclose(result.pvalue, erfc(sqrt(expected / 2)), rtol=1e-12)


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("transform", range(8))
def test_equal_odds_small_cells(adjust, transform):
    # Independent 120-digit Decimal monotone cell-odds bisection.
    # Tiny fitted off-diagonal cells must not be obtained by subtracting
    # two approximately 1e14 diagonal counts. Symmetries preserve Q.
    tables = np.dstack([
        [[17592186044416, 4], [3, 35184372088832]],
        [[35184372088832, 7], [5, 105553116266496]],
    ])
    if transform & 1:
        tables = tables[::-1, :, :]
    if transform & 2:
        tables = tables[:, ::-1, :]
    if transform & 4:
        tables = tables.transpose(1, 0, 2)[:, :, ::-1]
    expected = 0.5555352755861875 if adjust else 0.5647268822325966
    result = ctab.StratifiedTable(tables).test_equal_odds(adjust=adjust)
    assert_allclose(result.statistic, expected, rtol=2e-13, atol=0)
    assert_allclose(result.pvalue, erfc(sqrt(expected / 2)), rtol=2e-13)
    assert isinstance(result.statistic, np.float64)
    assert isinstance(result.pvalue, np.float64)
    assert set(vars(result)) == {"statistic", "pvalue"}


@pytest.mark.parametrize("adjust", [False, True])
def test_equal_odds_near_homogeneous_absolute_tolerance(adjust):
    # Counts near 1e15 limit recoverable residual precision. The statistic is
    # of order 1e-17 with a large relative error, so this only checks that it
    # is finite and tiny, not that it is accurate.
    tables = [[[3 * k * 10**14, 6 * k * 10**14],
               [k * 10**14, 2 * k * 10**14]]
              for k in [1, 2, 3] * 3]
    tables[-1][0][1] += 1
    result = ctab.StratifiedTable(tables).test_equal_odds(adjust=adjust)
    assert np.isfinite(result.statistic)
    assert 0 <= result.statistic < 1e-15
    assert_allclose(result.pvalue, 1, atol=1e-14, rtol=0)


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("shift_zeros", [False, True])
def test_equal_odds_observed_zeros(adjust, shift_zeros):
    # Positive margins and finite positive common OR, even without shifting.
    # Independent 120-digit monotone cell-odds bisection, before/after shift.
    tables = [[[0, 7], [5, 20]], [[11, 3], [9, 4]], [[6, 8], [5, 7]]]
    expected = {False: {False: 2.000284881564185, True: 2.000214395536559}, True: {False: 1.251899415048697, True: 1.2518903428029755}}[shift_zeros][adjust]
    st = ctab.StratifiedTable(tables, shift_zeros=shift_zeros)
    result = st.test_equal_odds(adjust=adjust)
    assert_allclose(result.statistic, expected, rtol=2e-13)
    # Chi-square(2) survival is exp(-Q/2), independently of scipy's path.
    assert_allclose(result.pvalue, np.exp(-expected / 2), rtol=2e-13)
    shifted = np.dstack(tables).astype(float)
    if shift_zeros:
        shifted[:, :, 0] += 0.5
    assert_equal(st.table, shifted)


class Check2x2Mixin:
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
        assert_equal(
            self.tbl_obj.summary().as_text(), self.tbl_data_obj.summary().as_text()
        )

    def test_summary(self):

        assert_equal(self.tbl_obj.summary().as_text(), self.summary_string)


class Test2x2_1(Check2x2Mixin):

    @classmethod
    def setup_class(cls):
        data = np.zeros((8, 2))
        data[:, 0] = [0, 0, 1, 1, 0, 0, 1, 1]
        data[:, 1] = [0, 1, 0, 1, 0, 1, 0, 1]
        cls.data = np.asarray(data)
        cls.table = np.asarray([[2, 2], [2, 2]])

        cls.oddsratio = 1.0
        cls.log_oddsratio = 0.0
        cls.log_oddsratio_se = np.sqrt(2)
        cls.oddsratio_confint = [0.062548836166112329, 15.987507702689751]
        cls.oddsratio_pvalue = 1.0
        cls.riskratio = 1.0
        cls.log_riskratio = 0.0
        cls.log_riskratio_se = 1 / np.sqrt(2)
        cls.riskratio_pvalue = 1.0
        cls.riskratio_confint = [0.25009765325990629, 3.9984381579173824]
        cls.log_riskratio_confint = [-1.3859038243496782, 1.3859038243496782]
        ss = [
            "               Estimate   SE   LCB    UCB   p-value",
            "---------------------------------------------------",
            "Odds ratio        1.000        0.063 15.988   1.000",
            "Log odds ratio    0.000 1.414 -2.772  2.772   1.000",
            "Risk ratio        1.000        0.250  3.998   1.000",
            "Log risk ratio    0.000 0.707 -1.386  1.386   1.000",
            "---------------------------------------------------",
        ]
        cls.summary_string = "\n".join(ss)
        cls.initialize()


@pytest.mark.parametrize(
    "table, shift_zeros",
    [
        ([[40, 1], [1, 40]], False),
        ([[30, 1], [1, 30]], False),
    ],
)
def test_nominal_association_small_pvalue(table, shift_zeros):
    """p-value must not be lost to cancellation in the far upper tail.

    GH#10274. The p-value was formed as ``1 - chi2.cdf(...)``. At a large
    statistic the CDF rounds to exactly 1.0 and the subtraction cancels the
    whole tail, returning 0.0 for a probability that is representable.
    """
    from scipy import stats

    rslt = ctab.Table(np.asarray(table, dtype=float),
                      shift_zeros=shift_zeros).test_nominal_association()

    expected = stats.chi2.sf(rslt.statistic, rslt.df)

    assert rslt.pvalue > 0
    assert_allclose(rslt.pvalue, expected, rtol=1e-12)


def test_homogeneity_small_pvalue():
    """Stuart-Maxwell p-value must survive the far upper tail. GH#10274."""
    from scipy import stats

    table = np.asarray(
        [[18, 36, 36], [1, 18, 36], [1, 1, 18]], dtype=float
    )
    rslt = ctab.SquareTable(table, shift_zeros=False).homogeneity()

    expected = stats.chi2.sf(rslt.statistic, rslt.df)

    assert rslt.pvalue > 0
    assert_allclose(rslt.pvalue, expected, rtol=1e-12)


def test_stratified_test_null_odds_small_pvalue():
    """StratifiedTable.test_null_odds p-value must not cancel. GH#10274."""
    from scipy import stats

    table = np.asarray(
        [[[60, 60], [1, 1]], [[1, 1], [60, 60]]], dtype=float
    )
    rslt = ctab.StratifiedTable(table).test_null_odds()

    expected = stats.chi2.sf(rslt.statistic, 1)

    assert rslt.pvalue > 0
    assert_allclose(rslt.pvalue, expected, rtol=1e-12)


def test_negative_counts_rejected():
    # negative counts used to flow through every statistic silently,
    # producing e.g. a negative odds ratio
    with pytest.raises(ValueError, match="non-negative"):
        ctab.Table2x2([[2, 2], [2, -1]])
    with pytest.raises(ValueError, match="non-negative"):
        ctab.SquareTable(np.asarray([[1, 2], [3, -0.5]]))
    with pytest.raises(ValueError, match="non-negative"):
        ctab.Table([[1, -2], [3, 4]])


def test_stratified_negative_counts_rejected():
    # companion to the base-Table check: StratifiedTable keeps its own
    # constructor, and negative counts used to flow through every statistic
    with pytest.raises(ValueError, match="non-negative"):
        ctab.StratifiedTable(np.asarray([[[2, 2], [2, -1]], [[3, 1], [1, 3]]], dtype=float))
    with pytest.raises(ValueError, match="non-negative"):
        ctab.StratifiedTable([[[2, 2], [2, 2]], [[3, 1], [1, -2]]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_counts_rejected(bad):
    # NaN is not negative and inf passed the check for negative counts, so that
    # every statistic was nan or inf without an error
    table = np.array([[2.0, 2.0], [2.0, bad]])
    for cls in (ctab.Table, ctab.SquareTable, ctab.Table2x2):
        with pytest.raises(ValueError, match="must be finite"):
            cls(table)
        with pytest.raises(ValueError, match="must be finite"):
            cls(table, shift_zeros=False)
    with pytest.raises(ValueError, match="must be finite"):
        ctab.Table(pd.DataFrame(table))
    with pytest.raises(ValueError, match="must be finite"):
        ctab.Table([[1, 2, 3], [4, bad, 6]])
    with pytest.raises(ValueError, match="must be finite"):
        ctab.StratifiedTable([table, [[3, 1], [1, 3]]])
    with pytest.raises(ValueError, match="must be finite"):
        ctab.StratifiedTable(np.stack([table, [[3.0, 1], [1, 3]]], axis=2))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("exact", [True, False])
def test_mcnemar_nonfinite_counts_rejected(bad, position, exact):
    # for exact=True this was an obscure error of int() for NaN and inf, and
    # for a diagonal cell the invalid count was not used
    table = np.array([[5.0, 3.0], [2.0, 6.0]])
    table[position] = bad
    with pytest.raises(ValueError, match="must be finite"):
        ctab.mcnemar(table, exact=exact)
    with pytest.raises(ValueError, match="must be finite"):
        ctab.mcnemar(pd.DataFrame(table), exact=exact)


@pytest.mark.parametrize("exact", [True, False])
def test_mcnemar_negative_counts_rejected(exact):
    # the counts were used without a check, the statistic for a negative
    # off-diagonal count is meaningless
    for table in ([[5, -1], [2, 6]], [[5, 1], [-2, 6]], [[-5, 1], [2, 6]]):
        with pytest.raises(ValueError, match="non-negative"):
            ctab.mcnemar(np.asarray(table), exact=exact)


def test_table2x2_confint_invalid_alpha_raises():
    # alpha outside (0, 1) previously returned (inf, 0.0) silently
    t2 = ctab.Table2x2(np.array([[10.0, 5.0], [8.0, 12.0]]))
    for confint in (
        t2.oddsratio_confint,
        t2.log_oddsratio_confint,
        t2.riskratio_confint,
        t2.log_riskratio_confint,
    ):
        with pytest.raises(ValueError, match="alpha must be in the range"):
            confint(alpha=2)
        with pytest.raises(ValueError, match="alpha must be in the range"):
            confint(alpha=0)
    lo, hi = t2.oddsratio_confint()
    assert np.isfinite([lo, hi]).all()
    assert 0 < lo < hi


@pytest.mark.parametrize("adjust", [False, True])
def test_equal_odds_reciprocal_overflow_legacy(adjust):
    # This unsupported scaling must retain the old NaN/warning behavior,
    # rather than introducing an OverflowError from accurate accumulation.
    tables = [[[1e-308, 1e-308], [1, 1]], [[1e-308, 1e-308], [1, 1]]]
    # WASM has no floating-point exception support; keep the numerical checks.
    warning_context = (
        nullcontext() if PYTHON_IMPL_WASM else
        pytest.warns(RuntimeWarning, match="invalid value encountered in divide")
    )
    with warning_context:
        result = ctab.StratifiedTable(tables).test_equal_odds(adjust=adjust)
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)


@pytest.mark.parametrize("adjust", [False, True])
def test_equal_odds_overflow_preserves_legacy_warnings(adjust):
    warning_context = (
        nullcontext() if PYTHON_IMPL_WASM else pytest.warns(RuntimeWarning)
    )
    with warning_context:
        table = ctab.StratifiedTable(np.full((2, 2, 2), 1e308))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = table.test_equal_odds(adjust=adjust)
    expected = ["invalid value encountered in divide"] * 2
    if adjust:
        expected.append("overflow encountered in reduce")
    if not PYTHON_IMPL_WASM:
        assert [str(warning.message) for warning in caught] == expected
    assert all(warning.category is RuntimeWarning for warning in caught)
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)


@pytest.mark.parametrize("adjust", [False, True])
def test_breslow_day_empty_fallback(adjust):
    # The internal helper must leave empty inputs to the existing public
    # validation, not reshape/reduce them or divide by an empty variance sum.
    assert ctab._breslow_day_statistic(np.empty((2, 2, 0)), 1.0, adjust) is None


@pytest.mark.parametrize("adjust", [False, True])
def test_breslow_day_margin_overflow_fallback(adjust):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ctab._breslow_day_statistic(
            np.full((2, 2, 2), 1e308), 1.0, adjust
        )
    assert result is None


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("exponent", [510, 511])
def test_equal_odds_unrepresentable_fitted_legacy(adjust, exponent):
    # The true fitted counts equal the observations, but the discriminant
    # overflows. At exponent 510 the helper computes zero fitted cells;
    # at 511 its numerator also overflows, giving inf / inf. Both must
    # fall back silently rather than broaden the supported scaling range.
    table = ctab.StratifiedTable(np.full((2, 2, 2), 2.0**exponent))
    assert table.oddsratio_pooled == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ctab._breslow_day_statistic(
            table.table, table.oddsratio_pooled, adjust
        ) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = table.test_equal_odds(adjust=adjust)
    if exponent == 510:
        # Legacy e11=inf and v11=inf leave inf / inf in the statistic.
        expected = ["overflow encountered in square",
                    "divide by zero encountered in divide",
                    "invalid value encountered in divide"]
        if adjust:
            expected.append("invalid value encountered in scalar divide")
    else:
        # Legacy c=-inf, b**2=inf, and (1-OR)*c = 0*(-inf) is NaN.
        expected = ["overflow encountered in multiply",
                    "overflow encountered in square",
                    "invalid value encountered in multiply"]
    messages = [str(warning.message) for warning in caught]
    if not PYTHON_IMPL_WASM:
        if exponent == 510 and adjust:
            # NumPy 1.23 calls scalar division "double_scalars".
            messages[-1] = messages[-1].replace("double_scalars", "scalar divide")
        assert messages == expected
    assert all(warning.category is RuntimeWarning for warning in caught)
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)


@pytest.mark.parametrize("adjust", [False, True])
def test_equal_odds_unrepresentable_reciprocal_legacy(adjust):
    # With OR=1, the fitted first row is 2**-1030. Its reciprocal is
    # already infinite, so fsum returns inf (does not raise OverflowError)
    # and the variance becomes zero. Legacy 0 / 0 for e11 stays NaN.
    tables = [[[2.0**-1030, 2.0**-1030], [1, 1]]] * 2
    table = ctab.StratifiedTable(tables)
    assert table.oddsratio_pooled == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ctab._breslow_day_statistic(
            table.table, table.oddsratio_pooled, adjust
        ) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = table.test_equal_odds(adjust=adjust)
    if not PYTHON_IMPL_WASM:
        assert [str(warning.message) for warning in caught] == [
            "invalid value encountered in divide"
        ]
    assert all(warning.category is RuntimeWarning for warning in caught)
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)


@pytest.mark.parametrize("adjust", [False, True])
def test_equal_odds_statistic_sum_overflow_legacy(adjust):
    # Exact binary scales give pooled OR=2**-512. In each of the first
    # two strata, fitted a and variance round to 2**-1023 and delta to 1.
    # Each chi-square term is finite (2**1023), but their sum overflows
    # math.fsum, with or without Tarone centering. No arithmetic is mocked.
    scale = 2.0**511
    tables = [[[1, 0], [0, scale]]] * 2
    tables += [[[0, scale], [scale, 0]]] * 8
    table = ctab.StratifiedTable(tables)
    assert table.oddsratio_pooled == 2.0**-512
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ctab._breslow_day_statistic(
            table.table, table.oddsratio_pooled, adjust
        ) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = table.test_equal_odds(adjust=adjust)
    # The legacy root subtraction rounds e11 to zero in the first two
    # strata, making v11=0 and Q=inf. The Tarone correction is finite.
    if not PYTHON_IMPL_WASM:
        assert [str(warning.message) for warning in caught] == [
            "divide by zero encountered in divide"
        ] * 2
    assert all(warning.category is RuntimeWarning for warning in caught)
    assert_equal(result.statistic, np.inf)
    assert_equal(result.pvalue, 0.0)


def test_null_odds_correction_no_overshoot():
    # The continuity correction is only applied if |sum(a - E(a))| >= 0.5,
    # as in R's mantelhaen.test; otherwise it would enlarge the statistic.
    # Single rare event: sum(a - E(a)) = -0.01, R: X2 = 0.0101, p = 0.92
    st = ctab.StratifiedTable([np.array([[0, 1], [1, 98]])])
    r0 = st.test_null_odds(correction=False)
    r1 = st.test_null_odds(correction=True)
    assert_allclose(r0.statistic, 0.0001 / 0.0099, rtol=1e-10)
    assert_allclose(r1.statistic, r0.statistic, rtol=1e-12)
    assert r1.pvalue > 0.9
    # sum(a - E(a)) = 0 exactly: statistic is 0 with or without correction
    tables = [np.array([[3, 2], [2, 3]]), np.array([[2, 3], [3, 2]])]
    r1 = ctab.StratifiedTable(tables).test_null_odds(correction=True)
    assert_allclose(r1.statistic, 0, atol=1e-14)
    assert_allclose(r1.pvalue, 1, atol=1e-14)
    # |sum(a - E(a))| >= 0.5: correction applied as before
    tables = [np.array([[10, 5], [5, 10]]), np.array([[8, 4], [6, 9]])]
    st = ctab.StratifiedTable(tables)
    delta = np.sum(st.table[0, 0, :] - st._apb * st._apc / st._n)
    denom = np.sum(st._apb * st._apc * st._bpd * st._cpd / (st._n**2 * (st._n - 1)))
    assert_allclose(
        st.test_null_odds(correction=True).statistic,
        (abs(delta) - 0.5) ** 2 / denom,
        rtol=1e-12,
    )


# Reference values from R 4.6.1, stats::mantelhaen.test(x, correct=TRUE) and
# correct=FALSE, where x is an array of dim c(2, 2, K) with
# x[, , k] <- matrix(c(a, c, b, d), 2, 2), i.e. the stratum [[a, b], [c, d]].
# The single stratum was padded in R with a stratum [[0, 0], [1, 1]] that
# contributes nothing to either sum.
# Columns: strata, sum(a - E(a)), corrected statistic and p-value,
# uncorrected statistic and p-value
mh_r_reference = [
    # |sum(a - E(a))| < 0.5: R does not apply the correction
    (
        [[[4, 3], [3, 1]], [[5, 0], [5, 2]]],
        0.378787878787879,
        (0.131651008973333, 0.716726320820142),
        (0.131651008973333, 0.716726320820142),
    ),
    # |sum(a - E(a))| > 0.5
    (
        [[[10, 10], [10, 10]], [[12, 8], [9, 11]]],
        1.5,
        (0.195244055068836, 0.658587409353435),
        (0.439299123904881, 0.507460932504393),
    ),
    # |sum(a - E(a))| = 0.5 exactly, the correction removes everything
    (
        [[[3, 0], [3, 0]], [[4, 0], [3, 1]]],
        0.5,
        (0.0, 1.0),
        (1.0, 0.317310507862914),
    ),
    # Each stratum has |a - E(a)| = 0.4 but the sum is 0.8
    (
        [[[1, 0], [2, 2]], [[4, 11], [2, 8]]],
        0.8,
        (0.0652173913043478, 0.798431964808643),
        (0.463768115942029, 0.495868184540851),
    ),
    # A single stratum
    (
        [[[98, 1], [1, 1]]],
        0.960396039603964,
        (5.51538108356299, 0.0188499660989098),
        (24.0001020304052, 9.63305959467338e-07),
    ),
]


@pytest.mark.parametrize("tables, delta, corrected, uncorrected", mh_r_reference)
def test_null_odds_correction_r(tables, delta, corrected, uncorrected):
    st = ctab.StratifiedTable([np.asarray(t, dtype=float) for t in tables])
    actual = np.sum(st.table[0, 0, :] - st._apb * st._apc / st._n)
    assert_allclose(actual, delta, rtol=1e-10)

    for correction, expected in ((True, corrected), (False, uncorrected)):
        res = st.test_null_odds(correction=correction)
        assert_allclose(res.statistic, expected[0], rtol=1e-10, atol=1e-14)
        assert_allclose(res.pvalue, expected[1], rtol=1e-10)


@pytest.mark.parametrize(
    "tables",
    [
        [[[2, 0], [0, 0]]],
        [[[2, 0], [0, 0]], [[3, 0], [0, 0]]],
    ],
)
@pytest.mark.parametrize("correction", [True, False])
def test_null_odds_zero_variance(tables, correction):
    # The statistic is 0 / 0; R's mantelhaen.test returns NaN as well. With
    # the correction the statistic used to be inf and the p-value 0.
    st = ctab.StratifiedTable([np.asarray(t, dtype=float) for t in tables])
    # There are no floating point exceptions, and so no warning, in WASM
    if PYTHON_IMPL_WASM:
        warns = nullcontext()
    else:
        warns = pytest.warns(RuntimeWarning, match="invalid value")
    with warns:
        res = st.test_null_odds(correction=correction)
    assert np.isnan(res.statistic)
    assert np.isnan(res.pvalue)
