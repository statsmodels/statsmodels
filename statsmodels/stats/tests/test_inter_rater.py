"""

Created on Mon Dec 10 09:18:14 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest

from statsmodels.stats.inter_rater import (
    aggregate_raters,
    cohens_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    to_table,
)
from statsmodels.tools.testing import Holder

table0 = np.asarray(
    """\
1 	0 	0 	0 	0 	14 	1.000
2 	0 	2 	6 	4 	2 	0.253
3 	0 	0 	3 	5 	6 	0.308
4 	0 	3 	9 	2 	0 	0.440
5 	2 	2 	8 	1 	1 	0.330
6 	7 	7 	0 	0 	0 	0.462
7 	3 	2 	6 	3 	0 	0.242
8 	2 	5 	3 	2 	2 	0.176
9 	6 	5 	2 	1 	0 	0.286
10 	0 	2 	2 	3 	7 	0.286""".split(),
    float,
).reshape(10, -1)

table1 = table0[:, 1:-1]

table10 = [[0, 4, 1], [0, 8, 0], [0, 1, 5]]

# Fleiss 1971, Fleiss has only the transformed table
diagnoses = np.array(
    [
        [4, 4, 4, 4, 4, 4],
        [2, 2, 2, 5, 5, 5],
        [2, 3, 3, 3, 3, 5],
        [5, 5, 5, 5, 5, 5],
        [2, 2, 2, 4, 4, 4],
        [1, 1, 3, 3, 3, 3],
        [3, 3, 3, 3, 5, 5],
        [1, 1, 3, 3, 3, 4],
        [1, 1, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
        [1, 4, 4, 4, 4, 4],
        [1, 2, 4, 4, 4, 4],
        [2, 2, 2, 3, 3, 3],
        [1, 4, 4, 4, 4, 4],
        [2, 2, 4, 4, 4, 5],
        [3, 3, 3, 3, 3, 5],
        [1, 1, 1, 4, 5, 5],
        [1, 1, 1, 1, 1, 2],
        [2, 2, 4, 4, 4, 4],
        [1, 3, 3, 5, 5, 5],
        [5, 5, 5, 5, 5, 5],
        [2, 4, 4, 4, 4, 4],
        [2, 2, 4, 5, 5, 5],
        [1, 1, 4, 4, 4, 4],
        [1, 4, 4, 4, 4, 5],
        [2, 2, 2, 2, 2, 4],
        [1, 1, 1, 1, 5, 5],
        [2, 2, 4, 4, 4, 4],
        [1, 3, 3, 3, 3, 3],
        [5, 5, 5, 5, 5, 5],
    ]
)
diagnoses_rownames = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
]
diagnoses_colnames = [
    "rater1",
    "rater2",
    "rater3",
    "rater4",
    "rater5",
    "rater6",
]


def test_fleiss_kappa():
    # currently only example from Wikipedia page
    kappa_wp = 0.210
    assert_almost_equal(fleiss_kappa(table1), kappa_wp, decimal=3)


def test_fleis_randolph():
    # reference numbers from online calculator
    # http://justusrandolph.net/kappa/#dInfo
    table = [[7, 0], [7, 0]]
    assert_equal(fleiss_kappa(table, method="unif"), 1)

    table = [[6.99, 0.01], [6.99, 0.01]]
    # % Overall Agreement 0.996671
    # Fixed Marginal Kappa: -0.166667
    # Free Marginal Kappa: 0.993343
    assert_allclose(fleiss_kappa(table), -0.166667, atol=6e-6)
    assert_allclose(fleiss_kappa(table, method="unif"), 0.993343, atol=6e-6)

    table = [[7, 1], [3, 5]]
    # % Overall Agreement 0.607143
    # Fixed Marginal Kappa: 0.161905
    # Free Marginal Kappa: 0.214286
    assert_allclose(fleiss_kappa(table, method="fleiss"), 0.161905, atol=6e-6)
    assert_allclose(fleiss_kappa(table, method="randolph"), 0.214286, atol=6e-6)

    table = [[7, 0], [0, 7]]
    # % Overall Agreement 1.000000
    # Fixed Marginal Kappa: 1.000000
    # Free Marginal Kappa: 1.000000
    assert_allclose(fleiss_kappa(table), 1)
    assert_allclose(fleiss_kappa(table, method="uniform"), 1)

    table = [[6, 1, 0], [0, 7, 0]]
    # % Overall Agreement 0.857143
    # Fixed Marginal Kappa: 0.708333
    # Free Marginal Kappa: 0.785714
    assert_allclose(fleiss_kappa(table), 0.708333, atol=6e-6)
    assert_allclose(fleiss_kappa(table, method="rand"), 0.785714, atol=6e-6)


class CheckCohens:

    def test_results(self):
        res = self.res
        res2 = self.res2

        res_ = [
            res.kappa,
            res.std_kappa,
            res.kappa_low,
            res.kappa_upp,
            res.std_kappa0,
            res.z_value,
            res.pvalue_one_sided,
            res.pvalue_two_sided,
        ]

        assert_almost_equal(res_, res2, decimal=4)
        assert_equal(str(res), self.res_string)


class TestUnweightedCohens(CheckCohens):
    # comparison to printout of a SAS example
    @classmethod
    def setup_class(cls):
        # temporary: res instance is at last position
        cls.res = cohens_kappa(table10)
        res10_sas = [0.4842, 0.1380, 0.2137, 0.7547]
        res10_sash0 = [0.1484, 3.2626, 0.0006, 0.0011]  # for test H0:kappa=0
        cls.res2 = res10_sas + res10_sash0  # concatenate

        cls.res_string = """\
                  Simple Kappa Coefficient
              --------------------------------
              Kappa                     0.4842
              ASE                       0.1380
              95% Lower Conf Limit      0.2137
              95% Upper Conf Limit      0.7547

                 Test of H0: Simple Kappa = 0

              ASE under H0              0.1484
              Z                         3.2626
              One-sided Pr >  Z         0.0006
              Two-sided Pr > |Z|        0.0011""" + "\n"

    def test_option(self):
        kappa = cohens_kappa(table10, return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)


class TestWeightedCohens(CheckCohens):
    # comparison to printout of a SAS example
    @classmethod
    def setup_class(cls):
        # temporary: res instance is at last position
        cls.res = cohens_kappa(table10, weights=[0, 1, 2])
        res10w_sas = [0.4701, 0.1457, 0.1845, 0.7558]
        res10w_sash0 = [0.1426, 3.2971, 0.0005, 0.0010]  # for test H0:kappa=0
        cls.res2 = res10w_sas + res10w_sash0  # concatenate

        cls.res_string = """\
                  Weighted Kappa Coefficient
              --------------------------------
              Kappa                     0.4701
              ASE                       0.1457
              95% Lower Conf Limit      0.1845
              95% Upper Conf Limit      0.7558

                 Test of H0: Weighted Kappa = 0

              ASE under H0              0.1426
              Z                         3.2971
              One-sided Pr >  Z         0.0005
              Two-sided Pr > |Z|        0.0010""" + "\n"

    def test_option(self):
        kappa = cohens_kappa(table10, weights=[0, 1, 2], return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)


def test_cohenskappa_weights():
    # some tests for equivalent results with different options
    rs = np.random.RandomState(9743678)
    table = rs.randint(0, 10, size=(5, 5)) + 5 * np.eye(5)

    # example aggregation, 2 groups of levels
    mat = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    table_agg = np.dot(np.dot(mat, table), mat.T)
    res1 = cohens_kappa(table, weights=np.arange(5) > 2, wt="linear")
    res2 = cohens_kappa(table_agg, weights=np.arange(2), wt="linear")
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

    # equivalence toeplitz with linear for special cases
    res1 = cohens_kappa(table, weights=2 * np.arange(5), wt="linear")
    res2 = cohens_kappa(table, weights=2 * np.arange(5), wt="toeplitz")
    res3 = cohens_kappa(table, weights=res1.weights[0], wt="toeplitz")
    # 2-Dim weights
    res4 = cohens_kappa(table, weights=res1.weights)

    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

    assert_almost_equal(res1.kappa, res3.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res3.var_kappa, decimal=14)

    assert_almost_equal(res1.kappa, res4.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res4.var_kappa, decimal=14)

    # equivalence toeplitz with quadratic for special cases
    res1 = cohens_kappa(table, weights=5 * np.arange(5) ** 2, wt="toeplitz")
    res2 = cohens_kappa(table, weights=5 * np.arange(5), wt="quadratic")
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)


anxiety = np.array(
    [
        3,
        3,
        3,
        4,
        5,
        5,
        2,
        3,
        5,
        2,
        2,
        6,
        1,
        5,
        2,
        2,
        1,
        2,
        4,
        3,
        3,
        6,
        4,
        6,
        2,
        4,
        2,
        4,
        3,
        3,
        2,
        3,
        3,
        3,
        2,
        2,
        1,
        3,
        3,
        4,
        2,
        1,
        4,
        4,
        3,
        2,
        1,
        6,
        1,
        1,
        1,
        2,
        3,
        3,
        1,
        1,
        3,
        3,
        2,
        2,
    ]
).reshape(20, 3, order="F")
anxiety_rownames = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
]
anxiety_colnames = [
    "rater1",
    "rater2",
    "rater3",
]


def test_cohens_kappa_irr():

    ck_w3 = Holder()
    ck_w4 = Holder()

    # >r = kappa2(anxiety[,1:2], c(0,0,0,1,1,1))
    # > cat_items(r, pref="ck_w3.")
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,0,1,1,1)"
    ck_w3.irr_name = "Kappa"
    ck_w3.value = 0.1891892
    ck_w3.stat_name = "z"
    ck_w3.statistic = 0.5079002
    ck_w3.p_value = 0.6115233

    # > r = kappa2(anxiety[,1:2], c(0,0,1,1,2,2))
    # > cat_items(r, pref="ck_w4.")
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2,2)"
    ck_w4.irr_name = "Kappa"
    ck_w4.value = 0.2820513
    ck_w4.stat_name = "z"
    ck_w4.statistic = 1.257410
    ck_w4.p_value = 0.2086053

    ck_w1 = Holder()
    ck_w2 = Holder()
    ck_w3 = Holder()
    ck_w4 = Holder()
    # > r = kappa2(anxiety[,2:3])
    # > cat_items(r, pref="ck_w1.")
    ck_w1.method = "Cohen's Kappa for 2 Raters (Weights: unweighted)"
    ck_w1.irr_name = "Kappa"
    ck_w1.value = -0.006289308
    ck_w1.stat_name = "z"
    ck_w1.statistic = -0.0604067
    ck_w1.p_value = 0.9518317

    # > r = kappa2(anxiety[,2:3], "equal")
    # > cat_items(r, pref="ck_w2.")
    ck_w2.method = "Cohen's Kappa for 2 Raters (Weights: equal)"
    ck_w2.irr_name = "Kappa"
    ck_w2.value = 0.1459075
    ck_w2.stat_name = "z"
    ck_w2.statistic = 1.282472
    ck_w2.p_value = 0.1996772

    # > r = kappa2(anxiety[,2:3], "squared")
    # > cat_items(r, pref="ck_w3.")
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: squared)"
    ck_w3.irr_name = "Kappa"
    ck_w3.value = 0.2520325
    ck_w3.stat_name = "z"
    ck_w3.statistic = 1.437451
    ck_w3.p_value = 0.1505898

    # > r = kappa2(anxiety[,2:3], c(0,0,1,1,2))
    # > cat_items(r, pref="ck_w4.")
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2)"
    ck_w4.irr_name = "Kappa"
    ck_w4.value = 0.2391304
    ck_w4.stat_name = "z"
    ck_w4.statistic = 1.223734
    ck_w4.p_value = 0.2210526

    all_cases = [
        (ck_w1, None, None),
        (ck_w2, None, "linear"),
        (ck_w2, np.arange(5), None),
        (ck_w2, np.arange(5), "toeplitz"),
        (ck_w3, None, "quadratic"),
        (ck_w3, np.arange(5) ** 2, "toeplitz"),
        (ck_w3, 4 * np.arange(5) ** 2, "toeplitz"),
        (ck_w4, [0, 0, 1, 1, 2], "toeplitz"),
    ]

    # Note R:irr drops the missing category level 4 and uses the reduced matrix
    r = np.histogramdd(anxiety[:, 1:], ([1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 6, 7]))

    for res2, w, wt in all_cases:
        msg = repr(w) + repr(wt)
        res1 = cohens_kappa(r[0], weights=w, wt=wt)
        assert_almost_equal(res1.kappa, res2.value, decimal=6, err_msg=msg)
        assert_almost_equal(res1.z_value, res2.statistic, decimal=5, err_msg=msg)
        assert_almost_equal(res1.pvalue_two_sided, res2.p_value, decimal=6, err_msg=msg)


def test_fleiss_kappa_irr():
    fleiss = Holder()
    # > r = kappam.fleiss(diagnoses)
    # > cat_items(r, pref="fleiss.")
    fleiss.method = "Fleiss' Kappa for m Raters"
    fleiss.irr_name = "Kappa"
    fleiss.value = 0.4302445
    fleiss.stat_name = "z"
    fleiss.statistic = 17.65183
    fleiss.p_value = 0
    data_, _ = aggregate_raters(diagnoses)
    res1_kappa = fleiss_kappa(data_)
    assert_almost_equal(res1_kappa, fleiss.value, decimal=7)


def test_to_table():
    data = diagnoses
    res1 = to_table(data[:, :2] - 1, 5)
    res0 = np.asarray(
        [[(data[:, :2] - 1 == [i, j]).all(1).sum() for j in range(5)] for i in range(5)]
    )
    assert_equal(res1[0], res0)

    res2 = to_table(data[:, :2])
    assert_equal(res2[0], res0)

    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    res3 = to_table(data[:, :2], bins)
    assert_equal(res3[0], res0)

    # more than 2 columns
    res4 = to_table(data[:, :3] - 1, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    res5 = to_table(data[:, :3] - 1, bins=5)
    assert_equal(res4[0].sum(-1), res0)
    assert_equal(res5[0].sum(-1), res0)


def test_aggregate_raters():
    data = diagnoses
    data_, categories = aggregate_raters(data)
    colsum = np.array([26, 26, 30, 55, 43])
    assert_equal(data_.sum(0), colsum)
    assert_equal(np.unique(diagnoses), categories)


def test_to_table_rejects_1d_data():
    # 1-d data used to crash with a bare "not enough values to unpack"
    with pytest.raises(ValueError, match="2-d array"):
        to_table(np.array([1, 2, 3]))


def test_cohens_kappa_rejects_non_square_table():
    # a 1-d table used to crash with a raw AxisError from np.diag
    with pytest.raises(ValueError, match="square 2-d array"):
        cohens_kappa(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="square 2-d array"):
        cohens_kappa(np.ones((2, 3)))


def test_fleiss_kappa_rejects_1d_table():
    # a 1-d table used to crash with a bare "not enough values to unpack"
    with pytest.raises(ValueError, match="2-d array"):
        fleiss_kappa(np.array([5, 5, 5]))


# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", section C:
# 12 units (rows) rated by 4 observers (columns), 7 ratings missing.
# https://repository.upenn.edu/asc_papers/43
kripp_data = np.array(
    [
        [1, 1, np.nan, 1],
        [2, 2, 3, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [4, 4, 4, 4],
        [1, 1, 2, 1],
        [2, 2, 2, 2],
        [np.nan, 5, 5, 5],
        [np.nan, np.nan, 1, 1],
        [np.nan, 3, np.nan, np.nan],
    ]
)


def _alpha_from_pairs(data, level):
    # Alpha from its definition, 1 - D_o / D_e, enumerating the ordered pairs
    # of ratings (Krippendorff 2011, section E) instead of building the
    # coincidence matrix.
    units = [row[~np.isnan(row)] for row in np.asarray(data, dtype=float)]
    units = [u for u in units if len(u) >= 2]
    values = np.concatenate(units)
    freq = {v: np.sum(values == v) for v in np.unique(values)}

    def delta2(c, k):
        if level == "nominal":
            return float(c != k)
        if level == "ordinal":
            lo, hi = min(c, k), max(c, k)
            between = sum(f for v, f in freq.items() if lo <= v <= hi)
            return (between - (freq[c] + freq[k]) / 2) ** 2
        if level == "interval":
            return (c - k) ** 2
        return 0.0 if c + k == 0 else ((c - k) / (c + k)) ** 2

    n = len(values)
    # within subjects, each pair is weighted by 1 / (m_u - 1)
    within = sum(
        delta2(u[i], u[j]) / (len(u) - 1)
        for u in units
        for i in range(len(u))
        for j in range(len(u))
        if i != j
    )
    # all pairs of pairable ratings, regardless of subject
    between = sum(
        delta2(values[i], values[j]) for i in range(n) for j in range(n) if i != j
    )
    d_o = within / n
    d_e = between / (n * (n - 1))
    return 1 - d_o / d_e


@pytest.mark.parametrize(
    "level, expected",
    [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849), ("ratio", 0.797)],
)
def test_krippendorff_alpha_published(level, expected):
    alpha = krippendorff_alpha(kripp_data, level=level)
    # the published values are rounded to three decimals
    assert_allclose(alpha, expected, atol=5e-4)
    assert_allclose(alpha, _alpha_from_pairs(kripp_data, level), rtol=1e-12)


@pytest.mark.parametrize(
    "level, expected",
    [
        ("nominal", 0.743421052631579),
        ("ordinal", 0.815387503754881),
        ("interval", 0.849107142857143),
        ("ratio", 0.797402774711612),
    ],
)
def test_krippendorff_alpha_irr(level, expected):
    # R 4.5.3, irr 0.85; kripp.alpha expects raters in rows
    # > nmm <- matrix(c(1, 2, 3, 3, 2, 1, 4, 1, 2, NA, NA, NA,
    # +                 1, 2, 3, 3, 2, 2, 4, 1, 2, 5, NA, 3,
    # +                 NA, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, NA,
    # +                 1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, NA), nrow = 4, byrow = TRUE)
    # > format(kripp.alpha(nmm, "nominal")$value, digits = 15)
    alpha = krippendorff_alpha(kripp_data, level=level)
    assert_allclose(alpha, expected, rtol=1e-13)


@pytest.mark.parametrize(
    "level, expected",
    [
        ("nominal", 0.43340982828202856),
        ("ordinal", 0.33585752217398357),
        ("interval", 0.28804962598066008),
        ("ratio", 0.2400102941476886),
    ],
)
def test_krippendorff_alpha_irr_complete(level, expected):
    # irr 0.85 kripp.alpha weights every pair by 1 instead of 1 / (m_u - 1)
    # when the data has no missing value, which is wrong for 3 or more
    # raters. The reference values are computed with one extra subject that
    # has a single rating, which makes irr use 1 / (m_u - 1) and does not
    # change alpha, since that subject has no pair of ratings.
    # R 4.5.3, irr 0.85; the category codes are the leading digits of the
    # labels in irr's diagnoses, which gives the `diagnoses` array above
    # > x <- sapply(diagnoses, function(v) as.integer(substr(as.character(v), 1, 1)))
    # > x <- rbind(x, c(1, NA, NA, NA, NA, NA))
    # > format(kripp.alpha(t(x), "nominal")$value, digits = 17)
    alpha = krippendorff_alpha(diagnoses, level=level)
    assert_allclose(alpha, expected, rtol=1e-13)


def test_krippendorff_alpha_published_fractions():
    # Exact values from the coincidence matrix printed in Krippendorff (2011),
    # section C: the off-diagonal coincidences o_ck (c < k) are
    # o_12 = o_23 = 4/3 and o_13 = o_14 = o_24 = o_34 = 1/3, the margins are
    # n_c = (9, 13, 10, 5, 3) and n = 40.
    # nominal: sum o_ck = 4 and sum n_c n_k = 608
    assert_allclose(krippendorff_alpha(kripp_data), 1 - 39 * 4 / 608, rtol=1e-13)
    # interval: sum o_ck (c - k)**2 = 26 / 3 and sum n_c n_k (c - k)**2 = 2240
    assert_allclose(
        krippendorff_alpha(kripp_data, level="interval"),
        1 - 39 * (26 / 3) / 2240,
        rtol=1e-13,
    )


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval", "ratio"])
def test_krippendorff_alpha_random_vs_pairs(level):
    rng = np.random.default_rng(1234)
    n_checked = 0
    while n_checked < 5:
        n_sub, n_rat = rng.integers(3, 15), rng.integers(2, 5)
        base = rng.integers(0, 4, n_sub)
        noise = rng.integers(0, 4, (n_sub, n_rat))
        data = np.where(rng.random((n_sub, n_rat)) < 0.6, base[:, None], noise)
        data = data.astype(float)
        data[rng.random(data.shape) < 0.2] = np.nan
        pairable = data[(~np.isnan(data)).sum(1) >= 2]
        if len(np.unique(pairable[~np.isnan(pairable)])) < 2:
            # alpha is undefined, see test_krippendorff_alpha_undefined
            continue
        # zeros are included on purpose, they exercise c + k = 0 for "ratio"
        expected = _alpha_from_pairs(data, level)
        assert_allclose(krippendorff_alpha(data, level=level), expected, rtol=1e-12)
        n_checked += 1


def test_krippendorff_alpha_fleiss_kappa():
    # Without missing ratings, nominal alpha is a function of Fleiss' kappa,
    # alpha = 1 - (1 - kappa) * (n - 1) / n with n the number of ratings.
    kappa = fleiss_kappa(aggregate_raters(diagnoses)[0])
    n = diagnoses.size
    assert_allclose(
        krippendorff_alpha(diagnoses), 1 - (1 - kappa) * (n - 1) / n, rtol=1e-13
    )


def test_krippendorff_alpha_scotts_pi():
    # two raters: Scott's pi with pooled category proportions
    data = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 1], [1, 1], [0, 0]])
    p_observed = np.mean(data[:, 0] == data[:, 1])
    p_cat = np.bincount(data.ravel()) / data.size
    pi = (p_observed - (p_cat**2).sum()) / (1 - (p_cat**2).sum())
    n = data.size
    assert_allclose(krippendorff_alpha(data), 1 - (1 - pi) * (n - 1) / n, rtol=1e-13)


@pytest.mark.parametrize(
    "level, transform",
    [
        ("nominal", lambda x: 10 - 3 * x),
        ("ordinal", np.exp),
        ("interval", lambda x: 2.5 * x - 7),
        ("ratio", lambda x: 0.4 * x),
    ],
)
def test_krippendorff_alpha_invariance(level, transform):
    # nominal: any relabeling; ordinal: increasing transform; interval:
    # affine transform; ratio: change of unit
    alpha = krippendorff_alpha(kripp_data, level=level)
    assert_allclose(krippendorff_alpha(transform(kripp_data), level=level), alpha)
    # order of subjects and raters does not matter
    shuffled = kripp_data[::-1][:, [2, 0, 3, 1]]
    assert_allclose(krippendorff_alpha(shuffled, level=level), alpha)


def test_krippendorff_alpha_unpairable_subjects_ignored():
    # subjects with fewer than two ratings do not contribute
    extra = np.array([[np.nan, 7, np.nan, np.nan], [np.nan] * 4])
    data = np.vstack([kripp_data, extra])
    assert_allclose(krippendorff_alpha(data), krippendorff_alpha(kripp_data))


def test_krippendorff_alpha_input_types():
    data = [list(row) for row in kripp_data]
    df = pd.DataFrame(kripp_data, columns=list("ABCD"))
    alpha = krippendorff_alpha(kripp_data, level="ordinal")
    assert_allclose(krippendorff_alpha(data, level="ordinal"), alpha)
    assert_allclose(krippendorff_alpha(df, level="ordinal"), alpha)
    assert isinstance(alpha, float)


def test_krippendorff_alpha_perfect_agreement():
    data = np.array([[1, 1, np.nan], [2, 2, 2], [3, np.nan, 3]])
    for level in ["nominal", "ordinal", "interval", "ratio"]:
        assert_equal(krippendorff_alpha(data, level=level), 1.0)


def test_krippendorff_alpha_undefined():
    # no subject has two ratings
    data = np.array([[1, np.nan], [np.nan, 2], [3, np.nan]])
    assert np.isnan(krippendorff_alpha(data))
    # all pairable ratings are equal; the single rating 2 is not pairable
    data = np.array([[1, 1], [1, 1], [2, np.nan]])
    assert np.isnan(krippendorff_alpha(data, level="interval"))


def test_krippendorff_alpha_raises():
    with pytest.raises(ValueError, match="2-d array"):
        krippendorff_alpha(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="ndim"):
        krippendorff_alpha(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="level"):
        krippendorff_alpha(kripp_data, level="circular")
    with pytest.raises(ValueError, match="inf"):
        krippendorff_alpha(np.array([[1, np.inf], [1, 2]]))
    with pytest.raises(ValueError, match="nonnegative"):
        krippendorff_alpha(np.array([[1, -1], [1, 2]]), level="ratio")
