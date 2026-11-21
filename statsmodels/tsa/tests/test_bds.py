"""
Test for BDS test for IID time series

References
----------

Kanzler, Ludwig. 1998.
BDS: MATLAB Module to Calculate Brock, Dechert & Scheinkman Test for
Independence.
Statistical Software Components. Boston College Department of Economics.
http://ideas.repec.org/c/boc/bocode/t871803.html.

"""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
from scipy import stats

from statsmodels.tsa._bds import (
    correlation_sum,
    correlation_sums,
    distance_indicators,
)
from statsmodels.tsa.stattools import bds
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

curdir = Path(__file__).resolve().parent
data_file = Path(curdir).joinpath("results", "bds_data.csv")
data = pd.read_csv(data_file, header=None)

res_file = Path(curdir).joinpath("results", "bds_results.csv")
results = pd.read_csv(res_file, header=None)


class CheckBDS:
    """
    Test BDS

    Test values from Kanzler's MATLAB program bds.
    """
    def test_stat(self):
        assert_almost_equal(self.res[0], self.bds_stats, DECIMAL_8)

    def test_pvalue(self):
        assert_almost_equal(self.res[1], self.pvalues, DECIMAL_8)


class TestBDSSequence(CheckBDS):
    """
    BDS Test on np.arange(1,26)
    """
    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 1]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])

        cls.data = data[0][data[0].notna()]
        cls.res = bds(cls.data, 5)


class TestBDSNormal(CheckBDS):
    """
    BDS Test on np.random.normal(size=25)
    """
    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 2]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])

        cls.data = data[1][data[1].notna()]
        cls.res = bds(cls.data, 5)


class TestBDSCombined(CheckBDS):
    """
    BDS Test on np.r_[np.random.normal(size=25), np.random.uniform(size=25)]
    """
    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 3]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])

        cls.data = data[2][data[2].notna()]
        cls.res = bds(cls.data, 5)


class TestBDSGDPC1(CheckBDS):
    """
    BDS Test on GDPC1: 1947Q1 - 2013Q1

    References
    ----------
    http://research.stlouisfed.org/fred2/series/GDPC1
    """
    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 4]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])

        cls.data = data[3][data[3].notna()]
        cls.res = bds(cls.data, 5)


def _bds_reference(x, max_dim, epsilon):
    # The BDS statistic from its definition (Broock et al., 1996), with the
    # 1-dimensional correlation sum of the effect taken over the same
    # observations as the m-dimensional one (Kanzler, 1999, footnote 10)
    n = len(x)
    close = np.abs(x[:, None] - x[None, :]) < epsilon

    def corr_sum(first, dim):
        # The share of the pairs s < t of m-histories that end at
        # observations from first on and are within epsilon in each lag
        pairs = [(s, t) for s in range(first, n) for t in range(s + 1, n)]
        within = [all(close[s - j, t - j] for j in range(dim)) for s, t in pairs]
        return np.mean(within)

    c = corr_sum(0, 1)
    k = (np.sum(close.sum(1) ** 2) - 3 * close.sum() + 2 * n) / (
        n * (n - 1) * (n - 2)
    )
    out = []
    for m in range(2, max_dim + 1):
        variance = 4 * (
            k**m
            + 2 * sum(k ** (m - j) * c ** (2 * j) for j in range(1, m))
            + (m - 1) ** 2 * c ** (2 * m)
            - m**2 * k * c ** (2 * m - 2)
        )
        effect = corr_sum(m - 1, m) - corr_sum(m - 1, 1) ** m
        out.append(np.sqrt(n - m + 1) * effect / np.sqrt(variance))
    return np.array(out)


@pytest.mark.parametrize("max_dim", [2, 4])
def test_bds_definition(max_dim):
    x = np.random.default_rng(0).standard_normal(15)
    epsilon = 1.5 * x.std(ddof=1)
    expected = _bds_reference(x, max_dim, epsilon)
    stat, pvalue = bds(x, max_dim)
    # 0-dimensional arrays for max_dim = 2
    expected_shape = (max_dim - 1,) if max_dim > 2 else ()
    assert stat.shape == pvalue.shape == expected_shape
    assert_allclose(stat, np.squeeze(expected), rtol=1e-12)
    assert_allclose(pvalue, 2 * stats.norm.sf(np.abs(stat)), rtol=1e-12)
    # epsilon given directly, instead of as a multiple of the std. dev.
    stat_eps, _ = bds(x, max_dim, epsilon=epsilon)
    assert_allclose(stat_eps, stat, rtol=1e-12)
    expected = _bds_reference(x, max_dim, 0.8)
    stat_eps, _ = bds(x, max_dim, epsilon=0.8)
    assert_allclose(stat_eps, np.squeeze(expected), rtol=1e-12)


@pytest.mark.parametrize("form", SERIES_FORMS)
def test_bds_input_forms(form):
    x = np.random.default_rng(1).standard_normal(40)
    stat, pvalue = bds(series_forms(x)[form], 3)
    expected_stat, expected_pvalue = bds(x, 3)
    assert_allclose(stat, expected_stat, rtol=1e-14)
    assert_allclose(pvalue, expected_pvalue, rtol=1e-14)


def test_bds_int_input():
    x = np.arange(30) % 7
    assert_allclose(bds(x, 3)[0], bds(x.astype(float), 3)[0], rtol=1e-14)


@pytest.mark.parametrize("max_dim", [1, 30])
def test_bds_max_dim_out_of_range(max_dim):
    x = np.random.default_rng(2).standard_normal(30)
    with pytest.raises(ValueError, match="Maximum embedding dimension"):
        bds(x, max_dim)


@pytest.mark.parametrize("form", ["list", "tuple", "series"])
def test_distance_indicators(form):
    x = np.array([0.0, 0.5, 2.0, 2.2])
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    res = distance_indicators(series_forms(x)[form], epsilon=1.0)
    assert res.dtype == bool
    np.testing.assert_array_equal(res, expected)
    # epsilon is distance times the standard deviation, 1.0 here
    distance = 1.0 / x.std(ddof=1)
    np.testing.assert_array_equal(distance_indicators(x, distance=distance), expected)


@pytest.mark.parametrize(
    "form", ["column", "dataframe", "two columns", "row", "1x1", "3d"]
)
def test_distance_indicators_2d_deprecated(form):
    # Input with more than one dimension returned the indicators of each
    # column along a last axis, and for a row those of each element with
    # itself. It is deprecated, and gives the same result with a warning.
    x = np.array([0.0, 0.5, 2.0, 2.2])
    near = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    # 2 * x: only 2.0 and 2.2 are closer than 1.0
    near2 = np.eye(4, dtype=bool)
    near2[2, 3] = near2[3, 2] = True
    column = "a column with shape \\(nobs, 1\\) will give the \\(nobs, nobs\\)"
    other = "It will raise a ValueError after statsmodels 0.17"
    value, expected, future = {
        "column": (x[:, None], near[:, :, None], column),
        "dataframe": (pd.DataFrame({"x": x}), near[:, :, None], column),
        "two columns": (np.c_[x, 2 * x], np.stack([near, near2], axis=-1), other),
        "row": (x[None, :], np.ones((1, 1, 4), dtype=bool), other),
        "1x1": (np.array([[1.0]]), np.ones((1, 1, 1), dtype=bool), column),
        "3d": (x[:, None, None], near[:, :, None, None], other),
    }[form]
    ndim = np.ndim(value)
    match = f"^x must be 1-D. Passing an array with {ndim} dimensions .*{future}"
    with pytest.warns(FutureWarning, match=match):
        res = distance_indicators(value, epsilon=1.0)
    assert res.dtype == bool
    np.testing.assert_array_equal(res, expected)


def test_distance_indicators_errors():
    x = np.arange(5.0)
    with pytest.raises(ValueError, match="epsilon"):
        distance_indicators(x, epsilon=0.0)
    with pytest.raises(ValueError, match="distance multiplier"):
        distance_indicators(x, distance=-1.0)


def test_correlation_sums():
    # The correlation sum of embedding dimension m is the share of the pairs
    # of m-histories that are within epsilon in every lag. Here all
    # consecutive values are within epsilon, and no other pairs are.
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    indicators = distance_indicators(x, epsilon=1.5)
    for m in range(1, 4):
        nhist = 6 - m + 1
        npairs = nhist * (nhist - 1) / 2
        corrsum, joint = correlation_sum(indicators, m)
        assert joint.shape == (nhist, nhist)
        assert_allclose(corrsum, (nhist - 1) / npairs, rtol=1e-15)
    expected = [correlation_sum(indicators, m)[0] for m in range(1, 5)]
    assert_allclose(correlation_sums(indicators, 4), [expected], rtol=1e-15)
    sums = correlation_sums(indicators.astype(float), 4)
    assert_allclose(sums, [expected], rtol=1e-15)


def test_correlation_sum_errors():
    with pytest.raises(ValueError, match="must be a matrix"):
        correlation_sum(np.ones(5, dtype=bool), 2)
    with pytest.raises(ValueError, match="must be symmetric"):
        correlation_sum(np.ones((5, 4), dtype=bool), 2)
