"""
Tests for statsmodels.stats.multivariate_tools

Regression tests: cc_stats had no test coverage and raised AttributeError on
numpy >= 2.0 because it used the removed alias ``np.product``. cc_ranktest
had no test coverage at all, and could raise TypeError whenever cancorr's
underlying eigenvalue solver returned a complex dtype (which happens for
some inputs even though the canonical correlations are mathematically real).
"""
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats as sp_stats

from statsmodels.stats.multivariate_tools import (
    CCRankTestResult,
    CCStatsResult,
    PartialProjectResult,
    cancorr,
    cc_ranktest,
    cc_stats,
    partial_project,
)


@pytest.fixture
def data():
    rs = np.random.RandomState(12345)
    x1 = rs.standard_normal((200, 3))
    x2 = rs.standard_normal((200, 2))
    return x1, x2


@pytest.fixture
def data_r():
    # Same data used to generate the reference values below with R 4.6.1:
    #
    #   cc <- cancor(x1, x2, xcenter = TRUE, ycenter = TRUE)
    #   fit <- lm(x1 ~ x2)
    #   summary(manova(fit))$stats["x2", "Pillai"]
    #   summary(manova(fit), test = "Wilks")$stats["x2", "Wilks"]
    #   summary(manova(fit), test = "Hotelling-Lawley")$stats["x2", "Hotelling-Lawley"]
    #   summary(manova(fit), test = "Roy")$stats["x2", "Roy"]
    rs = np.random.RandomState(20260821)
    x1 = rs.standard_normal((40, 3))
    x2 = x1[:, :2] * 0.5 + rs.standard_normal((40, 2))
    x2 = np.hstack([x2, rs.standard_normal((40, 1))])
    return x1, x2


# ---------------------------------------------------------------------
# cancorr
# ---------------------------------------------------------------------

def test_cancorr_perfect_correlation():
    # x with itself has all canonical correlations equal to one, so
    # Wilks' Lambda collapses to zero.
    rs = np.random.RandomState(12345)
    x1 = rs.standard_normal((50, 2))
    assert_allclose(cancorr(x1, x1), np.ones(2), atol=1e-8)


def test_cancorr_shape():
    rs = np.random.RandomState(12345)
    x1 = rs.standard_normal((50, 2))
    _x2 = x1 * 0.7 + rs.standard_normal((50, 2))
    x2 = np.hstack([_x2, rs.standard_normal((50, 1))])
    cc = cancorr(x1, x2)
    assert len(cc) == 2


def test_cancorr_against_r(data_r):
    # Reference values from R's base `cancor` function.
    x1, x2 = data_r
    cc = cancorr(x1, x2)
    cc_expected = [0.532338026574400, 0.344170637520608, 0.114617183600323]
    assert_allclose(np.real(cc), cc_expected, rtol=1e-7)


def test_cancorr_returns_real_dtype(data_r):
    # Regression test: eigvals of the (generally non-symmetric) auxiliary
    # matrix can come back with complex dtype even though the underlying
    # canonical correlations are mathematically real. cancorr must not leak
    # that dtype since cc_ranktest feeds the squared values straight into
    # scipy.stats.chi2.sf, which raises TypeError on complex input.
    x1, x2 = data_r
    cc = cancorr(x1, x2)
    assert not np.iscomplexobj(cc)
    # cc_ranktest must not raise despite this.
    cc_ranktest(x1, x2)


def test_cancorr_standardize_same_as_demean():
    # Rescaling by the standard deviation should not change the canonical
    # correlation coefficients (see docstring).
    rs = np.random.RandomState(7)
    x1 = rs.standard_normal((80, 2)) * 3 + 1
    x2 = rs.standard_normal((80, 2)) * 0.2 - 4
    cc_demean = cancorr(x1, x2, demean=True, standardize=False)
    cc_std = cancorr(x1, x2, demean=True, standardize=True)
    assert_allclose(cc_demean, cc_std, rtol=1e-10)


def test_cancorr_demean_false_differs():
    # Without demeaning, the location of the data enters the projection and
    # generally changes the canonical correlations.
    rs = np.random.RandomState(10)
    x1 = rs.standard_normal((80, 2)) + 5
    x2 = rs.standard_normal((80, 2)) + 3
    cc_demean = cancorr(x1, x2, demean=True)
    cc_raw = cancorr(x1, x2, demean=False)
    assert not np.allclose(cc_demean, cc_raw)


def test_cancorr_accepts_array_like():
    rs = np.random.RandomState(9)
    x1 = rs.standard_normal((40, 2))
    x2 = rs.standard_normal((40, 2))
    cc_ndarray = cancorr(x1, x2)
    cc_list = cancorr(x1.tolist(), x2.tolist())
    cc_df = cancorr(pd.DataFrame(x1), pd.DataFrame(x2))
    assert_allclose(cc_list, cc_ndarray, rtol=1e-12)
    assert_allclose(cc_df, cc_ndarray, rtol=1e-12)


# ---------------------------------------------------------------------
# cc_ranktest
# ---------------------------------------------------------------------

def test_cc_ranktest_fullrank_univariate_matches_formula():
    # With one variable on each side, the canonical correlation is the
    # Pearson correlation (an independent computation via np.corrcoef), and
    # the fullrank LM/Wald statistics reduce to simple closed-form
    # expressions in that correlation.
    rs = np.random.RandomState(11)
    x1 = rs.standard_normal((120, 1))
    x2 = x1 * 0.6 + rs.standard_normal((120, 1))

    r = np.corrcoef(x1.ravel(), x2.ravel())[0, 1]
    nobs = x1.shape[0]
    expected_lm = nobs * r**2
    expected_wald = nobs * r**2 / (1 - r**2)
    expected_df = 1

    value, pvalue, df, cc, w_value, w_pvalue = cc_ranktest(x1, x2, fullrank=True)
    assert_allclose(value, expected_lm, rtol=1e-10)
    assert_allclose(w_value, expected_wald, rtol=1e-10)
    assert df == expected_df
    assert_allclose(pvalue, sp_stats.chi2.sf(expected_lm, expected_df), rtol=1e-10)
    assert_allclose(w_pvalue, sp_stats.chi2.sf(expected_wald, expected_df), rtol=1e-10)
    assert_allclose(cc, [abs(r)], rtol=1e-10)


def test_cc_ranktest_fullrank_matches_smallest_cc(data):
    x1, x2 = data
    nobs = x1.shape[0]
    cc = cancorr(x1, x2)
    cc2_min = cc[-1] ** 2

    value, pvalue, df, cc_out, w_value, w_pvalue = cc_ranktest(x1, x2, fullrank=True)
    assert_allclose(value, nobs * cc2_min, rtol=1e-12)
    assert_allclose(w_value, nobs * cc2_min / (1 - cc2_min), rtol=1e-12)
    assert_allclose(cc_out, cc, rtol=1e-12)


def test_cc_ranktest_not_fullrank_matches_cumulative_formula(data):
    x1, x2 = data
    nobs, k1 = x1.shape
    _, k2 = x2.shape
    cc = cancorr(x1, x2)
    cc2 = cc ** 2
    r = np.arange(min(k1, k2))[::-1]
    expected_df = (k1 - r) * (k2 - r)
    expected_value = nobs * cc2[::-1].cumsum()
    expected_wald = nobs * (cc2 / (1 - cc2))[::-1].cumsum()

    value, pvalue, df, cc_out, w_value, w_pvalue = cc_ranktest(x1, x2, fullrank=False)
    assert_allclose(value, expected_value, rtol=1e-12)
    assert_allclose(w_value, expected_wald, rtol=1e-12)
    assert_allclose(df, expected_df)
    assert len(value) == min(k1, k2)


@pytest.mark.parametrize("fullrank", [True, False])
def test_cc_ranktest_return_object(data, fullrank):
    x1, x2 = data
    legacy = cc_ranktest(x1, x2, fullrank=fullrank)
    obj = cc_ranktest(x1, x2, fullrank=fullrank, return_object=True)

    assert isinstance(obj, CCRankTestResult)
    slots = obj.__slots__
    for slot, b in zip(slots, legacy, strict=True):
        a = getattr(obj, slot)
        assert_allclose(a, b, rtol=1e-12)

    assert_allclose(obj.statistic, legacy[0], rtol=1e-12)
    assert_allclose(obj.pvalue, legacy[1], rtol=1e-12)
    assert_allclose(obj.df, legacy[2], rtol=1e-12)
    assert_allclose(obj.ccorr, legacy[3], rtol=1e-12)
    assert_allclose(obj.wald_statistic, legacy[4], rtol=1e-12)
    assert_allclose(obj.wald_pvalue, legacy[5], rtol=1e-12)


def test_cc_ranktest_accepts_array_like(data):
    x1, x2 = data
    legacy = cc_ranktest(x1, x2)
    from_list = cc_ranktest(x1.tolist(), x2.tolist())
    from_df = cc_ranktest(pd.DataFrame(x1), pd.DataFrame(x2))
    for a, b in zip(legacy, from_list, strict=True):
        assert_allclose(a, b, rtol=1e-10)
    for a, b in zip(legacy, from_df, strict=True):
        assert_allclose(a, b, rtol=1e-10)


# ---------------------------------------------------------------------
# cc_stats
# ---------------------------------------------------------------------

def test_cc_stats_runs(data):
    x1, x2 = data
    res = cc_stats(x1, x2)
    expected_keys = {
        "canonical correlation coefficient",
        "eigenvalues",
        "Pillai's Trace",
        "Wilk's Lambda",
        "Hotelling's Trace",
        "Roy's Largest Root",
        "df_resid",
        "df_m",
    }
    assert expected_keys == set(res)
    assert np.isfinite(res["Wilk's Lambda"])


def test_cc_stats_identities(data):
    # The four classical MANOVA statistics are exact functions of the
    # squared canonical correlations.
    x1, x2 = data
    res = cc_stats(x1, x2)
    cc2 = cancorr(x1, x2) ** 2
    lam = cc2 / (1 - cc2)

    assert_allclose(res["Wilk's Lambda"], np.prod(1 - cc2), rtol=1e-12)
    assert_allclose(res["Pillai's Trace"], cc2.sum(), rtol=1e-12)
    assert_allclose(res["Hotelling's Trace"], lam.sum(), rtol=1e-12)
    assert_allclose(res["Roy's Largest Root"], lam.max(), rtol=1e-12)


def test_cc_stats_univariate_equals_correlation():
    # With one variable on each side the canonical correlation is the
    # absolute Pearson correlation and Wilks' Lambda is 1 - r ** 2.
    rs = np.random.RandomState(12345)
    x1 = rs.standard_normal((100, 1))
    x2 = x1 * 0.7 + rs.standard_normal((100, 1))

    r = np.corrcoef(x1.ravel(), x2.ravel())[0, 1]
    res = cc_stats(x1, x2)

    assert_allclose(res["canonical correlation coefficient"], [abs(r)], rtol=1e-10)
    assert_allclose(res["Wilk's Lambda"], 1 - r**2, rtol=1e-10)
    assert_allclose(res["Pillai's Trace"], r**2, rtol=1e-10)


def test_cc_stats_against_r(data_r):
    # Reference values from R's `summary(manova(lm(x1 ~ x2)))` for the four
    # classical MANOVA test statistics, which are equivalent to cc_stats
    # with demean=True (the default).
    x1, x2 = data_r
    res = cc_stats(x1, x2)
    assert_allclose(res["Pillai's Trace"], 0.414974301044939, rtol=1e-7)
    assert_allclose(res["Wilk's Lambda"], 0.623431470196253, rtol=1e-7)
    assert_allclose(res["Hotelling's Trace"], 0.543129049437308, rtol=1e-7)
    assert_allclose(res["Roy's Largest Root"], 0.395447053063982, rtol=1e-7)


def test_cc_stats_demean_false():
    # Not demeaning changes the projection (the mean is no longer
    # partialled out), so the results generally differ from demean=True,
    # but the returned statistics must still be exact functions of
    # cancorr(..., demean=False).
    rs = np.random.RandomState(4)
    x1 = rs.standard_normal((60, 2)) + 5
    x2 = rs.standard_normal((60, 2)) + 3
    res_demean = cc_stats(x1, x2, demean=True)
    res_raw = cc_stats(x1, x2, demean=False)
    assert not np.allclose(res_demean["Wilk's Lambda"], res_raw["Wilk's Lambda"])

    cc2 = cancorr(x1, x2, demean=False) ** 2
    assert_allclose(res_raw["Pillai's Trace"], cc2.sum(), rtol=1e-12)


def test_cc_stats_return_object(data):
    x1, x2 = data
    res_dict = cc_stats(x1, x2)
    res_obj = cc_stats(x1, x2, return_object=True)

    assert isinstance(res_obj, CCStatsResult)
    assert_allclose(res_obj.ccorr, res_dict["canonical correlation coefficient"])
    assert_allclose(res_obj.eigenvalues, res_dict["eigenvalues"])
    assert_allclose(res_obj.pillai_trace, res_dict["Pillai's Trace"])
    assert_allclose(res_obj.wilks_lambda, res_dict["Wilk's Lambda"])
    assert_allclose(res_obj.hotelling_trace, res_dict["Hotelling's Trace"])
    assert_allclose(res_obj.roys_largest_root, res_dict["Roy's Largest Root"])
    assert res_obj.df_resid == res_dict["df_resid"]
    assert res_obj.df_model == res_dict["df_m"]


def test_cc_stats_accepts_array_like(data):
    x1, x2 = data
    res_ndarray = cc_stats(x1, x2)
    res_list = cc_stats(x1.tolist(), x2.tolist())
    res_df = cc_stats(pd.DataFrame(x1), pd.DataFrame(x2))
    for key in res_ndarray:
        assert_allclose(res_list[key], res_ndarray[key], rtol=1e-12)
        assert_allclose(res_df[key], res_ndarray[key], rtol=1e-12)


# ---------------------------------------------------------------------
# partial_project
# ---------------------------------------------------------------------

def test_partial_project_matches_lstsq():
    # Cross-check against numpy's lstsq, an independent least-squares
    # implementation, rather than only checking internal consistency.
    rs = np.random.RandomState(3)
    endog = rs.standard_normal((50, 2))
    exog = np.column_stack([np.ones(50), rs.standard_normal((50, 2))])
    res = partial_project(endog, exog)
    assert isinstance(res, PartialProjectResult)

    expected_params = np.linalg.lstsq(exog, endog, rcond=None)[0]
    assert_allclose(res.params, expected_params, rtol=1e-10)
    assert_allclose(res.fittedvalues, exog.dot(expected_params), rtol=1e-10)
    assert_allclose(res.resid, endog - exog.dot(expected_params), rtol=1e-10)


def test_partial_project_orthogonality():
    # The projected residual must be orthogonal to every column of exog:
    # the defining property of an OLS projection, verified independently of
    # the pinv-based implementation.
    rs = np.random.RandomState(5)
    endog = rs.standard_normal((60, 3))
    exog = np.column_stack([np.ones(60), rs.standard_normal((60, 2))])
    res = partial_project(endog, exog)
    cross = exog.T.dot(res.resid)
    assert_allclose(cross, np.zeros_like(cross), atol=1e-8)


def test_partial_project_accepts_array_like():
    rs = np.random.RandomState(6)
    endog = rs.standard_normal((30, 2))
    exog = rs.standard_normal((30, 2))
    res_ndarray = partial_project(endog, exog)
    res_list = partial_project(endog.tolist(), exog.tolist())
    res_df = partial_project(pd.DataFrame(endog), pd.DataFrame(exog))
    assert_allclose(res_list.resid, res_ndarray.resid, rtol=1e-10)
    assert_allclose(res_df.resid, res_ndarray.resid, rtol=1e-10)


def test_partial_project_1d_input():
    # array_like inserts a trailing axis for 1-D input, so a single-column
    # endog/exog specified as a plain 1-D array must still work.
    rs = np.random.RandomState(8)
    endog = rs.standard_normal(40)
    exog = rs.standard_normal(40)
    res = partial_project(endog, exog)
    assert res.params.shape == (1, 1)
    assert res.resid.shape == (40, 1)
