"""
Tests for statsmodels.stats.multivariate_tools

Regression tests: cc_stats had no test coverage and raised AttributeError on
numpy >= 2.0 because it used the removed alias ``np.product``.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.stats.multivariate_tools import cancorr, cc_stats


@pytest.fixture
def data():
    rs = np.random.RandomState(12345)
    x1 = rs.standard_normal((200, 3))
    x2 = rs.standard_normal((200, 2))
    return x1, x2


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


def test_cancorr_perfect_correlation():
    # x with itself has all canonical correlations equal to one, so
    # Wilks' Lambda collapses to zero.
    rs = np.random.RandomState(12345)
    x1 = rs.standard_normal((50, 2))
    assert_allclose(cancorr(x1, x1), np.ones(2), atol=1e-8)
