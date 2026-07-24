"""Tests for the Hamilton (2018) regression-based trend-cycle filter.

Reference
---------
Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
Review of Economics and Statistics, 100(5), 831-843.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.tsa.filters.hamilton_filter import hamilton_filter

# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
_QUARTERLY = RNG.standard_normal(200)  # 200 quarters
_SHORT = RNG.standard_normal(15)


# ---------------------------------------------------------------------------
# Output shapes and NaN locations
# ---------------------------------------------------------------------------


def test_output_shapes_default():
    x = np.ones(50)
    cycle, trend = hamilton_filter(x)
    assert cycle.shape == (50,)
    assert trend.shape == (50,)


def test_output_shapes_custom_h_p():
    T = 80
    x = RNG.standard_normal(T)
    cycle, trend = hamilton_filter(x, h=12, p=6)
    assert cycle.shape == (T,)
    assert trend.shape == (T,)


def test_nan_prefix_default():
    """First p+h-1 = 4+8-1 = 11 values must be NaN (default h=8, p=4)."""
    cycle, trend = hamilton_filter(_QUARTERLY)
    assert np.all(np.isnan(cycle[:11]))
    assert np.all(np.isnan(trend[:11]))


def test_nan_prefix_custom():
    """First p+h-1 values NaN for custom h, p."""
    h, p = 3, 2
    T = 30
    x = RNG.standard_normal(T)
    cycle, trend = hamilton_filter(x, h=h, p=p)
    prefix = p + h - 1  # = 4
    assert np.all(np.isnan(cycle[:prefix]))
    assert np.all(np.isnan(trend[:prefix]))


def test_finite_after_nan_prefix():
    cycle, trend = hamilton_filter(_QUARTERLY)
    assert np.all(np.isfinite(cycle[11:]))
    assert np.all(np.isfinite(trend[11:]))


# ---------------------------------------------------------------------------
# Mathematical identities
# ---------------------------------------------------------------------------


def test_trend_plus_cycle_equals_x():
    """Trend + cycle must equal x wherever both are defined."""
    x = _QUARTERLY
    cycle, trend = hamilton_filter(x)
    assert_allclose(trend[11:] + cycle[11:], x[11:], atol=1e-10)


def test_trend_plus_cycle_equals_x_custom():
    h, p = 5, 3
    x = RNG.standard_normal(60)
    cycle, trend = hamilton_filter(x, h=h, p=p)
    start = p + h - 1
    assert_allclose(trend[start:] + cycle[start:], x[start:], atol=1e-10)


def test_cycle_mean_near_zero():
    """OLS residuals have mean zero (constant included in regressors)."""
    cycle, _ = hamilton_filter(_QUARTERLY)
    assert_allclose(np.nanmean(cycle), 0.0, atol=1e-10)


def test_n_finite_obs():
    """Number of finite observations equals T - p - h + 1."""
    T, h, p = 100, 8, 4
    x = RNG.standard_normal(T)
    cycle, _ = hamilton_filter(x, h=h, p=p)
    n_finite = np.sum(np.isfinite(cycle))
    assert n_finite == T - p - h + 1


# ---------------------------------------------------------------------------
# Correctness against direct OLS
# ---------------------------------------------------------------------------


def test_matches_direct_ols():
    """Hamilton filter should match explicit OLS regression."""
    x = _QUARTERLY
    h, p = 8, 4
    T = len(x)
    n_obs = T - p - h + 1

    Y = x[p + h - 1 :]
    cols = [x[p - 1 - j : T - h - j] for j in range(p)]
    cols.append(np.ones(n_obs))
    X = np.column_stack(cols)
    params, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    ols_cycle = Y - X @ params

    # hamilton_filter returns (cycle, trend)
    ham_cycle, _ = hamilton_filter(x, h=h, p=p)
    assert_allclose(ham_cycle[p + h - 1 :], ols_cycle, atol=1e-10)


# ---------------------------------------------------------------------------
# Pandas integration
# ---------------------------------------------------------------------------


def test_pandas_series_input():
    s = pd.Series(_QUARTERLY, name="gdp")
    cycle, trend = hamilton_filter(s)
    assert isinstance(cycle, pd.Series)
    assert isinstance(trend, pd.Series)


def test_pandas_name_suffix():
    s = pd.Series(_QUARTERLY, name="gdp")
    cycle, trend = hamilton_filter(s)
    assert cycle.name == "gdp_cycle"
    assert trend.name == "gdp_trend"


def test_pandas_index_preserved():
    idx = pd.date_range("2000Q1", periods=len(_QUARTERLY), freq="QE")
    s = pd.Series(_QUARTERLY, index=idx, name="gdp")
    cycle, _ = hamilton_filter(s)
    assert (cycle.index == idx).all()


def test_pandas_dataframe_column():
    df = pd.DataFrame({"gdp": _QUARTERLY, "cpi": _QUARTERLY * 0.5})
    cycle, trend = hamilton_filter(df["gdp"])
    assert isinstance(cycle, pd.Series)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_raises_h_zero():
    with pytest.raises(ValueError, match="h must be a positive integer"):
        hamilton_filter(_QUARTERLY, h=0)


def test_raises_p_zero():
    with pytest.raises(ValueError, match="p must be a positive integer"):
        hamilton_filter(_QUARTERLY, p=0)


def test_raises_too_short():
    with pytest.raises(ValueError, match="x must have at least 2p \\+ h"):
        hamilton_filter(np.ones(5), h=8, p=4)


@pytest.mark.parametrize("h", [3, 7])
@pytest.mark.parametrize("p", [2, 5, 7, 11])
def test_minimum_length_works(h, p):
    # There are p+1 regressors, so require at least p+1 observations
    # Requires at least 2 * p + h observations
    x = RNG.standard_normal(2 * p + h)
    cycle, trend = hamilton_filter(x, h=h, p=p)
    # Minimum number of non-nan values is the same as the number of regressors
    assert np.sum(np.isfinite(cycle)) == (p + 1), np.sum(np.isfinite(cycle))
    with pytest.raises(ValueError, match="x must have at least 2p \\+ h"):
        hamilton_filter(x[:-1], h=h, p=p)


# ---------------------------------------------------------------------------
# Monthly and annual recommended settings
# ---------------------------------------------------------------------------


def test_monthly_settings():
    """Monthly defaults: h=24, p=12 (two years ahead, one year lags)."""
    T = 120  # 10 years monthly
    x = RNG.standard_normal(T)
    cycle, trend = hamilton_filter(x, h=24, p=12)
    assert cycle.shape == (T,)
    start = 24 + 12 - 1  # = 35
    assert np.all(np.isnan(cycle[:start]))
    assert np.all(np.isfinite(cycle[start:]))


def test_annual_settings():
    """Annual example: h=2, p=4."""
    T = 50
    x = RNG.standard_normal(T)
    cycle, trend = hamilton_filter(x, h=2, p=4)
    start = 5
    assert np.all(np.isnan(cycle[:start]))
    assert np.all(np.isfinite(cycle[start:]))


# ---------------------------------------------------------------------------
# Integration with statsmodels filters API
# ---------------------------------------------------------------------------


def test_accessible_via_tsa_filters():
    from statsmodels.tsa.filters.hamilton_filter import hamilton_filter as hf

    assert callable(hf)


def test_importable_from_api():
    from statsmodels.tsa.filters.api import hamilton_filter as hf

    assert callable(hf)


# ---------------------------------------------------------------------------
# Stationarity property (I(1) input → stationary cycle)
# ---------------------------------------------------------------------------


def test_stationary_cycle_from_random_walk():
    """Hamilton filter produces stationary cycle from a random walk."""
    T = 500
    rw = np.cumsum(RNG.standard_normal(T))
    cycle, _ = hamilton_filter(rw)
    c = cycle[~np.isnan(cycle)]
    # Rough stationarity check: variance of first half ≈ variance of second half
    n = len(c) // 2
    ratio = np.var(c[:n]) / np.var(c[n:])
    assert 0.3 < ratio < 3.0  # generous: just not trending wildly
