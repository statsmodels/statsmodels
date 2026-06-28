"""
Tests for LocalProjections (Jordà 2005) impulse response estimator.

Reference
---------
Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local
Projections. American Economic Review, 95(1), 161-182.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.vector_ar.lp import (
    LocalProjections,
    LocalProjectionsResults,
    _hac_bandwidth,
    _nw_cov,
)


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------


def _make_var1(T=300, n=2, seed=42):
    """Simulate a stable VAR(1) with known coefficient matrix A."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.2], [0.0, 0.4]])  # lower-triangular, stable
    y = np.zeros((T, n))
    e = rng.standard_normal((T, n))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + e[t]
    return y, A


def _make_ar1(T=500, phi=0.6, seed=0):
    """Simulate a univariate AR(1): y_t = phi * y_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + e[t]
    return y


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


def test_hac_bandwidth_zero_horizon():
    bw = _hac_bandwidth(0)
    assert bw >= 1


def test_hac_bandwidth_positive_horizon():
    bw = _hac_bandwidth(5)
    assert bw >= 5


def test_hac_bandwidth_override():
    assert _hac_bandwidth(10, nw_lags=3) == 3


def test_nw_cov_shape():
    rng = np.random.default_rng(0)
    T, k = 100, 4
    X = rng.standard_normal((T, k))
    resid = rng.standard_normal(T)
    V = _nw_cov(X, resid, nlags=3)
    assert V.shape == (k, k)


def test_nw_cov_symmetric():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 3))
    resid = rng.standard_normal(80)
    V = _nw_cov(X, resid, nlags=4)
    assert_allclose(V, V.T, atol=1e-12)


def test_nw_cov_psd():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 3))
    resid = rng.standard_normal(100)
    V = _nw_cov(X, resid, nlags=5)
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals >= -1e-10)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_init_1d_endog():
    y = np.random.randn(100)
    lp = LocalProjections(y)
    assert lp.endog.shape == (100, 1)


def test_init_2d_endog():
    y = np.random.randn(150, 3)
    lp = LocalProjections(y, shock_idx=1)
    assert lp.endog.shape == (150, 3)


def test_init_default_shock_idx():
    lp = LocalProjections(np.random.randn(100, 2))
    assert lp.shock_idx == [0]


def test_init_multiple_shocks():
    lp = LocalProjections(np.random.randn(100, 3), shock_idx=[0, 2])
    assert lp.shock_idx == [0, 2]


def test_init_int_shock_idx():
    lp = LocalProjections(np.random.randn(100, 3), shock_idx=2)
    assert lp.shock_idx == [2]


def test_init_bad_shock_idx_raises():
    with pytest.raises(ValueError, match="out of range"):
        LocalProjections(np.random.randn(100, 2), shock_idx=5)


def test_init_negative_lags_raises():
    with pytest.raises(ValueError, match="lags"):
        LocalProjections(np.random.randn(100, 2), lags=-1)


def test_init_negative_horizons_raises():
    with pytest.raises(ValueError, match="horizons"):
        LocalProjections(np.random.randn(100, 2), horizons=-1)


def test_init_bad_trend_raises():
    with pytest.raises(ValueError, match="trend"):
        LocalProjections(np.random.randn(100, 2), trend="x")


def test_init_exog_wrong_length_raises():
    with pytest.raises(ValueError, match="exog"):
        LocalProjections(np.random.randn(100, 2),
                         exog=np.random.randn(50, 2))


def test_init_too_few_obs_raises():
    y = np.random.randn(10, 2)
    with pytest.raises(ValueError, match="Not enough"):
        LocalProjections(y, lags=5, horizons=10).fit()


def test_nobs_computed_correctly():
    T, lags, horizons = 200, 4, 8
    lp = LocalProjections(np.random.randn(T, 2), lags=lags, horizons=horizons)
    assert lp._nobs == T - lags - horizons


# ---------------------------------------------------------------------------
# IRF shape and basic properties
# ---------------------------------------------------------------------------


def test_fit_returns_results_object():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=1, horizons=5).fit()
    assert isinstance(res, LocalProjectionsResults)


def test_irfs_shape_univariate():
    y = _make_ar1()
    res = LocalProjections(y, lags=2, horizons=8).fit()
    assert res.irfs.shape == (9, 1, 1)


def test_irfs_shape_multivariate():
    rng = np.random.default_rng(7)
    y = rng.standard_normal((200, 3))
    res = LocalProjections(y, lags=1, horizons=6).fit()
    assert res.irfs.shape == (7, 3, 1)


def test_irfs_shape_multiple_shocks():
    y, _ = _make_var1(T=300, n=2)
    res = LocalProjections(y, shock_idx=[0, 1], lags=1, horizons=5).fit()
    assert res.irfs.shape == (6, 2, 2)


def test_stderr_shape_matches_irfs():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=2, horizons=6).fit()
    assert res.stderr.shape == res.irfs.shape


def test_stderr_nonnegative():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=2, horizons=6).fit()
    assert np.all(res.stderr >= 0)


# ---------------------------------------------------------------------------
# Accuracy: LP-IRF vs. true VAR(1) MA representation
# ---------------------------------------------------------------------------


def test_h0_response_equals_one_by_construction():
    """At horizon 0, response of variable i to its own shock is 1
    because the shock IS the contemporaneous value."""
    y, _ = _make_var1(T=500)
    res = LocalProjections(y, shock_idx=0, lags=2, horizons=0).fit()
    # y0's response to shock in y0 at h=0 must be 1 (OLS on current value).
    assert_allclose(res.irfs[0, 0, 0], 1.0, atol=1e-10)


def test_ar1_irfs_match_phi_powers():
    """LP-IRF for AR(1) at horizon h should be close to phi^h.

    LP estimators are consistent but have higher variance than the indirect
    VAR-based IRF, so we use a large sample and a generous tolerance.
    """
    phi = 0.6
    y = _make_ar1(T=5000, phi=phi, seed=7)
    res = LocalProjections(y, shock_idx=0, lags=1, horizons=6).fit()
    true_irfs = phi ** np.arange(7)
    assert_allclose(res.irfs[:, 0, 0], true_irfs, atol=0.08)


def test_var1_irfs_match_ma_coefficients():
    """LP-IRF at h should be close to A^h[:, shock_idx] for a VAR(1)."""
    y, A = _make_var1(T=2000, seed=123)
    res = LocalProjections(y, shock_idx=0, lags=1, horizons=6).fit()
    Ah = np.eye(2)
    for h in range(7):
        assert_allclose(res.irfs[h, :, 0], Ah[:, 0], atol=0.08)
        Ah = A @ Ah


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def test_conf_int_shape():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=1, horizons=5).fit()
    ci = res.conf_int(alpha=0.05)
    assert ci.shape == (6, 2, 1, 2)


def test_conf_int_lower_le_irf_le_upper():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=1, horizons=5).fit()
    ci = res.conf_int(alpha=0.05)
    assert np.all(ci[..., 0] <= res.irfs)
    assert np.all(res.irfs <= ci[..., 1])


def test_conf_int_width_increases_with_alpha():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=1, horizons=5).fit()
    width_90 = (res.conf_int(alpha=0.10)[..., 1]
                - res.conf_int(alpha=0.10)[..., 0])
    width_95 = (res.conf_int(alpha=0.05)[..., 1]
                - res.conf_int(alpha=0.05)[..., 0])
    assert np.all(width_95 >= width_90)


# ---------------------------------------------------------------------------
# Cumulative effects
# ---------------------------------------------------------------------------


def test_cumulative_effects_shape():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=1, horizons=6).fit()
    ce = res.cumulative_effects()
    assert ce.shape == res.irfs.shape


def test_cumulative_effects_h0_equals_irf_h0():
    y, _ = _make_var1()
    res = LocalProjections(y, lags=1, horizons=6).fit()
    assert_allclose(res.cumulative_effects()[0], res.irfs[0], atol=1e-12)


def test_cumulative_effects_monotone_for_positive_irf():
    """If all IRFs are positive, cumsum must be non-decreasing."""
    # Use an AR(1) with positive phi so all IRFs are positive.
    y = _make_ar1(T=1000, phi=0.7, seed=5)
    res = LocalProjections(y, shock_idx=0, lags=1, horizons=6).fit()
    if np.all(res.irfs >= 0):
        ce = res.cumulative_effects()
        assert np.all(np.diff(ce[:, 0, 0]) >= -1e-10)


# ---------------------------------------------------------------------------
# Trend specifications
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trend", ["n", "c", "ct"])
def test_fit_with_all_trend_types(trend):
    y, _ = _make_var1(T=250)
    res = LocalProjections(y, lags=1, horizons=5, trend=trend).fit()
    assert res.irfs.shape == (6, 2, 1)
    assert np.all(np.isfinite(res.irfs))


# ---------------------------------------------------------------------------
# Exogenous controls
# ---------------------------------------------------------------------------


def test_fit_with_exog():
    rng = np.random.default_rng(10)
    T = 200
    y = rng.standard_normal((T, 2))
    exog = rng.standard_normal((T, 2))
    res = LocalProjections(y, lags=1, horizons=4, exog=exog).fit()
    assert res.irfs.shape == (5, 2, 1)
    assert np.all(np.isfinite(res.irfs))


# ---------------------------------------------------------------------------
# nw_lags override
# ---------------------------------------------------------------------------


def test_nw_lags_override():
    y, _ = _make_var1(T=300)
    res1 = LocalProjections(y, lags=1, horizons=5, nw_lags=3).fit()
    res2 = LocalProjections(y, lags=1, horizons=5, nw_lags=6).fit()
    # IRF point estimates must be identical (same OLS).
    assert_allclose(res1.irfs, res2.irfs, atol=1e-12)
    # SEs will differ.
    assert not np.allclose(res1.stderr, res2.stderr)


# ---------------------------------------------------------------------------
# Zero-lag model
# ---------------------------------------------------------------------------


def test_zero_lags():
    y, _ = _make_var1(T=200)
    res = LocalProjections(y, lags=0, horizons=4).fit()
    assert res.irfs.shape == (5, 2, 1)
    assert np.all(np.isfinite(res.irfs))


# ---------------------------------------------------------------------------
# Pandas input
# ---------------------------------------------------------------------------


def test_pandas_dataframe_input():
    import pandas as pd
    y, _ = _make_var1(T=200)
    df = pd.DataFrame(y, columns=["gdp", "infl"])
    lp = LocalProjections(df, shock_idx=0, lags=1, horizons=4)
    assert lp.endog_names == ["gdp", "infl"]
    assert lp.shock_names == ["gdp"]
    res = lp.fit()
    assert res.irfs.shape == (5, 2, 1)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr():
    y, _ = _make_var1(T=200)
    res = LocalProjections(y, lags=1, horizons=6).fit()
    r = repr(res)
    assert "LocalProjectionsResults" in r
    assert "horizons=6" in r
