"""Tests for OneWayLS pairwise f-test degrees of freedom (gh-9777).

The pairwise comparison between two groups must use df_denom based on the
combined nobs of that pair (nobs_g1 + nobs_g2 - 2*k), not the total sample
size across all groups.  Using the full-model df gives an anticonservative
test — it overstates power and understates p-values.
"""

import numpy as np
import pytest

from statsmodels.sandbox.regression.onewaygls import OneWayLS


def _make_groups(seed=0):
    """Return (y, x, groups) with three groups of different sizes."""
    rng = np.random.RandomState(seed)
    n = [20, 25, 15]
    groups = np.repeat([0, 1, 2], n)
    x = np.column_stack([np.ones(sum(n)), rng.randn(sum(n))])
    y = x[:, 1] + rng.randn(sum(n)) * 0.5
    return y, x, groups


def test_pairwise_df_denom_is_pair_nobs_not_total():
    """Each pairwise df_denom must equal nobs_pair - 2*k, not total - 2*k*G."""
    y, x, groups = _make_groups()
    ols = OneWayLS(y, x, groups=groups)
    _, summarytable = ols.ftest_summary()

    k = x.shape[1]  # number of regressors (2 here)
    counts = np.bincount(groups)

    pair_dfs = {pair: info[2] for pair, info in summarytable if isinstance(pair, tuple)}

    for (g1, g2), df in pair_dfs.items():
        expected_df = counts[g1] + counts[g2] - 2 * k
        assert df == expected_df, (
            f"Pairwise df_denom for ({g1},{g2}) is {df}, expected {expected_df}. "
            "df_denom must be based on the pair nobs, not the full sample."
        )


def test_pairwise_df_strictly_less_than_full_model_df():
    """Pairwise df must be smaller than the joint-model residual df."""
    y, x, groups = _make_groups()
    ols = OneWayLS(y, x, groups=groups)
    ols.fitjoint()
    _, summarytable = ols.ftest_summary()

    full_df = ols.lsjoint.df_resid
    pair_dfs = [info[2] for pair, info in summarytable if isinstance(pair, tuple)]

    for df in pair_dfs:
        assert df < full_df, (
            f"Pairwise df_denom {df} is not less than full-model df_resid {full_df}."
        )


def test_overall_ftest_uses_full_model_df():
    """The 'all' F-test must still use the full joint-model df_denom."""
    y, x, groups = _make_groups()
    ols = OneWayLS(y, x, groups=groups)
    ols.fitjoint()
    _, summarytable = ols.ftest_summary()

    full_df = ols.lsjoint.df_resid
    all_entry = next(info for pair, info in summarytable if pair == "all")
    assert all_entry[2] == full_df


def test_pairwise_pvalue_is_finite_and_in_range():
    """All pairwise p-values must be finite and in [0, 1]."""
    y, x, groups = _make_groups()
    ols = OneWayLS(y, x, groups=groups)
    _, summarytable = ols.ftest_summary()

    for pair, (fval, pval, df_d, df_n) in summarytable:
        if isinstance(pair, tuple):
            assert np.isfinite(pval), f"p-value for {pair} is not finite"
            assert 0.0 <= pval <= 1.0, f"p-value {pval} for {pair} out of [0,1]"
            assert fval >= 0.0, f"F-value {fval} for {pair} must be non-negative"


@pytest.mark.parametrize("het", [False, True])
def test_pairwise_df_correct_for_het(het):
    """Pairwise df fix works regardless of the het= flag."""
    y, x, groups = _make_groups(seed=7)
    ols = OneWayLS(y, x, groups=groups, het=het)
    _, summarytable = ols.ftest_summary()

    k = x.shape[1]
    counts = np.bincount(groups)
    for pair, info in summarytable:
        if isinstance(pair, tuple):
            g1, g2 = pair
            expected_df = counts[g1] + counts[g2] - 2 * k
            assert info[2] == expected_df
