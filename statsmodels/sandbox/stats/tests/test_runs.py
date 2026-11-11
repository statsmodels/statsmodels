"""
Tests corresponding to sandbox.stats.runs
"""
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
import pytest
from scipy import stats

from statsmodels.sandbox.stats.runs import (
    Runs,
    RunsProb,
    TotalRunsProb,
    cochrans_q,
    mcnemar,
    median_test_ksample,
    runstest_1samp,
    runstest_2samp,
    symmetry_bowker,
)


def test_mean_cutoff():
    x = [1] * 5 + [2] * 6 + [3] * 8
    cutoff = "mean"
    expected = (-4.007095978613213, 6.146988816717466e-05)
    results = runstest_1samp(x, cutoff=cutoff, correction=False)
    assert_almost_equal(expected, results)


def test_median_cutoff():
    x = [1] * 5 + [2] * 6 + [3] * 8
    cutoff = "median"
    expected = (-3.944254410803499, 8.004864125547193e-05)
    results = runstest_1samp(x, cutoff=cutoff, correction=False)
    assert_almost_equal(expected, results)


def test_numeric_cutoff():
    x = [1] * 5 + [2] * 6 + [3] * 8
    cutoff = 2
    expected = (-3.944254410803499, 8.004864125547193e-05)
    results = runstest_1samp(x, cutoff=cutoff, correction=False)
    assert_almost_equal(expected, results)


def test_invalid_string_cutoff_raises():
    x = [1] * 5 + [2] * 6 + [3] * 8
    with pytest.raises(ValueError, match="cutoff"):
        runstest_1samp(x, cutoff="not-a-cutoff")


def test_single_run():
    x = [1] * 10
    expected = (-2.8856349, 0.0039062)
    results = runstest_1samp(x)
    assert_almost_equal(expected, results)


class TestRunsClass:
    def test_basic_attributes(self):
        x1 = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])
        r = Runs(x1)
        assert r.n_runs == 11
        assert r.n_pos == (x1 == 1).sum()
        assert_array_equal(r.runs, [3, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1])
        # every run length is accounted for as either pos or neg
        assert r.runs_pos.sum() + r.runs_neg.sum() == len(x1)

    def test_all_same_value_is_single_run(self):
        r = Runs(np.ones(6))
        assert r.n_runs == 1
        assert_array_equal(r.runs, [6])

    def test_runs_test_matches_runstest_1samp(self):
        x1 = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])
        z, p = Runs(x1).runs_test()
        z2, p2 = runstest_1samp(x1, cutoff=0.5)
        assert_allclose((z, p), (z2, p2))


def test_runstest_2samp_no_ties():
    rs = np.random.RandomState(0)
    x = rs.normal(size=20)
    y = rs.normal(loc=3, size=20)
    z, p = runstest_2samp(x, y)
    # very separated samples should show up as far too few runs (large
    # negative z, small p-value)
    assert z < 0
    assert p < 0.01


def test_runstest_2samp_with_groups_argument():
    rs = np.random.RandomState(0)
    x = rs.normal(size=15)
    y = rs.normal(size=15)
    combined = np.concatenate([x, y])
    groups = np.concatenate([np.zeros(15), np.ones(15)])
    z1, p1 = runstest_2samp(combined, groups=groups)
    z2, p2 = runstest_2samp(x, y)
    assert_allclose((z1, p1), (z2, p2))


def test_runstest_2samp_requires_y_or_groups():
    with pytest.raises(ValueError, match="either y or groups"):
        runstest_2samp(np.arange(10))


def test_runstest_2samp_wrong_number_of_groups_raises():
    with pytest.raises(ValueError, match="not exactly two groups"):
        runstest_2samp(
            np.arange(9), groups=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )


def test_runstest_2samp_with_ties(capsys):
    # forces the tie-breaking branch (adds/subtracts a small eps and takes
    # the max-p-value result of the two perturbations); needs float input,
    # see test_runstest_2samp_with_ties_integer_input_raises below
    x = np.array([1, 2, 2, 3, 3, 4], dtype=float)
    y = np.array([2, 3, 3, 4, 5, 5], dtype=float)
    z, p = runstest_2samp(x, y)
    assert np.isfinite(z)
    assert 0 <= p <= 1
    captured = capsys.readouterr()
    assert "ties detected" in captured.out


@pytest.mark.xfail(
    reason=(
        "BUG: runstest_2samp's tie-breaking branch does "
        "`xx = x.copy(); xx[groups == gruni[0]] += eps` where eps is a "
        "float. If the input arrays are integer-dtyped (a reasonable "
        "thing to pass for count/discrete data, which is exactly when "
        "ties are most likely), `xx` is also integer-dtyped and the "
        "in-place += with a float eps raises UFuncTypeError under numpy's "
        "same_kind casting rule. The same call with float-dtyped input "
        "(otherwise identical) works fine, see test_runstest_2samp_with_ties."
    ),
    raises=TypeError,
    strict=True,
)
def test_runstest_2samp_with_ties_integer_input_raises():
    x = np.array([1, 2, 2, 3, 3, 4])
    y = np.array([2, 3, 3, 4, 5, 5])
    runstest_2samp(x, y)


class TestTotalRunsProb:
    def test_pdf_sums_to_one_over_full_support(self):
        trp = TotalRunsProb(7, 9)
        support = np.arange(2, trp.n + 1)
        assert_allclose(trp.pdf(support).sum(), 1.0, atol=1e-10)

    def test_cdf_at_max_is_one(self):
        trp = TotalRunsProb(7, 9)
        assert_allclose(trp.cdf(trp.n), 1.0, atol=1e-10)

    def test_cdf_matches_cumulative_pdf(self):
        trp = TotalRunsProb(7, 9)
        r = 11
        support = np.arange(2, r + 1)
        assert_allclose(trp.cdf(r), trp.pdf(support).sum(), atol=1e-10)


def test_runsprob_pdf_matches_reference_values():
    # reference values ported from this module's own inert (not run as a
    # doctest) triple-quoted example block; loose tolerance since these
    # are sums of many comb()-based terms and differ slightly at the 1e-6
    # level from the scipy version originally used to compute them
    vals = [
        np.sum([RunsProb().pdf(xi, k, 16, 10 / 16.0) for xi in range(16)])
        for k in range(3)
    ]
    expected = [0.99999332193894064, 0.99999999999999367, 1.0]
    assert_allclose(vals, expected, atol=1e-4)


class TestMedianTestKsample:
    def test_statistic_matches_scipy(self):
        rs = np.random.RandomState(0)
        x = rs.randn(60)
        groups = rs.randint(0, 3, 60)
        (stat, _), _, _ = median_test_ksample(x, groups)
        xli = [x[groups == g] for g in np.unique(groups)]
        expected_stat, _, _, _ = stats.median_test(*xli)
        assert_allclose(stat, expected_stat)

    @pytest.mark.xfail(
        reason=(
            "BUG: median_test_ksample hardcodes ddof=1 in its "
            "stats.chisquare(table.ravel(), expected.ravel(), ddof=1) "
            "call. For a (2, ngroups) contingency table, the correct "
            "degrees of freedom is (2-1)*(ngroups-1) = ngroups-1, which "
            "requires ddof=ngroups (since chisquare's dof = ncells-1-ddof "
            "and ncells=2*ngroups here), not the hardcoded 1. The reported "
            "chi-square statistic matches scipy.stats.median_test exactly "
            "(see test_statistic_matches_scipy), but with the wrong ddof "
            "the p-value does not -- e.g., for this dataset the sandbox "
            "version reports p=0.300 where the correct value (matching "
            "scipy.stats.median_test) is p=0.087."
        ),
        raises=AssertionError,
        strict=True,
    )
    def test_pvalue_matches_scipy(self):
        rs = np.random.RandomState(0)
        x = rs.randn(60)
        groups = rs.randint(0, 3, 60)
        (_, pval), _, _ = median_test_ksample(x, groups)
        xli = [x[groups == g] for g in np.unique(groups)]
        _, expected_pval, _, _ = stats.median_test(*xli)
        assert_allclose(pval, expected_pval)

    def test_pvalue_matches_scipy_with_correct_ddof(self):
        # demonstrates the fix: ddof should be ngroups, not 1
        rs = np.random.RandomState(0)
        x = rs.randn(60)
        groups = rs.randint(0, 3, 60)
        ngroups = len(np.unique(groups))
        (stat, _), table, expected = median_test_ksample(x, groups)
        corrected = stats.chisquare(table.ravel(), expected.ravel(), ddof=ngroups)
        xli = [x[groups == g] for g in np.unique(groups)]
        expected_stat, expected_pval, _, _ = stats.median_test(*xli)
        assert_allclose(corrected.statistic, expected_stat)
        assert_allclose(corrected.pvalue, expected_pval)


def test_cochrans_q_matches_current_implementation():
    from statsmodels.stats.contingency_tables import cochrans_q as cochrans_q_new

    rs = np.random.RandomState(0)
    data = rs.randint(0, 2, (50, 4))
    with pytest.warns(FutureWarning, match="Deprecated"):
        stat, pval = cochrans_q(data)
    expected = cochrans_q_new(data)
    assert_allclose((stat, pval), (expected.statistic, expected.pvalue))


def test_mcnemar_from_table_wrong_shape_raises():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(ValueError, match="table needs to be 2 by 2"):
            mcnemar(np.eye(3))


def test_symmetry_bowker_symmetric_table_gives_zero_statistic():
    table = np.array([[5, 3, 2], [3, 6, 1], [2, 1, 4]])
    with pytest.warns(FutureWarning, match="Deprecated"):
        stat, pval, df = symmetry_bowker(table)
    assert_allclose(stat, 0.0, atol=1e-10)
    assert_allclose(pval, 1.0)
    assert df == 3


def test_symmetry_bowker_asymmetric_table_matches_manual_formula():
    table = np.array([[5, 8, 2], [1, 6, 1], [2, 9, 4]])
    with pytest.warns(FutureWarning, match="Deprecated"):
        stat, pval, df = symmetry_bowker(table)

    upp_idx = np.triu_indices(3, 1)
    tril = table.T[upp_idx]
    triu = table[upp_idx]
    expected_stat = ((tril - triu) ** 2 / (tril + triu)).sum()
    assert_allclose(stat, expected_stat)
    assert df == 3
    assert_allclose(pval, stats.chi2.sf(expected_stat, df))


def test_symmetry_bowker_requires_square_table():
    with pytest.warns(FutureWarning, match="Deprecated"):
        with pytest.raises(ValueError, match="table needs to be square"):
            symmetry_bowker(np.ones((2, 3)))
