"""
Array valued arguments of functions that broadcast over them

The power, sample size, confidence interval and equivalence functions below
accept arrays for the arguments listed in `CASES`, and the result for an
array is the array of the results for the elements. The checks of the
arguments have to keep that working and still reject invalid elements.
"""
from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.stats import power as smpower
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multivariate import (
    test_cov_blockdiagonal as cov_blockdiagonal,
    test_cov_diagonal as cov_diagonal,
    test_cov_spherical as cov_spherical,
)
from statsmodels.stats.nonparametric import rank_compare_2indep
from statsmodels.stats.oneway import equivalence_oneway
from statsmodels.stats.proportion import (
    binom_test,
    binom_tost_reject_interval,
    power_binom_tost,
    power_proportions_2indep,
    power_ztost_prop,
    samplesize_proportions_2indep_onetail,
)
from statsmodels.stats.rates import (
    confint_poisson,
    confint_quantile_poisson,
    nonequivalence_poisson_2indep,
    power_equivalence_neginb_2indep,
    power_equivalence_poisson_2indep,
    power_negbin_ratio_2indep,
    power_poisson_diff_2indep,
    power_poisson_ratio_2indep,
    tolerance_int_poisson,
    tost_poisson_2indep,
)
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW, zconfint, ztest

NAN = np.nan
_rs = np.random.default_rng(987654)
X1 = _rs.standard_normal(40)
X2 = _rs.standard_normal(25) + 0.3
GROUPS = [_rs.standard_normal(30), _rs.standard_normal(30) + 0.1, _rs.standard_normal(30)]
COV = np.eye(3) + 0.2
TABLE = np.array([[10, 5], [4, 12]])
ALPHA = (0.05, 0.1)
BAD_ALPHA = ([0.05, 1.5], [0.05, 0.0], [0.05, 1.0], [0.05, NAN], [-0.1, 0.05])
MATCH_ALPHA = "alpha must be in the range"


def _pvalue(func, **kwds):
    return func(**kwds).pvalue


def _tost_poisson(**kwds):
    return tost_poisson_2indep(**kwds).pvalue


def _nonequiv_poisson(**kwds):
    return nonequivalence_poisson_2indep(**kwds).pvalue


def _equiv_oneway(**kwds):
    return equivalence_oneway(GROUPS, **kwds).pvalue


# (id, callable, fixed arguments, argument, two valid values, invalid arrays, match)
CASES = [
    ("confint_poisson-alpha", partial(confint_poisson, method="exact-c"), dict(count=10, exposure=2.0), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("tolerance_int_poisson-alpha", partial(tolerance_int_poisson, method="exact-c"), dict(count=10, exposure=2.0, prob=0.9), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("tolerance_int_poisson-prob", partial(tolerance_int_poisson, method="exact-c"), dict(count=10, exposure=2.0), "prob", (0.9, 0.95), ([0.9, 1.0], [0.9, NAN], [0.0, 0.9]), "prob must be in the range"),
    ("confint_quantile_poisson-alpha", partial(confint_quantile_poisson, method="exact-c"), dict(count=10, exposure=2.0, prob=0.5), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("confint_quantile_poisson-prob", partial(confint_quantile_poisson, method="exact-c"), dict(count=10, exposure=2.0), "prob", (0.5, 0.6), ([0.5, 1.0], [0.5, NAN], [0.0, 0.5]), "prob must be in the range"),
    ("tost_poisson_2indep-low", _tost_poisson, dict(count1=10, exposure1=100, count2=12, exposure2=100, upp=1.5), "low", (0.5, 0.6), ([0.5, 2.0], [0.5, NAN]), "equivalence interval"),
    ("tost_poisson_2indep-upp", _tost_poisson, dict(count1=10, exposure1=100, count2=12, exposure2=100, low=0.5), "upp", (1.5, 1.6), ([1.5, 0.2], [1.5, NAN]), "equivalence interval"),
    ("nonequivalence_poisson_2indep-low", _nonequiv_poisson, dict(count1=10, exposure1=100, count2=12, exposure2=100, upp=1.5), "low", (0.5, 0.6), ([0.5, 2.0], [0.5, NAN]), "equivalence interval"),
    ("nonequivalence_poisson_2indep-upp", _nonequiv_poisson, dict(count1=10, exposure1=100, count2=12, exposure2=100, low=0.5), "upp", (1.5, 1.6), ([1.5, 0.2], [1.5, NAN]), "equivalence interval"),
    ("power_poisson_ratio_2indep-alpha", partial(power_poisson_ratio_2indep, value=1, alternative="two-sided", return_results=False), dict(rate1=0.1, rate2=0.15, nobs1=300), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_poisson_ratio_2indep-dispersion", partial(power_poisson_ratio_2indep, value=1, alternative="two-sided", return_results=False), dict(rate1=0.1, rate2=0.15, nobs1=300), "dispersion", (1.0, 1.5), ([1.0, -0.5], [1.0, NAN]), "dispersion must be non-negative"),
    ("power_poisson_diff_2indep-alpha", partial(power_poisson_diff_2indep, return_results=False), dict(rate1=0.1, rate2=0.15, nobs1=300), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_negbin_ratio_2indep-alpha", partial(power_negbin_ratio_2indep, dispersion=0.5, return_results=False), dict(rate1=0.1, rate2=0.15, nobs1=500), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_negbin_ratio_2indep-dispersion", partial(power_negbin_ratio_2indep, return_results=False), dict(rate1=0.1, rate2=0.15, nobs1=500), "dispersion", (0.2, 0.5), ([0.2, -0.5], [0.2, NAN]), "dispersion must be non-negative"),
    ("power_equivalence_poisson_2indep-alpha", power_equivalence_poisson_2indep, dict(rate1=0.1, rate2=0.1, nobs1=1000, low=0.8, upp=1.25), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_equivalence_poisson_2indep-low", power_equivalence_poisson_2indep, dict(rate1=0.1, rate2=0.1, nobs1=1000, upp=1.25), "low", (0.8, 0.85), ([0.8, 2.0], [0.8, NAN]), "equivalence interval"),
    ("power_equivalence_poisson_2indep-upp", power_equivalence_poisson_2indep, dict(rate1=0.1, rate2=0.1, nobs1=1000, low=0.8), "upp", (1.25, 1.3), ([1.25, 0.5], [1.25, NAN]), "equivalence interval"),
    ("power_equivalence_neginb_2indep-alpha", partial(power_equivalence_neginb_2indep, dispersion=0.5), dict(rate1=0.1, rate2=0.1, nobs1=1000, low=0.8, upp=1.25), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_equivalence_neginb_2indep-dispersion", power_equivalence_neginb_2indep, dict(rate1=0.1, rate2=0.1, nobs1=1000, low=0.8, upp=1.25), "dispersion", (0.5, 0.7), ([0.5, -0.5], [0.5, NAN]), "dispersion must be non-negative"),
    ("power_equivalence_neginb_2indep-low", partial(power_equivalence_neginb_2indep, dispersion=0.5), dict(rate1=0.1, rate2=0.1, nobs1=1000, upp=1.25), "low", (0.8, 0.85), ([0.8, 2.0], [0.8, NAN]), "equivalence interval"),
    ("power_equivalence_neginb_2indep-upp", partial(power_equivalence_neginb_2indep, dispersion=0.5), dict(rate1=0.1, rate2=0.1, nobs1=1000, low=0.8), "upp", (1.25, 1.3), ([1.25, 0.5], [1.25, NAN]), "equivalence interval"),
    ("samplesize_proportions_2indep_onetail-alpha", samplesize_proportions_2indep_onetail, dict(diff=0.1, prop2=0.2, power=0.8), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("samplesize_proportions_2indep_onetail-power", samplesize_proportions_2indep_onetail, dict(diff=0.1, prop2=0.2), "power", (0.8, 0.9), ([0.8, 1.0], [0.8, 0.0], [0.8, NAN]), "power must be in the range"),
    ("samplesize_proportions_2indep_onetail-diff", samplesize_proportions_2indep_onetail, dict(prop2=0.2, power=0.8), "diff", (0.1, 0.15), ([0.1, 0.9], [0.1, -0.3], [0.1, NAN]), "diff must keep"),
    ("samplesize_proportions_2indep_onetail-prop2", samplesize_proportions_2indep_onetail, dict(diff=0.1, power=0.8), "prop2", (0.2, 0.3), ([0.2, 1.2], [0.2, -0.1], [0.2, NAN]), "prop2 must be in the range"),
    ("power_proportions_2indep-diff", partial(power_proportions_2indep, return_results=False), dict(prop2=0.2, nobs1=100), "diff", (0.1, 0.15), ([0.1, 0.9], [0.1, -0.3], [0.1, NAN]), "diff must keep"),
    ("power_proportions_2indep-prop2", partial(power_proportions_2indep, return_results=False), dict(diff=0.1, nobs1=100), "prop2", (0.2, 0.3), ([0.2, 1.2], [0.2, -0.1], [0.2, NAN]), "prop2 must be in the range"),
    ("binom_test-prop", partial(binom_test, alternative="larger"), dict(count=3, nobs=10), "prop", (0.3, 0.5), ([0.3, 1.5], [0.3, -0.1], [0.3, NAN]), "p must be in range"),
    ("binom_test-nobs", partial(binom_test, alternative="smaller"), dict(count=3, prop=0.5), "nobs", (10, 12), ([10, 0], [10, -4], [10, NAN]), "nobs must be positive"),
    ("binom_test-count", partial(binom_test, alternative="larger"), dict(nobs=10, prop=0.5), "count", (3, 4), ([3, -1],), "count must be non-negative"),
    ("binom_test-count-exceeds-nobs", partial(binom_test, alternative="smaller"), dict(nobs=10, prop=0.5), "count", (3, 4), ([3, 11],), "count must not exceed nobs"),
    ("binom_tost_reject_interval-alpha", binom_tost_reject_interval, dict(low=0.4, upp=0.6, nobs=100), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("binom_tost_reject_interval-low", binom_tost_reject_interval, dict(upp=0.6, nobs=100, alpha=0.05), "low", (0.4, 0.45), ([0.4, 0.7], [0.4, NAN]), "equivalence interval"),
    ("binom_tost_reject_interval-upp", binom_tost_reject_interval, dict(low=0.4, nobs=100, alpha=0.05), "upp", (0.6, 0.65), ([0.6, 0.3], [0.6, NAN]), "equivalence interval"),
    ("power_binom_tost-alpha", power_binom_tost, dict(low=0.4, upp=0.6, nobs=100, p_alt=0.5), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_binom_tost-low", power_binom_tost, dict(upp=0.6, nobs=100, p_alt=0.5, alpha=0.05), "low", (0.4, 0.45), ([0.4, 0.7], [0.4, NAN]), "equivalence interval"),
    ("power_binom_tost-upp", power_binom_tost, dict(low=0.4, nobs=100, p_alt=0.5, alpha=0.05), "upp", (0.6, 0.65), ([0.6, 0.3], [0.6, NAN]), "equivalence interval"),
    ("power_ztost_prop-alpha", power_ztost_prop, dict(low=0.4, upp=0.6, nobs=200, p_alt=0.5), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("power_ztost_prop-low", power_ztost_prop, dict(upp=0.6, nobs=200, p_alt=0.5, alpha=0.05), "low", (0.4, 0.45), ([0.4, 0.7], [0.4, NAN]), "equivalence interval"),
    ("TTestPower-alpha", lambda **kw: smpower.TTestPower().power(**kw), dict(effect_size=0.5, nobs=20), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("NormalIndPower-alpha", lambda **kw: smpower.NormalIndPower().power(**kw), dict(effect_size=0.5, nobs1=20), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("FTestPower-alpha", lambda **kw: smpower.FTestPower().power(**kw), dict(effect_size=0.3, df_num=3, df_denom=40), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("FTestPowerF2-alpha", lambda **kw: smpower.FTestPowerF2().power(**kw), dict(effect_size=0.1, df_num=3, df_denom=40), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("log_oddsratio_confint-alpha", lambda **kw: Table2x2(TABLE).log_oddsratio_confint(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("oddsratio_confint-alpha", lambda **kw: Table2x2(TABLE).oddsratio_confint(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("log_riskratio_confint-alpha", lambda **kw: Table2x2(TABLE).log_riskratio_confint(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("riskratio_confint-alpha", lambda **kw: Table2x2(TABLE).riskratio_confint(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("zconfint-alpha", partial(zconfint, X1), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("tconfint_mean-alpha", lambda **kw: DescrStatsW(X1).tconfint_mean(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("tconfint_mean-alpha-larger", lambda **kw: DescrStatsW(X1).tconfint_mean(alternative="larger", **kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("tconfint_mean-alpha-smaller", lambda **kw: DescrStatsW(X1).tconfint_mean(alternative="smaller", **kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("tconfint_diff-alpha", lambda **kw: CompareMeans(DescrStatsW(X1), DescrStatsW(X2)).tconfint_diff(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("rank_compare_2indep-conf_int-alpha", lambda **kw: rank_compare_2indep(X1, X2).conf_int(**kw), dict(), "alpha", ALPHA, BAD_ALPHA, MATCH_ALPHA),
    ("zconfint-ddof", partial(zconfint, X1), dict(), "ddof", (1.0, 0.0), ([1.0, -1.0], [1.0, NAN]), "ddof must be non-negative"),
    ("ztest-ddof", lambda **kw: ztest(X1, **kw)[1], dict(), "ddof", (1.0, 0.0), ([1.0, -1.0], [1.0, NAN]), "ddof must be non-negative"),
    ("equivalence_oneway-equiv_margin", _equiv_oneway, dict(), "equiv_margin", (0.5, 0.8), ([0.5, 0.0], [0.5, -1.0], [0.5, NAN]), "equiv_margin must be positive"),
    ("test_cov_spherical-nobs", lambda **kw: cov_spherical(COV, **kw).pvalue, dict(), "nobs", (50, 80), ([50, 0], [50, -5], [50, NAN]), "nobs must be positive"),
    ("test_cov_diagonal-nobs", lambda **kw: cov_diagonal(COV, **kw).pvalue, dict(), "nobs", (50, 80), ([50, 0], [50, -5], [50, NAN]), "nobs must be positive"),
    ("test_cov_blockdiagonal-nobs", lambda **kw: cov_blockdiagonal(COV, block_len=[1, 2], **kw).pvalue, dict(), "nobs", (50, 80), ([50, 0], [50, -5], [50, NAN]), "nobs must be positive"),
]
IDS = [c[0] for c in CASES]


def _leaves(result):
    """float arrays in a (nested) tuple of results"""
    if isinstance(result, (tuple, list)):
        return [leaf for item in result for leaf in _leaves(item)]
    return [np.asarray(result, dtype=float)]


@pytest.mark.parametrize("func, fixed, arg, values", [c[1:5] for c in CASES], ids=IDS)
def test_array_argument_is_elementwise(func, fixed, arg, values):
    expected = [_leaves(func(**fixed, **{arg: value})) for value in values]
    result = _leaves(func(**fixed, **{arg: np.array(values)}))
    assert len(result) == len(expected[0])
    for k, leaf in enumerate(result):
        desired = np.stack([e[k] for e in expected])
        assert_allclose(np.broadcast_to(leaf, desired.shape), desired, rtol=1e-10)


@pytest.mark.parametrize("func, fixed, arg, bad, match", [c[1:4] + c[5:] for c in CASES], ids=IDS)
def test_array_argument_invalid_element(func, fixed, arg, bad, match):
    # one invalid element is enough, including NaN, for arrays and Series
    for values in bad:
        with pytest.raises(ValueError, match=match):
            func(**fixed, **{arg: np.array(values)})
    with pytest.raises(ValueError, match=match):
        func(**fixed, **{arg: pd.Series(bad[0])})
