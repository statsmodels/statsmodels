"""
Generic tests for statsmodels.stats.base.LimitedIterationMixin and every
dataclass across the codebase that uses it.

Each of these classes is a frozen, slotted dataclass whose ``__iter__``,
``__getitem__`` and ``__len__`` are restricted to a documented subset of
fields (``_iter_fields``), unlike a plain ``NamedTuple`` where every field
participates. This module builds a minimal, valid instance of every such
class and checks that the tuple protocol matches ``_iter_fields`` exactly,
that indexing out of that range raises ``IndexError``, and that the
dataclass is genuinely frozen and slotted.
"""
import dataclasses

import numpy as np
import pytest

from statsmodels.base._parameter_inference import ScoreTestResult
from statsmodels.discrete._diagnostics_count import (
    ChisquareProbResult,
    DispersionResults,
    ZeroinflationJHResult,
    ZeroModificationTestResult,
)
from statsmodels.stats.base import LimitedIterationMixin
from statsmodels.stats.diagnostic_gen import ChisquareBinningResult
from statsmodels.stats.effect_size import NoncentralityChisquareResult
from statsmodels.stats.meta_analysis import HomogeneityTestResult
from statsmodels.stats.multivariate import (
    CovOnewayResult,
    CovTestResult,
    HotellingResult,
)
from statsmodels.stats.nonparametric import (
    ProbSuperiorResult,
    RankCompareResult,
    TostProbSuperiorResult,
)
from statsmodels.stats.oneway import (
    AnovaResult,
    EquivalenceOnewayResult,
    ScaleAnovaResult,
    ScaleEquivalenceResult,
)
from statsmodels.stats.proportion import (
    Proportions2indepTestResult,
    ScoreTestProportionsResult,
    TostProportionsResult,
)
from statsmodels.stats.rates import (
    NonequivalencePoissonResult,
    PoissonTest2indepResult,
    PoissonTestResult,
    PowerDiffResult,
    PowerEquivalenceResult,
    PowerNegbinRatioResult,
    PowerRatioResult,
    TostPoissonResult,
)
from statsmodels.tsa.stattools import (
    ADFullerResult,
    DieboldMarianoResult,
    KPSSResult,
    RURResult,
)

# ---------------------------------------------------------------------
# Nested helper instances needed as fields of some of the classes below.
# ---------------------------------------------------------------------
_p2i = PoissonTest2indepResult(
    statistic=1.1,
    pvalue=0.2,
    distribution="normal",
    compare="ratio",
    method="score",
    alternative="two-sided",
    rates=(2.0, 1.5),
    ratio=1.3,
    diff=0.5,
    value=1.0,
    rates_cmle=None,
)
_prop2i = Proportions2indepTestResult(
    statistic=1.1,
    pvalue=0.2,
    compare="diff",
    method="score",
    diff=0.1,
    ratio=1.2,
    odds_ratio=1.3,
    variance=0.01,
    alternative="two-sided",
    value=0.0,
    prop1_null=None,
    prop2_null=None,
)
_probsup = ProbSuperiorResult(
    statistic=0.5, pvalue=0.3, df=None, distribution="normal"
)
_nc_chi2 = NoncentralityChisquareResult(
    nc=1.0,
    confint=(0.5, 1.5),
    nc_umvue=1.0,
    nc_lzd=1.0,
    nc_krs=1.0,
    nc_median=1.0,
    name="Noncentrality for chisquare-distributed random variable",
)


# Each entry: (class, kwargs used to build a valid instance).
CASES = [
    (ScoreTestResult, dict(
        statistic=1.0, pvalue=0.1, distribution="chi2", k_constraint=2,
    )),
    (ChisquareProbResult, dict(
        statistic=1.0, pvalue=0.1, df=2, diff1=np.array([0.1, -0.1]),
        res_aux=None, distribution="chi2",
    )),
    (DispersionResults, dict(
        statistic=np.array([1.0, 2.0]), pvalue=np.array([0.1, 0.2]),
        method=["a", "b"], alternative=["two-sided", "two-sided"], name="x",
    )),
    (ZeroinflationJHResult, dict(
        statistic=1.0, pvalue=0.1, df=1, rank_score=1, distribution="chi2",
    )),
    (ZeroModificationTestResult, dict(
        statistic=1.0, pvalue=0.1, pvalue_smaller=0.2, pvalue_larger=0.8,
        chi2=1.0, pvalue_chi2=0.1, df_chi2=1, distribution="normal",
    )),
    (HomogeneityTestResult, dict(
        statistic=1.0, pvalue=0.1, df=2.0, distribution="chi2",
    )),
    (PoissonTestResult, dict(
        statistic=1.0, pvalue=0.1, distribution="normal", method="score",
        alternative="two-sided", rate=0.5, nobs=20.0,
    )),
    (PoissonTest2indepResult, dict(
        statistic=1.0, pvalue=0.1, distribution="normal", compare="ratio",
        method="score", alternative="two-sided", rates=(2.0, 1.5),
        ratio=1.3, diff=0.5, value=1.0, rates_cmle=None,
    )),
    (TostPoissonResult, dict(
        statistic=1.0, pvalue=0.1, method="score", compare="ratio",
        equiv_limits=(0.5, 2.0), results_larger=_p2i, results_smaller=_p2i,
        title="tost",
    )),
    (NonequivalencePoissonResult, dict(
        statistic=1.0, pvalue=0.1, method="score", results_larger=_p2i,
        results_smaller=_p2i, title="nonequiv",
    )),
    (PowerRatioResult, dict(
        power=0.8, p_pooled=None, std_null=0.1, std_alt=0.1, nobs_1=20.0,
        nobs_2=20.0, nobs_ratio=1.0, alpha=0.05,
    )),
    (PowerEquivalenceResult, dict(
        power=0.8, power_margins=np.array([0.1, 0.2]), std_null_low=0.1,
        std_null_upp=0.1, std_alt=0.1, nobs_1=20.0, nobs_2=20.0,
        nobs_ratio=1.0, alpha=0.05,
    )),
    (PowerDiffResult, dict(
        power=0.8, rates_alt=(2.0, 1.5), std_null=0.1, std_alt=0.1,
        nobs_1=20.0, nobs_2=20.0, nobs_ratio=1.0, alpha=0.05,
    )),
    (PowerNegbinRatioResult, dict(
        power=0.8, std_null=0.1, std_alt=0.1, nobs_1=20.0, nobs_2=20.0,
        nobs_ratio=1.0, alpha=0.05,
    )),
    (ScoreTestProportionsResult, dict(
        statistic=1.0, pvalue=0.1, compare="diff", method="score",
        variance=0.01, alternative="two-sided", prop1_null=0.5, prop2_null=0.5,
    )),
    (Proportions2indepTestResult, dict(
        statistic=1.0, pvalue=0.1, compare="diff", method="score", diff=0.1,
        ratio=1.2, odds_ratio=1.3, variance=0.01, alternative="two-sided",
        value=0.0, prop1_null=None, prop2_null=None,
    )),
    (TostProportionsResult, dict(
        statistic=1.0, pvalue=0.1, compare="diff", method="score",
        results_larger=_prop2i, results_smaller=_prop2i, title="tost",
    )),
    (AnovaResult, dict(
        statistic=1.0, pvalue=0.1, df=(2.0, 30.0), df_num=2.0, df_denom=30.0,
        nobs_total=33.0, n_groups=3, means=np.array([1.0, 2.0, 3.0]),
        nobs=np.array([10, 11, 12]), vars_=np.array([1.0, 1.0, 1.0]),
        use_var="unequal", welch_correction=True,
    )),
    (EquivalenceOnewayResult, dict(
        statistic=1.0, pvalue=0.1, effectsize=0.1, crit_f=3.0, crit_es=0.2,
        reject=True, power_zero=0.5, df=(2, 30), f_stat=1.0,
        type_effectsize="eta2",
    )),
    (ScaleAnovaResult, dict(
        statistic=1.0, pvalue=0.1, df=(2.0, 30.0), df_num=2.0, df_denom=30.0,
        nobs_total=33.0, n_groups=3, means=np.array([1.0, 2.0, 3.0]),
        nobs=np.array([10, 11, 12]), vars_=np.array([1.0, 1.0, 1.0]),
        use_var="unequal", welch_correction=True, data_transformed=[1, 2, 3],
    )),
    (ScaleEquivalenceResult, dict(
        statistic=1.0, pvalue=0.1, effectsize=0.1, crit_f=3.0, crit_es=0.2,
        reject=True, power_zero=0.5, df=(2, 30), f_stat=1.0,
        type_effectsize="eta2", x_transformed=[1, 2, 3],
    )),
    (ProbSuperiorResult, dict(
        statistic=1.0, pvalue=0.1, df=None, distribution="normal",
    )),
    (TostProbSuperiorResult, dict(
        statistic=1.0, pvalue=0.1, results_larger=_probsup,
        results_smaller=_probsup, title="tost",
    )),
    (RankCompareResult, dict(
        statistic=1.0, pvalue=0.1, s1=1.0, s2=1.0, var1=1.0, var2=1.0,
        var=1.0, var_prob=1.0, nobs_1=10, nobs_2=10, nobs=20, mean1=1.0,
        mean2=1.0, prob1=0.5, prob2=0.5, somersd1=0.0, somersd2=0.0,
        df=None, use_t=False,
    )),
    (ChisquareBinningResult, dict(
        statistic=1.0, pvalue=0.1, df=2, freqs=np.array([1.0, 2.0]),
        probs=np.array([0.5, 0.5]), noncentrality=_nc_chi2,
        resid_pearson=np.array([0.1, -0.1]),
        chi2_stat_groups=np.array([0.5, 0.5]), indices=[[0], [1]],
    )),
    (HotellingResult, dict(
        statistic=1.0, pvalue=0.1, df=(3, 47), t2=4.0, distribution="F",
    )),
    (CovTestResult, dict(
        statistic=1.0, pvalue=0.1, df=6.0, distribution="chi2",
        null="equal value", cov_null=None,
    )),
    (CovOnewayResult, dict(
        statistic=1.0, pvalue=0.1, statistic_base=1.0, statistic_chi2=1.0,
        pvalue_chi2=0.1, df_chi2=2.0, distribution_chi2="chi2",
        statistic_f=1.0, pvalue_f=0.1, df_f=(2, 30), distribution_f="F",
    )),
    (ADFullerResult, dict(
        statistic=-2.0, pvalue=0.3, lags=1, nobs=100,
        critical_values={"1%": -3.5, "5%": -2.9, "10%": -2.6}, icbest=None,
        resstore=None,
    )),
    (DieboldMarianoResult, dict(
        statistic=1.0, pvalue=0.1, lags=1, harvey_adj_factor=None,
    )),
    (KPSSResult, dict(
        statistic=0.3, pvalue=0.1, lags=3,
        critical_values={"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
        resstore=None,
    )),
    (RURResult, dict(
        statistic=1.5, pvalue=0.5,
        critical_values={"10%": 1.35, "5%": 1.21, "2.5%": 1.10, "1%": 0.98},
        resstore=None,
    )),
]


def _values_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


@pytest.mark.parametrize("cls, kwargs", CASES, ids=[c[0].__name__ for c in CASES])
def test_limited_iteration_mixin_protocol(cls, kwargs):
    assert issubclass(cls, LimitedIterationMixin)
    instance = cls(**kwargs)

    iter_fields = instance._iter_fields
    expected = tuple(kwargs[name] for name in iter_fields)

    # __iter__ / unpacking / list()
    values = tuple(instance)
    assert len(values) == len(expected)
    for got, exp in zip(values, expected, strict=True):
        assert _values_equal(got, exp)
    assert len(list(instance)) == len(iter_fields)

    # __len__
    assert len(instance) == len(iter_fields)

    # __getitem__ (int)
    for i, exp in enumerate(expected):
        assert _values_equal(instance[i], exp)
    # __getitem__ out of range
    with pytest.raises(IndexError):
        instance[len(iter_fields)]
    # __getitem__ (slice)
    assert len(instance[:]) == len(iter_fields)

    # unpacking assignment for the common 1- and 2-field cases
    if len(iter_fields) == 2:
        a, b = instance
        assert _values_equal(a, expected[0])
        assert _values_equal(b, expected[1])
    elif len(iter_fields) == 1:
        (a,) = instance
        assert _values_equal(a, expected[0])

    # every constructor kwarg is reachable via attribute access, not just
    # the ones that participate in iteration
    for name, val in kwargs.items():
        assert _values_equal(getattr(instance, name), val)

    # frozen: assigning to any field raises
    first_field = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, first_field, getattr(instance, first_field))

    # slots: no instance __dict__
    assert not hasattr(instance, "__dict__")

    # repr never raises, and mentions the class name
    assert type(instance).__name__ in repr(instance)


_POWER_CASES = [
    c for c in CASES
    if c[0] in (PowerRatioResult, PowerEquivalenceResult, PowerDiffResult,
                PowerNegbinRatioResult)
]


@pytest.mark.parametrize(
    "cls, kwargs", _POWER_CASES, ids=[c[0].__name__ for c in _POWER_CASES]
)
def test_power_results_array_protocol(cls, kwargs):
    # The four Power*Result classes additionally define __array__ so that
    # np.asarray()/assert_allclose() treat them like the scalar `power`.
    instance = cls(**kwargs)
    arr = np.asarray(instance)
    assert arr.shape == ()
    assert float(arr) == kwargs["power"]


def test_rank_compare_result_methods_still_work():
    # RankCompareResult is the one LimitedIterationMixin class (besides
    # DispersionResults) with real methods beyond the dataclass fields;
    # make sure restricting __iter__/__getitem__/__len__ didn't break them.
    kwargs = dict(CASES[[c[0] for c in CASES].index(RankCompareResult)][1])
    res = RankCompareResult(**kwargs)
    ci = res.conf_int()
    assert len(ci) == 2
    sup = res.test_prob_superior()
    assert hasattr(sup, "statistic")
    assert hasattr(sup, "pvalue")


def test_dispersion_results_summary_frame_still_works():
    kwargs = dict(CASES[[c[0] for c in CASES].index(DispersionResults)][1])
    res = DispersionResults(**kwargs)
    frame = res.summary_frame()
    assert list(frame.columns) == ["statistic", "pvalue", "method", "alternative"]
    assert len(frame) == 2


def test_adfuller_kpss_rur_repr_no_stale_attribute():
    # ADFullerResult, KPSSResult and RURResult each define a custom
    # __repr__ that references self.statistic; a prior bug referenced the
    # nonexistent self.stat instead and raised AttributeError.
    adf = ADFullerResult(**dict(CASES[[c[0] for c in CASES].index(ADFullerResult)][1]))
    kpss_res = KPSSResult(**dict(CASES[[c[0] for c in CASES].index(KPSSResult)][1]))
    rur = RURResult(**dict(CASES[[c[0] for c in CASES].index(RURResult)][1]))
    for res in (adf, kpss_res, rur):
        text = repr(res)
        assert f"{res.statistic:0.5f}" in text
        assert f"{res.pvalue:0.5f}" in text
