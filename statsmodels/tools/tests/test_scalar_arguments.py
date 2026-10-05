"""
Arguments that have to be scalars

The functions below validate scalar arguments with `float_like` or
`int_like` and then compare the scalar. An array, a list, a string or None is
a TypeError that names the argument, and a value that is not in the valid range
is a ValueError. NaN is not in any range, which the comparison of the form
``x <= 0`` does not catch.
"""
import numpy as np
import pytest

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.robust.scale import iqr, mad
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.multitest import (
    fdrcorrection,
    fdrcorrection_twostage,
    multipletests,
)
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.stats.oneway import confint_effectsize_oneway, confint_noncentrality
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    multinomial_proportions_confint,
    proportions_ztest,
)
from statsmodels.stats.stattools import robust_kurtosis
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.vector_ar.var_model import VAR

NAN = np.nan
RNG = np.random.default_rng(20261004)
X = sm.add_constant(RNG.standard_normal((60, 2)))
Y = X @ [1.0, 0.5, -0.3] + RNG.standard_normal(60)
COUNT_Y = RNG.poisson(np.exp(0.2 * X[:, 1] + 0.5))
OLS_RES = sm.OLS(Y, X).fit()
GLM_RES = GLM(COUNT_Y, X, family=sm.families.Poisson()).fit()
PVALS = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
SERIES = np.cumsum(RNG.standard_normal(120)) + RNG.standard_normal(120)
VAR_RES = VAR(RNG.standard_normal((120, 2))).fit(1)
RANGE = [NAN, 0.0, 1.0, -0.1, 1.5]
POSITIVE = [NAN, 0.0, -1.0]

# (id, callable of the argument, a valid value, invalid values)
FLOAT_CASES = [
    ("results.conf_int", lambda v: OLS_RES.conf_int(alpha=v), 0.1, RANGE),
    ("regression get_prediction.conf_int", lambda v: OLS_RES.get_prediction(X[:3]).conf_int(alpha=v), 0.1, RANGE),
    ("glm get_prediction.conf_int", lambda v: GLM_RES.get_prediction(X[:3]).conf_int(alpha=v), 0.1, RANGE),
    ("glm get_prediction.conf_int delta", lambda v: GLM_RES.get_prediction(X[:3]).conf_int(method="delta", alpha=v), 0.1, RANGE),
    ("t_test.conf_int", lambda v: OLS_RES.t_test("x1 = 0").conf_int(alpha=v), 0.1, RANGE),
    ("combine_effects", lambda v: combine_effects(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.03]), alpha=v), 0.1, RANGE),
    ("multipletests", lambda v: multipletests(PVALS, alpha=v), 0.1, RANGE),
    ("fdrcorrection", lambda v: fdrcorrection(PVALS, alpha=v), 0.1, [NAN, 0.0, -0.1, 1.5]),
    ("fdrcorrection_twostage", lambda v: fdrcorrection_twostage(PVALS, alpha=v), 0.1, RANGE),
    ("confint_mvmean_fromstats alpha", lambda v: confint_mvmean_fromstats(np.array([1.0, 2.0]), np.eye(2), 50, np.eye(2), alpha=v), 0.1, RANGE),
    ("confint_mvmean_fromstats nobs", lambda v: confint_mvmean_fromstats(np.array([1.0, 2.0]), np.eye(2), v, np.eye(2)), 50, POSITIVE),
    ("confint_noncentrality", lambda v: confint_noncentrality(3.0, (2, 30), alpha=v), 0.1, RANGE),
    ("confint_effectsize_oneway alpha", lambda v: confint_effectsize_oneway(3.0, (2, 30), alpha=v), 0.1, RANGE),
    ("confint_effectsize_oneway nobs", lambda v: confint_effectsize_oneway(3.0, (2, 30), nobs=v), 33, POSITIVE),
    ("multinomial_proportions_confint", lambda v: multinomial_proportions_confint([10, 20, 30], alpha=v), 0.1, RANGE),
    ("proportions_ztest prop_var", lambda v: proportions_ztest(10, 50, value=0.2, prop_var=v), 0.3, [NAN, 1.0, -0.1, 1.5]),
    ("confint_proportions_2indep", lambda v: confint_proportions_2indep(10, 50, 15, 60, alpha=v), 0.1, RANGE),
    ("GLM.fit scale", lambda v: GLM(COUNT_Y, X, family=sm.families.Poisson()).fit(scale=v), 1.5, POSITIVE),
    ("mad", lambda v: mad(SERIES, c=v), 0.6745, POSITIVE),
    ("iqr", lambda v: iqr(SERIES, c=v), 1.349, POSITIVE),
    ("hpfilter", lambda v: hpfilter(SERIES, lamb=v), 1600.0, POSITIVE),
    ("bkfilter low", lambda v: bkfilter(SERIES, low=v, high=32), 6.0, [NAN, 0.0, 40.0]),
    ("bkfilter high", lambda v: bkfilter(SERIES, low=6, high=v), 32.0, [NAN, 6.0, 2.0]),
    ("cffilter low", lambda v: cffilter(SERIES, low=v, high=32), 6.0, [NAN, 1.0, 40.0]),
    ("cffilter high", lambda v: cffilter(SERIES, low=6, high=v), 32.0, [NAN, 6.0, 2.0]),
    ("KDEUnivariate.fit bw", lambda v: KDEUnivariate(SERIES).fit(bw=v), 0.5, POSITIVE),
    ("VARResults.test_whiteness", lambda v: VAR_RES.test_whiteness(nlags=6, signif=v), 0.05, RANGE),
    ("robust_kurtosis ab", lambda v: robust_kurtosis(SERIES, ab=(5.0, v)), 50.0, [NAN, 3.0, 100.0]),
    ("robust_kurtosis dg", lambda v: robust_kurtosis(SERIES, dg=(2.5, v)), 25.0, [NAN, 2.0, 100.0]),
]
INT_CASES = [
    ("variance_inflation_factor", lambda v: variance_inflation_factor(X, v), 1, [-1, 3, 99]),
    ("bkfilter K", lambda v: bkfilter(SERIES, K=v), 12, [-1]),
    ("VAR forecast steps", lambda v: VAR_RES.forecast(SERIES[:2, None] * np.ones((2, 2)), v), 3, [-1]),
    ("VAR irf periods", VAR_RES.irf, 3, [-1]),
]
FLOAT_IDS = [c[0] for c in FLOAT_CASES]
INT_IDS = [c[0] for c in INT_CASES]

# The wrong types of each argument. A string or None are not wrong for the
# arguments where they select a default or a method.
ALL_BAD_TYPES = ("array", "list", "str", "None", "bool")
NOT_BAD = {
    "GLM.fit scale": ("str", "None", "bool"),
    # an array raises ValueError in the truth test that selects the default
    "proportions_ztest prop_var": ("array", "None"),
    "KDEUnivariate.fit bw": ("str",),
    "confint_effectsize_oneway nobs": ("None",),
}


def _bad_types(name, good):
    bad = {
        "array": np.array([good, good]),
        "list": [good, good],
        "str": "a",
        "None": None,
        "bool": True,
    }
    return [bad[k] for k in ALL_BAD_TYPES if k not in NOT_BAD.get(name, ())]


@pytest.mark.parametrize("name, func, good", [c[:3] for c in FLOAT_CASES], ids=FLOAT_IDS)
def test_float_argument_is_a_scalar(name, func, good):
    func(good)
    func(np.float32(good))
    func(np.float64(good))
    for bad in _bad_types(name, good):
        with pytest.raises(TypeError):
            func(bad)


@pytest.mark.parametrize("func, bad_values", [c[1:2] + c[3:] for c in FLOAT_CASES], ids=FLOAT_IDS)
def test_float_argument_values(func, bad_values):
    for bad in bad_values:
        with pytest.raises(ValueError):
            func(bad)


@pytest.mark.parametrize("func, good", [c[1:3] for c in INT_CASES], ids=INT_IDS)
def test_int_argument_is_a_scalar(func, good):
    func(good)
    func(np.int64(good))
    func(float(good))  # a whole float is an integer for int_like
    for bad in (good + 0.5, [good, good], "a", None, True):
        with pytest.raises((TypeError, ValueError)):
            func(bad)


@pytest.mark.parametrize("func, bad_values", [c[1:2] + c[3:] for c in INT_CASES], ids=INT_IDS)
def test_int_argument_values(func, bad_values):
    for bad in bad_values:
        with pytest.raises(ValueError):
            func(bad)
