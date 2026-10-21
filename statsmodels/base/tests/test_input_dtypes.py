"""
Tests that models give the same results for integer, boolean and pandas
nullable inputs as for the same values stored as float.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.discrete.discrete_model import Logit, NegativeBinomial, Poisson
from statsmodels.genmod import families
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

NOBS = 100
GROUPS = np.repeat(np.arange(20), 5)
MISSING = [3, 40]
# the estimates are equal up to the rounding of the linear algebra
TOL = {"rtol": 1e-6, "atol": 1e-8}


def _data():
    rng = np.random.default_rng(10477)
    x = rng.integers(0, 10, size=(NOBS, 2))
    exog = np.column_stack([np.ones(NOBS, dtype=int), x])
    group_effect = rng.integers(-3, 4, 20)[GROUPS]
    # overdispersed counts so that alpha of NegativeBinomial is identified
    mu = np.exp(0.5 + 0.1 * x[:, 0])
    endog = {
        "continuous": 3 * x[:, 0]
        - 2 * x[:, 1]
        + group_effect
        + rng.integers(-5, 6, NOBS),
        "count": rng.negative_binomial(2, 2 / (2 + mu)),
        "binary": (rng.random(NOBS) < 0.4).astype(int),
    }
    return endog, exog


ENDOG, EXOG = _data()

MODELS = {
    "OLS": ("continuous", lambda y, x, **kw: OLS(y, x, **kw).fit()),
    "WLS": (
        "continuous",
        lambda y, x, **kw: WLS(y, x, weights=np.arange(1.0, NOBS + 1), **kw).fit(),
    ),
    "QuantReg": ("continuous", lambda y, x, **kw: QuantReg(y, x, **kw).fit()),
    "RLM": ("continuous", lambda y, x, **kw: RLM(y, x, **kw).fit()),
    "GLM-Gaussian": ("continuous", lambda y, x, **kw: GLM(y, x, **kw).fit()),
    "GLM-Poisson": (
        "count",
        lambda y, x, **kw: GLM(y, x, family=families.Poisson(), **kw).fit(),
    ),
    "GLM-Binomial": (
        "binary",
        lambda y, x, **kw: GLM(y, x, family=families.Binomial(), **kw).fit(),
    ),
    "Logit": ("binary", lambda y, x, **kw: Logit(y, x, **kw).fit(disp=0)),
    "Poisson": ("count", lambda y, x, **kw: Poisson(y, x, **kw).fit(disp=0)),
    "NegativeBinomial": (
        "count",
        lambda y, x, **kw: NegativeBinomial(y, x, **kw).fit(disp=0),
    ),
    "MixedLM": (
        "continuous",
        lambda y, x, **kw: MixedLM(y, x, groups=GROUPS, **kw).fit(),
    ),
    "GEE": ("continuous", lambda y, x, **kw: GEE(y, x, groups=GROUPS, **kw).fit()),
}


def _convert(values, dtype):
    if dtype in ("int64", "int32"):
        return np.asarray(values).astype(dtype)
    if np.ndim(values) == 1:
        return pd.Series(values, dtype=dtype)
    return pd.DataFrame(values).astype(dtype)


@pytest.mark.parametrize("dtype", ["int64", "int32", "Int64"])
@pytest.mark.parametrize("model", list(MODELS))
def test_integer_inputs(model, dtype):
    kind, fit = MODELS[model]
    endog = ENDOG[kind]
    expected = fit(endog.astype(float), EXOG.astype(float))
    res = fit(_convert(endog, dtype), _convert(EXOG, dtype))
    assert_allclose(np.asarray(res.params), np.asarray(expected.params), **TOL)


@pytest.mark.parametrize("model", list(MODELS))
def test_bool_dummy_exog(model):
    kind, fit = MODELS[model]
    endog = ENDOG[kind].astype(float)
    exog = pd.DataFrame(
        {"const": 1.0, "x0": EXOG[:, 1].astype(float), "d": EXOG[:, 2] > 4}
    )
    expected = fit(endog, exog.astype(float))
    res = fit(endog, exog)
    assert_allclose(np.asarray(res.params), np.asarray(expected.params), **TOL)


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("model", list(MODELS))
def test_nullable_with_missing(model, dtype):
    # pd.NA in nullable inputs is dropped in the same way as nan in floats
    if model == "MixedLM":
        pytest.skip("missing='drop' is not supported by MixedLM, see GH 7978")
    kind, fit = MODELS[model]
    endog = ENDOG[kind].astype(float)
    endog[MISSING] = np.nan
    expected = fit(endog, EXOG.astype(float), missing="drop")
    endog_na = pd.Series(ENDOG[kind], dtype=dtype)
    endog_na[MISSING] = pd.NA
    res = fit(endog_na, _convert(EXOG, dtype), missing="drop")
    assert res.nobs == NOBS - len(MISSING)
    assert_allclose(np.asarray(res.params), np.asarray(expected.params), **TOL)


@pytest.mark.parametrize("dtype", [bool, "boolean"])
@pytest.mark.parametrize("model", ["GLM-Binomial", "Logit"])
def test_boolean_endog(model, dtype):
    _, fit = MODELS[model]
    endog = ENDOG["binary"]
    expected = fit(endog.astype(float), EXOG.astype(float))
    res = fit(pd.Series(endog.astype(bool), dtype=dtype), EXOG)
    assert_allclose(np.asarray(res.params), np.asarray(expected.params), **TOL)


TS_MODELS = {
    "AutoReg": lambda y: AutoReg(y, 2).fit(),
    "ARIMA": lambda y: ARIMA(y, order=(1, 0, 0), trend="c").fit(),
}


@pytest.mark.parametrize("dtype", ["int64", "int32", "Int64"])
@pytest.mark.parametrize("model", list(TS_MODELS))
def test_integer_time_series(model, dtype):
    rng = np.random.default_rng(5445)
    endog = np.cumsum(rng.integers(-3, 4, NOBS)) + 50
    fit = TS_MODELS[model]
    expected = fit(endog.astype(float))
    res = fit(_convert(endog, dtype))
    assert_allclose(np.asarray(res.params), np.asarray(expected.params), **TOL)
