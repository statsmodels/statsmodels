"""
eval_env has the same meaning in every from_formula: None or 1 is the
namespace that calls from_formula, an integer k the namespace k stack frames
above from_formula, -1 an empty namespace, and a dict holds the variables.
"""

import numpy as np
import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import HAVE_FORMULAIC, HAVE_PATSY, FormulaManager
import statsmodels.formula.api as smf
from statsmodels.genmod.qif import QIF
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.othermod.betareg import BetaModel
from statsmodels.regression.rolling import RollingOLS

ENGINES = (["patsy"] if HAVE_PATSY else []) + (["formulaic"] if HAVE_FORMULAIC else [])

NOBS = 40
# w in the namespace that calls from_formula, one frame above it, and a dict
CALLER = np.linspace(1.0, 2.0, NOBS)
ABOVE = -np.linspace(1.0, 2.0, NOBS)
OTHER = np.linspace(3.0, 4.0, NOBS)

_rng = np.random.default_rng(20261126)
DATA = pd.DataFrame(
    {
        "y": _rng.standard_normal(NOBS),
        "x": _rng.standard_normal(NOBS),
        "g": np.repeat(np.arange(8), 5),
        "t": _rng.exponential(size=NOBS),
        "yb": np.tile([0, 1, 1, 0, 1], 8),
        "yo": np.tile([0, 1, 2, 1, 0], 8),
        "y01": _rng.uniform(0.05, 0.95, NOBS),
    }
)

CASES = [
    "ols",
    "gee",
    "phreg",
    "mixedlm",
    "mixedlm-re_formula",
    "conditional_logit",
    "qif",
    "ordered_model",
    "beta_model",
    "rolling_ols",
]


@pytest.fixture(params=ENGINES)
def engine(request, monkeypatch):
    monkeypatch.setattr(statsmodels.formula.options, "formula_engine", request.param)
    return request.param


def _caller(case, **kwargs):
    # The namespace that calls from_formula: every model is created here
    w = CALLER  # noqa: F841
    data = DATA
    if case == "ols":
        mod = smf.ols("y ~ x + w", data, **kwargs)
    elif case == "gee":
        mod = smf.gee("y ~ x + w", "g", data, **kwargs)
    elif case == "phreg":
        mod = smf.phreg("t ~ x + w", data, **kwargs)
    elif case == "mixedlm":
        mod = smf.mixedlm("y ~ x + w", data, groups="g", **kwargs)
    elif case == "mixedlm-re_formula":
        mod = smf.mixedlm("y ~ x", data, groups="g", re_formula="0 + w", **kwargs)
        return np.asarray(mod.exog_re)[:, 0]
    elif case == "conditional_logit":
        mod = smf.conditional_logit("yb ~ 0 + x + w", data, groups="g", **kwargs)
    elif case == "qif":
        mod = QIF.from_formula("y ~ x + w", "g", data, **kwargs)
    elif case == "ordered_model":
        mod = OrderedModel.from_formula("yo ~ 0 + x + w", data, distr="logit", **kwargs)
    elif case == "beta_model":
        mod = BetaModel.from_formula("y01 ~ x + w", data, **kwargs)
    else:
        mod = RollingOLS.from_formula("y ~ x + w", data, window=20, **kwargs)
        return np.asarray(mod.data.orig_exog["w"])
    names = list(mod.exog_names)
    return np.asarray(mod.exog)[:, names.index("w")]


def _above(case, **kwargs):
    # One stack frame above the namespace that calls from_formula
    w = ABOVE  # noqa: F841
    return _caller(case, **kwargs)


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "eval_env, expected",
    [(None, CALLER), (1, CALLER), (np.int64(1), CALLER), (2, ABOVE)],
    ids=["None", "1", "int64-1", "2"],
)
def test_eval_env_frames(engine, case, eval_env, expected):
    kwargs = {} if eval_env is None else {"eval_env": eval_env}
    np.testing.assert_allclose(_above(case, **kwargs), expected)


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("case", CASES)
def test_eval_env_dict(engine, case):
    np.testing.assert_allclose(_above(case, eval_env={"w": OTHER}), OTHER)
    # A name that is in neither data nor the dict is not looked up elsewhere
    with pytest.raises(FormulaManager().factor_evaluation_error):
        _above(case, eval_env={"v": OTHER})
