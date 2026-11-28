"""
Formulas that from_formula evaluates itself (second formulas of a model) use
the same eval_env as the model's formula: None or 1 is the namespace that
calls from_formula, an integer k the namespace k stack frames above it, and
a dict holds the variables.
"""

import numpy as np
import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import HAVE_FORMULAIC, HAVE_PATSY, FormulaManager
import statsmodels.formula.api as smf
from statsmodels.formula.formulatools import get_matrices_eval_env
from statsmodels.othermod.betareg import BetaModel
from statsmodels.regression.process_regression import ProcessMLE

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
        "time": np.tile(np.arange(5.0), 8),
        "y01": _rng.uniform(0.05, 0.95, NOBS),
    }
)

CASES = [
    "beta_model-precision",
    "process_mle-formula",
    "process_mle-scale",
    "process_mle-smooth",
    "process_mle-noise",
    "gee-dep_data",
]


@pytest.fixture(params=ENGINES)
def engine(request, monkeypatch):
    monkeypatch.setattr(statsmodels.formula.options, "formula_engine", request.param)
    return request.param


def _caller(case, **kwargs):
    # The namespace that calls from_formula: every model is created here
    w = CALLER  # noqa: F841
    data = DATA
    if case == "beta_model-precision":
        mod = BetaModel.from_formula(
            "y01 ~ x", data, exog_precision_formula="0 + w", **kwargs
        )
        return np.asarray(mod.exog_precision)[:, 0]
    if case == "gee-dep_data":
        mod = smf.gee("y ~ x", "g", data, dep_data="0 + w", **kwargs)
        return np.asarray(mod.dep_data)[:, 0]
    part = case.split("-")[1]
    formulas = {
        "formula": "y ~ x",
        "scale": "0 + x",
        "smooth": "0 + x",
        "noise": "0 + x",
    }
    formulas[part] = "y ~ 0 + w" if part == "formula" else "0 + w"
    mod = ProcessMLE.from_formula(
        formulas["formula"],
        data,
        scale_formula=formulas["scale"],
        smooth_formula=formulas["smooth"],
        noise_formula=formulas["noise"],
        time="time",
        groups="g",
        **kwargs,
    )
    exog = {
        "formula": mod.exog,
        "scale": mod.exog_scale,
        "smooth": mod.exog_smooth,
        "noise": mod.exog_noise,
    }[part]
    return np.asarray(exog)[:, 0]


def _above(case, **kwargs):
    # One stack frame above the namespace that calls from_formula
    w = ABOVE  # noqa: F841
    return _caller(case, **kwargs)


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "eval_env, expected",
    [(None, CALLER), (1, CALLER), (2, ABOVE)],
    ids=["None", "1", "2"],
)
def test_second_formula_eval_env(engine, case, eval_env, expected):
    kwargs = {} if eval_env is None else {"eval_env": eval_env}
    np.testing.assert_allclose(_above(case, **kwargs), expected)


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
# patsy warned about its own empty environment for eval_env=-1 before 0.16.0
@pytest.mark.filterwarnings("ignore:EvalEnvironment is deprecated:FutureWarning")
@pytest.mark.parametrize("case", CASES)
def test_second_formula_eval_env_empty(engine, case):
    with pytest.raises(FormulaManager().factor_evaluation_error):
        _above(case, eval_env=-1)


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.filterwarnings("ignore:EvalEnvironment is deprecated:FutureWarning")
def test_get_matrices_eval_env_empty(engine):
    # -1 is an empty namespace, not the namespace of statsmodels' own code,
    # where pd is defined
    eval_env = get_matrices_eval_env(-1)
    mgr = FormulaManager()
    with pytest.raises(mgr.factor_evaluation_error):
        mgr.get_matrices("~ pd.Series(y01)", DATA, eval_env=eval_env)
    out = mgr.get_matrices("~ pd.Series(y01)", DATA, eval_env={"pd": pd})
    np.testing.assert_allclose(np.asarray(out)[:, 1], DATA["y01"])


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("case", CASES)
def test_second_formula_eval_env_dict(engine, case):
    np.testing.assert_allclose(_above(case, eval_env={"w": OTHER}), OTHER)
    with pytest.raises(FormulaManager().factor_evaluation_error):
        _above(case, eval_env={"v": OTHER})
