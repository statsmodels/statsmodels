"""
from_formula of RecursiveLS, BinomialBayesMixedGLM, PoissonBayesMixedGLM,
ARDL and UECM take eval_env with the same meaning as other models: None or 1
is the namespace that calls from_formula, an integer k the namespace k stack
frames above it, -1 an empty namespace, and a dict holds the variables.
"""

import numpy as np
import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import HAVE_FORMULAIC, HAVE_PATSY, FormulaManager
from statsmodels.genmod.bayes_mixed_glm import (
    BinomialBayesMixedGLM,
    PoissonBayesMixedGLM,
)
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.tsa.ardl import ARDL, UECM

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
        "yb": np.tile([0, 1, 1, 0, 1], 8),
        "yc": np.tile([0, 1, 3, 2, 1], 8),
    }
)

CASES = [
    "recursive_ls",
    "binomial_bayes_mixed_glm",
    "binomial_bayes_mixed_glm-vc_formulas",
    "poisson_bayes_mixed_glm",
    "poisson_bayes_mixed_glm-vc_formulas",
    "ardl",
    "ardl-fixed",
    "uecm",
]


@pytest.fixture(params=ENGINES)
def engine(request, monkeypatch):
    monkeypatch.setattr(statsmodels.formula.options, "formula_engine", request.param)
    return request.param


def _column(mod, name="w"):
    return np.asarray(mod.exog)[:, list(mod.exog_names).index(name)]


def _caller(case, **kwargs):
    # The namespace that calls from_formula: every model is created here
    w = CALLER  # noqa: F841
    data = DATA
    if case == "recursive_ls":
        return _column(RecursiveLS.from_formula("y ~ x + w", data, **kwargs))
    if case.startswith(("binomial", "poisson")):
        binomial = case.startswith("binomial")
        cls = BinomialBayesMixedGLM if binomial else PoissonBayesMixedGLM
        endog = "yb" if binomial else "yc"
        if case.endswith("vc_formulas"):
            mod = cls.from_formula(f"{endog} ~ x", {"a": "0 + w"}, data, **kwargs)
            return mod.exog_vc.toarray()[:, 0]
        mod = cls.from_formula(f"{endog} ~ x + w", {"a": "0 + C(g)"}, data, **kwargs)
        return np.asarray(mod.exog)[:, list(mod.fep_names).index("w")]
    if case == "ardl":
        mod = ARDL.from_formula("y ~ w", data, 1, 0, **kwargs)
        return np.asarray(mod.data.orig_exog["w"])
    if case == "ardl-fixed":
        mod = ARDL.from_formula("y ~ x | w", data, 1, 0, **kwargs)
        return np.asarray(mod.fixed["w"])
    mod = UECM.from_formula("y ~ w", data, 1, 1, **kwargs)
    return np.asarray(mod.data.orig_exog["w"])


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
def test_from_formula_eval_env(engine, case, eval_env, expected):
    kwargs = {} if eval_env is None else {"eval_env": eval_env}
    np.testing.assert_allclose(_above(case, **kwargs), expected)


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("case", CASES)
def test_from_formula_eval_env_dict(engine, case):
    np.testing.assert_allclose(_above(case, eval_env={"w": OTHER}), OTHER)
    with pytest.raises(FormulaManager().factor_evaluation_error):
        _above(case, eval_env={"v": OTHER})


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
# patsy warned about its own empty environment for eval_env=-1 before 0.15.1
@pytest.mark.filterwarnings("ignore:EvalEnvironment is deprecated:FutureWarning")
@pytest.mark.parametrize("case", CASES)
def test_from_formula_eval_env_empty(engine, case):
    with pytest.raises(FormulaManager().factor_evaluation_error):
        _above(case, eval_env=-1)
