from statsmodels.compat.python import PYTHON_IMPL_WASM

import json
import os
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import FormulaManager, LinearConstraintValues

PATSY_ERROR_LOCK = threading.Lock()

HAS_FORMULAIC = HAS_PATSY = False
try:
    import patsy

    HAS_PATSY = True
except ImportError:
    pass

try:
    import formulaic

    HAS_FORMULAIC = True
except ImportError:
    pass

no_formulaic = pytest.mark.skipif(
    HAS_FORMULAIC, reason="can only run when patsy is installed and formulaic is not."
)

no_patsy = pytest.mark.skipif(
    HAS_PATSY, reason="can only run when formulaic is installed and patsy is not."
)

require_formulaic = pytest.mark.skipif(not HAS_FORMULAIC, reason="Requires formulaic")
require_patsy = pytest.mark.skipif(not HAS_PATSY, reason="Requires formulaic")

ENGINES = ["patsy"] if HAS_PATSY else []
ENGINES += ["formulaic"] if HAS_FORMULAIC else []


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


FORMULAS = [
    "y ~ 1 + C(d):a + b + a + C(e)",
    "y ~ 1 + a + C(e) + a:C(e) + C(f)",
    "y ~ 1 + a + C(e) + a:C(e) + C(f) + C(f):C(d)",
    "y ~ 1 + C(e) + C(f):C(e) + a + a:b + C(d):a:b + b + C(d)",
    "y ~ a - 1",
    "y ~ a + C(e) - 1",
    "y ~ a + C(e) + a:C(e) + C(f)  - 1",
    "y ~ a + C(e) + a:C(e) + C(f) + C(f):C(d)  - 1",
    "y ~ a + C(d):a:b + a:b + b + C(d) + C(e) + C(f):C(e)  - 1",
    (
        "y ~ d:f:c:b + C(f):c:b + a:b + a + C(d):a:b + c:b + "
        "b + C(d) + C(e) + C(f):C(e) - 1"
    ),
    (
        "y ~ 1 + C(f):C(d):a +  C(d):a + c + a:b + C(e):b + "
        "C(f):a + a + C(d) + b  +C(f):C(d) + C(d):C(e) + C(f)"
    ),
]


@pytest.fixture(params=FORMULAS)
def formula(request):
    return request.param


def g(x):
    return x**2


@pytest.fixture
def data(request):
    return pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7.2],
            "x": [1, 2, 3, 9, 8, 7, 6.3],
            "z": [-1, 2, -3, 9, -8, 7, -6.1],
            "c": ["a", "a", "b", "a", "b", "b", "b"],
        }
    )


def check_type(arr, engine):
    if engine == "patsy":
        return isinstance(arr, pd.DataFrame)
    else:
        return isinstance(arr, formulaic.ModelMatrix)


@pytest.mark.thread_unsafe("changes global formula_engine variable")
def test_engine_options_engine(engine):
    default = statsmodels.formula.options.formula_engine
    assert default in ("patsy", "formulaic")

    statsmodels.formula.options.formula_engine = engine
    mgr = FormulaManager()
    assert mgr.engine == engine

    if HAS_FORMULAIC:
        statsmodels.formula.options.formula_engine = "formulaic"
        mgr = FormulaManager()
        assert mgr.engine == "formulaic"
        if HAS_PATSY:
            mgr = FormulaManager(engine="patsy")
            assert mgr.engine == "patsy"

    statsmodels.formula.options.formula_engine = default


@pytest.mark.thread_unsafe(reason="changes global ordering parameter")
@pytest.mark.parametrize("ordering", ["degree", "sort", "none", "legacy"])
def test_engine_options_order(ordering):
    default = statsmodels.formula.options.ordering
    assert default in ("degree", "sort", "none", "legacy")

    statsmodels.formula.options.ordering = ordering
    assert statsmodels.formula.options.ordering == ordering
    statsmodels.formula.options.ordering = default

    with pytest.raises(ValueError):
        statsmodels.formula.options.ordering = "unknown"


@require_formulaic
@pytest.mark.thread_unsafe(reason="changes global ordering parameter")
def test_engine_options_order_effect(data):
    default = statsmodels.formula.options.ordering
    statsmodels.formula.options.ordering = "degree"
    mgr = FormulaManager(engine="formulaic")
    _, rhs0 = mgr.get_matrices("y ~ 1 + x + z + c", data)
    statsmodels.formula.options.ordering = "sort"
    _, rhs1 = mgr.get_matrices("y ~ 1 + x + z + c", data)
    statsmodels.formula.options.ordering = "none"
    _, rhs2 = mgr.get_matrices("y ~ 1 + x + c + z", data)
    assert len(rhs0.columns) == 4
    assert len(rhs1.columns) == 4
    assert len(rhs2.columns) == 4
    assert list(rhs0.columns) != list(rhs1.columns)
    assert list(rhs0.columns) != list(rhs2.columns)
    assert list(rhs1.columns) != list(rhs2.columns)
    statsmodels.formula.options.ordering = default


@require_formulaic
@pytest.mark.thread_unsafe(reason="changes global ordering parameter")
@pytest.mark.parametrize("ordering", ["degree", "sort", "none", "legacy"])
def test_get_matrices_model_spec_any_ordering(data, ordering, monkeypatch):
    # A ModelSpec was parsed again as a formula unless the ordering was legacy
    monkeypatch.setattr(statsmodels.formula.options, "ordering", ordering)
    mgr = FormulaManager(engine="formulaic")
    _, rhs = mgr.get_matrices("y ~ 1 + x + c", data)
    new = data.iloc[[4, 1, 6]]
    out = mgr.get_matrices(rhs.model_spec, new)
    assert list(out.columns) == list(rhs.columns)
    expected = pd.DataFrame(
        {
            "Intercept": 1.0,
            "x": new["x"],
            "c[T.b]": (new["c"] == "b").astype(float),
        },
        index=new.index,
    )
    np.testing.assert_allclose(
        np.asarray(out[list(expected.columns)], dtype=float), expected.to_numpy()
    )


def test_engine_options_err():
    with pytest.raises(ValueError, match="Invalid formula engine option"):
        statsmodels.formula.options.formula_engine = "unknown"


def test_engine(engine):
    mgr = FormulaManager(engine=engine)
    assert mgr.engine == engine

    with pytest.raises(ValueError):
        FormulaManager(engine="other")


def test_single_array(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "1 + x + z"
    output = mgr.get_matrices(fmla, data)
    assert check_type(output, engine)


def test_two_arrays(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_matrices(fmla, data)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert check_type(output[0], engine)
    assert check_type(output[1], engine)


@pytest.mark.parametrize("formula", ["y ~ 1 + x + c", "y ~ x + c - 1", "x + c"])
@pytest.mark.parametrize("with_lhs_data", [True, False])
def test_get_matrices_prediction_rhs_only(engine, data, formula, with_lhs_data):
    # prediction=True evaluates only the right-hand side: the left-hand side
    # need not be in the data, and only the design matrix is returned
    mgr = FormulaManager(engine=engine)
    fitted = mgr.get_matrices(formula, data)
    rhs = fitted[1] if isinstance(fitted, tuple) else fitted
    new = data.iloc[[5, 0, 2]]
    if not with_lhs_data:
        new = new.drop(columns="y")
    out = mgr.get_matrices(formula, new, prediction=True)
    assert check_type(out, engine)
    assert list(out.columns) == list(rhs.columns)
    np.testing.assert_allclose(
        np.asarray(out, dtype=float), np.asarray(rhs.loc[new.index], dtype=float)
    )


@require_patsy
def test_get_matrices_prediction_model_desc(data):
    mgr = FormulaManager(engine="patsy")
    _, rhs = mgr.get_matrices("y ~ x + c", data)
    desc = patsy.ModelDesc.from_formula("y ~ x + c")
    new = data.iloc[[5, 0, 2]].drop(columns="y")
    out = mgr.get_matrices(desc, new, prediction=True)
    assert list(out.columns) == list(rhs.columns)
    np.testing.assert_allclose(np.asarray(out), np.asarray(rhs.loc[new.index]))


def test_get_column_names(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_matrices(fmla, data)
    names = mgr.get_column_names(output[0])
    assert names == ["y"]

    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "z"]

    names = mgr.get_column_names(mgr.get_model_spec(output[1]))
    assert names == ["Intercept", "x", "z"]


@require_patsy
def test_get_empty_eval_patsy(data):
    mgr = FormulaManager(engine="patsy")
    fmla = "y ~ 1 + x + g(z)"
    output = mgr.get_matrices(fmla, data)
    assert check_type(output[0], "patsy")
    assert check_type(output[1], "patsy")
    names = mgr.get_column_names(output[0])
    assert names == ["y"]
    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "g(z)"]

    output = mgr.get_matrices(fmla, data, eval_env=0)
    assert check_type(output[0], "patsy")
    assert check_type(output[1], "patsy")
    names = mgr.get_column_names(output[0])
    assert names == ["y"]
    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "g(z)"]


@require_patsy
def test_get_empty_eval_patsy_errors(data):
    # Threaded use causes call stack depth issues
    mgr = FormulaManager(engine="patsy")
    fmla = "y ~ 1 + x + g(z)"
    with PATSY_ERROR_LOCK:
        with pytest.raises(patsy.PatsyError):
            mgr.get_matrices(fmla, data, eval_env=2)

        with pytest.raises(patsy.PatsyError):
            with pytest.warns(FutureWarning, match="EvalEnvironment is deprecate"):
                mgr.get_matrices(fmla, data, eval_env=patsy.eval.EvalEnvironment({}))

        # The empty environment is not a deprecated EvalEnvironment
        with pytest.raises(patsy.PatsyError):
            mgr.get_matrices(fmla, data, eval_env=mgr.get_empty_eval_env())


@require_formulaic
def test_get_empty_eval_formulaic(data):
    mgr = FormulaManager(engine="formulaic")
    fmla = "y ~ 1 + x + g(z)"
    output = mgr.get_matrices(fmla, data)
    assert check_type(output[0], "formulaic")
    assert check_type(output[1], "formulaic")

    output = mgr.get_matrices(fmla, data, eval_env=0)
    assert check_type(output[0], "formulaic")
    assert check_type(output[1], "formulaic")
    names = mgr.get_column_names(output[0])
    assert names == ["y"]
    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "g(z)"]

    eval_env = mgr.get_empty_eval_env()
    eval_env["g"] = g
    output = mgr.get_matrices(fmla, data, eval_env=eval_env)
    assert check_type(output[0], "formulaic")
    assert check_type(output[1], "formulaic")
    names = mgr.get_column_names(output[0])
    assert names == ["y"]
    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "g(z)"]


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("model", ["ols", "gee", "mixedlm", "rolling_ols"])
def test_eval_env_minus_one(engine, model, monkeypatch, recwarn):
    # eval_env=-1 evaluates the formula in an empty namespace, without a
    # deprecation warning
    import statsmodels.formula.api as smf
    from statsmodels.regression.rolling import RollingOLS

    monkeypatch.setattr(statsmodels.formula.options, "formula_engine", engine)
    rng = np.random.default_rng(1234)
    frame = pd.DataFrame(
        {
            "y": rng.standard_normal(40),
            "x": rng.standard_normal(40),
            "g": np.repeat(np.arange(8), 5),
        }
    )
    w = rng.standard_normal(40)  # noqa: F841

    def design(rhs, **kwargs):
        if model == "ols":
            return smf.ols("y ~ " + rhs, frame, **kwargs).exog
        if model == "gee":
            return smf.gee("y ~ " + rhs, "g", frame, **kwargs).exog
        if model == "mixedlm":
            mod = smf.mixedlm("y ~ x", frame, groups="g", re_formula=rhs, **kwargs)
            return mod.exog_re
        return RollingOLS.from_formula("y ~ " + rhs, frame, window=20, **kwargs).exog

    clean = design("1 + x", eval_env=-1)
    with pytest.raises(FormulaManager().factor_evaluation_error):
        design("1 + x + w", eval_env=-1)
    future = [rec for rec in recwarn if issubclass(rec.category, FutureWarning)]
    assert not future
    np.testing.assert_allclose(clean, design("1 + x"))


def test_has_intercept(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_matrices(fmla, data)
    assert mgr.has_intercept(mgr.spec)
    assert mgr.has_intercept(mgr.get_model_spec(output[1]))

    fmla = "y ~ x + z - 1"
    output = mgr.get_matrices(fmla, data)
    assert not mgr.has_intercept(mgr.spec)
    assert not mgr.has_intercept(mgr.get_model_spec(output[1]))


def test_get_intercept_idx(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_matrices(fmla, data)
    result = mgr.intercept_idx(mgr.spec)
    np.testing.assert_equal(result, np.array([True, False, False]))
    result = mgr.intercept_idx(mgr.get_model_spec(output[1]))
    np.testing.assert_equal(result, np.array([True, False, False]))

    fmla = "y ~ x + z - 1"
    output = mgr.get_matrices(fmla, data)
    result = mgr.intercept_idx(mgr.spec)
    np.testing.assert_equal(result, np.array([False, False]))
    result = mgr.intercept_idx(mgr.get_model_spec(output[1]))
    np.testing.assert_equal(result, np.array([False, False]))


def test_get_intercept_idx_terms_not_columns(engine, data):
    # One element per term: the categorical term has two columns here
    mgr = FormulaManager(engine=engine)
    frame = data.assign(d=list("pqrpqrp"))
    _, rhs = mgr.get_matrices("y ~ d + x", frame)
    assert rhs.shape[1] == 4
    names = mgr.get_term_names(mgr.spec)
    expected = [name in ("Intercept", "1") for name in names]
    np.testing.assert_equal(mgr.intercept_idx(mgr.spec), expected)
    assert len(expected) == 3


def test_remove_intercept(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_matrices(fmla, data)
    orig = mgr.spec.terms[:]
    result = mgr.remove_intercept(mgr.spec.terms)
    assert result == orig[1:]

    result = mgr.remove_intercept(mgr.get_model_spec(output[1]).terms)
    assert result == orig[1:]

    fmla = "y ~ x + z - 1"
    output = mgr.get_matrices(fmla, data)
    orig = mgr.spec.terms[:]
    result = mgr.remove_intercept(mgr.spec.terms)
    assert result == orig

    result = mgr.remove_intercept(mgr.get_model_spec(output[1]).terms)
    assert result == orig


def test_default_value():
    from statsmodels.formula._manager import _Default, _NoDefault

    assert isinstance(_NoDefault, _Default)
    _NoDefault.__str__()
    assert str(_NoDefault) == "<no default value>"
    assert repr(_NoDefault) == "<no default value>"


def test_get_spec(engine):
    mgr = FormulaManager(engine=engine)
    spec = mgr.get_spec("y ~ 1 + x + z")

    if engine == "patsy":
        assert isinstance(spec, patsy.desc.ModelDesc)
        assert len(spec.lhs_termlist) == 1
        assert len(spec.rhs_termlist) == 3
    else:
        assert isinstance(spec, formulaic.formula.Formula)
        assert len(spec.lhs) == 1
        assert len(spec.rhs) == 3


def test_get_na_action(engine, data):
    mgr = FormulaManager(engine=engine)
    result = mgr.get_na_action("drop")
    if engine == "patsy":
        assert result.on_NA == "drop"
        assert result.NA_types == ("None", "NaN")
    else:
        assert result == "drop"

    result = mgr.get_na_action(action="raise", types=["None"])
    if engine == "patsy":
        assert result.on_NA == "raise"
        assert result.NA_types == ("None",)
    else:
        assert result == "raise"


def test_get_model_spec(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    lhs, rhs = mgr.get_matrices(fmla, data)
    result = mgr.get_model_spec(rhs)
    if engine == "patsy":
        assert isinstance(result, patsy.design_info.DesignInfo)
    else:
        assert isinstance(result, formulaic.model_spec.ModelSpec)

    result = mgr.get_model_spec(data, optional=True)
    assert result is None


@pytest.mark.parametrize(
    "method",
    [
        "get_term_names",
        "get_column_names",
        "get_term_name_slices",
        "get_description",
        "get_factors",
        "is_categorical",
    ],
)
def test_spec_or_frame_without_spec_raises(engine, data, method):
    mgr = FormulaManager(engine=engine)
    func = getattr(mgr, method)
    # The spec and a frame that carries it are accepted
    rhs = mgr.get_matrices("x + c", data)
    expected = func(mgr.spec)
    assert func(rhs) == expected
    match = "spec_or_frame must be a"
    with pytest.raises(TypeError, match=match):
        func(np.ones((3, 2)))
    # A DataFrame without a model specification attached
    with pytest.raises(TypeError, match=match):
        func(data)
    # A column with the attribute's name is not a model specification
    attr = "design_info" if engine == "patsy" else "model_spec"
    with pytest.raises(TypeError, match=match):
        func(data.assign(**{attr: 1.0}))
    others = [eng for eng in ENGINES if eng != engine]
    for other in others:
        # A design or a spec from the other engine
        other_mgr = FormulaManager(engine=other)
        other_rhs = other_mgr.get_matrices("x + c", data)
        with pytest.raises(TypeError, match=match):
            func(other_rhs)
        with pytest.raises(TypeError, match=match):
            func(other_mgr.spec)


def test_na_action(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    missing = data.copy()
    missing.iloc[::3, 1] = np.nan
    dropper = mgr.get_na_action("drop")
    lhs, rhs = mgr.get_matrices(fmla, missing, na_action=dropper)
    assert rhs.shape[0] == lhs.shape[0]
    assert rhs.shape[0] == 4

    raiser = mgr.get_na_action("raise")
    if engine == "patsy":
        formula_exception = patsy.PatsyError
    else:
        formula_exception = ValueError

    with pytest.raises(formula_exception):
        mgr.get_matrices(fmla, missing, na_action=raiser)

    if engine == "patsy":
        return
    ignorer = mgr.get_na_action("ignore")
    lhs, rhs = mgr.get_matrices(fmla, missing, na_action=ignorer)
    assert lhs.shape[0] == rhs.shape[0]
    assert rhs.shape[0] == missing.shape[0]


@pytest.mark.parametrize("formula", ["y ~ 1 + x + z", "1 + x + z"])
@pytest.mark.parametrize("pandas", [True, False])
@pytest.mark.parametrize("dropped", [True, False])
def test_na_action_drop_keeps_labels(engine, data, formula, pandas, dropped):
    # Rows dropped for missing values: the rows left keep their labels and
    # the mask marks the dropped rows, for one- and two-sided formulas
    mgr = FormulaManager(engine=engine)
    frame = data.copy()
    frame.index = [20, 21, 22, 23, 24, 25, 26]
    if dropped:
        frame.loc[[21, 24], "x"] = np.nan
    dropper = mgr.get_na_action("drop")
    out = mgr.get_matrices(formula, frame, pandas=pandas, na_action=dropper)
    kept = frame.index[frame["x"].notna()]
    np.testing.assert_equal(np.asarray(mgr.missing_mask), frame["x"].isna())
    rhs = out[1] if isinstance(out, tuple) else out
    assert mgr.get_column_names(rhs) == ["Intercept", "x", "z"]
    expected = np.column_stack(
        [np.ones(len(kept)), frame.loc[kept, "x"], frame.loc[kept, "z"]]
    )
    np.testing.assert_allclose(np.asarray(rhs, dtype=float), expected)
    if isinstance(out, tuple):
        lhs_values = np.asarray(out[0], dtype=float)[:, 0]
        np.testing.assert_allclose(lhs_values, frame.loc[kept, "y"])
    if pandas:
        assert list(rhs.index) == list(kept)
        if isinstance(out, tuple):
            assert list(out[0].index) == list(kept)
    else:
        assert isinstance(rhs, np.ndarray)
        assert not isinstance(rhs, pd.DataFrame)
    if engine == "formulaic":
        assert list(mgr.missing_mask.index) == list(frame.index)
        # The spec records the output type, as without missing values
        assert mgr.spec.output == ("pandas" if pandas else "numpy")


@pytest.mark.parametrize("formula", ["y ~ 1 + x + z", "1 + x + z"])
def test_na_action_drop_keeps_index_freq(engine, data, formula):
    # Without missing values, the index of the data is kept with its freq
    mgr = FormulaManager(engine=engine)
    frame = data.copy()
    frame.index = pd.date_range("2000-01-01", periods=7, freq="QS-JAN")
    dropper = mgr.get_na_action("drop")
    out = mgr.get_matrices(formula, frame, na_action=dropper)
    for mat in out if isinstance(out, tuple) else (out,):
        assert mat.index.equals(frame.index)
        assert mat.index.freq == frame.index.freq


@require_formulaic
def test_na_action_drop_unsupported_structure(data):
    # Only a left- and a right-hand side are supported, also when dropping
    mgr = FormulaManager(engine="formulaic")
    formula = formulaic.Formula(lhs="y", rhs="x", other="z")
    with pytest.raises(ValueError, match="not currently supported"):
        mgr.get_matrices(formula, data, na_action=mgr.get_na_action("drop"))


def test_get_term_name_slices(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z + c"
    lhs, rhs = mgr.get_matrices(fmla, data)
    slices = mgr.get_term_name_slices(rhs)
    for i, key in enumerate(slices):
        assert slices[key] == slice(i, i + 1, None)

    slices = mgr.get_term_name_slices(mgr.get_model_spec(rhs))
    for i, key in enumerate(slices):
        assert slices[key] == slice(i, i + 1, None)


@pytest.mark.parametrize(
    "constraint",
    [
        ["Intercept = 0", "x + z = 1"],
        np.eye(4),
        (np.eye(4), np.zeros(4)),
        {"Intercept": 0, "z": 1},
        "x + z = 1",
    ],
)
def test_get_linear_constraints(engine, data, constraint):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z + c"
    lhs, rhs = mgr.get_matrices(fmla, data)
    constraints = mgr.get_linear_constraints(constraint, rhs.columns)
    assert isinstance(constraints[0], np.ndarray)
    assert isinstance(constraints[1], np.ndarray)
    assert isinstance(constraints[2], list)
    assert isinstance(constraints, LinearConstraintValues)

    constraints = mgr.get_linear_constraints(constraint, rhs.columns)
    assert isinstance(constraints[0], np.ndarray)
    assert isinstance(constraints[1], np.ndarray)
    assert isinstance(constraints[2], list)
    assert isinstance(constraints, LinearConstraintValues)


def test_bad_constraint(engine, data):
    mgr = FormulaManager(engine=engine)
    with pytest.raises(ValueError):
        mgr.get_linear_constraints(["x = 0", 7], ["Intercept", "x", "z", "c"])


@pytest.mark.parametrize(
    "constraint, matrix, values",
    [
        ("x1 = 0, x2 = 2*x1 + 1", [[0, 1, 0], [0, -2, 1]], [0, 1]),
        (["x1 = 0", "x2 = 1"], [[0, 1, 0], [0, 0, 1]], [0, 1]),
        ({"x1": 0, "x2": 1.5}, [[0, 1, 0], [0, 0, 1]], [0, 1.5]),
        (np.array([[0, 1, 0], [0, 0, 1]]), [[0, 1, 0], [0, 0, 1]], [0, 0]),
        ([[0, 1, 0]], [[0, 1, 0]], [0]),
        ((np.array([[0, 1, 0]]), np.array([2])), [[0, 1, 0]], [2]),
        (
            (np.array([[0, 1, 0], [0, 0, 1]]), np.array([[2], [3]])),
            [[0, 1, 0], [0, 0, 1]],
            [2, 3],
        ),
    ],
    ids=[
        "str",
        "list-of-str",
        "dict",
        "int-matrix",
        "list-of-lists",
        "matrix-and-values",
        "matrix-and-column-values",
    ],
)
def test_get_linear_constraints_values(engine, constraint, matrix, values):
    names = ["Intercept", "x1", "x2"]
    lc = FormulaManager(engine=engine).get_linear_constraints(constraint, names)
    # Both engines return float arrays, values as a column
    assert lc.constraint_matrix.dtype == np.float64
    assert lc.constraint_values.dtype == np.float64
    np.testing.assert_allclose(lc.constraint_matrix, matrix)
    np.testing.assert_allclose(
        lc.constraint_values, np.asarray(values, dtype=float)[:, None]
    )
    assert list(lc.variable_names) == names


def test_get_linear_constraints_empty_list(engine):
    mgr = FormulaManager(engine=engine)
    with pytest.raises(ValueError, match="Constraints must be non-empty"):
        mgr.get_linear_constraints([], ["Intercept", "x1"])


@no_formulaic
def test_formula_manager_no_formulaic():
    with pytest.raises(ImportError):
        FormulaManager(engine="formulaic")


@no_patsy
def test_formula_manager_no_patsy():
    with pytest.raises(ImportError):
        FormulaManager(engine="patsy")


@require_formulaic
@require_patsy
def test_legacy_orderer(formula):
    rs = np.random.RandomState(0)
    n = 100
    data = pd.DataFrame(
        {
            "y": rs.standard_normal(n),
            "a": rs.standard_normal(n),
            "b": rs.standard_normal(n),
            "c": rs.standard_normal(n),
            "d": pd.Series(
                rs.choice(["a", "b", "c"], size=n),
                dtype="category",
            ),
            "e": pd.Series(
                rs.choice(["a", "b", "c"], size=n),
                dtype="category",
            ),
            "f": pd.Series(
                rs.choice(["a", "b", "c"], size=n),
                dtype="category",
            ),
        }
    )
    mgr = FormulaManager(engine="formulaic")
    ordered_formula = mgr._legacy_orderer(formula, data, 0)
    mm = mgr.get_matrices(ordered_formula, data)
    _, patsy_rhs = patsy.dmatrices(formula, data, return_type="dataframe")

    index = list(patsy_rhs.columns)
    patsy_rhs.columns = index
    assert list(mm[1].columns) == list(patsy_rhs.columns)


def test_get_term_name(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    names = mgr.get_term_names(rhs)
    assert isinstance(names, list)
    for name in names:
        assert isinstance(name, str)


def test_get_description(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    description = mgr.get_description(rhs)
    assert isinstance(description, str)
    description = mgr.get_description(mgr.get_model_spec(rhs))
    assert isinstance(description, str)


def test_get_factor_categories(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    spec = mgr.get_model_spec(rhs)

    term_idx = mgr.get_term_names(spec).index("c")
    term = spec.terms[term_idx]
    factor = term.factors[0]
    cats = mgr.get_factor_categories(factor, spec)
    assert cats == ("a", "b")


def test_get_factors(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    spec = mgr.get_model_spec(rhs)

    factors = mgr.get_factors(spec)
    names = [mgr.get_term_name(factor) for factor in factors]
    assert sorted(names) == ["c", "x", "z"]
    assert mgr.get_factors(rhs) == factors
    factor = factors[names.index("c")]
    assert mgr.get_factor_categories(factor, spec) == ("a", "b")


def test_get_contrast_matrix(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    spec = mgr.get_model_spec(rhs)

    term_idx = mgr.get_term_names(spec).index("c")
    term = spec.terms[term_idx]
    factor = term.factors[0]
    cm = mgr.get_contrast_matrix(term, factor, spec)
    assert isinstance(cm, np.ndarray)
    np.testing.assert_allclose(cm, np.array([[0.0], [1.0]]))


def test_get_slice(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    spec = mgr.get_model_spec(rhs)
    _slice = mgr.get_slice(spec, "c")
    assert _slice == slice(1, 2, None)


def test_get_term_name_slices_alt(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    tns = mgr.get_term_name_slices(rhs)
    keys = ("c", "x", "z")
    keys += ("Intercept",) if engine == "patsy" else ("1",)
    for key in keys:
        assert key in tns
        assert isinstance(tns[key], slice)
    tns = mgr.get_term_name_slices(mgr.get_model_spec(rhs))
    for key in keys:
        assert key in tns
        assert isinstance(tns[key], slice)


def test_get_term_name_alt(engine, data):
    mgr = FormulaManager(engine=engine)
    fomula = "y ~ 1 + x + z + c"
    _, rhs = mgr.get_matrices(fomula, data)
    model_spec = mgr.get_model_spec(rhs)
    terms = model_spec.terms[:]  # copy
    names = ("Intercept",) if engine == "patsy" else ("1",)
    names += ("c", "x", "z")
    for term, name in zip(terms, names, strict=True):
        term_name = mgr.get_term_name(term)
        assert term_name == name, (terms, names)


@pytest.mark.parametrize(
    "lhs, expected",
    [("s", True), ("C(y)", True), ("y", False), ("I(y)", False), ("1", False)],
)
def test_is_categorical(engine, lhs, expected):
    df = pd.DataFrame({"y": [0, 1, 1, 0], "s": list("abba"), "x": [1.0, 2, 3, 4]})
    mgr = FormulaManager(engine=engine)
    lhs_mat, _ = mgr.get_matrices(f"{lhs} ~ x", df)
    assert mgr.is_categorical(lhs_mat) is expected


def test_engine_options_err_message():
    allowed = " or ".join(ENGINES)
    with pytest.raises(ValueError, match=f"Must be {allowed}\\.$"):
        statsmodels.formula.options.formula_engine = "unknown"


def test_engine_options_order_err_message():
    match = "Must be 'degree', 'sort', 'none', or 'legacy'\\.$"
    with pytest.raises(ValueError, match=match):
        statsmodels.formula.options.ordering = "unknown"


def test_engine_error_types(engine, data):
    mgr = FormulaManager(engine=engine)
    with PATSY_ERROR_LOCK:
        with pytest.raises(mgr.factor_evaluation_error, match="undefined_variable"):
            mgr.get_matrices("y ~ x + undefined_variable", data)
        # Neither engine can evaluate a formula on a list
        with pytest.raises(mgr.formula_materializer_error):
            with pytest.warns(DeprecationWarning, match="Using list"):
                mgr.get_matrices("~ x", [[1.0], [2.0]])


def _expected_design(data, columns):
    def column(name):
        if name == "Intercept":
            return np.ones(data.shape[0])
        if name == "c[T.b]":
            return (data["c"] == "b").astype(float)
        return data[name]

    return np.column_stack([column(name) for name in columns])


def test_get_matrices_numpy_output(engine, data):
    mgr = FormulaManager(engine=engine)
    lhs, rhs = mgr.get_matrices("y ~ x + c", data, pandas=False)
    for mat in (lhs, rhs):
        assert isinstance(mat, np.ndarray)
        assert not isinstance(mat, pd.DataFrame)
    columns = mgr.get_column_names(rhs)
    assert sorted(columns) == ["Intercept", "c[T.b]", "x"]
    np.testing.assert_allclose(np.asarray(rhs), _expected_design(data, columns))
    np.testing.assert_allclose(np.asarray(lhs)[:, 0], data["y"])


@pytest.mark.parametrize("data_type", ["dict", "dict-of-scalars", "recarray"])
def test_get_matrices_other_data(engine, data, data_type):
    mgr = FormulaManager(engine=engine)
    if data_type == "dict":
        other = {key: val.to_numpy() for key, val in data.items()}
        expected = data
    elif data_type == "dict-of-scalars":
        other = {"x": 2.5, "z": -1.0}
        expected = pd.DataFrame({"x": [2.5], "z": [-1.0]})
    else:
        other = data[["x", "z"]].to_records(index=False)
        expected = data
    with pytest.warns(DeprecationWarning, match="Using"):
        out = mgr.get_matrices("~ x + z", other)
    columns = mgr.get_column_names(out)
    assert columns == ["Intercept", "x", "z"]
    np.testing.assert_allclose(
        np.asarray(out, dtype=float), _expected_design(expected, columns)
    )


@require_formulaic
@require_patsy
def test_get_matrices_formulaic_eval_environment(data):
    mgr = FormulaManager(engine="formulaic")
    w = np.arange(7.0) ** 2
    env = patsy.eval.EvalEnvironment([{"w": w}])
    with pytest.warns(FutureWarning, match="EvalEnvironment is deprecated"):
        _, rhs = mgr.get_matrices("y ~ x + w", data, eval_env=env)
    np.testing.assert_allclose(np.asarray(rhs["w"]), w)


@require_formulaic
def test_get_matrices_formulaic_eval_env_type(data):
    mgr = FormulaManager(engine="formulaic")
    with pytest.raises(TypeError, match="must be an int or a dict"):
        mgr.get_matrices("y ~ x", data, eval_env="env")


@require_formulaic
def test_get_matrices_formulaic_unsupported_structure(data):
    formula = formulaic.Formula(lhs="y", rhs="x", other="z")
    mgr = FormulaManager(engine="formulaic")
    with pytest.raises(ValueError, match="not currently supported"):
        mgr.get_matrices(formula, data)


def test_get_slice_term_name(engine, data):
    mgr = FormulaManager(engine=engine)
    _, rhs = mgr.get_matrices("y ~ x + z + c", data)
    spec = mgr.spec
    for term in spec.terms:
        name = mgr.get_term_name(term)
        loc = mgr.get_slice(spec, term)
        assert mgr.get_slice(spec, name) == loc
        assert mgr.get_term_name_slices(spec)[name] == loc
        np.testing.assert_allclose(
            np.asarray(rhs, dtype=float)[:, loc],
            _expected_design(data, mgr.get_column_names(spec)[loc]),
        )


def test_remove_intercept_in_place(engine, data):
    mgr = FormulaManager(engine=engine)
    mgr.get_matrices("y ~ x + c", data)
    terms = list(mgr.spec.terms)
    out = mgr.remove_intercept(terms)
    assert out is terms
    assert [mgr.get_term_name(term) for term in out] == ["c", "x"]
    # No intercept to remove
    assert mgr.remove_intercept(out) == out


# Selection of the formula engine when statsmodels.formula is imported, run
# in a subprocess that blocks the imports of the engines
ENGINE_SELECTION = """
import importlib
import json
import os
import sys


def fresh_import(env, blocked):
    for name in list(sys.modules):
        if name == "statsmodels.formula" or name.startswith("statsmodels.formula."):
            del sys.modules[name]
    saved = {name: sys.modules.get(name) for name in blocked}
    sys.modules.update({name: None for name in blocked})
    if env is None:
        os.environ.pop("SM_FORMULA_ENGINE", None)
    else:
        os.environ["SM_FORMULA_ENGINE"] = env
    try:
        return importlib.import_module("statsmodels.formula")
    finally:
        for name, module in saved.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


def engine(env, blocked):
    return fresh_import(env, blocked).options.formula_engine


def set_option(env, blocked, value):
    fresh_import(env, blocked).options.formula_engine = value


def manager(env, blocked, value):
    fresh_import(env, blocked)
    from statsmodels.formula._manager import FormulaManager

    return FormulaManager(engine=value).engine


cases = {
    "invalid": (engine, "unknown", []),
    "formulaic": (engine, "formulaic", []),
    "patsy": (engine, "patsy", []),
    "no-patsy": (engine, None, ["patsy"]),
    "no-patsy-env-patsy": (engine, "patsy", ["patsy"]),
    "no-patsy-option": (set_option, None, ["patsy"], "patsy"),
    "no-patsy-manager": (manager, None, ["patsy"], "patsy"),
    "no-formulaic": (engine, None, ["formulaic"]),
    "no-formulaic-env-formulaic": (engine, "formulaic", ["formulaic"]),
    "no-formulaic-manager": (manager, None, ["formulaic"], "formulaic"),
    "neither": (engine, None, ["patsy", "formulaic"]),
}
results = {}
for key, (func, *args) in cases.items():
    try:
        results[key] = ["ok", func(*args)]
    except Exception as exc:
        results[key] = [type(exc).__name__, str(exc)]
print(json.dumps(results))
"""


@pytest.mark.skipif(PYTHON_IMPL_WASM, reason="no subprocesses in Pyodide")
@pytest.mark.skipif(
    not (HAS_PATSY and HAS_FORMULAIC), reason="requires patsy and formulaic"
)
def test_engine_selection_on_import():
    env = {k: v for k, v in os.environ.items() if k != "SM_FORMULA_ENGINE"}
    proc = subprocess.run(
        [sys.executable, "-c", ENGINE_SELECTION],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert results["invalid"] == [
        "ValueError",
        "Invalid value for SM_FORMULA_ENGINE: unknown",
    ]
    assert results["formulaic"] == ["ok", "formulaic"]
    assert results["patsy"] == ["ok", "patsy"]
    # Without patsy, formulaic is the default and the only option
    assert results["no-patsy"] == ["ok", "formulaic"]
    assert results["no-patsy-env-patsy"][0] == "RuntimeError"
    assert "patsy is not installed" in results["no-patsy-env-patsy"][1]
    assert results["no-patsy-option"] == [
        "ValueError",
        "Invalid formula engine option. Must be formulaic.",
    ]
    assert results["no-patsy-manager"] == [
        "ImportError",
        "patsy is not available. Please install patsy.",
    ]
    assert results["no-formulaic"] == ["ok", "patsy"]
    assert results["no-formulaic-env-formulaic"][0] == "RuntimeError"
    assert "formulaic is not installed" in results["no-formulaic-env-formulaic"][1]
    assert results["no-formulaic-manager"] == [
        "ImportError",
        "formulaic is not available. Please install formulaic.",
    ]
    assert results["neither"] == [
        "ImportError",
        "One of patsy or formulaic must be installed to use statsmodels",
    ]
