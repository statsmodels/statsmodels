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
    eval_env = mgr.get_empty_eval_env()
    with PATSY_ERROR_LOCK:
        with pytest.raises(patsy.PatsyError):
            mgr.get_matrices(fmla, data, eval_env=2)

        with pytest.raises(patsy.PatsyError):
            with pytest.warns(FutureWarning, match="EvalEnvironment is deprecate"):
                mgr.get_matrices(fmla, data, eval_env=eval_env)


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
