import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import FormulaManager

try:
    import formulaic
    import patsy
except ImportError:
    pytestmark = pytest.mark.skip(reason="patsy or formulaic not installed")


@pytest.fixture(params=["patsy", "formulaic"])
def engine(request):
    return request.param


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


def test_engine_options():
    default = statsmodels.formula.options.formula_engine
    assert default in ("patsy", "formulaic")

    statsmodels.formula.options.formula_engine = "patsy"
    mgr = FormulaManager()
    assert mgr.engine == "patsy"
    mgr = FormulaManager(engine="formulaic")
    assert mgr.engine == "formulaic"

    statsmodels.formula.options.formula_engine = "formulaic"
    mgr = FormulaManager()
    assert mgr.engine == "formulaic"

    mgr = FormulaManager(engine="patsy")
    assert mgr.engine == "patsy"

    statsmodels.formula.options.formula_engine = default


def test_engine(engine):
    mgr = FormulaManager(engine=engine)
    assert mgr.engine == engine


def test_single_array(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "1 + x + z"
    output = mgr.get_arrays(fmla, data)
    assert check_type(output, engine)


def test_two_arrays(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_arrays(fmla, data)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert check_type(output[0], engine)
    assert check_type(output[1], engine)


def test_get_column_names(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + z"
    output = mgr.get_arrays(fmla, data)
    names = mgr.get_column_names(output[0])
    assert names == ["y"]

    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "z"]


def test_get_empty_eval(engine, data):
    mgr = FormulaManager(engine=engine)
    fmla = "y ~ 1 + x + g(z)"
    output = mgr.get_arrays(fmla, data)
    names = mgr.get_column_names(output[0])
    assert names == ["y"]

    names = mgr.get_column_names(output[1])
    assert names == ["Intercept", "x", "z"]


"""
 'get_empty_eval_env',
 'get_linear_constraints',
 'get_model_spec',
 'get_na_action',
 'get_spec',
 'get_term_name_slices',
 'has_intercept',
 'intercept_idx',
 'remove_intercept',
 'spec'
"""
