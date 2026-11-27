import numpy as np
import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula import formulatools
from statsmodels.formula._manager import HAVE_FORMULAIC, HAVE_PATSY, FormulaManager
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import handle_formula_data

ENGINES = (["patsy"] if HAVE_PATSY else []) + (["formulaic"] if HAVE_FORMULAIC else [])


@pytest.fixture(params=ENGINES)
def engine(request, monkeypatch):
    monkeypatch.setattr(statsmodels.formula.options, "formula_engine", request.param)
    return request.param


@pytest.fixture
def data():
    return pd.DataFrame(
        {"y": [1.0, 3.0, 2.0, 5.0], "x": [0.5, 1.5, 1.0, 3.0]},
        index=[10, 11, 12, 13],
    )


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
def test_handle_formula_data_exog_must_be_none(engine, data):
    with pytest.raises(ValueError, match="X must be None"):
        handle_formula_data(data[["y"]], data[["x"]], "y ~ x")


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
@pytest.mark.parametrize("nobs", [1, 2, 4])
@pytest.mark.parametrize("formula", ["y ~ x", "~ x", "x"])
def test_handle_formula_data_model_spec(engine, data, nobs, formula):
    # The right-hand-side spec is returned whatever the number of rows
    data = data.iloc[:nobs]
    result, missing_mask, model_spec = handle_formula_data(data, None, formula)
    if formula.startswith("y"):
        endog, exog = result
        np.testing.assert_allclose(np.asarray(endog)[:, 0], data["y"])
        assert list(exog.index) == list(data.index)
    else:
        exog = result
    mgr = FormulaManager()
    assert mgr.get_column_names(model_spec) == ["Intercept", "x"]
    assert mgr.get_column_names(exog) == ["Intercept", "x"]
    np.testing.assert_allclose(np.asarray(exog)[:, 1], data["x"])
    assert missing_mask is None


@pytest.mark.thread_unsafe(reason="changes the global formula engine")
def test_handle_formula_data_missing_mask(engine, data):
    data = data.copy()
    data.loc[11, "x"] = np.nan
    (endog, exog), missing_mask, _ = handle_formula_data(data, None, "y ~ x")
    np.testing.assert_equal(np.asarray(missing_mask), [False, True, False, False])
    assert list(exog.index) == [10, 12, 13]
    np.testing.assert_allclose(np.asarray(endog)[:, 0], [1.0, 2.0, 5.0])


class CustomFormula:
    def __init__(self, text):
        self.text = text


class CustomFormulaSubclass(CustomFormula):
    pass


@pytest.fixture
def handler_calls(monkeypatch):
    calls = []

    def handler(Y, X, formula, depth=0, missing="drop"):
        calls.append((Y, X, formula, depth, missing))
        return handle_formula_data(Y, X, formula.text, depth=depth, missing=missing)

    monkeypatch.setitem(formulatools.formula_handler, CustomFormula, handler)
    return calls


@pytest.mark.thread_unsafe(reason="changes the global formula_handler registry")
@pytest.mark.parametrize("cls", [CustomFormula, CustomFormulaSubclass])
def test_formula_handler_is_called(handler_calls, data, cls):
    formula = cls("y ~ x")
    with pytest.warns(FutureWarning, match="formula_handler is deprecated"):
        out = handle_formula_data(data, None, formula, depth=1, missing="raise")
    assert len(handler_calls) == 1
    y_arg, x_arg, formula_arg, depth, missing = handler_calls[0]
    assert y_arg is data
    assert x_arg is None
    assert formula_arg is formula
    assert depth == 1
    assert missing == "raise"
    (endog, exog), missing_mask, model_spec = out
    np.testing.assert_allclose(np.asarray(endog)[:, 0], data["y"])
    np.testing.assert_allclose(np.asarray(exog)[:, 1], data["x"])
    assert FormulaManager().get_column_names(model_spec) == ["Intercept", "x"]


@pytest.mark.thread_unsafe(reason="changes the global formula_handler registry")
def test_formula_handler_from_formula(handler_calls, data):
    # A registered handler is used by the formula API
    with pytest.warns(FutureWarning, match="formula_handler is deprecated"):
        res = ols(CustomFormula("y ~ x"), data).fit()
    expected = ols("y ~ x", data).fit()
    np.testing.assert_allclose(res.params, expected.params, rtol=1e-12)
    assert list(res.params.index) == ["Intercept", "x"]
    assert len(handler_calls) == 1
