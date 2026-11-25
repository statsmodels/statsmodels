import numpy as np
import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import HAVE_FORMULAIC, HAVE_PATSY, FormulaManager
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
