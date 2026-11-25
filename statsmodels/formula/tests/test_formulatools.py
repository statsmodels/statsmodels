import pandas as pd
import pytest

import statsmodels.formula
from statsmodels.formula._manager import HAVE_FORMULAIC, HAVE_PATSY
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
