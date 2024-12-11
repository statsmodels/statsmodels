import numpy as np
import pandas

from statsmodels.tools import data


def test_missing_data_pandas():
    """
    Fixes GH: #144
    """
    X = np.random.random((10, 5))
    X[1, 2] = np.nan
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(rnames.tolist(), [0, 2, 3, 4, 5, 6, 7, 8, 9])


def test_dataframe():
    X = np.random.random((10, 5))
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(vals, df.values)
    np.testing.assert_equal(rnames.tolist(), df.index.tolist())
    np.testing.assert_equal(cnames, df.columns.tolist())


def test_formula_engine_use_detection_577():
    x = np.random.random((10, 2))
    df = pandas.DataFrame(x, columns=["var1", "var2"])
    from statsmodels.formula._manager import FormulaManager
    mgr = FormulaManager()
    if mgr.engine == "patsy":
        test_func = data._is_using_patsy
    else:
        test_func = data._is_using_formulaic
    endog = mgr.get_matrices("var1 - 1", df, pandas=False)
    assert test_func(endog, None)

    exog = mgr.get_matrices("var2 - 1", df, pandas=False)
    assert test_func(endog, exog)


def test_as_array_with_name_series():
    s = pandas.Series([1], name="hello")
    arr, name = data._as_array_with_name(s, "not_used")
    np.testing.assert_array_equal(np.array([1]), arr)
    assert name == "hello"


def test_as_array_with_name_array():
    arr, name = data._as_array_with_name(np.array([1]), "default")
    np.testing.assert_array_equal(np.array([1]), arr)
    assert name == "default"
