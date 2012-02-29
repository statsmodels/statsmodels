import pandas
import numpy as np

from scikits.statsmodels.tools import data

def test_missing_data_pandas():
    """
    Fixes GH: #144
    """
    X = np.random.random((10,5))
    X[1,2] = np.nan
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(rnames, [0,2,3,4,5,6,7,8,9])

def test_structarray():
    X = np.random.random((10,)).astype([('var1', 'f8'),
                                        ('var2', 'f8'),
                                        ('var3', 'f8')])
    vals, cnames, rnames = data.interpret_data(X)
    np.testing.assert_equal(cnames, X.dtype.names)
    np.testing.assert_equal(vals, X.view((float,3)))
    np.testing.assert_equal(rnames, None)

def test_recarray():
    X = np.random.random((10,)).astype([('var1', 'f8'),
                                        ('var2', 'f8'),
                                        ('var3', 'f8')])
    vals, cnames, rnames = data.interpret_data(X.view(np.recarray))
    np.testing.assert_equal(cnames, X.dtype.names)
    np.testing.assert_equal(vals, X.view((float,3)))
    np.testing.assert_equal(rnames, None)


def test_dataframe():
    X = np.random.random((10,5))
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(vals, df.values)
    np.testing.assert_equal(rnames, df.index)
    np.testing.assert_equal(cnames, df.columns)
