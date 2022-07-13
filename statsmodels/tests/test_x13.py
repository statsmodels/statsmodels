import pandas as pd
from statsmodels.tsa.x13 import _make_var_names


def test_make_var_names():
    # Check that _make_var_names return the corrent name when exog is pandas Series
    exog = pd.Series([1,2,3], name='abc')
    assert _make_var_names(exog) == exog.name
