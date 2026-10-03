import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from statsmodels.datasets import statecrime
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import (
    reset_ramsey,
    variance_inflation_factor,
)
from statsmodels.tools import add_constant


def test_reset_stata():
    data = statecrime.load_pandas().data
    mod = OLS(data.violent, add_constant(data[["murder", "hs_grad"]]))
    res = mod.fit()
    stat = reset_ramsey(res, degree=4)
    assert_almost_equal(stat.fvalue, 1.52, decimal=2)
    assert_almost_equal(stat.pvalue, 0.2221, decimal=4)

    exog_idx = list(data.columns).index("urban")
    data_arr = np.asarray(data)
    vif = variance_inflation_factor(data_arr, exog_idx, standardize=False)
    assert_almost_equal(vif, 16.4394, decimal=4)

    exog_idx = list(data.columns).index("urban")
    vif_df = variance_inflation_factor(data, exog_idx, standardize=False)
    assert_almost_equal(vif_df, 16.4394, decimal=4)


def test_reset_ramsey_degree_validation():
    # degrees below 2 have no powers to test and used to leak a bare numpy
    # error from the empty vander matrix
    rs = np.random.RandomState(12345)
    endog = rs.standard_normal(40)
    exog = np.column_stack([np.ones(40), rs.standard_normal((40, 2))])
    res = OLS(endog, exog).fit()
    with pytest.raises(ValueError, match="degree must be an integer >= 2"):
        reset_ramsey(res, degree=-1)
    with pytest.raises(ValueError, match="degree must be an integer >= 2"):
        reset_ramsey(res, degree=1)
    with pytest.raises(TypeError, match="degree"):
        reset_ramsey(res, degree=2.5)
