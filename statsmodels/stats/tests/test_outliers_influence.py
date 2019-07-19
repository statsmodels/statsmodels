import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.datasets import statecrime
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import reset_ramsey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

data = statecrime.load_pandas().data


def test_reset_stata():
    mod = OLS(data.violent, add_constant(data[['murder', 'hs_grad']]))
    res = mod.fit()
    stat = reset_ramsey(res, degree=4)
    assert_almost_equal(stat.fvalue[0, 0], 1.52, decimal=2)
    assert_almost_equal(stat.pvalue, 0.2221, decimal=4)


def test_variance_inflation_factor():    
    # Test Case: perfect collinear with constant component
    exog = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]]).T
    exog = add_constant(exog)
    vif = variance_inflation_factor(exog, 1)
    assert_almost_equal(vif, float('+inf'), decimal=4)

    # Test Case: partial collinear with constant component
    exog = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 0, 1, 1, 1, 1]]).T
    exog = add_constant(exog)
    vif = variance_inflation_factor(exog, 1)
    assert_almost_equal(vif, 5., decimal=4)

    # Test Case: no collinear with constant component
    exog = np.array([[0, 0, 0, 0, 2, 2, 2, 2], [0, 2, 0, 2, 0, 2, 0, 2]]).T
    exog = add_constant(exog)
    vif = variance_inflation_factor(exog, 1)
    assert_almost_equal(vif, 1., decimal=4)
