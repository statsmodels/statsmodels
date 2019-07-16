from numpy.testing import assert_almost_equal

from statsmodels.datasets import statecrime
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import reset_ramsey
from statsmodels.tools import add_constant

data = statecrime.load_pandas().data


def test_reset_stata():
    mod = OLS(data.violent, add_constant(data[['murder', 'hs_grad']]))
    res = mod.fit()
    stat = reset_ramsey(res, degree=4)
    assert_almost_equal(stat.fvalue[0, 0], 1.52, decimal=2)
    assert_almost_equal(stat.pvalue, 0.2221, decimal=4)
