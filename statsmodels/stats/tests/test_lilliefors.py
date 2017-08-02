'''
Test Lilliefors corrected K-S tests for normality and exponential distributions

Reference values from R packages:
    `nortest` (lillie.test)
    `KScorrect` (LcKS)

Note : p-values rounded to within the bounds of the Lilliefors tables.
        R implementation provides more precise p-values with a Monte Carlo
        simulation.

Jacob C. Kimmel
2 August 2017
'''

from numpy.testing import assert_, assert_raises, assert_almost_equal
from scipy import stats
import nose
from statsmodels.stats._lilliefors import lilliefors
import numpy as np

class TestLilliefors(object):

    def test_normal(self):
        np.random.seed(3975)
        x_n = stats.norm.rvs(size=500)
        d_ks_norm, p_norm = lilliefors(x_n, dist='norm')
        # shift normal distribution > 0 to exactly mirror R `KScorrect` test
        # R `KScorrect` requires all values tested for exponential
        # distribution to be > 0
        d_ks_exp, p_exp = lilliefors(x_n+np.abs(x_n.min()) + 0.001, dist='exp')
        # assert normal
        assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
        assert_almost_equal(p_norm, 0.2000, decimal=3)
        # assert exp
        assert_almost_equal(d_ks_exp, 0.3436007, decimal=3)
        assert_almost_equal(p_exp, 0.01, decimal=3)


    def test_expon(self):
        np.random.seed(3975)
        x_e = stats.expon.rvs(size=500)
        d_ks_norm, p_norm = lilliefors(x_e, dist='norm')
        d_ks_exp, p_exp = lilliefors(x_e, dist='exp')
        # assert normal
        assert_almost_equal(d_ks_norm, 0.15581, decimal=3)
        assert_almost_equal(p_norm, 2.2e-16, decimal=3)
        # assert exp
        assert_almost_equal(d_ks_exp, 0.02763748, decimal=3)
        assert_almost_equal(p_exp, 0.200, decimal=3)

    def test_pval_bounds(self):
        x = np.arange(1,10)
        d_ks_n, p_n = lilliefors(x, dist='norm')
        d_ks_e, p_e = lilliefors(x, dist='exp')
        assert_almost_equal(p_n, 0.200, decimal=7)
        assert_almost_equal(p_e, 0.200, decimal=7)
