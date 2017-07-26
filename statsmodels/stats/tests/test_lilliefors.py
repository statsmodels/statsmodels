from numpy.testing import assert_, assert_raises
from scipy import stats
import nose
from statsmodels.stats._lilliefors import lilliefors
import numpy as np

class TestLilliefors(object):

    def test_normal(self):
        np.random.seed(3975)
        x_n = stats.norm.rvs(size=5000)
        d_ks, p = lilliefors(x_n, dist='norm')
        assert_(p > 0.05)

    def test_expon(self):
        np.random.seed(3975)
        x_e = stats.expon.rvs(size=5000)
        d_ks, p = lilliefors(x_e, dist='exp')
        assert_(p > 0.05)
