import numpy as np
import pandas as pd
from statsmodels.stats.descriptivestats import (sign_test, DescrStats)
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose)

def test_sign_test():
    x = [7.8, 6.6, 6.5, 7.4, 7.3, 7., 6.4, 7.1, 6.7, 7.6, 6.8]
    M, p = sign_test(x, mu0=6.5)
    # from R SIGN.test(x, md=6.5)
    # from R
    assert_almost_equal(p, 0.02148, 5)
    # not from R, we use a different convention
    assert_equal(M, 4)

class CheckExternalMixin(object):

    @classmethod
    def get_descriptives(cls):
        cls.descriptive = DescrStats(cls.data)

    def test_nobs(self):
        nobs = self.descriptive.nobs.values
        assert_equal(nobs, self.nobs)

    def test_mean(self):
        mn = self.descriptive.mean.values
        assert_allclose(mn, self.mean, rtol=1e-4)

    def test_var(self):
        var = self.descriptive.var.values
        assert_allclose(var, self.var, rtol=1e-4)

    def test_std(self):
        std = self.descriptive.std.values
        assert_allclose(std, self.std, rtol=1e-4)

    def test_percentiles(self):
        per = self.descriptive.percentiles().values
        assert_almost_equal(per, self.per, 1)

class TestSim1(CheckExternalMixin):

        # Taken from R
        nobs = 20
        mean = 0.56930
        var = 0.72281079
        std = 0.85018
        per = [[-0.95387327],
               [-0.86025485],
               [-0.27005201],
               [0.06545155],
               [0.40537786],
               [1.09762186],
               [1.77440291],
               [1.88622475],
               [2.16995951]]

        @classmethod
        def setup_class(cls):
            np.random.seed(0)
            cls.data = np.random.normal(size=20)
            cls.get_descriptives()
