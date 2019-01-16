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
        var = 0.760853
        std = 0.872269
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

class TestSim2(CheckExternalMixin):


        data = [[25, 'Bob',  True,  1.2],
                [41, 'John',  False, 0.5],
                [30, 'Alice', True,  0.3]]
        nobs = [3, 3]
        mean = [32.000000, 0.666667]
        var = [67.000000, 0.223333]
        std = [8.185353, 0.472582]
        per = [[25.10,  0.304],
               [25.50,  0.320],
               [26.00,  0.340],
               [27.50,  0.400],
               [30.00,  0.500],
               [35.50,  0.850],
               [38.80,  1.060],
               [39.90,  1.130],
               [40.78,  1.186]]

        @classmethod
        def setup_class(cls):
            cls.get_descriptives()

class TestSim3(CheckExternalMixin):

        data = np.array([[1,2,3,4,5,6],
                          [6,5,4,3,2,1],
                          [9,9,9,9,9,9]])
        nobs = [3, 3, 3, 3, 3, 3]
        mean = [5.33333333, 5.33333333, 5.33333333, 5.33333333, 5.33333333,
            5.33333333]
        var = [16.33333333, 12.33333333, 10.33333333, 10.33333333, 12.33333333,
            16.33333333]
        std = [4.04145188, 3.51188458, 3.21455025, 3.21455025, 3.51188458,
            4.04145188]
        per = [[1.1 , 2.06, 3.02, 3.02, 2.06, 1.1 ],
               [1.5 , 2.3 , 3.1 , 3.1 , 2.3 , 1.5 ],
               [2.  , 2.6 , 3.2 , 3.2 , 2.6 , 2.  ],
               [3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 ],
               [6.  , 5.  , 4.  , 4.  , 5.  , 6.  ],
               [7.5 , 7.  , 6.5 , 6.5 , 7.  , 7.5 ],
               [8.4 , 8.2 , 8.  , 8.  , 8.2 , 8.4 ],
               [8.7 , 8.6 , 8.5 , 8.5 , 8.6 , 8.7 ],
               [8.94, 8.92, 8.9 , 8.9 , 8.92, 8.94]]


        @classmethod
        def setup_class(cls):
            cls.get_descriptives()

class TestSim4(CheckExternalMixin):

        t3 = TestSim3()
        data = pd.DataFrame(t3.data)
        nobs = t3.nobs
        mean = t3.mean
        var = t3.var
        std = t3.std
        per = t3.per

        @classmethod
        def setup_class(cls):
            cls.get_descriptives()
