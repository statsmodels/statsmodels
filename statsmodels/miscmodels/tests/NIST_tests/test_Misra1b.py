
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
 
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS
from numpy.testing import assert_almost_equal, assert_

class TestNonlinearLS(object):
    pass

class funcMisra1b(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-(1+b2*x/2)**(-2))

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return np.column_stack([1-(1+b2*x/2)**(-2),b1*x*(1+b2*x/2)**(-3)])

class TestMisra1b(TestNonlinearLS):

    def setup(self):
        x = np.array([77.599999999999994,
                   114.90000000000001,
                   141.09999999999999,
                   190.80000000000001,
                   239.90000000000001,
                   289.0,
                   332.80000000000001,
                   378.39999999999998,
                   434.80000000000001,
                   477.30000000000001,
                   536.79999999999995,
                   593.10000000000002,
                   689.10000000000002,
                   760.0])
        y = np.array([10.07,
                   14.73,
                   17.940000000000001,
                   23.93,
                   29.609999999999999,
                   35.18,
                   40.020000000000003,
                   44.82,
                   50.759999999999998,
                   55.049999999999997,
                   61.009999999999998,
                   66.400000000000006,
                   75.469999999999999,
                   81.780000000000001])
        mod1 = funcMisra1b(y, x)
        self.res_start1 = mod1.fit(start_value=[500.0, 0.0001])
        mod2 = funcMisra1b(y, x)
        self.res_start2 = mod2.fit(start_value=[300.0, 0.00020000000000000001])

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        assert_almost_equal(res1.params,[337.99746162999998, 0.00039039091286999998],decimal=3)
        assert_almost_equal(res2.params,[337.99746162999998, 0.00039039091286999998],decimal=3)
