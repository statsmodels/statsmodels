
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

class funcDanWood(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*x**b2

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return np.column_stack([x**b2,b1*np.log(x)*x**b2])

class TestDanWood(TestNonlinearLS):

    def setup(self):
        x = np.array([1.3089999999999999,
                   1.4710000000000001,
                   1.49,
                   1.5649999999999999,
                   1.611,
                   1.6799999999999999])
        y = np.array([2.1379999999999999,
                   3.4209999999999998,
                   3.597,
                   4.3399999999999999,
                   4.8819999999999997,
                   5.6600000000000001])
        mod1 = funcDanWood(y, x)
        self.res_start1 = mod1.fit(start_value=[1.0, 5.0])
        mod2 = funcDanWood(y, x)
        self.res_start2 = mod2.fit(start_value=[0.69999999999999996, 4.0])

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        assert_almost_equal(res1.params,[0.76886226176000005, 3.8604055870999998],decimal=3)
        assert_almost_equal(res2.params,[0.76886226176000005, 3.8604055870999998],decimal=3)
