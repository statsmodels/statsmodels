
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

class funcChwirut2(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.exp(-b1*x)/(b2+b3*x)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.column_stack([(-x)*np.exp(-b1*x)/(b2+b3*x),(-1)*np.exp(-b1*x)/(b2+b3*x)**2,(-x)*np.exp(-b1*x)/(b2+b3*x)**2])

class TestChwirut2(TestNonlinearLS):

    def setup(self):
        x = np.array([0.5,
                   1.0,
                   1.75,
                   3.75,
                   5.75,
                   0.875,
                   2.25,
                   3.25,
                   5.25,
                   0.75,
                   1.75,
                   2.75,
                   4.75,
                   0.625,
                   1.25,
                   2.25,
                   4.25,
                   0.5,
                   3.0,
                   0.75,
                   3.0,
                   1.5,
                   6.0,
                   3.0,
                   6.0,
                   1.5,
                   3.0,
                   0.5,
                   2.0,
                   4.0,
                   0.75,
                   2.0,
                   5.0,
                   0.75,
                   2.25,
                   3.75,
                   5.75,
                   3.0,
                   0.75,
                   2.5,
                   4.0,
                   0.75,
                   2.5,
                   4.0,
                   0.75,
                   2.5,
                   4.0,
                   0.5,
                   6.0,
                   3.0,
                   0.5,
                   2.75,
                   0.5,
                   1.75])
        y = np.array([92.900000000000006,
                   57.100000000000001,
                   31.050000000000001,
                   11.5875,
                   8.0250000000000004,
                   63.600000000000001,
                   21.399999999999999,
                   14.25,
                   8.4749999999999996,
                   63.799999999999997,
                   26.800000000000001,
                   16.462499999999999,
                   7.125,
                   67.299999999999997,
                   41.0,
                   21.149999999999999,
                   8.1750000000000007,
                   81.5,
                   13.119999999999999,
                   59.899999999999999,
                   14.619999999999999,
                   32.899999999999999,
                   5.4400000000000004,
                   12.56,
                   5.4400000000000004,
                   32.0,
                   13.949999999999999,
                   75.799999999999997,
                   20.0,
                   10.42,
                   59.5,
                   21.670000000000002,
                   8.5500000000000007,
                   62.0,
                   20.199999999999999,
                   7.7599999999999998,
                   3.75,
                   11.81,
                   54.700000000000003,
                   23.699999999999999,
                   11.550000000000001,
                   61.299999999999997,
                   17.699999999999999,
                   8.7400000000000002,
                   59.200000000000003,
                   16.300000000000001,
                   8.6199999999999992,
                   81.0,
                   4.8700000000000001,
                   14.619999999999999,
                   81.700000000000003,
                   17.170000000000002,
                   81.299999999999997,
                   28.899999999999999])
        mod1 = funcChwirut2(y, x)
        self.res_start1 = mod1.fit(start_value=[0.10000000000000001, 0.01, 0.02])
        mod2 = funcChwirut2(y, x)
        self.res_start2 = mod2.fit(start_value=[0.14999999999999999, 0.0080000000000000002, 0.01])

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        assert_almost_equal(res1.params,[0.16657666536999999, 0.0051653291286000002, 0.012150007096],decimal=3)
        assert_almost_equal(res2.params,[0.16657666536999999, 0.0051653291286000002, 0.012150007096],decimal=3)
