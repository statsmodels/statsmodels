
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

class funcLanczos3(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6 = params
        return b1*np.exp(-b2*x) + b3*np.exp(-b4*x) + b5*np.exp(-b6*x)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6 = params
        return np.column_stack([np.exp(-b2*x),-b1*x*np.exp(-b2*x),np.exp(-b4*x),-b3*x*np.exp(-b4*x),np.exp(-b6*x),-b5*x*np.exp(-b6*x)])

class TestLanczos3(TestNonlinearLS):

    def setup(self):
        x = np.array([0.0,
                   0.050000000000000003,
                   0.10000000000000001,
                   0.14999999999999999,
                   0.20000000000000001,
                   0.25,
                   0.29999999999999999,
                   0.34999999999999998,
                   0.40000000000000002,
                   0.45000000000000001,
                   0.5,
                   0.55000000000000004,
                   0.59999999999999998,
                   0.65000000000000002,
                   0.69999999999999996,
                   0.75,
                   0.80000000000000004,
                   0.84999999999999998,
                   0.90000000000000002,
                   0.94999999999999996,
                   1.0,
                   1.05,
                   1.1000000000000001,
                   1.1499999999999999])
        y = np.array([2.5133999999999999,
                   2.0442999999999998,
                   1.6684000000000001,
                   1.3664000000000001,
                   1.1232,
                   0.92689999999999995,
                   0.76790000000000003,
                   0.63890000000000002,
                   0.53380000000000005,
                   0.44790000000000002,
                   0.37759999999999999,
                   0.31969999999999998,
                   0.27200000000000002,
                   0.23250000000000001,
                   0.19969999999999999,
                   0.17230000000000001,
                   0.14929999999999999,
                   0.13009999999999999,
                   0.1138,
                   0.10000000000000001,
                   0.088300000000000003,
                   0.078299999999999995,
                   0.069800000000000001,
                   0.062399999999999997])
        mod1 = funcLanczos3(y, x)
        self.res_start1 = mod1.fit(start_value=[1.2, 0.29999999999999999, 5.5999999999999996, 5.5, 6.5, 7.5999999999999996])
        mod2 = funcLanczos3(y, x)
        self.res_start2 = mod2.fit(start_value=[0.5, 0.69999999999999996, 3.6000000000000001, 4.2000000000000002, 4.0, 6.2999999999999998])

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        assert_almost_equal(res1.params,[0.086816414977000003, 0.95498101504999999, 0.84400777462999999, 2.9515951831999998, 1.5825685901, 4.9863565084000001],decimal=3)
        assert_almost_equal(res2.params,[0.086816414977000003, 0.95498101504999999, 0.84400777462999999, 2.9515951831999998, 1.5825685901, 4.9863565084000001],decimal=3)
