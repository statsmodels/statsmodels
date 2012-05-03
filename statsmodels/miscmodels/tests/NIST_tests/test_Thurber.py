
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

class funcThurber(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6, b7 = params
        return (b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6, b7 = params
        return np.column_stack([1/(1 + b5*x + b6*x**2 + b7*x**3),x/(1 + b5*x + b6*x**2 + b7*x**3),x**2/(1 + b5*x + b6*x**2 + b7*x**3),x**3/(1 + b5*x + b6*x**2 + b7*x**3),(-x)*(b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)**2,(-x**2)*(b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)**2,(-x**3)*(b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)**2])

class TestThurber(TestNonlinearLS):

    def setup(self):
        x = np.array([-3.0670000000000002,
                   -2.9809999999999999,
                   -2.9209999999999998,
                   -2.9119999999999999,
                   -2.8399999999999999,
                   -2.7970000000000002,
                   -2.702,
                   -2.6989999999999998,
                   -2.633,
                   -2.4809999999999999,
                   -2.363,
                   -2.3220000000000001,
                   -1.5009999999999999,
                   -1.46,
                   -1.274,
                   -1.212,
                   -1.1000000000000001,
                   -1.046,
                   -0.91500000000000004,
                   -0.71399999999999997,
                   -0.56599999999999995,
                   -0.54500000000000004,
                   -0.40000000000000002,
                   -0.309,
                   -0.109,
                   -0.10299999999999999,
                   0.01,
                   0.11899999999999999,
                   0.377,
                   0.79000000000000004,
                   0.96299999999999997,
                   1.006,
                   1.115,
                   1.5720000000000001,
                   1.841,
                   2.0470000000000002,
                   2.2000000000000002])
        y = np.array([80.573999999999998,
                   84.248000000000005,
                   87.263999999999996,
                   87.194999999999993,
                   89.075999999999993,
                   89.608000000000004,
                   89.867999999999995,
                   90.100999999999999,
                   92.405000000000001,
                   95.853999999999999,
                   100.696,
                   101.06,
                   401.67200000000003,
                   390.72399999999999,
                   567.53399999999999,
                   635.31600000000003,
                   733.05399999999997,
                   759.08699999999999,
                   894.20600000000002,
                   990.78499999999997,
                   1090.1089999999999,
                   1080.914,
                   1122.643,
                   1178.3510000000001,
                   1260.5309999999999,
                   1273.5139999999999,
                   1288.3389999999999,
                   1327.5429999999999,
                   1353.8630000000001,
                   1414.509,
                   1425.2080000000001,
                   1421.384,
                   1442.962,
                   1464.3499999999999,
                   1468.7049999999999,
                   1447.894,
                   1457.6279999999999])
        mod1 = funcThurber(y, x)
        self.res_start1 = mod1.fit(start_value=[1000.0, 1000.0, 400.0, 40.0, 0.69999999999999996, 0.29999999999999999, 0.029999999999999999])
        mod2 = funcThurber(y, x)
        self.res_start2 = mod2.fit(start_value=[1300.0, 1500.0, 500.0, 75.0, 1.0, 0.40000000000000002, 0.050000000000000003])

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        assert_almost_equal(res1.params,[1288.13968, 1491.0792535, 583.23836876999997, 75.416644290999997, 0.96629502864000005, 0.39797285796999998, 0.049727297348999999],decimal=3)
        assert_almost_equal(res2.params,[1288.13968, 1491.0792535, 583.23836876999997, 75.416644290999997, 0.96629502864000005, 0.39797285796999998, 0.049727297348999999],decimal=3)
