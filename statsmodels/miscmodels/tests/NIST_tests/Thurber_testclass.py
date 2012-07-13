
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcThurber(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6, b7 = params
        return (b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)

class funcThurber_J(NonlinearLS):

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

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([-3.067,
                   -2.981,
                   -2.921,
                   -2.912,
                   -2.84,
                   -2.797,
                   -2.702,
                   -2.699,
                   -2.633,
                   -2.481,
                   -2.363,
                   -2.322,
                   -1.501,
                   -1.46,
                   -1.274,
                   -1.212,
                   -1.1,
                   -1.046,
                   -0.915,
                   -0.714,
                   -0.566,
                   -0.545,
                   -0.4,
                   -0.309,
                   -0.109,
                   -0.103,
                   0.01,
                   0.119,
                   0.377,
                   0.79,
                   0.963,
                   1.006,
                   1.115,
                   1.572,
                   1.841,
                   2.047,
                   2.2])
        y = np.array([80.574,
                   84.248,
                   87.264,
                   87.195,
                   89.076,
                   89.608,
                   89.868,
                   90.101,
                   92.405,
                   95.854,
                   100.696,
                   101.06,
                   401.672,
                   390.724,
                   567.534,
                   635.316,
                   733.054,
                   759.087,
                   894.206,
                   990.785,
                   1090.109,
                   1080.914,
                   1122.643,
                   1178.351,
                   1260.531,
                   1273.514,
                   1288.339,
                   1327.543,
                   1353.863,
                   1414.509,
                   1425.208,
                   1421.384,
                   1442.962,
                   1464.35,
                   1468.705,
                   1447.894,
                   1457.628])
        self.Degrees_free=30
        self.Res_stddev=13.714600784
        self.Res_sum_squares=5642.7082397
        self.start_value2=[1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05]
        self.Cert_parameters=[1288.13968, 1491.0792535, 583.23836877, 75.416644291, 0.96629502864, 0.39797285797, 0.049727297349]
        self.start_value1=[1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03]
        self.Cert_stddev=[4.6647963344, 39.571156086, 28.698696102, 5.567537027, 0.031333340687, 0.014984928198, 0.0065842344623]
        self.Nobs_data=37
        self.nparams=7

        mod1 = funcThurber(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcThurber(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcThurber_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcThurber_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
