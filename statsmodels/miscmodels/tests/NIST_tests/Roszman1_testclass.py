
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcRoszman1(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return b1 - b2*x - np.arctan(b3/(x-b4))/np.pi

class funcRoszman1_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return b1 - b2*x - np.arctan(b3/(x-b4))/np.pi

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return np.column_stack([np.ones(25),-x,1/(np.pi*(x-b4)*(1+(b3/(x-b4))**2)),b3/(np.pi*(x-b4)**2*(1+(b3/(x-b4))**2))])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([-4868.68,
                   -4868.09,
                   -4867.41,
                   -3375.19,
                   -3373.14,
                   -3372.03,
                   -2473.74,
                   -2472.35,
                   -2469.45,
                   -1894.65,
                   -1893.4,
                   -1497.24,
                   -1495.85,
                   -1493.41,
                   -1208.68,
                   -1206.18,
                   -1206.04,
                   -997.92,
                   -996.61,
                   -996.31,
                   -834.94,
                   -834.66,
                   -710.03,
                   -530.16,
                   -464.17])
        y = np.array([0.252429,
                   0.252141,
                   0.251809,
                   0.297989,
                   0.296257,
                   0.295319,
                   0.339603,
                   0.337731,
                   0.33382,
                   0.38951,
                   0.386998,
                   0.438864,
                   0.434887,
                   0.427893,
                   0.471568,
                   0.461699,
                   0.461144,
                   0.513532,
                   0.506641,
                   0.505062,
                   0.535648,
                   0.533726,
                   0.568064,
                   0.612886,
                   0.624169])
        self.Degrees_free=21
        self.Res_stddev=0.004854298406
        self.Res_sum_squares=0.00049484847331
        self.start_value2=[0.2, -5e-06, 1200.0, -150.0]
        self.Cert_parameters=[0.20196866396, -6.1953516256e-06, 1204.4556708, -181.34269537]
        self.start_value1=[0.1, -1e-05, 1000.0, -100.0]
        self.Cert_stddev=[0.019172666023, 3.2058931691e-06, 74.050983057, 49.573513849]
        self.Nobs_data=25
        self.nparams=4

        mod1 = funcRoszman1(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcRoszman1(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcRoszman1_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcRoszman1_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
