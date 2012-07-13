
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcLanczos3(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6 = params
        return b1*np.exp(-b2*x) + b3*np.exp(-b4*x) + b5*np.exp(-b6*x)

class funcLanczos3_J(NonlinearLS):

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

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([0.0,
                   0.05,
                   0.1,
                   0.15,
                   0.2,
                   0.25,
                   0.3,
                   0.35,
                   0.4,
                   0.45,
                   0.5,
                   0.55,
                   0.6,
                   0.65,
                   0.7,
                   0.75,
                   0.8,
                   0.85,
                   0.9,
                   0.95,
                   1.0,
                   1.05,
                   1.1,
                   1.15])
        y = np.array([2.5134,
                   2.0443,
                   1.6684,
                   1.3664,
                   1.1232,
                   0.9269,
                   0.7679,
                   0.6389,
                   0.5338,
                   0.4479,
                   0.3776,
                   0.3197,
                   0.272,
                   0.2325,
                   0.1997,
                   0.1723,
                   0.1493,
                   0.1301,
                   0.1138,
                   0.1,
                   0.0883,
                   0.0783,
                   0.0698,
                   0.0624])
        self.Degrees_free=18
        self.Res_stddev=2.9923229172e-05
        self.Res_sum_squares=1.6117193594e-08
        self.start_value2=[0.5, 0.7, 3.6, 4.2, 4.0, 6.3]
        self.Cert_parameters=[0.086816414977, 0.95498101505, 0.84400777463, 2.9515951832, 1.5825685901, 4.9863565084]
        self.start_value1=[1.2, 0.3, 5.6, 5.5, 6.5, 7.6]
        self.Cert_stddev=[0.017197908859, 0.097041624475, 0.041488663282, 0.10766312506, 0.058371576281, 0.034436403035]
        self.Nobs_data=24
        self.nparams=6

        mod1 = funcLanczos3(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcLanczos3(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcLanczos3_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcLanczos3_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
