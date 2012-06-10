
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcLanczos2(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6 = params
        return b1*np.exp(-b2*x) + b3*np.exp(-b4*x) + b5*np.exp(-b6*x)

class funcLanczos2_J(NonlinearLS):

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
        return np.column_stack([np.exp(-b2*x),(-x)*b1*np.exp(-b2*x),np.exp(-b4*x),(-x)*b3*np.exp(-b4*x),np.exp(-b6*x),(-x)*b5*np.exp(-b6*x)])

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
                   2.04433,
                   1.6684,
                   1.36642,
                   1.12323,
                   0.92689,
                   0.767934,
                   0.638878,
                   0.533784,
                   0.447936,
                   0.377585,
                   0.319739,
                   0.272013,
                   0.232497,
                   0.199659,
                   0.17227,
                   0.149341,
                   0.13007,
                   0.113812,
                   0.100042,
                   0.0883321,
                   0.0783354,
                   0.0697669,
                   0.0623931])
        self.Degrees_free=18
        self.Res_stddev=1.1130395851e-06
        self.Res_sum_squares=2.2299428125e-11
        self.start_value2=[0.5, 0.7, 3.6, 4.2, 4.0, 6.3]
        self.Cert_parameters=[0.096251029939, 1.0057332849, 0.86424689056, 3.0078283915, 1.5529016879, 5.00287981]
        self.start_value1=[1.2, 0.3, 5.6, 5.5, 6.5, 7.6]
        self.Cert_stddev=[0.00066770575477, 0.0033989646176, 0.0017185846685, 0.0041707005856, 0.0023744381417, 0.0013958787284]
        self.Nobs_data=24
        self.nparams=6

        mod1 = funcLanczos2(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcLanczos2(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcLanczos2_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcLanczos2_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
