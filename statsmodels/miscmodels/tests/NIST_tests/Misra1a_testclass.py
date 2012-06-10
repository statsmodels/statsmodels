
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMisra1a(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-np.exp(-b2*x))

class funcMisra1a_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-np.exp(-b2*x))

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return np.column_stack([1-np.exp(-b2*x),b1*x*np.exp(-b2*x)])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([77.6,
                   114.9,
                   141.1,
                   190.8,
                   239.9,
                   289.0,
                   332.8,
                   378.4,
                   434.8,
                   477.3,
                   536.8,
                   593.1,
                   689.1,
                   760.0])
        y = np.array([10.07,
                   14.73,
                   17.94,
                   23.93,
                   29.61,
                   35.18,
                   40.02,
                   44.82,
                   50.76,
                   55.05,
                   61.01,
                   66.4,
                   75.47,
                   81.78])
        self.Degrees_free=12
        self.Res_stddev=0.1018787633
        self.Res_sum_squares=0.12455138894
        self.start_value2=[250.0, 0.0005]
        self.Cert_parameters=[238.94212918, 0.00055015643181]
        self.start_value1=[500.0, 0.0001]
        self.Cert_stddev=[2.7070075241, 7.2668688436e-06]
        self.Nobs_data=14
        self.nparams=2

        mod1 = funcMisra1a(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcMisra1a(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcMisra1a_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcMisra1a_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
