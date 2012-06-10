
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMisra1c(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-(1+2*b2*x)**(-.5))

class funcMisra1c_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-(1+2*b2*x)**(-.5))

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return np.column_stack([(1-(1+2*b2*x)**(-.5)),b1*x*(1+2*b2*x)**(-1.5)])

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
        self.Res_stddev=0.058428615257
        self.Res_sum_squares=0.040966836971
        self.start_value2=[600.0, 0.0002]
        self.Cert_parameters=[636.42725809, 0.00020813627256]
        self.start_value1=[500.0, 0.0001]
        self.Cert_stddev=[4.6638326572, 1.7728423155e-06]
        self.Nobs_data=14
        self.nparams=2

        mod1 = funcMisra1c(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcMisra1c(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcMisra1c_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcMisra1c_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
