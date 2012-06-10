
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMGH10(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return b1 * np.exp(b2/(x+b3))

class funcMGH10_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return b1 * np.exp(b2/(x+b3))

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.column_stack([np.exp(b2/(x+b3)),b1*np.exp(b2/(x+b3))*(1/(x+b3)), b1 * np.exp(b2/(x+b3)) * (-b2/(x+b3)**2)])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([50.0,
                   55.0,
                   60.0,
                   65.0,
                   70.0,
                   75.0,
                   80.0,
                   85.0,
                   90.0,
                   95.0,
                   100.0,
                   105.0,
                   110.0,
                   115.0,
                   120.0,
                   125.0])
        y = np.array([34780.0,
                   28610.0,
                   23650.0,
                   19630.0,
                   16370.0,
                   13720.0,
                   11540.0,
                   9744.0,
                   8261.0,
                   7030.0,
                   6005.0,
                   5147.0,
                   4427.0,
                   3820.0,
                   3307.0,
                   2872.0])
        self.Degrees_free=13
        self.Res_stddev=2.6009740065
        self.Res_sum_squares=87.945855171
        self.start_value2=[0.02, 4000.0, 250.0]
        self.Cert_parameters=[0.005609636471, 6181.3463463, 345.22363462]
        self.start_value1=[2.0, 400000.0, 25000.0]
        self.Cert_stddev=[0.00015687892471, 23.309021107, 0.78486103508]
        self.Nobs_data=16
        self.nparams=3

        mod1 = funcMGH10(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcMGH10(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcMGH10_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcMGH10_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
