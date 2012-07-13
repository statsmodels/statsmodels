
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMGH09(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return b1*(x**2+x*b2) / (x**2+x*b3+b4)

class funcMGH09_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return b1*(x**2+x*b2) / (x**2+x*b3+b4)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return np.column_stack([(x**2+x*b2) / (x**2+x*b3+b4), (b1*x*b2) / (x**2+x*b3+b4), ((-1)*b1*(x**2+x*b2) / (x**2+x*b3+b4)**2) * x,(-1)*b1*(x**2+x*b2) / (x**2+x*b3+b4)**2])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([4.0,
                   2.0,
                   1.0,
                   0.5,
                   0.25,
                   0.167,
                   0.125,
                   0.1,
                   0.0833,
                   0.0714,
                   0.0625])
        y = np.array([0.1957,
                   0.1947,
                   0.1735,
                   0.16,
                   0.0844,
                   0.0627,
                   0.0456,
                   0.0342,
                   0.0323,
                   0.0235,
                   0.0246])
        self.Degrees_free=7
        self.Res_stddev=0.0066279236551
        self.Res_sum_squares=0.00030750560385
        self.start_value2=[0.25, 0.39, 0.415, 0.39]
        self.Cert_parameters=[0.19280693458, 0.19128232873, 0.12305650693, 0.13606233068]
        self.start_value1=[25.0, 39.0, 41.5, 39.0]
        self.Cert_stddev=[0.011435312227, 0.19633220911, 0.080842031232, 0.090025542308]
        self.Nobs_data=11
        self.nparams=4

        mod1 = funcMGH09(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcMGH09(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcMGH09_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcMGH09_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
