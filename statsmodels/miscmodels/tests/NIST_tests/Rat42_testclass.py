
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcRat42(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return b1 / (1+np.exp(b2-b3*x))

class funcRat42_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return b1 / (1+np.exp(b2-b3*x))

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.column_stack([1 / (1+np.exp(b2-b3*x)), -b1*np.exp(b2-b3*x) / (1+np.exp(b2-b3*x))**2, b1*np.exp(b2-b3*x)* x / (1+np.exp(b2-b3*x))**2])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([9.0,
                   14.0,
                   21.0,
                   28.0,
                   42.0,
                   57.0,
                   63.0,
                   70.0,
                   79.0])
        y = np.array([8.93,
                   10.8,
                   18.59,
                   22.33,
                   39.35,
                   56.11,
                   61.73,
                   64.62,
                   67.08])
        self.Degrees_free=6
        self.Res_stddev=1.1587725499
        self.Res_sum_squares=8.0565229338
        self.start_value2=[75.0, 2.5, 0.07]
        self.Cert_parameters=[72.462237576, 2.6180768402, 0.067359200066]
        self.start_value1=[100.0, 1.0, 0.1]
        self.Cert_stddev=[1.7340283401, 0.088295217536, 0.0034465663377]
        self.Nobs_data=9
        self.nparams=3

        mod1 = funcRat42(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcRat42(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcRat42_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcRat42_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
