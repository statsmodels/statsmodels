
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcDanWood(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*x**b2

class funcDanWood_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*x**b2

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return np.column_stack([x**b2,b1*np.log(x)*x**b2])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([1.309,
                   1.471,
                   1.49,
                   1.565,
                   1.611,
                   1.68])
        y = np.array([2.138,
                   3.421,
                   3.597,
                   4.34,
                   4.882,
                   5.66])
        self.Degrees_free=4
        self.Res_stddev=0.032853114039
        self.Res_sum_squares=0.0043173084083
        self.start_value2=[0.7, 4.0]
        self.Cert_parameters=[0.76886226176, 3.8604055871]
        self.start_value1=[1.0, 5.0]
        self.Cert_stddev=[0.01828197386, 0.051726610913]
        self.Nobs_data=6
        self.nparams=2

        mod1 = funcDanWood(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcDanWood(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcDanWood_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcDanWood_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
