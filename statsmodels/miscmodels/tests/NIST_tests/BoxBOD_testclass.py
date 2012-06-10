
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcBoxBOD(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-np.exp(-b2*x))

class funcBoxBOD_J(NonlinearLS):

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
        x = np.array([1.0,
                   2.0,
                   3.0,
                   5.0,
                   7.0,
                   10.0])
        y = np.array([109.0,
                   149.0,
                   149.0,
                   191.0,
                   213.0,
                   224.0])
        self.Degrees_free=4
        self.Res_stddev=17.088072423
        self.Res_sum_squares=1168.0088766
        self.start_value2=[100.0, 0.75]
        self.Cert_parameters=[213.80940889, 0.54723748542]
        self.start_value1=[1.0, 1.0]
        self.Cert_stddev=[12.354515176, 0.10455993237]
        self.Nobs_data=6
        self.nparams=2

        mod1 = funcBoxBOD(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcBoxBOD(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcBoxBOD_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcBoxBOD_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
