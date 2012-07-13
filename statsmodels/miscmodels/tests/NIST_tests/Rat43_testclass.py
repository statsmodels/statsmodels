
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcRat43(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return b1 / (1+np.exp(b2-b3*x))**(1/b4)

class funcRat43_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return b1 / (1+np.exp(b2-b3*x))**(1/b4)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4 = params
        return np.column_stack([1 / (1+np.exp(b2-b3*x))**(1/b4), (-1/b4)* b1 * np.exp(b2-b3*x)/ ((1+np.exp(b2-b3*x))**(1/b4 + 1)), (x/b4)* b1 * np.exp(b2-b3*x) / ((1+np.exp(b2-b3*x))**(1/b4 + 1)), (b1/b4**2) * (1+np.exp(b2-b3*x))**(-1/b4) * np.log(1+np.exp(b2-b3*x))])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([1.0,
                   2.0,
                   3.0,
                   4.0,
                   5.0,
                   6.0,
                   7.0,
                   8.0,
                   9.0,
                   10.0,
                   11.0,
                   12.0,
                   13.0,
                   14.0,
                   15.0])
        y = np.array([16.08,
                   33.83,
                   65.8,
                   97.2,
                   191.55,
                   326.2,
                   386.87,
                   520.53,
                   590.03,
                   651.92,
                   724.93,
                   699.56,
                   689.96,
                   637.56,
                   717.41])
        self.Degrees_free=9
        self.Res_stddev=28.262414662
        self.Res_sum_squares=8786.404908
        self.start_value2=[700.0, 5.0, 0.75, 1.3]
        self.Cert_parameters=[699.6415127, 5.2771253025, 0.75962938329, 1.2792483859]
        self.start_value1=[100.0, 10.0, 1.0, 1.0]
        self.Cert_stddev=[16.302297817, 2.0828735829, 0.19566123451, 0.68761936385]
        self.Nobs_data=15
        self.nparams=4

        mod1 = funcRat43(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcRat43(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcRat43_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcRat43_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
