
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMGH17(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5 = params
        return b1 + b2*np.exp(-x*b4) + b3*np.exp(-x*b5)

class funcMGH17_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5 = params
        return b1 + b2*np.exp(-x*b4) + b3*np.exp(-x*b5)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5 = params
        return np.column_stack([np.ones(33),np.exp(-x*b4),np.exp(-x*b5),(-x)*b2*np.exp(-x*b4),(-x)*b3*np.exp(-x*b5)])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([0.0,
                   10.0,
                   20.0,
                   30.0,
                   40.0,
                   50.0,
                   60.0,
                   70.0,
                   80.0,
                   90.0,
                   100.0,
                   110.0,
                   120.0,
                   130.0,
                   140.0,
                   150.0,
                   160.0,
                   170.0,
                   180.0,
                   190.0,
                   200.0,
                   210.0,
                   220.0,
                   230.0,
                   240.0,
                   250.0,
                   260.0,
                   270.0,
                   280.0,
                   290.0,
                   300.0,
                   310.0,
                   320.0])
        y = np.array([0.844,
                   0.908,
                   0.932,
                   0.936,
                   0.925,
                   0.908,
                   0.881,
                   0.85,
                   0.818,
                   0.784,
                   0.751,
                   0.718,
                   0.685,
                   0.658,
                   0.628,
                   0.603,
                   0.58,
                   0.558,
                   0.538,
                   0.522,
                   0.506,
                   0.49,
                   0.478,
                   0.467,
                   0.457,
                   0.448,
                   0.438,
                   0.431,
                   0.424,
                   0.42,
                   0.414,
                   0.411,
                   0.406])
        self.Degrees_free=28
        self.Res_stddev=0.0013970497866
        self.Res_sum_squares=5.4648946975e-05
        self.start_value2=[0.5, 1.5, -1.0, 0.01, 0.02]
        self.Cert_parameters=[0.37541005211, 1.9358469127, -1.4646871366, 0.01286753464, 0.022122699662]
        self.start_value1=[50.0, 150.0, -100.0, 1.0, 2.0]
        self.Cert_stddev=[0.0020723153551, 0.22031669222, 0.22175707739, 0.00044861358114, 0.00089471996575]
        self.Nobs_data=33
        self.nparams=5

        mod1 = funcMGH17(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcMGH17(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcMGH17_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcMGH17_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
