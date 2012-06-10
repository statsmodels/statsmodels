
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcLanczos1(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6 = params
        return b1*np.exp(-b2*x) + b3*np.exp(-b4*x) + b5*np.exp(-b6*x)

class funcLanczos1_J(NonlinearLS):

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
                   2.044333373291,
                   1.668404436564,
                   1.366418021208,
                   1.123232487372,
                   0.9268897180037,
                   0.7679338563728,
                   0.6388775523106,
                   0.5337835317402,
                   0.4479363617347,
                   0.377584788435,
                   0.3197393199326,
                   0.2720130773746,
                   0.2324965529032,
                   0.1996589546065,
                   0.1722704126914,
                   0.1493405660168,
                   0.1300700206922,
                   0.1138119324644,
                   0.1000415587559,
                   0.0883320908454,
                   0.0783354401935,
                   0.06976693743449,
                   0.06239312536719])
        self.Degrees_free=18
        self.Res_stddev=8.9156129349e-14
        self.Res_sum_squares=1.4307867721e-25
        self.start_value2=[0.5, 0.7, 3.6, 4.2, 4.0, 6.3]
        self.Cert_parameters=[0.095100000027, 1.0000000001, 0.86070000013, 3.0000000002, 1.5575999998, 5.0000000001]
        self.start_value1=[1.2, 0.3, 5.6, 5.5, 6.5, 7.6]
        self.Cert_stddev=[5.3347304234e-11, 2.7473038179e-10, 1.3576062225e-10, 3.3308253069e-10, 1.8815731448e-10, 1.1057500538e-10]
        self.Nobs_data=24
        self.nparams=6

        mod1 = funcLanczos1(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcLanczos1(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcLanczos1_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcLanczos1_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
