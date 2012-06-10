
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcChwirut2(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.exp(-b1*x)/(b2+b3*x)

class funcChwirut2_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.exp(-b1*x)/(b2+b3*x)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.column_stack([(-x)*np.exp(-b1*x)/(b2+b3*x),(-1)*np.exp(-b1*x)/(b2+b3*x)**2,(-x)*np.exp(-b1*x)/(b2+b3*x)**2])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([0.5,
                   1.0,
                   1.75,
                   3.75,
                   5.75,
                   0.875,
                   2.25,
                   3.25,
                   5.25,
                   0.75,
                   1.75,
                   2.75,
                   4.75,
                   0.625,
                   1.25,
                   2.25,
                   4.25,
                   0.5,
                   3.0,
                   0.75,
                   3.0,
                   1.5,
                   6.0,
                   3.0,
                   6.0,
                   1.5,
                   3.0,
                   0.5,
                   2.0,
                   4.0,
                   0.75,
                   2.0,
                   5.0,
                   0.75,
                   2.25,
                   3.75,
                   5.75,
                   3.0,
                   0.75,
                   2.5,
                   4.0,
                   0.75,
                   2.5,
                   4.0,
                   0.75,
                   2.5,
                   4.0,
                   0.5,
                   6.0,
                   3.0,
                   0.5,
                   2.75,
                   0.5,
                   1.75])
        y = np.array([92.9,
                   57.1,
                   31.05,
                   11.5875,
                   8.025,
                   63.6,
                   21.4,
                   14.25,
                   8.475,
                   63.8,
                   26.8,
                   16.4625,
                   7.125,
                   67.3,
                   41.0,
                   21.15,
                   8.175,
                   81.5,
                   13.12,
                   59.9,
                   14.62,
                   32.9,
                   5.44,
                   12.56,
                   5.44,
                   32.0,
                   13.95,
                   75.8,
                   20.0,
                   10.42,
                   59.5,
                   21.67,
                   8.55,
                   62.0,
                   20.2,
                   7.76,
                   3.75,
                   11.81,
                   54.7,
                   23.7,
                   11.55,
                   61.3,
                   17.7,
                   8.74,
                   59.2,
                   16.3,
                   8.62,
                   81.0,
                   4.87,
                   14.62,
                   81.7,
                   17.17,
                   81.3,
                   28.9])
        self.Degrees_free=51
        self.Res_stddev=3.171713304
        self.Res_sum_squares=513.04802941
        self.start_value2=[0.15, 0.008, 0.01]
        self.Cert_parameters=[0.16657666537, 0.0051653291286, 0.012150007096]
        self.start_value1=[0.1, 0.01, 0.02]
        self.Cert_stddev=[0.03830328681, 0.00066621605126, 0.0015304234767]
        self.Nobs_data=54
        self.nparams=3

        mod1 = funcChwirut2(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcChwirut2(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcChwirut2_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcChwirut2_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
