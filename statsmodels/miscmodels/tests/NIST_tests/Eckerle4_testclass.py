
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcEckerle4(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return (b1/b2) * np.exp(-0.5*((x-b3)/b2)**2)

class funcEckerle4_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return (b1/b2) * np.exp(-0.5*((x-b3)/b2)**2)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3 = params
        return np.column_stack([(1/b2) * np.exp(-0.5*((x-b3)/b2)**2),(-b1/b2**2) * np.exp(-0.5*((x-b3)/b2)**2) + (b1/b2) * np.exp(-0.5*((x-b3)/b2)**2) * ((x-b3)/b2**3), (b1/b2) * np.exp(-0.5*((x-b3)/b2)**2) * ((x-b3)/b2)])

class TestNonlinearLS(object):
    def setup(self):
        x = np.array([400.0,
                   405.0,
                   410.0,
                   415.0,
                   420.0,
                   425.0,
                   430.0,
                   435.0,
                   436.5,
                   438.0,
                   439.5,
                   441.0,
                   442.5,
                   444.0,
                   445.5,
                   447.0,
                   448.5,
                   450.0,
                   451.5,
                   453.0,
                   454.5,
                   456.0,
                   457.5,
                   459.0,
                   460.5,
                   462.0,
                   463.5,
                   465.0,
                   470.0,
                   475.0,
                   480.0,
                   485.0,
                   490.0,
                   495.0,
                   500.0])
        y = np.array([0.0001575,
                   0.0001699,
                   0.000235,
                   0.0003102,
                   0.0004917,
                   0.000871,
                   0.0017418,
                   0.00464,
                   0.0065895,
                   0.0097302,
                   0.0149002,
                   0.023731,
                   0.0401683,
                   0.0712559,
                   0.1264458,
                   0.2073413,
                   0.2902366,
                   0.3445623,
                   0.3698049,
                   0.3668534,
                   0.3106727,
                   0.2078154,
                   0.1164354,
                   0.0616764,
                   0.03372,
                   0.0194023,
                   0.0117831,
                   0.0074357,
                   0.0022732,
                   0.00088,
                   0.0004579,
                   0.0002345,
                   0.0001586,
                   0.0001143,
                   7.1e-05])
        self.Degrees_free=32
        self.Res_stddev=0.0067629245447
        self.Res_sum_squares=0.0014635887487
        self.start_value2=[1.5, 5.0, 450.0]
        self.Cert_parameters=[1.5543827178, 4.0888321754, 451.54121844]
        self.start_value1=[1.0, 10.0, 500.0]
        self.Cert_stddev=[0.015408051163, 0.046803020753, 0.046800518816]
        self.Nobs_data=35
        self.nparams=3

        mod1 = funcEckerle4(y, x)
        self.res_start1 = mod1.fit(self.start_value1)
        mod2 = funcEckerle4(y, x)
        self.res_start2 = mod2.fit(self.start_value2)
        mod1_J = funcEckerle4_J(y, x)
        self.resJ_start1 = mod1_J.fit(self.start_value1)
        mod2_J = funcEckerle4_J(y, x)
        self.resJ_start2 = mod2_J.fit(self.start_value2)
