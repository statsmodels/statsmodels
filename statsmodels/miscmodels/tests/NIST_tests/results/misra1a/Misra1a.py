DESCRIPTION = '''
This script contains the NonlinearLS class derived model
definitions using both the analytical jacobian and the 
numerical derivative jacobian.
Also it contains result statistics for the particular model
obtained from the NIST files and Gretl
'''

import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMisra1a(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-np.exp(-b2*x))

class funcMisra1a_J(NonlinearLS):

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

class Misra1a(object):
    '''
    Results for nonlinear regression fitted NIST model y = b1*(1-exp[-b2*x]) + e 
    with Misra data
    '''
    def __init__(self):

        self.params = [238.94212918, 0.00055015643181]
        self.bse = [2.7070075241, 7.2668688436e-06]
        self.df_resid = 12
        self.df_model = 1
        self.ssr = 0.12455138894
        self.ser=0.1018787633
        self.start_value2=[250.0, 0.0005]
        self.start_value1=[500.0, 0.0001]
        self.nobs=14
        self.nparams=2

        self.scale = 0.010379282412
        self.mse_resid = 0.010379282412
        self.rsquared = 0.99998158011
        self.rsquared_adj = 0.999980045119
        self.tvalues = [88.2680013486642, 75.7074993836574]
        self.pvalues = np.array([2.98563089249121e-18, 1.87789823791948e-17])
        self.aic = -22.3790400838
        self.bic = -21.1009254245
        self.hqc = -22.4973529587
        self.fittedvalues = np.array([9.986266361548,14.636752697085,17.846722502904,
                                      23.810184059939,29.543687342833,35.124386448710,
                                      39.977050598374,44.906423598336,50.834674167910,
                                      55.181915646716,61.099791805535,66.523811163933,
                                      75.393791802036,81.650357800287])
        self.resid = np.array([0.083733638452,0.093247302915,0.093277497096,
                               0.119815940061,0.066312657167,0.055613551290,
                               0.042949401626,-0.086423598336,-0.074674167910,
                              -0.131915646716,-0.089791805535,-0.123811163933,
                               0.076208197964,0.129642199713])

    def conf_int(self):
        conf_int = [(233.044,244.840),(0.000534323, 0.000565990)]
        return conf_int

    def cov_params(self):
        cov_params = np.array([[7.32789,-1.96474e-05],[-1.96474e-05,5.28074e-11]])
        return cov_params

class funcMisra1a(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-np.exp(-b2*x))

class funcMisra1a_J(NonlinearLS):

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



class Misra1aWNLS(object):
    '''
    Results for weighted nonlinear regression fitted NIST model  
    y = b1*(1-exp[-b2*x]) + e with Misra data.

    27/6/2012
    Results obtained from R statistical package because Gretl
    does not provide the api for weighted NLS
    '''
    def __init__(self):
        self.start_value1 = Misra1a().start_value1
        self.params = [2.366400e+02, 5.565599e-04]
        self.bse = [2.530e+00,6.869e-06]
        self.df_resid = 12
        self.df_model = 1
        self.ssr = 0.07781363
        self.ser=0.08053

        self.weights=np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
                               0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49])
        self.scale = 0.08053**2
        self.mse_resid = 0.08053**2
        self.tvalues = [93.53, 81.03]
        self.fittedvalues = np.array([10.00270, 14.65911, 17.87252, 23.84093,
                                      29.57737, 35.15917, 40.01135, 44.93882,
                                      50.86285, 55.20562, 61.11549, 66.53015,
                                      75.38053, 81.61993])
        self.resid = np.array([0.067304043,  0.070886316, 0.067476455,
                               0.089071402,  0.032630248, 0.020826813,
                               0.008646553, -0.118824531, -0.102854753,
                              -0.155624291, -0.105485516, -0.130150299,
                               0.089467629,  0.160068499])
    def conf_int(self):
        conf_int = [(2.312589e+02, 2.423067e+02),(5.415904e-04, 5.715623e-04)]
        return conf_int

    def cov_params(self):
        cov_params = np.array([[6.4020681851, -1.735380e-05],
                              [-0.0000173538, 4.718318e-11]])
        return cov_params
