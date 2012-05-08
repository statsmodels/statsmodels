import os

import numpy as np
from numpy import genfromtxt
cur_dir = os.path.dirname(os.path.abspath(__file__))

class ARIMA111(object):
    def __init__(self, method="mle"):
        if method == "mle":
            # from stata
            from arima111_results import results
            self.__dict__ = results
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma**2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:] # no idea why this initial value
            self.linear = self.y[1:]
            self.k_diff = 1
            #their bse are OPG
            #self.bse = np.diag(self.cov_params) ** .5
            # from gretl
            self.arroots = [1.0640 + 0j]
            self.maroots = [1.2971 + 0j]
            self.hqic = 496.8653
            self.aic_gretl = 491.5112
            self.bic_gretl = 504.7442
            self.bse = [.205811, .0457010, .0897565]
            self.tvalues = [4.280, 20.57, -8.590]
            self.pvalues = [1.87e-5, 5.53e-94, 8.73e-18]
            self.cov_params = [[0.0423583,   -0.00167449,    0.00262911],
                               [-0.00167449, 0.00208858,    -0.0035068],
                               [0.00262911, -0.0035068, 0.00805622]]
            self.bse = np.diag(np.sqrt(self.cov_params))
            forecast = genfromtxt(open(cur_dir+"/arima111_forecasts.csv"),
                            delimiter=",", skip_header=1, usecols=[1,2,3,4,5])
            self.forecast = forecast[203:,1]
            self.fcerr = forecast[203:,2]
            self.fc_conf_int = forecast[203:,3:]
        else:
            pass

class ARIMA212(object):
    def __init__(self, method="mle"):
        if method == "mle":
            # from gretl
            self.arroots = [1.0456, -1.3641]
            self.maroots = [-1.0768, 1.2210]
            self.hqic = 488.2071
            self.aic_gretl = 480.1760
            self.bic_gretl = 500.0256

