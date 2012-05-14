import os

import numpy as np
from numpy import genfromtxt
cur_dir = os.path.dirname(os.path.abspath(__file__))

forecast_results = genfromtxt(open(cur_dir+"/results_arima_forecasts.csv",
                        "rb"), names=True, delimiter=",", dtype=float)

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
            # from stata
            #forecast = genfromtxt(open(cur_dir+"/arima111_forecasts.csv"),
            #                delimiter=",", skip_header=1, usecols=[1,2,3,4,5])
            #self.forecast = forecast[203:,1]
            #self.fcerr = forecast[203:,2]
            #self.fc_conf_int = forecast[203:,3:]
            # from gretl
            self.forecast = forecast_results['fc111c'][-25:]
            self.forecasterr = forecast_results['fc111cse'][-25:]
            self.forecast_dyn = forecast_results['fc111cdyn'][-25:]
            self.forecasterr_dyn = forecast_results['fc111cdynse'][-25:]
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

class ARIMA211(object):
    def __init__(self, method="mle"):
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
            self.arroots = [1.027 + 0j, 5.7255+ 0j]
            self.maroots = [1.1442+0j]
            self.hqic = 496.5314
            self.aic_gretl = 489.8388
            self.bic_gretl = 506.3801
            #self.bse = [0.248376, 0.102617, 0.0871312, 0.0696346]
            self.tvalues = [3.468, 11.14, -1.941, 12.55]
            self.pvalues = [.0005, 8.14e-29, .0522, 3.91e-36]
            cov_params = np.array([
                    [0.0616906,  -0.00250187, 0.0010129,    0.00260485],
                    [0, 0.0105302,   -0.00867819,   -0.00525614],
                    [ 0 ,0,         0.00759185,    0.00361962],
                    [ 0 ,0,0,                      0.00484898]])
            self.cov_params = cov_params + cov_params.T - \
                              np.diag(np.diag(cov_params))
            self.bse = np.diag(np.sqrt(self.cov_params))
            self.forecast = forecast_results['fc211c'][-25:]
            self.forecasterr = forecast_results['fc211cse'][-25:]
            self.forecast_dyn = forecast_results['fc211cdyn'][-25:]
            self.forecasterr_dyn = forecast_results['fc211cdynse'][-25:]


