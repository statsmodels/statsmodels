"""
Robust linear models
"""
import numpy as np

from models import utils
from models.regression import WLS, GLS
from models.robust import norms, scale

from models.model import LikelihoodModel

class Model(WLS):

    niter = 20
    scale_est = 'MAD'

    def __init__(self, endog, exog, M=norms.Hampel()):
        self.M = M
        self.weights = 1
        self._endog = endog
        self._exog = exog
        self.initialize()   # is this still needed

    def __iter__(self):
        self.iter = 0
        self.dev = np.inf
        return self

    def deviance(self, results=None):
        """
        Return (unnormalized) log-likelihood from M estimator.

        Note that self.scale is interpreted as a variance in OLSModel, so
        we divide the residuals by its sqrt.
        """
        if results is None:
            results = self.results
        return self.M((self._endog - results.predict) / np.sqrt(results.scale)).sum()

    def next(self):
        results = self.results
        self.weights = self.M.weights((self._endog - results.predict) / np.sqrt(results.scale))
#        self.initialize(self.design)
        self.initialize()
        results = WLS.fit(self, self._endog)
        self.scale = results.scale = self.estimate_scale(results)
        self.iter += 1
        return results

    def cont(self, results, tol=1.0e-5):
        """
        Continue iterating, or has convergence been obtained?
        """
        if self.iter >= Model.niter:
            return False

        curdev = self.deviance(results)
        if np.fabs((self.dev - curdev) / curdev) < tol:
            return False
        self.dev = curdev

        return True

    def estimate_scale(self, results):
        """
        Note that self.scale is interpreted as a variance in OLSModel, so
        we return MAD(resid)**2 by default.
        """
        resid = self._endog - results.predict
        if self.scale_est == 'MAD':
            return scale.MAD(resid)**2
        elif self.scale_est == 'Huber2':
            return scale.huber(resid)**2
        else:
            return scale.scale_est(self, resid)**2

    def fit(self):

        iter(self)
        self.results = WLS(self._endog, self._exog).fit()   # does it know the weights?
        self.scale = self.results.scale = self.estimate_scale(self.results)

        while self.cont(self.results):
            self.results = self.next()

        return self.results

class RLM(LikelihoodModel):
    def __init__(self, endog, exog, M=norms.Hampel()):
        self.M = M
        self._endog = endog
        self._exog = exog
        self.initialize()

    def initialize(self):
        self.history = {'deviance' : [np.inf], 'params' : [], 'scale' : []}
        self.iteration = 0
#        self.y = self._endog
        self.calc_params = np.linalg.pinv(self._exog)
        self.normalized_cov_params = np.dot(self.calc_params,
                                        np.transpose(self.calc_params))
#        self.df_resid = self._exog.shape[0] - utils.rank(self._exog)
        self.df_resid = np.nan  # to avoid estimating residuals in GLS
        self.df_model = utils.rank(self._exog)-1

#    @property
#    def scale(self):
#        return self.results.scale   # does this pull from here or GLS?

    def score(self, params):
        pass

    def information(self, params):
        pass

    def deviance(self, tmp_results):
        '''
        Returns the (unnormalized) log-likelihood from the M estimator.

        Note that self.scale is interpreted as a variance, so we divide
        the residuals by its sqrt.
        '''
        return self.M((self._endog - tmp_results.predict)/\
#                    np.sqrt(tmp_results.scale)).sum()
                    tmp_results.scale).sum()

    def update_history(self, tmp_results):
        self.history['deviance'].append(self.deviance(tmp_results))
        self.history['params'].append(tmp_results.params)
        self.history['scale'].append(tmp_results.scale)

    def estimate_scale(self, results):
        '''
        Note that self.scale is interpreted as a variance in OLSModel, so
        we return MAD(resid)**2 by default.
        '''
# Figure out why this ^ is.
        resid = self._endog - results.predict
        if self.scale_est == 'MAD':
#            return scale.MAD(resid)**2
            return scale.MAD(resid)
        elif self.scale_est == "Huber":
#            return scale.huber(resid)**2
            return scale.huber(resid)
        else:
            return scale.scale_est(self, resid)**2

    def fit(self, maxiter=100, tol=1e-5, scale_est='MAD'):
        self.scale_est = scale_est  # is this the best place to put this
                                    # are the other scales implemented?
        self.results = WLS(self._endog, self._exog).fit()   # initial guess is just OLS
        self.results.scale = self.estimate_scale(self.results)  # overwrite scale estimate
        self.update_history(self.results)
        self.iteration += 1
        while ((np.fabs(self.history['deviance'][self.iteration]-\
                self.history['deviance'][self.iteration-1])) > tol and \
                self.iteration < maxiter):
            self.weights = self.M.weights((self._endog - self.results.predict) \
#                        /np.sqrt(self.results.scale))
                        /self.results.scale)    # why all the squaring and roots?
#                        /scale)
            self.results = WLS(self._endog, self._exog, weights=self.weights).fit()
            self.results.scale = self.estimate_scale(self.results)  # iteratively update scale
# M&P suggests to use a constant weight, can be iterative or constant of a "resistant" fit
            self.update_history(self.results)
            self.iteration += 1
        return self.results

# needs its own results class, so it doesn't have all the cruft from GLS/WLS

if __name__=="__main__":
#NOTE: This is to be removed
#Delivery Time Data is taken from Montgomery and Peck
    import models

#delivery time(minutes)
    endog = np.array([16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8.00, 17.83,
    79.24, 21.50, 40.33, 21.00, 13.50, 19.75, 24.00, 29.00, 15.35, 19.00,
    9.50, 35.10, 17.90, 52.32, 18.75, 19.83, 10.75])

#number of cases, distance (Feet)
    exog = np.array([[7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6,
    7, 3, 17, 10, 26, 9, 8, 4], [560, 220, 340, 80, 150, 330, 110, 210, 1460,
    605, 688, 215, 255, 462, 448, 776, 200, 132, 36, 770, 140, 810, 450, 635,
    150]])
    exog = exog.T
    exog = models.functions.add_constant(exog)

    model_ols = models.regression.OLS(endog, exog)
    results_ols = model_ols.fit()

    model_huber = RLM(endog, exog, M=norms.HuberT(t=2.))
    # default norm is Hampel, but R uses Huber
    results_huber = model_huber.fit(scale_est="MAD")
    # this is default. just being explicit

    model_ramsaysE = RLM(endog, exog, M=norms.RamsayE())
    results_ramsaysE = model_ramsaysE.fit()

    model_andrewWave = RLM(endog, exog, M=norms.AndrewWave())
    results_andrewWave = model_andrewWave.fit()

    model_hampel = RLM(endog, exog, M=norms.Hampel(a=1.7,b=3.4,c=8.5))
    results_hampel = model_hampel.fit()

### Stack Loss Data ###
    from models.datasets.stackloss.data import load
    data = load()
    data.exog = models.functions.add_constant(data.exog)

# First guess from L1-norm
    Z = lambda x,y: np.sum(x+y)


    m1 = RLM(data.endog, data.exog, M=norms.HuberT())
    results1 = m1.fit()
    m2 = RLM(data.endog, data.exog, M=norms.Hampel())
    results2 = m2.fit()
    m3 = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
    results3 = m3.fit()
# Huber scale estimate do not currently work
#    m4 = RLM(data.endog, data.exog, M=norms.HuberT())
#    results4 = m1.fit(scale_est="Huber")
#    m5 = RLM(data.endog, data.exog, M=norms.Hampel())
#    results5 = m2.fit(scale_est="Huber")
#    m6 = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results6 = m3.fit(scale_est="Huber")




#    print '''Least squares fit
#%s
#Huber Params, t = 2.
#%s
#Ramsay's E Params
#%s
#Andrew's Wave Params
#%s
#Hampel's 17A Function
#%s
#'''

#% (results_ols.params, results_huber.params, results_ramsaysE.params,
#            results_andrewWave.params, results_hampel.params)








