"""
Robust linear models
"""
import numpy as np

from models import utils
from models.regression import WLS, GLS
from models.robust import norms, scale
from scipy.stats import norm as Gaussian # can get rid of this once scale is sorted out

from models.model import LikelihoodModel, LikelihoodModelResults

class Model(WLS):

    niter = 20
    scale_est = 'MAD'

    def __init__(self, endog, exog, M=norms.Hampel()):
        self.M = M
        self.weights = 1
        self._endog = endog
        self._exog = exog
        self.initialize()

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
    def __init__(self, endog, exog, M=norms.HuberT()):
        """
        """
        self.M = M
        self._endog = endog
        self._exog = exog
        self.initialize()

    def initialize(self):
        self.history = {'deviance' : [np.inf], 'params' : [np.inf],
            'weights' : [np.inf], 'sresid' : [np.inf], 'scale' : []}
        self.iteration = 0
        self.calc_params = np.linalg.pinv(self._exog)
        self.normalized_cov_params = np.dot(self.calc_params,
                                        np.transpose(self.calc_params))
#        self.df_resid = self._exog.shape[0] - utils.rank(self._exog)
        self.df_resid = np.float(self._exog.shape[0] - utils.rank(self._exog))
        self.df_model = np.float(utils.rank(self._exog)-1)
        self.nobs = float(self._endog.shape[0])

    def score(self, params):
        pass

    def information(self, params):
        pass

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.

        Note that self.scale is interpreted as a variance, so we divide
        the residuals by its sqrt.
        """
        return self.M((self._endog - tmp_results.predict)/\
#                    np.sqrt(tmp_results.scale)).sum()
                    tmp_results.scale).sum()

    def update_history(self, tmp_results):
# TODO: Figure out why the first estimate gets copied to the end of these
        self.history['deviance'].append(self.deviance(tmp_results))
        self.history['params'].append(tmp_results.params)
        self.history['scale'].append(tmp_results.scale)
        self.history['sresid'].append(tmp_results.resid/tmp_results.scale)
        self.history['weights'].append(tmp_results._model.weights)

    def estimate_scale(self, resid):
        """
        Note that self.scale is interpreted as a variance in OLSModel, so
        we return MAD(resid)**2 by default.
        """
        if self.scale_est == 'MAD':
#            return scale.MAD(resid)**2
            return scale.MAD(resid)
        if self.scale_est == 'stand_MAD':
            return scale.stand_MAD(resid)
        elif self.scale_est == "Huber":
##            return scale.huber(resid)**2
            #must initialize an instance to specify norm, etc.
#            s = scale.Huber(norm=self.M)
#            return scale.huber(resid)[1]    # what do you do with loc?
            # why are we iterating within the iterator, it should be
# alternating?
            if self.scale == None:
# what to use for first guess?
#                self.scale = 1. #?
#                self.scale = (1/self.df_resid)*np.sum(resid**2) # sigma**2
                self.scale = models.robust.scale.MAD(resid)
                # OLS estimate
            d = 1.5
            h = self.df_resid/self.nobs * (d**2 + (1-d**2)*Gaussian.cdf(d)-.5\
                -d*np.sqrt(2*np.pi)*np.exp(-.5*d**2))
#            test = np.less_equal(np.fabs(resid/self.scale), d)
            rw = np.clip(resid, -d*self.scale, d*self.scale
            chi = test * ((resid/self.scale)**2)/2 + (1-test)*(d**2)/2
            return np.sqrt(1/(self.nobs*h) * np.sum(chi)*self.scale**2)

#        else:
#            return scale.scale_est(self, resid)**2
#        return np.median(np.fabs(resid))/Gaussian.ppf(3/4.)

    def fit(self, maxiter=100, tol=1e-8, scale_est='MAD', init=None, cov='H1',
            update_scale=True, conv='dev'):
        """
        Iterated reweighted least squares for robust regression.

        The IRLS routine runs until the deviance function as converged to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------

        maxiter : scalar
            The maximum number of iterations to try. Default is 100.

        tol : float
            The convergence tolerance of the estimate.
            Defaults is 1e1-8

        scale_est : string
            'MAD', 'stand_MAD', or 'Huber'
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'MAD' (median absolute deviation.  Other options are
            use 'stand_MAD' for the median absolute deviation standardized
            around zero and 'Huber' for Huber's Proposal 2.  See
            models.robust.scale for
            more information.

        init : string
            Allows initial estimates for the parameters.
            Default is None, which means that the least squares estimate
            is used.  Currently it is the only available choice.

        cov : string
            'H1', 'H2', or 'H3'
            Indicates how the covariance matrix is estimated.  Default is 'H1'
            See Huber (1981) p 173 for more information.

        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Defaults is True.

        conv : string
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "resids" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).
            The default is "dev".


        Returns
        -------
        results : object
            The RLM results class
        """
        self.scale_est = scale_est
        if not cov.upper() in ["H1","H2","H3"]:
            raise AttributeError, "Covariance matrix %s not understood" % cov
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if not conv in ["weights","coefs","dev","resid"]:
            raise AttributeError, "Convergence argument %s not understood" \
                % conv
        if not init:
            # initial guess is OLS by default, ie., weights = 1
            wls_results = WLS(self._endog, self._exog).fit()
            self.scale = None # only necessary for Huber proposal 2
            self.scale = self.estimate_scale(wls_results.resid)
        self.update_history(wls_results)
        self.iteration = 1  # so these don't accumulate across fits
        if conv == 'coefs':
            criterion = self.history['params']
        elif conv == 'dev':
            criterion = self.history['deviance']
        elif conv == 'resid':
            criterion = self.history['sresid']
        elif conv == 'weights':
            criterion = self.history['weights']
        while (np.all(np.fabs(criterion[self.iteration]-\
                criterion[self.iteration-1]) > tol) and \
                self.iteration < maxiter):
            print self.scale
            self.weights = self.M.weights((self._endog - wls_results.predict) \
                        /self.scale)
            wls_results = WLS(self._endog, self._exog,
                                    weights=self.weights).fit()
            if update_scale is True:
                self.scale = self.estimate_scale(wls_results.resid)
            self.update_history(wls_results)
            self.iteration += 1
        self.results = RLMResults(self, wls_results.params,
                            self.normalized_cov_params, self.scale)
        self.results.wls_results = wls_results  # for debugging
        return self.results

class RLMResults(LikelihoodModelResults):
    """
    Class to contain RLM results
    """
    def __init__(self, model, params, normalized_cov_params, scale):
        super(RLMResults, self).__init__(model, params,
                normalized_cov_params, scale)
        self._get_results(model)

    def _get_results(self, model):
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.fitted_values = np.dot(model._exog, self.params)
        self.hat = np.dot(model._exog,model.calc_params)
        self.resid = model._endog - self.fitted_values   # before bcov
        self.sresid = self.resid/self.scale
        self.calc_params = model.calc_params    # for bcov,
                                                # this is getting sloppy
        self.bcov_unscaled = self.cov_params(scale=1)
        self.nobs = model.nobs
        self.weights = model.weights
        m = np.mean(model.M.psi_deriv(self.resid/self.scale))
        var_psiprime = np.var(model.M.psi_deriv(self.resid/self.scale))
        k = 1 + (self.df_model+1)/self.nobs * var_psiprime/m**2
# these might better be called unbiased rather than scaled
        if model.cov == "H1":
# THE BELOW IS CORRECT!!! # Based on Huber (1973) 8.14
# SAS documentation and Huber 1981 is misleading...
            self.bcov_scaled = k**2 * (1/self.df_resid*\
                np.sum(model.M.psi(self.sresid)**2)*self.scale**2)\
                /((1/self.nobs*np.sum(model.M.psi_deriv(self.sresid)))**2)\
                *model.normalized_cov_params
        else:
# needs to be optimized
# Correctly based on Huber 1973
            W = np.dot(model.M.psi_deriv(self.sresid)*model._exog.T,model._exog)
            W_inv = np.linalg.inv(W)
# should be
# [W_jk]^-1 = [SUM(psi_deriv(Sr_i)*x_ij*x_jk)]^-1
# where Sr are the STANDARDIZED (!) residuals
            if model.cov == "H2":
# This is correct, based on Huber (1973) 8.13
                self.bcov_scaled = k*(1/self.df_resid)*np.sum(\
                        model.M.psi(self.sresid)**2)*self.scale**2\
                        /((1/self.nobs)*np.sum(\
                        model.M.psi_deriv(self.sresid)))*W_inv
# This is correct vis-a-vis the SAS results
            elif model.cov == "H3":
                self.bcov_scaled = k**-1*1/self.df_resid*np.sum(\
                    model.M.psi(self.sresid)**2)*self.scale**2\
                    *np.dot(np.dot(W_inv, np.dot(model._exog.T,model._exog)),\
                    W_inv)
        self.bse = np.sqrt(np.diag(self.bcov_scaled))

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

#    model_ols = models.regression.OLS(endog, exog)
#    results_ols = model_ols.fit()

#    model_huber = RLM(endog, exog, M=norms.HuberT(t=2.))
#    results_huber = model_huber.fit(scale_est="stand_MAD", update_scale=False)

#    model_ramsaysE = RLM(endog, exog, M=norms.RamsayE())
#    results_ramsaysE = model_ramsaysE.fit(update_scale=False)

#    model_andrewWave = RLM(endog, exog, M=norms.AndrewWave())
#    results_andrewWave = model_andrewWave.fit(update_scale=False)

#    model_hampel = RLM(endog, exog, M=norms.Hampel(a=1.7,b=3.4,c=8.5)) # convergence problems with scale changed, not with 2,4,8 though?
#    results_hampel = model_hampel.fit(update_scale=False)

#######################
### Stack Loss Data ###
#######################
    from models.datasets.stackloss.data import load
    data = load()
    data.exog = models.functions.add_constant(data.exog)
#############
### Huber ###
#############
#    m1_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber1 = m1_Huber.fit()
#    m2_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber2 = m2_Huber.fit(cov="H2")
#    m3_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber3 = m3_Huber.fit(cov="H3")
##############
### Hampel ###
##############
#    m1_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel1 = m1_Hampel.fit()
#    m2_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel2 = m2_Hampel.fit(cov="H2")
#    m3_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel3 = m3_Hampel.fit(cov="H3")
################
### Bisquare ###
################
#    m1_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare1 = m1_Bisquare.fit()
#    m2_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare2 = m2_Bisquare.fit(cov="H2")
#    m3_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare3 = m3_Bisquare.fit(cov="H3")


##############################################
# Huber's Proposal 2 scaling                 #
##############################################

################
### Huber'sT ###
################
    m1_Huber_H = RLM(data.endog, data.exog, M=norms.HuberT())
    results_Huber1_H = m1_Huber_H.fit(scale_est = 'Huber')
#    m2_Huber_H
#    m3_Huber_H
# Huber scale estimate do not currently work
#    m4 = RLM(data.endog, data.exog, M=norms.HuberT())
#    results4 = m1.fit(scale_est="Huber")
#    m5 = RLM(data.endog, data.exog, M=norms.Hampel())
#    results5 = m2.fit(scale_est="Huber")
#    m6 = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results6 = m3.fit(scale_est="Huber")




#    print """Least squares fit
#%s
#Huber Params, t = 2.
#%s
#Ramsay's E Params
#%s
#Andrew's Wave Params
#%s
#Hampel's 17A Function
#%s
#""" % (results_ols.params, results_huber.params, results_ramsaysE.params,
#            results_andrewWave.params, results_hampel.params)

