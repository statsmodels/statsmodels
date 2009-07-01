"""
General linear models
--------------------

"""

import numpy as np
from models import family, utils
from models.regression import WLS
from models.model import LikelihoodModel
from scipy import derivative, comb

# Note: STATA uses either iterated reweighted least squares optimization
#       of the deviation
# or the default mle using Newton-Raphson - which one is "quasi"likelihood?

# Note: only these combos make sense for family and link
#              + ident log logit probit cloglog pow opow nbinom loglog logc
# Gaussian     |   x    x                        x
# inv Gaussian |   x    x                        x
# binomial     |   x    x    x     x       x     x    x           x      x
# Poission     |   x    x                        x
# neg binomial |   x    x                        x          x
# gamma        |   x    x                        x
#

# Note need to correct for "dispersion"?

# Would GLM or GeneralLinearModel be a better class name?
class Model(WLS):
    '''
    Notes
    -----
    This uses iterative reweighted least squares.

    References
    ----------
    Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach.
        SAGE QASS Series.

    Green, PJ. 1984.  "Iteratively reweighted least squares for maximum
        likelihood estimation, and some robust and resistant alternatives."
        Journal of the Royal Statistical Society, Series B, 46, 149-192.

    '''

    maxiter = 10
#    @property
#    def scale(self):
#        return self.results.scale

    def __init__(self, endog, exog, family=family.Gaussian()):
        self.family = family
        super(Model, self).__init__(endog, exog, weights=1)

    def __iter__(self):
        self.iter = 0
        self.dev = np.inf
        return self

#    def llf(self, b, Y):
#        pass

    def deviance(self, Y=None, results=None, scale = 1.):
        """
        Return (unnormalized) log-likelihood for GLM.

        Note that self.scale is interpreted as a variance in old_model, so
        we divide the residuals by its sqrt.
        """

# NOTE: Is old_model just WLSModel.whiten?
        if results is None:
            results = self.results
        if Y is None:
            Y = self._endog
        return self.family.deviance(Y, results.mu) / scale

    def next(self):
        results = self.results
        Y = self._endog
        self.weights = self.family.weights(results.mu)
        if self.weights.ndim == 2:
            print 'family weights are not 1d'   # to be taken out
            self.weights = self.weights.ravel()
        self.initialize()
        Z = results.predict + self.family.link.deriv(results.mu) * (Y - results.mu)
        # TODO: this had to changed to execute properly
        # is this correct? Why? I don't understand super.... -- JT

        newresults = super(Model, self).fit(Z)
        newresults.Y = Y
        newresults.mu = self.family.link.inverse(newresults.predict)
        self.iter += 1
        return newresults

    def cont(self, tol=1.0e-05):
        """
        Continue iterating, or has convergence been obtained?
        """
        if self.iter >= Model.maxiter:
            return False

        curdev = self.deviance(results=self.results)

        if np.fabs((self.dev - curdev) / curdev) < tol:
            return False

        self.dev = curdev # this ie Deviance in STATA
        return True

    def estimate_scale(self, Y=None, results=None):
        """
        Return Pearson\'s X^2 estimate of scale.
        """

        if results is None:
            results = self.results
        if Y is None:
            Y = self._endog
        resid = Y - results.mu          # This gives the response residual
# This is the (1/df) Pearson in STATA
        return ((np.power(resid, 2) / self.family.variance(results.mu)).sum()
                / results.df_resid)

    def fit(self):
        iter(self)
        self.results = super(Model, self).fit(
            self.family.link.initialize(self._endog)) # calls WLS.fit with
                                            # Y, where Y is the result
                                            # of the link function on the mean
                                            # of Y
        self.results.mu = self.family.link.inverse(self.results.predict)
                                            # returns inverse of link
                                            # on the predicted values
                                            # predict has been overwritten
                                            # and holds self.link(mu)
                                            # which is just the mean vector!?
#        self.results.scale = self.estimate_scale()
                                            # uses Pearson's X2 as
                                            # as default scaling
        while self.cont():
            self.results = self.next()
#            self.results.scale = self.estimate_scale()
#        self.results.scale = 1.
        return self.results

class GLMBinomial(LikelihoodModel):
    '''
    Notes
    -----
    This uses iterative reweighted least squares.

    References
    ----------
    Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach.
        SAGE QASS Series.

    Green, PJ. 1984.  "Iteratively reweighted least squares for maximum
        likelihood estimation, and some robust and resistant alternatives."
        Journal of the Royal Statistical Society, Series B, 46, 149-192.

    Hardin, J.W. and Hilbe, J. 2007.  Generalized Linear Models and Extensions.
        Stata Corp.



    '''
    def initialize(self):
# BIG NOTE: Is the data binary or proportional?  Need to check this
# need to check overdispersion defaults
        self.family = family.Binomial()
        self.endog = self._endog
        self.exog = self._exog
#        if self.family is family.Binomial()    # need to check this...string property, isinstance?
        if (self.endog.ndim > 1 and self.endog.shape[1] > 1): # greedy logic
            self.y = self.endog[:,0]    # successes
            self.k = self.endog[:,0] + self.endog[:,1]    # total trials
#            self.deviance = self.binom_dev
            self.deviance = lambda mu: 2*np.sum(self.y*np.log(self.y/mu)/
                + (self.k - self.y)*np.log((self.k-self.y)/(self.k-mu)))    # - or +?
# Gill p.58 and Hardin 9.21 say +
        else:
            self.k = 1.
            self.y = self.endog # then self.endog is binary
            self.deviance = lambda mu: 2 * np.sum(np.log(1/mu)) # always assumes# that reponse variable of interest == 1(?), or do I need to code the two conditions and then reduce?
        self.history = { 'predict' : [], 'params' : [np.inf], 'logL' : [], 'deviance' : [np.inf]}
        self.iteration = 0
        self.last_result = np.inf
        self.nobs = self._endog.shape[0]

        ### copied from OLS initialize()?? ###
        self.calc_params = np.linalg.pinv(self._exog)
        self.normalized_cov_params = np.dot(self.calc_params,
                                        np.transpose(self.calc_params))
        self.df_resid = self._exog.shape[0]
        self.df_model = utils.rank(self._exog)-1

#    def binom_dev(mu):
#        conditions = [(),(),()]
#        dev = np.piecewise(self.y,

    def llf(self, results):
        n = self.nobs
# TODO        y = np.sum(self._endog)   # number of "successes" UPDATE FOR PROPORTIONAL
        p = self.inverse(results.predict)
        llf = y * np.log(p/(1-p)) - (-n*np.log(1-p)) + np.log(comb(n,y))
        return llf

    def score(self, params):
        pass

    def information(self, params):
        pass

    def update_history(self, tmp_result, mu):
        self.history['params'].append(tmp_result.params)
        self.history['predict'].append(tmp_result.predict)
#        deviance = 2*np.sum(self.y*np.log(self.y/mu)/
#            - (self.k - self.y)*np.log((self.k-self.y))/(self.k-mu))
        self.history['deviance'].append(self.deviance(mu))

    def inverse(self, z):      # temporary
        return np.exp(z)/(1+np.exp(z))

    def link(self, mu):
        return np.log(mu/(1-mu))

    def fit(self, maxiter=100, method='IRLS', tol=1e-5):
#TODO: method='newton'
# for IRLS
# initial value for Newton method (which is a value of the coefs!)
# can often just  be the Theta for the constant only model
# (probably analytically derived)
# cf Hardin 27, 125
        mu =  (self.y + 0.5)/(self.k + 1)   # continuity correction
# OR
#        wls_endog = self.inverse(self._endog.mean()) * np.ones((self.nobs))
        wls_exog = self.exog
        eta = np.log(mu/(self.k - mu)) # First guess at linear predictor
        self.iteration+=1
#        while ((self.history['params'][self.iteration-1]-\
#                self.history['params'][self.iteration]).all()>tol\
#                and self.iteration < maxiter):
        self.history['deviance'].append(self.deviance(mu))
        while ((np.fabs(self.history['deviance'][self.iteration]-\
                    self.history['deviance'][self.itermation-1])) > tol):
# which one for binomial?  same for bernoulli...
#            w = mu*(self.k - mu) # weights based on variance
            w = mu*(1-mu/self.k) # if it's on the variance, then it's this!?
            wls_endog = eta + self.k*(self.y - mu)/(mu*(self.k - mu))
            wls_results = WLS(wls_endog, wls_exog, weights=w).fit()
            eta = np.dot(self.exog, wls_results.params)
            mu = self.k/(1+np.exp(-eta))
            # ugly clip
            mu = np.clip(mu, np.finfo(np.float).eps, mu-1e-05)
#            mu = np.where(mu < np.finfo(np.float).eps, np.finfo(np.float).eps, mu)
#            mu = np.where(mu == self.k, mu-np.finfo
            # clip all zeros to 0+eps, else leave
            # mu is bounded by [0,k)
            self.update_history(wls_results, mu)    # pass mu or make it an attr?
            self.iteration += 1
        self.results = wls_results
        return self.results










