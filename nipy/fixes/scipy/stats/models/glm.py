"""
General linear models
--------------------

"""

import numpy as np
from nipy.fixes.scipy.stats.models import family
from nipy.fixes.scipy.stats.models.regression import WLS
from nipy.fixes.scipy.stats.models.model import LikelihoodModel
from scipy import derivative, comb
from nipy.fixes.scipy.stats.models import utils

# Note: STATA uses either iterated reweighted least squares optimization
#       of the deviation
# or the default mle using Newton-Raphson

# Note: only these combos make sense for family and link
#              + ident log logit probit cloglog pow opow nbinom loglog logc
# Gaussian     |   x    x                        x
# inv Gaussian |   x    x                        x
# binomial     |   x    x    x     x       x     x    x           x      x
# Poission     |   x    x                        x
# neg binomial |   x    x                        x          x
# gamma        |   x    x                        x
#


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
    @property
    def scale(self):
        return self.results.scale

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

        self.dev = curdev # this is Deviance in STATA
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
        self.results.scale = self.estimate_scale()
                                            # uses Pearson's X2 as
                                            # as default scaling
        while self.cont():
            self.results = self.next()
            self.results.scale = self.estimate_scale()

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
        self.family = family.Binomial()
        self.history = { 'predict' : [], 'params' : [np.inf], 'logL' : []}
        self.iteration = 0
        self.last_result = np.inf
        self.nobs = self._endog.shape[0]

        ### copied from OLS initialize()?? ###
        self.calc_params = np.linalg.pinv(self._exog)
        self.normalized_cov_params = np.dot(self.calc_params,
                                        np.transpose(self.calc_params))
        self.df_resid = self._exog.shape[0]
        self.df_model = utils.rank(self._exog)-1

    def llf(self, results):
        n = self.nobs
        y = np.sum(self._endog)   # number of "successes"
        p = self.inverse(results.predict)
        llf = y * np.log(p/(1-p)) - (-n*np.log(1-p)) + np.log(comb(n,y))
        return llf

    def score(self, params):
        pass

    def information(self, params):
        pass

    def update_history(self, tmp_result):
        self.history['params'].append(tmp_result.params)
        self.history['predict'].append(tmp_result.predict)

    def inverse(self, z):      # temporary
        return np.exp(z)/(1+np.exp(z))

    def link(self, mu):
        return np.log(mu/(1-mu))

    def next(self, wls_results):
# or is mu below always the link on the mean response?
        mu = self.inverse(wls_results.predict)
        var = mu * (1 - mu/self.nobs)
        weights = 1/var * derivative(self.inverse, wls_results.predict, dx=1e-02, n=1, order=3)**2
        raw_input('pause')
        wls_exog = self._exog    # this is redundant
        wls_endog = (self._endog - mu)*derivative(self.link, mu, dx=1e-02,
                n=1, order=3) + self.history['predict'][self.iteration - 1]
                # - offset? cf. Hardin p 29
        wls_results = WLS(wls_endog, wls_exog, weights).fit()
        self.iteration +=1
        return wls_results


    def fit(self, maxiter=100, method='IRLS', tol=1e-10):
#TODO: method='newton'
# for IRLS
# initial value can be inverse of link of the mean of the response
# note that this is NOT what our initial values are, so it's probably pretty
# robust to any choice that's in the support
# OR for Binomial(n_i,p_i) it can be n_i(y_i + .5)/(n_i + 1)
# note that for non binomial it's
# (y_i + y_bar)/2
# cf Hardin page 31
# initial value for Newton method (which is a value of the coefs!)
# can often just  be the Theta for the constant only model
# (probably analytically derived)
# cf Hardin 27
#        wls_endog =  self.nobs * (self._endog + .5)/(self.nobs + 1)
# OR
        wls_endog = self.inverse(self._endog.mean()) * np.ones((self.nobs))
        wls_exog = self._exog
        weights = 1.
        wls_results = WLS(wls_endog, wls_exog, weights).fit()
        self.update_history(wls_results)
        eta = wls_results.predict
        self.iteration+=1
        while ((self.history['params'][self.iteration-1]-\
                self.history['params'][self.iteration]).all()>tol\
                and self.iteration < maxiter):
            wls_results=self.next(wls_results)
            self.update_history(wls_results)
        self.results = wls_results
        return self.results










