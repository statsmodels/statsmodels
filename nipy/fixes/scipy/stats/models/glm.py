"""
General linear models
--------------------

"""

import numpy as np
from models import family, utils
from models.regression import WLS,GLS
from models.model import LikelihoodModel, LikelihoodModelResults
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
        self.results.scale = self.estimate_scale()
                                            # uses Pearson's X2 as
                                            # as default scaling
        while self.cont():
            self.results = self.next()
            self.results.scale = self.estimate_scale()
        return self.results

# Would GLM or GeneralLinearModel be a better class name?
#class glzm(WLSModel):
class GLMtwo(LikelihoodModel):
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
#    @property
#    def scale(self):
#        return self.results.scale

    def __init__(self, endog, exog, family=family.Gaussian()):
        self.family = family
        self._endog = endog
        self._exog = exog
        self.initialize()

    def initialize(self):
        self.history = { 'predict' : [], 'params' : [np.inf],
                         'logL' : [], 'deviance' : [np.inf]}
        self.iteration = 0
        self.y = self._endog

         ### copied from OLS initialize() All needed?? ###
        self.calc_params = np.linalg.pinv(self._exog)
        self.normalized_cov_params = np.dot(self.calc_params,
                                        np.transpose(self.calc_params))
        self.df_model = utils.rank(self._exog)-1
        self.df_resid = self._exog.shape[0] - utils.rank(self._exog)


    def score(self, params):
        pass

    def information(self, params):
        pass

    def update_history(self, tmp_result, mu):
        self.history['params'].append(tmp_result.params)
        self.history['predict'].append(tmp_result.predict)
        self.history['deviance'].append(self.family.deviance(self.y, mu))

    def estimate_scale(self, mu):
        """
        Return scale.

        Pearson\'s X^2 estimate of scale.
        Residual Deviance estimate of scale
        1.
        Float
        """
#TODO: Fix docstring

        if not self.scaletype:
            if isinstance(self.family, (family.Binomial, family.Poisson)):
                return np.array(1.)
            else:
                resid = self.y - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum() \
                    / self.df_resid)

        if isinstance(self.scaletype, float):
            return np.array(self.scaletype)

        if isinstance(self.scaletype, str):
            if self.scaletype.lower() == 'x2':
                resid = self.y - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum() \
                    / self.df_resid)
            elif self.scaletype.lower() == 'dev':
                return self.family.deviance(self.y, mu)/self.df_resid
            else:
                raise ValueError, "Scale %s with type %s not understood" %\
                    (self.scaletype,type(self.scaletype))

        else:
            raise ValueError, "Scale %s with type %s not understood" %\
                (self.scaletype, type(self.scaletype))

    def fit(self, maxiter=100, method='IRLS', tol=1e-8, data_weights=1.,
            scale=None):
        '''
        Fits a glm model based

        parameters
        ----------
        scale : string or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default is `X2` for Gamma, Gaussian, and Inverse Gaussian
                `X2` is Pearson's chi-squared divided by the residual
                degrees of freedom
            The default is 1 for Binomial and Poisson
            `dev` scales by the deviance divided by the residual
                    degrees of freedome

        data_weights : array-like
            Number of trials for each observation. Used for binomial data.
        '''
        if self._endog.shape[0] != self._exog.shape[0]:
            raise ValueError, 'size of Y does not match shape of design'
        # Why have these checks here and not in the base class or model?
        self.data_weights = data_weights
        if np.shape(self.data_weights) == () and self.data_weights>1:
            self.data_weights = self.data_weights * np.ones((self._exog.shape[0]))
        self.scaletype = scale
        if isinstance(self.family, family.Binomial):
            self.y = self.family.initialize(self.y)
            mu = (self.y + 0.5)/2    # starting mu for binomial
        else: mu = (self.y + self.y.mean())/2. # starting mu for nonbinomial
        wls_exog = self._exog
        eta = self.family.predict(mu)
        self.iteration += 1
        self.history['deviance'].append(self.family.deviance(self.y, mu) / 1.)
# scale is assumed to be 1. for the initial predictor
        while ((np.fabs(self.history['deviance'][self.iteration]-\
                    self.history['deviance'][self.iteration-1])) > tol and \
                    self.iteration < maxiter):
            self.weights = data_weights*self.family.weights(mu)
#            if self.weights.ndim == 2:  # not sure what this corrected for?
#                if not self.weights.size == self.Y.shape[0]:
#                    print 'weights too large', self.weights.shape
#                else:
#                    print 'familiy weights are not 1d', self.weights.shape
#                    self.weights = self.weights.ravel()
            wls_endog = eta + self.family.link.deriv(mu) * (self.y-mu)  # - offset
            wls_results = WLS(wls_endog, wls_exog, self.weights).fit()
            eta = np.dot(self._exog, wls_results.params) # + offset
            mu = self.family.fitted(eta)
            self.update_history(wls_results, mu)
            self.scale = self.estimate_scale(mu)
            self.iteration += 1
        self.mu = mu
        self.results = GLMResults(self, wls_results.params, self.scale,
                    wls_results)
        return self.results

class GLMResults(LikelihoodModelResults):   # could inherit from RegressionResults?
    '''
    Class to contain GLM results
    '''

    _llf = None

    def __init__(self, model, params, scale, tmp_results):
#TODO: need to streamline init sig...
        super(GLMResults, self).__init__(model, params,
                normalized_cov_params=None, scale=scale)
        self._get_results(model, tmp_results)

    def _get_results(self, model, tmp_results):
        self.tmp_results = tmp_results # temporarily here
        self.nobs = model.y.shape[0]    # this should be a model attribute
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.mu = model.mu
        self.bse = np.sqrt(np.diag(tmp_results.cov_params(scale=model.scale)))
        self.resid_response = model.data_weights*(model.y - model.mu)
            # data_weights needed for binomial(n)
        self.resid_pearson = np.sqrt(model.data_weights)*(model.y-model.mu)/\
                np.sqrt(model.family.variance(model.mu))
        self.resid_working = model.data_weights * (self.resid_response/\
                    model.family.link.deriv(model.mu))
        self.resid_anscombe = model.family.resid_anscombe(model.y,model.mu)
        self.resid_dev = model.family.devresid(model.y, model.mu)
        self.pearsonX2 = np.sum(self.resid_pearson**2)
        self.predict = np.dot(model._exog,self.params)
        null = WLS(model.y,np.ones((len(model.y),1)),weights=model.data_weights).\
                fit().predict # predicted values of constant only fit
        self.null_deviance = model.family.deviance(model.y,null)
        self.deviance = model.family.deviance(model.y,model.mu)

    @property
    def llf(self):
        if self._llf is None:
            self._llf = self.model.family.logL(self.model.y,
                    self.model.mu, scale=self.scale)
        return self._llf

    def information_criteria(self):
        llf = self.llf
        aic = -2 * llf + 2*(self.df_model+1)
        bic = self.model.family.deviance(self.model.y,self.model.mu) - \
                (self.df_resid)*np.log(self.nobs)
        # this doesn't appear to be correct?
        return dict(aic=aic, bic=bic)
