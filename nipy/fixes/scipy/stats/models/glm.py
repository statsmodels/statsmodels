"""
General linear models
--------------------

"""

import numpy as np
from models import family, tools
from models.regression import WLS,GLS   #don't need gls, might for mlogit
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

class GLM(LikelihoodModel):
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
    def __init__(self, endog, exog, family=family.Gaussian()):
        self.family = family
        self._endog = endog
        self._exog = exog
        self.initialize()

    def initialize(self):
        self.history = { 'predict' : [], 'params' : [np.inf],
                         'deviance' : [np.inf]}
        self.iteration = 0
        self.y = self._endog
        self.calc_params = np.linalg.pinv(self._exog)
        self.normalized_cov_params = np.dot(self.calc_params,
                                        np.transpose(self.calc_params))
        self.df_model = tools.rank(self._exog)-1
        self.df_resid = self._exog.shape[0] - tools.rank(self._exog)

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
        Estimates the dispersion/scale.

        Depends on the specification of scale in the fit method.

        Default for Binomial and Poisson families is 1.

        Default for the other families is Pearson's Chi-Square estimate.

        See fit() for other options.

        Parameters
        ----------
        mu : array
            mu is the mean response estimate

        Returns
        ----------
        Estimate of scale
        """
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
        Fits a glm model

        parameters
        ----------
        scale : string or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default is `X2` for Gamma, Gaussian, and Inverse Gaussian
                `X2` is Pearson's chi-squared divided by the residual
                degrees of freedom
            The default is 1 for Binomial and Poisson
            `dev` scales by the deviance divided by the residual
                    degrees of freedom

        data_weights : array-like
            Number of trials for each observation. Used for only for
            binomial data.
        '''
        if self._endog.shape[0] != self._exog.shape[0]:
            raise ValueError, 'size of Y does not match shape of design'
#FIXME: Why have these checks here and not in the base class or model?
        if np.shape(data_weights) != () and not isinstance(self.family,
                family.Binomial):
            raise AttributeError, "Data weights are only to be supplied for\
the Binomial family"
        self.data_weights = data_weights
        if np.shape(self.data_weights) == () and self.data_weights>1:
            self.data_weights = self.data_weights * np.ones((self._exog.shape[0]))
        self.scaletype = scale
        if isinstance(self.family, family.Binomial):
            self.y = self.family.initialize(self.y)
        mu = self.family.starting_mu(self.y)
        wls_exog = self._exog
        eta = self.family.predict(mu)
        self.iteration += 1
        self.history['deviance'].append(self.family.deviance(self.y, mu) / 1.)
        while ((np.fabs(self.history['deviance'][self.iteration]-\
                    self.history['deviance'][self.iteration-1])) > tol and \
                    self.iteration < maxiter):
            self.weights = data_weights*self.family.weights(mu)
#FIXME: What were these added to correct?
#            if self.weights.ndim == 2:
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
        self.results = GLMResults(self, wls_results.params,
                wls_results.normalized_cov_params, self.scale)
        self.results.bse = np.sqrt(np.diag(wls_results.cov_params(\
                scale=self.scale)))
        return self.results

class GLMResults(LikelihoodModelResults):
    '''
    Class to contain GLM results
    '''

    _llf = None

    def __init__(self, model, params, normalized_cov_params, scale):
        super(GLMResults, self).__init__(model, params,
                normalized_cov_params=normalized_cov_params, scale=scale)
        self._get_results(model)

    def _get_results(self, model):
        self.nobs = model.y.shape[0]
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.mu = model.mu
        self.calc_params = model.calc_params
#        self.bse = np.sqrt(np.diag(tmp_results.cov_params(scale=model.scale)))
        self.resid_response = model.data_weights*(model.y - model.mu)
        self.resid_pearson = np.sqrt(model.data_weights)*(model.y-model.mu)/\
                np.sqrt(model.family.variance(model.mu))
        self.resid_working = model.data_weights * (self.resid_response/\
                    model.family.link.deriv(model.mu))
        self.resid_anscombe = model.family.resid_anscombe(model.y,model.mu)
        self.resid_dev = model.family.devresid(model.y, model.mu)
        self.pearsonX2 = np.sum(self.resid_pearson**2)
        self.predict = np.dot(model._exog,self.params)
        null = WLS(model.y,np.ones((len(model.y),1)),
            weights=model.data_weights).fit().predict
        # null is the predicted values of constant only fit
        self.null_deviance = model.family.deviance(model.y,null)
        self.deviance = model.family.deviance(model.y,model.mu)
        self.aic = -2 * self.llf + 2*(self.df_model+1)
        self.bic = self.model.family.deviance(self.model.y,self.model.mu) -\
                (self.df_resid)*np.log(self.nobs)

    @property
    def llf(self):
        if self._llf is None:
            if isinstance(self.model.family, family.NegativeBinomial):
                self._llf = self.model.family.logL(self.model.y,
                    predicted=self.predict)
            else:
                self._llf = self.model.family.logL(self.model.y,
                    self.model.mu, scale=self.scale)
        return self._llf
