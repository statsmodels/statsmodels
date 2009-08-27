"""
Generalized linear models
"""

import numpy as np
import family, tools
from regression import WLS#,GLS #might need for mlogit
from model import LikelihoodModel, LikelihoodModelResults

__all__ = ['GLM']

class GLM(LikelihoodModel):
    '''
    Generalized Linear Models class

    GLM inherits from statsmodels.LikelihoodModel

    Parameters
    -----------
    endog : array-like
        1d array of endogenous response variable.  This array can be
        1d or 2d for Binomial family models.  See examples/glm.py for more
        information
TODO: fix example location
    exog : array-like
        n x p design / exogenous data array
    family : family class instance
        The default is Gaussian.  If for example, you would like to
        specify the binomial distribution and you have imported
        statsmodels then family = statsmodels.family.Binomial().
        Each family can take a link instance as an argument.  See
        statsmodels.family.family for more information.

    Attributes
    -----------
    df_model : float
        Model degrees of freedom is equal to p - 1, where p is the number
        of regressors.  Note that the intercept is not reported as a
        degree of freedom.
    df_resid : float
        Residual degrees of freedom is equal to the number of observation n
        minus the number of regressors p.

    endog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    history : dict
        Contains information about the iterations. Its keys are `fittedvalues`,
        `deviance`, and `params`.
    iteration : int
        The number of iterations that fit has run.  Initialized at 0.
    family : family class instance
        A pointer to the distribution family of the model.
    mu : array
        The mean response of the transformed variable.  `mu` is the value of
        the inverse of the link function at eta, where eta is the linear
        predicted value of the WLS fit of the transformed variable.  `mu` is
        only available after fit is called.  See
        statsmodels.family.family.fitted of the distribution family for more
        information.
    normalized_cov_params : array
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)
    pinv_wexog : array
        The pseudoinverse of the design / exogenous data array.  Note that
        GLM has no whiten method, so this is just the pseudo inverse of the
        design.
        The pseudoinverse is approximately equal to (X.T X)^(-1)X.T
    scale : float
        The estimate of the scale / dispersion of the model fit.  Only
        available after fit is called.  See GLM.fit and GLM.estimate_scale
        for more information.
    scaletype : str
        The scaling used for fitting the model.  This is only available after
        fit is called.  The default is None.  See GLM.fit for more information.
    weights : array
        The value of the weights after the last iteration of fit.  Only
        available after fit is called.  See statsmodels.family.family for
        the specific distribution weighting functions.

    Methods
    -------
    estimate_scale
        Estimates the dispersion / scale of the model.
    fit
        Fits the generalized linear model using iteratively reweighted least
        squares.
    information
        Returns the Fisher information matrix.
        Not yet implemented.
    initialize
        (Re)initialize the design.  This resets history and number of
        iterations.
    loglike
        Returns the loglikelihood at `params` for a given distribution family.
    newton
        Used to fit the model via Newton-Raphson.
        Not yet implemented.
    predict
        Returns the linear predictor of the model.
        dot(X,params).  See the fitted method of the specific family
        for the inverse of the link function.
    score
        Returns the score matrix of the model.
        Not yet implemented.


    Examples
    --------
    >>> from scikits import statsmodels as models
    >>> from scikits.statsmodels.datasets.scotland import Load
    >>> data = Load()
    >>> data.exog = models.tools.add_constant(data.exog)

    Instantiate a gamma family model with the default link function.

    >>> gamma_model = models.GLM(data.endog, data.exog,
            family=models.family.Gamma())
    >>> gamma_results = gamma_model.fit()
    >>> gamma_results.params
    array([  4.96176830e-05,   2.03442259e-03,  -7.18142874e-05,
         1.11852013e-04,  -1.46751504e-07,  -5.18683112e-04,
        -2.42717498e-06,  -1.77652703e-02])
    >>> gamma.scale
    0.0035842831734919055
    >>> gamma_results.deviance
    0.087388516416999198
    >>>gamma_results.pearsonX2
    0.086022796163805704
    >>> gamma_results.llf
    -83.017202161073527

    See also
    --------
    statsmodels.family.*

    Notes
    -----
    Only the following combinations make sense for family and link
                 + ident log logit probit cloglog pow opow nbinom loglog logc
    Gaussian     |   x    x                        x
    inv Gaussian |   x    x                        x
    binomial     |   x    x    x     x       x     x    x           x      x
    Poission     |   x    x                        x
    neg binomial |   x    x                        x          x
    gamma        |   x    x                        x

    Not all of these link functions are currently available.

    References
    ----------
    Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach.
        SAGE QASS Series.

    Green, PJ. 1984.  "Iteratively reweighted least squares for maximum
        likelihood estimation, and some robust and resistant alternatives."
        Journal of the Royal Statistical Society, Series B, 46, 149-192.

    Hardin, J.W. and Hilbe, J.M. 2007.  "Generalized Linear Models and
        Extensions."  2nd ed.  Stata Press, College Station, TX.

    McCullagh, P. and Nelder, J.A.  1989.  "Generalized Linear Models." 2nd ed.
        Chapman & Hall, Boca Rotan.
    '''

    def __init__(self, endog, exog, family=family.Gaussian()):
        if endog.shape[0] != exog.shape[0]:
            raise ValueError, 'size of endog does not match shape of exog'
        self.family = family
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)
        self.initialize()

    def initialize(self):
        """
        Initialize a generalized linear model.
        """
#TODO: intended for public use?
        self.history = { 'fittedvalues' : [], 'params' : [np.inf],
                         'deviance' : [np.inf]}
        self.iteration = 0
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                                        np.transpose(self.pinv_wexog))
        self.df_model = tools.rank(self.exog)-1
        self.df_resid = self.exog.shape[0] - tools.rank(self.exog)

    def score(self, params):
        """
        Score matrix.  Not yet implemeneted
        """
        raise NotImplementedError

    def loglike(self, *args):
        """
        Loglikelihood function.

        Each distribution family has its own loglikelihood function.
        See statsmodels.family.family
        """
        return self.family.loglike(*args)

    def information(self, params):
        """
        Fisher information matrix.  Not yet implemented.
        """
        raise NotImplementedError

    def _update_history(self, tmp_result, mu):
        """
        Helper method to update history during iterative fit.
        """
        self.history['params'].append(tmp_result.params)
        self.history['fittedvalues'].append(tmp_result.fittedvalues)
        self.history['deviance'].append(self.family.deviance(self.endog, mu))

    def estimate_scale(self, mu):
        """
        Estimates the dispersion/scale.

        Type of scale can be chose in the fit method.

        Parameters
        ----------
        mu : array
            mu is the mean response estimate

        Returns
        --------
        Estimate of scale

        Notes
        -----
        The default scale for Binomial and Poisson families is 1.  The default
        for the other families is Pearson's Chi-Square estimate.

        See also
        --------
        statsmodels.glm.fit for more information
        """
        if not self.scaletype:
            if isinstance(self.family, (family.Binomial, family.Poisson)):
                return np.array(1.)
            else:
                resid = self.endog - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum() \
                    / self.df_resid)

        if isinstance(self.scaletype, float):
            return np.array(self.scaletype)

        if isinstance(self.scaletype, str):
            if self.scaletype.lower() == 'x2':
                resid = self.endog - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum() \
                    / self.df_resid)
            elif self.scaletype.lower() == 'dev':
                return self.family.deviance(self.endog, mu)/self.df_resid
            else:
                raise ValueError, "Scale %s with type %s not understood" %\
                    (self.scaletype,type(self.scaletype))

        else:
            raise ValueError, "Scale %s with type %s not understood" %\
                (self.scaletype, type(self.scaletype))

    def predict(self, exog, params=None):
        """
        Return linear predicted values for a design matrix

        Parameters
        ---------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters / coefficients of a GLM.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        if self._results is None and params is None:
            raise ValueError, "If the model has not been fit, then you must \
specify the params argument."
        if self._results is not None:
            return np.dot(exog, self.results.params)
        else:
            return np.dot(exog, params)

    def fit(self, maxiter=100, method='IRLS', tol=1e-8, data_weights=1.,
            scale=None):
        '''
        Fits a generalized linear model for a given family.

        parameters
        ----------
         data_weights : array-like or scalar, only used with Binomial
            Number of trials for each observation. Used for only for
            binomial data when `endog` is specified as a 2d array of
            (successes, failures). Note that this argument will be
            dropped in the future.
        maxiter : int, optional
            Default is 100.
        method : string
            Default is 'IRLS' for iteratively reweighted least squares.  This
            is currently the only method available for GLM fit.
        scale : string or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        tol : float
            Convergence tolerance.  Default is 1e-8.
        '''
        if np.shape(data_weights) != () and not isinstance(self.family,
                family.Binomial):
            raise ValueError, "Data weights are only to be supplied for\
the Binomial family"
        self.data_weights = data_weights
        if np.shape(self.data_weights) == () and self.data_weights>1:
            self.data_weights = self.data_weights *\
                    np.ones((self.exog.shape[0]))
        self.scaletype = scale
        if isinstance(self.family, family.Binomial):
# thisc checks what kind of data is given for Binomial.  family will need a reference to
# endog if this is to be removed from the preprocessing
            self.endog = self.family.initialize(self.endog)
        mu = self.family.starting_mu(self.endog)
        wlsexog = self.exog
        eta = self.family.predict(mu)
        self.iteration += 1
        self.history['deviance'].append(self.family.deviance(self.endog, mu))
            # first guess on the deviance is assumed to be scaled by 1.
        while((np.fabs(self.history['deviance'][self.iteration]-\
                    self.history['deviance'][self.iteration-1])) > tol and \
                    self.iteration < maxiter):
            self.weights = data_weights*self.family.weights(mu)
            wlsendog = eta + self.family.link.deriv(mu) * (self.endog-mu)
                # - offset
            wls_results = WLS(wlsendog, wlsexog, self.weights).fit()
            eta = np.dot(self.exog, wls_results.params) # + offset
            mu = self.family.fitted(eta)
            self._update_history(wls_results, mu)
            self.scale = self.estimate_scale(mu)
            self.iteration += 1
        self.mu = mu
        glm_results = GLMResults(self, wls_results.params,
                wls_results.normalized_cov_params, self.scale)
        glm_results.bse = np.sqrt(np.diag(wls_results.cov_params(\
                scale=self.scale)))
        return glm_results

    @property
    def results(self):
        """
        A property that returns a GLMResults class.

        Notes
        -----
        Calls fit if it has not already been called.  The default values for
        fit are used.  If the data_weights argument needs to be supplied for
        the Binomial family, then you should directly call fit.
        """
        if self._results is None:
            self._results = self.fit()
        return self._results
#TODO: remove dataweights argument and have it calculated from endog
# note that data_weights is not documented because I'm going to remove it.
# make the number of trials an argument to Binomial if constant and 1d endog


class GLMResults(LikelihoodModelResults):
    '''
    Class to contain GLM results.

    GLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelReesults

    Attributes
    ----------
    aic : float
        Akaike Information Criterion
        -2 * `llf` + 2*(`df_model` + 1)
    bic : float
        Bayes Information Criterion
        `deviance` - `df_resid` * log(`nobs`)
    deviance : float
        See statsmodels.family.family for the distribution-specific deviance
        functions.
    df_model : float
        See GLM.df_model
    df_resid : float
        See GLM.df_resid
    fittedvalues : array
        Linear predicted values for the fitted model.
        dot(exog, params)
    llf : float
        Value of the loglikelihood function evalued at params.
        See statsmodels.family.family for distribution-specific loglikelihoods.
    model : class instance
        Pointer to GLM model instance that called fit.
    mu : array
        See GLM docstring.
    nobs : float
        The number of observations n.
    normalized_cov_params : array
        See GLM docstring
    null_deviance : float
        The value of the deviance function for the model fit with a constant
        as the only regressor.
    params : array
        The coefficients of the fitted model.  Note that interpretation
        of the coefficients often depends on the distribution family and the
        data.
    pearsonX2 : array
        Pearson's Chi-Squared statistic is defined as the sum of the squares
        of the Pearson residuals.
    pinv_wexog : array
        See GLM docstring.
    resid_anscombe : array
        Anscombe residuals.  See statsmodels.family.family for distribution-
        specific Anscombe residuals.
    resid_dev : array
        Deviance residuals.  See statsmodels.family.family for distribution-
        specific deviance residuals.
    resid_pearson : array
        Pearson residuals.  The Pearson residuals are defined as
        (`endog` - `mu`)/sqrt(VAR(`mu`)) where VAR is the distribution
        specific variance function.  See statsmodels.family.family and
        statsmodels.family.varfuncs for more information.
    resid_response : array
        Respnose residuals.  The response residuals are defined as
        `endog` - `fittedvalues`
    resid_working : array
        Working residuals.  The working residuals are defined as
        `resid_response`/link'(`mu`).  See statsmodels.family.links for the
        derivatives of the link functions.  They are defined analytically.
    scale : array
        The estimate of the scale / dispersion for the model fit.
        See GLM.fit and GLM.estimate_scale for more information.
    stand_errors : array
        The standard errors of the fitted GLM.
still named bse

    Methods
    -------
    conf_int
        Returns the confidence intervals for the parameter estimates.  See
        statsmodels.model.LikelihoodModelResults.conf_int for more information.
        Note that the confidence interval for the GLMs are based on the
        standard normal distribution.
    cov_params
        Returns the estimated covariance matrix scaled by `scale`.
    f_test
        Compute an F test / F contrast for a contrast matrix.
        See statsmodels.model.LikelihoodModelResults.f_test for more
        information.  Note that the f_test method for GLMs is untested.
    t
        Return the t-values for the parameter estimates.  Note that the
        z values are more appropriate for GLMs.  A convenenience function
        is not yet implemented for z values.
    t_test
        t test linear restrictions for the model.
        See statsmodels.model.LikelihoodModelResults.t_test for more
        information.  Note that t_test for GLMS is untested.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''
#TODO: add a z value function to LLMResults
    _llf = None

    def __init__(self, model, params, normalized_cov_params, scale):
        super(GLMResults, self).__init__(model, params,
                normalized_cov_params=normalized_cov_params, scale=scale)
        self._get_results(model)

    def _get_results(self, model):
        self.nobs = model.endog.shape[0]
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.mu = model.mu
        self.pinv_wexog = model.pinv_wexog
#        self.bse = np.sqrt(np.diag(tmp_results.cov_params(scale=model.scale)))
        self.resid_response = model.data_weights*(model.endog - model.mu)
        self.resid_pearson = np.sqrt(model.data_weights)*(model.endog-\
                model.mu)/np.sqrt(model.family.variance(model.mu))
        self.resid_working = model.data_weights * (self.resid_response/\
                    model.family.link.deriv(model.mu))
        self.resid_anscombe = model.family.resid_anscombe(model.endog,model.mu)
        self.resid_dev = model.family.devresid(model.endog, model.mu)
        self.pearsonX2 = np.sum(self.resid_pearson**2)
        self.fittedvalues = np.dot(model.exog, self.params)
        null = WLS(model.endog,np.ones((len(model.endog),1)),
            weights=model.data_weights).fit().fittedvalues
        # null is the predicted values of constant only fit
        self.null_deviance = model.family.deviance(model.endog,null)
        self.deviance = model.family.deviance(model.endog,model.mu)
        self.aic = -2 * self.llf + 2*(self.df_model+1)
        self.bic = self.deviance - self.df_resid*np.log(self.nobs)

    @property
    def llf(self):
        if self._llf is None:
            if isinstance(self.model.family, family.NegativeBinomial):
                self._llf = self.model.family.loglike(self.model.endog,
                    fittedvalues=self.fittedvalues)
            else:
                self._llf = self.model.family.loglike(self.model.endog,
                    self.model.mu, scale=self.scale)
        return self._llf

#TODO: write summary method to return a string
