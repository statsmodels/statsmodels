"""
Created on Tue Dec 20 20:24:20 2011

Author: Josef Perktold
License: BSD-3

"""

import numpy as np

from statsmodels.regression.linear_model import GLS, OLS, WLS


def atleast_2dcols(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


class GLSHet2(GLS):
    """
    WLS with heteroscedasticity that depends on explanatory variables

    Notes
    -----
    Mixing GLS sigma and weights for heteroscedasticity might not make
    sense.

    This should probably be rewritten following the pattern of GLSAR;
    the stopping criteria could also be improved in GLSAR, e.g. change
    in rho.
    """

    def __init__(self, endog, exog, exog_var, sigma=None):
        self.exog_var = atleast_2dcols(exog_var)
        super(self.__class__, self).__init__(endog, exog, sigma=sigma)

    def fit(self, lambd=1.):
        # maybe iterate
        # preliminary estimate
        res_gls = GLS(self.endog, self.exog, sigma=self.sigma).fit()
        res_resid = OLS(res_gls.resid**2, self.exog_var).fit()
        # or  log-link
        # res_resid = OLS(np.log(res_gls.resid**2), self.exog_var).fit()
        # here I could use whiten and current instance instead of delegating
        # but this is easier
        # see pattern of GLSAR, calls self.initialize and self.fit
        res_wls = WLS(self.endog, self.exog, weights=1./res_resid.fittedvalues).fit()

        res_wls._results.results_residual_regression = res_resid
        return res_wls


class GLSHet(WLS):
    """
    A regression model with an estimated heteroscedasticity.

    A subclass of WLS, that additionally estimates the weight matrix as a
    function of additional explanatory variables.

    Parameters
    ----------
    endog : array_like
        The dependent variable.
    exog : array_like
        The independent variables.
    exog_var : array_like, 1d or 2d
        Regressors, explanatory variables for the variance.
    weights : array_like or None
        If weights are given, then they are used in the first step
        estimation.
    link : link function or None
        If None, then the variance is assumed to be a linear combination of
        the exog_var. If given, then ... not tested yet.

    Attributes
    ----------
    history : dict
        Contains the parameter estimates in both regressions for each
        iteration, populated after calling `iterative_fit`.

    Notes
    -----
    Except for `iterative_fit`, all methods are inherited from WLS.

    The results instance additionally has a
    ``results_residual_regression`` attribute, an OLS result instance
    from the heteroscedasticity estimation.

    GLSHet is considered to be experimental.

    `fit` is just standard WLS fit for fixed weights.
    `iterative_fit` updates the estimate for weights, see its docstring.

    The two alternatives for handling heteroscedasticity in the data are to
    use heteroscedasticity robust standard errors or estimate the
    heteroscedasticity directly.
    Estimating heteroscedasticity and using weighted least squares produces
    smaller confidence intervals for the estimated parameters than the
    heteroscedasticity robust standard errors if the heteroscedasticity is
    correctly specified. If the heteroscedasticity is incorrectly specified
    then the estimated covariance is inconsistent.

    Stock and Watson, for example, argue in favor of using OLS with
    heteroscedasticity robust standard errors instead of GLSHet since we are
    seldom sure enough about the correct specification (in economics).

    GLSHet has asymptotically the same distribution as WLS if the true
    weights are known. In both cases the asymptotic distribution of the
    parameter estimates is the normal distribution.

    The assumption of the model::

        y = X*beta + u,
        with E(u) = 0, E(X*u) = 0, var(u_i) = z_i*gamma
        or for vector of all observations Sigma = diag(Z*gamma)

    where::

        y : endog (nobs)
        X : exog  (nobs, k_vars)
        Z : exog_var (nobs, k_vars2)
        beta, gamma estimated parameters

    If a link is specified, then the heteroscedasticity is::

        var(u_i) = link.inverse(z_i*gamma), or
        link(var(u_i)) = z_i*gamma

    for example for the log-link::

        var(u_i) = exp(z_i*gamma)

    The link option is not tested yet.
    """

    def __init__(self, endog, exog, exog_var=None, weights=None, link=None):
        self.exog_var = atleast_2dcols(exog_var)
        if weights is None:
            weights = np.ones(endog.shape)
        if link is not None:
            self.link = link
            self.linkinv = link.inverse   # as defined in families.links
        else:
            self.link = lambda x: x  # no transformation
            self.linkinv = lambda x: x

        super(self.__class__, self).__init__(endog, exog, weights=weights)

    def iterative_fit(self, maxiter=3):
        """
        Perform an iterative two-step procedure to estimate a WLS model

        The model is assumed to have heteroskedastic errors.
        The variance is estimated by OLS regression of the link transformed
        squared residuals on Z, i.e.::

           link(sigma_i) = x_i*gamma.

        Parameters
        ----------
        maxiter : int, optional
            The number of iterations.

        Returns
        -------
        RegressionResults
            The results of the final WLS fit, with the OLS result instance
            from the last heteroscedasticity estimation attached as the
            ``results_residual_regression`` attribute.

        Notes
        -----
        maxiter=1: returns the estimate based on the given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        Possible extension: stop iteration if the change in parameter
        estimates is smaller than some tolerance.

        Repeated calls to iterative_fit will do one redundant pinv_wexog
        calculation. Calling iterative_fit(maxiter) once does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        """
        import collections
        self.history = collections.defaultdict(list)  # not really necessary
        res_resid = None  # if maxiter < 2 no updating
        for i in range(maxiter):
            # pinv_wexog is cached
            if hasattr(self, "pinv_wexog"):
                del self.pinv_wexog
            # self.initialize()
            # print 'wls self',
            results = self.fit()
            self.history["self_params"].append(results.params)
            if not i == maxiter-1:  # s kip for last iteration, could break instead
                # print 'ols',
                self.results_old = results  # for debugging
                # estimate heteroscedasticity
                res_resid = OLS(self.link(results.resid**2), self.exog_var).fit()
                self.history["ols_params"].append(res_resid.params)
                # update weights
                self.weights = 1./self.linkinv(res_resid.fittedvalues)
                self.weights /= self.weights.max()  # not required
                self.weights[self.weights < 1e-14] = 1e-14  # clip
                # print 'in iter', i, self.weights.var() # debug, do weights change
                self.initialize()

        # note results is the wrapper, results._results is the results instance
        results._results.results_residual_regression = res_resid
        return results
