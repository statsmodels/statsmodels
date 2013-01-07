"""
This module implements a basic nonlinear regression model, 
NonlinearLS.

Models are specified with a model function, an endogenous response variable 
and an exogenous design matrix and are fit using their `fit` method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

D. C. Montgomery and E.A. Peck. "Introduction to Linear Regression
    Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models:

R. Davidson and J.G. MacKinnon.  "Econometric Theory and Methods," Oxford,
    2004.

W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.
"""

__docformat__ = 'restructuredtext en'

__all__ = ['NonlinearLS']

import inspect
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats.stats import ss
import scipy.stats as stats
from scipy.optimize import curve_fit
from statsmodels.tools.tools import (add_constant, rank,
                                             recipr, chain_dot)
from statsmodels.tools.decorators import (resettable_cache,
        cache_readonly, cache_writable)
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap

class NonlinearLS(base.LikelihoodModel):
    """
    Nonlinear least squares model with a general covariance structure.

    Parameters
    ----------
    f : callable
           model function f(x, ...). Takes exogenous variable(s) are the first 
           argument, and parameters as separate subsequent arguments.
    endog : array-like
           endog is a 1-d vector that contains the response/independent variable
    exog : array-like
           exog is a n x p vector where n is the number of observations and p is
           the number of regressors/dependent variables including the intercept
           if one is included in the data.
    sigma : scalar or array
           `sigma` is the weighting matrix of the covariance.
           The default is None for no scaling.  If `sigma` is a scalar, it is
           assumed that `sigma` is an n x n diagonal matrix with the given
           scalar, `sigma` as the value of each diagonal element.  If `sigma`
           is an n-length vector, then `sigma` is assumed to be a diagonal
           matrix with the given `sigma` on the diagonal.  This should be the
           same as WLS.

    Attributes
    ----------
    df_model : float
        p - 1, where p is the number of parameters.
    df_resid : float
        Number of observations n less the number of parameters p.
    llf : float
        The value of the likelihood function of the fitted model.
    nobs : float
        The number of observations n.
    normalized_cov_params : array
        p x p array :math:`(X^{T}\Sigma^{-1}X)^{-1}`
    results : RegressionResults instance
        A property that returns the RegressionResults class if fit.
    sigma : array
        `sigma` is the n x n covariance structure of the error terms.
    wexog : array
        Design matrix whitened by `cholsigmainv`
    wendog : array
        Response variable whitened by `cholsigmainv`

    Notes
    -----
    If sigma is a function of the data making one of the regressors
    a constant, then the current postestimation statistics will not be correct.


    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> ols_resid = sm.OLS(data.endog, data.exog).fit().resid
    >>> res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()
    >>> rho = res_fit.params

    `rho` is a consistent estimator of the correlation of the residuals from
    an OLS fit of the longley data.  It is assumed that this is the true rho
    of the AR process data.

    >>> from scipy.linalg import toeplitz
    >>> order = toeplitz(np.arange(16))
    >>> sigma = rho**order

    `sigma` is an n x n matrix of the autocorrelation structure of the
    data.

    >>> gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
    >>> gls_results = gls_model.results

    """

    def __init__(self, f, endog, exog, sigma=None):
        self.f = f

        if sigma is not None:
            self.sigma = np.asarray(sigma)
        else:
            self.sigma = sigma
        if self.sigma is not None and not self.sigma.shape == (): #greedy logic
            nobs = int(endog.shape[0])
            if self.sigma.ndim == 1 or np.squeeze(self.sigma).ndim == 1:
                if self.sigma.shape[0] != nobs:
                    raise ValueError("sigma is not the correct dimension.  \
Should be of length %s, if sigma is a 1d array" % nobs)
            elif self.sigma.shape[0] != nobs and \
                    self.sigma.shape[1] != nobs:
                raise ValueError("expected an %s x %s array for sigma" % \
                        (nobs, nobs))
        if self.sigma is not None:
            nobs = int(endog.shape[0])
            if self.sigma.shape == ():
                self.sigma = np.diag(np.ones(nobs)*self.sigma)
            if np.squeeze(self.sigma).ndim == 1:
                self.sigma = np.diag(np.squeeze(self.sigma))
            self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(\
                    self.sigma)).T
        super(NonlinearLS, self).__init__(endog, exog)

    def initialize(self):
        #print "calling initialize, now whitening"  #for debugging
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # overwrite nobs from class Model:
        self.nobs = float(self.wexog.shape[0])
        self.df_resid = float(self.nobs - 
                              len(inspect.getargspec(self.f).args) + 1)
        self.df_model = float(len(inspect.getargspec(self.f).args) - 2)

    def whiten(self, X):
        """
        NonlinearLS whiten method.

        Parameters
        -----------
        X : array-like
            Data to be whitened.

        Returns
        -------
        np.dot(cholsigmainv,X)

        See Also
        --------
        regression.NonlinearLS
        """
        X = np.asarray(X)
        if np.any(self.sigma) and not self.sigma==():
            return np.dot(self.cholsigmainv, X)
        else:
            return X

    def fit(self, **kwargs):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Returns
        -------
        A RegressionResults class instance.

        See Also
        ---------
        regression.RegressionResults

        """
        exog = self.wexog
        endog = self.wendog
        
        if "Dfun" in kwargs:
            dfun = kwargs["Dfun"]
            def dfun_wrapper(var, xs, ys, f):
                return dfun(xs, *var)
                
            kwargs["Dfun"] = dfun_wrapper
        
        beta, self.normalized_cov_params = curve_fit(self.f, exog.T, endog, sigma=self.sigma,
                                                     **kwargs)

        self._data.xnames = ["x%s" % i for i in range(1, len(beta) + 1)]
        
        lfit = RegressionResults(self, beta,
                                 normalized_cov_params=self.normalized_cov_params)
        
        return RegressionResultsWrapper(lfit)
        
    def predict(self, params=None, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array-like, optional after fit has been called
            Parameters of a linear model
        exog : array-like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        if exog is None:
            exog = self.wexog
        if params is None:
            params = self.params
        return np.array(self.f(exog.T, *params))

    def loglike(self, params):
        """
        Returns the value of the gaussian loglikelihood function at params.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        params : array-like
            The parameter estimates

        Returns
        -------
        loglike : float
            The value of the loglikelihood function for a NonlinearLS Model.


        Notes
        -----
        The loglikelihood function for the normal distribution is

        .. math:: -\\frac{n}{2}\\log\\left(Y-\\hat{Y}\\right)-\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)-\\frac{1}{2}\\log\\left(\\left|\\Sigma\\right|\\right)

        Y and Y-hat are whitened.

        """
#TODO: combine this with OLS/WLS loglike and add _det_sigma argument
        nobs2 = self.nobs / 2.0
        SSR = ss(self.wendog - np.array(self.f(self.wexog.T, *params)))
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with likelihood constant
        if np.any(self.sigma) and self.sigma.ndim == 2:
#FIXME: robust-enough check?  unneeded if _det_sigma gets defined
            llf -= .5*np.log(np.linalg.det(self.sigma))
            # with error covariance matrix
        return llf




class RegressionResults(base.LikelihoodModelResults):
    """
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    Returns
    -------
    **Attributes**

    aic
        Aikake's information criteria :math:`-2llf + 2(df_model+1)`
    bic
        Bayes' information criteria :math:`-2llf + \log(n)(df_model+1)`
    bse
        The standard errors of the parameter estimates.
    pinv_wexog
        See specific model class docstring
    centered_tss
        The total sum of squares centered about the mean
    cov_HC0
        See HC0_se below.  Only available after calling HC0_se.
    cov_HC1
        See HC1_se below.  Only available after calling HC1_se.
    cov_HC2
        See HC2_se below.  Only available after calling HC2_se.
    cov_HC3
        See HC3_se below.  Only available after calling HC3_se.
    df_model :
        Model degress of freedom. The number of regressors p - 1 for the
        constant  Note that df_model does not include the constant even though
        the design does.  The design is always assumed to have a constant
        in calculating results for now.
    df_resid
        Residual degrees of freedom. n - p.  Note that the constant *is*
        included in calculating the residual degrees of freedom.
    ess
        Explained sum of squares.  The centered total sum of squares minus
        the sum of squared residuals.
    fvalue
        F-statistic of the fully specified model.  Calculated as the mean
        squared error of the model divided by the mean squared error of the
        residuals.
    f_pvalue
        p-value of the F-statistic
    fittedvalues
        The predicted the values for the original (unwhitened) design.
    het_scale
        Only available if HC#_se is called.  See HC#_se for more information.
    HC0_se
        White's (1980) heteroskedasticity robust standard errors.
        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        where e_i = resid[i]
        HC0_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC0, which is the full heteroskedasticity
        consistent covariance matrix and also `het_scale`, which is in
        this case just resid**2.  HCCM matrices are only appropriate for OLS.
    HC1_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as sqrt(diag(n/(n-p)*HC_0)
        HC1_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC1, which is the full HCCM and also `het_scale`,
        which is in this case n/(n-p)*resid**2.  HCCM matrices are only
        appropriate for OLS.
    HC2_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC2_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC2, which is the full HCCM and also `het_scale`,
        which is in this case is resid^(2)/(1-h_ii).  HCCM matrices are only
        appropriate for OLS.
    HC3_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC3_se is a property.  It is not evaluated until it is called.
        When it is called the RegressionResults instance will then have
        another attribute cov_HC3, which is the full HCCM and also `het_scale`,
        which is in this case is resid^(2)/(1-h_ii)^(2).  HCCM matrices are
        only appropriate for OLS.
    model
        A pointer to the model instance that called fit() or results.
    mse_model
        Mean squared error the model. This is the explained sum of squares
        divided by the model degrees of freedom.
    mse_resid
        Mean squared error of the residuals.  The sum of squared residuals
        divided by the residual degrees of freedom.
    mse_total
        Total mean squared error.  Defined as the uncentered total sum of
        squares divided by n the number of observations.
    nobs
        Number of observations n.
    normalized_cov_params
        See specific model class docstring
    params
        The parameter values that minimize the least squares criterion.
    pvalues
        The two-tailed p values for the t-stats of the params.
    resid
        The residuals of the model.
    rsquared
        R-squared of a model with an intercept.  This is defined here as
        1 - `ssr`/`centered_tss`
    rsquared_adj
        Adjusted R-squared.  This is defined here as
        1 - (n-1)/(n-p)*(1-`rsquared`)
    scale
        A scale factor for the covariance matrix.
        Default value is ssr/(n-p).  Note that the square root of `scale` is
        often called the standard error of the regression.
    ssr
        Sum of squared (whitened) residuals.
    uncentered_tss
        Uncentered sum of squares.  Sum of the squared values of the
        (whitened) endogenous response variable.
    wresid
        The residuals of the transformed/whitened regressand and regressor(s)
    """

    # For robust covariance matrix properties
    _HC0_se = None
    _HC1_se = None
    _HC2_se = None
    _HC3_se = None

    _cache = {} # needs to be a class attribute for scale setter?

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(RegressionResults, self).__init__(model, params,
                                                 normalized_cov_params,
                                                 scale)
        self._cache = resettable_cache()

    def __str__(self):
        self.summary()

##    def __repr__(self):
##        print self.summary()

    def conf_int(self, alpha=.05, cols=None):
        """
        Returns the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
            `cols` specifies which confidence intervals to return

        Notes
        -----
        The confidence interval is based on Student's t-distribution.
        """
        bse = self.bse
        params = self.params
        dist = stats.t
        q = dist.ppf(1 - alpha / 2, self.df_resid)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = params[cols] - q * bse[cols]
            upper = params[cols] + q * bse[cols]
        return np.asarray(zip(lower, upper))

    @cache_readonly
    def df_resid(self):
        return self.model.df_resid

    @cache_readonly
    def df_model(self):
        return self.model.df_model

    @cache_readonly
    def nobs(self):
        return float(self.model.wexog.shape[0])

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params, self.model.exog)

    @cache_readonly
    def wresid(self):
        return self.model.wendog - self.model.predict(self.params,
                self.model.wexog)

    @cache_readonly
    def resid(self):
        return self.model.endog - self.model.predict(self.params,
                self.model.exog)

#    def _getscale(self):
#        val = self._cache.get("scale", None)
#        if val is None:
#            val = ss(self.wresid) / self.df_resid
#            self._cache["scale"] = val
#        return val

#    def _setscale(self, val):
#        self._cache.setdefault("scale", val)

#    scale = property(_getscale, _setscale)

#TODO: fix writable example
    @cache_writable()
    def scale(self):
        wresid = self.wresid
        return np.dot(wresid, wresid) / self.df_resid

    @cache_readonly
    def ssr(self):
        wresid = self.wresid
        return np.dot(wresid, wresid)

    @cache_readonly
    def centered_tss(self):
        centered_wendog = self.model.wendog - np.mean(self.model.wendog)
        return np.dot(centered_wendog, centered_wendog)

    @cache_readonly
    def uncentered_tss(self):
        wendog = self.model.wendog
        return np.dot(wendog, wendog)

    @cache_readonly
    def ess(self):
        return self.centered_tss - self.ssr

# Centered R2 for models with intercepts
# have a look in test_regression.test_wls to see
# how to compute these stats for a model without intercept,
# and when the weights are a (linear?) function of the data...
    @cache_readonly
    def rsquared(self):
        return 1 - self.ssr/self.centered_tss

    @cache_readonly
    def rsquared_adj(self):
        return 1 - (self.nobs - 1)/self.df_resid * (1 - self.rsquared)

    @cache_readonly
    def mse_model(self):
        return self.ess/self.df_model

    @cache_readonly
    def mse_resid(self):
        return self.ssr/self.df_resid

    @cache_readonly
    def mse_total(self):
        return self.uncentered_tss/self.nobs

    @cache_readonly
    def fvalue(self):
        return self.mse_model/self.mse_resid

    @cache_readonly
    def f_pvalue(self):
        return stats.f.sf(self.fvalue, self.df_model, self.df_resid)

    @cache_readonly
    def bse(self):
        diags = np.diag(self.normalized_cov_params)
        return np.sqrt([n if n > 0 else 0
                        for n in diags])

    @cache_readonly
    def pvalues(self):
        return stats.t.sf(np.abs(self.tvalues), self.df_resid)*2

    @cache_readonly
    def aic(self):
        return -2 * self.llf + 2 * (self.df_model + 1)

    @cache_readonly
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1)

# Centered R2 for models with intercepts
# have a look in test_regression.test_wls to see
# how to compute these stats for a model without intercept,
# and when the weights are a (linear?) function of the data...

#TODO: make these properties reset bse
    def _HCCM(self, scale):
        H = np.dot(self.model.pinv_wexog,
            scale[:,None]*self.model.pinv_wexog.T)
        return H

    @property
    def HC0_se(self):
        """
        See statsmodels.RegressionResults
        """
        if self._HC0_se is None:
            self.het_scale = self.resid**2 # or whitened residuals? only OLS?
            self.cov_HC0 = self._HCCM(self.het_scale)
            self._HC0_se = np.sqrt(np.diag(self.cov_HC0))
        return self._HC0_se

    @property
    def HC1_se(self):
        """
        See statsmodels.RegressionResults
        """
        if self._HC1_se is None:
            self.het_scale = self.nobs/(self.df_resid)*(self.resid**2)
            self.cov_HC1 = self._HCCM(self.het_scale)
            self._HC1_se = np.sqrt(np.diag(self.cov_HC1))
        return self._HC1_se

    @property
    def HC2_se(self):
        """
        See statsmodels.RegressionResults
        """
        if self._HC2_se is None:
            # probably could be optimized
            h = np.diag(chain_dot(self.model.exog,
                                  self.normalized_cov_params,
                                  self.model.exog.T))
            self.het_scale = self.resid**2/(1-h)
            self.cov_HC2 = self._HCCM(self.het_scale)
            self._HC2_se = np.sqrt(np.diag(self.cov_HC2))
        return self._HC2_se

    @property
    def HC3_se(self):
        """
        See statsmodels.RegressionResults
        """
        if self._HC3_se is None:
            # above probably could be optimized to only calc the diag
            h = np.diag(chain_dot(self.model.exog,
                                  self.normalized_cov_params,
                                  self.model.exog.T))
            self.het_scale=(self.resid/(1-h))**2
            self.cov_HC3 = self._HCCM(self.het_scale)
            self._HC3_se = np.sqrt(np.diag(self.cov_HC3))
        return self._HC3_se

#TODO: this needs a test
    def norm_resid(self):
        """
        Residuals, normalized to have unit length and unit variance.

        Returns
        -------
        An array wresid/sqrt(scale)

        Notes
        -----
        This method is untested
        """
        if not hasattr(self, 'resid'):
            raise ValueError('need normalized residuals to estimate standard '
                             'deviation')
        return self.wresid * recipr(np.sqrt(self.scale))

    def compare_f_test(self, restricted):
        '''use F test to test whether restricted model is correct

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current
            model. The result instance of the restricted model is required to
            have two attributes, residual sum of squares, `ssr`, residual
            degrees of freedom, `df_resid`.

        Returns
        -------
        f_value : float
            test statistic, F distributed
        p_value : float
            p-value of the test statistic
        df_diff : int
            degrees of freedom of the restriction, i.e. difference in df between
            models

        Notes
        -----
        See mailing list discussion October 17,

        '''
        ssr_full = self.ssr
        ssr_restr = restricted.ssr
        df_full = self.df_resid
        df_restr = restricted.df_resid

        df_diff = (df_restr - df_full)
        f_value = (ssr_restr - ssr_full) / df_diff / ssr_full * df_full
        p_value = stats.f.sf(f_value, df_diff, df_full)
        return f_value, p_value, df_diff

    def compare_lr_test(self, restricted):
        '''
        Likelihood ratio test to test whether restricted model is correct

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current model.
            The result instance of the restricted model is required to have two
            attributes, residual sum of squares, `ssr`, residual degrees of
            freedom, `df_resid`.

        Returns
        -------
        lr_stat : float
            likelihood ratio, chisquare distributed with df_diff degrees of
            freedom
        p_value : float
            p-value of the test statistic
        df_diff : int
            degrees of freedom of the restriction, i.e. difference in df between
            models

        Notes
        -----

        .. math:: D=-2\\log\\left(\\frac{\\mathcal{L}_{null}}
           {\\mathcal{L}_{alternative}}\\right)

        where :math:`\mathcal{L}` is the likelihood of the model. With :math:`D`
        distributed as chisquare with df equal to difference in number of
        parameters or equivalently difference in residual degrees of freedom

        TODO: put into separate function, needs tests
        '''
#        See mailing list discussion October 17,
        llf_full = self.llf
        llf_restr = restricted.llf
        df_full = self.df_resid
        df_restr = restricted.df_resid

        lrdf = (df_restr - df_full)
        lrstat = -2*(llf_restr - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, lrdf)

        return lrstat, lr_pvalue, lrdf


    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        #TODO: import where we need it (for now), add as cached attributes
        from statsmodels.stats.stattools import (jarque_bera,
                omni_normtest, durbin_watson)
        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        #TODO: reuse condno from somewhere else ?
        #condno = np.linalg.cond(np.dot(self.wexog.T, self.wexog))
        wexog = self.model.wexog
        eigvals = np.linalg.linalg.eigvalsh(np.dot(wexog.T, wexog))
        eigvals = np.sort(eigvals) #in increasing order
        condno = np.sqrt(eigvals[-1]/eigvals[0])

        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                          omni=omni, omnipv=omnipv, condno=condno,
                          mineigval=eigvals[0])

#        #TODO not used yet
#        diagn_left_header = ['Models stats']
#        diagn_right_header = ['Residual stats']

        #TODO: requiring list/iterable is a bit annoying
        #need more control over formatting
        #TODO: default don't work if it's not identically spelled

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None), #[self.df_resid]), #TODO: spelling
                    ('Df Model:', None), #[self.df_model])
                    ]

        top_right = [('R-squared:', ["%#8.3f" % self.rsquared]),
                     ('Adj. R-squared:', ["%#8.3f" % self.rsquared_adj]),
                     ('F-statistic:', ["%#8.4g" % self.fvalue] ),
                     ('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
                     ('Log-Likelihood:', None), #["%#6.4g" % self.llf]),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
                     ]

        diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                      ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                      ('Skew:', ["%#6.3f" % skew]),
                      ('Kurtosis:', ["%#6.3f" % kurtosis])
                      ]

        diagn_right = [('Durbin-Watson:', ["%#8.3f" % durbin_watson(self.wresid)]),
                       ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                       ('Prob(JB):', ["%#8.3g" % jbpv]),
                       ('Cond. No.', ["%#8.3g" % condno])
                       ]


        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        #create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                          yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=.05,
                             use_t=True)

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")

        #add warnings/notes, added to text format only
        etext =[]
        if eigvals[0] < 1e-10:
            wstr = \
'''The smallest eigenvalue is %6.3g. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.''' \
                    % eigvals[0]
            etext.append(wstr)
        elif condno > 1000:  #TODO: what is recommended
            wstr = \
'''The condition number is large, %6.3g. This might indicate that there are
strong multicollinearity or other numerical problems.''' % condno
            etext.append(wstr)
        
        diags = np.diag(self.normalized_cov_params)
        if any(diags < 0):
            wstr = \
'''There were negative variance estimates, which have been set to 0. This 
might indicate problems with the model.'''
            etext.append(wstr)

        if etext:
            smry.add_extra_txt(etext)

        return smry

#        top = summary_top(self, gleft=topleft, gright=diagn_left, #[],
#                          yname=yname, xname=xname,
#                          title=self.model.__class__.__name__ + ' ' +
#                          "Regression Results")
#        par = summary_params(self, yname=yname, xname=xname, alpha=.05,
#                             use_t=False)
#
#        diagn = summary_top(self, gleft=diagn_left, gright=diagn_right,
#                          yname=yname, xname=xname,
#                          title="Linear Model")
#
#        return summary_return([top, par, diagn], return_fmt=return_fmt)


    def summary_old(self, yname=None, xname=None, returns='text'):
        """returns a string that summarizes the regression results

        Parameters
        -----------
        yname : string, optional
            Default is `Y`
        xname : list of strings, optional
            Default is `X.#` for # in p the number of regressors

        Returns
        -------
        String summarizing the fit of a linear model.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> ols_results = sm.OLS(data.endog, data.exog).fit()
        >>> print ols_results.summary()
        ...

        Notes
        -----
        All residual statistics are calculated on whitened residuals.
        """
        import time
        from statsmodels.iolib.table import SimpleTable
        from statsmodels.stats.stattools import (jarque_bera,
                omni_normtest, durbin_watson)

        if yname is None:
            yname = self.model.endog_names
        if xname is None:
            xname = self.model.exog_names
        modeltype = self.model.__class__.__name__

        llf, aic, bic = self.llf, self.aic, self.bic
        JB, JBpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        t = time.localtime()

        part1_fmt = dict(
            data_fmts = ["%s"],
            empty_cell = '',
            colwidths = 15,
            colsep=' ',
            row_pre = '| ',
            row_post = '|',
            table_dec_above='=',
            table_dec_below='',
            header_dec_below=None,
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = "r",
            stubs_align = "l",
            fmt = 'txt'
        )
        part2_fmt = dict(
            #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            data_fmts = ["%#10.4g","%#10.4g","%#6.4f","%#6.4f"],
            #data_fmts = ["%#15.4F","%#15.4F","%#15.4F","%#14.4G"],
            empty_cell = '',
            colwidths = 14,
            colsep=' ',
            row_pre = '| ',
            row_post = ' |',
            table_dec_above='=',
            table_dec_below='=',
            header_dec_below='-',
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = 'r',
            stubs_align = 'l',
            fmt = 'txt'
        )
        part3_fmt = dict(
            #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            data_fmts = ["%#10.4g","%#10.4g","%#10.4g","%#6.4g"],
            empty_cell = '',
            colwidths = 15,
            colsep='   ',
            row_pre = '| ',
            row_post = '  |',
            table_dec_above=None,
            table_dec_below='-',
            header_dec_below='-',
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = 'r',
            stubs_align = 'l',
            fmt = 'txt'
        )

        # Print the first part of the summary table
        part1data = [[yname],
                     [modeltype],
                     ['Least Squares'],
                     [time.strftime("%a, %d %b %Y",t)],
                     [time.strftime("%H:%M:%S",t)],
                     [self.nobs],
                     [self.df_resid],
                     [self.df_model]]
        part1header = None
        part1title = 'Summary of Regression Results'
        part1stubs = ('Dependent Variable:',
                      'Model:',
                      'Method:',
                      'Date:',
                      'Time:',
                      '# obs:',
                      'Df residuals:',
                      'Df model:')
        part1 = SimpleTable(part1data,
                            part1header,
                            part1stubs,
                            title=part1title,
                            txt_fmt = part1_fmt)

        ########  summary Part 2   #######

        part2data = zip([self.params[i] for i in range(len(xname))],
                        [self.bse[i] for i in range(len(xname))],
                        [self.tvalues[i] for i in range(len(xname))],
                        [self.pvalues[i] for i in range(len(xname))])
        part2header = ('parameter', 'std. error', 't-statistic', 'prob.')
        part2stubs = xname
        #dfmt={'data_fmt':["%#12.6g","%#12.6g","%#10.4g","%#5.4g"]}
        part2 = SimpleTable(part2data,
                            part2header,
                            part2stubs,
                            title=None,
                            txt_fmt = part2_fmt)

        #self.summary2 = part2
        ########  summary Part 3   #######

        part3Lheader = ['Models stats']
        part3Rheader = ['Residual stats']
        part3Lstubs = ('R-squared:',
                       'Adjusted R-squared:',
                       'F-statistic:',
                       'Prob (F-statistic):',
                       'Log likelihood:',
                       'AIC criterion:',
                       'BIC criterion:',)
        part3Rstubs = ('Durbin-Watson:',
                       'Omnibus:',
                       'Prob(Omnibus):',
                       'JB:',
                       'Prob(JB):',
                       'Skew:',
                       'Kurtosis:')
        part3Ldata = [[self.rsquared], [self.rsquared_adj],
                      [self.fvalue],
                      [self.f_pvalue],
                      [llf],
                      [aic],
                      [bic]]
        part3Rdata = [[durbin_watson(self.wresid)],
                      [omni],
                      [omnipv],
                      [JB],
                      [JBpv],
                      [skew],
                      [kurtosis]]
        part3L = SimpleTable(part3Ldata, part3Lheader, part3Lstubs,
                             txt_fmt = part3_fmt)
        part3R = SimpleTable(part3Rdata, part3Rheader, part3Rstubs,
                             txt_fmt = part3_fmt)
        part3L.extend_right(part3R)
        ########  Return Summary Tables ########
        # join table parts then print
        if returns == 'text':
            return str(part1) + '\n' +  str(part2) + '\n' + str(part3L)
        elif returns == 'tables':
            return [part1, part2 ,part3L]
        elif returns == 'csv':
            return part1.as_csv() + '\n' + part2.as_csv() + '\n' + \
                   part3L.as_csv()
        elif returns == 'latex':
            print('not available yet')
        elif returns == 'html':
            print('not available yet')

class RegressionResultsWrapper(wrap.ResultsWrapper):

    _attrs = {
        'chisq' : 'columns',
        'sresid' : 'rows',
        'weights' : 'rows',
        'wresid' : 'rows',
        'bcov_unscaled' : 'cov',
        'bcov_scaled' : 'cov',
        'HC0_se' : 'columns',
        'HC1_se' : 'columns',
        'HC2_se' : 'columns',
        'HC3_se' : 'columns'
    }

    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._attrs,
                                   _attrs)

    _methods = {
        'norm_resid' : 'rows',
    }

    _wrap_methods = wrap.union_dicts(
                        base.LikelihoodResultsWrapper._wrap_methods,
                        _methods)
wrap.populate_wrapper(RegressionResultsWrapper,
                      RegressionResults)
