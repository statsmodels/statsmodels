"""
This module implements some standard regression models:

Generalized Least Squares (GLS),
Ordinary Least Squares (OLS),
and Weighted Least Squares (WLS),
as well as an GLS model with autoregressive error terms GLSAR(p)

Models are specified with an endogenous response variable and an
exogenous design matrix and are fit using their `fit` method.

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

__all__ = ['GLS', 'WLS', 'OLS', 'GLSAR']

from string import join as sjoin    #
from csv import reader              # These are for read_array

import numpy as np
from scipy.linalg import norm, toeplitz
from scipy import stats
from scipy.stats.stats import ss
from model import LikelihoodModel, LikelihoodModelResults
import tools
from tools import add_constant

class GLS(LikelihoodModel):
    """
    Generalized least squares model with a general covariance structure.

    Parameters
    ----------
    endog : array-like
        `endog` is a 1-d vector that contains the response/independent variable
    exog : array-like
        `exog` is a n x p vector where n is the number of observations and
        p is the number of regressors/dependent variables including the intercept
        if one is included in the data.
    sigma : scalar or array
       `sigma` is the weighting matrix of the covariance.
       The default is None for no scaling.  If `sigma` is a scalar, it is
       assumed that `sigma` is an n x n diagonal matrix with the given sclar,
       `sigma` as the value of each diagonal element.  If `sigma` is an
       n-length vector, then `sigma` is assumed to be a diagonal matrix
       with the given `sigma` on the diagonal.  This should be the same as WLS.

    Attributes
    ----------
    pinv_wexog : array
        `pinv_wexog` is the p x n Moore-Penrose pseudoinverse
        of the whitened design matrix. In matrix notation it is approximately
        [X^(T)(sigma)^(-1)X]^(-1)X^(T)psi
        where psi is cholsigmainv.T
    cholsimgainv : array
        n x n upper triangular matrix such that in matrix notation
        (cholsigmainv^(T) cholsigmainv) = (sigma)^(-1).
        It is the transpose of the Cholesky decomposition of the pseudoinverse
        of sigma.
    df_model : float
        The model degrees of freedom is equal to p - 1, where p is the number
        of regressors.  Note that the intercept is not reported as a degree
        of freedom.
    df_resid : float
        The residual degrees of freedom is equal to the number of observations
        n less the number of parameters p.  Note that the intercept is counted as
        using a degree of freedom for the degrees of freedom of the
        residuals.
    llf : float
        The value of the likelihood function of the fitted model.
    nobs : float
        The number of observations n
    normalized_cov_params : array
        `normalized_cov_params` is a p x p array
        In matrix notation this can be written (X^(T) sigma^(-1) X)^(-1)
    sigma : array
        `sigma` is the n x n covariance matrix of the error terms or
        E(resid resid.T)
        where E is the expectations operator.
    wexog : array
        `wexog` is the whitened design matrix.  In matrix notation
        (cholsigmainv exog)
    wendog : array
        The whitened response /dependent variable.  In matrix notation
        (cholsigmainv endog)

    Methods
    -------
    fit
       Solves the least squares minimization.
       Note that using the model's results property is equivalent to
       calling fit.
    information
        Returns the Fisher information matrix for a given set of parameters.
        Not yet implemented
    initialize
        (Re)-initialize a model.
#TODO: should this be a public method?
    loglike
        Obtain the loglikelihood for a given set of parameters.
    newton
        Used to solve the maximum likelihood problem.
    predict
        Returns the fitted values given the parameters and exogenous design.
    score
        Score function.
    whiten
        Returns the input premultiplied by cholsigmainv

    Examples
    --------
    >>> import numpy as np
    >>> import scikits.statsmodels as models
    >>> from scikits.statsmodels.tools import add_constant
    >>> from scikits.statsmodels.datasets.longley.data import load
    >>> data = load()
    >>> data.exog = add_constant(data.exog)
    >>> ols_tmp = models.OLS(data.endog, data.exog).results
    >>> rho = np.corrcoef(ols_tmp.resid[1:],ols_tmp.resid[:-1])[0][1]

    `rho` is the correlation of the residuals from an OLS fit of the longley
    data.  It is assumed that this is the true rho of the data, and it
    will be used to estimate the structure of heteroskedasticity.

    >>> from scipy.linalg import toeplitz
    >>> order = toeplitz(np.arange(16))
    >>> sigma = rho**order

    `sigma` is an n x n matrix of the autocorrelation structure of the
    data.

    >>> gls_model = models.GLS(data.endog, data.exog, sigma=sigma)
    >>> gls_results = gls_model.results

    Notes
    -----
    If sigma is a function of the data making one of the regressors
    a constant, then the current postestimation statistics will not be correct.
    """

    def __init__(self, endog, exog, sigma=None):
        if sigma is not None:
            self.sigma = np.asarray(sigma)
        else:
            self.sigma = sigma
        if self.sigma is not None and not self.sigma.shape == (): #greedy logic
            nobs = int(endog.shape[0])
            if self.sigma.ndim == 1 or np.squeeze(self.sigma).ndim == 1:
                if self.sigma.shape[0] != nobs:
                    raise ValueError, "sigma is not the correct dimension.  \
Should be of length %s, if sigma is a 1d array" % nobs
            elif self.sigma.shape[0] != nobs and \
                    self.sigma.shape[1] != nobs:
                raise ValueError, "expected an %s x %s array for sigma" % \
                        (nobs, nobs)
        if self.sigma is not None:
            nobs = int(endog.shape[0])
            if self.sigma.shape == ():
                self.sigma = np.diag(np.ones(nobs)*self.sigma)
            if np.squeeze(self.sigma).ndim == 1:
                self.sigma = np.diag(np.squeeze(self.sigma))
            self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(\
                    self.sigma)).T
        super(GLS, self).__init__(endog, exog)

    def initialize(self):
        self.wexog = self.whiten(self._exog)
        self.wendog = self.whiten(self._endog)
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                                         np.transpose(self.pinv_wexog))
        self.df_resid = self.nobs - tools.rank(self._exog)
#       Below assumes that we have a constant
        self.df_model = float(tools.rank(self._exog)-1)

    def whiten(self, X):
        """
        GLS whiten method.

        Parameters
        -----------
        X : array-like
            Data to be whitened.

        Returns
        -------
        np.dot(cholsigmainv,X)

        See Also
        --------
        regression.GLS
        """
        X = np.asarray(X)
        if np.any(self.sigma) and not self.sigma==():
            return np.dot(self.cholsigmainv, X)
        else:
            return X

    def fit(self):
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

        Notes
        -----
        Currently it is assumed that all models will have an intercept /
        constant in the design matrix for postestimation statistics.

        The fit method uses the pseudoinverse of the design/exogenous variables
        to solve the least squares minimization.

        """
#TODO: add a full_output keyword so that only lite results needed for
# IRLS are calculated?
        beta = np.dot(self.pinv_wexog, self.wendog)
        # should this use lstsq instead?
        # worth a comparison at least...though this is readable
        lfit = RegressionResults(self, beta,
                       normalized_cov_params=self.normalized_cov_params)
        return lfit

    @property
    def results(self):
        """
        A property that returns a RegressionResults class.

        Notes
        -----
        Calls fit, if it has not already been called.
        """
        if self._results is None:
            self._results = self.fit()
        return self._results

    def predict(self, exog, params=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ---------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        #JP: this doesn't look correct for GLMAR
        #SS: it needs its own predict method
        if self._results is None and params is None:
            raise ValueError, "If the model has not been fit, then you must specify the params argument."
        if self._results is not None:
            return np.dot(exog, self.results.params)
        else:
            return np.dot(exog, params)

    def loglike(self, params):
        """
        Returns the value of the gaussian loglikelihood function at params.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        `params` : array-like
            The parameter estimates

        Returns
        -------
        The value of the loglikelihood function for a GLS Model.

        Formula
        -------
        ..math :: -\frac{n}{2}\text{\ensuremath{\log}}\left(Y-\hat{Y}\right)-\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)-\frac{1}{2}\text{log}\left(\left|\Sigma\right|\right)\]

        Notes
        -----
        Y and Y-hat are whitened.
        """
#TODO: combine this with OLS/WLS loglike and add _det_sigma argument
        nobs2 = self.nobs / 2.0
        SSR = ss(self.wendog - np.dot(self.wexog,params))
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with likelihood constant
        if np.any(self.sigma) and self.sigma.ndim == 2:
#FIXME: robust-enough check?  unneeded if _det_sigma gets defined
            llf -= .5*np.log(np.linalg.det(self.sigma))
            # with error covariance matrix
        return llf

class WLS(GLS):
    """
    A regression model with diagonal but non-identity covariance structure.

    The weights are presumed to be (proportional to) the inverse of the
    variance of the observations.  That is, if the variables are to be
    transformed by 1/sqrt(W) you must supply weights = 1/W.  Note that this
    is different than the behavior for GLS with a diagonal Sigma, where you
    would just supply W.

    Parameters
    ----------

    endog : array-like
        n length array containing the response variabl
    exog : array-like
        n x p array of design / exogenous data
    weights : array-like, optional
        1d array of weights.  If you supply 1/W then the variables are pre-
        multiplied by 1/sqrt(W).  If no weights are supplied the default value
        is 1 and WLS reults are the same as OLS.

    Attributes
    ----------
    weights : array
        The stored weights supplied as an argument.

    See regression.GLS

    Methods
    -------
    whiten
        Returns the input scaled by sqrt(W)

    See regression.GLS

    Examples
    ---------
    >>> import numpy as np
    >>> import scikits.statsmodels as models
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = models.tools.add_constant(X)
    >>> wls_model = models.WLS(Y,X, weights=range(1,8))
    >>> results = wls_model.fit()
    >>> results.params
    array([ 0.0952381 ,  2.91666667])
    >>> results.t()
    array([ 0.35684428,  2.0652652 ])
    <T test: effect=2.9166666666666674, sd=1.4122480109543243, t=2.0652651970780505, p=0.046901390323708769, df_denom=5>
    >>> print results.f_test([1,0])
    <F test: F=0.12733784321528099, p=0.735774089183, df_denom=5, df_num=1>

    Notes
    -----
    If the weights are a function of the data, then the postestimation statistics
    such as fvalue and mse_model might not be correct, as the package does not
    yet support no-constant regression.
    """
#FIXME: bug in fvalue or f_test for this example?
#UPDATE the bug is in fvalue, f_test is correct vs. R
#mse_model is calculated incorrectly according to R
#same fixed used for WLS in the tests doesn't work
#mse_resid is good
    def __init__(self, endog, exog, weights=1.):
        weights = np.array(weights)
        if weights.shape == ():
            self.weights = weights
        else:
            design_rows = exog.shape[0]
            if not(weights.shape[0] == design_rows and
                   weights.size == design_rows) :
                raise ValueError(\
                    'Weights must be scalar or same length as design')
            self.weights = weights.reshape(design_rows)
        super(WLS, self).__init__(endog, exog)

    def whiten(self, X):
        """
        Whitener for WLS model, multiplies each column by sqrt(self.weights)

        Parameters
        ----------
        X : array-like
            Data to be whitened

        Returns
        -------
        sqrt(weights)*X
        """
        X = np.asarray(X)
        if X.ndim == 1:
            return X * np.sqrt(self.weights)
        elif X.ndim == 2:
            if np.shape(self.weights) == ():
                whitened = np.sqrt(self.weights)*X
            else:
                whitened = np.sqrt(self.weights)[:,None]*X
            return whitened

    def loglike(self, params):
        """
        Returns the value of the gaussian loglikelihood function at params.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array-like
            The parameter estimates.

        Returns
        -------
        The value of the loglikelihood function for a WLS Model.

        Formula
        --------
        .. math :: -\frac{n}{2}\text{\ensuremath{\log}}\left(Y-\hat{Y}\right)-\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)-\frac{1}{2}\text{log}\left(\left|W\right|\right)\]

        W is treated as a diagonal matrix for the purposes of the formula.
        """
        nobs2 = self.nobs / 2.0
        SSR = ss(self.wendog - np.dot(self.wexog,params))
        #SSR = ss(self._endog - np.dot(self._exog,params))
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        if np.all(self.weights != 1):    #FIXME: is this a robust-enough check?
            llf -= .5*np.log(np.multiply.reduce(1/self.weights)) # with weights
        return llf

class OLS(WLS):
    """
    A simple ordinary least squares model.

    Parameters
    ----------
    endog : array-like
         1d vector of response/dependent variable
    exog: array-like
        Column ordered (observations in rows) design matrix.

    Methods
    -------
    See regression.GLS


    Attributes
    ----------
    weights : scalar
        The OLS model has an attribute weights = array(1.0).  This is due to
        its inheritance from WLS.

    See regression.GLS

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> import scikits.statsmodels as models
    >>>
    >>> Y = [1,3,4,5,2,3,4],
    >>> X = range(1,8) #[:,np.newaxis]
    >>> X = models.tools.add_constant(X)
    >>>
    >>> model = models.OLS(Y,X)
    >>> results = model.fit()
    >>> # or results = model.results
    >>> results.params
    array([ 0.25      ,  2.14285714])
    >>> results.t()
    array([ 0.98019606,  1.87867287])
    >>> print results.t_test([0,1])
    <T test: effect=2.1428571428571423, sd=1.1406228159050935, t=1.8786728732554485, p=0.059539737780605395, df_denom=5>
    >>> print results.f_test(np.identity(2))
    <F test: F=19.460784313725501, p=0.00437250591095, df_denom=5, df_num=2>

    Notes
    -----
    OLS, as the other models, assumes that the design matrix contains a constant.
    """
    def __init__(self, endog, exog=None):
        super(OLS, self).__init__(endog, exog)

    def loglike(self, params):
        '''
        The likelihood function for the clasical OLS model.

        Parameters
        ----------
        params : array-like
            The coefficients with which to estimate the loglikelihood.

        Returns
        -------
        The concentrated likelihood function evaluated at params.
        '''
        nobs2 = self.nobs/2.
        return -nobs2*np.log(2*np.pi)-nobs2*np.log(1/(2*nobs2) *\
                np.dot(np.transpose(self._endog -
                    np.dot(self._exog, params)),
                    (self._endog - np.dot(self._exog,params)))) -\
                    nobs2

    def whiten(self, Y):
        """
        OLS model whitener does nothing: returns Y.
        """
        return Y

class GLSAR(GLS):
    """
    A regression model with an AR(p) covariance structure.

    The linear autoregressive process of order p--AR(p)--is defined as:
        TODO

    Examples
    --------
    >>> import scikits.statsmodels as models
    >>> X = range(1,8)
    >>> X = models.tools.add_constant(X)
    >>> Y = [1,3,4,5,8,10,9]
    >>> model = models.GLSAR(Y, X, rho=2)
    >>> for i in range(6):
    ...    results = model.fit()
    ...    print "AR coefficients:", model.rho
    ...    rho, sigma = models.regression.yule_walker(results.resid,
    ...                 order=model.order)
    ...    model = models.GLSAR(Y, X, rho)
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.620642   -0.88654567]
    AR coefficients: [-0.61887622 -0.88137957]
    AR coefficients: [-0.61894058 -0.88152761]
    AR coefficients: [-0.61893842 -0.88152263]
    >>> results.params
    array([ 1.58747943, -0.56145497])
    >>> results.t()
    array([ 30.796394  ,  -2.66543144])
    >>> print results.t_test([0,1])
    <T test: effect=-0.56145497223945595, sd=0.21064318655324663, t=-2.6654314408481032, p=0.022296117189135045, df_denom=5>
    >>> import numpy as np
    >>> print(results.f_test(np.identity(2)))
    <F test: F=2762.4281271616205, p=2.4583312696e-08, df_denom=5, df_num=2>

    Or, equivalently

    >>> model2 = models.GLSAR(Y, X, rho=2)
    >>> res = model2.iterative_fit(maxiter=6)
    >>> model2.rho
    array([-0.61893842, -0.88152263])

    Notes
    -----
    GLSAR is considered to be experimental for now, since it has not been
    adequately tested.
    """
    def __init__(self, endog, exog=None, rho=1):
        if isinstance(rho, np.int):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0,1]:
                raise ValueError, "AR parameters must be a scalar or a vector"
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        if exog is None:
            #JP this looks wrong, should be a regression on constant
            #results for rho estimate now identical to yule-walker on y
            #super(AR, self).__init__(endog, add_constant(endog))
            super(GLSAR, self).__init__(endog, np.ones((endog.shape[0],1)))
        else:
            super(GLSAR, self).__init__(endog, exog)

    def iterative_fit(self, maxiter=3):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated simultaneously.

        Parameters
        -----------
        endog : array-like

        exog : array-like, optional

        maxiter : integer, optional
            the number of iterations
        """
#TODO: update this after going through example.
        for i in range(maxiter-1):
            self.initialize()
            results = self.fit()
            self.rho, _ = yule_walker(results.resid,
                                      order=self.order, df=None)
                                        #note that the X passed is different for
                                      #univariate.  Why this X anyway?
        self._results = self.fit() #final estimate
        return self._results # add missing return

    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure.

        Parameters:
        -----------
        X : array-like
            The data to be whitened

        Returns
        -------
        TODO
        """
#TODO: notation for AR process
        X = np.asarray(X, np.float64)
        _X = X.copy()
        for i in range(self.order):
            _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
        return _X

def yule_walker(X, order=1, method="unbiased", df=None, inv=False):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    Parameters:
    -----------
    X : array-like
        1d array
    order : integer, optional
        The order of the autoregressive process.  Default is 1.
    method : string, optional
       Method can be "unbiased" or "mle" and this determines
       denominator in estimate of autocorrelation function (ACF)
       at lag k. If "mle", the denominator is n=X.shape[0], if
       "unbiased" the denominator is n-k.
       The default is unbiased.
    df : integer, optional
       Specifies the degrees of freedom. If df is supplied,
       then it is assumed the X has df degrees of
       freedom rather than n.  Default is None.
    inv : Bool
        If inv is True the inverse of R is also returned.
        The default is False.

    Returns
    -------
    rho
        The autoregressive coefficients
    sigma
        TODO

    Examples
    --------
    >>> import scikits.statsmodels as models
    >>> from scikits.statsmodels.datasets.sunspots.data import load
    >>> data = load()
    >>> rho, sigma = models.regression.yule_walker(data.endog,
        order=4, method="mle")
    >>> rho
    array([ 1.28310031, -0.45240924, -0.20770299,  0.04794365])
    >>> sigma
    16.808022730464351

    """
#TODO: define R better, look back at notes and technical notes on YW.
#First link here is useful
#http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YuleWalkerAndMore.htm
    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError, "ACF estimation method must be 'unbiased' \
        or 'MLE'"
    X = np.array(X)
    X -= X.mean()                  # automatically demean's X
    n = df or X.shape[0]

    if method == "unbiased":        # this is df_resid ie., n - p
        denom = lambda k: n - k
    else:
        denom = lambda k: n
    if len(X.shape) != 1:
        raise ValueError, "expecting a vector to estimate AR parameters"
    r = np.zeros(order+1, np.float64)
    r[0] = (X**2).sum() / denom(0)
    for k in range(1,order+1):
        r[k] = (X[0:-k]*X[k:]).sum() / denom(k)
    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:]*rho).sum()
    if inv == True:
        return rho, np.sqrt(sigmasq), np.linalg.inv(R)
    else:
        return rho, np.sqrt(sigmasq)

class RegressionResults(LikelihoodModelResults):
    """
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    Attributes
    -----------
    aic
        Aikake's information criteria
        -2 * llf + 2*(df_model+1)
    bic
        Bayes' information criteria
        -2 * llf + log(n)*(df_model+1)
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
    df_model
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
        The linear coefficients that minimize the least squares criterion.  This
        is usually called Beta for the classical linear model.
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
    stand_errors
        The standard errors of the parameter estimates
    uncentered_tss
        Uncentered sum of squares.  Sum of the squared values of the
        (whitened) endogenous response variable.
    wresid
        The residuals of the transformed/whitened regressand and regressor(s)

    Methods
    -------
    cov_params
        This is the estimated covariance matrix scaled by `scale`
        See statsmodels.model.cov_params
    conf_int
        Returns 1 - alpha % confidence intervals for the estimates
        See statsmodels.model.conf_int()
    f_test
        F test (sometimes called F contrast) returns a ContrastResults instance
        given an array of linear restrictions.
        See statsmodels.model.f_test
    norm_resid
        Returns the (whitened) residuals normalized to have unit length.
    summary
        Returns a string summarizing the fit of a linear regression model.
    t
        Returns the t-value for the (optional) given columns.  Calling t()
        without an argument returns all of the t-values.
        See statsmodels.model.t
    t_test
        T tests (sometimes called T contrast) returns a ContrastResults instance
        given a 1d array of linear restrictions.  There is not yet full support
        for multiple an array of multiple t-tests.
    """

    # For robust covariance matrix properties
    _HC0_se = None
    _HC1_se = None
    _HC2_se = None
    _HC3_se = None

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(RegressionResults, self).__init__(model, params,
                                                 normalized_cov_params,
                                                 scale)
        self._get_results()

    def _get_results(self):
        '''
        This contains the results that are the same across models
        '''
        self.fittedvalues = self.model.predict(self.model._exog, self.params)
        self.wresid = self.model.wendog - \
                self.model.predict(self.model.wexog,self.params)
        self.resid = self.model._endog - self.fittedvalues
        self.pinv_wexog = self.model.pinv_wexog # needed?
        self.scale = ss(self.wresid) / self.model.df_resid
        self.nobs = float(self.model.wexog.shape[0])
        self.df_resid = self.model.df_resid
        self.df_model = self.model.df_model

# if not hascons?
#            self.ess = np.dot(self.params,(np.dot(self.model.wexog.T,
#                self.model.wendog)))
#            self.uncentered_tss =
#            self.centered_tss = ssr + ess
#            self.ssr = ss(self.model.wendog)- self.ess
#            self.rsquared
#            self.rsquared_adj
#        else:

#        self.ess = ss(self.fittedvalues - np.mean(self.model.wendog))
        self.ssr = ss(self.wresid)
        self.centered_tss = ss(self.model.wendog - \
                np.mean(self.model.wendog))
        self.uncentered_tss = ss(self.model.wendog)
        self.ess = self.centered_tss - self.ssr
# Centered R2 for models with intercepts
# have a look in test_regression.test_wls to see
# how to compute these stats for a model without intercept,
# and when the weights are a (linear?) function of the data...
        self.rsquared = 1 - self.ssr/self.centered_tss
        self.rsquared_adj = 1 - (self.model.nobs - 1)/(self.df_resid)*\
                (1 - self.rsquared)
        self.mse_model = self.ess/self.model.df_model
        self.mse_resid = self.ssr/self.model.df_resid
        self.mse_total = self.uncentered_tss/(self.nobs)
        self.fvalue = self.mse_model/self.mse_resid
        self.f_pvalue = stats.f.sf(self.fvalue, self.model.df_model,
                self.model.df_resid)
        self.bse = np.sqrt(np.diag(self.cov_params()))
#TODO: change to stand_errors or something
        self.llf = self.model.loglike(self.params)
        self.aic = -2 * self.llf + 2*(self.model.df_model+1)
        self.bic = -2 * self.llf + np.log(self.model.nobs)*\
                (self.model.df_model+1)
        self.pvalues = stats.t.sf(np.abs(self.t()), self.model.df_resid)*2

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
            h=np.diag(np.dot(np.dot(self.model._exog, self.normalized_cov_params),
                    self.model._exog.T)) # probably could be optimized
            self.het_scale= self.resid**2/(1-h)
            self.cov_HC2 = self._HCCM(self.het_scale)
            self._HC2_se = np.sqrt(np.diag(self.cov_HC2))
        return self._HC2_se

    @property
    def HC3_se(self):
        """
        See statsmodels.RegressionResults
        """
        if self._HC3_se is None:
            h=np.diag(np.dot(np.dot(self.model._exog,
                self.normalized_cov_params),self.model._exog.T))
            # above probably could be optimized to only calc the diag
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
            raise ValueError, 'need normalized residuals to estimate standard\
 deviation'
        return self.wresid * tools.recipr(np.sqrt(self.scale))

    def summary(self, yname=None, xname=None):
        """
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
        >>> from scikits import statsmodels as models
        >>> from scikits.statsmodels.datasets.longley.data import load
        >>> data = load()
        >>> data.exog = models.tools.add_constant(data.exog)
        >>> ols_results = models.OLS(data.endog, data.exog).results
        >>> print ols_results.summary()
        ...

        Notes
        -----
        All residual statistics are calculated on whitened residuals.
        """
        if yname is None:
            yname = 'Y'
        if xname is None:
            xname = ['X.%d' %
                (i) for i in range(self.model._exog.shape[1])]
        modeltype = self.model.__class__.__name__

#FIXME: better string formatting for alignment
        import time # delay import until here?
        import os

        # local time & date
        t = time.localtime()

        # extra stats
        from postestimation import jarque_bera, omni_norm_test, durbin_watson
        llf, aic, bic = self.llf, self.aic, self.bic
        JB, JBpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_norm_test(self.wresid)

        fit_sum = os.linesep+''.join(["="]*80)+os.linesep
        fit_sum += "Dependent Variable: " + yname + os.linesep
        fit_sum += "Model: " + modeltype + os.linesep
        fit_sum += "Method: Least Squares" + os.linesep
        fit_sum += "Date: " + time.strftime("%a, %d %b %Y",t) + os.linesep
        fit_sum += "Time: " + time.strftime("%H:%M:%S",t) + os.linesep
        fit_sum += '# obs:        %5.0f' % self.nobs + os.linesep
        fit_sum += 'Df residuals: %5.0f' % self.df_resid + os.linesep
        fit_sum += 'Df model:     %5.0f' % self.df_model + os.linesep
        fit_sum += ''.join(["="]*80) + os.linesep
        fit_sum += 'variable       coefficient       std. error     \
t-statistic       prob.'+os.linesep
        fit_sum += ''.join(["="]*80) + os.linesep
        for i in range(len(xname)):
            fit_sum += '''%-10s    %#12.6g     %#12.6g      %#10.4g       \
%#5.4g''' % tuple([xname[i],self.params[i],self.bse[i],self.t()[i],
                    self.pvalues[i]]) + os.linesep
        fit_sum += ''.join(["="]*80) + os.linesep
        fit_sum += 'Models stats                         Residual stats' +\
os.linesep
        fit_sum += ''.join(["="]*80) + os.linesep
        fit_sum += 'R-squared            %-8.5g         Durbin-Watson stat  \
% -8.4g' % tuple([self.rsquared, durbin_watson(self.wresid)]) + os.linesep
        fit_sum += 'Adjusted R-squared   %-8.5g         Omnibus stat        \
% -8.4g' % tuple([self.rsquared_adj, omni]) + os.linesep
        fit_sum += 'F-statistic          %-8.5g         Prob(Omnibus stat)  \
% -8.4g' % tuple([self.fvalue, omnipv]) + os.linesep
        fit_sum += 'Prob (F-statistic)   %-8.5g         JB stat             \
% -8.4g' % tuple([self.f_pvalue, JB]) + os.linesep
        fit_sum += 'Log likelihood       %-8.5g         Prob(JB)            \
% -8.4g' % tuple([llf, JBpv]) + os.linesep
        fit_sum += 'AIC criterion        %-8.5g         Skew                \
% -8.4g' % tuple([aic, skew]) + os.linesep
        fit_sum += 'BIC criterion        %-8.4g         Kurtosis            \
% -8.4g' % tuple([bic, kurtosis]) + os.linesep
        fit_sum += ''.join(["="]*80) + os.linesep
        return fit_sum
