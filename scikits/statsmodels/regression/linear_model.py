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

import numpy as np
from scipy.linalg import norm, toeplitz, lstsq, calc_lwork
from scipy import stats
from scipy.stats.stats import ss
from scikits.statsmodels.tools.tools import (add_constant, rank,
                                             recipr, chain_dot)
from scikits.statsmodels.tools.decorators import (resettable_cache,
        cache_readonly, cache_writable)
import scikits.statsmodels.base.model as base
import scikits.statsmodels.base.wrapper as wrap

class GLS(base.LikelihoodModel):
    """
    Generalized least squares model with a general covariance structure.

    Parameters
    ----------
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
    pinv_wexog : array
        `pinv_wexog` is the p x n Moore-Penrose pseudoinverse of `wexog`.
    cholsimgainv : array
        The transpose of the Cholesky decomposition of the pseudoinverse.
    df_model : float
        p - 1, where p is the number of regressors including the intercept.
        of freedom.
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
    >>> import scikits.statsmodels.api as sm
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

    def __init__(self, endog, exog, sigma=None):
#TODO: add options igls, for iterative fgls if sigma is None
#TODO: default is sigma is none should be two-step GLS
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
        super(GLS, self).__init__(endog, exog)

    def initialize(self):
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # overwrite nobs from class Model:
        self.nobs = float(self.wexog.shape[0])
        self.df_resid = self.nobs - rank(self.exog)
#       Below assumes that we have a constant
        self.df_model = float(rank(self.exog)-1)

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

    def fit(self, method="pinv", **kwargs):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str
            Can be "pinv", "qr", or "mle".  "pinv" uses the
            Moore-Penrose pseudoinverse to solve the least squares problem.
            "svd" uses the Singular Value Decomposition.  "qr" uses the
            QR factorization.  "mle" fits the model via maximum likelihood.
            "mle" is not yet implemented.

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
        exog = self.wexog
        endog = self.wendog

        if method == "pinv":
            if ((not hasattr(self, 'pinv_wexog')) or
                (not hasattr(self, 'normalized_cov_params'))):
                self.pinv_wexog = pinv_wexog = np.linalg.pinv(self.wexog)
                self.normalized_cov_params = np.dot(pinv_wexog,
                                                 np.transpose(pinv_wexog))
            beta = np.dot(self.pinv_wexog, endog)

        elif method == "qr":
            if ((not hasattr(self, '_exog_Q')) or
                (not hasattr(self, 'normalized_cov_params'))):
                Q, R = np.linalg.qr(exog)
                self._exog_Q, self._exog_R = Q, R
                self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
            else:
                Q, R = self._exog_Q, self._exog_R

            beta = np.linalg.solve(R,np.dot(Q.T,endog))

            # no upper triangular solve routine in numpy/scipy?
        lfit = RegressionResults(self, beta,
                       normalized_cov_params=self.normalized_cov_params)
        return RegressionResultsWrapper(lfit)

    def predict(self, params, exog=None):
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
        #JP: this doesn't look correct for GLMAR
        #SS: it needs its own predict method
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

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
            The value of the loglikelihood function for a GLS Model.


        Notes
        -----
        The loglikelihood function for the normal distribution is

        .. math:: -\\frac{n}{2}\\log\\left(Y-\\hat{Y}\\right)-\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)-\\frac{1}{2}\\log\\left(\\left|\\Sigma\\right|\\right)

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

    **Methods**

    whiten
        Returns the input scaled by sqrt(W)


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



    Examples
    ---------
    >>> import numpy as np
    >>> import scikits.statsmodels.api as sm
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> wls_model = sm.WLS(Y,X, weights=range(1,8))
    >>> results = wls_model.fit()
    >>> results.params
    array([ 0.0952381 ,  2.91666667])
    >>> results.tvalues
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

        Notes
        --------
        .. math:: -\\frac{n}{2}\\log\\left(Y-\\hat{Y}\\right)-\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)-\\frac{1}{2}log\\left(\\left|W\\right|\\right)

        where :math:`W` is a diagonal matrix
        """
        nobs2 = self.nobs / 2.0
        SSR = ss(self.wendog - np.dot(self.wexog,params))
        #SSR = ss(self.endog - np.dot(self.exog,params))
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        if np.all(self.weights != 1):    #FIXME: is this a robust-enough check?
            llf -= .5*np.log(np.multiply.reduce(1/self.weights)) # with weights
        return llf

class OLS(WLS):
    """
    A simple ordinary least squares model.

    **Methods**

    inherited from regression.GLS

    Parameters
    ----------
    endog : array-like
         1d vector of response/dependent variable
    exog: array-like
        Column ordered (observations in rows) design matrix.


    Attributes
    ----------
    weights : scalar
        Has an attribute weights = array(1.0) due to inheritance from WLS.

    See regression.GLS

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> import scikits.statsmodels.api as sm
    >>>
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8) #[:,np.newaxis]
    >>> X = sm.add_constant(X)
    >>>
    >>> model = sm.OLS(Y,X)
    >>> results = model.fit()
    >>> # or results = model.results
    >>> results.params
    array([ 0.25      ,  2.14285714])
    >>> results.tvales
    array([ 0.98019606,  1.87867287])
    >>> print results.t_test([0,1])
    <T test: effect=2.1428571428571423, sd=1.1406228159050935, t=1.8786728732554485, p=0.059539737780605395, df_denom=5>
    >>> print results.f_test(np.identity(2))
    <F test: F=19.460784313725501, p=0.00437250591095, df_denom=5, df_num=2>

    Notes
    -----
    OLS, as the other models, assumes that the design matrix contains a constant.
    """
#TODO: change example to use datasets.  This was the point of datasets!
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
                np.dot(np.transpose(self.endog -
                    np.dot(self.exog, params)),
                    (self.endog - np.dot(self.exog,params)))) -\
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
    >>> import scikits.statsmodels.api as sm
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> Y = [1,3,4,5,8,10,9]
    >>> model = sm.GLSAR(Y, X, rho=2)
    >>> for i in range(6):
    ...    results = model.fit()
    ...    print "AR coefficients:", model.rho
    ...    rho, sigma = sm.regression.yule_walker(results.resid,
    ...                 order=model.order)
    ...    model = sm.GLSAR(Y, X, rho)
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.620642   -0.88654567]
    AR coefficients: [-0.61887622 -0.88137957]
    AR coefficients: [-0.61894058 -0.88152761]
    AR coefficients: [-0.61893842 -0.88152263]
    >>> results.params
    array([ 1.58747943, -0.56145497])
    >>> results.tvalues
    array([ 30.796394  ,  -2.66543144])
    >>> print results.t_test([0,1])
    <T test: effect=-0.56145497223945595, sd=0.21064318655324663, t=-2.6654314408481032, p=0.022296117189135045, df_denom=5>
    >>> import numpy as np
    >>> print(results.f_test(np.identity(2)))
    <F test: F=2762.4281271616205, p=2.4583312696e-08, df_denom=5, df_num=2>

    Or, equivalently

    >>> model2 = sm.GLSAR(Y, X, rho=2)
    >>> res = model2.iterative_fit(maxiter=6)
    >>> model2.rho
    array([-0.61893842, -0.88152263])

    Notes
    -----
    GLSAR is considered to be experimental.
    """
    def __init__(self, endog, exog=None, rho=1):
        if isinstance(rho, np.int):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0,1]:
                raise ValueError("AR parameters must be a scalar or a vector")
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
        ----------
        maxiter : integer, optional
            the number of iterations
        """
#TODO: update this after going through example.
        for i in range(maxiter-1):
            self.initialize()
            results = self.fit()
            self.rho, _ = yule_walker(results.resid,
                                      order=self.order, df=None)
        self._results = self.fit() #final estimate
        return self._results # add missing return

    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure.

        Parameters
        ----------
        X : array-like
            The data to be whitened

        Returns
        -------
        TODO
        """
#TODO: notation for AR process
        X = np.asarray(X, np.float64)
        _X = X.copy()
        #dimension handling is not DRY
        # I think previous code worked for 2d because of single index rows in np
        if X.ndim == 1:
            for i in range(self.order):
                _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
            return _X[self.order:]
        elif X.ndim == 2:
            for i in range(self.order):
                _X[(i+1):,:] = _X[(i+1):,:] - self.rho[i] * X[0:-(i+1),:]
                return _X[self.order:,:]

def yule_walker(X, order=1, method="unbiased", df=None, inv=False, demean=True):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    Parameters
    ----------
    X : array-like
        1d array
    order : integer, optional
        The order of the autoregressive process.  Default is 1.
    method : string, optional
       Method can be "unbiased" or "mle" and this determines denominator in
       estimate of autocorrelation function (ACF) at lag k. If "mle", the
       denominator is n=X.shape[0], if "unbiased" the denominator is n-k.
       The default is unbiased.
    df : integer, optional
       Specifies the degrees of freedom. If `df` is supplied, then it is assumed
       the X has `df` degrees of freedom rather than `n`.  Default is None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho
        The autoregressive coefficients
    sigma
        TODO

    Examples
    --------
    >>> import scikits.statsmodels.api as sm
    >>> from scikits.statsmodels.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma = sm.regression.yule_walker(data.endog,       \
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
        raise ValueError("ACF estimation method must be 'unbiased' or 'MLE'")
    X = np.array(X)
    if demean:
        X -= X.mean()                  # automatically demean's X
    n = df or X.shape[0]

    if method == "unbiased":        # this is df_resid ie., n - p
        denom = lambda k: n - k
    else:
        denom = lambda k: n
    if X.ndim > 1 and X.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")
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
        return np.sqrt(np.diag(self.cov_params()))

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

        Parametersnorm
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
        scikits.statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        #TODO: import where we need it (for now), add as cached attributes
        from scikits.statsmodels.stats.stattools import (jarque_bera,
                omni_normtest, durbin_watson)
        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        #TODO: reuse condno from somewhere else ?
        #condno = np.linalg.cond(np.dot(self.wexog.T, self.wexog))
        wexog = self.model.wexog
        eigvals = np.linalg.linalg.eigvalsh(np.dot(wexog.T, wexog))
        eigvals = np.sort(eigvals) #in increasing order
        condno = eigvals[-1]/eigvals[0]

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
        from scikits.statsmodels.iolib.summary import Summary
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
        >>> import scikits.statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> ols_results = sm.OLS(data.endog, data.exog).results
        >>> print ols_results.summary()
        ...

        Notes
        -----
        All residual statistics are calculated on whitened residuals.
        """
        import time
        from scikits.statsmodels.iolib.table import SimpleTable
        from scikits.statsmodels.stats.stattools import (jarque_bera,
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
        part2header = ('coefficient', 'std. error', 't-statistic', 'prob.')
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


if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    data = sm.datasets.longley.load()
    data.exog = add_constant(data.exog)
    ols_results = OLS(data.endog, data.exog).fit() #results
    gls_results = GLS(data.endog, data.exog).fit() #results
    print(ols_results.summary())
    tables = ols_results.summary(returns='tables')
    csv = ols_results.summary(returns='csv')
"""
    Summary of Regression Results
=======================================
| Dependent Variable:            ['y']|
| Model:                           OLS|
| Method:                Least Squares|
| Date:               Tue, 29 Jun 2010|
| Time:                       22:32:21|
| # obs:                          16.0|
| Df residuals:                    9.0|
| Df model:                        6.0|
===========================================================================
|            coefficient       std. error      t-statistic           prob.|
---------------------------------------------------------------------------
| x1             15.0619          84.9149           0.1774          0.8631|
| x2             -0.0358           0.0335          -1.0695          0.3127|
| x3             -2.0202           0.4884          -4.1364        0.002535|
| x4             -1.0332           0.2143          -4.8220       0.0009444|
| x5             -0.0511           0.2261          -0.2261          0.8262|
| x6           1829.1515         455.4785           4.0159        0.003037|
| const    -3482258.6346      890420.3836          -3.9108        0.003560|
===========================================================================
|                        Models stats                      Residual stats |
---------------------------------------------------------------------------
| R-squared:                 0.995479    Durbin-Watson:           2.55949 |
| Adjusted R-squared:        0.992465    Omnibus:                0.748615 |
| F-statistic:                330.285    Prob(Omnibus):          0.687765 |
| Prob (F-statistic):     4.98403e-10    JB:                     0.352773 |
| Log likelihood:            -109.617    Prob(JB):               0.838294 |
| AIC criterion:              233.235    Skew:                   0.419984 |
| BIC criterion:              238.643    Kurtosis:                2.43373 |
---------------------------------------------------------------------------
"""
