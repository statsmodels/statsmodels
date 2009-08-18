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

'Introduction to Linear Regression Analysis', Douglas C. Montgomery,
    Elizabeth A. Peck, G. Geoffrey Vining. Wiley, 2006.

Econometrics references for regression models:

R. Davidson and J.G. MacKinnon.  "Econometric Theory and Methods," Oxford,
    2004.

W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.



"""

__docformat__ = 'restructuredtext en'

__all__ = ['GLS', 'WLS', 'OLS', 'AR']

from string import join as sjoin    #
from csv import reader              # These are for read_array

import numpy as np
from scipy.linalg import norm, toeplitz
from models.model import LikelihoodModel, LikelihoodModelResults
from models import tools
from models.tools import add_constant
from scipy import stats
from scipy.stats.stats import ss

class GLS(LikelihoodModel):
    """
    Generalized least squares model with a general covariance structure.

    Parameters
    ----------
    endog : array-like
        `endog` is a 1-d vector that contains the response/independent variable

    exog : array-like
        `exog` is a nobs x p vector where nobs is the number of observations and
        p is the number of regressors/dependent variables including the intercept
        if one is included in the data.

    sigma : scalar or array
       `sigma` is the weighting matrix of the covariance.
       The default is None for no scaling.  If `sigma` is a scalar, it is
       assumed that `sigma` is an n x n diagonal matrix with the given sclar,
       `sigma` as the value of each diagonal element.  If `sigma` is an
       n-length vector, then `sigma` is assumed to be a diagonal matrix
       with the given `sigma` on the diagonal.  This should be the same as WLS
??? is it the same was WLS?

       See the docs/gls.rst for more info.

    Attributes
    ----------
    calc_params : array
        `calc_params` is the p x n Moore-Penrose pseudoinverse
        of the design matrix. In matrix notation it is approximately
        (X^(T)X)^(-1)X^(T)

    df_model : scalar
        The model degrees of freedom is equal to p - 1, where p is the number
        of regressors.  Note that the intercept is not included.

    df_resid : scalar
        The residual degrees of freedom is equal to the number of observations
        less the number of parameters.  Note that the intercept is counted
        as a regressor when calculating the degrees of freedom of the
        residuals.

    llf : float
        `llf` is the value of the maximum likelihood function of the model.

    normalized_cov_params : array
        `normalized_cov_params` is a p x p array that is the inverse of ...
        In matrix notation this can be written (X^(T)X)^(-1)

    sigma :

    wdesign : array
        `wdesign` is the whitened design matrix.  If sigma is not a scalar
        this is np.dot(cholsigmainv,Y).  Where cholsigmainv is the lower-
        triangular Cholesky factor of the transpose of the pseudoinverse of
        sigma.  In matrix notation this can be written L^(T)X where LL^(T)
        is approximately Sigma^(+), the pseudoinverse of Sigma.
#FIXME: Check explanation, try to write in terms of matrices and inverses
#        only rather than pseudoinverses and factors.

    Methods
    -------
    fit
       Solves the least squares minimization.

    information
        Returns the Fisher information matrix.

    initialize
        (Re)-initialize a model.

    newton
        Used to solve the maximum likelihood problem.
        Not currently implemented.

    fittedvalues
        Returns the fitted values given the parameters and exogenous design.

    score
        Score function.

    whiten
        TODO

    Examples
    --------
    >>>import numpy as np
    >>>import models
    >>>from models.tools import add_constant
    >>>from models.datasets.longley.data import load
    >>>data = load()
    >>>data.exog = add_constant(data.exog)
    >>>ols_tmp = models.OLS(data.endog, data.exog).results
    >>>rho = np.corrcoef(ols_tmp.resid[1:],ols_model.resid[:-1])[0][1]

    `rho` is the correlation of the residuals from an OLS fit of the longley
    data.  It is assumed that this is the true rho of the data, and it
    will be used to estimate the structure of heteroskedasticity.

    >>>from scipy.linalg import toeplitz
    >>>order = toeplitz(np.arange(16))
    >>>sigma = rho**order

    `sigma` is an n x n matrix of the autocorrelation structure of the
    data.

    >>>gls_model = models.GLS(data.endog, data.exog, sigma=sigma)
    >>>gls_results = gls_model.results

    *Assume* that the error terms in the OLS fit above are uncorrelated.
    Then sigma will be a diagonal matrix.

    >>>s = np.var(ols_tmp

    >>>

    """
    def __init__(self, endog, exog, sigma=None):
#TODO: make sure checks work
#TODO: are these assumptions about what to do with sigma correct
# they results in difference from WLS
# ie., wresid, etc. when they should be the same?
# why should they be the same is the whitener is different?
# do the math...
# wresid should be the same between GLS and WLS with the same weights?
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
        self.wdesign = self.whiten(self._exog)
        #JP: calc_params is not an informative name, but anything better?
        #SS: gen_inv?  it's the generalized inverse
        # ie., beta = calc_params dot y
        self.calc_params = np.linalg.pinv(self.wdesign)
        self.normalized_cov_params = np.dot(self.calc_params,
                                         np.transpose(self.calc_params))
        self.df_resid = self.nobs - tools.rank(self._exog)
#       Below assumes that we will always have a constant
        self.df_model = tools.rank(self._exog)-1

    def whiten(self, Y):
        if np.any(self.sigma) and not self.sigma==():
            return np.dot(self.cholsigmainv, Y)
        else:
            return Y

    def fit(self):
        """
        Full fit of the model including estimate of covariance matrix,
        (whitened) residuals and scale.

        Returns
        -------
        adjRsq
            Adjusted R-squared
        bse
            The standard errors of the parameter estimates
        cTSS
            The total sum of squares centered about the mean
        df_resid
            Residual degrees of freedom
        df_model
            Model degress of freedom
        ESS
            Explained sum of squares
        F
            F-statistic
        F_p
            F-statistic p-value
        MSE_model
            Mean squared error the model
        MSE_resid
            Mean squared error of the residuals
        MSE_total
            Total mean squared error
        fittedvalues
            The predict the values for a given design
        resid
            The residuals of the model.
        Rsq
            R-squared of a model with an intercept
        scale
            A scale factor for the covariance matrix.
            Default value is SSR/(n-k)
            Otherwise, determined by the `robust` keyword argument
        SSR
            Sum of squared residuals
        uTSS
            Uncentered sum of squares
        Z
            The whitened response variable
        """
        wendog = self.whiten(self._endog)
# shouldn't actual whitened endog be
#       wendog = np.dot(np.linalg.inv(self.sigma),self._endog)
# or equivalently
#       wendog = np.dot(np.dot(self.cholsigmainv.T,self.cholsigmainv),self._endog)
        beta = np.dot(self.calc_params, wendog)
        # should this use lstsq instead?
        lfit = RegressionResults(self, beta,
                       normalized_cov_params=self.normalized_cov_params)
        lfit.fittedvalues = self.predict(self._exog, beta)
# D&M says that WLS and GLS postestimation stats should be
# on transformed data, but this screws up RLM for now?
        lfit.wresid = wendog - self.predict(self.wdesign, lfit.params)
        lfit.resid = self._endog - lfit.fittedvalues
        lfit.wendog = wendog   # not a good name wendog analogy to wdesign
        lfit.df_resid = self.df_resid
        lfit.df_model = self.df_model
        lfit.calc_params = self.calc_params
        self._summary(lfit)
        return lfit

    @property
    def results(self):
        if self._results is None:
            self._results = self.fit()
        return self._results

    def _summary(self, lfit):
        """
        Private method to call additional statistics for GLS.
        Meant to be overwritten by subclass as needed(?).
        """
        lfit.scale = ss(lfit.wresid) / self.df_resid
        lfit.nobs = float(self.wdesign.shape[0])
        lfit.SSR = ss(lfit.wresid)
        lfit.cTSS = ss(lfit.wendog-np.mean(lfit.wendog))
#TODO: Z or Y here?  Need to have tests in GLS.
#JP what does c and u in front of TSS stand for?
#c is centered and u is uncentered
#JP I think, it should be Y instead of Z, are the following results correct, with Z?
#TODO: more robust tests for WLS or GLS, to see if Y or Z is used.
# I think Y as well, but Z = Y for OLS

        lfit.uTSS = ss(lfit.wendog)
# Centered R2 for models with intercepts
        lfit.Rsq = 1 - lfit.SSR/lfit.cTSS
        lfit.ESS = ss(lfit.fittedvalues - np.mean(lfit.wendog))
        lfit.SSR = ss(lfit.resid)
        lfit.adjRsq = 1 - (lfit.nobs - 1)/(lfit.nobs - (lfit.df_model+1))\
                *(1 - lfit.Rsq)
        lfit.MSE_model = lfit.ESS/lfit.df_model
        lfit.MSE_resid = lfit.SSR/lfit.df_resid
        lfit.MSE_total = lfit.uTSS/(lfit.df_model+lfit.df_resid)
        lfit.F = lfit.MSE_model/lfit.MSE_resid
        lfit.F_p = 1 - stats.f.cdf(lfit.F, lfit.df_model, lfit.df_resid)
        lfit.bse = np.sqrt(np.diag(lfit.cov_params()))
        lfit.llf = self.loglike(lfit.params)
        lfit.aic = -2 * lfit.llf + 2*(self.df_model+1)
        lfit.bic = -2 * lfit.llf + np.log(lfit.nobs)*(self.df_model+1)
        lfit.pvalues = stats.t.sf(np.abs(lfit.t()), lfit.df_resid)
        lfit.PostEstimation = PostRegression(lfit)

    def predict(self, design, params=None):
        """
        Return linear predicted values from a design matrix.
        """
        #JP: this doesn't look correct for GLMAR
        #SS: it needs its own predict method
        if self._results is None and params is None:
            raise ValueError, "If the model has not been fit, then you must specify the params argument."
        if self._results is not None:
            return np.dot(design, self.results.params)
        else:
            return np.dot(design, params)

    def loglike(self, params):
        """
        Returns the value of the gaussian loglikelihood function at b.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        `params` : array-like
            The parameter estimates.

        Returns
        -------
        The value of the loglikelihood function for an OLS Model.

        Notes
        -----
        The Likelihood Function is
        .. math:: \ell(\boldsymbol{y},\hat{\beta},\hat{\sigma})=
        -\frac{n}{2}(1+\log2\pi-\log n)-\frac{n}{2}\log\text{SSR}(\hat{\beta})

        The AIC is
        .. math:: \text{AIC}=\log\frac{SSR}{n}+\frac{2K}{n}

        The BIC (or Schwartz Criterion) is
        .. math:: \text{BIC}=\log\frac{SSR}{n}+\frac{K}{n}\log n

        """
        nobs2 = self.nobs / 2.0
        SSR = ss(self._endog - np.dot(self._exog,params))
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        return llf


class WLS(GLS):
    """
    A regression model with diagonal but non-identity covariance structure.
    The weights are presumed to be (proportional to the) inverse of the
    variance of the observations.

    Parameters
    ----------

    Methods
    -------

    Attributes
    ----------

    Examples
    ---------
    >>>import numpy as np
    >>>import models
    >>>import numpy as np
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = models.tools.add_constant(X)
    >>> model1 = models.WLS(Y,X, weights=range(1,8))
    >>> results = model1.fit()
    >>> results.params
    array([ 0.0952381 ,  2.91666667])
    >>> results.t()
    array([ 0.61890419,  3.58195812])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=2.9166666666666674, sd=0.81426598880777334, t=3.5819581153538946, p=0.0079210215745977308, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=81.213965087281849, p=0.00015411866558, df_denom=5, df_num=2>    """

    def __init__(self, endog, exog, weights=1):
        weights = np.array(weights)
        if weights.shape == ():
            self.weights = weights
        else:
            design_rows = exog.shape[0]
            if not(weights.shape[0] == design_rows and
                   weights.size == design_rows) :
                raise ValueError(
                    'Weights must be scalar or same length as design')
            self.weights = weights.reshape(design_rows)
        super(WLS, self).__init__(endog, exog)

    def whiten(self, X):
        """
        Whitener for WLS model, multiplies by sqrt(self.weights)
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

class OLS(WLS):
    """
    A simple ordinary least squares model.

    Parameters
    ----------
        `endog` : array-like
            1d vector of response/dependent variable

        `exog`: array-like
            Column ordered (observations in rows) design matrix.

    Methods
    -------

    Attributes
    ----------
    design : ndarray
        This is the design, or X, matrix.
    wdesign : ndarray
        This is the whitened design matrix.
        design = wdesign by default for the OLS, though models that
        inherit from the OLS will whiten the design.
    calc_params : ndarray
        This is the Moore-Penrose pseudoinverse of the whitened design matrix.
    normalized_cov_params : ndarray
        np.dot(calc_params, calc_params.T)
    df_resid : integer
        Degrees of freedom of the residuals.
        Number of observations less the rank of the design.
    df_model : integer
        Degres of freedome of the model.
        The rank of the design.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from models.tools import add_constant
    >>> from models.regression import OLS
    >>>
    >>> Y = [1,3,4,5,2,3,4],
    >>> X = range(1,8)
    >>> X = add_constant(X)
    >>>
    >>> model = OLS(Y,X)
    >>> results = model.fit()
    >>>
    >>> results.params
    array([ 0.25      ,  2.14285714])
    >>> results.t()
    array([ 0.98019606,  1.87867287])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=2.14285714286, sd=1.14062281591, t=1.87867287326, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=19.4607843137, df_denom=5, df_num=2>
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

class AR(GLS):
    """
    A regression model with an AR(p) covariance structure.

    The linear autoregressive process of order p--AR(p)--is defined as:
        TODO

    Examples
    --------
    >>> import numpy as N
    >>> import numpy.random as R
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import AR
    >>>
    >>> data={'Y':[1,3,4,5,8,10,9],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = AR(f.design(), 2)
    >>> for i in range(6):
    ...     results = model.fit(data['Y'])
    ...     print "AR coefficients:", model.rho
    ...     rho, sigma = model.yule_walker(data["Y"] - results.fittedvalues)
    ...     model = AR(model.design, rho)
    ...
### NOTE ### the above call to yule_walker needs an order = model.order
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
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=-0.561454972239, sd=0.210643186553, t=-2.66543144085, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=2762.42812716, df_denom=5, df_num=2>
    >>>
    >>> model.rho = np.array([0,0])
    >>> model.iterative_fit(data['Y'], niter=3)
    >>> print model.rho
    [-0.61887622 -0.88137957]

    New Example
    --------
    import numpy as np
    from models.tools import add_constant
    from models.regression import AR, yule_walker

    X = np.arange(1,8)
    X = add_constant(X)
    Y = np.array((1, 3, 4, 5, 8, 10, 9))
    model = AR(Y, X, rho=2)
    for i in range(6):
        results = model.fit()
        print "AR coefficients:", model.rho
        rho, sigma = yule_walker(results.resid, order = model.order)
        model = AR(Y, X, rho)
    results.params
    results.t() # is this correct? it does equal params/bse
    print results.Tcontrast([0,1])  # are sd and t correct? vs
    print results.Fcontrast(np.eye(2))

    #equivalently
    model2 = AR(Y, X, rho=2)
    model2.iterative_fit(maxiter=6)
    model2.rho
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
            super(AR, self).__init__(endog, np.ones((endog.shape[0],1)))
        else:
            super(AR, self).__init__(endog, exog)

    def iterative_fit(self, maxiter=3):
        """
        Perform an iterative two-stage procedure to estimate AR(p)
        parameters and regression coefficients simultaneously.

        :Parameters:
            Y : TODO
                TODO
            niter : ``integer``
                the number of iterations
        """
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

        :Parameters:
            X : TODO
                TODO
        """
        X = np.asarray(X, np.float64)
        _X = X.copy()
        for i in range(self.order):
            _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
        return _X

def yule_walker(X, order=1, method="unbiased", df=None, inv=False):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    :Parameters:
        X : a 1d ndarray
        method : ``string``
               Method can be "unbiased" or "mle" and this determines
               denominator in estimate of autocorrelation function (ACF)
               at lag k. If "mle", the denominator is n=r.shape[0], if
               "unbiased" the denominator is n-k.
        df : ``integer``
               Specifies the degrees of freedom. If df is supplied,
               then it is assumed the X has df degrees of
               freedom rather than n.
    """

    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError, "ACF estimation method must be 'unbiased' \
        or 'MLE'"
    X = np.array(X, np.float64)  # don't touch the data? JP fixed, this modified data
    X -= X.mean()                  # automatically demean's X
    n = df or X.shape[0]    # is df_resid the degrees of freedom?
                            # no it's n I think or n-1

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
    """
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(RegressionResults, self).__init__(model, params,
                                                 normalized_cov_params,
                                                 scale)
#TODO: this needs a test
    def norm_resid(self):
        """
        Residuals, normalized to have unit length.

        Note: residuals are whitened residuals.

        Notes
        -----
        Is this supposed to return "stanardized residuals," residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i/sqrt(MS_E)

        Where MS_E = SSE/(n - k)

        See: Montgomery and Peck 3.2.1 p. 68
             Davidson and MacKinnon 15.2 p 662

        """
        if not hasattr(self, 'resid'):
            raise ValueError, 'need normalized residuals to estimate standard deviation'
        return self.resid * tools.recipr(np.sqrt(self.scale))

#TODO: these need to be tested
class PostRegression(object):
    def __init__(self, results):
        self.results = results
        self.y_varnm = 'Y'
        self.x_varnm = ['X.%d' %
                (i) for i in range(self.results.model._exog.shape[1])]
        self.modeltype = \
                str(self.results.model.__class__).split(' ')[1].strip('\'>')

    def durbin_watson(self):
        """
        Calculates the Durbin-Waston statistic
        """
        diff_resids = np.diff(self.results.resid,1)
        dw = np.dot(diff_resids,diff_resids) / \
                np.dot(self.results.resid,self.results.resid);
        return dw

    def omni_norm_test(self):
        """
        Omnibus test for normality
        """
        return stats.normaltest(self.results.resid)

    def jarque_bera(self):
        """
        Calculate residual skewness, kurtosis, and do the JB test for normality
        """

        # Calculate residual skewness and kurtosis
        skew = stats.skew(self.results.resid)
        kurtosis = 3 + stats.kurtosis(self.results.resid)

        # Calculate the Jarque-Bera test for normality
        JB = (self.results.nobs/6) * (np.square(skew) + (1/4)*np.square(kurtosis-3))
        JBpv = 1-stats.chi2.cdf(JB,2);

        return JB, JBpv, skew, kurtosis

#FIXME: print sig digits for better alignment
    def summary(self):
        """
        Printing model output to screen
        """

        import time # delay import until here?
        import os

        # local time & date
        t = time.localtime()

        # extra stats
        llf, aic, bic = self.results.llf, self.results.aic, self.results.bic
        JB, JBpv, skew, kurtosis = self.jarque_bera()
        omni, omnipv = self.omni_norm_test()


        fit_sum = os.linesep+'=============================================================================='+os.linesep
        fit_sum += "Dependent Variable: " + self.y_varnm + os.linesep
        fit_sum += "Model: " + self.modeltype + os.linesep
        fit_sum += "Method: Least Squares" + os.linesep
        fit_sum += "Date: " + time.strftime("%a, %d %b %Y",t) + os.linesep
        fit_sum += "Time: " + time.strftime("%H:%M:%S",t) + os.linesep
        fit_sum += '# obs:        %5.0f' % self.results.nobs + os.linesep
        fit_sum += 'Df residuals: %5.0f' % self.results.df_resid + os.linesep
        fit_sum += 'Df model:     %5.0f' % self.results.df_model + os.linesep
        fit_sum += '=============================================================================='+os.linesep
        fit_sum += 'variable     coefficient     std. Error      t-statistic     prob.'+\
                os.linesep
        fit_sum += '==============================================================================' + os.linesep
        for i in range(len(self.x_varnm)):
            fit_sum += '''% -5s          % -5.6f     % -5.6f     % -5.6f     % -5.6f''' % tuple([self.x_varnm[i],self.results.params[i],self.results.bse[i],self.results.t()[i],self.results.pvalues[i]]) + os.linesep
        fit_sum += '==============================================================================' + os.linesep
        fit_sum += 'Models stats                         Residual stats' + os.linesep
        fit_sum += '==============================================================================' + os.linesep
        fit_sum += 'R-squared            % -5.6f         Durbin-Watson stat  % -5.6f' % tuple([self.results.Rsq, self.durbin_watson()]) + os.linesep
        fit_sum += 'Adjusted R-squared   % -5.6f         Omnibus stat        % -5.6f' % tuple([self.results.adjRsq, omni]) + os.linesep
        fit_sum += 'F-statistic          % -5.6f         Prob(Omnibus stat)  % -5.6f' % tuple([self.results.F, omnipv]) + os.linesep
        fit_sum += 'Prob (F-statistic)   % -5.6f         JB stat             % -5.6f' % tuple([self.results.F_p, JB]) + os.linesep
        fit_sum += 'Log likelihood       % -5.6f         Prob(JB)            % -5.6f' % tuple([llf, JBpv]) + os.linesep
        fit_sum += 'AIC criterion        % -5.6f         Skew                % -5.6f' % tuple([aic, skew]) + os.linesep
        fit_sum += 'BIC criterion        % -5.6f         Kurtosis            % -5.6f' % tuple([bic, kurtosis]) + os.linesep
        fit_sum += '==============================================================================' + os.linesep
        return fit_sum

### The below is replicated by np.io

def read_design(desfile, delimiter=',', try_integer=True):
    """
    Return a record array with the design.
    The columns are first cast as numpy.float, if this fails, its
    dtype is unchanged.

    If try_integer is True and a given column can be cast as float,
    it is then tested to see if it can be cast as numpy.int.

    >>> design = [["id","age","gender"],[1,23.5,"male"],[2,24.1,"female"],[3,24.5,"male"]]
    >>> read_design(design)
    recarray([(1, 23.5, 'male'), (2, 24.100000000000001, 'female'),
    (3, 24.5, 'male')],
    dtype=[('id', '<i4'), ('age', '<f8'), ('gender', '|S6')])
    >>> design = [["id","age","gender"],[1,23.5,"male"],[2,24.1,"female"],[3.,24.5,"male"]]
    >>> read_design(design)
    recarray([(1, 23.5, 'male'), (2, 24.100000000000001, 'female'),
    (3, 24.5, 'male')],
    dtype=[('id', '<i4'), ('age', '<f8'), ('gender', '|S6')])
    >>> read_design(design, try_integer=False)
    recarray([(1.0, 23.5, 'male'), (2.0, 24.100000000000001, 'female'),
    (3.0, 24.5, 'male')],http://soccernet.espn.go.com/news/story?id=655585&sec=global&cc=5901
    dtype=[('id', '<f8'), ('age', '<f8'), ('gender', '|S6')])
    >>>

    Notes
    -----
    This replicates np.recfromcsv pretty closely.  The only difference I can
    can see is the try_integer will cast integers to floats.  np.io should
    be preferred especially if we import division from __future__.
    """

    if type(desfile) == type("string"):
        desfile = file(desfile)
        _reader = reader(desfile, delimiter=delimiter)
    else:
        _reader = iter(desfile)
    colnames = _reader.next()
    predesign = np.rec.fromrecords([row for row in _reader], names=colnames)

    # Try to typecast each column to float, then int

    dtypes = predesign.dtype.descr
    newdescr = []
    newdata = []
    for name, descr in dtypes:
        x = predesign[name]
        try:
            y = np.asarray(x.copy(), np.float) # cast as float
            if np.alltrue(np.equal(x, y)):
                if try_integer:
                    z = y.astype(np.int) # cast as int
                    if np.alltrue(np.equal(y, z)):
                        newdata.append(z)
                        newdescr.append(z.dtype.descr[0][1])
                    else:
                        newdata.append(y)
                        newdescr.append(y.dtype.descr[0][1])
                else:
                    newdata.append(y)
                    newdescr.append(y.dtype.descr[0][1])
        except:
            newdata.append(x)
            newdescr.append(descr)

    return np.rec.fromarrays(newdata, formats=sjoin(newdescr, ','), names=colnames)
