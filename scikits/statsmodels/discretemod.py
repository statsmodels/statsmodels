"""
Limited dependent variable and qualitative variables.

Includes binary outcomes, count data, (ordered) ordinal data and limited
dependent variables.

General References
--------------------

A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.  Cambridge,
    1998

G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
    Cambridge, 1983.

W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
"""

__all__ = ["Poisson","Logit","Probit","MNLogit"]

import numpy as np
from model import LikelihoodModel, LikelihoodModelResults
import tools
from decorators import *
from regression import OLS
from scipy import stats, factorial, special, optimize # opt just for nbin
#import numdifftools as nd #This will be removed when all have analytic hessians

#TODO: add options for the parameter covariance/variance
# ie., OIM, EIM, and BHHH see Green 21.4

def _isdummy(X):
    """
    Given an array X, returns a boolean column index for the dummy variables.

    Parameters
    ----------
    X : array-like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 2, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _isdummy(X)
    >>> ind
    array([ True, False, False,  True,  True], dtype=bool)
    """
    X = np.asarray(X)
    if X.ndim > 1:
        ind = np.zeros(X.shape[1]).astype(bool)
    max = (np.max(X, axis=0) == 1)
    min = (np.min(X, axis=0) == 0)
    remainder = np.all(X % 1. == 0, axis=0)
    ind = min & max & remainder
    if X.ndim == 1:
        ind = np.asarray([ind])
    return ind

def _iscount(X):
    """
    Given an array X, returns a boolean column index for count variables.

    Parameters
    ----------
    X : array-like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 10, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _iscount(X)
    >>> ind
    array([ True, False, False,  True,  True], dtype=bool)
    """
    X = np.asarray(X)
    remainder = np.all(X % 1. == 0, axis = 0)
    dummy = _isdummy(X)
    remainder -= dummy
    return remainder

class DiscreteModel(LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    scikits.statsmodels.model.LikelihoodModel.
    """
    def __init___(endog, exog):
        super(DiscreteModel, self).__init__(endog, exog)

    def initialize(self):
        """
        Initialize is called by
        scikits.statsmodels.model.LikelihoodModel.__init__
        and should contain any preprocessing that needs to be done for a model.
        """
        self.df_model = float(tools.rank(self.exog) - 1) # assumes constant
        self.df_resid = float(self.exog.shape[0] - tools.rank(self.exog))

    def cdf(self, X):
        """
        The cumulative distribution function of the model.
        """
        raise NotImplementedError

    def pdf(self, X):
        """
        The probability density (mass) function of the model.
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=35, full_output=0,
            disp=1, callback=None, **kwargs):
        if start_params is None and isinstance(self, MNLogit):
            start_params = np.zeros((self.exog.shape[1]*\
                    (self.wendog.shape[1]-1)))
        mlefit = super(DiscreteModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        if isinstance(self, MNLogit):
            mlefit.params = mlefit.params.reshape(-1, self.exog.shape[1])
        discretefit = DiscreteResults(self, mlefit)
        return discretefit

class Poisson(DiscreteModel):
    """
    Poisson model for count data

    Parameters
    ----------
    endog : array-like
        1-d array of the response variable.
    exog : array-like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is included
        in the data.

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    nobs : float
        The number of observations of the model.

    Methods
    -------
    cdf
    fit
    hessian
    information
    initialize
    loglike
    pdf
    predict
    score
    """

    def cdf(self, X):
        """
        Poisson model cumulative distribution function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the model.  See notes.

        Returns
        -------
        The value of the Poisson CDF at each point.

        Notes
        -----
        The CDF is defined as

        .. math:: \\exp\left(-\\lambda\\right)\\sum_{i=0}^{y}\\frac{\\lambda^{i}}{i!}

        where :math:`\\lambda` assumes the loglinear model. I.e.,

        .. math:: \\ln\\lambda_{i}=X\\beta

        The parameter `X` is :math:`X\\beta` in the above formula.
        """
        y = self.endog
#        xb = np.dot(self.exog, params)
        return stats.poisson.cdf(y, np.exp(X))

    def pdf(self, X):
        """
        Poisson model probability mass function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the model.  See notes.

        Returns
        -------
        The value of the Poisson PMF at each point.

        Notes
        --------
        The PMF is defined as

        .. math:: \\frac{e^{-\\lambda_{i}}\\lambda_{i}^{y_{i}}}{y_{i}!}

        where :math:`\\lambda` assumes the loglinear model. I.e.,

        .. math:: \\ln\\lambda_{i}=X\\beta

        The parameter `X` is :math:`X\\beta` in the above formula.
        """
        y = self.endog
#        xb = np.dot(self.exog,params)
        return stats.poisson.pmf(y, np.exp(X))

    def loglike(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        --------
        .. math :: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        return np.sum(-np.exp(XB) +  endog*XB - np.log(factorial(endog)))

    def score(self, params):
        """
        Poisson model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        The score vector of the model evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\lambda_{i}\\right)x_{i}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=X\\beta
        """

        X = self.exog
        L = np.exp(np.dot(X,params))
        return np.dot(self.endog - L,X)

    def hessian(self, params):
        """
        Poisson model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        The Hessian matrix evaluated at params

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i=1}^{n}\\lambda_{i}x_{i}x_{i}^{\\prime}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=X\\beta

        """
        X = self.exog
        L = np.exp(np.dot(X,params))
        return -np.dot(L*X.T, X)

#    def fit(self, start_params=None, maxiter=35, method='newton',
#            tol=1e-08):
#        """
#        Fits the Poisson model.
#
#        Parameters
#        ----------
#        start_params : array-like, optional
#            The default is a 0 vector.
#        maxiter : int, optional
#            Maximum number of iterations.  The default is 35.
#        method : str, optional
#            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
#        tol : float, optional
#            The convergence tolerance for the solver.  The default is
#            1e-08.
#
#        Returns
#        --------
#        DiscreteResults object
#
#        See also
#        --------
#        scikits.statsmodels.model.LikelihoodModel
#        scikits.statsmodels.sandbox.discretemod.DiscreteResults
#        scipy.optimize
#        """
#
#        mlefit = super(Poisson, self).fit(start_params=start_params,
#            maxiter=maxiter, method=method, tol=tol)
#        params = mlefit.params
#        mlefit = DiscreteResults(self, params, self.hessian(params))
#        return mlefit

class NbReg(DiscreteModel):
    pass

class Logit(DiscreteModel):
    """
    Binary choice logit model

    Parameters
    ----------
    endog : array-like
        1-d array of the response variable.
    exog : array-like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is included
        in the data.

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    nobs : float
        The number of observations of the model.

    Methods
    --------
    cdf
    fit
    hessian
    information
    initialize
    loglike
    pdf
    predict
    score
    """

    def cdf(self, X):
        """
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        1/(1 + exp(-X))

        Notes
        ------
        In the logit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta\\right)=\\text{Prob}\\left(Y=1|x\\right)=\\frac{e^{x^{\\prime}\\beta}}{1+e^{x^{\\prime}\\beta}}
        """
        X = np.asarray(X)
        return 1/(1+np.exp(-X))

    def pdf(self, X):
        """
        The logistic probability density function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        np.exp(-x)/(1+np.exp(-X))**2

        Notes
        -----
        In the logit model,

        .. math:: \\lambda\\left(x^{\\prime}\\beta\\right)=\\frac{e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{2}}
        """
        X = np.asarray(X)
        return np.exp(-X)/(1+np.exp(-X))**2

    def loglike(self, params):
        """
        Log-likelihood of logit model.

        Parameters
        -----------
        params : array-like
            The parameters of the logit model.

        Returns
        -------
        The log-likelihood function of the logit model.  See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        q = 2*self.endog - 1
        X = self.exog
        return np.sum(np.log(self.cdf(q*np.dot(X,params))))

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        The score vector of the model evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """

        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return np.dot(y - L,X)

    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        The Hessian evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return -np.dot(L*(1-L)*X.T,X)

#    def fit(self, start_params=None, maxiter=35, method='newton',
#            tol=1e-08):
#        """
#        Fits the binary logit model.
#
#        Parameters
#        ----------
#        start_params : array-like, optional
#            The default is a 0 vector.
#        maxiter : int, optional
#            Maximum number of iterations.  The default is 35.
#        method : str, optional
#            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
#        tol : float, optional
#            The convergence tolerance for the solver.  The default is
#            1e-08.
#
#        Returns
#        --------
#        DiscreteResults object
#
#        See also
#        --------
#        scikits.statsmodels.model.LikelihoodModel
#        scikits.statsmodels.sandbox.discretemod.DiscreteResults
#        scipy.optimize
#        """
#        mlefit = super(Logit, self).fit(start_params=start_params,
#            maxiter=maxiter, method=method, tol=tol)
#        params = mlefit.params
#        mlefit = DiscreteResults(self, params, self.hessian(params))
#        return mlefit


class Probit(DiscreteModel):
    """
    Binary choice Probit model

    Parameters
    ----------
    endog : array-like
        1-d array of the response variable.
    exog : array-like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is included
        in the data.

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    nobs : float
        The number of observations of the model.

    Methods
    --------
    cdf
    fit
    hessian
    information
    initialize
    loglike
    pdf
    predict
    score
    """

    def cdf(self, X):
        """
        Probit (Normal) cumulative distribution function

        Parameters
        ----------
        X : array-like
            The linear predictor of the model (XB).

        Returns
        --------
        The cdf evaluated at `X`.

        Notes
        -----
        This function is just an alias for scipy.stats.norm.cdf
        """
        return stats.norm.cdf(X)

    def pdf(self, X):
        """
        Probit (Normal) probability density function

        Parameters
        ----------
        X : array-like
            The linear predictor of the model (XB).

        Returns
        --------
        The pdf evaluated at X.

        Notes
        -----
        This function is just an alias for scipy.stats.norm.pdf

        """
        X = np.asarray(X)
        return stats.norm.pdf(X)


    def loglike(self, params):
        """
        Log-likelihood of probit model (i.e., the normal distribution).

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        The log-likelihood evaluated at params

        Notes
        -----
        .. math:: \\ln L=\\sum_{i}\\ln\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """

        q = 2*self.endog - 1
        X = self.exog
        return np.sum(np.log(self.cdf(q*np.dot(X,params))))

    def score(self, params):
        """
        Probit model score (gradient) vector

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        The score vector of the model evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left[\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\\right]x_{i}

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """
        y = self.endog
        X = self.exog
        XB = np.dot(X,params)
        q = 2*y - 1
        L = q*self.pdf(q*XB)/self.cdf(q*XB)
        return np.dot(L,X)

    def hessian(self, params):
        """
        Probit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        The Hessian evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\lambda_{i}\\left(\\lambda_{i}+x_{i}^{\\prime}\\beta\\right)x_{i}x_{i}^{\\prime}
        where
        .. math:: \\lambda_{i}=\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}
        and :math:`q=2y-1`
        """
        X = self.exog
        XB = np.dot(X,params)
        q = 2*self.endog - 1
        L = q*self.pdf(q*XB)/self.cdf(q*XB)
        return np.dot(-L*(L+XB)*X.T,X)

#    def fit(self, start_params=None, maxiter=35, method='newton',
#            tol=1e-08):
#        """
#        Fits the binary probit model.
#
#        Parameters
#        ----------
#        start_params : array-like, optional
#            The default is a 0 vector.
#        maxiter : int, optional
#            Maximum number of iterations.  The default is 35.
#        method : str, optional
#            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
#        tol : float, optional
#            The convergence tolerance for the solver.  The default is
#            1e-08.
#
#        Returns
#        --------
#        DiscreteResults object
#
#        See also
#        --------
#        scikits.statsmodels.model.LikelihoodModel
#        scikits.statsmodels.sandbox.discretemod.DiscreteResults
#        scipy.optimize
#        """
#        mlefit = super(Probit, self).fit(start_params=start_params,
#            maxiter=maxiter, method=method, tol=tol)
#        params = mlefit.params
#        mlefit = DiscreteResults(self, params, self.hessian(params))
#        return mlefit


class MNLogit(DiscreteModel):
    """
    Multinomial logit model

    Parameters
    ----------
    endog : array-like
        `endog` is an 1-d vector of the endogenous response.  `endog` can
        contain strings, ints, or floats.  Note that if it contains strings,
        every distinct string will be a category.  No stripping of whitespace
        is done.
    exog : array-like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is included
        in the data.

    Attributes
    ----------
    J : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one.
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    names : dict
        A dictionary mapping the column number in `wendog` to the variables
        in `endog`.
    nobs : float
        The number of observations of the model.
    wendog : array
        An n x j array where j is the number of unique categories in `endog`.
        Each column of j is a dummy variable indicating the category of
        each observation. See `names` for a dictionary mapping each column to
        its category.

    Methods
    --------
    cdf
    fit
    hessian
    information
    initialize
    loglike
    pdf
    predict
    score

    Notes
    -----
    See developer notes for further information on `MNLogit` internals.
    """

    def initialize(self):
        """
        Preprocesses the data for MNLogit.

        Turns the endogenous variable into an array of dummies and assigns
        J and K.
        """
        super(MNLogit, self).initialize()
        #This is also a "whiten" method as used in other models (eg regression)
        wendog, self.names = tools.categorical(self.endog, drop=True,
                dictnames=True)
        self.wendog = wendog    # don't drop first category
        self.J = float(wendog.shape[1])
        self.K = float(self.exog.shape[1])
        self.df_model *= (self.J-1) # for each J - 1 equation.
        self.df_resid = self.nobs - self.df_model - (self.J-1)


    def _eXB(self, params, exog=None):
        """
        A private method used by the cdf.

        Returns
        -------
        :math:`\exp(\beta_{j}^{\prime}x_{i})`

        where :math:`j = 0,1,...,J`

        Notes
        -----
        A row of ones is appended for the dropped category.
        """
        if exog == None:
            exog = self.exog
        eXB = np.exp(np.dot(params.reshape(-1, exog.shape[1]), exog.T))
        eXB = np.vstack((np.ones((1, self.nobs)), eXB))
        return eXB

    def pdf(self, eXB):
        """
        NotImplemented
        """
        pass

    def cdf(self, eXB):
        """
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        eXB : array
            The exponential predictor of the model exp(XB).

        Returns
        --------
        The cdf evaluated at `eXB`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        num = eXB
        denom = eXB.sum(axis=0)
        return num/denom[None,:]

    def loglike(self, params):
        """
        Log-likelihood of the multinomial logit model.

        Parameters
        ----------
        params : array-like
            The parameters of the multinomial logit model.

        Returns
        -------
        The log-likelihood function of the logit model.  See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)
        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        d = self.wendog
        eXB = self._eXB(params)
        logprob = np.log(self.cdf(eXB))
        return (d.T * logprob).sum()

    def score(self, params):
        """
        Score matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the multinomial logit model.

        Returns
        --------
        The 2-d score vector of the multinomial logit model evaluated at
        `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`

        In the multinomial model ths score matrix is K x J-1 but is returned
        as a flattened array to work with the solvers.
        """
        eXB = self._eXB(params)
        firstterm = self.wendog[:,1:].T - self.cdf(eXB)[1:,:]
        return np.dot(firstterm, self.exog).flatten()

    def hessian(self, params):
        """
        Multinomial logit Hessian matrix of the log-likelihood

        Parameters
        -----------
        params : array-like
            The parameters of the model

        Returns
        -------
        The Hessian evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}

        where
        :math:`\boldsymbol{1}\left(j=l\right)` equals 1 if `j` = `l` and 0
        otherwise.

        The actual Hessian matrix has J**2 * K x K elements. Our Hessian
        is reshaped to be square (J*K, J*K) so that the solvers can use it.

        This implementation does not take advantage of the symmetry of
        the Hessian and could probably be refactored for speed.
        """
        X = self.exog
        eXB = self._eXB(params)
        pr = self.cdf(eXB)
        partials = []
        J = self.wendog.shape[1] - 1
        K = self.exog.shape[1]
        for i in range(J):
            for j in range(J): # this loop assumes we drop the first col.
                if i == j:
                    partials.append(\
                        -np.dot((pr[i+1,:]*(1-pr[j+1,:]))[None,:]*X.T,X))
                else:
                    partials.append(-np.dot(pr[i+1,:]*-pr[j+1,:][None,:]*X.T,X))
        H = np.array(partials)
        # the developer's notes on multinomial should clear this math up
        H = np.transpose(H.reshape(J,J,K,K), (0,2,1,3)).reshape(J*K,J*K)
        return H

#    def fit(self, start_params=None, maxiter=35, method='newton',
#            tol=1e-08):
#        """
#        Fits the multinomial logit model.
#
#        Parameters
#        ----------
#        start_params : array-like, optional
#            The default is a 0 vector.
#        maxiter : int, optional
#            Maximum number of iterations.  The default is 35.
#        method : str, optional
#            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
#        tol : float, optional
#            The convergence tolerance for the solver.  The default is
#            1e-08.
#
#        Notes
#        -----
#        The reference category is always the first column of `wendog` for now.
#        """
#        if start_params == None:
#            start_params = np.zeros((self.exog.shape[1]*\
#                    (self.wendog.shape[1]-1)))
#        mlefit = super(MNLogit, self).fit(start_params=start_params,
#                maxiter=maxiter, method=method, tol=tol)
#        params = mlefit.params.reshape(-1, self.exog.shape[1])
#        mlefit = DiscreteResults(self, params, self.hessian(params))
#        return mlefit

#TODO: Weibull can replaced by a survival analsysis function
# like stat's streg (The cox model as well)
#class Weibull(DiscreteModel):
#    """
#    Binary choice Weibull model
#
#    Notes
#    ------
#    This is unfinished and untested.
#    """
##TODO: add analytic hessian for Weibull
#    def initialize(self):
#        pass
#
#    def cdf(self, X):
#        """
#        Gumbell (Log Weibull) cumulative distribution function
#        """
##        return np.exp(-np.exp(-X))
#        return stats.gumbel_r.cdf(X)
#        # these two are equivalent.
#        # Greene table and discussion is incorrect.
#
#    def pdf(self, X):
#        """
#        Gumbell (LogWeibull) probability distribution function
#        """
#        return stats.gumbel_r.pdf(X)
#
#    def loglike(self, params):
#        """
#        Loglikelihood of Weibull distribution
#        """
#        X = self.exog
#        cdf = self.cdf(np.dot(X,params))
#        y = self.endog
#        return np.sum(y*np.log(cdf) + (1-y)*np.log(1-cdf))
#
#    def score(self, params):
#        y = self.endog
#        X = self.exog
#        F = self.cdf(np.dot(X,params))
#        f = self.pdf(np.dot(X,params))
#        term = (y*f/F + (1 - y)*-f/(1-F))
#        return np.dot(term,X)
#
#    def hessian(self, params):
#        hess = nd.Jacobian(self.score)
#        return hess(params)
#
#    def fit(self, start_params=None, method='newton', maxiter=35, tol=1e-08):
## The example had problems with all zero start values, Hessian = 0
#        if start_params is None:
#            start_params = OLS(self.endog, self.exog).fit().params
#        mlefit = super(Weibull, self).fit(start_params=start_params,
#                method=method, maxiter=maxiter, tol=tol)
#        return mlefit
#
class NegBinTwo(DiscreteModel):
    """
    NB2 Negative Binomial model.

    Note: This is not working yet
    """
#NOTE: to use this with the solvers, the likelihood fit will probably
# need to be amended to have args, so that we can pass the ancillary param
# if not we can just stick the alpha param on the end of the beta params and
# amend all the methods to reflect this
# if we try to keep them separate I think we'd have to use a callback...
# need to check variance function, then derive score vector, and hessian
# loglike should be fine...
# also, alpha should maybe always be lnalpha to contrain it to be positive

#    def pdf(self, X, alpha):
#        a1 = alpha**-1
#        term1 = special.gamma(X + a1)/(special.agamma(X+1)*special.gamma(a1))

    def loglike(self, params):
        """
        Loglikelihood for NB2 model

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """
        alpha = params[-1]
        params = params[:-1]
        a1 = alpha**-1
        y = self.endog
        J = special.gammaln(y+a1) - special.gammaln(a1)
# See Cameron and Trivedi 1998 for a simplification of the above
# writing a convenience function using the log summation, *might*
# be more accurate
        XB = np.dot(self.exog,params)
        return np.sum(J - np.log(factorial(y)) - \
                (y+a1)*np.log(1+alpha*np.exp(XB))+y*np.log(alpha)+y*XB)

    def score(self, params):
        """
        Score vector for NB2 model
        """
        import numdifftools as nd
        y = self.endog
        X = self.exog
        jfun = nd.Jacobian(self.loglike)
        return jfun(params)[-1]
        dLda2 = jfun(params)[-1]
        alpha = params[-1]
        params = params[:-1]
        XB = np.dot(X,params)
        mu = np.exp(XB)
        a1 = alpha**-1
        f1 = lambda x: 1./((x-1)*x/2. + x*a1)
        cond = y>0
        dJ = np.piecewise(y, cond, [f1,1./a1])
# if y_i < 1, this equals zero!  Not noted in C &T
        dLdB = np.dot((y-mu)/(1+alpha*mu),X)
        return dLdB
#
#        dLda = np.sum(1/alpha**2 * (np.log(1+alpha*mu) - dJ) + \
#                (y-mu)/(alpha*(1+alpha*mu)))
#        scorevec = np.zeros((len(dLdB)+1))
#        scorevec[:-1] = dLdB
#        scorevec[-1] = dLda
#        scorevec[-1] = dLda2[-1]
#        return scorevec

    def hessian(self, params):
        """
        Hessian of NB2 model.  Currently uses numdifftools
        """
#        d2dBdB =
#        d2da2 =
        import numdifftools as nd
        Hfun = nd.Jacobian(self.score)
        return Hfun(params)[-1]
# is the numerical hessian block diagonal?  or is it block diagonal by assumption?

    def fit(self, start_params=None, maxiter=35, method='bfgs', tol=1e-08):
#        start_params = [0]*(self.exog.shape[1])+[1]
# Use poisson fit as first guess.
        start_params = Poisson(self.endog, self.exog).fit().params
        start_params = np.roll(np.insert(start_params, 0, 1), -1)
        mlefit = super(NegBinTwo, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        return mlefit


### Results Class ###

#class DiscreteResults(object):
#TODO: these need to return z scores
class DiscreteResults(LikelihoodModelResults):
    """
    A results class for the discrete dependent variable models.

    Parameters
    ----------
    model : A DiscreteModel instance
    params : array-like
        The parameters of a fitted model.
    hessian : array-like
        The hessian of the fitted model.
    scale : float
        A scale parameter for the covariance matrix.


    Returns
    -------
    *Attributes*

    aic : float
        Akaike information criterion.  -2*(`llf` - p) where p is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. -2*`llf` + ln(`nobs`)*p where p is the
        number of regressors including the intercept.
    bse : array
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : array
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; -2*(`llnull` - `llf`)
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. 1 - (`llf`/`llnull`)

    Methods
    -------
    margeff
        Get marginal effects of the fitted model.
    conf_int

    """

    def __init__(self, model, mlefit):
#        super(DiscreteResults, self).__init__(model, params,
#                np.linalg.inv(-hessian), scale=1.)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.nobs
        self._cache = resettable_cache()
        self.__dict__.update(mlefit.__dict__)

    @cache_readonly
    def bse(self):
        bse = np.sqrt(np.diag(self.cov_params()))
        if self.params.ndim == 1 or self.params.shape[1] == 1:
            return bse
        else:
            return bse.reshape(self.params.shape)

    @cache_readonly
    def llf(self):
        model = self.model
        return model.loglike(self.params)

    @cache_readonly
    def prsquared(self):
        return 1 - self.llf/self.llnull

    @cache_readonly
    def llr(self):
        return -2*(self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        return stats.chisqprob(self.llr, self.df_model)

    @cache_readonly
    def llnull(self):
        model = self.model # will this use a new instance?
#TODO: what parameters to pass to fit?
        null = model.__class__(model.endog, np.ones(model.nobs)).fit(disp=0)
        return null.llf

    @cache_readonly
    def fittedvalues(self):
        return np.dot(self.model.exog, self.params)

    @cache_readonly
    def aic(self):
        if hasattr(self.model, "J"):
            return -2*(self.llf - (self.df_model+self.model.J-1))
        else:
            return -2*(self.llf - (self.df_model+1))

    @cache_readonly
    def bic(self):
        if hasattr(self.model, "J"):
            return -2*self.llf + np.log(self.nobs)*\
                    (self.df_model+self.model.J-1)
        else:
            return -2*self.llf + np.log(self.nobs)*(self.df_model+1)

    def conf_int(self, alpha=.05, cols=None):
        if hasattr(self.model, "J"):
            confint = super(DiscreteResults, self).conf_int(alpha=alpha,
                    cols=cols)
            return confint.transpose(0,2,1).reshape(self.model.J-1,self.model.K,2)
        else:
            return super(DiscreteResults, self).conf_int(alpha=alpha, cols=cols)
    conf_int.__doc__ = LikelihoodModelResults.conf_int.__doc__
#TODO: does the above work?

#TODO: the baove and the below will change if we merge the mixin branch
    def t(self, column=None):
        if hasattr(self.model, "J"):
            #TODO: make this more robust once this is sorted
            if column is None:
                column = range(int(self.model.K))
            else:
                column = np.asarray(column)
            return self.params/self.bse[:,column]
        else:
            return super(DiscreteResults, self).t(column=column)
    t.__doc__ = LikelihoodModelResults.t.__doc__


    def margeff(self, params=None, at='overall', method='dydx', atexog=None,
        dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        params : array-like, optional
            The parameters.
        at : str, optional
            Options are:
            'overall', The average of the marginal effects at each observation.
            'mean', The marginal effects at the mean of each regressor.
            'median', The marginal effects at the median of each regressor.
            'zero', The marginal effects at zero for each regressor.
            'all', The marginal effects at each observation.
            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            'dydx' - dy/dx - No transformation is made and marginal effects
                are returned.  This is the default.
            'eyex' - estimate elasticities of variables in `exog` --
                d(lny)/d(lnx)
            'dyex' - estimate semielasticity -- dy/d(lnx)
            'eydx' - estimate semeilasticity -- d(lny)/dx
            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array-like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
#TODO:
#        factor : None or dictionary, optional
#            If a factor variable is present (it must be an integer, though
#            of type float), then `factor` may be a dict with the zero-indexed
#            column of the factor and the value should be the base-outcome.

        model = self.model
        method = method.lower()
        at = at.lower()
        if params is None:
            params = self.params
        else:
            params = np.asarray(params)
            # could prob use a shape check here (do we even need this option?)
        if not at in ['overall','mean','median','zero','all']:
            raise ValueError, "%s not a valid option for `at`." % at

        exog = model.exog.copy() # copy because values are changed
        ind = exog.var(0) != 0 # index for non-constants

        # get user instructions
        if dummy == True or count == True:
            if method in ['dyex','eyex']:
                raise ValueError, "%s not allowed for discrete \
variables" % method
            if at in ['median', 'zero']:
                raise ValueError, "%s not allowed for discrete \
variables" % at
            if dummy:
                dummy_ind = _isdummy(exog)
            if count:
                count_ind = _iscount(exog)
        if atexog is not None:
            if not isinstance(atexog, dict):
                raise ValueError, "exog, if not None, should be a dict. \
Got %s" % type(atexog)
            for key in atexog:
                exog[:,key] = atexog[key]

        if at == 'mean':
            exog[:,ind] = exog.mean(0)[ind]
        elif at == 'median':
            exog[:,ind] = np.median(exog, axis=0)[ind]
        elif at == 'zero':
            exog[:,ind] = 0
        if method not in ['dydx','eyex','dyex','eydx']:
            raise ValueError, "method is not understood.  Got %s" % method
        # group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if isinstance(model, (Probit, Logit)):
            effects = np.dot(model.pdf(np.dot(exog,params))[:,None],
                    params[None,:])
        # group 2 oprobit, ologit, gologit, mlogit, biprobit
        #TODO
        # group 3 poisson, nbreg, zip, zinb
        elif isinstance(model, (Poisson)):
            effects = np.exp(np.dot(exog, params))[:,None]*params[None,:]
        fittedvalues = np.dot(exog, params) #TODO: add a predict method
                                            # that takes an exog kwd
        if 'ex' in method:
            effects *= exog
        if 'dy' in method:
            if at == 'all':
                effects = effects[:,ind]
            elif at == 'overall':
                effects = effects.mean(0)[ind]
            else:
                effects = effects[0,ind]
        if 'ey' in method:
            effects /= model.cdf(fittedvalues[:,None])
            if at == 'all':
                effects = effects[:,ind]
            elif at == 'overall':
                effects = effects.mean(0)[ind]
            else:
                effects = effects[0,ind]
        if dummy == True:
            for i, tf in enumerate(dummy_ind):
                if tf == True:
                    exog0 = exog.copy()
                    exog0[:,i] = 0
                    fittedvalues0 = np.dot(exog0,params)
                    exog1 = exog.copy()
                    exog1[:,i] = 1
                    fittedvalues1 = np.dot(exog1, params)
                    effect0 = model.cdf(np.dot(exog0, params))
                    effect1 = model.cdf(np.dot(exog1, params))
                    if 'ey' in method:
                        effect0 /= model.cdf(fittedvalues0)
                        effect1 /= model.cdf(fittedvalues1)
                    effects[i] = (effect1 - effect0).mean()
        if count == True:
            for i, tf in enumerate(count_ind):
                if tf == True:
                    exog0 = exog.copy()
                    exog1 = exog.copy()
                    exog1[:,i] += 1
                    effect0 = model.cdf(np.dot(exog0, params))
                    effect1 = model.cdf(np.dot(exog1, params))
#TODO: compute discrete elasticity correctly
#Stata doesn't use the midpoint method or a weighted average.
#Check elsewhere
                    if 'ey' in method:
#                        #TODO: don't know if this is theoretically correct
                        fittedvalues0 = np.dot(exog0,params)
                        fittedvalues1 = np.dot(exog1,params)
#                        weight1 = model.exog[:,i].mean()
#                        weight0 = 1 - weight1
                        wfv = (.5*model.cdf(fittedvalues1) + \
                                .5*model.cdf(fittedvalues0))
                        effects[i] = ((effect1 - effect0)/wfv).mean()
                    effects[i] = (effect1 - effect0).mean()
        # Set standard error of the marginal effects by Delta method.
        self.margfx_se = None
        self.margfx = effects
        return effects

if __name__=="__main__":
    import numpy as np
    import scikits.statsmodels as sm
# Scratch work for negative binomial models
# dvisits was written using an R package, I can provide the dataset
# on request until the copyright is cleared up
#TODO: request permission to use dvisits
    data2 = np.genfromtxt('./dvisits.txt', names=True)
# note that this has missing values for Accident
    endog = data2['doctorco']
    exog = data2[['sex','age','agesq','income','levyplus','freepoor',
            'freerepa','illness','actdays','hscore','chcond1',
            'chcond2']].view(float).reshape(len(data2),-1)
    exog = sm.add_constant(exog, prepend=True)
    poisson_mod = Poisson(endog, exog)
    poisson_res = poisson_mod.fit()
#    nb2_mod = NegBinTwo(endog, exog)
#    nb2_res = nb2_mod.fit()
# solvers hang (with no error and no maxiter warn...)
# haven't derived hessian (though it will be block diagonal) to check
# newton, note that Lawless (1987) has the derivations
# appear to be something wrong with the score?
# according to Lawless, traditionally the likelihood is maximized wrt to B
# and a gridsearch on a to determin ahat?
# or the Breslow approach, which is 2 step iterative.
    nb2_params = [-2.190,.217,-.216,.609,-.142,.118,-.497,.145,.214,.144,
            .038,.099,.190,1.077] # alpha is last
    # taken from Cameron and Trivedi
# the below is from Cameron and Trivedi as well
#    endog2 = np.array(endog>=1, dtype=float)
# skipped for now, binary poisson results look off?


