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
from scikits.statsmodels.model import LikelihoodModel, LikelihoodModelResults
from scikits.statsmodels import tools
from scikits.statsmodels.decorators import *
from scikits.statsmodels.regression import OLS
from scipy import stats, factorial, special, optimize # opt just for nbin
import numdifftools as nd #This will be removed when all have analytic hessians

#TODO: all of the fit methods that call super can be taken out
# once the ResultsClass is settled, except of course the ones
# that need to do some preprocessing before fit is called

#TODO: add options for the parameter covariance/variance
# ie., OIM, EIM, and BHHH see Green 21.4

def isdummy(X):
    """
    Given an array X, returns a column index for the dummy variables.

    Examples
    --------
    >>> X = np.random.randint(0, 2, size=(15,5)).view(float)
    >>> X = np.asarray(X, dtype=float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = isdummy(X)
    >>> ind
    (array([0, 3, 4]),)
    """
    nobs = X.shape[0]
    sums = X.sum(0)
    colsums = np.logical_or(X == 1, X == 0).sum(0)
    ind = np.where(np.logical_and(colsums == nobs, sums/nobs!=1))
    return ind

def add_factorial(X):
    """
    Returns a vector of descending numbers added sequential.

    For instance, if given [5, 4, 0, 2], returns [15, 10, 0, 3].
    """
    X = np.asarray(X)
    return X/2. * X + (1-X)
# or equivalently
#    return X*(X+1)/2.

class DiscreteModel(LikelihoodModel):
    """
    Template for discrete choice models.

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

    Notes
    -----
    The cdf method is not implemented for the Poisson.
    """

    def pdf(self, params):
        """
        Poisson model probability mass function

        Parameters
        -----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        The value of the Poisson PMF at each point.

        Notes
        --------
        The pmf is defined as

        .. math:: \\frac{e^{-\\lambda_{i}}\\lambda_{i}^{y_{i}}}{y_{i}!}

        where :math:`\\lambda` assumes the loglinear model. I.e.,

        .. math:: \\ln\\lambda_{i}=X\\beta

        This method is not used in the fitting algorithm.
        """
        y = self.endog
        X = self.exog
        L = np.exp(np.dot(X,params))
        return (np.exp(-L)*L**y)/factorial(y)

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

    def fit(self, start_params=None, maxiter=35, method='newton',
            tol=1e-08):
        """
        Fits the Poisson model.

        Parameters
        ----------
        start_params : array-like, optional
            The default is a 0 vector.
        maxiter : int, optional
            Maximum number of iterations.  The default is 35.
        method : str, optional
            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
        tol : float, optional
            The convergence tolerance for the solver.  The default is
            1e-08.

        Returns
        --------
        DiscreteResults object

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel
        scikits.statsmodels.sandbox.discretemod.DiscreteResults
        scipy.optimize
        """

        mlefit = super(Poisson, self).fit(start_params=start_params,
            maxiter=maxiter, method=method, tol=tol)
        params = mlefit.params
        mlefit = DiscreteResults(self, params, self.hessian(params))
        return mlefit

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

    def fit(self, start_params=None, maxiter=35, method='newton',
            tol=1e-08):
        """
        Fits the binary logit model.

        Parameters
        ----------
        start_params : array-like, optional
            The default is a 0 vector.
        maxiter : int, optional
            Maximum number of iterations.  The default is 35.
        method : str, optional
            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
        tol : float, optional
            The convergence tolerance for the solver.  The default is
            1e-08.

        Returns
        --------
        DiscreteResults object

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel
        scikits.statsmodels.sandbox.discretemod.DiscreteResults
        scipy.optimize
        """
        mlefit = super(Logit, self).fit(start_params=start_params,
            maxiter=maxiter, method=method, tol=tol)
        params = mlefit.params
        mlefit = DiscreteResults(self, params, self.hessian(params))
        return mlefit


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

    def fit(self, start_params=None, maxiter=35, method='newton',
            tol=1e-08):
        """
        Fits the binary probit model.

        Parameters
        ----------
        start_params : array-like, optional
            The default is a 0 vector.
        maxiter : int, optional
            Maximum number of iterations.  The default is 35.
        method : str, optional
            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
        tol : float, optional
            The convergence tolerance for the solver.  The default is
            1e-08.

        Returns
        --------
        DiscreteResults object

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel
        scikits.statsmodels.sandbox.discretemod.DiscreteResults
        scipy.optimize
        """
        mlefit = super(Probit, self).fit(start_params=start_params,
            maxiter=maxiter, method=method, tol=tol)
        params = mlefit.params
        mlefit = DiscreteResults(self, params, self.hessian(params))
        return mlefit


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
        #This is also a "whiten" method as used in other models
        wendog, self.names = tools.categorical(self.endog, drop=True,
                dictnames=True)
        self.wendog = wendog    # don't drop first category
        self.J = float(wendog.shape[1])
        self.K = float(self.exog.shape[1])

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
        if exog is None:
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

    def fit(self, start_params=None, maxiter=35, method='newton',
            tol=1e-08):
        """
        Fits the multinomial logit model.

        Parameters
        ----------
        start_params : array-like, optional
            The default is a 0 vector.
        maxiter : int, optional
            Maximum number of iterations.  The default is 35.
        method : str, optional
            `method` can be 'newton', 'ncg', 'bfgs'. The default is 'newton'.
        tol : float, optional
            The convergence tolerance for the solver.  The default is
            1e-08.

        Notes
        -----
        The reference category is always the first column of `wendog` for now.
        """
        if start_params is None:
            start_params = np.zeros((self.exog.shape[1]*\
                    (self.wendog.shape[1]-1)))
        mlefit = super(MNLogit, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        mlefit.params = mlefit.params.reshape(-1, self.exog.shape[1])
        return mlefit


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

    Attributes
    ----------
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
    margeff - Get marginal effects of the fitted model.
    """

    def __init__(self, model, params, hessian, scale=1.):
        super(DiscreteResults, self).__init__(model, params,
                np.linalg.inv(-hessian), scale=1.)
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = resettable_cache()

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params()))

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
        null = model.__class__(model.endog, np.ones(model.nobs)).fit()
        return null.llf

    @cache_readonly
    def aic(self):
        return -2*(self.llf - (self.df_model+1))

    @cache_readonly
    def bic(self):
        return -2*self.llf + np.log(self.nobs)*(self.df_model+1)

    def margeff(self, params=None, loc='meanfx', method='dydx', exog=None,
        nodiscrete=None):
        """
        Parameters
        ----------
        params : array-like, optional
            The parameters.
        loc : str, optional
            'meanfx' - The average of the marginal effects over all of the
                exogenous variables.
            'atmeanfx' - The marginal effects at the mean of the exogenous
                variables.
            'all' - returns the marginal effects at each observation.
        method : str not implemented yet, optional
            'dydx' dy/dx
            'eyex' dlogy/dlogx
            'dyex' dy/dlogx
            'eydx' dlogy/dx
        exog : array-like, optional
            Optionally, you can provide the exogenous variables over which
                to get the marginal effects.
        nodiscrete : None or list of ints of dummy variable columns
            If None, treats dummy variables as continuous.  Default
            If a list of columns which contain dummy variables is provided the
            marginal effects account for the dummies.

        Notes
        -----
        Only the defaults are available now.  This is not yet finished or tested.
        """
        model = self.model
        if params is None:
            params = self.params
        if exog is None:
            exog = model.exog
            ind = np.where(exog.var(0) != 0)[0] # don't report the constant
        elif exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                return np.dot(model.pdf(np.dot(exog,params),params))
#TODO: this definitely needs to be tested
#                ind = np.arange(len(exog))
#                exog = exog
            if exog.ndim > 1:
                ind = np.where(exog.var(0) != 0)[0]
        if loc is 'meanfx' and not nodiscrete:
            if not isinstance(model, MNLogit):
                change = np.dot(model.pdf(np.dot(exog,params))[:,None],
                    params[None,:]).mean(0)
#            elif isinstance(model, MNLogit):
#                change = np.dot(model.pdf(model._eXB(params, exog)),
#                    params[None,:]).mean(0)



        return change[ind]

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


