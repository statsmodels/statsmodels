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
from scipy.special import gammaln
from scipy import stats, special, optimize # opt just for nbin
import statsmodels.tools.tools as tools
from statsmodels.tools.decorators import (resettable_cache,
        cache_readonly)
from statsmodels.regression.linear_model import OLS
from scipy import stats, special, optimize # opt just for nbin
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap

#TODO: add options for the parameter covariance/variance
# ie., OIM, EIM, and BHHH see Green 21.4

#### margeff helper functions ####
#NOTE: todo marginal effects for group 2
# group 2 oprobit, ologit, gologit, mlogit, biprobit

def _check_margeff_args(at, method):
    """
    Checks valid options for margeff
    """
    if at not in ['overall','mean','median','zero','all']:
        raise ValueError("%s not a valid option for `at`." % at)
    if method not in ['dydx','eyex','dyex','eydx']:
        raise ValueError("method is not understood.  Got %s" % method)

def _check_discrete_args(at, method):
    """
    Checks the arguments for margeff if the exogenous variables are discrete.
    """
    if method in ['dyex','eyex']:
        raise ValueError("%s not allowed for discrete variables" % method)
    if at in ['median', 'zero']:
        raise ValueError("%s not allowed for discrete variables" % at)

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
    remainder = np.logical_and(np.all(X % 1. == 0, axis = 0),
                               X.var(0) != 0)
    dummy = _isdummy(X)
    remainder -= dummy
    return remainder

def _get_margeff_exog(exog, at, atexog, ind):
    if atexog is not None: # user supplied
        if not isinstance(atexog, dict):
            raise ValueError("atexog should be a dict not %s"\
                    % type(atexog))
        for key in atexog:
            exog[:,key] = atexog[key]
    if at == 'mean':
        exog = np.atleast_2d(exog.mean(0))
    elif at == 'median':
        exog = np.atleast_2d(np.median(exog, axis=0))
    elif at == 'zero':
        exog = np.zeros((1,exog.shape[1]))
        exog[0,~ind] = 1
    return exog

def _get_count_effects(effects, exog, count_ind, method, model, params):
    for i, tf in enumerate(count_ind):
        if tf == True:
            exog0 = exog.copy()
            effect0 = model.predict(params, exog0)
            wf1 = model.predict
            exog0[:,i] += 1
            effect1 = model.predict(params, exog0)
    #TODO: compute discrete elasticity correctly
    #Stata doesn't use the midpoint method or a weighted average.
    #Check elsewhere
            if 'ey' in method:
                pass
                ##TODO: don't know if this is theoretically correct
                #fittedvalues0 = np.dot(exog0,params)
                #fittedvalues1 = np.dot(exog1,params)
                #weight1 = model.exog[:,i].mean()
                #weight0 = 1 - weight1
                #wfv = (.5*model.cdf(fittedvalues1) + \
                        #        .5*model.cdf(fittedvalues0))
                #effects[i] = ((effect1 - effect0)/wfv).mean()
            effects[i] = (effect1 - effect0).mean()
    return effects


def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    for i, tf in enumerate(dummy_ind):
        if tf == True:
            exog0 = exog.copy() # only copy once, can we avoid a copy?
            exog0[:,i] = 0
            effect0 = model.predict(params, exog0)
            #fittedvalues0 = np.dot(exog0,params)
            exog0[:,i] = 1
            effect1 = model.predict(params, exog0)
            if 'ey' in method:
                effect0 = np.log(effect0)
                effect1 = np.log(effect1)
            effects[i] = (effect1 - effect0).mean() # mean for overall
    return effects

def _effects_at(effects, at, ind):
    if at == 'all':
        effects = effects[:,ind]
    elif at == 'overall':
        effects = effects.mean(0)[ind]
    else:
        effects = effects[0,ind]
    return effects


#### Private Model Classes ####

class DiscreteModel(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    statsmodels.model.LikelihoodModel.
    """
    def __init__(self, endog, exog):
        super(DiscreteModel, self).__init__(endog, exog)
        self.raise_on_perfect_prediction = True

    def initialize(self):
        """
        Initialize is called by
        statsmodels.model.LikelihoodModel.__init__
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

    def _check_perfect_pred(self, params):
        endog = self.endog
        fittedvalues = self.cdf(np.dot(self.exog, params))
        if (self.raise_on_perfect_prediction and 
                np.allclose(fittedvalues - endog, 0)):
            msg = "Perfect separation detected, results not available"
            raise PerfectSeparationError(msg)

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """
        Fit the model using maximum likelihood.

        The rest of the docstring is from
        statsmodels.LikelihoodModel.fit
        """
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass # make a function factory to have multiple call-backs
        mlefit = super(DiscreteModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        return mlefit # up to subclasses to wrap results

    fit.__doc__ += base.LikelihoodModel.fit.__doc__

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.
        """
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None):
        """
        This should implement the derivative of the non-linear function
        """
        raise NotImplementedError

class BinaryModel(DiscreteModel):
    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            Fitted parameters of the model.
        exog : array-like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used.
        linear : bool, optional
            If True, returns the linear predictor dot(exog,params).  Else,
            returns the value of the cdf at the linear predictor.

        Returns
        -------
        array
            Fitted values at exog.
        """
        if exog is None:
            exog = self.exog
        if not linear:
            return self.cdf(np.dot(exog, params))
        else:
            return np.dot(exog, params)

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        bnryfit = super(BinaryModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        discretefit = BinaryResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

    def _derivative_exog(self, params, exog=None):
        """
        For computing marginal effects.
        """
        #note, this form should be appropriate for
        ## group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog == None:
            exog = self.exog
        return np.dot(self.pdf(np.dot(exog, params))[:,None], params[None,:])

class MultinomialModel(BinaryModel):
    def initialize(self):
        """
        Preprocesses the data for MNLogit.

        Turns the endogenous variable into an array of dummies and assigns
        J and K.
        """
        super(MultinomialModel, self).initialize()
        #This is also a "whiten" method as used in other models (eg regression)
        wendog, ynames = tools.categorical(self.endog, drop=True,
                dictnames=True)
        self._ynames_map = ynames
        self.wendog = wendog    # don't drop first category
        self.J = float(wendog.shape[1])
        self.K = float(self.exog.shape[1])
        self.df_model *= (self.J-1) # for each J - 1 equation.
        self.df_resid = self.exog.shape[0] - self.df_model - (self.J-1)


    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            2d array of fitted parameters of the model. Should be in the
            order returned from the model.
        exog : array-like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used.
        linear : bool, optional
            If True, returns the linear predictor dot(exog,params).  Else,
            returns the value of the cdf at the linear predictor.

        Notes
        -----
        Column 0 is the base case, the rest conform to the rows of params
        shifted up one for the base case.
        """
        if exog is None: # do here to accomodate user-given exog
            exog = self.exog
        pred = super(MultinomialModel, self).predict(params, exog, linear)
        if linear:
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        if start_params is None:
            start_params = np.zeros((self.K * (self.J-1)))
        else:
            start_params = np.asarray(start_params)
        callback = lambda x : None # placeholder until check_perfect_pred
        # skip calling super to handle results from LikelihoodModel
        mnfit = base.LikelihoodModel.fit(self, start_params = start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = MultinomialResults(self, mnfit)
        return MultinomialResultsWrapper(mnfit)
    fit.__doc__ = DiscreteModel.fit.__doc__

class CountModel(DiscreteModel):
    def __init__(self, endog, exog, offset=None, exposure=None):
        super(CountModel, self).__init__(endog, exog)
        self._check_inputs(offset, exposure) # attaches if needed

    def _check_inputs(self, offset, exposure):
        if offset is not None:
            offset = np.asarray(offset)
            if offset.shape[0] != self.endog.shape[0]:
                raise ValueError("offset is not the same length as endog")
            self.offset = offset

        if exposure is not None:
            exposure = np.log(exposure)
            if exposure.shape[0] != self.endog.shape[0]:
                raise ValueError("exposure is not the same length as endog")
            self.exposure = exposure

    #TODO: are these two methods only for Poisson? or also Negative Binomial?
    def predict(self, params, exog=None, exposure=None, offset=None,
                linear=False):
        """
        Predict response variable of a count model given exogenous variables.

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        #TODO: add offset tp
        if exog is None:
            exog = self.exog
            offset = getattr(self, 'offset', 0)
            exposure = getattr(self, 'exposure', 0)

        else:
            if exposure is None:
                exposure = 0
            else:
                exposure = np.log(exposure)
            if offset is None:
                offset = 0

        if not linear:
            return np.exp(np.dot(exog, params) + exposure + offset) # not cdf
        else:
            return np.dot(exog, params) + exposure + offset
            return super(CountModel, self).predict(params, exog, linear)

    def _derivative_exog(self, params, exog=None):
        """
        """
        # group 3 poisson, nbreg, zip, zinb
        if exog == None:
            exog = self.exog
        return self.predict(params, exog)[:,None] * params[None,:]

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        cntfit = super(CountModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        discretefit = CountResults(self, cntfit)
        return CountResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

class OrderedModel(DiscreteModel):
    pass

#### Public Model Classes ####

class Poisson(CountModel):
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
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        #np.sum(stats.poisson.logpmf(endog, np.exp(XB)))
        return np.sum(-np.exp(XB) +  endog*XB - gammaln(endog+1))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        The log likelihood for each observation of the model evaluated at `params`

        Notes
        --------
        .. math :: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        #np.sum(stats.poisson.logpmf(endog, np.exp(XB)))
        return -np.exp(XB) +  endog*XB - gammaln(endog+1)



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
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X,params) + offset + exposure)
        return np.dot(self.endog - L, X)

    def jac(self, params):
        """
        Poisson model Jacobian of the log-likelihood for each observation

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
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X,params) + offset + exposure)
        return (self.endog - L)[:,None] * X

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
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X,params) + exposure + offset)
        return -np.dot(L*X.T, X)

class NbReg(DiscreteModel):
    pass

class Logit(BinaryModel):
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

    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.

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
        return np.log(self.cdf(q*np.dot(X,params)))

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

    def jac(self, params):
        """
        Logit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        jac : ndarray, (nobs, k)
            The derivative of the loglikelihood evaluated at `params` for each
            observation

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """

        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X, params))
        return (y - L)[:,None] * X

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

class Probit(BinaryModel):
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
        return stats.norm._cdf(X)

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
        return stats.norm._pdf(X)


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
        return np.sum(np.log(np.clip(self.cdf(q*np.dot(X,params)),1e-20,
            1)))

    def loglikeobs(self, params):
        """
        Log-likelihood of probit model for each observation

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
        return np.log(np.clip(self.cdf(q*np.dot(X,params)), 1e-20, 1))


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
        # clip to get rid of invalid divide complaint
        L = q*self.pdf(q*XB)/np.clip(self.cdf(q*XB), 1e-20, 1-1e-20)
        return np.dot(L,X)

    def jac(self, params):
        """
        Probit model Jacobian for each observation

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
        # clip to get rid of invalid divide complaint
        L = q*self.pdf(q*XB)/np.clip(self.cdf(q*XB), 1e-20, 1-1e-20)
        return L[:,None] * X

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

class MNLogit(MultinomialModel):
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
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    J : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one.
    names : dict
        A dictionary mapping the column number in `wendog` to the variables
        in `endog`.
    wendog : array
        An n x j array where j is the number of unique categories in `endog`.
        Each column of j is a dummy variable indicating the category of
        each observation. See `names` for a dictionary mapping each column to
        its category.

    Notes
    -----
    See developer notes for further information on `MNLogit` internals.
    """
    def pdf(self, eXB):
        """
        NotImplemented
        """
        pass

    def cdf(self, X):
        """
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        X : array
            The linear predictor of the model XB.

        Returns
        --------
        The cdf evaluated at `XB`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        return eXB/eXB.sum(1)[:,None]

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
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog,params)))
        return np.sum(d * logprob)

    def loglikeobs(self, params):
        """
        Log-likelihood of the multinomial logit model for each observation.

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
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog,params)))
        return d * logprob

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
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:,1:] - self.cdf(np.dot(self.exog,
                                                  params))[:,1:]
        #NOTE: might need to switch terms if params is reshaped
        return np.dot(firstterm.T, self.exog).flatten()

    def jac(self, params):
        """
        Jabobian matrix for multinomial logit model log-likelihood

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
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:,1:] - self.cdf(np.dot(self.exog,
                                                  params))[:,1:]
        #NOTE: might need to switch terms if params is reshaped
        return (firstterm[:,:,None] * self.exog[:,None,:]).reshape(self.exog.shape[0], -1)

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
        :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0
        otherwise.

        The actual Hessian matrix has J**2 * K x K elements. Our Hessian
        is reshaped to be square (J*K, J*K) so that the solvers can use it.

        This implementation does not take advantage of the symmetry of
        the Hessian and could probably be refactored for speed.
        """
        params = params.reshape(self.K, -1, order='F')
        X = self.exog
        pr = self.cdf(np.dot(X,params))
        partials = []
        J = self.wendog.shape[1] - 1
        K = self.exog.shape[1]
        for i in range(J):
            for j in range(J): # this loop assumes we drop the first col.
                if i == j:
                    partials.append(\
                        -np.dot(((pr[:,i+1]*(1-pr[:,j+1]))[:,None]*X).T,X))
                else:
                    partials.append(-np.dot(((pr[:,i+1]*-pr[:,j+1])[:,None]*X).T,X))
        H = np.array(partials)
        # the developer's notes on multinomial should clear this math up
        H = np.transpose(H.reshape(J,J,K,K), (0,2,1,3)).reshape(J*K,J*K)
        return H


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

class NBin(CountModel):
    """
    Negative Binomial model.
    """
    #def pdf(self, X, alpha):
    #    a1 = alpha**-1
    #    term1 = special.gamma(X + a1)/(special.agamma(X+1)*special.gamma(a1))

    def _check_inputs(self, offset, exposure):
        if offset is not None or exposure is not None:
            raise ValueError("offset and exposure not implemented yet")

    def loglike(self, params):
        """
        Loglikelihood for negative binomial model

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """
        lnalpha = params[-1]
        params = params[:-1]
        a1 = np.exp(lnalpha)**-1
        y = self.endog
        J = special.gammaln(y+a1) - special.gammaln(a1) - special.gammaln(y+1)
        mu = np.exp(np.dot(self.exog,params))
        pdf = a1*np.log(a1/(a1+mu)) + y*np.log(mu/(mu+a1))
        llf = np.sum(J+pdf)
        return llf

    def loglikeobs(self, params):
        """
        Loglikelihood for negative binomial model

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """
        lnalpha = params[-1]
        params = params[:-1]
        a1 = np.exp(lnalpha)**-1
        y = self.endog
        J = special.gammaln(y+a1) - special.gammaln(a1) - special.gammaln(y+1)
        mu = np.exp(np.dot(self.exog,params))
        pdf = a1*np.log(a1/(a1+mu)) + y*np.log(mu/(mu+a1))
        llf = J + pdf
        return llf

    def score(self, params, full=False):
        """
        Score vector for NB2 model
        """
        lnalpha = params[-1]
        params = params[:-1]
        a1 = np.exp(lnalpha)**-1
        y = self.endog[:,None]
        exog = self.exog
        mu = np.exp(np.dot(exog,params))[:,None]
        dparams = exog*a1 * (y-mu)/(mu+a1)



        da1 = -1*np.exp(lnalpha)**-2
        dalpha = (special.digamma(a1+y) - special.digamma(a1) + np.log(a1)\
                        - np.log(a1+mu) - (a1+y)/(a1+mu) + 1)

        #multiply above by constant outside of the sum to reduce rounding error
        if full:
            return np.column_stack([dparams, dalpha])
        #JP: what's full, and why is there no da1?

        return np.r_[dparams.sum(0), da1*dalpha.sum()]

    def hessian(self, params):
        """
        Hessian of NB2 model.  Currently uses numdifftools
        """
        lnalpha = params[-1]
        params = params[:-1]
        a1 = np.exp(lnalpha)**-1

        exog = self.exog
        y = self.endog[:,None]
        mu = np.exp(np.dot(exog,params))[:,None]

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim+1,dim+1))
        const_arr = a1*mu*(a1+y)/(mu+a1)**2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i,j] = np.sum(-exog[:,i,None]*exog[:,j,None] *\
                                const_arr, axis=0)
        hess_arr[np.triu_indices(dim, k=1)] = hess_arr.T[np.triu_indices(dim,
                                                        k =1)]

        # for dl/dparams dalpha
        da1 = -1*np.exp(lnalpha)**-2
        dldpda = np.sum(mu*exog*(y-mu)*da1/(mu+a1)**2 , axis=0)
        hess_arr[-1,:-1] = dldpda
        hess_arr[:-1,-1] = dldpda

        # for dl/dalpha dalpha
        #NOTE: polygamma(1,x) is the trigamma function
        da2 = 2*np.exp(lnalpha)**-3
        dalpha = da1 * (special.digamma(a1+y) - special.digamma(a1) + \
                    np.log(a1) - np.log(a1+mu) - (a1+y)/(a1+mu) + 1)
        dada = (da2*dalpha/da1 + da1**2 * (special.polygamma(1,a1+y) - \
                    special.polygamma(1,a1) + 1/a1 -1/(a1+mu) + \
                    (y-mu)/(mu+a1)**2)).sum()
        hess_arr[-1,-1] = dada

        return hess_arr


    def fit(self, start_params=None, maxiter=35, method='bfgs', tol=1e-08):
        # start_params = [0]*(self.exog.shape[1])+[1]
        # Use poisson fit as first guess.
        start_params = Poisson(self.endog, self.exog).fit(disp=0).params
        start_params = np.r_[start_params, 0.1]
        mlefit = super(NegBinTwo, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        return mlefit


### Results Class ###

class DiscreteResults(base.LikelihoodModelResults):
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
    """

    def __init__(self, model, mlefit):
        #super(DiscreteResults, self).__init__(model, params,
        #        np.linalg.inv(-hessian), scale=1.)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = resettable_cache()
        self.nobs = model.exog.shape[0]
        self.__dict__.update(mlefit.__dict__)

    def __getstate__(self):
        try:
            #remove unpicklable callback
            self.mle_settings['callback'] = None
        except (AttributeError, KeyError):
            pass
        return self.__dict__

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
        model = self.model
        #TODO: what parameters to pass to fit?
        null = model.__class__(model.endog, np.ones(self.nobs)).fit(disp=0)
        return null.llf

    @cache_readonly
    def resid(self):
        model = self.model
        endog = model.endog
        exog = model.exog
        #        M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = the two individuals who share a covariate pattern
        # use unique row pattern?
        #TODO: is this common to all models?  logit uses Pearson, should have options
        #These are the deviance residuals
        M = 1
        p = model.predict(self.params)
        Y_0 = np.where(exog==0)
        Y_M = np.where(exog == M)
        res = np.zeros_like(endog)
        res = -(1-endog)*np.sqrt(2*M*np.abs(np.log(1-p))) + \
                endog*np.sqrt(2*M*np.abs(np.log(p)))
        return res

    @cache_readonly
    def fittedvalues(self):
        return np.dot(self.model.exog, self.params)

    @cache_readonly
    def aic(self):
        return -2*(self.llf - (self.df_model+1))

    @cache_readonly
    def bic(self):
        return -2*self.llf + np.log(self.nobs)*(self.df_model+1)

    def _get_endog_name(self, yname, yname_list):
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return yname, yname_list

    def margeff(self, at='overall', method='dydx', atexog=None, dummy=False,
            count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semielasticity -- dy/d(lnx)
            - 'eydx' - estimate semeilasticity -- d(lny)/dx

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
        #factor : None or dictionary, optional
        #    If a factor variable is present (it must be an integer, though
        #    of type float), then `factor` may be a dict with the zero-indexed
        #    column of the factor and the value should be the base-outcome.

        # get local variables
        model = self.model
        params = self.params
        method = method.lower()
        at = at.lower()
        exog = model.exog.copy() # copy because values are changed
        ind = exog.var(0) != 0 # index for non-constants

        _check_margeff_args(at, method)

        # handle discrete exogenous variables
        if dummy:
            _check_discrete_args(at, method)
            dummy_ind = _isdummy(exog)
        if count:
            _check_discrete_args(at, method)
            count_ind = _iscount(exog)

        # get the exogenous variables
        exog = _get_margeff_exog(exog, at, atexog, ind)

        # get base marginal effects, handled by sub-classes
        effects = model._derivative_exog(params, exog)

        if 'ex' in method:
            effects *= exog
        if 'ey' in method:
            effects /= model.predict(params, exog)[:,None]

        effects = _effects_at(effects, at, ind)

        if dummy == True:
            effects = _get_dummy_effects(effects, exog, dummy_ind, method,
                                         model, params)

        if count == True:
            effects = _get_count_effects(effects, exog, count_ind, method,
                                         model, params)

        # Set standard error of the marginal effects by Delta method.
        self.margfx_se = None
        self.margfx = effects
        return effects


    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
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

        top_left = [('Dep. Variable:', None),
                     ('Model:', [self.model.__class__.__name__]),
                     ('Method:', ['MLE']),
                     ('Date:', None),
                     ('Time:', None),
                     #('No. iterations:', ["%d" % self.mle_retvals['iterations']]),
                     ('converged:', ["%s" % self.mle_retvals['converged']])
                      ]

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Pseudo R-squ.:', ["%#6.4g" % self.prsquared]),
                     ('Log-Likelihood:', None),
                     ('LL-Null:', ["%#8.5g" % self.llnull]),
                     ('LLR p-value:', ["%#6.4g" % self.llr_pvalue])
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        #boiler plate
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        yname, yname_list = self._get_endog_name(yname, yname_list)
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, #[],
                          yname=yname, xname=xname, title=title)
        # for parameters, etc
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=.05,
                             use_t=False)

        #diagnostic table not used yet
        #smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
        #                   yname=yname, xname=xname,
        #                   title="")
        return smry


class CountResults(DiscreteResults):
    pass

class OrderedResults(DiscreteResults):
    pass

class BinaryResults(DiscreteResults):
    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
        smry = super(BinaryResults, self).summary(yname, xname, title, alpha,
                     yname_list)
        fittedvalues = self.model.cdf(self.fittedvalues)
        absprederror = np.abs(self.model.endog - fittedvalues)
        predclose_sum = (absprederror < 1e-4).sum()
        predclose_frac = predclose_sum / len(fittedvalues)

        #add warnings/notes
        etext = []
        if predclose_sum == len(fittedvalues): #nobs?
            wstr = "Complete Separation: The results show that there is"
            wstr += "complete separation.\n"
            wstr += "In this case the Maximum Likelihood Estimator does "
            wstr += "not exist and the parameters\n"
            wstr += "are not identified."
            etext.append(wstr)
        elif predclose_frac > 0.1:  #TODO: get better diagnosis
            wstr = "Possibly complete quasi-separation: A fraction "
            wstr += "%4.2f of observations can be\n" % predclose_frac
            wstr += "perfectly predicted. This might indicate that there "
            wstr += "is complete\nquasi-separation. In this case some "
            wstr += "parameters will not be identified."
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry
    summary.__doc__ = DiscreteResults.summary.__doc__

class MultinomialResults(DiscreteResults):
    def _maybe_convert_ynames_int(self, ynames):
        # see if they're integers
        try:
            for i in ynames:
                if ynames[i] % 1 == 0:
                    ynames[i] = str(int(ynames[i]))
        except TypeError:
            pass
        return ynames

    def _get_endog_name(self, yname, yname_list):
        model = self.model
        if yname is None:
            yname = model.endog_names
        if yname_list is None:
            ynames = model._ynames_map
            ynames = self._maybe_convert_ynames_int(ynames)
            # use range below to ensure sortedness
            ynames = [ynames[key] for key in range(int(model.J))]
            ynames = ['='.join([yname, name]) for name in ynames]
            yname_list = ynames[1:] # assumes first variable is dropped
        return yname, yname_list

    @cache_readonly
    def bse(self):
        bse = np.sqrt(np.diag(self.cov_params()))
        return bse.reshape(self.params.shape, order='F')

    @cache_readonly
    def aic(self):
        return -2*(self.llf - (self.df_model+self.model.J-1))

    @cache_readonly
    def bic(self):
        return -2*self.llf + np.log(self.nobs)*(self.df_model+self.model.J-1)

    def conf_int(self, alpha=.05, cols=None):
        confint = super(DiscreteResults, self).conf_int(alpha=alpha,
                                                            cols=cols)
        return confint.transpose(2,0,1)


#### Results Wrappers ####

class OrderedResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(OrderedResultsWrapper, OrderedResults)

class CountResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(CountResultsWrapper, CountResults)

class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(BinaryResultsWrapper, BinaryResults)

class MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(MultinomialResultsWrapper, MultinomialResults)


if __name__=="__main__":
    import numpy as np
    import statsmodels.api as sm
# Scratch work for negative binomial models
# dvisits was written using an R package, I can provide the dataset
# on request until the copyright is cleared up
#TODO: request permission to use dvisits
    data2 = np.genfromtxt('../datasets/dvisits/dvisits.csv', names=True)
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
    data = sm.datasets.randhie.load()
    nbreg = NBin
    mod = nbreg(data.endog, data.exog.view((float,9)))
#FROM STATA:
    params = np.asarray([-.05654133,  -.21214282, .0878311, -.02991813, .22903632,
            .06210226, .06799715, .08407035, .18532336])
    bse = [0.0062541, 0.0231818, 0.0036942, 0.0034796, 0.0305176, 0.0012397,
            0.0198008, 0.0368707, 0.0766506]
    lnalpha = .31221786
    mod.loglike(np.r_[params,np.exp(lnalpha)])
    poiss_res = Poisson(data.endog, data.exog.view((float,9))).fit()
    func = lambda x: -mod.loglike(x)
    grad = lambda x: -mod.score(x)
    from scipy import optimize
#    res1 = optimize.fmin_l_bfgs_b(func, np.r_[poiss_res.params,.1],
#                        approx_grad=True)
    res1 = optimize.fmin_bfgs(func, np.r_[poiss_res.params,.1], fprime=grad)
    from statsmodels.sandbox.regression.numdiff import approx_hess_cs
#    np.sqrt(np.diag(-np.linalg.inv(approx_hess_cs(np.r_[params,lnalpha], mod.loglike))))
#NOTE: this is the hessian in terms of alpha _not_ lnalpha
    hess_arr = mod.hessian(res1)


