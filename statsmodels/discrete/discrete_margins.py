#Splitting out maringal effects to see if they can be generalized

import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly, resettable_cache

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
    Given an array X, returns the column indices for the dummy variables.

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
    return np.where(ind)[0]

def _iscount(X):
    """
    Given an array X, returns the column indices for count variables.

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
    remainder = np.logical_and(np.logical_and(np.all(X % 1. == 0, axis = 0),
                               X.var(0) != 0), np.all(X >= 0, axis=0))
    dummy = _isdummy(X)
    remainder = np.where(remainder)[0].tolist()
    for idx in dummy:
        remainder.remove(idx)
    return np.array(remainder)

def _get_margeff_exog(exog, at, atexog, ind):
    if atexog is not None: # user supplied
        if isinstance(atexog, dict):
            # assumes values are singular or of len(exog)
            for key in atexog:
                exog[:,key] = atexog[key]
        elif isinstance(atexog, np.ndarray): #TODO: handle DataFrames
            if atexog.ndim == 1:
                nvars = len(atexog)
            else:
                nvars = atexog.shape[1]
            try:
                assert nvars == exog.shape[1]
            except:
                raise ValueError("atexog does not have the same number "
                        "of variables as exog")
            exog = atexog

    #NOTE: we should fill in atexog after we process at
    if at == 'mean':
        exog = np.atleast_2d(exog.mean(0))
    elif at == 'median':
        exog = np.atleast_2d(np.median(exog, axis=0))
    elif at == 'zero':
        exog = np.zeros((1,exog.shape[1]))
        exog[0,~ind] = 1
    return exog

def _get_count_effects(effects, exog, count_ind, method, model, params):
    """
    If there's a count variable, the predicted difference is taken by
    subtracting one and adding one to exog then averaging the difference
    """
    # this is the index for the effect and the index for count col in exog
    for i_count, i_exog in count_ind:
        exog0 = exog.copy()
        exog0[:,i_exog] -= 1
        effect0 = model.predict(params, exog0)
        exog0[:,i_exog] += 2
        effect1 = model.predict(params, exog0)
        #NOTE: done by analogy with dummy effects but untested bc
        # stata doesn't handle both count and eydx anywhere
        if 'ey' in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[i_count] = ((effect1 - effect0)/2).mean() # mean for overall
    return effects

def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    If there's a dummy variable, the predicted difference is taken at
    0 and 1
    """
    # this is the index for the effect and the index for dummy col in exog
    for i_dummy, i_exog in dummy_ind:
        exog0 = exog.copy() # only copy once, can we avoid a copy?
        exog0[:,i_exog] = 0
        effect0 = model.predict(params, exog0)
        #fittedvalues0 = np.dot(exog0,params)
        exog0[:,i_exog] = 1
        effect1 = model.predict(params, exog0)
        if 'ey' in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[i_dummy] = (effect1 - effect0).mean() # mean for overall
    return effects

def _effects_at(effects, at, ind):
    if at == 'all':
        effects = effects[:,ind]
    elif at == 'overall':
        effects = effects.mean(0)[ind]
    else:
        effects = effects[0,ind]
    return effects

def margeff_cov_params_dummy(model, cov_margins, params, exog, dummy_ind,
        method):
    """
    For discrete regressors the marginal effect is

    \Delta F = cdf(XB) | d = 1 - cdf(XB) | d = 0

    The row of the Jacobian for this variable is given by

    pdf(XB)*X | d = 1 - pdf(XB)*X | d = 0
    """
    for i_dummy, i_exog in dummy_ind:
        exog0 = exog.copy()
        exog1 = exog.copy()
        exog0[:,i_exog] = 0
        exog1[:,i_exog] = 1
        dfdb0 = model._derivative_predict(params, exog0, method)
        dfdb1 = model._derivative_predict(params, exog1, method)
        dfdb = (dfdb1 - dfdb0)
        if dfdb.ndim >= 2: # for overall
            dfdb = dfdb.mean(0)
        cov_margins[i_exog, :] = dfdb # how each F changes with change in B
    return cov_margins

def margeff_cov_params_count(model, cov_margins, params, exog, count_ind,
                             method):
    """
    For discrete regressors the marginal effect is

    \Delta F = cdf(XB) | d += 1 - cdf(XB) | d -= 1

    The row of the Jacobian for this variable is given by

    (pdf(XB)*X | d += 1 - pdf(XB)*X | d -= 1) / 2
    """
    for i_dummy, i_exog in count_ind:
        exog0 = exog.copy()
        exog0[:,i_exog] -= 1
        dfdb0 = model._derivative_predict(params, exog0, method)
        exog0[:,i_exog] += 2
        dfdb1 = model._derivative_predict(params, exog0, method)
        dfdb = (dfdb1 - dfdb0)
        if dfdb.ndim >= 2: # for overall
            dfdb = dfdb.mean(0) / 2
        cov_margins[i_exog, :] = dfdb # how each F changes with change in B
    return cov_margins

def margeff_cov_params(model, params, exog, cov_params, at, derivative,
                       dummy_ind, count_ind, method):
    """
    Computes the variance-covariance of marginal effects by the delta method.

    Parameters
    ----------
    model : model instance
        The model that returned the fitted results. Its pdf method is used
        for computing the Jacobian of discrete variables in dummy_ind and
        count_ind
    params : array-like
        estimated model parameters
    exog : array-like
        exogenous variables at which to calculate the derivative
    cov_params : array-like
        The variance-covariance of the parameters
    at : str
       Options are:

        - 'overall', The average of the marginal effects at each
          observation.
        - 'mean', The marginal effects at the mean of each regressor.
        - 'median', The marginal effects at the median of each regressor.
        - 'zero', The marginal effects at zero for each regressor.
        - 'all', The marginal effects at each observation.

        Only overall has any effect here.

    derivative : function or array-like
        If a function, it returns the marginal effects of the model with
        respect to the exogenous variables evaluated at exog. Expected to be
        called derivative(params, exog). This will be numerically
        differentiated. Otherwise, it can be the Jacobian of the marginal
        effects with respect to the parameters.
    dummy_ind : array-like
        Indices of the columns of exog that contain dummy variables
    count_ind : array-like
        Indices of the columns of exog that contain count variables

    Notes
    -----
    For continuous regressors, the variance-covariance is given by

    Asy. Var[MargEff] = [d margeff / d params] V [d margeff / d params]'

    where V is the parameter variance-covariance.

    The outer Jacobians are computed via numerical differentiation if
    derivative is a function.
    """
    if callable(derivative):
        from statsmodels.sandbox.regression.numdiff import approx_fprime_cs
        params = params.ravel(order='F') # for Multinomial
        jacobian_mat = approx_fprime_cs(params, derivative, args=(exog,method))
        if at == 'overall':
            jacobian_mat = np.mean(jacobian_mat, axis=1)
        else:
            jacobian_mat = jacobian_mat.squeeze() # exog was 2d row vector
        if dummy_ind is not None:
            jacobian_mat = margeff_cov_params_dummy(model, jacobian_mat, params,
                                exog, dummy_ind, method)
        if count_ind is not None:
            jacobian_mat = margeff_cov_params_count(model, jacobian_mat, params,
                                exog, count_ind, method)
    else:
        jacobian_mat = derivative

    #NOTE: this won't go through for at == 'all'
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)

def margeff_cov_with_se(model, params, exog, cov_params, at, derivative,
                        dummy_ind, count_ind, method):
    """
    See margeff_cov_params.

    Same function but returns both the covariance of the marginal effects
    and their standard errors.
    """
    cov_me = margeff_cov_params(model, params, exog, cov_params, at,
                                              derivative, dummy_ind,
                                              count_ind, method)
    return cov_me, np.sqrt(np.diag(cov_me))

def margeff():
    pass

_transform_names = dict(dydx='dy/dx',
                        eyex='d(lny)/d(lnx)',
                        dyex='dy/d(lnx)',
                        eydx='d(lny)/dx')

class Margins(object):
    """
    Mostly a do nothing class. Lays out the methods expected of a sub-class.

    This is just a sketch of what we may want out of a general margins class.
    I (SS) need to look at details of other models.
    """
    def __init__(self, results, get_margeff, derivative, dist=None,
                       margeff_args=()):
        self._cache = resettable_cache()
        self.results = results
        self.dist = dist
        self._get_margeff = get_margeff
        self.get_margeff(margeff_args)

    def _reset(self):
        self._cache = resettable_cache()

    def get_margeff(self, *args, **kwargs):
        self._reset()
        self.margeff = self._get_margeff(*args)

    @cache_readonly
    def tvalues(self):
        raise NotImplementedError

    @cache_readonly
    def cov_margins(self):
        raise NotImplementedError

    @cache_readonly
    def margins_se(self):
        raise NotImplementedError

    def get_frame(self):
        raise NotImplementedError

    @cache_readonly
    def pvalues(self):
        raise NotImplementedError

    def conf_int(self, alpha=.05):
        raise NotImplementedError

    def summary(self, alpha=.05):
        raise NotImplementedError

#class DiscreteMargins(Margins):
class DiscreteMargins(object):
    def __init__(self, results, args=()):
        self._cache = resettable_cache()
        self.results = results
        self.get_margeff(*args)

    def _reset(self):
        self._cache = resettable_cache()

    def get_margeff(self, *args, **kwargs):
        self._reset()
        self.margeff = self._get_margeff(*args)

    @cache_readonly
    def tvalues(self):
        return self.margeff / self.margeff_se

    def get_frame(self, alpha=.05):
        from pandas import DataFrame
        names = [_transform_names[self.margeff_options['method']],
                                  'Std. Err.', 'z', 'Pr(>|z|)',
                                  'Conf. Int. Low', 'Cont. Int. Hi.']
        ind = self.results.model.exog.var(0) != 0 # True if not a constant
        exog_names = self.results.model.exog_names
        var_names = [name for i,name in enumerate(exog_names) if ind[i]]
        table = np.column_stack((self.margeff, self.margeff_se, self.tvalues,
                                 self.pvalues, self.conf_int(alpha)))
        return DataFrame(table, columns=names, index=var_names)

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.tvalues)) * 2

    def conf_int(self, alpha=.05):
        me_se = self.margeff_se
        q = norm.ppf(1 - alpha / 2)
        lower = self.margeff - q * me_se
        upper = self.margeff + q * me_se
        return np.asarray(zip(lower, upper))

    def summary(self, alpha=.05):
        raise NotImplementedError

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False,
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
        self._reset() # always reset the cache when this is called
        #TODO: if at is not all or overall, we can also put atexog values
        # in summary table head
        method = method.lower()
        at = at.lower()
        self.margeff_options = dict(method=method, at=at)

        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy() # copy because values are changed
        ind = exog.var(0) != 0 # index for non-constants

        _check_margeff_args(at, method)

        if np.any(~ind):
            const_idx = np.where(~ind)[0]
        else:
            const_idx = None

        # handle discrete exogenous variables
        if dummy:
            _check_discrete_args(at, method)
            dummy_ind = _isdummy(exog)
            exog_ind = dummy_ind.copy()
            # adjust back for a constant because effects doesn't have one
            if const_idx is not None:
                dummy_ind[dummy_ind > const_idx] -= 1
            if dummy_ind.size == 0: # don't waste your time
                dummy = False
                dummy_ind = None # this gets passed to stand err func
            else:
                dummy_ind = zip(dummy_ind, exog_ind[:])
        else:
            dummy_ind = None

        if count:
            _check_discrete_args(at, method)
            count_ind = _iscount(exog)
            exog_ind = count_ind.copy()
            # adjust back for a constant because effects doesn't have one
            if const_idx is not None:
                count_ind[count_ind > const_idx] -= 1
            if count_ind.size == 0: # don't waste your time
                count = False
                count_ind = None # for stand err func
            else:
                count_ind = zip(count_ind, exog_ind)
        else:
            count_ind = None

        # get the exogenous variables
        exog = _get_margeff_exog(exog, at, atexog, ind)

        # get base marginal effects, handled by sub-classes
        effects = model._derivative_exog(params, exog, method)

        J = getattr(model, 'J', 1)
        ind = np.tile(ind, J) # adjust for multi-equation.
        effects = _effects_at(effects, at, ind)

        if dummy:
            effects = _get_dummy_effects(effects, exog, dummy_ind, method,
                                         model, params)

        if count:
            effects = _get_count_effects(effects, exog, count_ind, method,
                                         model, params)

        # Set standard error of the marginal effects by Delta method.
        margeff_cov, margeff_se = margeff_cov_with_se(model, params, exog,
                                                results.cov_params(), at,
                                                model._derivative_exog,
                                                dummy_ind, count_ind,
                                                method)

        # reshape for multi-equation
        if J > 1:
            K = model.K - np.any(ind) # subtract constant
            self.margeff = effects.reshape(K, J, order='F')
            self.margeff_se = margeff_se[ind].reshape(K, J, order='F')
            self.margeff_cov = margeff_cov[ind][:, ind]
        else:

            # don't care about at constant
            self.margeff_cov = margeff_cov[ind][:, ind]
            self.margeff_se = margeff_se[ind]
            self.margeff = effects

