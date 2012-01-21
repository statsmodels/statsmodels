import numpy as np
import scikits.statsmodels.base.model as base
import scikits.statsmodels.base.wrapper as wrap
from scikits.statsmodels.tools.decorators import (cache_readonly, transform2,
            set_transform, unset_transform)
from scikits.statsmodels.sandbox.regression.numdiff import (approx_hess_cs,
    approx_fprime_cs, approx_hess, approx_fprime)
from scipy.stats import norm

#TODO: clean-up estimation, delete webuse, write specific results class,
#      figure out good separation for params and scale, can't concentrate
#      likelihood, generalize margins from discrete for use here as well

#class LogLike(object):
#    """
#    The Loglikelihood parametrization due to Olsen (1978).
#
#    Concentrates out \sigma and will allow easier score and hessian
#    calculations. The results are similar to truncated regression cf.
#    Greene 19.3.3, 7th edition.
#    """

def _olsen_reparam(params):
    """
    Go from true parameters to gamma and theta of Olsen.

    gamma = beta/sigma
    theta = 1/sigma
    """
    beta, sigma  = params[:-1], params[-1]
    theta = 1./sigma
    gamma = beta*theta
    return np.r_[gamma, theta]

def _reparam_olsen(params):
    """
    Go from gamma and theta of Olsen to beta and sigma

    sigma = 1/theta
    beta = gamma/theta
    """
    gamma, theta  = params[:-1], params[-1]
    sigma = 1./theta
    beta = gamma*sigma
    return np.r_[beta, sigma]

def inverse_mills_above(X):
    """
    Return the inverse Mills ratio for truncation from below.

    Parameters
    ----------
    X : array
        Argument to inverse Mills ratio
        
    Returns
    -------
    imr : array
        norm.pdf(X)/norm.cdf(X)
    """
    return (-norm.pdf(X)/norm.cdf(X))

def inverse_mills_below(X):
    """
    Return the inverse Mills ratio for truncation from above.

    Parameters
    ----------
    X : array
        Argument to inverse Mills ratio
        
    Returns
    -------
    imr : array
        norm.pdf(X)/norm.cdf(X)

    Notes
    -----
    This is also the hazard function for the standard normal distribution.
    """
    return norm.pdf(X)/(1-norm.cdf(X))

def _null_params(model):
    """
    Returns maximum liklihood estimates of the null (constant-only model)
    """
    endog = model.endog
    left = model.left
    right = model.right
    res = Tobit(endog, np.ones_like(endog), left=left,
                right=right).fit(method='bfgs', start_params = [0.,endog.std()],
                        disp=0)
    k_zeros = model.exog.shape[1] - 1
    params = np.r_[[0] * k_zeros, res.params]
    params[-1] = np.log(params[-1])
    #params = _olsen_reparam(params)
    return params

def _olsmle_params(model):
    """
    Does OLS on full endog and exog then scales the estimates by the
    number of non-limit observations.

    Notes
    -----
    This is just a cute trick based on an "empirical regularity" reported in
    Greene 7th edition.
    """
    ols_res = sm.OLS(model.endog, model.exog).fit()
    params = ols_res.params
    sigma = ols_res.scale ** .5
    n_goodobs = len(self._center_endog)
    params = params/n_goonobs
    #return _olsen_reparam(np.r_[params, sigma])
    return np.r_[params, np.log(sigma)]

def _ols_params(model):
    """
    Does OLS on uncensored endog and exog and returns the parameters.

    Notes
    -----
    Shouldn't use this because consistency for Tobit is only guaranteed
    if the estimator for the starting values is consistent. Cf. Olsen's and
    Amemiya's papers.
    """
    pass

_start_param_funcs = {'null' : _null_params, 'olsmle' : _olsmle_params,
                'ols' : _ols_params}

def _predict_xb(model, ab_args, params, sigma, exog):
    """
    """
    return np.dot(exog, params)

def _predict_pr(model, ab_args, params, sigma, exog):
    """
    Probability of being uncensored.
    """
    a,b = ab_args
    mu = _predict_xb(model, (), params, sigma, exog)
    return norm.cdf((mu - a)/sigma) - norm.cdf((mu - b)/sigma)

def _predict_e(model, ab_args, params, sigma, exog):
    """
    Conditional on being uncensored
    """
    a,b = ab_args
    mu = _predict_xb(model, (), params, sigma, exog)
    if np.isfinite(a) and np.isfinite(b):
        alpha_a = (a-mu)/sigma
        alpha_b = (b-mu)/sigma
        return mu + sigma * (norm.pdf(alpha_a) - norm.pdf(alpha_b))/ \
                 (norm.cdf(alpha_b) - norm.cdf(alpha_a))
    elif np.isfinite(a): # from below only
        alpha = (a - mu)/sigma
        return mu + sigma * inverse_mills_below(alpha)
    elif np.isfinite(b): # from above only
        alpha = (b - mu)/sigma
        return mu + sigma * inverse_mills_above(alpha)
    else: # both infinite
        return mu

def _predict_ystar(model, ab_args, params, sigma, exog):
    """
    Unconditional expected value
    """
    a,b = ab_args
    e = _predict_e(model, ab_args, params, sigma, exog)
    pr = _predict_pr(model, ab_args, params, sigma, exog)
    if np.isfinite(a) and np.isfinite(b): 
        pra = _predict_pr(model, (-np.inf,a), params, sigma, exog)
        return pr * e + pra * a + (1 - pr - pra) * b
    elif np.isfinite(a):
        return e*pr + (1-pr) * a
    elif np.isfinite(b):
        return e*pr + (1-pr) * b
    else:    
        return _predict_xb(model, (), params, sigma, exog)

_predict_funcs = dict(xb    = _predict_xb, 
                      pr    = _predict_pr,
                      e     = _predict_e,
                      ystar = _predict_ystar)

def _clean_type(typ):
    args = ()
    if typ == 'xb':
        return typ, args

    import re
    # match a,b
    ab_pattern = "\(.*,.*\)"
    ab = re.search(ab_pattern, typ)
    if ab:
        ab = ab.group()[1:-1] # drop parens
        args = tuple(map(float, ab.split(",")))

    # match the predict type
    pr_pattern = "^pr"
    e_pattern = "^e"
    ystar_pattern = "^ystar"

    if re.search(pr_pattern, typ):
        return "pr", args
    elif re.search(e_pattern, typ):
        return "e", args
    elif re.search(ystar_pattern, typ):
        return "ystar", args
    else:
        raise ValueError("Predict type %s not understood" % typ)

def _exponentiate_sigma(params):
    params = params.copy() # do this or it gets modified several times
    params[-1] = np.exp(params[-1])
    return params

#####        End Helper Functions        #####


##### Tobit Model in transformed parameter space #####

class TobitOlsen(base.LikelihoodModel):
    def __init__(self, endog, exog, left=True, right=True):
        super(Tobit, self).__init__(endog, exog)
        # set up censoring
        self._transparams = False # need to exp(sigma) to keep positive in fit
        self._init_censored(left, right)
        self._set_loglike()

    def _init_censored(self, left, right):
        if left is False and right is False:
            raise ValueError("Must give a censoring level on the right or "
                             "left.")
        endog = self.endog
        exog = self.exog
        if left is True:
            left = self.endog.min()
        elif left is False:
            left = -np.inf
        if right is True:
            right = self.endog.max()
        elif right is False:
            right = np.inf
        self.left, self.right = left, right

        # do this all here, so that we only have to do the indexing once
        left_idx = endog <= left # do we need to attach these?
        right_idx = endog >= right
        self._left_endog = endog[left_idx]
        self.n_lcens = left_idx.sum()
        self.n_rcens = right_idx.sum()
        self._right_endog = endog[right_idx]
        self._left_exog = exog[left_idx]
        self._right_exog = exog[right_idx]
        center_idx = ~np.logical_or(left_idx, right_idx)
        self._center_endog = endog[center_idx]
        self._center_exog = exog[center_idx]

    def _set_loglike(self):
        """
        Use this so we don't have a lot of if checks in each loglike call
        """
        left, right = self.left, self.right
        if np.isfinite(left) and np.isinf(right): # left-censored
            self.loglike_ = self._loglike_olsen_left
            self.score_ = self._score_left
            self.hessian_ = self._hessian_left_olsen
        elif np.isinf(left) and np.isfinite(right): # right-censored
            self.loglike_ = self._loglike_right
            #self.score = self._score_right
            #self.hessian = self._hessian_right
        else: # left and right censored
            self.loglike_ = self._loglike_both
            #self.score = self._score_both
            #self.hessian = self._hessian_both

    #implemented here, so we don't have to repeat reparams stuff
    #@transform2(_olsen_reparam)
    def loglike(self, params):
        params = params.copy()
        params[-1] = np.exp(params[-1])
        params = _olsen_reparam(params)
        return self.loglike_(params)

    def score(self, params):
        params = params.copy()
        params[-1] = np.exp(params[-1])
        params = _olsen_reparam(params)
        return self.score_(params)


    #@transform2(_olsen_reparam)
    def _score_left(self, params):
        #return approx_fprime(params, self.loglike)
        gamma, theta = params[:-1], params[-1]

        # score for gamma
        left, left_exog = self.left, self._left_exog
        imr = -inverse_mills_above(left*theta - np.dot(left_exog, gamma))
        gamma_left = np.dot(left_exog.T, imr)

        center_endog = self._center_endog * theta
        center_exog = self._center_exog
        center_resid = (center_endog - np.dot(center_exog, gamma))
        gamma_center = np.dot(center_exog.T, center_resid)
        dLdGamma = gamma_center - gamma_left

        # score for theta
        theta_left = np.sum(left * imr)
        theta_center = np.sum(1/theta - center_resid * self._center_endog)
        dLdTheta = theta_center + theta_left
        return np.r_[dLdGamma, dLdTheta]

    #@transform2(_olsen_reparam)
    def hessian(self, params):
        # this is the hessian for left-censored at zero
        #gamma, theta = params[:-1], params[-1]
        #predicted = -np.dot(self.exog, gamma)
        #fF = norm.pdf(predicted)/norm.cdf(predicted)
        #D = fF * (-predicted - fF)
        loglike = self.loglike
        return approx_hess(params, loglike, epsilon=1e-4)[0]

    def _hessian_left_olsen(self, params):
        gamma, theta = params[:-1], params[-1]

        # get set up
        center_endog = self._center_endog
        center_exog = self._center_exog
        left, left_exog = self.left, self._left_exog
        left_resid = left*theta - np.dot(left_exog, gamma)
        imr = -inverse_mills_above(left_resid)[:,None]

        # dLdGamma2 for uncensored
        dLdGamma2_center = -np.dot(center_exog.T, center_exog)
        # dLdGamma2 for censored
        dLdGamma2_left = np.dot((left_exog * imr * (left_resid[:,None] + imr)).T,
                left_exog)
        # dLdGamma2
        dLdGamma2 = dLdGamma2_center - dLdGamma2_left

        # dLdGammadTheta for uncensored
        dLdGammadTheta_center = np.dot(center_exog.T, center_endog)
        # dLdGammadTheta for censored
        dLdGammadTheta_left = np.dot((imr * left * (left_resid[:,None] + imr)).T,
                                left_exog)
        # dLdGammadTheta
        dLdGammadTheta = dLdGammadTheta_center + dLdGammadTheta_left


        # dLdTheta2 for uncensored
        dLdTheta2_center = -(np.dot(center_endog, center_endog) - \
                            len(center_endog)/theta**2)
        # dLdTheta2 for censored
        dLdTheta2_left = np.sum(imr * left ** 2 * ( 1 - imr))
        dLdTheta2 = dLdTheta2_center + dLdTheta2_left

        # put it all together
        k = center_exog.shape[1] + 1
        hess = np.zeros((k,k))
        hess[:-1,:-1] = dLdGamma2
        hess[-1,:-1] = dLdGammadTheta
        hess[:-1,-1] = dLdGammadTheta
        hess[-1,-1] = dLdTheta2
        return hess



    def _loglike_left(self, params, sigma):
        left_exog = (self.left - np.dot(self._left_exog, params)) / sigma
        left_like = np.sum(norm.logcdf(left_exog))
            # can get overflow from the above if cdf is very small
            # seems to mean bad starting values, but might need a check
            #if np.isinf(left_like) and left_like < 0:
            #    left_like = -1e4
        return left_like

    def _loglike_right(self, params, sigma):
        right_exog = (np.dot(self._right_exog, params) - self.right) / sigma
        right_like = np.sum(norm.logcdf(right_exog))
        #if np.isinf(right_like) and left_like < 0:
        #    left_like = -1e4
        return right_like

    def _loglike_olsen_left(self, params):
        gamma, theta = params[:-1], params[-1]

        left, left_exog = self.left, self._left_exog
        llf_left = np.sum(norm.logcdf(left*theta - np.dot(left_exog, gamma)))

        center_endog = self._center_endog * theta
        center_exog = self._center_exog
        llf_center = -.5 * np.sum((np.log(2*np.pi) - np.log(theta**2) + \
                     (center_endog - np.dot(center_exog, gamma))**2))
        return llf_center + llf_left

    def _loglike_olsen_right(self, params):
        gamma, theta = params[:-1], params[-1]

        right = self.right
        center_endog = self._center_endog * theta
        center_exog = self._center_exog
        llf_center = -.5 * np.sum((np.log(2*np.pi) - np.log(theta**2) + \
                     (center_endog - np.dot(center_exog, gamma))**2))
        right_exog = self._right_exog
        llf_right = np.sum(norm.logcdf(np.dot(right_exog, gamma)-right*theta))
        return llf_right + llf_center

    #def loglike_(self, params): #NOTE: these needs sigma as last parameter
    #    sigma = params[-1]
    #    if self._transparams:
    #        sigma = np.exp(sigma)
    #    params = params[:-1]

    #    left_like = 0
    #    if self._left_exog.size:
    #        left_like = self._loglike_left(params, sigma)

    #    right_like = 0
    #    if self._right_exog.size:
    #        right_like = self._loglike_right(params, sigma)

    #    center_endog = self._center_endog
    #    center_exog = self._center_exog
    #    center_like = np.sum(norm.logpdf((center_endog - np.dot(center_exog,
    #                               params)) / sigma) - np.log(sigma))
    #    llf = left_like + right_like + center_like
    #    return llf

    def _get_start_params(self, method='null'):
        """
        Get starting parameters from an OLS estimation

        Parameters
        ----------
        method : str {'null', 'olsmle', 'ols'}
            The method to use to get the starting values. It is recommended
            to use the default. See the private functions _null_params,
            _olsmle_params, and _ols_params for more information.
        """
        return _start_param_funcs[method](self)

    #@unset_transform # likelihood expects not to transform
    def fit(self, start_params = None, method='newton', **kwargs):
        """
        Notes
        -----
        It is not recommended to change the default solver or starting
        parameters. Please report performance issues to the developers.
        """
        if start_params is None:
            start_params = self._get_start_params()
        mlefit = super(Tobit, self).fit(start_params=start_params,
                        method=method, **kwargs)
        return TobitResults(self, mlefit)

#####       Tobit Log-likelihood functions    #####

def _loglike_uncens_obs(self, params, sigma):
    """
    Compute the loglikelihood for the uncensored observations only.
    """
    center_endog = self._center_endog
    center_exog = self._center_exog
    center_like = np.sum(norm.logpdf((center_endog - np.dot(center_exog,
                                    params)) / sigma) - np.log(sigma))
    return center_like

def _score_uncens_obs(self, params, sigma):
    """
    Computes the score for the uncensored observations only.
    """
    endog = self._center_endog
    exog = self._center_exog
    scaled_center_exog = (endog - np.dot(exog, params))/sigma
    dLdB_center = np.sum(exog/sigma * scaled_center_exog[:,None], axis=0)
    dLdSigma_center = np.sum(scaled_center_exog**2 - 1, axis=0)
    return np.r_[dLdB_center, dLdSigma_center]

    #####       Left-censored stuff

def _loglike_left_cens_obs(self, params, sigma):
    """
    Computes the log-likelihood for the left-censored observations only.
    """
    alpha = (self.left - np.dot(self._left_exog, params)) / sigma
    left_like = np.sum(norm.logcdf(alpha))
    # can get overflow from the above if cdf is very small
    return left_like

def _score_left_cens_obs(self, params, sigma):
    """
    Computes the score for the left-censored observations only
    """
    left_exog = self._left_exog
    alpha = (self.left - np.dot(left_exog, params)) / sigma
    imr = -inverse_mills_above(alpha)
    dLdB_left = -np.sum(left_exog/sigma * imr[:,None], axis=0)
    dLdSigma_left = -np.sum(imr * alpha, axis=0)
    score_l = np.r_[dLdB_left, dLdSigma_left]
    return score_l


    #####       Right-censored stuff

def _loglike_right_cens_obs(self, params, sigma):
    """
    Computes the log-likelihood for the right-censored observations only.
    """
    alpha = (np.dot(self._right_exog, params) - self.right) / sigma
    right_like = np.sum(norm.logcdf(alpha))
    return right_like

def _score_right_cens_obs(self, params, sigma):
    right_exog = self._right_exog
    alpha = (np.dot(right_exog, params) - self.right) / sigma
    imr = -inverse_mills_above(alpha)
    dLdB_right = np.sum(right_exog/sigma * imr[:,None], axis=0)
    dLdSigma_right = -np.sum(alpha * imr, axis=0)
    return np.r_[dLdB_right, dLdSigma_right]

#### Model classes

class Tobit(base.LikelihoodModel):
    """
    Tobit model for limited dependent variable regression.

    Parameters
    ----------
    endog : array-like
        Endogenous variable
    exog : array-like
        Exogenous variable(s)
    left : bool or float
        If True, censoring is assumed to be at the lowest value of endog.
        If False, no censoring in the left tail of the distribution.
        If a floating point value is given, this is the censoring level on
        the left.
    right : bool or float
        If True, censoring is assumed to be at the highest value of endog.
        If False, no censoring in the right tail of the distribution.
        If a floating point value is given, this is the censoring level on 
        the right.

    Notes
    -----
    ystar is the observed variable, that proxies for the unobserved latent
    variable y.

              y, if a < y < b
    ystar = { a, if y <= a
              b, if y >= b
    """
    def __init__(self, endog, exog, left=True, right=False):
        super(Tobit, self).__init__(endog, exog)
        self._transparams = False
        # set up censoring
        self._init_censored(left, right)
        self._set_loglike()

    def _init_censored(self, left, right):
        if left is False and right is False:
            raise ValueError("Must give a censoring level on the right or "
                             "left.")
        endog = self.endog
        exog = self.exog
        if left is True:
            left = self.endog.min()
        elif left is False:
            left = -np.inf
        if right is True:
            right = self.endog.max()
        elif right is False:
            right = np.inf
        self.left, self.right = left, right

        # do this all here, so that we only have to do the indexing once
        left_idx = endog <= left # do we need to attach these?
        right_idx = endog >= right
        self._left_endog = endog[left_idx]
        self.n_lcens = left_idx.sum()
        self.n_rcens = right_idx.sum()
        self._right_endog = endog[right_idx]
        self._left_exog = exog[left_idx]
        self._right_exog = exog[right_idx]
        center_idx = ~np.logical_or(left_idx, right_idx)
        self._center_endog = endog[center_idx]
        self._center_exog = exog[center_idx]

    def _set_loglike(self):
        """
        Use this so we don't have a lot of if checks in each loglike call
        """
        left, right = self.left, self.right
        if np.isfinite(left) and np.isinf(right): # left-censored
            self._loglike = self._loglike_left
            self._score = self._score_left
            #self.hessian_ = self._hessian_left
        elif np.isinf(left) and np.isfinite(right): # right-censored
            self._loglike = self._loglike_right
            self._score = self._score_right
            #self.hessian = self._hessian_right
        else: # left and right censored
            self._loglike = self._loglike_both
            self._score = self._score_both
            #self.hessian = self._hessian_both

    @transform2(_exponentiate_sigma) # keeps std. dev. positive
    def loglike(self, params):
        """
        Notes
        -----
        Assumes sigma is the last parameter
        """
        params, sigma = params[:-1], params[-1]
        return self._loglike(params, sigma)

    @transform2(_exponentiate_sigma)
    def score(self, params):
        params, sigma = params[:-1], params[-1]
        return self._score(params, sigma)

    def hessian(self, params):
        # this is the hessian for left-censored at zero
        loglike = self.loglike
        return approx_hess(params, loglike, epsilon=1e-4)[0]

    def _score_left(self, params, sigma):
        score_l = _score_left_cens_obs(self, params, sigma)
        score_u = _score_uncens_obs(self, params, sigma)
        return score_l + score_u

    def _loglike_left(self, params, sigma):
        """
        Computes the log-likelihood for the left-censored model.
        """
        llf_l = _loglike_left_cens_obs(self, params, sigma)
        llf_u = _loglike_uncens_obs(self, params, sigma)
        return llf_l + llf_u

    def _score_right(self, params, sigma):
        score_r = _score_right_cens_obs(self, params, sigma)
        score_u = _score_uncens_obs(self, params, sigma)
        return score_r + score_u

    def _loglike_right(self, params, sigma):
        """
        Computes the log-likelihood for the right-censored model.
        """
        llf_r = _loglike_right_cens_obs(self, params, sigma)
        llf_u = _loglike_uncens_obs(self, params, sigma)
        return llf_r + llf_u

    def _score_both(self, params, sigma):
        score_l = _score_left_cens_obs(self, params, sigma)
        score_r = _score_right_cens_obs(self, params, sigma)
        score_u = _score_uncens_obs(self, params, sigma)
        return score_l + score_r + score_u

    def _loglike_both(self, params, sigma):
        llf_l = _loglike_left_cens_obs(self, params, sigma)
        llf_r = _loglike_right_cens_obs(self, params, sigma)
        llf_u = _loglike_uncens_obs(self, params, sigma)
        return llf_l + llf_r + llf_u

    def _get_start_params(self, method='null'):
        """
        Get starting parameters from an OLS estimation

        Parameters
        ----------
        method : str {'null', 'olsmle', 'ols'}
            The method to use to get the starting values. It is recommended
            to use the default. See the private functions _null_params,
            _olsmle_params, and _ols_params for more information.
        """
        return _start_param_funcs[method](self)

    def predict(self, params, exog=None, typ='xb'):
        """
        Use the Tobit model for prediction

        Parameters
        ----------
        params : array
            The parameters of the model and sigma. The last number in params 
            should be sigma, the standard deviation of the model.
        exog : array, optional
            The exogenous variables at which to predict. The default is the
            given exog for the model.
        typ : str {'xb', 'pr(a,b)', 'e(a,b)', 'ystar(a,b)'}, optional
            The type of prediction to make.
            'xb'         - linear prediction
            'pr(a,b)'    - Probability that a < y < b. Where y is the latent
                           unobserved variable. a and b may be replaced 
                           by floats in the string. a and b may be replaced by
                           -inf or inf, respectively. If (a,b) is omitted 
                           entirely, they are inferred from the model.
            'e(a,b)'     - The expectation of y given a < y < b. Where y is the
                           latent unobserved variable. a and b may be replaced 
                           by floats in the string. a and b may be replaced by
                           -inf or inf, respectively. If (a,b) is omitted 
                           entirely, they are inferred from the model.
            'ystar(a,b)' - Expectation of the observed variable ystar where
                           ystar = max(a, min(y,b)). a and b may be replaced by
                           -inf or inf, respectively. If (a,b) is omitted 
                           entirely, they are inferred from the model.
        """
        typ, ab_args = _clean_type(typ)
        params, sigma = params[:-1], params[-1]
        if exog == None:
            exog = self.exog
        if not ab_args:
            ab_args = (self.left, self.right)
        return _predict_funcs[typ](self, ab_args, params, sigma, exog)

    @set_transform
    def fit(self, start_params = None, method='newton', **kwargs):
        """
        Fit the Tobit model

        Parameters
        ----------

        Notes
        -----
        It is not recommended to change the default solver or starting
        parameters.
        """
        if start_params is None:
            start_params = self._get_start_params()
        else:
            start_params = np.copy(start_params)

        mlefit = super(Tobit, self).fit(start_params=start_params,
                        method=method, **kwargs)
        return TobitResults(self, mlefit)

class TobitResults(base.LikelihoodModelResults):
    def __init__(self, model, mlefit):
        self.model = model
        params = mlefit.params
        # re-parameterize from exponential
        params[-1] = np.exp(params[-1])
        # re-parameterize from Olsen
        #params = _reparam_olsen(params)
        beta, sigma = params[:-1], params[-1]
        self.model_params = beta
        mlefit.params = np.r_[beta, sigma]
        self.__dict__.update(mlefit.__dict__)
        self.scale = sigma # overwrite from mlefit
        self._model_stats() # attach model statistics

    def _model_stats(self):
        model = self.model
        exog = model.exog
        self.nobs = nobs = exog.shape[0]
        self.n_lcens = model.n_lcens
        self.n_rcens = model.n_rcens
        self.n_ucens = nobs - self.n_lcens - self.n_rcens
        self.k_constant = k_constant = int(np.any(exog.var(0) == 0))
        self.k_params = k_params = exog.shape[1] - k_constant
        self.k_ancillary = 1 # number of ancillary parameters, sigma here
        self.df_model = k_params - self.k_constant + 1 # for sigma
        self.df_resid = nobs - self.df_model

    def cov_params(self):
        hess = self.model.hessian(self.params)
        return np.linalg.inv(-hess)

    @cache_readonly
    def chi2(self):
        return 2 * (self.llf-self.llnull)

    @cache_readonly
    def chi2_pvalue(self):
        from scipy.stats import chi2
        #NOTE: df == number of restrictions
        return chi2.sf(self.chi2, self.df_model)

    @cache_readonly
    def llnull(self):
        endog = self.model.endog
        left = self.model.left
        right = self.model.right
        res = Tobit(endog, np.ones_like(endog), left=left,
                     right=right).fit(method='bfgs', disp=0)
        try:
            assert not res.mle_retvals['warnflag']
        except:
            raise AssertionError("Null Likelihood did not converge. Please "
                    "report.")
        return res.llf

    def predict(self, exog=None, typ='xb'):
        return self.model.predict(self.params, exog, typ)
    #TODO: fix this to remove params, we need a general solution here
    predict.__doc__ = Tobit.predict.__doc__

def webuse(data, baseurl='http://www.stata-press.com/data/r11/'):
    """
    Parameters
    ----------
    data : str
    Name of dataset to fetch.

    Examples
    --------
    >>> dta = webuse('auto')

    Notes
    -----
    Make sure baseurl has trailing forward slash. Doesn't do any
    error checking in response URLs.
    """
    # lazy imports
    import pandas
    from scikits.statsmodels.iolib import genfromdta
    from urllib2 import urlopen
    from urlparse import urljoin
    from StringIO import StringIO

    url = urljoin(baseurl, data+'.dta')
    dta = urlopen(url)
    dta = StringIO(dta.read()) # make it truly file-like
    return genfromdta(dta)

if __name__ == "__main__":
    import pandas
    import scikits.statsmodels.api as sm
    #dta = webuse('auto')
    #df = pandas.DataFrame.from_records(dta)
    #df['rep78'] = df['rep78'].astype(float)
    #df.ix[df['rep78'] == -999, 'rep78'] = np.nan
    #df['weight'] = df['weight'].astype(float) #TODO: fix StataReader for ints
    #exog = sm.add_constant(df['weight'] / 1000, prepend=True)
    #endog = df['mpg']
    #auto_mod2 = Tobit(endog, exog, left=17, right=False).fit(method='bfgs', maxiter=1000)



    #bse = np.sqrt(np.diag(np.linalg.inv(-approx_hess(np.r_[mod.params,
    #                np.log(mod.scale)], mod.model.loglike)[0])))
    # now since we take the exponent in the likelihood of tobit, we have
    # f(x) = exp(log(x)) == exp(x)*1/x, so
    #bse[-1] = bse[-1] * mod.scale

    #check olsen reparams for non-zero censoring
    #auto_mod2 = Tobit(endog, exog, left=False, right=24).fit(method='bfgs')
    #self = auto_mod2.model
    #gamma = auto_mod2.model_params / auto_mod2.scale
    #theta = 1 / auto_mod2.scale
    #params = np.r_[gamma, theta]

    #mod3 = Tobit(endog, exog, left=17, right=24).fit(method='bfgs')

    # compare to Stata results in the manual, parameters and bse look good

    # replicate Fair's results
    data = sm.datasets.fair.load()
    endog = data.endog
    exog = sm.add_constant(data.exog, prepend=False)
    #mod = Tobit(endog, exog, left=0, right=False).fit(method='bfgs')

    # well this is interesting...

    #NOTE: newton's method doesn't work well, I think this is due to the
    #Hessian approximation
    #fair_mod = Tobit(endog, exog, left=0, right=False).fit(method='newton')
    #cov = np.linalg.inv(-approx_hess(mod.params, mod.model.loglike)[0])
    #bse = np.sqrt(np.diag(np.linalg.inv(-approx_hess(mod.params,
    #                    mod.model.loglike)[0])))
    # now since we take the exponent in the likelihood of tobit, we have
    # f(x) = exp(log(x)) == exp(x)*1/x, so
    #bse[-1] = bse[-1] * mod.scale

    # params from stata
    true_params = np.array([-1.530712955415909, -.1051385059749894, .1282901151942823, -.0277671231250877, -.9434969317125149, -.085975029751076, .3128387422989903, .0142119678926689, 7.836529294458556, 4.498874144181835])

    # ok
    print 'Newton'
    mod1 = Tobit(endog, exog, left=0, right=False).fit(method='newton',
            retall=1)

    print 'NM' # this doesn't work well,
    mod2 = Tobit(endog, exog, left=0, right=False).fit(method='nm',
                    maxiter=10000, maxfun=10000, ftol=1e-12, xtol=1e-12)
    print 'BFGS'
    mod3 = Tobit(endog, exog, left=0, right=False).fit(method='bfgs',
                        maxiter=250)
    print 'Powell'
    mod4 = Tobit(endog, exog, left=0, right=False).fit(method='powell',
                        xtol=1e-8, ftol=1e-8)
    print 'CG'
    mod5 = Tobit(endog, exog, left=0, right=False).fit(method='cg',
                                maxiter=1000)
    print 'NCG'
    mod6 = Tobit(endog, exog, left=0, right=False).fit(method='ncg')
