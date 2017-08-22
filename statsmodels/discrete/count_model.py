from __future__ import division

__all__ = ["TruncatedPoisson", "TruncatedNegativeBinomialP", "Hurdle"]

import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions import truncatedpoisson, truncatednegbin
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
                                                 Poisson, Logit, CountResults,
                                                 L1CountResults, Probit,
                                                 NegativeBinomial,
                                                 NegativeBinomialP,
                                                 _discrete_results_docs)
from statsmodels.tools.numdiff import (approx_fprime, approx_hess,
                                       approx_hess_cs, approx_fprime_cs)
from statsmodels.tools.decorators import resettable_cache, cache_readonly
from copy import deepcopy

class GenericTruncated(CountModel):
    __doc__ = """
    Generic Truncated model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    truncation : int, optional
        Truncation parameter specify truncation point out of the support
        of the distribution. pmf(k) = 0 for k <= truncation
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, truncation=0, offset=None,
                       exposure=None, missing='none', **kwargs):
        super(GenericTruncated, self).__init__(endog, exog, offset=offset,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)
        self.exog = self.exog[self.endog >= (truncation + 1)]
        self.endog = self.endog[self.endog >= (truncation + 1)]
        self.trunc = truncation

    def loglike(self, params):
        """
        Loglikelihood of Generic Truncated model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        --------

        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Truncated model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        --------

        """
        llf_main = self.model_main.loglikeobs(params)

        pmf = np.zeros_like(self.endog, dtype=np.float64)
        for i in range(self.trunc + 1):
            model = self.model_main.__class__(np.ones_like(self.endog) * i,
                                         self.exog)
            pmf +=  np.exp(model.loglikeobs(params))

        llf = llf_main - np.log(1 - pmf)

        return llf

    def score_obs(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        score_main = self.model_main.score_obs(params)

        pmf = np.zeros_like(self.endog, dtype=np.float64)
        score_trunc = np.zeros_like(score_main, dtype=np.float64)
        for i in range(self.trunc + 1):
            model = self.model_main.__class__(np.ones_like(self.endog) * i,
                                              self.exog)
            pmf_i =  np.exp(model.loglikeobs(params))
            score_trunc += (model.score_obs(params).T * pmf_i).T
            pmf += pmf_i

        dparams = score_main + (score_trunc.T / (1 - pmf)).T

        return dparams

    def score(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        return self.score_obs(params).sum(0)

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            start_params = model.fit(disp=0).params
        mlefit = super(GenericTruncated, self).fit(start_params=start_params,
                       maxiter=maxiter, disp=disp,
                       full_output=full_output, callback=lambda x:x,
                       **kwargs)

        zipfit = self.result(self, mlefit._results)
        result = self.result_wrapper(zipfit)

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):

        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1]
            alpha = alpha * np.ones(k_params)

        alpha_p = alpha
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            start_params = model.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = self.result_reg(self, cntfit)
        else:
            raise TypeError(
                    "argument method == %s, which is not handled" % method)

        return self.result_reg_wrapper(discretefit)

    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Truncated model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        """
        return approx_hess(params, self.loglike)

class Truncated(GenericTruncated):
    """
    Truncated model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    truncation : int, optional
        Truncation parameter specify truncation point out of the support
        of the distribution. pmf(k) = 0 for k <= truncation
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, model=Poisson,
                 distribution=truncatedpoisson, offset=None,
                 exposure=None, truncation=0, missing='none', **kwargs):
        super(Truncated, self).__init__(endog, exog, offset=offset,
                                               exposure=exposure,
                                               truncation=truncation,
                                               missing=missing, **kwargs)
        self.model_main = model(self.endog, self.exog,
                                offset=offset, exposure=exposure)
        self.k_extra = getattr(self.model_main, "k_extra", 0)
        self.exog_names.extend(list(set(self.model_main.exog_names) - set(self.exog_names)))
        self.model_dist = distribution
        self.result = GenericTruncatedResults
        self.result_wrapper = GenericTruncatedResultsWrapper
        self.result_reg = L1GenericTruncatedResults
        self.result_reg_wrapper = L1GenericTruncatedResultsWrapper   

class TruncatedPoisson(GenericTruncated):
    """
    Truncated Poisson model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    truncation : int, optional
        Truncation parameter specify truncation point out of the support
        of the distribution. pmf(k) = 0 for k <= truncation
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None,
                 truncation=0, missing='none', **kwargs):
        super(TruncatedPoisson, self).__init__(endog, exog, offset=offset,
                                               exposure=exposure,
                                               truncation=truncation,
                                               missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog,
                                  exposure=exposure,
                                  offset=offset)
        self.model_dist = truncatedpoisson
        self.result = TruncatedPoissonResults
        self.result_wrapper = TruncatedPoissonResultsWrapper
        self.result_reg = L1TruncatedPoissonResults
        self.result_reg_wrapper = L1TruncatedPoissonResultsWrapper

    def predict(self, params, exog=None, exposure=None, offset=None,
                which='mean', count_prob=None):
        """
        Paramaters
        ----------
        count_prob : array-like or int
            The counts for which you want the probabilities. If count_prob is
            None then the probabilities for each count from 0 to max(y) are
            given.

        Predict response variable of a count model given exogenous variables.
        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        if exog is None:
            exog = self.exog
        
        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)

        if offset is None:
            offset = getattr(self, 'offset', 0)

        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset

        if which == 'mean':
            return np.exp(linpred) / (1 - np.exp(-np.exp(linpred)))
        elif which == 'linear':
            return linpred
        elif which == 'prob':
            if count_prob is not None:
                counts = np.atleast_2d(count_prob)
            else:
                counts = np.atleast_2d(np.arange(0, np.max(self.endog)+1))
            mu = self.predict(params, exog=exog, exposure=exposure,
                              offset=offset)[:,None]
            return self.model_dist.pmf(counts, mu, self.trunc)
        else:
            raise TypeError(
                "argument wich == %s not handled" % which)

class TruncatedNegativeBinomialP(GenericTruncated):
    """
    Truncated Generalized Negative Binomial model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    truncation : int, optional
        Truncation parameter specify truncation point out of the support
        of the distribution. pmf(k) = 0 for k <= truncation
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None,
                 truncation=0, p=2, missing='none', **kwargs):
        super(TruncatedNegativeBinomialP, self).__init__(endog, exog, offset=offset,
                                               exposure=exposure,
                                               truncation=truncation,
                                               missing=missing, **kwargs)
        self.model_main = NegativeBinomialP(self.endog, self.exog,
                                            exposure=exposure,
                                            offset=offset, p=p)
        self.model_dist = truncatednegbin
        self.result = GenericTruncatedResults
        self.result_wrapper = GenericTruncatedResultsWrapper
        self.result_reg = L1GenericTruncatedResults
        self.result_reg_wrapper = L1GenericTruncatedResultsWrapper

    def predict(self, params, exog=None, exposure=None, offset=None,
                which='mean', count_prob=None):
        """
        Paramaters
        ----------
        count_prob : array-like or int
            The counts for which you want the probabilities. If count_prob is
            None then the probabilities for each count from 0 to max(y) are
            given.

        Predict response variable of a count model given exogenous variables.
        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        if exog is None:
            exog = self.exog
        
        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)

        if offset is None:
            offset = getattr(self, 'offset', 0)

        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset

        if which == 'mean':
            return np.exp(linpred) / (1 - np.exp(-np.exp(linpred)))
        elif which == 'linear':
            return linpred
        elif which == 'prob':
            if count_prob is not None:
                counts = np.atleast_2d(count_prob)
            else:
                counts = np.atleast_2d(np.arange(0, np.max(self.endog)+1))
            mu = self.predict(params, exog=exog, exposure=exposure,
                              offset=offset)[:,None]
            return self.model_dist.pmf(counts, mu, params[-1],
                self.model_main.parametrization ,self.trunc)
        else:
            raise TypeError(
                "argument wich == %s not handled" % which)

class GenericCensored(CountModel):
    __doc__ = """
    Generic Censored model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None,
                       exposure=None, missing='none', **kwargs):
        self.zero_idx = np.nonzero(endog == 0)[0]
        self.nonzero_idx = np.nonzero(endog)[0]
        super(GenericCensored, self).__init__(endog, exog, offset=offset,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)

    def loglike(self, params):
        """
        Loglikelihood of Generic Censored model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        --------

        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Censored model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        --------

        """
        llf_main = self.model_main.loglikeobs(params)
        
        llf = np.concatenate((llf_main[self.zero_idx],
            np.log(1 - np.exp(llf_main[self.nonzero_idx]))))

        return llf

    def score_obs(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        score_main = self.model_main.score_obs(params)
        llf_main = self.model_main.loglikeobs(params)
        
        score = np.concatenate((score_main[self.zero_idx],
            (score_main[self.nonzero_idx].T *
            -np.exp(llf_main[self.nonzero_idx]) /
            (1 - np.exp(llf_main[self.nonzero_idx]))).T))

        return score

    def score(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        return self.score_obs(params).sum(0)

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            start_params = model.fit(disp=0).params
        mlefit = super(GenericCensored, self).fit(start_params=start_params,
                       maxiter=maxiter, disp=disp,
                       full_output=full_output, callback=lambda x:x,
                       **kwargs)

        zipfit = self.result(self, mlefit._results)
        result = self.result_wrapper(zipfit)

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):

        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1]
            alpha = alpha * np.ones(k_params)

        alpha_p = alpha
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            start_params = model.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = self.result_reg(self, cntfit)
        else:
            raise TypeError(
                    "argument method == %s, which is not handled" % method)

        return self.result_reg_wrapper(discretefit)

    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Censored model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        """
        return approx_hess(params, self.loglike)

class Censored(GenericCensored):
    """
    Censored model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, model=Poisson,
                 distribution=truncatedpoisson, offset=None,
                 exposure=None, missing='none', **kwargs):
        super(Censored, self).__init__(endog, exog, offset=offset,
                                               exposure=exposure,
                                               missing=missing, **kwargs)
        self.model_main = model(np.zeros_like(self.endog), self.exog)
        self.model_dist = distribution
        self.result = GenericTruncatedResults
        self.result_wrapper = GenericTruncatedResultsWrapper
        self.result_reg = L1GenericTruncatedResults
        self.result_reg_wrapper = L1GenericTruncatedResultsWrapper 

class GenericHurdle(CountModel):
    __doc__ = """
    Generic Hurdle model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None,
                 exposure=None, missing='none', **kwargs):
        super(GenericHurdle, self).__init__(endog, exog, offset=offset,
                                            exposure=exposure,
                                            missing=missing, **kwargs)
        self.model1 = Censored(self.endog, self.exog, model=self.model_name1)
        self.model2 = Truncated(self.endog, self.exog, model=self.model_name2)
        self.exog_names.insert(0, 'inflate_const')
        self.k_extra1 = self
        self.k_extra2 = 0
        for i in range(self.exog.shape[1], 1, -1): 
            self.exog_names.insert(0, 'zero_x%d' % (i-1))

    def loglike(self, params):
        """
        Loglikelihood of Generic Hurdle model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        --------

        """
        k = int((len(params) - self.k_extra1 - self.k_extra2) / 2) + self.k_extra1
        return self.model1.loglike(params[:k]) + self.model2.loglike(params[k:])

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        results1 = self.model1.fit(start_params=start_params,
                       method=method, maxiter=maxiter, disp=disp,
                       full_output=full_output, callback=lambda x:x,
                       **kwargs)

        results2 = self.model2.fit(start_params=start_params,
                       method=method, maxiter=maxiter, disp=disp,
                       full_output=full_output, callback=lambda x:x,
                       **kwargs)        

        result = deepcopy(results1)
        result._results.model = self
        result.mle_retvals['converged'] = [results1.mle_retvals['converged'], results2.mle_retvals['converged']]
        result._results.params = np.append(results1._results.params, results2._results.params)
        result._results.df_model += results2._results.df_model

        modelfit = self.result(self, result._results, results1, results2)
        result = self.result_wrapper(modelfit)

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

class HurdlePoisson(GenericHurdle):
    """
    Poisson Poisson Hurdle model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}   

    def __init__(self, endog, exog, offset=None,
                       exposure=None, missing='none', **kwargs):
        self.model_name1 = Poisson
        self.model_name2 = Poisson
        super(HurdlePoisson, self).__init__(endog, exog, offset=offset,
                                            exposure=exposure,
                                            missing=missing, **kwargs)
        self.result = HurdlePoissonResults
        self.result_wrapper = HurdlePoissonResultsWrapper
        self.result_reg = L1HurdlePoissonResults
        self.result_reg_wrapper = L1HurdlePoissonResultsWrapper

class GenericTruncatedResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Generic Truncated",
                    "extra_attr" : ""}

    @cache_readonly
    def _dispersion_factor(self):
        mu = np.exp(self.predict(which='linear'))

        return (1 - mu / (np.exp(mu) - 1))

class TruncatedPoissonResults(GenericTruncatedResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Truncated Poisson",
                    "extra_attr" : ""}

class L1GenericTruncatedResults(L1CountResults, GenericTruncatedResults):
    pass

class L1TruncatedPoissonResults(L1CountResults, TruncatedPoissonResults):
    pass

class GenericTruncatedResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(GenericTruncatedResultsWrapper,
                      GenericTruncatedResults)

class TruncatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(TruncatedPoissonResultsWrapper,
                      TruncatedPoissonResults)

class L1GenericTruncatedResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1GenericTruncatedResultsWrapper,
                      L1GenericTruncatedResults)

class L1TruncatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1TruncatedPoissonResultsWrapper,
                      L1TruncatedPoissonResults)

class HurdlePoissonResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Hurdle model",
                    "extra_attr" : ""}
    
    def __init__(self, model, mlefit, model1, model2, cov_type='nonrobust', cov_kwds=None,
                 use_t=None):
        super(HurdlePoissonResults, self).__init__(model, mlefit,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        self.model1 = model1
        self.model2 = model2

    @cache_readonly
    def llnull(self):
        return self.model1._results.llnull + self.model2._results.llnull

    @cache_readonly
    def bse(self):
        return np.append(self.model1.bse, self.model2.bse)

class L1HurdlePoissonResults(L1CountResults, HurdlePoissonResults):
    pass

class HurdlePoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(HurdlePoissonResultsWrapper,
                      HurdlePoissonResults)
class L1HurdlePoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1HurdlePoissonResultsWrapper,
                      L1HurdlePoissonResults)

if __name__=="__main__":
    import numpy as np
    import statsmodels.api as sm

    data = sm.datasets.randhie.load()
    endog = data.endog
    exog = sm.add_constant(data.exog[:,:3], prepend=False)
    res1 = Truncated(endog, exog, model=NegativeBinomialP).fit(method="bfgs", maxiter=3500)

    print(res1.params)
    print(res1.llf)
    print(res1.aic)
    print(res1.df_model)
    print(res1.bse)
    print(res1.summary())