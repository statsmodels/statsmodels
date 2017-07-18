from __future__ import division

__all__ = ["TruncatedPoisson"]

import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions import truncatedpoisson
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
                                                 Poisson, Logit, CountResults,
                                                 L1CountResults, Probit,
                                                 _discrete_results_docs)
from statsmodels.tools.numdiff import (approx_fprime, approx_hess,
                                       approx_hess_cs, approx_fprime_cs)

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
        .. math:: \\ln L=\\sum_{y_{i}=0}\\ln(w_{i}+(1-w_{i})*P_{main\\_model})+
            \\sum_{y_{i}>0}(\\ln(1-w_{i})+L_{main\\_model})
            where P - pdf of main model, L - loglike function of main model.

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
        .. math:: \\ln L=\\ln(w_{i}+(1-w_{i})*P_{main\\_model})+
            \\ln(1-w_{i})+L_{main\\_model}
            where P - pdf of main model, L - loglike function of main model.

        for observations :math:`i=1,...,n`

        """
        llf_main = self.model_main.loglikeobs(params)

        pmf = np.zeros_like(self.endog, dtype=np.float64)
        for i in range(self.trunc + 1):
            model = self.model_main_name(np.ones_like(self.endog) * i,
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
            model = self.model_main_name(np.ones_like(self.endog) * i,
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
            model = self.model_main_name(self.endog, self.exog, offset=offset)
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
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main_name(self.endog, self.exog, offset=offset)
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
            return np.exp(linpred) / (1 - np.exp(np.exp(-linpred)))
        elif which == 'linear':
            return linpred
        elif which =='prob':
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

class TruncatedPoisson(GenericTruncated):
    """
    Poisson Truncated model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    exog_infl: array
        A reference to the zero-inflated exogenous design.
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
        self.model_main_name = Poisson
        self.model_main = Poisson(self.endog, self.exog)
        self.model_dist = truncatedpoisson
        self.result = TruncatedPoissonResults
        self.result_wrapper = TruncatedPoissonResultsWrapper
        self.result_reg = L1TruncatedPoissonResults
        self.result_reg_wrapper = L1TruncatedPoissonResultsWrapper

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
        self.model1 = self.model_name1(
            np.zeros_like(self.endog, dtype=np.float64), self.exog)
        self.model2 = self.model_name2(self.endog, self.exog)
        
    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Hurdle model

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

        for observations :math:`i=1,...,n`
        """
        llf = np.zeros_like(self.endog, dtype=np.float64)

        llf1 = self.model1.loglikeobs(params)
        llf2 = self.model2.loglikeobs(params)

        y = self.endog
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        llf[zero_idx] = llf1[zero_idx]
        llf[nonzero_idx] = np.log(1 - np.exp(llf1[nonzero_idx])) + llf2

        return llf

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
        return np.sum(self.loglikeobs(params))

    def score_obs(self, params):
        return approx_fprime(params, self.loglikeobs)

    def score(self, params):
        return np.sum(self.score_obs(params), axis=0)

    def hessian(self, params):
        return approx_hess(params, self.loglike)

class GenericTruncatedResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Generic Zero Inflated",
                    "extra_attr" : ""}

class TruncatedPoissonResults(GenericTruncatedResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Zero Inflated Poisson",
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


if __name__=="__main__":
    import numpy as np
    import statsmodels.api as sm
