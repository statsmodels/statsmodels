from __future__ import division

__all__ = ["PoissonZeroInflated"]

import numpy as np
import statsmodels.base.model as base
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
                                                 Poisson, Logit)
from statsmodels.tools.numdiff import (approx_fprime, approx_hess,
                                       approx_hess_cs, approx_fprime_cs)

class GenericZeroInflated(CountModel):
    __doc__ = """
    Generiz Zero Inflated model for count data

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
    def __init__(self, endog, exog, exog_infl=None, offset=None,
                       exposure=None, missing='none', **kwargs):
        super(GenericZeroInflated, self).__init__(endog, exog, offset=offset,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)

        if exog_infl is None:
            self.exog_infl = np.ones((endog.size, 1))
        else:
            self.exog_infl = exog_infl
        self.k_exog = exog.shape[1]
        self.k_inflate = exog_infl.shape[1]

    def loglike(self, params):
        """
        Loglikelihood of Generic Zero Inflated model

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
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Zero Inflated model

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
        .. math:: \\ln L_{i}=\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        for observations :math:`i=1,...,n`

        """
        params_infl = params[self.k_exog:]
        params_main = params[:self.k_exog]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w[w >= 1.] = np.nextafter(1, 0)
        llf_main = self.model_main.loglikeobs(params_main)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        llf = np.zeros_like(y)
        llf[zero_idx] = (np.log(w[zero_idx] +
            (1 - w[zero_idx]) * np.exp(llf_main[zero_idx])))
        llf[nonzero_idx] = np.log(1 - w[nonzero_idx]) + llf_main[nonzero_idx]

        return llf

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            start_params = mod_poi.fit(disp=0).params
            start_params = np.append(start_params, np.zeros(self.k_inflate))
        mlefit = super(GenericZeroInflated, self).fit(start_params=start_params,
                        maxiter=maxiter, disp=disp,
                        full_output=full_output, callback=lambda x:x,
                        **kwargs)

        #gpfit = GeneralizedPoissonResults(self, mlefit._results)
        #result = GeneralizedPoissonResultsWrapper(gpfit)
        result = mlefit

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

    def score(self, params):
        """
        Generic Zero Inflated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\lambda_{i}\\right)x_{i}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta
        """
        return approx_fprime(params, self.loglike)

    def hessian(self, params):
        """
        Generic Zero Inflated model Hessian matrix of the loglikelihood

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
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i=1}^{n}\\lambda_{i}x_{i}x_{i}^{\\prime}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta

        """
        return approx_hess(params, self.loglike)

class PoissonZeroInflated(GenericZeroInflated):
    """
    Poisson Zero Inflated model for count data

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
    def __init__(self, endog, exog, exog_infl=None, offset=None,
                       exposure=None, missing='none', **kwargs):
        super(PoissonZeroInflated, self).__init__(endog, exog, offset=offset,
                                                  exog_infl=exog_infl,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)
        self.model_main = Poisson(endog, exog)
        self.model_infl = Logit(np.zeros(exog_infl.shape[0]), exog_infl)

    def score(self, params):
        params_infl = params[self.k_exog:]
        params_main = params[:self.k_exog]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w[w == 1.] = np.nextafter(1, 0)
        score_main = self.model_main.score_obs(params_main)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        mu = self.model_main.predict(params_main)

        dmudb = (self.exog[zero_idx].T * mu[zero_idx]).T
        coeff = (1 + w[zero_idx] * (np.exp(mu[zero_idx]) - 1))
        dldp_zero = (dmudb.T * ((w[zero_idx] - 1) / coeff)).T.sum(0)
        dldp_nonzero = score_main[nonzero_idx].sum(0)
        dldp = dldp_zero + dldp_nonzero

        dldw_zero =  -self.exog_infl[zero_idx].T.dot(w[zero_idx]*(1-w[zero_idx])*(1 - np.exp(mu[zero_idx])) / coeff)
        dldw_nonzero = -self.exog_infl[nonzero_idx].T.dot(w[nonzero_idx])
        dldw = dldw_zero + dldw_nonzero

        return np.concatenate((dldp, dldw))


if __name__=="__main__":
    import numpy as np
    import statsmodels.api as sm
