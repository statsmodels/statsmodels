__all__ = [
    "HurdleCountModel",
    "TruncatedLFNegativeBinomialP",
    "TruncatedLFPoisson",
]

from copy import deepcopy
import warnings

import numpy as np
from scipy.linalg import block_diag

import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (
    CountModel,
    CountResults,
    DiscreteModel,
    GeneralizedPoisson,
    L1CountResults,
    NegativeBinomialP,
    Poisson,
    _discrete_results_docs,
    _l1_results_attr,
    _validate_l1_method,
)
from statsmodels.distributions.discrete import (
    truncatednegbin,
    truncatedpoisson,
)
import statsmodels.regression.linear_model as lm
from statsmodels.tools._decorators import cache_readonly
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    int_like,
    string_like,
)


class TruncatedLFGeneric(CountModel):
    __doc__ = """
    Generic Truncated model for count data

    .. versionadded:: 0.14.0

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    truncation : int
        Truncation parameter that specifies the truncation point out of
        the support of the distribution. pmf(k) = 0 for k <= truncation
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, truncation=0, offset=None,
                 exposure=None, missing="none", **kwargs):
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            missing=missing,
            **kwargs
            )
        mask = self.endog > truncation
        self.exog = self.exog[mask]
        self.endog = self.endog[mask]
        if offset is not None:
            self.offset = self.offset[mask]
        if exposure is not None:
            self.exposure = self.exposure[mask]

        self.trunc = truncation
        self.truncation = truncation  # needed for recreating model
        # We cannot set the correct df_resid here, not enough information
        self._init_keys.extend(["truncation"])
        self._null_drop_keys = []

    def loglike(self, params):
        """
        Log-likelihood of Generic Truncated model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Log-likelihood for observations of Generic Truncated model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        llf_main = self.model_main.loglikeobs(params)

        yt = self.trunc + 1

        # equivalent ways to compute truncation probability
        # pmf0 = np.zeros_like(self.endog, dtype=np.float64)
        # for i in range(self.trunc + 1):
        #     model = self.model_main.__class__(np.ones_like(self.endog) * i,
        #                                       self.exog)
        #     pmf0 += np.exp(model.loglikeobs(params))
        #
        # pmf1 = self.model_main.predict(
        #     params, which="prob", y_values=np.arange(yt)).sum(-1)

        pmf = self.predict(
            params, which="prob-base", y_values=np.arange(yt)).sum(-1)

        # Skip pmf = 1 to avoid warnings
        log_1_m_pmf = np.full_like(pmf, -np.inf)
        loc = pmf > 1
        log_1_m_pmf[loc] = np.nan
        loc = pmf < 1
        log_1_m_pmf[loc] = np.log(1 - pmf[loc])
        llf = llf_main - log_1_m_pmf

        return llf

    def score_obs(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e., the first derivative of the
            log-likelihood function, evaluated at `params`
        """
        score_main = self.model_main.score_obs(params)

        pmf = np.zeros_like(self.endog, dtype=np.float64)
        # TODO: can we rewrite to following without creating new models
        score_trunc = np.zeros_like(score_main, dtype=np.float64)
        for i in range(self.trunc + 1):
            model = self.model_main.__class__(
                np.ones_like(self.endog) * i,
                self.exog,
                offset=getattr(self, "offset", None),
                exposure=getattr(self, "exposure", None),
                )
            pmf_i = np.exp(model.loglikeobs(params))
            score_trunc += (model.score_obs(params).T * pmf_i).T
            pmf += pmf_i

        dparams = score_main + (score_trunc.T / (1 - pmf)).T

        return dparams

    def score(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e., the first derivative of the
            log-likelihood function, evaluated at `params`
        """
        return self.score_obs(params).sum(0)

    def fit(self, start_params=None, method="bfgs", maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type="nonrobust", cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog,
                                              offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                start_params = model.fit(disp=0).params

        # Todo: check how we can to this in __init__
        k_params = self.df_model + 1 + self.k_extra
        self.df_resid = self.endog.shape[0] - k_params

        mlefit = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            disp=disp,
            full_output=full_output,
            callback=lambda x: x,
            **kwargs
            )

        zipfit = self.result_class(self, mlefit._results)
        result = self.result_class_wrapper(zipfit)

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(
            self, start_params=None, method="l1",
            maxiter="defined_by_method", full_output=1, disp=1, callback=None,
            alpha=0, trim_mode="auto", auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):

        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1]
            alpha = alpha * np.ones(k_params)

        alpha_p = alpha
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog,
                                              offset=offset)
            start_params = model.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode,
                auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        if method in ["l1", "l1_cvxopt_cp"]:
            discretefit = self.result_class_reg(self, cntfit)
        else:
            raise TypeError(
                    f"argument method == {method}, which is not handled")

        return self.result_class_reg_wrapper(discretefit)

    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Truncated model Hessian matrix of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of the log-likelihood function,
            evaluated at `params`
        """
        return approx_hess(params, self.loglike)

    def predict(self, params, exog=None, exposure=None, offset=None,
                which="mean", y_values=None):
        """
        Predict response variable or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : array_like, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        offset : array_like, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : array_like, optional
            Log(exposure) is added to the linear predictor with coefficient
            equal to 1. If exposure is specified, then it will be logged by
            the method. The user does not need to log it first.
            Default is one if exog is not None, and it is the model exposure
            if exog is None.
        which : str, optional
            Statistic to predict. Default is 'mean'.

            - 'mean' : the conditional expectation of endog E(y | x)
            - 'mean-main' : mean parameter of truncated count model.
              Note, this is not the mean of the truncated distribution.
            - 'linear' : the linear predictor of the truncated count model.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).
              The probabilities in the truncated region are zero.
            - 'prob-base' : probabilities for untruncated base distribution.
              The probabilities are for each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).


        y_values : array_like, optional
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``

        Returns
        -------
        ndarray
            The predicted values, whose interpretation depends on `which`.

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        exog, offset, exposure = self._get_predict_arrays(
            exog=exog,
            offset=offset,
            exposure=exposure
            )

        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset

        if which == "mean":
            mu = np.exp(linpred)
            if self.truncation == 0:
                prob_main = self.model_main._prob_nonzero(mu, params)
                return mu / prob_main
            elif self.truncation == -1:
                return mu
            elif self.truncation > 0:
                counts = np.atleast_2d(np.arange(0, self.truncation + 1))
                # next is same as in prob-main below
                probs = self.model_main.predict(
                    params, exog=exog, exposure=np.exp(exposure),
                    offset=offset, which="prob", y_values=counts)
                prob_tregion = probs.sum(1)
                mean_tregion = (np.arange(self.truncation + 1) * probs).sum(1)
                mean = (mu - mean_tregion) / (1 - prob_tregion)
                return mean
            else:
                raise ValueError("unsupported self.truncation")
        elif which == "linear":
            return linpred
        elif which == "mean-main":
            return np.exp(linpred)
        elif which == "prob":
            if y_values is not None:
                counts = np.atleast_2d(y_values)
            else:
                counts = np.atleast_2d(np.arange(0, np.max(self.endog)+1))
            mu = np.exp(linpred)[:, None]
            if self.k_extra == 0:
                # poisson, no extra params
                probs = self.model_dist.pmf(counts, mu, self.trunc)
            elif self.k_extra == 1:
                p = self.model_main.parameterization
                probs = self.model_dist.pmf(counts, mu, params[-1],
                                            p, self.trunc)
            else:
                raise ValueError("k_extra is not 0 or 1")
            return probs
        elif which == "prob-base":
            if y_values is not None:
                counts = np.asarray(y_values)
            else:
                counts = np.arange(0, np.max(self.endog)+1)

            probs = self.model_main.predict(
                params, exog=exog, exposure=np.exp(exposure),
                offset=offset, which="prob", y_values=counts)
            return probs
        elif which == "var":
            mu = np.exp(linpred)
            counts = np.atleast_2d(np.arange(0, self.truncation + 1))
            # next is same as in prob-main below
            probs = self.model_main.predict(
                params, exog=exog, exposure=np.exp(exposure),
                offset=offset, which="prob", y_values=counts)
            prob_tregion = probs.sum(1)
            mean_tregion = (np.arange(self.truncation + 1) * probs).sum(1)
            mean = (mu - mean_tregion) / (1 - prob_tregion)
            mnc2_tregion = (np.arange(self.truncation + 1)**2 *
                            probs).sum(1)
            vm = self.model_main._var(mu, params)
            # uncentered 2nd moment
            mnc2 = (mu**2 + vm - mnc2_tregion) / (1 - prob_tregion)
            v = mnc2 - mean**2
            return v
        else:
            raise ValueError(
                f"argument which == {which} not handled")


class TruncatedLFPoisson(TruncatedLFGeneric):
    __doc__ = """
    Truncated Poisson model for count data

    .. versionadded:: 0.14.0

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    truncation : int
        Truncation parameter that specifies the truncation point out of
        the support of the distribution. pmf(k) = 0 for k <= truncation
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None, exposure=None,
                 truncation=0, missing="none", **kwargs):
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            truncation=truncation,
            missing=missing,
            **kwargs
            )
        self.model_main = Poisson(self.endog, self.exog,
                                  exposure=getattr(self, "exposure", None),
                                  offset=getattr(self, "offset", None),
                                  )
        self.model_dist = truncatedpoisson

        self.result_class = TruncatedLFPoissonResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _predict_mom_trunc0(self, params, mu):
        """
        Predict mean and variance of zero-truncated distribution

        experimental api, will likely be replaced by other methods

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.

        Returns
        -------
        m : ndarray
            Predicted mean of the zero-truncated distribution.
        var_ : ndarray
            Predicted variance of the zero-truncated distribution.
        """
        w = (1 - np.exp(-mu))  # prob of no truncation, 1 - P(y=0)
        m = mu / w
        var_ = m - (1 - w) * m**2
        return m, var_


class TruncatedLFNegativeBinomialP(TruncatedLFGeneric):
    __doc__ = """
    Truncated Generalized Negative Binomial model for count data

    .. versionadded:: 0.14.0

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    truncation : int
        Truncation parameter that specifies the truncation point out of
        the support of the distribution. pmf(k) = 0 for k <= truncation
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.
    p : int, optional
        P denotes parameterizations for NB regression. p=1 for NB-1 and
        p=2 for NB-2. Default is p=2.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None, exposure=None,
                 truncation=0, p=2, missing="none", **kwargs):
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            truncation=truncation,
            missing=missing,
            **kwargs
            )
        self.model_main = NegativeBinomialP(
            self.endog,
            self.exog,
            exposure=getattr(self, "exposure", None),
            offset=getattr(self, "offset", None),
            p=p
            )
        self.k_extra = self.model_main.k_extra
        self.exog_names.extend(self.model_main.exog_names[-self.k_extra:])
        self.model_dist = truncatednegbin

        self.result_class = TruncatedNegativeBinomialResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _predict_mom_trunc0(self, params, mu):
        """
        Predict mean and variance of zero-truncated distribution

        experimental api, will likely be replaced by other methods

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.

        Returns
        -------
        m : ndarray
            Predicted mean of the zero-truncated distribution.
        var_ : ndarray
            Predicted variance of the zero-truncated distribution.
        """
        # note: prob_zero and vm are distribution specific, rest is generic
        # when mean of base model is mu
        alpha = params[-1]
        p = self.model_main.parameterization
        prob_zero = (1 + alpha * mu**(p-1))**(- 1 / alpha)
        w = 1 - prob_zero  # prob of no truncation, 1 - P(y=0)
        m = mu / w
        vm = mu * (1 + alpha * mu**(p-1))  # variance of NBP
        # uncentered 2nd moment is vm + mu**2
        mnc2 = (mu**2 + vm) / w  # uses mnc2_tregion = 0
        var_ = mnc2 - m**2
        return m, var_


class TruncatedLFGeneralizedPoisson(TruncatedLFGeneric):
    __doc__ = """
    Truncated Generalized Poisson model for count data

    .. versionadded:: 0.14.0

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    truncation : int
        Truncation parameter that specifies the truncation point out of
        the support of the distribution. pmf(k) = 0 for k <= truncation
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.
    p : int, optional
        Dispersion power parameter for the GeneralizedPoisson model. p=1
        for GP-1 and p=2 for GP-2. Default is p=2.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None, exposure=None,
                 truncation=0, p=2, missing="none", **kwargs):
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            truncation=truncation,
            missing=missing,
            **kwargs
            )
        self.model_main = GeneralizedPoisson(
            self.endog,
            self.exog,
            exposure=getattr(self, "exposure", None),
            offset=getattr(self, "offset", None),
            p=p
            )
        self.k_extra = self.model_main.k_extra
        self.exog_names.extend(self.model_main.exog_names[-self.k_extra:])
        self.model_dist = None
        self.result_class = TruncatedNegativeBinomialResults

        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper


class _RCensoredGeneric(CountModel):
    __doc__ = """
    Generic right Censored model for count data

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None, exposure=None,
                 missing="none", **kwargs):
        self.zero_idx = np.nonzero(endog == 0)[0]
        self.nonzero_idx = np.nonzero(endog)[0]
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            missing=missing,
            **kwargs
            )

    def loglike(self, params):
        """
        Log-likelihood of Generic Censored model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Log-likelihood for observations of Generic Censored model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        llf_main = self.model_main.loglikeobs(params)

        llf = np.concatenate(
            (llf_main[self.zero_idx],
             np.log(1 - np.exp(llf_main[self.nonzero_idx])))
            )

        return llf

    def score_obs(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e., the first derivative of the
            log-likelihood function, evaluated at `params`
        """
        score_main = self.model_main.score_obs(params)
        llf_main = self.model_main.loglikeobs(params)

        score = np.concatenate((
            score_main[self.zero_idx],
            (score_main[self.nonzero_idx].T *
             -np.exp(llf_main[self.nonzero_idx]) /
             (1 - np.exp(llf_main[self.nonzero_idx]))).T
            ))

        return score

    def score(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e., the first derivative of the
            log-likelihood function, evaluated at `params`
        """
        return self.score_obs(params).sum(0)

    def fit(self, start_params=None, method="bfgs", maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type="nonrobust", cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog,
                                              offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                start_params = model.fit(disp=0).params
        mlefit = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            disp=disp,
            full_output=full_output,
            callback=lambda x: x,
            **kwargs
            )

        zipfit = self.result_class(self, mlefit._results)
        result = self.result_class_wrapper(zipfit)

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(
            self, start_params=None, method="l1",
            maxiter="defined_by_method", full_output=1, disp=1, callback=None,
            alpha=0, trim_mode="auto", auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):

        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1]
            alpha = alpha * np.ones(k_params)

        alpha_p = alpha
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog,
                                              offset=offset)
            start_params = model.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode,
                auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        if method in ["l1", "l1_cvxopt_cp"]:
            discretefit = self.result_class_reg(self, cntfit)
        else:
            raise TypeError(
                    f"argument method == {method}, which is not handled")

        return self.result_class_reg_wrapper(discretefit)

    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Censored model Hessian matrix of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of the log-likelihood function,
            evaluated at `params`
        """
        return approx_hess(params, self.loglike)


class _RCensoredPoisson(_RCensoredGeneric):
    __doc__ = """
    Censored Poisson model for count data

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None,
                 exposure=None, missing="none", **kwargs):
        super().__init__(endog, exog, offset=offset,
                         exposure=exposure,
                         missing=missing, **kwargs)
        self.model_main = Poisson(np.zeros_like(self.endog), self.exog)
        self.model_dist = None
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper


class _RCensoredGeneralizedPoisson(_RCensoredGeneric):
    __doc__ = """
    Censored Generalized Poisson model for count data

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None, p=2,
                 exposure=None, missing="none", **kwargs):
        super().__init__(
            endog, exog, offset=offset, exposure=exposure,
            missing=missing, **kwargs)

        self.model_main = GeneralizedPoisson(
            np.zeros_like(self.endog), self.exog)
        self.model_dist = None
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper


class _RCensoredNegativeBinomialP(_RCensoredGeneric):
    __doc__ = """
    Censored Negative Binomial model for count data

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None, p=2,
                 exposure=None, missing="none", **kwargs):
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            missing=missing,
            **kwargs
            )
        self.model_main = NegativeBinomialP(np.zeros_like(self.endog),
                                            self.exog,
                                            p=p
                                            )
        self.model_dist = None
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper


class _RCensored(_RCensoredGeneric):
    __doc__ = """
    Censored model for count data

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, model=Poisson,
                 distribution=truncatedpoisson, offset=None,
                 exposure=None, missing="none", **kwargs):
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            missing=missing,
            **kwargs
            )
        self.model_main = model(np.zeros_like(self.endog), self.exog)
        self.model_dist = distribution
        # fix k_extra and exog_names
        self.k_extra = k_extra = self.model_main.k_extra
        if k_extra > 0:
            self.exog_names.extend(self.model_main.exog_names[-k_extra:])

        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _prob_nonzero(self, mu, params):
        """
        Probability that count is not zero

        internal use in Censored model, will be refactored or removed
        """
        prob_nz = self.model_main._prob_nonzero(mu, params)
        return prob_nz


class HurdleCountModel(CountModel):
    __doc__ = """
    Hurdle model for count data

    .. versionadded:: 0.14.0

    {params}
    {extra_params}

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.

    Notes
    -----
    The parameters in the NegativeBinomial zero model are not identified if
    the predicted mean is constant. If there is no or only little variation in
    the predicted mean, then convergence might fail, hessian might not be
    invertible or parameter estimates will have large standard errors.

    References
    ----------
    not yet
    """.format(
           params=base._model_params_doc,
           extra_params="""offset : array_like, optional
        Offset is added to the linear prediction with coefficient equal to 1.
    dist : {'poisson', 'negbin'}, optional
        Log-likelihood type of count model family. Default is 'poisson'.
    zerodist : {'poisson', 'negbin'}, optional
        Log-likelihood type of zero hurdle model family. Default is
        'poisson'.
    p : int, optional
        Dispersion power parameter for the NegativeBinomialP count model.
        Used when dist='negbin'. Default is 2.
    pzero : int, optional
        Dispersion power parameter for the NegativeBinomialP zero hurdle
        model. Used when zerodist='negbin'. Default is 2.
    exposure : array_like, optional
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc,
       )

    def __init__(self, endog, exog, offset=None,
                 dist="poisson", zerodist="poisson",
                 p=2, pzero=2,
                 exposure=None, missing="none", **kwargs):

        if (offset is not None) or (exposure is not None):
            msg = "Offset and exposure are not yet implemented"
            raise NotImplementedError(msg)
        super().__init__(
            endog,
            exog,
            offset=offset,
            exposure=exposure,
            missing=missing,
            **kwargs
            )
        # Use self.exog rather than the exog argument: the base class has
        # already converted it to a 2-dimensional ndarray, so this also
        # accepts the array_like inputs (lists, Series, DataFrames) that
        # every other count model accepts.
        self.k_exog = self.exog.shape[1]
        self.k_extra1 = 0
        self.k_extra2 = 0

        self._initialize(dist, zerodist, p, pzero)
        self.result_class = HurdleCountResults
        self.result_class_wrapper = HurdleCountResultsWrapper
        self.result_class_reg = L1HurdleCountResults
        self.result_class_reg_wrapper = L1HurdleCountResultsWrapper

    def _initialize(self, dist, zerodist, p, pzero):
        if (dist not in ["poisson", "negbin"] or
                zerodist not in ["poisson", "negbin"]):
            raise NotImplementedError('dist and zerodist must be "poisson",'
                                      '"negbin"')

        if zerodist == "poisson":
            self.model1 = _RCensored(self.endog, self.exog, model=Poisson)
        elif zerodist == "negbin":
            self.model1 = _RCensored(self.endog, self.exog,
                                     model=NegativeBinomialP)
            self.k_extra1 += 1

        if dist == "poisson":
            self.model2 = TruncatedLFPoisson(self.endog, self.exog)
        elif dist == "negbin":
            self.model2 = TruncatedLFNegativeBinomialP(self.endog, self.exog,
                                                       p=p)
            self.k_extra2 += 1

    @property
    def _k_zero(self):
        """
        Index that splits the parameter vector into its two components.

        Parameters are ordered as the zero model's exog parameters, the zero
        model's extra parameters, the main model's exog parameters and then
        the main model's extra parameters. ``params[:_k_zero]`` therefore
        belongs to the zero model and ``params[_k_zero:]`` to the main model.
        """
        return self.k_exog + self.k_extra1

    def loglike(self, params):
        """
        Log-likelihood of Generic Hurdle model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
        """
        k = self._k_zero
        return (self.model1.loglike(params[:k]) +
                self.model2.loglike(params[k:]))

    def fit(self, start_params=None, method="bfgs", maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type="nonrobust", cov_kwds=None, use_t=None, **kwargs):

        if cov_type != "nonrobust":
            raise ValueError("robust cov_type currently not supported")

        k_zero = self._k_zero
        if start_params is None:
            start_params1 = None
            start_params2 = None
        else:
            start_params = array_like(start_params, "start_params", ndim=1)
            k_params = k_zero + self.k_exog + self.k_extra2
            if start_params.size != k_params:
                raise ValueError(
                    "start_params must have one entry per parameter. The "
                    f"model has {k_params} parameters, {k_zero} in the zero "
                    f"model and {k_params - k_zero} in the main model, but "
                    f"start_params has {start_params.size}."
                )
            start_params1 = start_params[:k_zero]
            start_params2 = start_params[k_zero:]

        results1 = self.model1.fit(
            start_params=start_params1,
            method=method, maxiter=maxiter, disp=disp,
            full_output=full_output, callback=lambda x: x,
            **kwargs
            )

        results2 = self.model2.fit(
            start_params=start_params2,
            method=method, maxiter=maxiter, disp=disp,
            full_output=full_output, callback=lambda x: x,
            **kwargs
            )

        result = deepcopy(results1)
        result._results.model = self
        result.mle_retvals["converged"] = [results1.mle_retvals["converged"],
                                           results2.mle_retvals["converged"]]
        result._results.params = np.append(results1._results.params,
                                           results2._results.params)
        # TODO: the following should be in __init__ or initialize
        result._results.df_model += results2._results.df_model
        # this looks wrong attr does not exist, always 0
        self.k_extra1 += getattr(results1._results, "k_extra", 0)
        self.k_extra2 += getattr(results2._results, "k_extra", 0)
        self.k_extra = (self.k_extra1 + self.k_extra2 + 1)
        xnames1 = ["zm_" + name for name in self.model1.exog_names]
        self.exog_names[:] = xnames1 + self.model2.exog_names

        # fix up cov_params,
        # we could use normalized cov_params directly, unless it's not used
        result._results.normalized_cov_params = None
        try:
            cov1 = results1._results.cov_params()
            cov2 = results2._results.cov_params()
            result._results.normalized_cov_params = block_diag(cov1, cov2)
        except ValueError as e:
            if "need covariance" not in str(e):
                # could be some other problem
                raise

        modelfit = self.result_class(self, result._results, results1, results2)
        result = self.result_class_wrapper(modelfit)

        return result

    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(
        self,
        start_params=None,
        method="l1",
        maxiter="defined_by_method",
        full_output=1,
        disp=1,
        callback=None,
        alpha=0,
        trim_mode="auto",
        auto_trim_tol=0.01,
        size_trim_tol=1e-4,
        qc_tol=0.03,
        **kwargs,
    ):
        r"""
        Fit the model using a regularized maximum likelihood.

        The zero model and the main model are penalized and fit separately,
        and the combined parameter vector is then refit jointly.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the log-likelihood
            maximization. The default is an array of zeros. Ordered as
            described under `alpha`.
        method : {'l1', 'l1_cvxopt_cp'}, optional
            See notes for details.
        maxiter : int or 'defined_by_method', optional
            Maximum number of iterations to perform.
            If 'defined_by_method', then use method defaults (see notes).
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable, optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        alpha : float or array_like, optional
            Non-negative. The weight multiplying the l1 penalty term. If a
            scalar, every exog parameter of both components is penalized by
            this value and the extra parameters are left unpenalized, for
            example the shape parameter of a NegativeBinomialP component.

            If an array, it must have one entry per parameter, ordered as the
            zero model's exog parameters, the zero model's extra parameters,
            the main model's exog parameters and then the main model's extra
            parameters. For example, a hurdle model with 3 exog variables,
            ``zerodist="poisson"`` and ``dist="negbin"`` takes an `alpha` of
            length 7, whose last entry is the weight on the main model's
            shape parameter.
        trim_mode : {'auto', 'size', 'off'}, optional
            If not 'off', trim (set to zero) parameters that would have been
            zero if the solver reached the theoretical minimum.
            If 'auto', trim params using the theory in the notes below.
            If 'size', trim params if they have very small absolute value.
        auto_trim_tol : float, optional
            Tolerance used when trim_mode == 'auto'.
        size_trim_tol : float, optional
            Tolerance used when trim_mode == 'size'.
        qc_tol : float, optional
            Print warning and do not allow auto trim when condition (ii) in
            the notes below is violated by this much.
        **kwargs
            Additional keyword arguments used when fitting the model.

        Returns
        -------
        L1HurdleCountResultsWrapper
            A results instance.

        Notes
        -----
        Using 'l1_cvxopt_cp' requires the cvxopt module.

        Optional arguments for the solvers (available in
        Results.mle_settings)::

            'l1'
                acc : float (default 1e-6)
                    Requested accuracy as used by slsqp
            'l1_cvxopt_cp'
                abstol : float
                    absolute accuracy (default: 1e-7).
                reltol : float
                    relative accuracy (default: 1e-6).
                feastol : float
                    tolerance for feasibility conditions (default: 1e-7).
                refinement : int
                    number of iterative refinement steps when solving KKT
                    equations (default: 1).

        Optimization methodology

        With :math:`L` the negative log likelihood, we solve the convex but
        non-smooth problem

        .. math:: \min_\beta L(\beta) + \sum_k\alpha_k |\beta_k|

        via the transformation to the smooth, convex, constrained problem
        in twice as many variables (adding the "added variables" :math:`u_k`)

        .. math:: \min_{\beta,u} L(\beta) + \sum_k\alpha_k u_k,

        subject to

        .. math:: -u_k \leq \beta_k \leq u_k.

        With :math:`\partial_k L` the derivative of :math:`L` in the
        :math:`k^{th}` parameter direction, theory dictates that, at the
        minimum, exactly one of two conditions holds:

        (i) :math:`|\partial_k L| = \alpha_k`  and  :math:`\beta_k \neq 0`
        (ii) :math:`|\partial_k L| \leq \alpha_k`  and  :math:`\beta_k = 0`
        """
        _validate_l1_method(method)
        if maxiter != "defined_by_method":
            maxiter = int_like(maxiter, "maxiter")
            if maxiter < 0:
                raise ValueError("maxiter must be non-negative")
        full_output = bool_like(full_output, "full_output")
        disp = bool_like(disp, "disp")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        trim_mode = string_like(
            trim_mode, "trim_mode", options=("auto", "size", "off"),
            lower=False
        )
        auto_trim_tol = float_like(auto_trim_tol, "auto_trim_tol")
        size_trim_tol = float_like(size_trim_tol, "size_trim_tol")
        qc_tol = float_like(qc_tol, "qc_tol")

        k_zero = self._k_zero
        k_params = k_zero + self.k_exog + self.k_extra2
        if np.size(alpha) == 1:
            # Do not penalize extra parameters if alpha is a scalar
            alpha = float_like(alpha, "alpha") * np.concatenate([
                np.ones(self.k_exog),
                np.zeros(self.k_extra1),
                np.ones(self.k_exog),
                np.zeros(self.k_extra2),
            ])
        else:
            alpha = array_like(alpha, "alpha", ndim=1)
            if alpha.size != k_params:
                raise ValueError(
                    "alpha must be a scalar or a 1-dimensional array with "
                    f"one entry per parameter. The model has {k_params} "
                    f"parameters, {k_zero} in the zero model and "
                    f"{k_params - k_zero} in the main model, but alpha has "
                    f"{alpha.size}."
                )
        if not np.all(alpha >= 0):
            raise ValueError("alpha must be non-negative")
        alpha1 = alpha[:k_zero]
        alpha2 = alpha[k_zero:]

        if start_params is None:
            start_params1 = None
            start_params2 = None
        else:
            start_params = array_like(start_params, "start_params", ndim=1)
            if start_params.size != k_params:
                raise ValueError(
                    "start_params must have one entry per parameter. The "
                    f"model has {k_params} parameters, {k_zero} in the zero "
                    f"model and {k_params - k_zero} in the main model, but "
                    f"start_params has {start_params.size}."
                )
            start_params1 = start_params[:k_zero]
            start_params2 = start_params[k_zero:]

        results1 = self.model1.fit_regularized(
            start_params=start_params1,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=False,
            callback=callback,
            alpha=alpha1,
            trim_mode=trim_mode,
            auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol,
            qc_tol=qc_tol,
            **kwargs,
        )
        start_params1 = results1.params
        results2 = self.model2.fit_regularized(
            start_params=start_params2,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=False,
            callback=callback,
            alpha=alpha2,
            trim_mode=trim_mode,
            auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol,
            qc_tol=qc_tol,
            **kwargs,
        )
        start_params2 = results2.params
        start_params = np.append(start_params1, start_params2)

        cntfit = super(CountModel, self).fit_regularized(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            alpha=alpha,
            trim_mode=trim_mode,
            auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol,
            qc_tol=qc_tol,
            **kwargs,
        )
        # params comes from this joint refit, not from either component fit,
        # so its own converged flag (already set by the call above) has to
        # be kept, not replaced by the two component flags.
        cntfit.mle_retvals["converged"] = [
            results1.mle_retvals["converged"],
            results2.mle_retvals["converged"],
            cntfit.mle_retvals["converged"],
        ]
        self.k_extra1 += getattr(results1._results, "k_extra", 0)
        self.k_extra2 += getattr(results2._results, "k_extra", 0)
        self.k_extra = (self.k_extra1 + self.k_extra2 + 1)
        xnames1 = ["zm_" + name for name in self.model1.exog_names]
        self.exog_names[:] = xnames1 + self.model2.exog_names
        # Deliberately keep the covariance that came back from the joint
        # refit. fit has no joint optimization step, so it has to assemble a
        # covariance out of the two component fits, but here
        # DiscreteModel.cov_params_func_l1 has already built one from the
        # joint hessian and the joint set of trimmed parameters. Replacing it
        # with a block diagonal of the component covariances would pair the
        # params of one optimization with the standard errors of another,
        # which shows up as a non-zero coefficient reported with a nan
        # standard error whenever the two disagree about trimming.

        hurdlefit = self.result_class_reg(
            self, cntfit, results_zero=results1, results_count=results2
        )
        return self.result_class_reg_wrapper(hurdlefit)

    def score_obs(self, params):
        """
        Hurdle model score (gradient) vector of the log-likelihood.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score_obs : ndarray, 2-D
            The derivative of the log-likelihood for each observation,
            evaluated at `params`, with shape (nobs, k_params).
        """
        k_zero = self._k_zero
        params_zero = params[:k_zero]
        params_main = params[k_zero:]
        score_zero = self.model1.score_obs(params_zero)
        # The score of the main model is only defined for non-zero elements of
        # endog due to under-the-hood masking in left-truncated models. Since
        # those entries contribute nothing to the gradient, we can just fill
        # them with 0.
        score_main = np.zeros((self.exog.shape[0], self.k_exog + self.k_extra2))
        truncated_score_main = self.model2.score_obs(params_main)
        nonzero_idx = np.nonzero(self.endog)[0]
        score_main[nonzero_idx, :] = truncated_score_main
        return np.hstack((score_zero, score_main))

    def score(self, params):
        return self.score_obs(params).sum(0)

    def hessian(self, params):
        """
        Hurdle model Hessian matrix of the log-likelihood. When the zero and
        main models are separately estimated, this is a block diagonal matrix of
        the two models' Hessians.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function, evaluated
            at `params`
        """
        k_zero = self._k_zero
        params_zero = params[:k_zero]
        params_main = params[k_zero:]
        hessian_zero = self.model1.hessian(params_zero)
        hessian_main = self.model2.hessian(params_main)
        hessian = block_diag(hessian_zero, hessian_main)
        return hessian

    def predict(self, params, exog=None, exposure=None,
                offset=None, which="mean", y_values=None):
        """
        Predict response variable or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : array_like, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        offset : array_like, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : array_like, optional
            Log(exposure) is added to the linear predictor with coefficient
            equal to 1. If exposure is specified, then it will be logged by
            the method. The user does not need to log it first.
            Default is one if exog is not None, and it is the model exposure
            if exog is None.
        which : str, optional
            Statistic to predict. Default is 'mean'.

            - 'mean' : the conditional expectation of endog E(y | x)
            - 'mean-main' : mean parameter of truncated count model.
              Note, this is not the mean of the truncated distribution.
            - 'linear' : the linear predictor of the truncated count model.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'prob-main' : probability of selecting the main model which is
              the probability of observing a nonzero count P(y > 0 | x).
            - 'prob-zero' : probability of observing a zero count. P(y=0 | x).
              This is equal to ``1 - prob-main``.
            - 'prob-trunc' : probability of truncation of the truncated count
              model. This is the probability of observing a zero count implied
              by the truncation model.
            - 'mean-nonzero' : expected value conditional on having observation
              larger than zero, E(y | X, y>0)
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).

        y_values : array_like, optional
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``

        Returns
        -------
        ndarray
            The predicted values, whose interpretation depends on `which`.

        Notes
        -----
        'prob-zero' / 'prob-trunc' is the ratio of probabilities of observing
        a zero count between hurdle model and the truncated count model.
        If this ratio is larger than one, then the hurdle model has an inflated
        number of zeros compared to the count model. If it is smaller than one,
        then the number of zeros is deflated.
        """
        which = which.lower()  # make it case insensitive
        no_exog = True if exog is None else False
        exog, offset, exposure = self._get_predict_arrays(
            exog=exog,
            offset=offset,
            exposure=exposure
            )

        exog_zero = None  # not yet
        if exog_zero is None:
            if no_exog:
                exog_zero = self.exog
            else:
                exog_zero = exog

        k_zeros = int((len(params) - self.k_extra1 - self.k_extra2) / 2
                      ) + self.k_extra1
        params_zero = params[:k_zeros]
        params_main = params[k_zeros:]

        lin_pred = (np.dot(exog, params_main[:self.exog.shape[1]]) +
                    exposure + offset)

        # this currently is mean_main, offset, exposure for zero part ?
        mu1 = self.model1.predict(params_zero, exog=exog)
        # prob that count model applies y>0 from zero model predict
        prob_main = self.model1.model_main._prob_nonzero(mu1, params_zero)
        prob_zero = (1 - prob_main)

        mu2 = np.exp(lin_pred)
        prob_ntrunc = self.model2.model_main._prob_nonzero(mu2, params_main)

        if which == "mean":
            return prob_main * np.exp(lin_pred) / prob_ntrunc
        elif which == "mean-main":
            return np.exp(lin_pred)
        elif which == "linear":
            return lin_pred
        elif which == "mean-nonzero":
            return np.exp(lin_pred) / prob_ntrunc
        elif which == "prob-zero":
            return prob_zero
        elif which == "prob-main":
            return prob_main
        elif which == "prob-trunc":
            return 1 - prob_ntrunc
        # not yet supported
        elif which == "var":
            # generic computation using results from submodels
            mu = np.exp(lin_pred)
            mt, vt = self.model2._predict_mom_trunc0(params_main, mu)
            var_ = prob_main * vt + prob_main * (1 - prob_main) * mt**2
            return var_
        elif which == "prob":
            probs_main = self.model2.predict(
                params_main, exog, np.exp(exposure), offset, which="prob",
                y_values=y_values)
            probs_main *= prob_main[:, None]
            probs_main[:, 0] = prob_zero
            return probs_main
        else:
            raise ValueError(f"which = {which} is not available")


class TruncatedLFGenericResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Generic Truncated",
        "extra_attr": ""}


class TruncatedLFPoissonResults(TruncatedLFGenericResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Truncated Poisson",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        if self.model.trunc != 0:
            msg = "dispersion is only available for zero-truncation"
            raise NotImplementedError(msg)

        mu = np.exp(self.predict(which="linear"))

        return (1 - mu / (np.exp(mu) - 1))


class TruncatedNegativeBinomialResults(TruncatedLFGenericResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description":
            "A results class for Truncated Negative Binomial",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        if self.model.trunc != 0:
            msg = "dispersion is only available for zero-truncation"
            raise NotImplementedError(msg)

        alpha = self.params[-1]
        p = self.model.model_main.parameterization
        mu = np.exp(self.predict(which="linear"))

        return (1 - alpha * mu**(p-1) / (np.exp(mu**(p-1)) - 1))


class L1TruncatedLFGenericResults(L1CountResults, TruncatedLFGenericResults):
    pass


class TruncatedLFGenericResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(TruncatedLFGenericResultsWrapper,
                      TruncatedLFGenericResults)


class L1TruncatedLFGenericResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(L1TruncatedLFGenericResultsWrapper,
                      L1TruncatedLFGenericResults)


class HurdleCountResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Hurdle model",
        "extra_attr": ""}

    def __init__(self, model, mlefit, results_zero, results_count,
                 cov_type="nonrobust", cov_kwds=None, use_t=None):
        super().__init__(
            model,
            mlefit,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
            )
        self.results_zero = results_zero
        self.results_count = results_count
        # TODO: this is to fix df_resid, should be automatic but is not
        self.df_resid = self.model.endog.shape[0] - len(self.params)

    @cache_readonly
    def llnull(self):
        return (self.results_zero._results.llnull +
                self.results_count._results.llnull)

    @cache_readonly
    def bse(self):
        return np.append(self.results_zero.bse, self.results_count.bse)


class L1HurdleCountResults(HurdleCountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Hurdle model fit by l1 regularization",
        "extra_attr": _l1_results_attr
    }

    def __init__(
        self,
        model,
        mlefit,
        results_zero,
        results_count,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=None,
    ):
        super().__init__(
            model=model,
            mlefit=mlefit,
            results_zero=results_zero,
            results_count=results_count,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t
        )
        # TODO: mixins might eliminate the need for the below duplicated code
        #  (cf. .discrete_model.L1CountResults)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been zeroed out.
        self.trimmed = mlefit.mle_retvals["trimmed"]
        self.nnz_params = (~self.trimmed).sum()

        # Set degrees of freedom. Adjust for extra parameters not included in
        # df_model. Unlike L1CountResults, df_resid does not add k_extra back,
        # so that it agrees with HurdleCountResults on an untrimmed fit.
        k_extra = getattr(self.model, "k_extra", 0)
        self.df_model = self.nnz_params - 1 - k_extra
        self.df_resid = self.model.endog.shape[0] - self.nnz_params

    @cache_readonly
    def bse(self):
        # Not HurdleCountResults.bse, which concatenates the standard errors
        # of the two component fits. params come from the joint refit, so the
        # standard errors have to come from the joint covariance as well, or
        # the two disagree about which parameters were trimmed. This is the
        # standard LikelihoodModelResults.bse.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.sqrt(np.diag(self.cov_params()))


class HurdleCountResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(HurdleCountResultsWrapper,
                      HurdleCountResults)


class L1HurdleCountResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(L1HurdleCountResultsWrapper,
                      L1HurdleCountResults)
