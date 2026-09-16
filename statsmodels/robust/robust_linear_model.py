"""
Robust linear models with support for the M-estimators listed under
:ref:`norms <norms>`

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York.  1981.

PJ Huber.  1973,  'The 1972 Wald Memorial Lectures: Robust Regression:
    Asymptotics, Conjectures, and Monte Carlo.'  The Annals of Statistics,
    1.5, 799-821.

R Venables, B Ripley. 'Modern Applied Statistics in S'  Springer, New York,
    2002.
"""

import numpy as np
from scipy import stats

import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.robust import norms, scale
from statsmodels.tools._decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.validation import string_like

__all__ = ["RLM"]


def _check_convergence(criterion, iteration, tol, maxiter):
    cond = np.abs(criterion[iteration] - criterion[iteration - 1])
    return not (np.any(cond > tol) and iteration < maxiter)


class RLM(base.LikelihoodModel):
    __doc__ = f"""
    Robust Linear Model

    Estimate a robust linear model via iteratively reweighted least squares
    given a robust criterion estimator.

    {base._model_params_doc}
    M : statsmodels.robust.norms.RobustNorm, optional
        The robust criterion function for downweighting outliers.
        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,
        TrimmedMean, Hampel, TukeyBiweight, TukeyQuartic, StudentT, and
        MQuantileNorm.  The default is HuberT().
        See statsmodels.robust.norms for more information.
    {base._missing_param_doc}

    Attributes
    ----------
    df_model : float
        The degrees of freedom of the model.  The number of regressors p less
        one for the intercept.  Note that the reported model degrees
        of freedom does not count the intercept as a regressor, though
        the model is assumed to have an intercept.
    df_resid : float
        The residual degrees of freedom.  The number of observations n
        less the number of regressors p.  Note that here p does include
        the intercept as using a degree of freedom.
    endog : ndarray
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exog : ndarray
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    M : statsmodels.robust.norms.RobustNorm
         See above.  Robust estimator instance instantiated.
    nobs : float
        The number of observations n
    pinv_wexog : ndarray
        The pseudoinverse of the design / exogenous data array.  Note that
        RLM has no whiten method, so this is just the pseudo inverse of the
        design.
    normalized_cov_params : ndarray
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.stackloss.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
    >>> rlm_results = rlm_model.fit()
    >>> rlm_results.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results.bse
    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])
    >>> rlm_results_HC2 = rlm_model.fit(cov="H2")
    >>> rlm_results_HC2.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results_HC2.bse
    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])
    >>> mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.Hampel())
    >>> rlm_hamp_hub = mod.fit(scale_est=sm.robust.scale.HuberScale())
    >>> rlm_hamp_hub.params
    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])
    """

    def __init__(self, endog, exog, M=None, missing="none", **kwargs):
        self._check_kwargs(kwargs)
        self.M = M if M is not None else norms.HuberT()
        super(base.LikelihoodModel, self).__init__(
            endog, exog, missing=missing, **kwargs
        )
        self._initialize()
        # things to remove_data
        self._data_attr.extend(["weights", "pinv_wexog"])

        # Populated by `fit`; declared here so they exist (as None) even
        # before `fit` has been called.
        self.cov = None
        self.scale_est = None
        self.scale = None
        self.weights = None

    def _initialize(self):
        """
        Initialize the model for the IRLS fit

        Resets the history and number of iterations.
        """
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(
            self.pinv_wexog, np.transpose(self.pinv_wexog)
        )
        self.df_resid = float(self.exog.shape[0] - np.linalg.matrix_rank(self.exog))
        self.df_model = float(np.linalg.matrix_rank(self.exog) - 1)
        self.nobs = float(self.endog.shape[0])

    def score(self, params):
        raise NotImplementedError

    def information(self, params):
        raise NotImplementedError

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix

        Parameters
        ----------
        params : array_like
            Parameters of a linear model
        exog : array_like, optional
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        ndarray
            The predicted values.
        """
        # copied from linear_model  # TODO: then is it needed?
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def loglike(self, params):
        raise NotImplementedError

    def deviance(self, tmp_results):
        """
        Return the (unnormalized) log-likelihood from the M estimator

        Parameters
        ----------
        tmp_results : Results
            Results from intermediate fit containing ``fittedvalues`` and
            ``scale`` used to compute the residuals.

        Returns
        -------
        float
            The value of the deviance.
        """
        tmp_resid = self.endog - tmp_results.fittedvalues
        return self.M(tmp_resid / tmp_results.scale).sum()

    def _update_history(self, tmp_results, history, conv):
        history["params"].append(tmp_results.params)
        history["scale"].append(self.scale)
        if conv == "dev":
            history["deviance"].append(self.deviance(tmp_results))
        elif conv == "sresid":
            history["sresid"].append(tmp_results.resid / tmp_results.scale)
        elif conv == "weights":
            history["weights"].append(tmp_results.model.weights)
        return history

    def _estimate_scale(self, resid, scale_est):
        """
        Estimate the scale based on the option provided to the fit method

        Parameters
        ----------
        resid : ndarray
            The residuals used to estimate the scale.
        scale_est : {"mad"} or callable
            The scale estimator requested in the call to `fit`.

        Returns
        -------
        float
            The estimated scale.
        """
        if isinstance(scale_est, str):
            _ = string_like(scale_est, "scale_est", options=("mad",))
            return scale.mad(resid, center=0)
        elif isinstance(scale_est, scale.HuberScale):
            return scale_est(self.df_resid, self.nobs, resid)
        else:
            try:
                return scale_est(self, resid)
            except TypeError:
                return scale_est(resid) * np.sqrt(self.nobs / self.df_resid)

    def fit(
        self,
        maxiter=50,
        tol=1e-8,
        scale_est="mad",
        cov="H1",
        update_scale=True,
        conv="dev",
        start_params=None,
        start_scale=None,
    ):
        """
        Fit the model using iteratively reweighted least squares

        The IRLS routine runs until the specified objective converges to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------
        conv : {"coefs", "dev", "sresid", "weights"}, optional
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "sresid" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).  The default is "dev".
        cov : {"H1", "H2", "H3"}, optional
            Indicates how the covariance matrix is estimated.  Default is 'H1'.
            See rlm.RLMResults for more information.
        maxiter : int, optional
            The maximum number of iterations to try. Default is 50.
        scale_est : {"mad"}, HuberScale or callable, optional
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation).  Other options are
            'HuberScale' for Huber's proposal 2. Huber's proposal 2 has
            optional keyword arguments d, tol, and maxiter for specifying the
            tuning constant, the convergence tolerance, and the maximum number
            of iterations. Custom callables can accept either ``resid`` or
            ``(model, resid)`` and must return the scale estimate. The ``model``
            object provides useful afftributes like ``nobs`` and ``df_reside``
            that may be needed in scale estimation. Due to backward compatability
            issues, single- argument callables use the same degrees-of-freedom
            correction as the built-in non-Huber scale estimators (nobs/df_resid).
            Scale estimates from two argument functions are used without
            modification. See statsmodels.robust.scale for more information or
            the examples below.
        tol : float, optional
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : bool, optional
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.
        start_params : array_like, optional
            Initial guess of the solution of the optimizer. If not provided,
            the initial parameters are computed using OLS.
        start_scale : float, optional
            Initial scale. If update_scale is False, then the scale will be
            fixed at this level for the estimation of the mean parameters.
            during iteration. If not provided, then the initial scale is
            estimated from the OLS residuals

        Returns
        -------
        results : statsmodels.robust.robust_linear_model.RLMResults
            Results instance

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.stackloss.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
        >>> rlm_results_mad = rlm_model.fit(scale_est="mad")

        >>> from statsmodels.robust import HuberScale
        >>> hs = HuberScale(d=1.345, tol=1e-6, maxiter=100)
        >>> rlm_results_mad = rlm_model.fit(scale_est=hs)

        Next we use a custom callable to estimate the scale. The callable can
        either accept a single argument (the residuals) or two arguments (the
        model and the residuals). In this example, we use a callable that computes
        the average absolute deviation of the residuals. Note that the scale
        estimate from this one-input callable is adjusted for degrees of freedom,
        so it will be different than the two-input callable version below.

        >>> import numpy as np
        >>> def avg_abs_deviation(resid):
        ...     median = np.median(resid)
        ...     c = np.sqrt(np.pi / 2)
        ...     return c * np.mean(np.abs(resid - median))
        >>> rlm_results_custom = rlm_model.fit(scale_est=avg_abs_deviation)
        >>> print(f"{rlm_results_custom.scale:.4f}")
        3.0997

        This second version accepts two versions. While the function value is the same
        it will behave differently in practice since the scale estimate is not adjusted
        for degrees of freedom.

        >>> def avg_abs_deviation_with_model(model, resid):
        ...     median = np.median(resid)
        ...     c = np.sqrt(np.pi / 2)
        ...     return c * np.mean(np.abs(resid - median))
        >>> rlm_results_custom_2 = rlm_model.fit(scale_est=avg_abs_deviation_with_model)
        >>> print(f"{rlm_results_custom_2.scale:.4f}")
        2.7686
        """
        # options are upper-cased for display/storage, unlike most other
        # string options in this codebase which are lower-cased
        cov = string_like(cov, "cov", options=("h1", "h2", "h3")).upper()
        conv = string_like(conv, "conv", options=("weights", "coefs", "dev", "sresid"))

        if start_params is None:
            wls_results = lm.WLS(self.endog, self.exog).fit()
        else:
            start_params = np.asarray(start_params, dtype=np.double).squeeze()
            start_params = np.atleast_1d(start_params)
            if start_params.shape[0] != self.exog.shape[1] or start_params.ndim != 1:
                raise ValueError(
                    f"start_params must by a 1-d array with {self.exog.shape[1]} "
                    "values"
                )
            fake_wls = reg_tools._MinimalWLS(
                self.endog,
                self.exog,
                weights=np.ones_like(self.endog),
                check_weights=False,
            )
            wls_results = fake_wls.results(start_params)

        if not start_scale:
            self.scale = self._estimate_scale(wls_results.resid, scale_est)
        elif start_scale:
            self.scale = start_scale
            if not update_scale:
                scale_est = "fixed"

        history = dict(params=[np.inf], scale=[])
        if conv == "coefs":
            criterion = history["params"]
        elif conv == "dev":
            history.update(dict(deviance=[np.inf]))
            criterion = history["deviance"]
        elif conv == "sresid":
            history.update(dict(sresid=[np.inf]))
            criterion = history["sresid"]
        else:  # conv == "weights"
            history.update(dict(weights=[np.inf]))
            criterion = history["weights"]

        # done one iteration so update
        history = self._update_history(wls_results, history, conv)
        iteration = 1
        converged = 0
        while not converged:
            if self.scale == 0.0:
                import warnings

                warnings.warn(
                    "Estimated scale is 0.0 indicating that the most"
                    " last iteration produced a perfect fit of the "
                    "weighted data.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
                break
            self.weights = self.M.weights(wls_results.resid / self.scale)
            wls_results = reg_tools._MinimalWLS(
                self.endog, self.exog, weights=self.weights, check_weights=True
            ).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid, scale_est)
            history = self._update_history(wls_results, history, conv)
            iteration += 1
            converged = _check_convergence(criterion, iteration, tol, maxiter)
        results = RLMResults(
            self,
            wls_results.params,
            self.normalized_cov_params,
            self.scale,
            cov=cov,
        )

        history["iteration"] = iteration
        results.fit_history = history
        results.fit_options = dict(
            cov=cov,
            scale_est=scale_est,
            norm=self.M.__class__.__name__,
            conv=conv,
        )
        # norm is not changed in fit, no old state

        return RLMResultsWrapper(results)


class RLMResults(base.LikelihoodModelResults):
    """
    Class to contain RLM results

    Attributes
    ----------
    bcov_scaled : ndarray
        p x p scaled covariance matrix specified in the model fit method.
        The default is H1. H1 is defined as
        ``k**2 * (1/df_resid*sum(M.psi(sresid)**2)*scale**2)/
        ((1/nobs*sum(M.psi_deriv(sresid)))**2) * (X.T X)^(-1)``

        where ``k = 1 + (df_model +1)/nobs * var_psiprime/m**2``
        where ``m = mean(M.psi_deriv(sresid))`` and
        ``var_psiprime = var(M.psi_deriv(sresid))``

        H2 is defined as
        ``k * (1/df_resid) * sum(M.psi(sresid)**2) *scale**2/
        ((1/nobs)*sum(M.psi_deriv(sresid)))*W_inv``

        H3 is defined as
        ``1/k * (1/df_resid * sum(M.psi(sresid)**2)*scale**2 *
        (W_inv X.T X W_inv))``

        where `k` is defined as above and
        ``W_inv = (M.psi_deriv(sresid) exog.T exog)^(-1)``

        See the technical documentation for cleaner formulae.
    bcov_unscaled : ndarray
        The usual p x p covariance matrix with scale set equal to 1.  It
        is then just equivalent to normalized_cov_params.
    bse : ndarray
        An array of the standard errors of the parameters.  The standard
        errors are taken from the robust covariance matrix specified in the
        argument to fit.
    chisq : ndarray
        An array of the chi-squared values of the parameter estimates.
    df_model : float
        See RLM.df_model
    df_resid : float
        See RLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `deviance`,
        `params`, `iteration` and the convergence criteria specified in
        `RLM.fit`, if different from `deviance` or `params`.
    fit_options : dict
        Contains the options given to fit.
    fittedvalues : ndarray
        The linear predicted values.  dot(exog, params)
    model : statsmodels.robust.robust_linear_model.RLM
        A reference to the model instance
    nobs : float
        The number of observations n
    normalized_cov_params : ndarray
        See RLM.normalized_cov_params
    params : ndarray
        The coefficients of the fitted model
    pvalues : ndarray
        The p values associated with `tvalues`. Note that `tvalues` are assumed
        to be distributed standard normal rather than Student's t.
    resid : ndarray
        The residuals of the fitted model.  endog - fittedvalues
    scale : float
        The type of scale is determined in the arguments to the fit method in
        RLM.  The reported scale is taken from the residuals of the weighted
        least squares in the last IRLS iteration if update_scale is True.  If
        update_scale is False, then it is the scale given by the first OLS
        fit before the IRLS iterations.
    sresid : ndarray
        The scaled residuals.
    tvalues : ndarray
        The "t-statistics" of params. These are defined as params/bse where
        bse are taken from the robust covariance matrix specified in the
        argument to fit.
    weights : ndarray
        The reported weights are determined by passing the scaled residuals
        from the last weighted least squares fit in the IRLS algorithm.

    See Also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """

    def __init__(self, model, params, normalized_cov_params, scale, cov="H1"):
        super().__init__(model, params, normalized_cov_params, scale)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.nobs
        # cov is a public constructor argument (not only reachable through
        # the already-validated RLM.fit), so it needs its own validation
        self.cov = string_like(cov, "cov", options=("h1", "h2", "h3")).upper()
        # Snapshot the final IRLS weights now, rather than deferring to
        # model.weights, so this result is unaffected by any later fit()
        # call on the same model instance.
        self.weights = model.weights
        self._cache = {}
        # for remove_data
        self._data_in_cache.extend(["sresid"])
        self._data_attr.extend(["weights"])

        self.cov_params_default = self.bcov_scaled
        # TODO: "pvals" should come from chisq on bse?

    @cache_readonly
    def fittedvalues(self):
        return np.dot(self.model.exog, self.params)

    @cache_readonly
    def resid(self):
        return self.model.endog - self.fittedvalues  # before bcov

    @cache_readonly
    def sresid(self):
        if self.scale == 0.0:
            sresid = self.resid.copy()
            sresid[:] = 0.0
            return sresid
        return self.resid / self.scale

    @cache_readonly
    def bcov_unscaled(self):
        return self.normalized_cov_params

    @cache_readonly
    def bcov_scaled(self):
        model = self.model
        m = np.mean(model.M.psi_deriv(self.sresid))
        var_psiprime = np.var(model.M.psi_deriv(self.sresid))
        k = 1 + (self.df_model + 1) / self.nobs * var_psiprime / m**2

        if self.cov == "H1":
            ss_psi = np.sum(model.M.psi(self.sresid) ** 2)
            s_psi_deriv = np.sum(model.M.psi_deriv(self.sresid))
            return (
                k**2
                * (1 / self.df_resid * ss_psi * self.scale**2)
                / ((1 / self.nobs * s_psi_deriv) ** 2)
                * model.normalized_cov_params
            )
        else:
            W = np.dot(model.M.psi_deriv(self.sresid) * model.exog.T, model.exog)
            W_inv = np.linalg.inv(W)
            # [W_jk]^-1 = [SUM(psi_deriv(Sr_i)*x_ij*x_jk)]^-1
            # where Sr are the standardized residuals
            if self.cov == "H2":
                # These are correct, based on Huber (1973) 8.13
                return (
                    k
                    * (1 / self.df_resid)
                    * np.sum(model.M.psi(self.sresid) ** 2)
                    * self.scale**2
                    / ((1 / self.nobs) * np.sum(model.M.psi_deriv(self.sresid)))
                    * W_inv
                )
            else:  # self.cov == "H3"
                return (
                    k**-1
                    * 1
                    / self.df_resid
                    * np.sum(model.M.psi(self.sresid) ** 2)
                    * self.scale**2
                    * np.dot(np.dot(W_inv, np.dot(model.exog.T, model.exog)), W_inv)
                )

    @cache_readonly
    def pvalues(self):
        return stats.norm.sf(np.abs(self.tvalues)) * 2

    @cache_readonly
    def bse(self):
        # Use cov_params_default (a plain attribute snapshotted once in
        # __init__) rather than bcov_scaled directly, so bse still works
        # after remove_data() has cleared model.exog.
        return np.sqrt(np.diag(self.cov_params_default))

    @cache_readonly
    def chisq(self):
        return (self.params / self.bse) ** 2

    def summary(self, yname=None, xname=None, title=None, alpha=0.05, return_fmt="text"):
        """
        Summarize the fitted model

        Parameters
        ----------
        yname : str, optional
            Name of the dependent variable (optional)
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` where
            `##` is the 0-based index of the regressor. Must match the
            number of parameters in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float, optional
            Significance level for the confidence intervals
        return_fmt : str, optional
            Unused

        Returns
        -------
        Summary
            Instance holding the summary tables and text, which can be
            printed or converted to various output formats.
        """
        top_left = [
            ("Dep. Variable:", None),
            ("Model:", None),
            ("Method:", ["IRLS"]),
            ("Norm:", [self.fit_options["norm"]]),
            ("Scale Est.:", [self.fit_options["scale_est"]]),
            ("Cov Type:", [self.fit_options["cov"]]),
            ("Date:", None),
            ("Time:", None),
            ("No. Iterations:", ["{:d}".format(self.fit_history["iteration"])]),
        ]
        top_right = [
            ("No. Observations:", None),
            ("Df Residuals:", None),
            ("Df Model:", None),
        ]

        if title is None:
            title = "Robust Linear Model Regression Results"

        # boiler plate
        from statsmodels.iolib.summary import Summary

        smry = Summary()
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,
            yname=yname,
            xname=xname,
            title=title,
        )
        smry.add_table_params(
            self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t
        )

        # add warnings/notes, added to text format only
        etext = []
        wstr = (
            "If the model instance has been used for another fit with "
            "different fit parameters, then the fit options might not be "
            "the correct ones anymore ."
        )
        etext.append(wstr)

        if etext:
            smry.add_extra_txt(etext)

        return smry

    def summary2(
        self, xname=None, yname=None, title=None, alpha=0.05, float_format="%.4f"
    ):
        """
        Experimental summary function for regression results

        Parameters
        ----------
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` where
            `##` is the 0-based index of the regressor. Must match the
            number of parameters in the model
        yname : str, optional
            Name of the dependent variable (optional)
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float, optional
            Significance level for the confidence intervals
        float_format : str, optional
            Print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        from statsmodels.iolib import summary2

        smry = summary2.Summary()
        smry.add_base(
            results=self,
            alpha=alpha,
            float_format=float_format,
            xname=xname,
            yname=yname,
            title=title,
        )

        return smry


class RLMResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(RLMResultsWrapper, RLMResults)
