"""
Generalized Pareto regression for threshold exceedances

The model is the peaks-over-threshold model of extreme value theory in which
the scale, and optionally the shape, of the generalized Pareto distribution
depend on explanatory variables.

References
----------
Coles, Stuart. An Introduction to Statistical Modeling of Extreme Values.
Springer, 2001. Chapter 4 and Section 6.3.

Davison, A. C., and R. L. Smith. "Models for Exceedances over High
Thresholds." Journal of the Royal Statistical Society, Series B 52, no. 3
(1990): 393-442.

Hosking, J. R. M., and J. R. Wallis. "Parameter and Quantile Estimation for
the Generalized Pareto Distribution." Technometrics 29, no. 3 (1987): 339-349.

Smith, Richard L. "Maximum Likelihood Estimation in a Class of Nonregular
Cases." Biometrika 72, no. 1 (1985): 67-90.
"""

import warnings

import numpy as np
from scipy import stats

from statsmodels.base.model import (
    GenericLikelihoodModel,
    GenericLikelihoodModelResults,
)
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools._decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning

# For |w| below this bound the functions of w = shape * endog / scale that are
# 0 / 0 at w = 0 are evaluated by their Taylor series. 30 terms give a
# truncation error below 1e-25 at |w| = 0.1, and the closed forms lose at
# most about eps / w**2 relative precision above it.
_W_SMALL = 0.1
_N_TERMS = 30
# log1p(w) / w
_COEF_H = np.array([(-1) ** (n + 1) / n for n in range(1, _N_TERMS + 1)])
# (log1p(w) - w / (1 + w)) / w**2
_COEF_G = np.array([(-1) ** n * (n - 1) / n for n in range(2, _N_TERMS + 2)])
# (-2 log1p(w) + 2 w / (1 + w) + w**2 / (1 + w)**2) / w**3
_COEF_K = np.array([(-1) ** n * (n - 1) * (n - 2) / n for n in range(3, _N_TERMS + 3)])


def _series(w, coef):
    """Evaluate the power series sum_j coef[j] * w**j by Horner's rule."""
    out = np.zeros_like(w)
    for c in coef[::-1]:
        out = out * w + c
    return out


def _small_w_eval(w, coef, closed_form):
    """Use the series for small |w| and the closed form elsewhere."""
    small = np.abs(w) < _W_SMALL
    w_large = np.where(small, 1.0, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        large_val = closed_form(w_large)
    return np.where(small, _series(w, coef), large_val)


def _log1p_over_w(w):
    return _small_w_eval(w, _COEF_H, lambda v: np.log1p(v) / v)


def _g_func(w):
    return _small_w_eval(w, _COEF_G, lambda v: (np.log1p(v) - v / (1 + v)) / v**2)


def _k_func(w):
    def closed_form(v):
        u = 1 + v
        return (-2 * np.log1p(v) + 2 * v / u + v**2 / u**2) / v**3

    return _small_w_eval(w, _COEF_K, closed_form)


def _pwm_genpareto(y):
    """
    Probability weighted moment estimates of the generalized Pareto

    Hosking and Wallis (1987), with plotting positions (j - 0.35) / n.

    Parameters
    ----------
    y : ndarray
        1d array of nonnegative exceedances.

    Returns
    -------
    shape : float
        Estimate of the shape parameter, in the sign convention of
        ``scipy.stats.genpareto`` (positive shape is a heavy tail).
    scale : float
        Estimate of the scale parameter.
    """
    y = np.sort(y)
    n = y.shape[0]
    p = (np.arange(1, n + 1) - 0.35) / n
    a0 = y.mean()
    a1 = np.mean((1 - p) * y)
    denom = a0 - 2 * a1
    if not denom > 0:
        # degenerate sample, e.g. all values equal; fall back to exponential
        return 0.0, a0
    shape = 2 - a0 / denom
    scale = 2 * a0 * a1 / denom
    return shape, scale


class GeneralizedParetoModel(GenericLikelihoodModel):
    r"""
    Generalized Pareto regression for exceedances over a threshold

    The response is the exceedance ``y = x - u`` of an observation ``x``
    over a threshold ``u``. Conditional on the explanatory variables it
    follows a generalized Pareto distribution with scale
    :math:`\sigma_i = \exp(x_i' \beta)` and shape
    :math:`\xi_i = z_i' \gamma`. The density is

    .. math::

       f(y_i) = \frac{1}{\sigma_i}
                \left(1 + \xi_i \frac{y_i}{\sigma_i}\right)^{-1 - 1/\xi_i}

    on the support :math:`1 + \xi_i y_i / \sigma_i > 0`, and the
    exponential density :math:`\exp(-y_i / \sigma_i) / \sigma_i` at
    :math:`\xi_i = 0`.

    Parameters
    ----------
    endog : array_like
        1d array of exceedances over the threshold. Values must be finite
        and nonnegative.
    exog : array_like, optional
        A nobs x k array of explanatory variables for the log of the scale
        parameter. An intercept is not included by default and should be
        added by the user (models specified using a formula include an
        intercept by default). See `statsmodels.tools.add_constant`. If
        None, then the log-scale is a constant.
    exog_shape : array_like, optional
        A nobs x k_shape array of explanatory variables for the shape
        parameter, which uses the identity link. If None, then the shape is a
        constant.
    **kwds
        Extra keywords that are handled by the super classes, for example
        ``missing``.

    Attributes
    ----------
    k_shape : int
        The number of shape parameters, which are the last ``k_shape``
        elements of ``params``.

    Notes
    -----
    .. versionadded:: 0.15.1

    Status: experimental. The api can change.

    The parameters are ordered as the coefficients of ``exog`` for the
    log-scale followed by the coefficients of ``exog_shape`` for the shape.
    The shape follows the sign convention of ``scipy.stats.genpareto``,
    positive shape is a heavy, Pareto-type tail and negative shape is a
    tail with a finite upper end point.

    The model assumes that the threshold is high enough that the
    exceedances are approximately generalized Pareto, by the
    Pickands-Balkema-de Haan theorem, and that the exceedances are
    independent given the explanatory variables. Threshold choice and
    declustering of dependent exceedances are left to the user.

    The usual asymptotic normality of the maximum likelihood estimator, and
    so the standard errors based on the Hessian, hold for a shape above
    -0.5 (Smith 1985). For a shape between -1 and -0.5 the estimator exists
    but is not regular, and below -1 the likelihood is unbounded and the
    maximum likelihood estimator does not exist. ``fit`` warns if a
    fitted shape is below -0.5.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from statsmodels.extremes.api import GeneralizedParetoModel
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(size=500)
    >>> exog = np.column_stack([np.ones(500), x])
    >>> y = stats.genpareto.rvs(0.2, scale=np.exp(0.5 + x), random_state=rng)
    >>> res = GeneralizedParetoModel(y, exog).fit()
    >>> print(res.summary())
    """

    def __init__(self, endog, exog=None, exog_shape=None, **kwds):
        endog_arr = np.asarray(endog, dtype=float)
        if endog_arr.ndim != 1:
            raise ValueError("endog must be 1-dimensional")
        nobs = endog_arr.shape[0]
        if exog is None:
            exog = np.ones((nobs, 1))
        if exog_shape is None:
            extra_names = ["shape"]
            exog_shape = np.ones((nobs, 1))
        else:
            if np.ndim(exog_shape) == 1:
                exog_shape = np.asarray(exog_shape)[:, None]
            if hasattr(exog_shape, "columns"):
                cols = exog_shape.columns
            else:
                cols = range(1, np.shape(exog_shape)[1] + 1)
            extra_names = [f"shape-{c}" for c in cols]

        kwds["extra_params_names"] = extra_names
        super().__init__(endog, exog, exog_shape=exog_shape, **kwds)

        self.exog_shape = np.asarray(self.exog_shape, dtype=float)
        if self.exog_shape.shape[0] != self.endog.shape[0]:
            raise ValueError("exog_shape must have the same number of rows as endog")
        if not np.all(np.isfinite(self.endog)) or np.any(self.endog < 0):
            raise ValueError("endog must be finite and nonnegative exceedances")
        if not np.any(self.endog > 0):
            raise ValueError("endog must have at least one positive exceedance")

        self.k_shape = self.exog_shape.shape[1]
        self._data_attr.append("exog_shape")
        # As in BetaModel, the shape intercept counts as the one extra
        # parameter and shape slopes count in df_model.
        self.k_extra = 1
        self.nobs = self.endog.shape[0]
        self.df_model = self.nparams - self.k_constant - self.k_extra
        self.df_resid = self.nobs - self.nparams
        del kwds["extra_params_names"]
        self._check_kwargs(kwds)
        self.results_class = GeneralizedParetoResults
        self.results_class_wrapper = GeneralizedParetoResultsWrapper

    def _get_exogs(self):
        return (self.exog, self.exog_shape)

    def _split_params(self, params):
        # Uses k_shape, not exog, so that it works after remove_data.
        params = np.asarray(params, dtype=float)
        k = len(params) - self.k_shape
        return params[:k], params[k:]

    def _linpreds(self, params, exog=None, exog_shape=None):
        """Return the log-scale and the shape for each observation."""
        if exog is None:
            exog = self.exog
        if exog_shape is None:
            exog_shape = self.exog_shape
        params_scale, params_shape = self._split_params(params)
        linpred = np.dot(exog, params_scale)
        shape = np.dot(exog_shape, params_shape)
        return linpred, shape

    def _parts(self, params):
        """Return t = y / scale, w = shape * t, and in_support for params."""
        linpred, shape = self._linpreds(params)
        t = self.endog * np.exp(-linpred)
        w = shape * t
        in_support = w > -1
        return linpred, shape, t, w, in_support

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of the generalized Pareto regression

        Parameters
        ----------
        params : array_like
            The parameters of the model, the coefficients of the log-scale
            followed by the coefficients of the shape.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. Observations outside of the support have a
            loglikelihood of ``-inf``.
        """
        linpred, _, t, w, ok = self._parts(params)
        w_ok = np.where(ok, w, 0.0)
        llf = -linpred - np.log1p(w_ok) - t * _log1p_over_w(w_ok)
        return np.where(ok, llf, -np.inf)

    def score_factor(self, params):
        """
        Derivatives of the loglikelihood with respect to the linear predictors

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        sf_scale : ndarray
            Derivative with respect to the linear predictor of the log-scale.
        sf_shape : ndarray
            Derivative with respect to the shape.

        Notes
        -----
        The values are nan for observations outside of the support.
        """
        _, _, t, w, ok = self._parts(params)
        w_ok = np.where(ok, w, 0.0)
        u = 1 + w_ok
        sf_scale = (t - 1) / u
        sf_shape = t**2 * _g_func(w_ok) - t / u
        return (
            np.where(ok, sf_scale, np.nan),
            np.where(ok, sf_shape, np.nan),
        )

    def score_obs(self, params):
        """
        Score, first derivative of the loglikelihood for each observation

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        score_obs : ndarray, 2d
            The first derivative of the loglikelihood function evaluated at
            params for each observation.
        """
        sf1, sf2 = self.score_factor(params)
        return np.column_stack(
            (sf1[:, None] * self.exog, sf2[:, None] * self.exog_shape)
        )

    def score(self, params):
        """
        Score, first derivative of the loglikelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        score : ndarray
            The first derivative of the loglikelihood function evaluated at
            params.
        """
        sf1, sf2 = self.score_factor(params)
        return np.concatenate((sf1 @ self.exog, sf2 @ self.exog_shape))

    def hessian_factor(self, params):
        """
        Second derivatives of the loglikelihood w.r.t. the linear predictors

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        tuple of ndarray
            The three factors ``(h_scale, h_cross, h_shape)`` for the
            log-scale, cross and shape parts of the observed Hessian.
        """
        _, shape, t, w, ok = self._parts(params)
        w_ok = np.where(ok, w, 0.0)
        u = 1 + w_ok
        h11 = -(1 + shape) * t / u**2
        h12 = -t * (t - 1) / u**2
        h22 = t**3 * _k_func(w_ok) + t**2 / u**2
        return tuple(np.where(ok, h, np.nan) for h in (h11, h12, h22))

    def hessian(self, params):
        """
        Observed Hessian, second derivative of the loglikelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        hessian : ndarray, 2d
            The Hessian of the loglikelihood evaluated at params.
        """
        h11, h12, h22 = self.hessian_factor(params)
        x, z = self.exog, self.exog_shape
        d11 = (x.T * h11) @ x
        d12 = (x.T * h12) @ z
        d22 = (z.T * h22) @ z
        return np.block([[d11, d12], [d12.T, d22]])

    def _start_params(self):
        """
        Starting values inside of the support

        The log-scale linear predictor starts from a least squares fit of
        ``log(endog)`` on ``exog``, rescaled by a probability weighted moment
        fit of the standardized exceedances. The shape starts at that
        constant estimate, or at zero if the constant estimate puts an
        observation outside of the support.
        """
        y = self.endog
        x, z = self.exog, self.exog_shape
        # exceedances at zero would give log(0); __init__ ensures mean(y) > 0
        logy = np.log(np.maximum(y, 1e-8 * y.mean()))
        fitted = x @ np.linalg.lstsq(x, logy, rcond=None)[0]
        y_std = y * np.exp(-fitted)
        shape0, scale0 = _pwm_genpareto(y_std)
        # PWM exists for shape < 1, keep the start in the regular region
        shape0 = float(np.clip(shape0, -0.45, 0.9))
        target = fitted + np.log(max(scale0, 1e-12))
        params_scale = np.linalg.lstsq(x, target, rcond=None)[0]
        params_shape = np.linalg.lstsq(z, np.full(self.nobs, shape0), rcond=None)[0]
        sp = np.concatenate((params_scale, params_shape))
        if not np.all(self._parts(sp)[4]):
            sp = np.concatenate((params_scale, np.zeros(self.k_shape)))
        return sp

    def fit(
        self,
        start_params=None,
        method="bfgs",
        maxiter=1000,
        disp=False,
        refine=True,
        **kwds,
    ):
        """
        Fit the model by maximum likelihood

        Parameters
        ----------
        start_params : array_like, optional
            Starting values for the parameters, the coefficients of the
            log-scale followed by the coefficients of the shape. The default
            is computed by ``_start_params``. All observations must be
            inside of the support at the starting values.
        method : str, optional
            The optimization method, see `LikelihoodModel.fit`. Default is
            "bfgs".
        maxiter : int, optional
            The maximum number of iterations. Default is 1000.
        disp : bool, optional
            If True, then convergence messages are printed. Default is False.
        refine : bool, optional
            If True (default) and `method` is not "newton", then the estimate
            of a converged optimization is refined by Newton steps with the
            analytic Hessian, and the result with the larger loglikelihood
            is returned. The gradient tolerance of "bfgs" alone leaves
            relative errors of about 1e-5 in the estimates.
        **kwds
            Additional keyword arguments for the optimizer and
            `LikelihoodModel.fit`. Only ``cov_type`` and ``cov_kwds`` are
            passed on to the refinement.

        Returns
        -------
        GeneralizedParetoResults
            The results instance.

        Raises
        ------
        ValueError
            If an observation is outside of the support at `start_params`.

        Warns
        -----
        EstimationWarning
            If a fitted shape is below -0.5, where the standard errors are
            not valid.
        """
        if start_params is None:
            start_params = self._start_params()
        else:
            start_params = np.asarray(start_params, dtype=float)
            if start_params.shape != (self.nparams,):
                raise ValueError(f"start_params must have length {self.nparams}")
            if not np.all(self._parts(start_params)[4]):
                raise ValueError(
                    "start_params put observations outside of the support, "
                    "1 + shape * endog / scale must be positive for all "
                    "observations"
                )

        res = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            disp=disp,
            **kwds,
        )
        if refine and method != "newton" and res.mle_retvals["converged"]:
            cov_kwds = {
                key: kwds[key] for key in ("cov_type", "cov_kwds") if key in kwds
            }
            # A refinement that does not converge is discarded below, so its
            # convergence warning is not relevant to the user.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                res_newton = super().fit(
                    start_params=res.params,
                    method="newton",
                    maxiter=50,
                    disp=disp,
                    **cov_kwds,
                )
            res = max(res, res_newton, key=lambda r: r.llf)
        shape_fitted = self.predict(res.params, which="shape")
        if np.min(shape_fitted) < -0.5:
            warnings.warn(
                "A fitted shape is below -0.5. The maximum likelihood "
                "estimator is not regular there and the standard errors "
                "are not valid.",
                EstimationWarning,
                stacklevel=2,
            )
        return GeneralizedParetoResultsWrapper(res)

    def predict(self, params, exog=None, exog_shape=None, which="mean"):
        """
        Predict the mean or the parameters of the distribution

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like, optional
            Explanatory variables for the log-scale. If None, then the
            model's ``exog`` is used.
        exog_shape : array_like, optional
            Explanatory variables for the shape. If None, then the model's
            ``exog_shape`` is used.
        which : {"mean", "scale", "shape", "linear"}, optional
            Statistic to predict. Default is "mean".

            - "mean" : mean exceedance ``scale / (1 - shape)``, which is
              infinite if the shape is 1 or larger.
            - "scale" : scale parameter.
            - "shape" : shape parameter.
            - "linear" : linear predictor of the log-scale.

        Returns
        -------
        ndarray
            The predicted values.
        """
        if exog is not None:
            exog = np.atleast_2d(np.asarray(exog, dtype=float))
        if exog_shape is not None:
            exog_shape = np.asarray(exog_shape, dtype=float)
            if exog_shape.ndim == 1:
                exog_shape = exog_shape[:, None]
        if exog is not None and exog_shape is None and self.k_shape == 1:
            exog_shape = np.ones((exog.shape[0], 1))
        linpred, shape = self._linpreds(params, exog=exog, exog_shape=exog_shape)
        if which == "linear":
            return linpred
        if which == "scale":
            return np.exp(linpred)
        if which == "shape":
            return shape
        if which == "mean":
            scale = np.exp(linpred)
            with np.errstate(divide="ignore"):
                return np.where(shape < 1, scale / (1 - shape), np.inf)
        raise ValueError(f"which = {which} is not available")

    def get_distribution(self, params, exog=None, exog_shape=None):
        """
        Frozen scipy distribution of the predictive distribution

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like, optional
            Explanatory variables for the log-scale.
        exog_shape : array_like, optional
            Explanatory variables for the shape.

        Returns
        -------
        scipy.stats.rv_frozen
            Frozen ``scipy.stats.genpareto`` with one shape and scale per
            observation.
        """
        scale = self.predict(params, exog=exog, exog_shape=exog_shape, which="scale")
        shape = self.predict(params, exog=exog, exog_shape=exog_shape, which="shape")
        return stats.genpareto(shape, scale=scale)


class GeneralizedParetoResults(GenericLikelihoodModelResults):
    """
    Results class for the generalized Pareto regression

    This class inherits from GenericLikelihoodModelResults and not all
    inherited methods might be appropriate in this case.
    """

    @cache_readonly
    def fitted_scale(self):
        """In-sample predicted scale"""
        return self.model.predict(self.params, which="scale")

    @cache_readonly
    def fitted_shape(self):
        """In-sample predicted shape"""
        return self.model.predict(self.params, which="shape")

    @cache_readonly
    def resid_exponential(self):
        """
        Residuals that are standard exponential if the model is correct

        Computed as ``log(1 + shape * endog / scale) / shape``, which is
        ``endog / scale`` at shape zero. These are the residuals used in
        exponential quantile plots for nonstationary threshold models
        (Coles 2001, Section 6.2.3).
        """
        t = self.model.endog / self.fitted_scale
        w = self.fitted_shape * t
        return t * _log1p_over_w(w)

    def get_distribution(self, exog=None, exog_shape=None):
        """
        Frozen scipy distribution of the predictive distribution

        Parameters
        ----------
        exog : array_like, optional
            Explanatory variables for the log-scale.
        exog_shape : array_like, optional
            Explanatory variables for the shape.

        Returns
        -------
        scipy.stats.rv_frozen
            Frozen ``scipy.stats.genpareto`` with one shape and scale per
            observation.
        """
        return self.model.get_distribution(
            self.params, exog=exog, exog_shape=exog_shape
        )

    def bootstrap(self, *args, **kwargs):
        raise NotImplementedError


class GeneralizedParetoResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(GeneralizedParetoResultsWrapper, GeneralizedParetoResults)
