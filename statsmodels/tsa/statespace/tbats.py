"""
Trigonometric, Box-Cox, ARMA Error, Trend and Seasonal (TBATS) Model

Author: Leoyzen Liu
License: Simplified-BSD
"""

from __future__ import division, absolute_import, print_function

from warnings import warn
from collections import OrderedDict

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.eval_measures import aic
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
import statsmodels.base.wrapper as wrap
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.transform import BoxCox
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate,
    unconstrain_stationary_univariate,
    solve_discrete_lyapunov,
    companion_matrix,
)
from statsmodels.tsa.stattools import _safe_arma_fit
from statsmodels.tsa.tsatools import detrend, lagmat

from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper, _handle_args
from .tools import fourier


def constrain_bound(x, low=0, high=1):
    from scipy.special import expit

    return expit(x) * (high - low) + low


def unconstrain_bound(x, lower=0, upper=1):
    from scipy.special import logit

    x = (x - lower) / (upper - lower)
    return logit(x)


class PeriodWrapper(object):
    def __init__(self, name, attribute, dtype=np.float32, multiple=False):
        self.name = name
        self.attribute = attribute
        self._attribute = "_" + attribute
        self._multiple = multiple
        self.dtype = dtype

    def __set__(self, obs, val):
        if val is None:
            k_period = 0
        elif np.isscalar(val):
            if self._multiple:
                val = np.asarray([val], dtype=self.dtype)
            else:
                val = self.dtype(val)
            k_period = 1
        else:
            if self._multiple:
                val = np.asarray(val, dtype=self.dtype)
            else:
                raise ValueError("%s must be a scalar of %s" % (self.name, self.dtype))
            k_period = len(val)
        setattr(obs, "k_period", k_period)
        setattr(obs, self._attribute, val)
        setattr(obs, "period_names", ["Seasonal"] * k_period)

    def __get__(self, obj, objtype):
        return getattr(obj, self._attribute, None)


def _tbats_seasonal_matrix(periods, k, dtype=np.float32):
    n_period = len(periods)
    n_k = len(k)
    tau = 2 * np.sum(k)
    if n_k != n_period:
        raise ValueError
    design = []
    transition = []
    for m, k in zip(periods, k):
        design = np.append(design, np.r_[np.ones(k), np.zeros(k)])
        lmbda = [2 * np.pi * (j + 1) / m for j in range(k)]
        c = np.diag(np.cos(lmbda))
        s = np.diag(np.sin(lmbda))
        A = np.bmat([[c, s], [-s, c]])
        transition.append(A)
    return tau, design, linalg.block_diag(*transition).astype(dtype)


def _bats_seasonal_matrix(periods):
    periods = np.asarray(periods, dtype="int")
    design = []
    transition = []
    tau = np.sum(periods)
    for m in periods:
        design = np.append(design, np.r_[np.zeros(m - 1), 1])
        A = np.transpose(companion_matrix(int(m)))
        A[0, -1] = 1
        transition.append(A)
    return tau, design, linalg.block_diag(*transition).astype("int")


def _k_f_test(x, periods, ks, exog=None, wls=False):
    for i, (period, k) in enumerate(zip(periods, ks)):
        if exog is None:
            exog = fourier(x, period, k, name="Seasonal" + str(i))
        else:
            exog = exog.join(
                fourier(x, period, k, name="Seasonal" + str(i)), how="outer"
            )
    if not wls:
        res = OLS(x, add_constant(exog)).fit(cov_kwd={"use_t": True})
    else:
        res = RLM(x, add_constant(exog))
        res = WLS(x, res.exog, weights=res.fit().weights).fit()
    return res


def seasonal_fourier_k_select_ic(y, periods, ic="aic", grid_search=False):
    periods = np.asarray(periods)
    ks = np.ones_like(periods, dtype=int)
    _ic = lambda obj: getattr(obj, ic)
    result = dict()
    max_k = np.ones_like(ks)

    for i, period in enumerate(periods):

        max_k[i] = period // 2
        # if i != 0:
        #     current_k = 2
        #     while current_k <= max_k[i]:
        #         if period % current_k != 0:
        #             current_k += 1
        #             continue

        #         latter = period / current_k

        #         if np.any(periods[:i] % latter == 0):
        #             max_k[i] = current_k - 1
        #             break
        #         else:
        #             current_k += 1
    print(max_k)
    resid = y
    current_resid = None
    last_diff = np.inf
    for i, period in enumerate(periods):
        base_test = _k_f_test(resid, (period,), (1,))
        min_ic = _ic(base_test)
        current_resid = base_test.resid
        # start search from 2
        current_k = 2
        while current_k <= max_k[i]:
            _ks = ks.copy()
            _ks[i] = current_k
            try:
                test = _k_f_test(resid, (period,), (current_k,))
                current_ic = _ic(test)
                ic_diff = np.abs(min_ic - current_ic)
                result.update({str(tuple(_ks.tolist())): current_ic})
                if test.f_pvalue > 0.001:
                    # not significant, continue searching
                    current_k += 1
                    continue
                if current_ic <= min_ic:
                    last_diff = ic_diff
                    min_k = current_k
                    ks[i] = min_k
                    min_ic = current_ic
                    current_resid = test.resid
                elif ic_diff < last_diff:
                    last_diff = ic_diff
                    # minor up, try one step more
                    continue
                elif not grid_search:
                    break
            except (ValueError, LinAlgError) as e:
                break
            finally:
                current_k += 1
        resid = current_resid
    print(ks, result)
    return ks


class TBATSModel(MLEModel, BoxCox):
    r"""
    Trigonometric, Box-Cox, ARMA Error, Trend and Seasonal(TBATS)

    This model incorporates BoxCox transformations, Fourier representations with time varying coefﬁcients, and ARMA error correction.

    Parameters
    ----------
    periods:   iterable of float or None or array, optional
        The period of the seasonal component. Default is None. Can be multiple
    k:  int or None, optional, should be same length of periods.
        The number of Fourier series paris(West & Harrison, 1997),(Harvey, 1989)
    order:  iterable, optional
        The (p,q) order of the model for the number of AR parameters, and MA parameters of error.
    period_names:    iterable or None, optional
        The names used for periods, should be the same length of periods

    use_exact_diffuse : bool, optional
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).

    See Also
    --------
    statsmodels.tsa.statespace.tbats.TBATSResults
    statsmodels.tsa.statespace.mlemodel.MLEModel

    Notes
    -----
    
    These models take the general form (see [1]_ Chapter 2.2 for all details)
    
    .. math::
        \begin{equation}
        y^{(\omega)}_t = \begin{cases}
        \frac{y^\omega_t - 1}{\omega};     \omega \ne 0 \\
        log{y_t};     \omega = 0
        \end{cases}        
        \end{equation} \\
        \begin{equation}
        y^{(\omega)}_t = l_{t-1} + \phi b_{t-1} + \sum^{T}_{i=1}s^{(i)}_{t-m_i} + d_t \\
        \end{equation} \\
        \begin{equation}
        l = l_{t-1} + \phi b_{t-1} + \alpha d_t \\        
        \end{equation} \\
        \begin{equation}
        b_t = (1-\phi)b + \phi b_{t-1} + \beta d_t \\        
        \end{equation} \\
        \begin{equation}
        s^{(i)}_t = s^{(i)}_{t-m_i} + \gamma_id_t \\   
        \end{equation} \\
        \begin{equation}     
        d_t = \sum^p_{i=1}\psi_id_{t-i} + \sum^q_{i-1}\theta_i\epsilon_{t-i}+\epsilon_t        
        \end{equation} \\
            
    where :math:`l_t` is the is the local level in period :math:`t`, :math: `b` is the long-run trend, :math:`b_t` is the short-run trend in period :math:`t`, :math:`s^{(i)}_t` represents the :math:`i`th seasonal component at time :math:`t`, :math:`d_t` denotes an ARMA(p, q) process and :math:`\epsilon_t` is a Gaussian white noise process with zero mean and constant variance :math:`\sigma^2`.
    
    The smoothing parameters are given by :math:`\alpha, \beta` and :math:`\gamma_i` for :math:`i=1,...,T`. :math:`\phi` is daping parameter for damped trend. 
    
    The damping factor is included in the level and measurement equations as well as the trend equation for consistency with [2]_, but identical predictions are obtained if it is excluded from the level and measurement equations.
    
    References
    ----------
    .. [1] Alysha M De Livera, Rob J Hyndman and Ralph D snyder. 2010
       Forecasting time series with complex seasonal patterns using exponential smoothing.
       MONASH University.
    """

    periods = PeriodWrapper("Seasonal Periods", "periods", np.float32, multiple=True)

    def __init__(
        self,
        endog,
        periods=None,
        k=None,
        order=(0, 0),
        period_names=None,
        use_trend=True,
        damped_trend=True,
        box_cox=True,
        exog=None,
        mle_regression=True,
        use_exact_diffuse=False,
        bc_bounds=(0, 1),
        enforce_stationarity=True,
        initial_params=None,
        initial_states=None,
        irregular=False,
        concentrate_scale=False,
        **kwargs,
    ):

        # Model options
        self.seasonal = periods is not None
        self.trend = use_trend
        self.boxcox = box_cox
        self.damping = damped_trend
        self.mle_regression = mle_regression
        self.use_exact_diffuse = use_exact_diffuse
        self.autoregressive = order != (0, 0)
        self.ar_order = order[0]
        self.periods = periods
        self.k = k
        self.order = order
        self.bc_bounds = bc_bounds
        self.irregular = irregular
        self.enforce_stationarity = enforce_stationarity
        self.concentrate_scale = concentrate_scale
        # level alpha for
        k_states = 1

        # Exogenous component
        self.k_exog = 0
        if exog is not None:
            exog_is_using_pandas = _is_using_pandas(exog, None)
            if not exog_is_using_pandas:
                exog = np.asarray(exog)

            # Make sure we have 2-dimensional array
            if exog.ndim == 1:
                if not exog_is_using_pandas:
                    exog = exog[:, None]
                else:
                    exog = pd.DataFrame(exog)

            self.k_exog = exog.shape[1]
        self.regression = self.k_exog > 0

        if periods is not None:
            self.seasonal = True
            self.k = k
            k_states += int(2 * np.sum(self.k))
            self.max_period = max(self.periods)
            if period_names is not None:
                self.period_names = period_names
        else:
            self.seasonal = False
            self.max_period = 0
            self.periods = (0,)
            self.k = (0,)

        self.trend = use_trend & True
        if self.trend:
            k_states += 1

        self.autoregressive = np.sum(order) != 0
        self.order = order

        p, q = self.order

        if self.autoregressive:
            k_states += p + q

        k_states = (
            1
            + self.trend
            + np.sum(self.k) * 2
            + np.sum(self.order)
            + (not self.mle_regression) * self.k_exog
        )
        k_posdef = 1
        # k_posdef = k_states
        # Handle non-default loglikelihood burn
        self._loglikelihood_burn = kwargs.get("loglikelihood_burn", None)
        # loglikelihood_burn = kwargs.get('loglikelihood_burn', int(self.max_period) + p + q)

        self.initial_params = initial_params
        self.initial_states = initial_states
        # Initialize the model base
        super().__init__(
            endog=endog,
            exog=exog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            **kwargs,
        )

        if self.concentrate_scale:
            self.ssm.filter_concentrated = True

        if box_cox:
            if np.any(self.data.endog <= 0):
                warn("To use boxcox transformation the endog must be positive")
                self.boxcox = False
            else:
                # use to calculate loglikeobs when using boxcox
                self.data.log_endog = np.log(self.data.endog)
        else:
            self.boxcox = False

        # self.ssm.loglikelihood_burn = loglikelihood_burn
        self.setup()
        # Initialize the state
        # self.initialize_state()
        self.initialize_default()

    def initialize(self):
        """
        Initialize the SARIMAX model.

        Notes
        -----
        These initialization steps must occur following the parent class
        __init__ function calls.
        """
        super().initialize()
        offset = 0

        # level
        self._level_state_offset = offset
        offset += 1

        # trend
        self._trend_state_offset = offset
        ## dampping trend
        offset += 1

        if self.seasonal:
            tau = int(2 * np.sum(self.k))

            self._seasonal_state_offset = offset

            offset += tau

        if self.autoregressive:
            p, q = self.order

            if p:
                self._ar_state_offset = offset
                offset += p
            if q:
                self._ma_state_offset = offset
                offset += q

    def setup(self):

        self.parameters = OrderedDict()
        self.parameters_obs_intercept = OrderedDict()

        # Initialize the fixed components of the state space matrices,
        i = 0  # state offset
        j = 0  # state covariance offset

        offset = 0

        if self.concentrate_scale:
            self["state_cov", 0, 0] = 1
        else:
            self.parameters["state_var"] = 1

        if self.irregular:
            self.parameters["obs_var"] = 1

        if self.boxcox:
            self.parameters["boxcox"] = 1

        # Level Setup
        self.parameters["level_alpha"] = 1
        self["design", 0, 0] = 1
        self["transition", 0, 0] = 1
        offset += 1

        # Trend Setup
        if self.trend:
            self.parameters["trend_beta"] = 1
            # self.parameters["trend_intercept"] = 1

            self["design", 0, offset] = 1
            self["transition", :2, 1] = 1
            self["selection", 1, 0] = 1
            offset += 1
            if self.damping:
                self.parameters["damping"] = 1

        if self.seasonal:
            tau, season_design, season_transition = _tbats_seasonal_matrix(
                self.periods, self.k
            )
            self["design", 0, offset : offset + tau] = season_design
            self["transition", offset : offset + tau, offset : offset + tau] = (
                season_transition
            )
            self.parameters["seasonal_gamma"] = 2 * self.k_period
            offset += tau

        if self.autoregressive:
            p, q = self.order
            start = 1 + self.trend
            if self.seasonal:
                start += tau
            if p:
                if p > 1:
                    np.fill_diagonal(
                        self["transition", start + 1 : start + p, start : start + p], 1
                    )
                self["selection", start, 0] = 1
                self.parameters["ar_coeff"] = p
                start += p
            if q:
                if q > 1:
                    np.fill_diagonal(
                        self["transition", start + 1 : start + q, start : start + q], 1
                    )
                self["selection", start, 0] = 1
                self.parameters["ma_coef"] = q

        if self.regression:
            if self.mle_regression:
                self.parameters_obs_intercept["reg_coeff"] = self.k_exog
            else:
                design = np.repeat(self.ssm["design", :, :, 0], self.nobs, axis=0)
                self.ssm["design"] = design.transpose()[np.newaxis, :, :]
                self.ssm["design", 0, offset : offset + self.k_exog, :] = (
                    self.exog.transpose()
                )
                self.ssm[
                    "transition",
                    offset : offset + self.k_exog,
                    offset : offset + self.k_exog,
                ] = np.eye(self.k_exog)

                offset += self.k_exog

        self.parameters.update(self.parameters_obs_intercept)

        self.k_obs_intercept = sum(self.parameters_obs_intercept.values())
        self.k_params = sum(self.parameters.values())

    def initialize_state(self, boxcox_lambda=None, initialization=None):
        from scipy.stats import linregress

        if self.boxcox:
            y, _ = self.transform_boxcox(self.data.endog, lmbda=boxcox_lambda)
        else:
            y = self.data.endog

        if self.seasonal:
            max_period = max(self.periods)
            resid, trend = hpfilter(y, lamb=(2 * np.sin(np.pi / (max_period))) ** (-4))
        else:
            resid, trend = hpfilter(y)

        slope, intercept, *_ = linregress(np.arange(trend.shape[0]), trend)

        offset = 0
        init = initialization or self.ssm.initialization

        init.set(offset, "known", [intercept])
        offset += 1

        if self.trend:
            init.set(offset, "known", [slope])
            offset += 1

        if self.seasonal:
            ols_mod = _k_f_test(resid, self.periods, self.k)
            tau = 2 * np.sum(self.k)
            init.set((offset, offset + tau), "known", ols_mod.params[1:])

    def initialize_default(self, approximate_diffuse_variance=None):
        if approximate_diffuse_variance is None:
            approximate_diffuse_variance = self.ssm.initial_variance
        if self.use_exact_diffuse:
            diffuse_type = "diffuse"
        else:
            diffuse_type = "approximate_diffuse"

            # Set the loglikelihood burn parameter, if not given in constructor
            if self._loglikelihood_burn is None:
                self.loglikelihood_burn = (
                    int(self.max_period)
                    + self.k_states
                    + np.sum(self.order)
                    # self.k_states
                )

        init = Initialization(
            self.k_states, approximate_diffuse_variance=approximate_diffuse_variance
        )

        offset = 1 + self.trend + 2 * np.sum(self.k)
        if self.autoregressive and self.enforce_stationarity:
            # start = offset
            length = self.order[0]
            # we set in initialize_state call
            init.set((0, offset), diffuse_type)
            init.set((offset, offset + length), "stationary")
            init.set((offset + length, self.k_states), diffuse_type)
        else:
            init.set((0, self.k_states), diffuse_type)

        # self.initialize_state(initialization=init)
        self.ssm.initialization = init

    def filter(self, params, **kwargs):
        kwargs.setdefault("results_class", TBATSResults)
        kwargs.setdefault("results_wrapper_class", TBATSResultsWrapper)
        return super().filter(params, **kwargs)

    def smooth(self, params, **kwargs):
        kwargs.setdefault("results_class", TBATSResults)
        kwargs.setdefault("results_wrapper_class", TBATSResultsWrapper)
        return super().smooth(params, **kwargs)

    def loglike(self, params, *args, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        See Also
        --------
        update : modifies the internal state of the state space model to
                 reflect new params

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        transformed, includes_fixed, complex_step, kwargs = _handle_args(
            MLEModel._loglike_param_names,
            MLEModel._loglike_param_defaults,
            *args,
            **kwargs,
        )

        params = self.handle_params(
            params, transformed=transformed, includes_fixed=includes_fixed
        )
        self.update(
            params, transformed=True, includes_fixed=True, complex_step=complex_step
        )

        if complex_step:
            kwargs["inversion_method"] = INVERT_UNIVARIATE | SOLVE_LU

        sse = self.ssm.loglike(complex_step=complex_step, **kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues, but the averaging is done
        # automatically in the base model `fit` method

        if self.boxcox:
            lmbda = params[(not self.concentrate_scale) + self.irregular]
            sse += (lmbda - 1) * np.nansum(
                self.data.log_endog[self.loglikelihood_burn :]
            )
        return sse

    def loglikeobs(
        self,
        params,
        transformed=True,
        includes_fixed=False,
        complex_step=False,
        **kwargs,
    ):
        params = self.handle_params(
            params, transformed=transformed, includes_fixed=includes_fixed
        )

        loglikeobs = super().loglikeobs(
            params,
            transformed=True,
            complex_step=complex_step,
            includes_fixed=includes_fixed,
            **kwargs,
        )

        if self.boxcox:
            lmbda = params[(not self.concentrate_scale) + self.irregular]
            bc_loglikeobs = (lmbda - 1) * self.data.log_endog
            loglikelihood_burn = kwargs.get(
                "loglikelihood_burn", self.loglikelihood_burn
            )
            bc_loglikeobs[:loglikelihood_burn] = 0
            loglikeobs += bc_loglikeobs
        return loglikeobs

    def fit(self, *args, **kwargs):
        kwargs.setdefault("maxiter", (self.k_params * 200) ** 2)
        return super().fit(*args, **kwargs)

    @property
    def start_params(self):

        if not hasattr(self, "parameters"):
            return []

        _start_params = {}

        endog = self.endog
        exog = self.exog

        if not self.concentrate_scale:
            _start_params["state_var"] = 1

        if self.irregular:
            _start_params["obs_var"] = 1

        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[~np.isnan(endog)]
            if exog is not None:
                exog = exog[mask]

        if self.boxcox:
            endog, _start_params["boxcox"] = self.transform_boxcox(
                self.data.endog, method="loglik", bounds=self.bc_bounds
            )
        # _start_params["level_alpha"] = 0.09
        _start_params["level_alpha"] = 0.09

        if self.seasonal:
            decomp_mod = seasonal_decompose(
                endog, period=int(np.max(self.max_period)), extrapolate_trend="freq"
            )
            trend1 = decomp_mod.trend
            resid = np.squeeze(decomp_mod.resid)
            # resid, trend1 = hpfilter(
            #     endog, lamb=(2 * np.sin(np.pi / (max(self.periods)))) ** (-4)
            # )
        else:
            resid = np.squeeze(detrend(endog, order=2))

        # level_var = np.var(trend1)
        # trend beta
        if self.trend:
            # cycle2, trend2 = hpfilter(trend1)
            _start_params["trend_beta"] = 0.05
            # we use the parameter matches hpfilter recommandation
            # _start_params["trend_beta"] = 1 / (2 * np.sin(np.pi / (max(self.periods)))) ** (-4)
            # trend_var = np.var(trend2)
            # level_var = np.var(cycle2)
            # _start_params["trend_intercept"] = 0
            if self.damping:
                _start_params["damping"] = 0.999

        # seasonal gammas
        if self.seasonal:
            n = self.k_period
            n_k = np.sum(self.k)
            _start_params["seasonal_gamma"] = [0.001] * n * 2
            resid = np.squeeze(_k_f_test(resid, self.periods, self.k).resid)
            # tau = 2 * np.sum(self.k)

            # _start_params['seasonal_gamma'] = ols_mod.params[1:]

        # _start_params["level_alpha"] = min(level_var / var_resid, 0.09)
        # if self.trend:
        # _start_params["trend_beta"] = min(trend_var / var_resid, 0.05)

        # Regression
        if self.regression and self.mle_regression:
            _start_params["reg_coeff"] = np.linalg.pinv(self.exog).dot(resid).tolist()
            resid = np.squeeze(resid - np.dot(self.exog, _start_params["reg_coeff"]))

        # AR
        if self.autoregressive:
            p, q = self.order
            if p:
                Y = resid[p:]
                X = lagmat(resid, p, trim="both")
                _start_params["ar_coeff"] = np.linalg.pinv(X).dot(Y).tolist()
                resid = np.squeeze(Y - np.dot(X, _start_params["ar_coeff"]))
            # if p:
            #     _start_params["ar_coeff"] = [0.01] * p
            if q:
                _start_params["ma_coef"] = [0.01] * q

        var_resid = np.var(resid)

        # if not self.concentrate_scale:
        #     _start_params["state_var"] = var_resid
        #     if self.irregular:
        #         _start_params["obs_var"] = var_resid
        # else:
        #     self["state_cov", 0, 0] = var_resid

        start_params = []
        for key in self.parameters.keys():
            if np.isscalar(_start_params[key]):
                start_params.append(_start_params[key])
            else:
                start_params.extend(_start_params[key])
        return start_params

    @property
    def param_names(self):

        if not hasattr(self, "parameters"):
            return []
        param_names = []
        p, q = self.order
        for key in self.parameters.keys():
            if key == "state_var":
                param_names.append("state.var")
            elif key == "obs_var":
                param_names.append("irregular.var")
            elif key == "level_alpha":
                param_names.append("alpha.level")
            elif key == "trend_beta":
                param_names.append("beta.trend")
            # elif key == "trend_intercept":
            #     param_names.append("intercept.trend")
            elif key == "damping":
                param_names.append("phi.damping")
            elif key == "seasonal_gamma":
                for i, name in enumerate(self.period_names, 1):
                    param_names.append("gamma1.{}".format(name))
                    param_names.append("gamma2.{}".format(name))
            elif key == "ar_coeff":
                for i in range(p):
                    param_names.append("ar.L%d.coefs" % (i + 1))
            elif key == "boxcox":
                param_names.append("lambda.boxcox")
            elif key == "ma_coef":
                for i in range(q):
                    param_names.append("ma.L%d.coefs" % (i + 1))
            elif key == "reg_coeff":
                param_names += [
                    "beta.%s.coefs" % self.exog_names[i] for i in range(self.k_exog)
                ]
            else:
                param_names.append(key)
        return param_names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation.

        Used primarily to enforce stationarity of the autoregressive lag
        polynomial, invertibility of the moving average lag polynomial, and
        positive variance parameters.

        Parameters
        ----------
        unconstrained : array_like
            Unconstrained parameters used by the optimizer.

        Returns
        -------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros_like(unconstrained)

        offset = 0
        # print(unconstrained)

        # Positive parameters: obs_cov, state_cov
        if not self.concentrate_scale:
            constrained[offset] = unconstrained[offset] ** 2
            offset += 1

        if self.irregular:
            constrained[offset] = unconstrained[offset] ** 2
            offset += 1

        if self.boxcox:
            # constrained[offset] = _ensure_bound_constrait(unconstrained[offset])
            bc_lower, bc_upper = self.bc_bounds
            constrained[offset] = constrain_bound(
                unconstrained[offset], low=bc_lower, high=bc_upper
            )
            offset += 1

        # Level Alpha
        ## we use constrains from Table 2 for stability (Hyndman et al. 2007)
        ## using usual region for now
        alpha = constrain_bound(unconstrained[offset], high=2)
        # alpha = unconstrained[offset] ** 2
        constrained[offset] = alpha
        offset += 1

        if self.trend:
            beta, phi = unconstrained[offset : offset + 2]
            if self.damping:
                # 0 < beta < (1 + phi)(2 - alpha)
                constrained[offset] = constrain_bound(
                    beta,
                    low=0,
                    high=(1 + phi) * (2 - alpha),
                )
            # beta = unconstrained[offset]
            else:
                # beta < (1 + phi)(2 - alpha)
                constrained[offset] = constrain_bound(beta, high=4 - 2 * alpha)

            offset += 1

            if self.damping:
                # 0 < phi <= 1
                constrained[offset] = constrain_bound(phi)
                offset += 1

        if self.seasonal:
            num_k = 2 * len(self.periods)
            _s = slice(offset, offset + num_k)
            constrained[_s] = constrain_bound(unconstrained[_s], high=2 - alpha)
            offset += num_k

        if self.autoregressive:
            p, q = self.order
            if p > 0:
                if self.enforce_stationarity:
                    constrained[offset : offset + p] = constrain_stationary_univariate(
                        unconstrained[offset : offset + p]
                    )
                else:
                    constrained[offset : offset + p] = unconstrained[
                        offset : offset + p
                    ]
                offset += p

            if q > 0:
                if self.enforce_stationarity:
                    constrained[offset : offset + q] = -constrain_stationary_univariate(
                        unconstrained[offset : offset + q]
                    )
                else:
                    constrained[offset : offset + q] = unconstrained[
                        offset : offset + q
                    ]
                offset += q

        constrained[offset:] = unconstrained[offset:]
        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Used primarily to reverse enforcement of stationarity of the
        autoregressive lag polynomial and invertibility of the moving average
        lag polynomial.

        Parameters
        ----------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Returns
        -------
        constrained : array_like
            Unconstrained parameters used by the optimizer.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros_like(constrained)

        offset = 0

        # Positive parameters: z, state_cov
        if not self.concentrate_scale:
            unconstrained[0] = constrained[0] ** 0.5
            offset += 1

        if self.irregular:
            unconstrained[offset] = constrained[offset] ** 0.5
            offset += 1

        if self.boxcox:
            bc_lower, bc_upper = self.bc_bounds
            unconstrained[offset] = unconstrain_bound(
                constrained[offset], lower=bc_lower, upper=bc_upper
            )
            offset += 1

        # alpha
        alpha = constrained[offset]
        unconstrained[offset] = unconstrain_bound(alpha, upper=2)
        offset += 1

        if self.trend:
            if self.damping:
                beta, phi = constrained[offset : offset + 2]
                unconstrained[offset] = unconstrain_bound(
                    beta, upper=(1 + phi) * (2 - alpha)
                )
                unconstrained[offset + 1] = unconstrain_bound(phi)
                offset += 2
            else:
                unconstrained[offset] = unconstrain_bound(
                    constrained[offset], upper=4 - 2 * alpha
                )

        if self.seasonal:
            num_k = 2 * len(self.periods)
            _s = slice(offset, offset + num_k)
            unconstrained[_s] = unconstrain_bound(constrained[_s], upper=2 - alpha)
            offset += num_k

        if self.autoregressive:
            p, q = self.order
            if p > 0:
                if self.enforce_stationarity:
                    unconstrained[offset : offset + p] = (
                        unconstrain_stationary_univariate(
                            constrained[offset : offset + p]
                        )
                    )
                else:
                    unconstrained[offset : offset + p] = constrained[
                        offset : offset + p
                    ]
                offset += p

            if q > 0:
                if self.enforce_stationarity:
                    unconstrained[offset : offset + q] = (
                        unconstrain_stationary_univariate(
                            constrained[offset : offset + q]
                        )
                    )
                else:
                    unconstrained[offset : offset + q] = constrained[
                        offset : offset + q
                    ]
                offset += q
        unconstrained[offset:] = constrained[offset:]
        return unconstrained

    def update(
        self, params, transformed=True, includes_fixed=False, complex_step=False
    ):
        params = super().update(
            params, transformed=transformed, includes_fixed=includes_fixed
        )
        # params = np.asarray(params, dtype=self.ssm.dtype)

        offset = matrix_offset = 0
        s_dtype = self["selection"].dtype

        # cov
        if not self.concentrate_scale:
            cov = params[0]
            # self["obs_cov", 0, 0] = cov
            # np.fill_diagonal(self["state_cov"], cov)
            self["state_cov", 0, 0] = cov
            offset += 1

        if self.irregular:
            self["obs_cov", 0, 0] = params[offset]
            offset += 1

        if self.boxcox:
            lmbda = params[offset]
            offset += 1
            endog_touse, lmbda = self.transform_boxcox(
                self.endog, lmbda, bounds=self.bc_bounds
            )
            self.ssm.bind(endog_touse)
            self.ssm._representations = {}
            self.ssm._statespaces = {}
        else:
            lmbda = None

        # level alpha
        alpha = params[offset]
        offset += 1
        self["selection", matrix_offset, 0] = alpha
        matrix_offset += 1

        if self.trend:
            beta = params[offset]
            offset += 1

            self["selection", matrix_offset, 0] = beta
            matrix_offset += 1

            # assign intercept b for trend
            # self["state_intercept", 1] = b

            # damped
            if self.damping:
                damped = params[offset]
                offset += 1

                self["transition", :2, 1] = damped
                self["design", 0, 1] = damped
                # self["state_intercept", 1] = (1 - damped) * b

        # Seasonal gamma
        if self.seasonal:
            n = self.k_period
            tau = int(2 * np.sum(self.k))
            gamma = params[offset : offset + 2 * n]
            offset += 2 * n

            j = 0
            gamma_selection = np.zeros(tau)
            # gamma_design = np.zeros(tau)
            for i, k in enumerate(self.k, 1):
                gamma_selection[j : j + 2 * k] = np.r_[
                    np.full(k, gamma[2 * i - 2], dtype=s_dtype),
                    np.full(k, gamma[2 * i - 1], dtype=s_dtype),
                ]
                # gamma_design[j: j + 2 * k] = np.r_[
                #     np.ones(k),
                #     np.zeros(k)
                # ]
                j += 2 * k

            self["selection", matrix_offset : matrix_offset + tau, 0] = gamma_selection
            # self['design', 0, matrix_offset: matrix_offset + tau] = gamma_design

            matrix_offset += tau

        if self.autoregressive:
            p, q = self.order
            col_indice = matrix_offset
            # AR
            if p:
                ar = params[offset : offset + p]

                col_indice = matrix_offset
                offset += p
                self["design", 0, matrix_offset : matrix_offset + p] = ar
                self["transition", 0, matrix_offset : matrix_offset + p] = alpha * ar
                if self.trend:
                    self["transition", 1, matrix_offset : matrix_offset + p] = beta * ar
                if self.seasonal:
                    self[
                        "transition",
                        1 + self.trend : col_indice,
                        matrix_offset : matrix_offset + p,
                    ] = (
                        gamma_selection[:, None] * ar
                    )
                self["transition", matrix_offset, matrix_offset : matrix_offset + p] = (
                    ar
                )
                matrix_offset += p

            # MA
            if q:
                ma = params[offset : offset + q]

                self["design", 0, -q:] = ma
                self["transition", 0, -q:] = alpha * ma
                if self.trend:
                    self["transition", 1, -q:] = beta * ma
                if self.seasonal:
                    self["transition", 1 + self.trend : col_indice, -q:] = (
                        gamma_selection[:, None] * ma
                    )
                self["transition", col_indice + p, -q:] = ma

                offset += q

        if self.regression and self.mle_regression and True:
            self.ssm["obs_intercept"] = np.dot(
                self.data.exog, params[offset : offset + self.k_exog]
            )[None, :]
            offset += self.k_exog

        # self.initialize_state(boxcox_lambda=float(lmbda) if lmbda else None)


class TBATSResults(MLEResults):
    """
    Class to hold results from fitting an unobserved components model.

    Parameters
    ----------
    model : UnobservedComponents instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the unobserved components
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super(TBATSResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs
        )

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        self.specification = Bunch(
            **{
                # Model options
                "level": True,
                "trend": self.model.trend,
                "irregular": self.model.irregular,
                "seasonal": self.model.seasonal,
                "arma_order": self.model.order,
                "box_cox": self.model.boxcox,
                "damped_trend": self.model.damping,
                "autoregressive": self.model.autoregressive,
                "concentrate_scale": self.model.concentrate_scale,
            }
        )

    @cache_readonly
    def fittedvalues(self):
        fittedvalues = super(TBATSResults, self).fittedvalues
        if self.model.boxcox:
            boxcox_lambda_offset = (
                not self.specification.concentrate_scale
            ) + self.specification.irregular
            fittedvalues = self.model.untransform_boxcox(
                fittedvalues, float(self.params[boxcox_lambda_offset])
            )
        return fittedvalues

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglike(self.params)

    @property
    def level(self):
        offset = 0
        out = Bunch(
            filtered=self.filtered_state[offset],
            filtered_cov=self.filtered_state_cov[offset, offset],
            smoothed=None,
            smoothed_cov=None,
            offset=0,
        )
        if self.smoothed_state is not None:
            out.smoothed = self.smoothed_state[offset]
        if self.smoothed_state_cov is not None:
            out.smoothed_cov = self.smoothed_state_cov[offset, offset]

        return out

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        resid = self.data.endog - self.fittedvalues
        if resid.shape[0] == 1:
            resid = resid[0, :]
        else:
            resid = resid.T
        return resid

    @cache_readonly
    def aic(self):
        return aic(self.llf, self.nobs, self.params.nonzero()[0].shape[0])

    @cache_readonly
    def mape(self):
        return np.nanmean(
            np.abs(self.resid / self.fittedvalues)[self.loglikelihood_burn :]
        )

    @cache_readonly
    def mae(self):
        return np.nanmean(np.abs(self.resid)[self.loglikelihood_burn :])

    @property
    def trend(self):
        """
        Estimates of of unobserved trend component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, trend is always the second component of the state vector
        # (because level is always present if trend is present)
        out = None
        spec = self.specification
        if spec.trend:
            offset = int(spec.level)
            out = Bunch(
                filtered=self.filtered_state[offset],
                filtered_cov=self.filtered_state_cov[offset, offset],
                smoothed=None,
                smoothed_cov=None,
                offset=offset,
            )
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def seasonal(self):
        out = None
        spec = self.specification
        if spec.seasonal:
            k = self.model.k
            offset = int(spec.level + spec.trend)
            filtered_state = []
            filtered_state_cov = []
            smoothed_state = []
            smoothed_state_cov = []
            for i in k:
                filtered_state.append(self.filtered_state[offset : offset + i].sum(0))
                filtered_state_cov.append(
                    self.filtered_state_cov[offset : offset + i, offset : offset + i]
                    .sum(0)
                    .sum(0)
                )
                if self.smoothed_state is not None:
                    smoothed_state.append(
                        self.smoothed_state[offset : offset + i].sum(0)
                    )
                if self.smoothed_state_cov is not None:
                    smoothed_state_cov.append(
                        self.smoothed_state_cov[
                            offset : offset + i, offset : offset + i
                        ]
                        .sum(0)
                        .sum(0)
                    )
                offset += 2 * i
            out = Bunch(
                filtered=np.array(filtered_state),
                filtered_cov=np.array(filtered_state_cov),
                smoothed=np.array(smoothed_state),
                smoothed_cov=smoothed_state_cov,
                offset=offset,
            )
        return out

    @property
    def arma_error(self):
        out = None
        spec = self.specification
        if spec.autoregressive:
            p, q = self.model.order
            offset = int(spec.level + spec.trend)
            if spec.seasonal:
                offset += 2 * np.sum(self.model.k)

            # if q > 0:
            #     filtered = self.filtered_state[offset] + self.filtered_state[offset + p]
            #     filtered_cov = self.filtered_state_cov[offset, offset] + self.filtered_state_cov[offset + p, offset + p]

            # else:
            filtered = self.filtered_state[offset]
            filtered_cov = self.filtered_state_cov[offset, offset]
            out = Bunch(
                filtered=filtered,
                filtered_cov=filtered_cov,
                smoothed=None,
                smoothed_cov=None,
                offset=offset,
            )

            if self.smoothed_state is not None:
                # if q > 0:
                #     out.smoothed = self.smoothed_state[offset] + self.smoothed_state[offset + p]
                # else:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                # if q > 0:
                #     out.smoothed_cov = self.smoothed_state_cov[offset, offset] + self.smoothed_state_cov[
                #         offset + p, offset + p]
                # else:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    def plot_components(
        self,
        which=None,
        alpha=0.05,
        start=None,
        end=None,
        observed=True,
        level=True,
        trend=True,
        seasonal=True,
        autoregressive=True,
        resid=True,
        figsize=None,
        legend_loc="upper left",
    ):

        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        from pandas.plotting import register_matplotlib_converters

        plt = _import_mpl()
        register_matplotlib_converters()
        # fig = create_mpl_fig(fig, figsize)

        # Determine which results we have
        if which is None:
            which = "filtered" if self.smoothed_state is None else "smoothed"

        spec = self.specification
        components = OrderedDict(
            [
                ("level", level and spec.level),
                ("trend", trend and spec.trend),
                ("arma_error", autoregressive and spec.autoregressive),
            ]
        )

        seasonal = seasonal and spec.seasonal

        llb = self.filter_results.loglikelihood_burn

        # Number of plots
        k_plots = observed + np.sum(list(components.values())) + resid
        if seasonal:
            k_plots += len(self.model.periods)

        # Get dates, if applicable
        if hasattr(self.data, "dates") and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
            start = self.model._get_index_loc(start)[0] if start else llb
            if isinstance(start, slice):
                start = start.start
            end = self.model._get_index_loc(end)[0] if end else None
            if isinstance(end, slice):
                end = end.stop
        else:
            dates = np.arange(len(self.resid))

        # Get the critical value for confidence intervals
        critical_value = norm.ppf(1 - alpha / 2.0)

        plot_idx = 0
        fig, axes = plt.subplots(k_plots, 1, figsize=figsize, sharex=True)
        if observed:
            ax = axes[plot_idx]
            plot_idx += 1

            # Plot the observed dataset
            ax.plot(
                dates[start:end],
                self.data.endog[start:end],
                color="k",
                label="Observed",
            )

            # Get the predicted values and confidence intervals
            predict = self.filter_results.forecasts[0]
            std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0, 0])
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors

            if spec.box_cox:
                lmbda = self.params[(not spec.concentrate_scale) + spec.irregular]
                predict = self.model.untransform_boxcox(predict, lmbda)
                ci_lower = self.model.untransform_boxcox(ci_lower, lmbda)
                ci_upper = self.model.untransform_boxcox(ci_upper, lmbda)

            # Plot
            ax.plot(
                dates[start:end], predict[start:end], label="One-step-ahead predictions"
            )
            ci_poly = ax.fill_between(
                dates[start:end], ci_lower[start:end], ci_upper[start:end], alpha=0.2
            )
            ci_label = "$%.3g \\%%$ confidence interval" % ((1 - alpha) * 100)

            # Proxy artist for fill_between legend entry
            # See e.g. http://matplotlib.org/1.3.1/users/legend_guide.html
            p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(p)
            labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)

            ax.set_title("Predicted vs observed")

        for component, is_plotted in components.items():
            if not is_plotted:
                continue

            ax = axes[plot_idx]
            plot_idx += 1

            component_bunch = getattr(self, component)

            # Check for a valid estimation type
            if which not in component_bunch:
                raise ValueError("Invalid type of state estimate.")

            which_cov = "%s_cov" % which

            # Get the predicted values
            value = component_bunch[which]

            # Plot
            state_label = "%s (%s)" % (component.title(), which)
            ax.plot(dates[start:end], value[start:end], label=state_label)

            # Get confidence intervals
            if which_cov in component_bunch:
                std_errors = np.sqrt(component_bunch["%s_cov" % which])
                ci_lower = value - critical_value * std_errors
                ci_upper = value + critical_value * std_errors
                ci_poly = ax.fill_between(
                    dates[start:end],
                    ci_lower[start:end],
                    ci_upper[start:end],
                    alpha=0.2,
                )
                ci_label = "$%.3g \\%%$ confidence interval" % ((1 - alpha) * 100)

            # Legend
            ax.legend(loc=legend_loc)

            ax.set_title("%s component" % component.title())

        if seasonal:
            component_bunch = self.seasonal

            for i, (m, k, name) in enumerate(
                zip(self.model.periods, self.model.k, self.model.period_names)
            ):
                ax = axes[plot_idx]
                plot_idx += 1

                which_cov = "%s_cov" % which

                # Get the predicted values
                value = component_bunch[which][i]

                # Plot
                state_label = "%s (%s, %s)" % (name, m, k)
                ax.plot(dates[start:end], value[start:end], label=state_label)

                # Get confidence intervals
                if which_cov in component_bunch:
                    std_errors = np.sqrt(component_bunch["%s_cov" % which][i])
                    ci_lower = value - critical_value * std_errors
                    ci_upper = value + critical_value * std_errors
                    ci_poly = ax.fill_between(
                        dates[start:end],
                        ci_lower[start:end],
                        ci_upper[start:end],
                        alpha=0.2,
                    )
                    ci_label = "$%.3g \\%%$ confidence interval" % ((1 - alpha) * 100)

                # Legend
                ax.legend(loc=legend_loc)

                ax.set_title(name + " component")

        if resid:
            ax = axes[plot_idx]
            plot_idx += 1

            value = self.resid
            ax.plot(dates[start:end], value[start:end], label="Remainder")
            ax.legend(loc=legend_loc)

            ax.set_title("Residuals component")
            # ax.set_ylabel('Remainder')

        # Add a note if first observations excluded
        if llb > 0:
            text = (
                "Note: The first %d observations are not shown, due to"
                " approximate diffuse initialization."
            )
            fig.text(0.02, 0.02, text % llb, fontsize="large")

        fig.tight_layout()
        return fig

    def forecast(self, steps=1, **kwargs):
        forecast_values = super().forecast(steps, **kwargs)
        if self.model.boxcox:
            boxcox_lambda_offset = (
                not self.specification.concentrate_scale
            ) + self.specification.irregular
            forecast_values = self.model.untransform_boxcox(
                forecast_values, self.params[boxcox_lambda_offset]
            )

        return forecast_values

    def get_prediction(self, start=None, end=None, dynamic=False, **kwargs):
        return super().get_prediction(start, end, dynamic, **kwargs)

    def summary(
        self, alpha=0.05, start=None, separate_params=True, title=None, **kwargs
    ):
        from statsmodels.iolib.summary import summary_params

        summary = super(TBATSResults, self).summary(
            alpha=alpha,
            start=start,
            model_name="TBATS",
            display_params=not separate_params,
            title=title,
        )
        if separate_params:

            def make_table(self, mask, title, strip_end=True):
                res = (
                    self,
                    self.params[mask],
                    self.bse[mask],
                    self.zvalues[mask],
                    self.pvalues[mask],
                    self.conf_int(alpha)[mask],
                )
                param_names = [
                    ".".join(name.split(".")[:-1]) if strip_end else name
                    for name in np.array(self.data.param_names)[mask].tolist()
                ]

                return summary_params(
                    res,
                    yname=None,
                    xname=param_names,
                    alpha=alpha,
                    use_t=False,
                    title=title,
                )

            offset = 0
            # Main Params
            mask = list()
            if not self.specification.concentrate_scale:
                mask.append(offset)
                offset += 1
            if self.specification.irregular:
                mask.append(offset)
                offset += 1
            if self.specification.box_cox:
                mask.append(offset)
                offset += 1
            table = make_table(self, mask, title, strip_end=False)
            summary.tables.append(table)

            # Level  & Trend
            mask = [offset]
            offset += 1
            title = "Level"

            # Trend
            if self.specification.trend:
                mask.append(offset)
                offset += 1
                title += " & Trend"

                if self.specification.damped_trend:
                    mask.append(offset)
                    offset += 1

            table = make_table(self, mask, title, strip_end=False)
            summary.tables.append(table)

            # Seasonal
            if self.specification.seasonal:
                for period, k, name in zip(
                    self.model.periods, self.model.k, self.model.period_names
                ):
                    mask = [offset, offset + 1]
                    offset += 2
                    title = name + "({:.2f}, {:.0f})".format(period, k)
                    table = make_table(self, mask, title)
                    summary.tables.append(table)

            if self.specification.autoregressive:
                p, q = self.model.order
                table = make_table(
                    self,
                    np.arange(offset, offset + p + q),
                    "ARMA Error" + str(self.model.order),
                )
                offset += p + q
                summary.tables.append(table)

        return summary

    summary.__doc__ = MLEResults.summary.__doc__


class TBATSResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._methods, _methods)


wrap.populate_wrapper(TBATSResultsWrapper, TBATSResults)


def _safe_tbats_fit(endog, model_kw, fit_kw, model_class=TBATSModel, **kwargs):
    try:
        if True:
            res = model_class(endog, **model_kw).fit(**fit_kw)
            return res
    except LinAlgError as e:
        # SVD convergence failure on badly misspecified models
        print(f"LinAlgError: {str(e)}", model_kw, fit_kw)
        return
    except Exception as e:  # no idea what happened
        print("error fits models:", model_kw, fit_kw)
        raise e
        # print(e.with_traceback())
        # return


def tbats_k_order_select_ic(
    y,
    periods,
    use_trend=None,
    use_box_cox=None,
    damped_trend=None,
    max_ar=4,
    max_ma=2,
    ic="aic",
    model_class=TBATSModel,
    return_params=False,
    **kwargs,
):
    from functools import partial
    from statsmodels.tsa.stattools import arma_order_select_ic

    model_kw = kwargs.get("model_kw", {})
    fit_kw = kwargs.get("fit_kw", {})

    # default model_kw
    model_kw.setdefault("use_trend", False if use_trend is None else use_trend)
    model_kw.setdefault("damped_trend", False if damped_trend is None else damped_trend)
    model_kw.setdefault("box_cox", False if use_box_cox is None else use_box_cox)
    model_kw.setdefault("order", (0, 0))

    # default fit_kw
    # we use low_memory mode to save memory
    fit_kw.setdefault("low_memory", True)

    grid_search = kwargs.pop("grid_search", False)

    periods = np.asarray(periods)
    periods.sort()
    n_periods = periods.size

    ks = np.ones(n_periods, dtype=int)
    data = np.asarray(y)

    model_kw["periods"] = periods
    model_kw["k"] = ks

    calc_aic = partial(_safe_tbats_fit, y, fit_kw=fit_kw, model_class=model_class)

    _ic = lambda obj: getattr(obj, ic)

    if np.any(data <= 0) or use_box_cox is not True:
        model_kw["box_cox"] = False
        y2 = data.copy()
    elif use_box_cox is True:
        bounds = model_kw.get("bc_bounds", (0, 2))
        y2 = BoxCox().transform_boxcox(data, bounds=bounds)
        y2 = np.log(data)
        model_kw["box_cox"] = True

    decomp_mod = seasonal_decompose(
        y2, period=int(np.max(periods)), extrapolate_trend="freq"
    )
    ks = seasonal_fourier_k_select_ic(
        decomp_mod.seasonal, periods, grid_search=grid_search, ic=ic
    )
    model_kw["k"] = ks
    resid = decomp_mod.resid

    # baseline
    best_model = calc_aic(model_kw=model_kw)
    min_ic = _ic(best_model)

    # Use Trend
    if use_trend is None:
        model_kw["use_trend"] = True
        mod = calc_aic(model_kw=model_kw)
        if mod and _ic(mod) < min_ic:
            best_model = mod
            min_ic = _ic(mod)
            if damped_trend is None:
                model_kw["damped_trend"] = True
                mod = calc_aic(model_kw=model_kw)
                if mod and _ic(mod) < min_ic:
                    model_kw["damped_trend"] = True
                    best_model = mod
                    min_ic = _ic(mod)
                else:
                    model_kw["damped_trend"] = False
        else:
            model_kw["use_trend"] = False

    elif use_trend is True:
        model_kw["use_trend"] = True
        mod = calc_aic(model_kw=model_kw)
        if mod:
            best_model = mod
            # overwrite the default ic
            min_ic = _ic(mod)
            if damped_trend is None:
                model_kw["damped_trend"] = True
                mod = calc_aic(model_kw=model_kw)
                if mod and _ic(mod) < min_ic:
                    model_kw["damped_trend"] = True
                    best_model = mod
                    min_ic = _ic(mod)
                else:
                    model_kw["damped_trend"] = False
        else:
            model_kw["use_trend"] = False

    order = getattr(
        arma_order_select_ic(best_model.resid, max_ar=max_ar, max_ma=max_ma, ic=ic),
        f"{ic}_min_order",
    )
    print(f"resid best arma order: {order}")
    model_kw["order"] = order
    mod = calc_aic(model_kw=model_kw)
    if mod is not None:
        print(f"arima {ic}: {_ic(mod)} for {order}, current best {ic}: {min_ic}")
    if mod and _ic(mod) < min_ic:
        best_model = mod
        min_ic = _ic(mod)
    else:
        model_kw["order"] = (0, 0)
    # model_kw.update(k=ks)

    if return_params:
        return model_kw
    elif not fit_kw.get("low_memory", True):
        return model_kw, best_model
    else:
        return model_kw, best_model.model.filter(best_model.params)
