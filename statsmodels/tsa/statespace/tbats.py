"""
Trigonometric, Box-Cox, ARMA Error, Trend and Seasonal (TBATS) Model

Author: Leoyzen Liu
License: Simplified-BSD
"""

from __future__ import absolute_import, division, print_function

from statsmodels.compat.pandas import Appender

from collections import OrderedDict
from warnings import warn

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import linalg

from statsmodels.base.data import PandasData
from statsmodels.base.model import LikelihoodModel
from statsmodels.base.transform import BoxCox
import statsmodels.base.wrapper as wrap
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic
from statsmodels.tools.sm_exceptions import EstimationWarning, ValueWarning
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation.validation import (
    array_like,
    bool_like,
    float_like,
    string_like,
)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import (
    companion_matrix,
    constrain_stationary_univariate,
    unconstrain_stationary_univariate,
)

from ._tbats_tools import _tbats_recursive_compute, _tbats_w_caculation
from .innovations import InnnovationsMLEModel
from .kalman_filter import (
    INVERT_UNIVARIATE,
    MEMORY_CONSERVE,
    MEMORY_NO_FORECAST,
    MEMORY_NO_FORECAST_MEAN,
    SOLVE_LU,
)
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper, _handle_args
from .tools import fourier


def constrain_bound(x, lower=1e-6, upper=1 - 1e-6):
    from scipy.special import expit

    if upper <= lower:
        return lower
    return expit(x) * (upper - lower) + lower


def unconstrain_bound(x, lower=1e-6, upper=1 - 1e-6):
    from scipy.special import logit

    if upper <= lower:
        return logit(lower)

    x = (x - lower) / (upper - lower)
    return logit(x)


def constrain_scale(x, lower=1e-6, upper=1 - 1e-6):
    from scipy.special import expit

    _x = expit(x)
    _xs = _x * (upper - lower) + lower
    return _x / _xs


def unconstrain_scale(x, lower=1e-6, upper=1 - 1e-6):
    from scipy.special import logit

    s = np.sum(x)
    s = (np.sum(x) - lower) / (upper - lower)
    return logit(x / s)


class PeriodWrapper(object):
    def __init__(self, name, attribute, dtype=np.float32, multiple=False):
        self.name = name
        self.attribute = attribute
        self._attribute = "_" + attribute
        self._multiple = multiple
        self.dtype = dtype

    def __set__(self, obs, val):
        if val is None:
            k_periods = 0
        elif np.isscalar(val):
            if self._multiple:
                val = np.asarray([val], dtype=self.dtype)
            else:
                val = self.dtype(val)
            k_periods = 1
        else:
            if self._multiple:
                val = np.asarray(val, dtype=self.dtype)
            else:
                msg = "%s must be a scalar of %s"
                raise ValueError(msg % (self.name, self.dtype))
            k_periods = len(val)
        setattr(obs, "k_periods", k_periods)
        setattr(obs, self._attribute, val)
        setattr(obs, "period_names", [f"Seasonal{i}" for i in range(k_periods)])

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
    return tau, design.astype(dtype), linalg.block_diag(*transition).astype(dtype)


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


def _k_f_test(x, periods, ks, exog=None, weights=None, robust=False):
    for i, (period, k) in enumerate(zip(periods, ks)):
        if exog is None:
            exog = fourier(x, period, k, name="Seasonal" + str(i))
        else:
            exog = exog.join(
                fourier(x, period, k, name="Seasonal" + str(i)), how="outer"
            )
    # if not wls:
    res = OLS(x, add_constant(exog, prepend=False)).fit(cov_kwd={"use_t": True})
    # else:
    # if weights is None and robust:
    #     weights = RLM(x, add_constant(exog)).weights
    # res = GLM(x, add_constant(exog), weights=weights, missing="drop").fit()
    return res


def seasonal_fourier_k_select_ic(
    y, periods, ic="aic", grid_search=False, restrict_periods=True
):
    periods = np.asarray(periods)
    ks = np.ones_like(periods, dtype=int)
    _ic = lambda obj: getattr(obj, ic)
    result = {}
    max_k = np.ones_like(ks)

    for i, period in enumerate(periods):

        max_k[i] = period // 2
        if restrict_periods and i != 0:
            current_k = 2
            while current_k <= max_k[i]:
                if period % current_k != 0:
                    current_k += 1
                    continue

                latter = period / current_k

                if np.any(periods[:i] % latter == 0):
                    max_k[i] = current_k - 1
                    break

                current_k += 1
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
                test = _k_f_test(resid, periods, _ks)
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


def _initialization_simple(
    endog, trend=False, seasonal=False, seasonal_periods=None, seasonal_harmonics=None
):
    # See Section 7.6 of Hyndman and Athanasopoulos
    nobs = len(endog)
    initial_trend = None
    initial_seasonal = None

    # Non-seasonal
    if seasonal is None or not seasonal:
        initial_level = endog[0]
    # Seasonal
    else:
        m = int(max(seasonal_periods))
        if nobs < 2 * m:
            raise ValueError(
                "Cannot compute initial seasonals using"
                " heuristic method with less than two full"
                " seasonal cycles in the data."
            )

        initial_level = np.mean(endog[:m])

        if trend is not None:
            initial_trend = (pd.Series(endog).diff(m)[m : 2 * m] / m).mean()

        seasonal_resid = endog[:m] - initial_level
        initial_seasonal = _k_f_test(
            seasonal_resid, seasonal_periods, seasonal_harmonics
        ).params[:-1]

    return initial_level, initial_trend, initial_seasonal


class TBATSModel(InnnovationsMLEModel, BoxCox):
    r"""
    Trigonometric, Box-Cox, ARMA Error, Trend and Seasonal(TBATS)

    This model incorporates BoxCox transformations, Fourier representations
    with time varying coefﬁcients, and ARMA error correction.

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
        y^{(\omega)}_t = l_{t-1} + \phi b_{t-1} + \sum^{T}_{i=1}s^{(i)}_{t-m_i} + d_t

    **Box-Cox**

    To avoid the problems with non-linear models, Box-Cox transformation is introduced
    to allow some types of non-linearity.

    .. math::
        y^{(\omega)}_t = \begin{cases}
        \frac{y^\omega_t - 1}{\omega};\quad \omega \ne 0 \\
        log{y_t};\quad\omega = 0
        \end{cases}

    The notation :math:`y^{(\omega)}_t ` is used to represent Box-Cox transformed
    observations with the parameter :math:`\omega`,
    where :math:`y_t` is the observation at time :math:`t`.

    **Level & Trend**

    The trend component is a dynamic extension of a regression model that
    includes an intercept and linear time-trend. It can be written:

    .. math::
        l = l_{t-1} + \phi b_{t-1} + \alpha d_t \\
        b_t = (1-\phi)b + \phi b_{t-1} + \beta d_t

    where the level is a generalization of the intercept term that can
    dynamically vary across time, and the trend is a generalization of the
    time-trend such that the slope can dynamically vary across time.

    :math:`l_t` is the is the local level in period :math:`t`,
    :math:`b` is the long-run trend,
    :math:`b_t` is the short-run trend in period :math:`t`.

    :math:`\alpha, \beta` are smoothing parameters for level and trend.
    :math:`\phi` is daping parameter for damped trend.

    The damping factor is included in the level and measurement equations
    as well as the trend equation for consistency with [2]_,
    but identical predictions are obtained if it is excluded
    from the level and measurement equations.

    **Trigonometric Season**

    .. math::
        s^{(i)}_t = s^{(i)}_{t-m_i} + \gamma_id_t \\

    :math:`\gamma_i` for :math:`i=1,...,T` are seasonal smoothing parameters.

    **ARMA Error**

    .. math::
        d_t = \sum^p_{i=1}\psi_id_{t-i} + \sum^q_{i-1}\theta_i\epsilon_{t-i}+\epsilon_t

    :math:`d_t` denotes an ARMA(p, q) process and :math:`\epsilon_t` is a Gaussian
    white noise process with zero mean and constant variance :math:`\sigma^2`.


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
        harmonics=None,
        order=(0, 0),
        period_names=None,
        trend=False,
        damped_trend=False,
        box_cox=False,
        exog=None,
        mle_regression=True,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=True,
        initialization_method="concentrated",
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        bounds=None,
        biasadj=True,
        check_admissible=True,
        **kwargs,
    ):

        # Model options
        self.seasonal = periods is not None
        self.trend = trend
        self.boxcox = bool_like(box_cox, "boxcox")
        self.damped_trend = bool_like(damped_trend, "damped_trend")
        self.mle_regression = bool_like(mle_regression, "mle_regression")
        self.autoregressive = order != (0, 0)
        self.ar_order = order[0]
        self.ma_order = order[1]
        self.periods = periods
        self.harmonics = harmonics
        self.order = order
        self.concentrate_scale = bool_like(concentrate_scale, "concentrate_scale")
        self.initialization_method = string_like(
            initialization_method, "initialization_method"
        ).lower()
        self.enforce_stationarity = bool_like(
            enforce_stationarity, "enforce_stationarity"
        )
        self.enforce_invertibility = bool_like(
            enforce_invertibility, "enforce_invertibility"
        )
        self.biasadj = bool_like(biasadj, "biasadj")
        self.check_admissible = bool_like(check_admissible, "check_admissible")
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = (
                [(1e-4, 1 - 1e-4)] * 2
                + [(0.8, 0.9999)]
                + [(1e-6, 1 - 1e-6)]
                + [(-1, 2)]
            )

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
            self.harmonics = harmonics
            self.k_harmonics = int(2 * np.sum(self.harmonics))
            k_states += self.k_harmonics
            self.max_period = max(self.periods)
            if period_names is not None:
                self.period_names = period_names
        else:
            self.seasonal = False
            self.k_harmonics = 0
            self.max_period = 0
            self.periods = tuple()
            self.harmonics = tuple()

        if self.trend:
            k_states += 1

        self.autoregressive = np.sum(order) != 0
        self.order = order

        p, q = self.order

        if self.autoregressive:
            k_states += p + q

        # cached $$\epsilon$
        self.k_hidden_states = 1

        k_states = (
            # self.k_hidden_states
            +1  # level
            + self.trend
            + self.k_harmonics
            + np.sum(self.order)
            + (not self.mle_regression) * self.k_exog
        )
        k_posdef = 1

        if self.boxcox and self.initialization_method not in ("concentrated", "estimated"):
            raise ValueError(f"{initialization_method} is not support when boxcox enabled.")

        if self.initialization_method not in [
            "concentrated",
            "estimated",
            "simple",
            "heuristic",
            "known",
        ]:
            raise ValueError(
                'Invalid initialization method "%s".' % initialization_method
            )

        if self.initialization_method == "known":
            if initial_level is None:
                raise ValueError(
                    "`initial_level` argument must be provided"
                    " when initialization method is set to"
                    ' "known".'
                )
            if initial_trend is None and self.trend:
                raise ValueError(
                    "`initial_trend` argument must be provided"
                    " for models with a trend component when"
                    ' initialization method is set to "known".'
                )
            if initial_seasonal is None and self.seasonal:
                raise ValueError(
                    "`initial_seasonal` argument must be provided"
                    " for models with a seasonal component when"
                    ' initialization method is set to "known".'
                )

        # init = Initialization(k_states, "known", constant=np.zeros(k_states))
        # Initialize the model base
        super().__init__(
            endog=endog,
            exog=exog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="known",
            constant=np.zeros(k_states + 1),
            stationary_cov=np.zeros((k_states + 1, k_states + 1)),
            **kwargs,
        )
        self._initial_state = None
        self._need_recompute_state = True

        if self.concentrate_scale:
            self.ssm.filter_concentrated = True

        if box_cox:
            if np.any(self.data.endog <= 0):
                warn("To use boxcox transformation the endog must be positive")
                self.boxcox = False
            else:
                # use to calculate loglikeobs when using boxcox
                self.data.log_endog = np.log(self.data.endog)
                self._boxcox_lambda = 1
        else:
            self.boxcox = False

        self.setup()

    @property
    def state_names(self):
        # if self.seasonal:
        #     state_names += [f"sasonal{i}" for i in range(1, len(self.periods) + 1)]
        state_names = ["level"]
        if self.trend:
            state_names += ["trend"]
        if self.seasonal:
            state_names += [
                f"seasonal{i}.S{j}"
                for i, k in enumerate(self.harmonics, start=1)
                for j in range(1, k + 1)
            ]
            state_names += [
                f"seasonal{i}.C{j}"
                for i, k in enumerate(self.harmonics, start=1)
                for j in range(1, k + 1)
            ]
        if self.ar_order:
            state_names += [f"ar.L{i}" for i in range(1, self.ar_order + 1)]
        if self.ma_order:
            state_names += [f"ma.L{i}" for i in range(1, self.ma_order + 1)]

        # hidden cached \epsilon
        state_names += ["error"]
        return state_names

    @property
    def _res_classes(self):
        return {"fit": (TBATSResults, TBATSResultsWrapper)}

    @Appender(LikelihoodModel.initialize.__doc__)
    def initialize(self):
        super().initialize()
        # cached components
        # offset = self.k_hidden_states
        offset = 0

        # level
        self._level_state_offset = offset
        offset += 1

        # trend
        if self.trend:
            self._trend_state_offset = offset
            ## dampping trend
            offset += 1

        if self.seasonal:
            tau = self.k_harmonics
            self._seasonal_state_offset = offset
            offset += tau

        self._arma_state_offset = offset
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

        offset = 0

        if self.concentrate_scale:
            self["cov"] = 1
        else:
            self.parameters["state_var"] = 1

        if self.boxcox:
            self.parameters["boxcox"] = 1

        # Level Setup
        self.parameters["level_alpha"] = 1
        self["w", 0, offset] = 1.0
        self["F", offset, offset] = 1.0
        # self["selection", 0, 0] = 1
        offset += 1

        # Trend Setup
        if self.trend:
            self.parameters["trend_beta"] = 1
            # self.parameters["trend_intercept"] = 1

            self["w", 0, offset] = 1
            self["F", offset - 1 : offset + 1, offset] = 1
            offset += 1
            if self.damped_trend:
                self.parameters["damped_trend"] = 1

        if self.seasonal:
            tau, season_design, season_transition = _tbats_seasonal_matrix(
                self.periods, self.harmonics
            )
            self["w", 0, offset : offset + tau] = season_design
            self["F", offset : offset + tau, offset : offset + tau] = season_transition
            self.parameters["seasonal_gamma"] = 2 * self.k_periods
            offset += tau

        if self.autoregressive:
            p, q = self.order
            if p:
                start = self._ar_state_offset
                if p > 1:
                    np.fill_diagonal(
                        self["F", start + 1 : start + p, start : start + p], 1
                    )
                self["g", start, 0] = 1
                # self["state_cov", start, start] = 1
                # self["selection", start, start] = 1
                self.parameters["ar_coeff"] = p
                start += p
            if q:
                start = self._ma_state_offset
                if q > 1:
                    np.fill_diagonal(
                        self["F", start + 1 : start + q, start : start + q], 1
                    )
                self["g", start, 0] = 1
                # self["state_cov", start, start] = 1
                self.parameters["ma_coef"] = q

                # reuse cached residual
                self["F", start, 0] = 1

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

    def _initialize_constant_statespace(self, initial_state):
        # Note: this should be run after `update` has already put any new
        # parameters into the transition matrix, since it uses the transition
        # matrix explicitly.

        # Due to timing differences, the state space representation integrates
        # the trend into the level in the "predicted_state" (only the
        # "filtered_state" corresponds to the timing of the exponential
        # smoothing models)
        self._initial_state = initial_state
        constant = np.zeros_like(self.initialization.constant, dtype=self.ssm.dtype)
        constant[:initial_state.shape[0]] = initial_state.astype(self.ssm.dtype)

        self.initialization.constant = constant

    def _initialize_stationary_cov_statespace(self):
        R = self.ssm["selection"]
        Q = self.ssm["state_cov"]
        self.initialization.stationary_cov = R.dot(Q).dot(R.T)

    def _compute_concentrated_states(self, params, *args, **kwargs):
        # Apply the usual filter, but keep forecasts
        kwargs["conserve_memory"] = MEMORY_CONSERVE & ~MEMORY_NO_FORECAST_MEAN
        # Compute the initial state vector
        # super().loglike(params, *args, **kwargs)
        self.ssm._filter(conserve_memory=MEMORY_CONSERVE & ~MEMORY_NO_FORECAST_MEAN)
        # self.ssm.loglike(**kwargs)
        y_tilde = np.array(self.ssm._kalman_filter.forecast_error, copy=True)

        # print(self.ssm._kalman_filter.converged, self.ssm._kalman_filter.period_converged)
        dtype = self.ssm.dtype
        Z = self["w"].astype(dtype)

        # g = self["g"]
        # F = self["F"]
        # y = np.array(self.ssm.endog, copy=True, dtype=dtype)
        # y_tilde = np.zeros_like(y)
        # y_hat = np.zeros_like(y)
        # x0 = np.zeros((self.k_states - 1, 1), dtype=dtype)
        # # print(Z.dtype, F.dtype, g.dtype, y.dtype, y_tilde.dtype, y_hat.dtype, x0.dtype)
        # _tbats_recursive_compute(Z, F, g, y, y_tilde, y_hat, x0)

        y_tilde = y_tilde[0]

        # De Livera et al. (2011)
        # cutoff_offset = self.k_hidden_states
        # T = self["transition", cutoff_offset:, cutoff_offset:]
        # R = self._internal_selection
        # # g = self["selection", cutoff_offset:]
        # # R = g.dot(self["state_cov"]).dot(g.T)
        # Z = self._internal_design
        # T = self["F"]
        # R = self["g"]
        # Z = self["w"]

        # D = T - R.dot(Z)
        D = self.D.astype(dtype)
        w = np.zeros((self.nobs, Z.shape[-1]), dtype=dtype)
        # print(D.dtype, Z.dtype, w.dtype)
        _tbats_w_caculation(D, Z[0, :], w)
        # w[0, :] = Z[0, :]
        # for i in range(self.nobs - 1):
        #     w[i + 1] = w[i].dot(D)

        if self.autoregressive:
            # remove arma bits
            # offset = np.sum(self.order)
            w = w[:, : self._arma_state_offset]

        # this makes sure that coefficient for all zeroes dimension will be zero
        # without this calculated coefficient may be very large and unrealistic
        # w[np.isclose(w, 0)] = 0
        # for i in range(w.shape[1]):
        #     if np.allclose(w[:, i], 0):
        #         w[:, i] = 0
        # print(self._ar_state_offset, R.shape)
        # raise

        # ensure nan is not conclude
        nan_mask = np.any(np.isnan(w) | np.isinf(w), axis=1) | (
            np.isnan(y_tilde) | np.isinf(y_tilde)
        )
        if np.all(nan_mask):
            warn("something wrong with parameter, all weights being nan", ValueWarning)
            return np.zeros((w.shape[0], 1))
        elif np.any(nan_mask):
            y_tilde = y_tilde[~nan_mask]
            w = w[~nan_mask]

        # constant is needed in auther's implementation
        # w = add_constant(w, prepend=False)
        w = w
        # mod_ols = GLM(y_tilde, w, missing="drop")
        # mod_ols = OLS(y_tilde, w, missing="drop")
        # mod_ols = RLM(y_tilde, w, missing="drop")

        # res_ols = mod_ols.fit()
        # return res_ols.params
        p, *_ = linalg.lstsq(w, y_tilde)
        return p

    @cache_readonly
    def _optim_param_scale(self):
        scale = []
        if not self.concentrate_scale:
            scale += [0.01]
        if self.boxcox:
            scale += [0.001]
        # scale += [1]  # alpha scale
        scale += [0.01]
        if self.trend:
            # scale += [1]  # beta scale
            scale += [0.01]
            if self.damped_trend:
                scale += [0.01]  # phi scale
        if self.k_harmonics > 0:
            scale += [1e-5] * self.k_periods * 2  # seasonal parameters scale
        if self.autoregressive:
            scale += [0.1] * np.sum(self.order)  # ARMA parameters scale
        _scale = np.r_[scale]
        return _scale.astype(self.ssm.dtype)

    @property
    def start_params(self):
        if not hasattr(self, "parameters"):
            return []

        # Make sure starting parameters aren't beyond or right on the bounds
        bounds = [(x[0] + 1e-3, x[1] - 1e-3) for x in self.bounds]

        _start_params = {}

        endog = self.endog
        exog = self.exog
        total_periods = np.sum(self.periods)

        if not self.concentrate_scale:
            _start_params["state_var"] = 1

        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[~np.isnan(endog)]
            if exog is not None:
                exog = exog[mask]

        if self.boxcox:
            if self.seasonal:
                window_length = int(max(self.periods))
            else:
                window_length = 4
            endog, _start_params["boxcox"] = self.transform_boxcox(
                self.endog[:, 0], bounds=bounds[4], window_length=window_length
            )
        else:
            endog = self.endog[:, 0]

        if total_periods > 16:
            _start_params["level_alpha"] = 0.09
            # _start_params["level_alpha"] = 1e-6
        else:
            _start_params["level_alpha"] = 0.09

        # trend beta
        if self.trend:
            if total_periods > 16:
                # _start_params["trend_beta"] = 5e-7
                _start_params["trend_beta"] = 0.05
            else:
                _start_params["trend_beta"] = 0.05
            if self.damped_trend:
                _start_params["damped_trend"] = np.clip(0.999, *bounds[2])

        # seasonal gammas
        if self.seasonal:
            n = self.k_periods
            # _start_params["seasonal_gamma"] = [0.001] * n * 2
            _start_params["seasonal_gamma"] = []
            for i, k in enumerate(self.harmonics):
                # _start_params["seasonal_gamma"].extend([0.001 / k / 10**i] * 2)
                _start_params["seasonal_gamma"].extend([0.001] * 2)

        # Regression
        if self.regression and self.mle_regression:
            _start_params["reg_coeff"] = [0.1]

        # AR
        if self.autoregressive:
            p, q = self.order
            # avoid transition to be singular
            if p:
                _start_params["ar_coeff"] = [0] * p
            if q:
                _start_params["ma_coef"] = [0] * q

        start_params = []
        for key in self.parameters.keys():
            if np.isscalar(_start_params[key]):
                start_params.append(_start_params[key])
            else:
                start_params.extend(_start_params[key])
        return np.asarray(start_params, dtype=self.ssm.dtype)

    @property
    def param_names(self):

        if not hasattr(self, "parameters"):
            return []
        param_names = []
        p, q = self.order
        for key in self.parameters.keys():
            if key == "state_var":
                param_names.append("state.var")
            elif key == "level_alpha":
                param_names.append("alpha.level")
            elif key == "trend_beta":
                param_names.append("beta.trend")
            # elif key == "trend_intercept":
            #     param_names.append("intercept.trend")
            elif key == "damped_trend":
                param_names.append("phi.damped_trend")
            elif key == "seasonal_gamma":
                for i, name in enumerate(self.period_names, 1):
                    param_names.append(f"gamma1.{name}")
                    param_names.append(f"gamma2.{name}")
            elif key == "ar_coeff":
                for i in range(p):
                    param_names.append(f"ar.L{i + 1:d}.coefs")
            elif key == "boxcox":
                param_names.append("lambda.boxcox")
            elif key == "ma_coef":
                for i in range(q):
                    param_names.append(f"ma.L{i + 1:d}.coefs")
            elif key == "reg_coeff":
                param_names += [
                    f"beta.{self.exog_names[i]}.coefs" for i in range(self.k_exog)
                ]
            else:
                param_names.append(key)
        return param_names

    @Appender(MLEModel.transform_params.__doc__)
    def transform_params(self, unconstrained):
        unconstrained = np.array(unconstrained, ndmin=1) * self._optim_param_scale
        constrained = np.zeros_like(unconstrained)

        offset = 0
        # print(unconstrained)

        # # Positive parameters: obs_cov, state_cov
        if not self.concentrate_scale:
            constrained[offset] = unconstrained[offset] ** 2
            offset += 1

        if self.boxcox:
            # constrained[offset] = _ensure_bound_constrait(unconstrained[offset])
            bc_lower, bc_upper = self.bounds[4]
            constrained[offset] = constrain_bound(
                unconstrained[offset], lower=bc_lower, upper=bc_upper
            )
            offset += 1

        # # Level Alpha
        # ## we use constrains from Table 2 for stability (Hyndman et al. 2007)
        # ## using usual region for now
        low, high = self.bounds[0]
        # alpha = constrain_bound(unconstrained[offset], lower=low, upper=high)
        alpha = unconstrained[offset]
        constrained[offset] = alpha
        offset += 1

        if self.trend:
            low, high = self.bounds[1]
            high = min(high, alpha)
            beta = unconstrained[offset]
            # constrained[offset] = constrain_bound(beta, lower=low, upper=high)
            constrained[offset] = beta

            offset += 1

            if self.damped_trend:
                low, high = self.bounds[2]
                # 0 < phi <= 1
                phi = unconstrained[offset]
                constrained[offset] = constrain_bound(phi, lower=low, upper=high)
                offset += 1

        if self.seasonal:
            # low, high = self.bounds[3]
            # high = min(high, (1 - alpha) / self.k)
            num_k = 2 * self.k_periods
            # _s = np.s_[offset:offset + num_k:2]
            _s = np.s_[offset : offset + num_k]
            # constrained[_s] = constrain_bound(unconstrained[_s], -1 + 1e-6, 1 - 1e-6)
            constrained[_s] = unconstrained[_s]
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
                if self.enforce_invertibility:
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

    @Appender(MLEModel.untransform_params.__doc__)
    def untransform_params(self, constrained):
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros_like(constrained)

        offset = 0

        # Positive parameters: z, state_cov
        if not self.concentrate_scale:
            unconstrained[0] = constrained[0] ** 0.5
            offset += 1

        if self.boxcox:
            bc_lower, bc_upper = self.bounds[4]
            unconstrained[offset] = unconstrain_bound(
                constrained[offset], lower=bc_lower, upper=bc_upper
            )
            offset += 1

        # alpha
        low, high = self.bounds[0]
        alpha = constrained[offset]
        unconstrained[offset] = alpha
        offset += 1

        if self.trend:
            low, high = self.bounds[1]
            high = min(high, alpha)
            unconstrained[offset] = constrained[offset]
            offset += 1
            if self.damped_trend:
                low, high = self.bounds[2]
                unconstrained[offset] = unconstrain_bound(
                    constrained[offset], lower=low, upper=high
                )
                offset += 1

        if self.seasonal:
            low, high = self.bounds[3]
            num_k = 2 * len(self.periods)
            _s = np.s_[offset : offset + num_k]
            unconstrained[_s] = constrained[_s]
            # unconstrained[_s] = unconstrain_bound(constrained[_s], -1 + 1e-6, 1 - 1e-6)
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
                if self.enforce_invertibility:
                    unconstrained[offset : offset + q] = (
                        unconstrain_stationary_univariate(
                            -constrained[offset : offset + q]
                        )
                    )
                else:
                    unconstrained[offset : offset + q] = constrained[
                        offset : offset + q
                    ]
                offset += q
        unconstrained[offset:] = constrained[offset:]
        return unconstrained / self._optim_param_scale

    def _reset_boxcox_lambda(self, lmbda):
        endog_touse, _ = self.transform_boxcox(
            self.endog, lmbda, bounds=self.bounds[-1]
        )
        self.ssm.bind(endog_touse)
        self.ssm._representations.clear()
        self.ssm._statespaces.clear()

    @Appender(MLEModel.update.__doc__)
    def update(
        self, params, transformed=True, includes_fixed=False, complex_step=False
    ):
        params = super().update(
            params, transformed=transformed, includes_fixed=includes_fixed
        )
        param_offset = 0
        state_offset = 0
        s_dtype = params.dtype

        # cov
        if not self.concentrate_scale:
            cov = params[0]
            # self["obs_cov", 0, 0] = cov
            np.fill_diagonal(self["state_cov"], cov)
            # self["state_cov", 0, 0] = cov
            param_offset += 1

        if self.boxcox:
            self._boxcox_lambda = params[param_offset]
            self._reset_boxcox_lambda(self._boxcox_lambda)
            param_offset += 1

        # level alpha
        alpha = params[param_offset]
        param_offset += 1
        self["g", self._level_state_offset, 0] = alpha
        state_offset += 1

        if self.trend:
            beta = params[param_offset]
            param_offset += 1

            # self["selection", matrix_offset, 0] = beta
            self["g", self._trend_state_offset, 0] = beta

            # assign intercept b for trend
            # self["state_intercept", 1] = b
            # state_offset += 1

            # damped
            if self.damped_trend:
                phi = params[param_offset]

                _offset = self._trend_state_offset
                self["F", _offset - 1 : _offset + 1, _offset] = phi
                self["w", 0, _offset] = phi
                # store in internal design matrix
                param_offset += 1

        # Seasonal gamma
        if self.seasonal:
            n = self.k_periods
            tau = self.k_harmonics
            gamma = params[param_offset : param_offset + 2 * n]
            param_offset += 2 * n
            _offset = self._seasonal_state_offset

            j = 0
            gamma_selection = []
            # gamma_design = np.zeros(tau)
            for i, k in enumerate(self.harmonics, 1):
                gamma_selection = np.r_[
                    gamma_selection, np.repeat(gamma[2 * i - 2 : 2 * i], k)
                ]
                j += 2 * k

            self["g", _offset : _offset + tau, 0] = gamma_selection.astype(s_dtype)

            state_offset += tau

        # if self.autoregressive:
        p, q = self.order
        # AR
        if p:
            ar = params[param_offset : param_offset + p]
            param_offset += p
            s = np.s_[self._ar_state_offset : self._ar_state_offset + p]

            self["w", 0, s] = ar
            # # self["design", 0, col_indice+1] = 1

            self["F", 0, s] = alpha * ar
            if self.trend:
                self["F", 1, s] = beta * ar
            if self.seasonal:
                self["F", self._seasonal_state_offset : self._ar_state_offset, s] = (
                    gamma_selection[:, None] * ar
                )

            self["F", self._ar_state_offset, s] = ar
            state_offset += p

        # MA
        if q:
            ma = params[param_offset : param_offset + q]
            s = np.s_[self._ma_state_offset:self._ma_state_offset + q]
            self["w", 0, s] = ma
            self["F", 0, s] = alpha * ma
            if self.trend:
                self["F", 1, s] = beta * ma
            if self.seasonal:
                self["F", self._seasonal_state_offset:self._arma_state_offset, s] = (
                    gamma_selection[:, None] * ma
                )
            self["F", self._arma_state_offset, s] = ma

            # # the change apply into internal design
            # self["g", state_offset, 0] = 1
            param_offset += q

        if self.regression and self.mle_regression and True:
            self.ssm["obs_intercept"] = np.dot(
                self.data.exog, params[param_offset : param_offset + self.k_exog]
            )[None, :]
            param_offset += self.k_exog

        self._need_recompute_state = True
        self._initialize_stationary_cov_statespace()

    @Appender(MLEModel.filter.__doc__)
    def filter(
        self,
        params,
        *args,
        transformed=True,
        includes_fixed=False,
        complex_step=False,
        cov_type=None,
        cov_kwds=None,
        return_ssm=False,
        results_class=None,
        results_wrapper_class=None,
        low_memory=False,
        **kwargs,
    ):
        if self.initialization_method == "concentrated":
            if self._need_recompute_state:
                initial_states = self._compute_concentrated_states(
                    params, *args, **kwargs
                )
                self._need_recompute_state = False
            else:
                initial_states = self._initial_state
            self._initialize_constant_statespace(initial_states)

        results = super().filter(
            params,
            transformed=transformed,
            includes_fixed=includes_fixed,
            complex_step=complex_step,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            return_ssm=return_ssm,
            results_class=results_class,
            results_wrapper_class=results_wrapper_class,
            low_memory=low_memory,
            **kwargs,
        )

        if self.initialization_method == "concentrated":
            self.ssm.initialization.constant = np.zeros(self.k_states)
        return results

    @Appender(MLEModel.smooth.__doc__)
    def smooth(
        self,
        params,
        *args,
        cov_type=None,
        cov_kwds=None,
        return_ssm=False,
        results_class=None,
        results_wrapper_class=None,
        **kwargs,
    ):
        if self.initialization_method == "concentrated":
            if self._need_recompute_state:
                initial_states = self._compute_concentrated_states(
                    params, *args, **kwargs
                )
                self._need_recompute_state = False
            else:
                initial_states = self._initial_state
            self._initialize_constant_statespace(initial_states)

        results = super().smooth(
            params,
            *args,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            return_ssm=return_ssm,
            results_class=results_class,
            results_wrapper_class=results_wrapper_class,
            **kwargs,
        )

        if self.initialization_method == "concentrated":
            self.ssm.initialization.constant = np.zeros(self.k_states)
        return results

    @Appender(MLEModel.loglike.__doc__)
    def loglike(self, params, *args, **kwargs):
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

        if self.check_admissible and not self._check_admissible():
            warn("model is not admissible. parameters maybe unstable")
            return -(10**10)

        if self.initialization_method == "concentrated":
            initial_states = self._compute_concentrated_states(
                params,
                transformed=True,
                includes_fixed=includes_fixed,
                complex_step=complex_step,
            )
            # initial_states = self._compute_concentrated_states(params, *args, **kwargs)
            self._need_recompute_state = False
            self._initialize_constant_statespace(initial_states)
            loglike = self.ssm.loglike()
            # reset the initial constant
            self.ssm.initialization.constant = np.zeros(self.k_states)
        else:
            loglike = super().loglike(params, *args, **kwargs)
        # if self.initialization_method == "concentrated":
        #     try:
        #         initial_states = self._compute_concentrated_states(params, *args, **kwargs)
        #         self._need_recompute_state = False
        #         self._initialize_constant_statespace(initial_states)
        #     except ValueError:
        #         warn(
        #             "model is not admissive. parameters maybe unstable"
        #         )
        #         return -(10**10)
        #     loglike = self.ssm.loglike()
        #     # reset the initial constant
        #     self.ssm.initialization.constant = np.zeros(self.k_states)
        # else:
        #     loglike = super().loglike(params, *args, **kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues, but the averaging is done
        # automatically in the base model `fit` method

        if self.boxcox:
            loglike += (self._boxcox_lambda - 1) * np.nansum(
                self.data.log_endog[self.loglikelihood_burn :]
            )

        return loglike

    @Appender(MLEModel.fit.__doc__)
    def fit(self, *args, **kwargs):
        kwargs.setdefault("method", "nm")
        kwargs.setdefault("maxiter", (self.k_params * 200) ** 2)
        return super().fit(*args, **kwargs)

    @Appender(BoxCox.untransform_boxcox.__doc__)
    def untransform_boxcox(self, x, lmbda, method="naive", variance=None):
        if method != "biasadj":
            return super().untransform_boxcox(x, lmbda, method)

        # Now compute the regression components as described in
        # De Livera et al. (2011), equation (12b).
        assert variance is not None, "variance should be provided when method='biasadj'"
        if np.isclose(lmbda, 0.0):
            return np.exp(x) * (1 + 0.5 * variance)
        else:
            t = lmbda * x + 1
            return np.power(t, 1 / lmbda) * (
                1 + 0.5 * variance * (1 - lmbda) / np.power(t, 2)
            )


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
        super().__init__(model, params, filter_results, cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()
        if model.autoregressive:
            initial_state_slice = np.s_[:model._ar_state_offset]
        else:
            initial_state_slice = np.s_[:-1]
        self.initial_state = model._initial_state[:model._arma_state_offset]

        if isinstance(self.data, PandasData):
            # index = self.data.row_labels
            self.initial_state = pd.DataFrame(
                [self.initial_state],
                columns=model.state_names[initial_state_slice],
            )
            # if model._index_dates and model._index_freq is not None:
            #     self.initial_state.index = index.shift(-1)[:1]

        self.specification = Bunch(
            **{
                # Model options
                "level": True,
                "k_hidden_states": self.model.k_hidden_states,
                "trend": self.model.trend,
                "seasonal": self.model.seasonal,
                "arma_order": self.model.order,
                "boxcox": self.model.boxcox,
                "damped_trend": self.model.damped_trend,
                "autoregressive": self.model.autoregressive,
                "concentrate_scale": self.model.concentrate_scale,
                "biasadj": self.model.biasadj,
            }
        )

    @cache_readonly
    def fittedvalues(self):
        fittedvalues = super().fittedvalues
        if self.model.boxcox:
            _fittedvalues = self.model.untransform_boxcox(
                fittedvalues, self.model._boxcox_lambda
            )
            if isinstance(fittedvalues, PandasData):
                fittedvalues = pd.Series(_fittedvalues, index=fittedvalues.index)
            else:
                fittedvalues = _fittedvalues
        return fittedvalues

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglike(self.params)

    @cache_readonly
    def compact_llf(self):
        r"""
        the simply loglikelihood  value absent inessential constants

        :math:`equiv_llf = -(nlog(SSE) - 2 * (\omega - 1) * log(y))`
        """
        llf = self.nobs * np.log(
            np.sum(
                self.filter_results.forecasts_error[0, self.loglikelihood_burn :] ** 2
            )
        )
        if self.specification.boxcox:
            llf -= (
                2
                * (self.model._boxcox_lambda - 1)
                * np.nansum(self.model.data.log_endog[self.loglikelihood_burn :])
            )
        return -llf

    @property
    def level(self):
        offset = self.model._level_state_offset
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
            offset = self.model._trend_state_offset
            # offset = int(spec.level)
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
            k = self.model.harmonics
            offset = self.model._seasonal_state_offset
            # offset = int(spec.level + spec.trend)
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
            offset = int(self.model._arma_state_offset)

            s = []
            if p > 0:
                s = np.r_[s, offset]
            elif q > 0:
                s = np.r_[s, offset]

            s = s.astype(int)

            filtered = self.filtered_state[s].sum(axis=0)
            filtered_cov = self.filtered_state_cov[s, s].sum(axis=0)
            out = Bunch(
                filtered=filtered,
                filtered_cov=filtered_cov,
                smoothed=None,
                smoothed_cov=None,
                offset=offset,
            )

            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[s].sum(axis=0)
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[s, s].sum(axis=0)
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
        biasadj=None,
    ):

        from pandas.plotting import register_matplotlib_converters
        from scipy.stats import norm

        from statsmodels.graphics.utils import _import_mpl

        plt = _import_mpl()
        register_matplotlib_converters()

        # Determine which results we have
        if which is None:
            which = "filtered" if self.smoothed_state is None else "smoothed"

        spec = self.specification
        biasadj = biasadj if biasadj is not None else spec.biasadj
        boxcox_method = "biasadj" if biasadj else "naive"
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
            xlim = (dates[start], dates[end or -1])
        else:
            dates = np.arange(len(self.resid))
            xlim = (0, len(self.resid) - 1)

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
            predict_cov = self.filter_results.forecasts_error_cov[0, 0]
            std_errors = np.sqrt(predict_cov)
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors

            if spec.boxcox:
                lmbda = self.model._boxcox_lambda
                predict = self.model.untransform_boxcox(
                    predict, lmbda, method=boxcox_method, variance=predict_cov
                )
                ci_lower = self.model.untransform_boxcox(
                    ci_lower, lmbda, method=boxcox_method, variance=predict_cov
                )
                ci_upper = self.model.untransform_boxcox(
                    ci_upper, lmbda, method=boxcox_method, variance=predict_cov
                )

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
            ax.set_xlim(xlim)

        for component, is_plotted in components.items():
            if not is_plotted:
                continue

            ax = axes[plot_idx]
            plot_idx += 1

            component_bunch = getattr(self, component)

            # Check for a valid estimation type
            if which not in component_bunch:
                raise ValueError("Invalid type of state estimate.")

            # Get the predicted values
            value = component_bunch[which]

            # Plot
            state_label = "%s (%s)" % (component.title(), which)
            ax.plot(dates[start:end], value[start:end], label=state_label)

            ax.set_title("%s component" % component.title())
            ax.set_xlim(xlim)

        if seasonal:
            component_bunch = self.seasonal

            for i, (m, k, name) in enumerate(
                zip(self.model.periods, self.model.harmonics, self.model.period_names)
            ):
                ax = axes[plot_idx]
                plot_idx += 1

                # Get the predicted values
                value = component_bunch[which][i]

                # Plot
                state_label = "%s(%s, %s)" % (name, m, k)
                ax.plot(dates[start:end], value[start:end], label=state_label)

                ax.set_title(state_label + " component")
                ax.set_xlim(xlim)

        if resid:
            ax = axes[plot_idx]
            plot_idx += 1

            value = self.resid
            ax.vlines(
                dates[start:end], [0], value[start:end], color="C0", label="Remainder"
            )
            ax.legend(loc=legend_loc)

            ax.set_title("Residuals")
            ax.set_xlim(xlim)
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

    @Appender(MLEResults.forecast.__doc__)
    def forecast(self, steps=1, **kwargs):
        forecast_values = super().forecast(steps, **kwargs)
        if self.model.boxcox:
            forecast_values = self.model.untransform_boxcox(
                forecast_values, self.model._boxcox_lambda
            )

        return forecast_values

    @Appender(MLEResults.simulate.__doc__)
    def simulate(
        self,
        nsimulations,
        measurement_shocks=None,
        state_shocks=None,
        initial_state=None,
        anchor=None,
        repetitions=None,
        exog=None,
        extend_model=None,
        extend_kwargs=None,
        pretransformed_measurement_shocks=True,
        pretransformed_state_shocks=True,
        pretransformed_initial_state=True,
        random_state=None,
        **kwargs,
    ):
        sim = super().simulate(
            nsimulations,
            measurement_shocks,
            state_shocks,
            initial_state,
            anchor,
            repetitions,
            exog,
            extend_model,
            extend_kwargs,
            pretransformed_measurement_shocks,
            pretransformed_state_shocks,
            pretransformed_initial_state,
            random_state,
            **kwargs,
        )
        if self.model.boxcox:
            sim[:] = self.model.untransform_boxcox(sim, self.model._boxcox_lambda)

        return sim

    @Appender(MLEResults.summary.__doc__)
    def summary(
        self, alpha=0.05, start=None, separate_params=True, title=None, **kwargs
    ):
        from statsmodels.iolib.summary import summary_params

        spec = self.specification

        summary = super().summary(
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
            if self.specification.boxcox:
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
                    self.model.periods, self.model.harmonics, self.model.period_names
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

        # if self.model.initialization_method != "estimated":
        params = np.array(self.initial_state)
        if params.ndim > 1:
            params = params[0]
        if isinstance(self.initial_state, PandasData):
            names = self.initial_state.columns
        else:
            if spec.autoregressive:
                names = self.model.state_names[:self.model._arma_state_offset]
            else:
                names = self.model.state_names
        param_header = [
            "initialization method: %s" % self.model.initialization_method
        ]
        params_stubs = names
        params_data = [[forg(params[i], prec=4)] for i in range(len(params))]

        initial_state_table = SimpleTable(
            params_data, param_header, params_stubs, txt_fmt=fmt_params
        )
        summary.tables.append(initial_state_table)

        return summary


class TBATSResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._methods, _methods)


wrap.populate_wrapper(TBATSResultsWrapper, TBATSResults)


def _safe_tbats_fit(endog, model_kw, fit_kw, model_class=TBATSModel, **kwargs):
    try:
        mod = model_class(endog, **model_kw)
        sp = kwargs.pop("start_params", None)
        if sp is not None:
            _sp = mod.start_params
            # we assume the order is correct
            _sp[: sp.shape[0]] = sp
            sp = _sp
        res = mod.fit(start_params=sp, **fit_kw)
        return res
    except LinAlgError as e:
        # SVD convergence failure on badly misspecified models
        warn(f"LinAlgError: {str(e)}, {model_kw}, {fit_kw}", RuntimeWarning)
        return
    except Exception as e:  # no idea what happened
        print(f"error fits models, error: {e}", model_kw, fit_kw)
        # raise e
        # raise e
        return None
        # print(e.with_traceback())
        # return


def tbats_k_order_select_ic(
    y,
    periods,
    use_trend=None,
    use_box_cox=None,
    damped_trend=None,
    max_ar=5,
    max_ma=5,
    ic="aic",
    model_class=TBATSModel,
    return_params=False,
    restrict_periods=True,
    **kwargs,
):
    from functools import partial
    from statsmodels.tsa.stattools import arma_order_select_ic

    model_kw = kwargs.get("model_kw", {})
    fit_kw = kwargs.get("fit_kw", {})

    # default model_kw
    model_kw.setdefault("trend", False if use_trend is None else use_trend)
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
    model_kw["harmonics"] = ks

    calc_aic = partial(_safe_tbats_fit, y, fit_kw=fit_kw, model_class=model_class)

    _ic = lambda obj: getattr(obj, ic) if obj is not None else -np.inf

    if np.any(data <= 0) or use_box_cox is not True:
        model_kw["box_cox"] = False
        y2 = data.copy()
    elif use_box_cox is True:
        bounds = model_kw.get("bounds", (-1, 2))
        window_length = int(max(periods))
        y2, *_ = BoxCox().transform_boxcox(
            data, bounds=bounds, window_length=window_length
        )
        # y2 = np.log(data)
        model_kw["box_cox"] = True

    decomp_mod = seasonal_decompose(
        y2, period=int(np.max(periods)), extrapolate_trend="freq"
    )
    ks = seasonal_fourier_k_select_ic(
        decomp_mod.seasonal,
        periods,
        grid_search=grid_search,
        ic=ic,
        restrict_periods=restrict_periods,
    )
    model_kw["harmonics"] = ks
    resid = decomp_mod.resid

    # baseline
    best_model = calc_aic(model_kw=model_kw)
    min_ic = _ic(best_model)

    # Use Trend
    if use_trend is None:
        model_kw["trend"] = True
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
            model_kw["trend"] = False

    elif use_trend is True:
        # alread set, continue
        pass
    else:
        model_kw["trend"] = False

    # best_order = (0, 0)
    # ar_range = np.arange(0, max_ar + 1)
    # ma_range = np.arange(0, max_ma + 1)
    # for ma in ma_range:
    #     for ar in ar_range:
    #         if (ar, ma) == (0, 0):
    #             continue
    #         model_kw["order"] = (ar, ma)
    #         mod = calc_aic(model_kw=model_kw, start_params=best_model.params)
    #         if mod and _ic(mod) < min_ic:
    #             best_model = mod
    #             min_ic = _ic(mod)
    #             best_order = (ar, ma)
    # model_kw["order"] = best_order
    # print(f"resid best arma order: {best_order}")

    # arma process
    # print(best_model.resid)
    arma_result = arma_order_select_ic(
        best_model.resid,
        max_ar=max_ar,
        max_ma=max_ma,
        ic=ic,
        model_kw=dict(trend="n", concentrate_scale=True, missing="drop"),
    )[ic].fillna(np.inf)
    # we manually handle the result because sometimes there will be nan in results.
    delta = np.ascontiguousarray(np.abs(arma_result.min().min() - arma_result))
    ncols = delta.shape[1]
    loc = np.argmin(delta)
    order = (loc // ncols, loc % ncols)
    # print(arma_result)

    print(f"resid best arma order: {order}")
    if order != (0, 0):
        model_kw["order"] = order
        mod = calc_aic(model_kw=model_kw)
        if mod is not None:
            print(f"arima {ic}: {_ic(mod)} for {order}, current best {ic}: {min_ic}")
        if mod and _ic(mod) < min_ic:
            best_model = mod
            min_ic = _ic(mod)
        else:
            model_kw["order"] = (0, 0)
    model_kw.update(k=ks)

    if return_params:
        return model_kw
    elif not fit_kw.get("low_memory", True):
        return model_kw, best_model
    else:
        return model_kw, best_model.model.smooth(best_model.params)
