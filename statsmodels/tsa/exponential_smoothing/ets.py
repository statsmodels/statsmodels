r"""
ETS models for time series analysis.

The ETS models are a family of time series models. They can be seen as a
generalization of simple exponential smoothing to time series that contain
trends and seasonalities. Additionally, they have an underlying state space
model.

An ETS model is specified by an error type (E; additive or multiplicative), a
trend type (T; additive or multiplicative, both damped or undamped, or none),
and a seasonality type (S; additive or multiplicative or none).
The following gives a very short summary, a more thorough introduction can be
found in [1]_.

Denote with :math:`\circ_b` the trend operation (addition or
multiplication), with :math:`\circ_d` the operation linking trend and dampening
factor :math:`\phi` (multiplication if trend is additive, power if trend is
multiplicative), and with :math:`\circ_s` the seasonality operation (addition
or multiplication).
Furthermore, let :math:`\ominus` be the respective inverse operation
(subtraction or division).

With this, it is possible to formulate the ETS models as a forecast equation
and 3 smoothing equations. The former is used to forecast observations, the
latter are used to update the internal state.

.. math::

    \hat{y}_{t|t-1} &= (l_{t-1} \circ_b (b_{t-1}\circ_d \phi))\circ_s s_{t-m}\\
    l_{t} &= \alpha (y_{t} \ominus_s s_{t-m})
             + (1 - \alpha) (l_{t-1} \circ_b (b_{t-1} \circ_d \phi))\\
    b_{t} &= \beta/\alpha (l_{t} \ominus_b l_{t-1})
             + (1 - \beta/\alpha) b_{t-1}\\
    s_{t} &= \gamma (y_t \ominus_s (l_{t-1} \circ_b (b_{t-1}\circ_d\phi))
             + (1 - \gamma) s_{t-m}

The notation here follows [1]_; :math:`l_t` denotes the level at time
:math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m` is the
number of seasonal periods, and :math:`\phi` a trend damping factor.
The parameters :math:`\alpha, \beta, \gamma` are the smoothing parameters,
which are called ``smoothing_level``, ``smoothing_trend``, and
``smoothing_seasonal``, respectively.

Note that the formulation above as forecast and smoothing equation does not
distinguish different error models -- it is the same for additive and
multiplicative errors. But the different error models lead to different
likelihood models, and therefore will lead to different fit results.

The error models specify how the true values :math:`y_t` are updated. In the
additive error model,

.. math::

    y_t = \hat{y}_{t|t-1} + e_t,

in the multiplicative error model,

.. math::

    y_t = \hat{y}_{t|t-1}\cdot (1 + e_t).

Using these error models, it is possible to formulate state space equations for
the ETS models:

.. math::

   y_t &= Y_t + \eta \cdot e_t\\
   l_t &= L_t + \alpha \cdot (M_e \cdot L_t + \kappa_l) \cdot e_t\\
   b_t &= B_t + \alpha \beta^* \cdot (M_e \cdot B_t + \kappa_b) \cdot e_t\\
   s_t &= S_t + (1 - \alpha) \gamma^* \cdot (M_e \cdot S_t+\kappa_s)\cdot e_t\\

with

.. math::

   B_t &= b_{t-1} \circ_d \phi\\
   L_t &= l_{t-1} \circ_b B_t\\
   S_t &= s_{t-m}\\
   Y_t &= L_t \circ_s S_t,

and

.. math::

   \eta &= \begin{cases}
               Y_t\quad\text{if error is multiplicative}\\
               1\quad\text{else}
           \end{cases}\\
   M_e &= \begin{cases}
               1\quad\text{if error is multiplicative}\\
               0\quad\text{else}
           \end{cases}\\

and, when using the additve error model,

.. math::

   \kappa_l &= \begin{cases}
               \frac{1}{S_t}\quad
               \text{if seasonality is multiplicative}\\
               1\quad\text{else}
           \end{cases}\\
   \kappa_b &= \begin{cases}
               \frac{\kappa_l}{l_{t-1}}\quad
               \text{if trend is multiplicative}\\
               \kappa_l\quad\text{else}
           \end{cases}\\
   \kappa_s &= \begin{cases}
               \frac{1}{L_t}\quad\text{if seasonality is multiplicative}\\
               1\quad\text{else}
           \end{cases}

When using the multiplicative error model

.. math::

   \kappa_l &= \begin{cases}
               0\quad
               \text{if seasonality is multiplicative}\\
               S_t\quad\text{else}
           \end{cases}\\
   \kappa_b &= \begin{cases}
               \frac{\kappa_l}{l_{t-1}}\quad
               \text{if trend is multiplicative}\\
               \kappa_l + l_{t-1}\quad\text{else}
           \end{cases}\\
   \kappa_s &= \begin{cases}
               0\quad\text{if seasonality is multiplicative}\\
               L_t\quad\text{else}
           \end{cases}

When fitting an ETS model, the parameters :math:`\alpha, \beta^*`, \gamma^*,
\phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \ldots, s_{-m}` are
selected as maximizers of log likelihood.

References
----------
.. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
   principles and practice*, 3rd edition, OTexts: Melbourne,
   Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
"""

from collections import OrderedDict
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import _distn_infrastructure, rv_continuous, rv_discrete

import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import bool_like, int_like, string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
    _initialization_heuristic,
)
from statsmodels.tsa.tsatools import freq_to_period

# Implementation details:

# * The smoothing equations are implemented only for models having all
#   components (trend, dampening, seasonality). When using other models, the
#   respective parameters (smoothing and initial parameters) are set to values
#   that lead to the reduced model (often zero).
#   The internal model is needed for smoothing (called from fit and loglike),
#   forecasts, and simulations.
# * Somewhat related to above: There are 3 sets of parameters: free params,
#   model params, and internal params.
#   - free params are what is passed by a user into the fit method
#   - model params are all parameters necessary for a model, and are for
#     example passed as argument to the likelihood function when
#     ``internal_params=False`` is set.
#   - internal params are what is used internally in the smoothing equations


class ETSModel(base.StateSpaceMLEModel):
    """
    ETS models.

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    error: str, optional
        The error model. "add" (default) or "mul".
    trend : str or None, optional
        The trend component model. "add", "mul", or None (default).
    damped_trend : bool, optional
        Whether or not an included trend component is damped. Default is False.
    seasonal : str, optional
        The seasonality model. "add", "mul", or None (default).
    seasonal_periods: int, optional
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Required if
        `seasonal` is not None.
    initialization_method : str, optional
        Method for initialization of the state space model. One of:

        * 'estimated' (default)
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_trend` and `initial_seasonal` if
        applicable.
        'heuristic' uses a heuristic based on the data to estimate initial
        level, trend, and seasonal state. 'estimated' uses the same heuristic
        as initial guesses, but then estimates the initial states as part of
        the fitting process.  Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float, optional
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`.  Only
        used if initialization is 'known'.
    bounds : iterable[tuple], optional
        An iterable containing bounds for the parameters. Must contain one
        element for every smoothing parameter, and optionally an element for
        every initial state, where each element is a tuple of the form (lower,
        upper).  Default is (0.0001, 0.9999) for the level, trend, and seasonal
        smoothing parameters and (0.8, 0.98) for the trend damping parameter,
        and (None, None) for the initial states (if `initialization_method` is
        'estimated').
        If `initialization_method` is 'estimated', either no bounds for the
        initial states should be given, or bounds for all initial states
        (level, trend, seasonal, depending on chosen model).
    """

    def __init__(
        self,
        endog,
        error="add",
        trend=None,
        damped_trend=False,
        seasonal=None,
        seasonal_periods=None,
        initialization_method="estimated",
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        bounds=None,
        dates=None,
        freq=None,
        missing="none",
    ):

        super().__init__(
            endog, exog=None, dates=dates, freq=freq, missing=missing
        )

        # MODEL DEFINITION
        # ================
        options = ("add", "mul", "additive", "multiplicative")
        # take first three letters of option -> either "add" or "mul"
        self.error = string_like(error, "error", options=options)[:3]
        self.trend = string_like(
            trend, "trend", options=options, optional=True
        )
        if self.trend is not None:
            self.trend = self.trend[:3]
        self.damped_trend = bool_like(damped_trend, "damped_trend")
        self.seasonal = string_like(
            seasonal, "seasonal", options=options, optional=True
        )
        if self.seasonal is not None:
            self.seasonal = self.seasonal[:3]

        self.has_trend = self.trend is not None
        self.has_seasonal = self.seasonal is not None

        if self.has_seasonal:
            self.seasonal_periods = int_like(
                seasonal_periods, "seasonal_periods", optional=True
            )
            if seasonal_periods is None:
                self.seasonal_periods = freq_to_period(self._index_freq)
            if self.seasonal_periods <= 1:
                raise ValueError("seasonal_periods must be larger than 1.")
        else:
            # in case the model has no seasonal component, we internally handle
            # this as if it had an additive seasonal component with
            # seasonal_periods=1, but restrict the smoothing parameter to 0 and
            # set the initial seasonal to 0.
            self.seasonal_periods = 1

        # reject invalid models
        if np.any(self.endog <= 0) and (
            self.error == "mul"
            or self.trend == "mul"
            or self.seasonal == "mul"
        ):
            raise ValueError(
                "endog must be strictly positive when using"
                "multiplicative error, trend or seasonal components."
            )
        if self.damped_trend and not self.has_trend:
            raise ValueError("Can only dampen the trend component")

        # INITIALIZATION METHOD
        # =====================
        self.set_initialization_method(
            initialization_method,
            initial_level,
            initial_trend,
            initial_seasonal,
        )

        # BOUNDS
        # ======
        if bounds is None:
            self.bounds = self._default_param_bounds()
        else:
            # first, check whether only smoothing parameter bounds or also
            # initial state bounds are provided
            if len(bounds) == self.k_params:
                self.bounds = bounds
            elif (
                self.initialization_method == "estimated"
                and len(bounds) == self._k_smoothing_params
            ):
                self.bounds = bounds + [(None, None)] * self._k_initial_states
            else:
                raise ValueError(
                    "`bounds` must be a list of tuples of boundary"
                    " values for each parameter"
                )

        # SMOOTHER
        # ========
        if self.trend == "add" or self.trend is None:
            if self.seasonal == "add" or self.seasonal is None:
                self._smoothing_func = smooth._ets_smooth_add_add
            else:
                self._smoothing_func = smooth._ets_smooth_add_mul
        else:
            if self.seasonal == "add" or self.seasonal is None:
                self._smoothing_func = smooth._ets_smooth_mul_add
            else:
                self._smoothing_func = smooth._ets_smooth_mul_mul

    def set_initialization_method(
        self,
        initialization_method,
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
    ):
        """
        Sets a new initialization method for the state space model.

        Parameters
        ----------
        initialization_method : str, optional
            Method for initialization of the state space model. One of:

            * 'estimated' (default)
            * 'heuristic'
            * 'known'

            If 'known' initialization is used, then `initial_level` must be
            passed, as well as `initial_trend` and `initial_seasonal` if
            applicable.
            'heuristic' uses a heuristic based on the data to estimate initial
            level, trend, and seasonal state. 'estimated' uses the same
            heuristic as initial guesses, but then estimates the initial states
            as part of the fitting process. Default is 'estimated'.
        initial_level : float, optional
            The initial level component. Only used if initialization is
            'known'.
        initial_trend : float, optional
            The initial trend component. Only used if initialization is
            'known'.
        initial_seasonal : array_like, optional
            The initial seasonal component. An array of length `seasonal`. Only
            used if initialization is 'known'.
        """
        if hasattr(self, "initialization_method"):
            orig_init_method = self.initialization_method
        self.initialization_method = string_like(
            initialization_method,
            "initialization_method",
            options=("estimated", "known", "heuristic"),
        )
        if self.initialization_method == "known":
            if initial_level is None:
                raise ValueError(
                    "`initial_level` argument must be provided"
                    ' when initialization method is set to "known".'
                )
            if self.has_trend and initial_trend is None:
                raise ValueError(
                    "`initial_trend` argument must be provided"
                    " for models with a trend component when"
                    ' initialization method is set to "known".'
                )
            if self.has_seasonal and initial_seasonal is None:
                raise ValueError(
                    "`initial_seasonal` argument must be provided"
                    " for models with a seasonal component when"
                    ' initialization method is set to "known".'
                )
        elif (
            self.initialization_method == "heuristic"
            or self.initialization_method == "estimated"
        ):
            (
                initial_level,
                initial_trend,
                initial_seasonal,
            ) = _initialization_heuristic(
                self.endog,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            )
        if not self.has_trend:
            initial_trend = 0
        if not self.has_seasonal:
            initial_seasonal = 0
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal

        # if this is not called from the constructor, we have to change the
        # bounds
        if hasattr(self, "bounds"):
            if orig_init_method == "estimated" and initialization_method in (
                "heuristic",
                "known",
            ):
                # if we change from 'estimated' to 'known' or 'heuristic', we
                # don't need the bounds for the initial states anymore
                self.bounds = self.bounds[0:self.k_params]
            elif (
                orig_init_method in ("heuristic", "known")
                and initialization_method == "estimated"
            ):
                # if it was heuristic or known and we change to estimated, we
                # have to add the bounds for the initial states
                self.bounds = (
                    self.bounds + [(None, None)] * self._k_initial_states
                )

        # we also have to reset the params index dictionaries
        self._internal_params_index = OrderedDict(
            zip(self._internal_param_names, np.arange(self._k_params_internal))
        )
        self._params_index = OrderedDict(
            zip(self.param_names, np.arange(self.k_params))
        )

    @staticmethod
    def prepare_data(data):
        """
        Prepare data for use in the state space representation
        """
        endog = np.array(data.orig_endog, order="C")
        if endog.ndim != 1:
            raise ValueError("endog must be 1-dimensional")
        return endog, None

    @property
    def param_names(self):
        param_names = ["smoothing_level"]
        if self.has_trend:
            param_names += ["smoothing_trend"]
        if self.has_seasonal:
            param_names += ["smoothing_seasonal"]
        if self.damped_trend:
            param_names += ["damping_trend"]

        # Initialization
        if self.initialization_method == "estimated":
            param_names += ["initial_level"]
            if self.has_trend:
                param_names += ["initial_trend"]
            if self.has_seasonal:
                param_names += [
                    f"initial_seasonal.{i}"
                    for i in range(self.seasonal_periods)
                ]
        return param_names

    @property
    def _internal_param_names(self):
        param_names = [
            "smoothing_level",
            "smoothing_trend",
            "smoothing_seasonal",
            "damping_trend",
            "initial_level",
            "initial_trend",
        ]
        param_names += [
            f"initial_seasonal.{i}" for i in range(self.seasonal_periods)
        ]
        return param_names

    @property
    def _k_states(self):
        return 1 + int(self.has_trend) + int(self.has_seasonal)  # level

    @property
    def _k_states_internal(self):
        return 2 + self.seasonal_periods

    @property
    def _k_smoothing_params(self):
        return self._k_states + int(self.damped_trend)

    @property
    def _k_initial_states(self):
        return (
            1
            + int(self.has_trend)
            + +int(self.has_seasonal) * self.seasonal_periods
        )

    @property
    def k_params(self):
        k_params = self._k_smoothing_params
        if self.initialization_method == "estimated":
            k_params += self._k_initial_states
        return k_params

    @property
    def _k_params_internal(self):
        return 4 + 2 + self.seasonal_periods

    def _internal_params(self, params):
        """
        Converts a parameter array passed from outside to the internally used
        full parameter array.
        """
        # internal params that are not needed are all set to zero, except phi,
        # which is one
        internal = np.zeros(self._k_params_internal)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index[name]
            internal[internal_idx] = params[i]
        if not self.damped_trend:
            internal[3] = 1  # phi is 4th parameter
        if self.initialization_method != "estimated":
            internal[4] = self.initial_level
            internal[5] = self.initial_trend
            internal[6:] = self.initial_seasonal
        return internal

    def _model_params(self, internal):
        """
        Converts internal parameters to model parameters
        """
        params = np.empty(self.k_params)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index[name]
            params[i] = internal[internal_idx]
        return params

    def _set_fixed_params(self, params):
        if self._has_fixed_params:
            for i, name in enumerate(self._fixed_params):
                idx = self._fixed_params_index[i]
                params[idx] = self._fixed_params[name]

    @property
    def _start_params(self):
        # Make sure starting parameters aren't beyond or right on the bounds
        bounds = []
        for b in self.bounds:
            lb = b[0] + 1e-3 if b[0] is not None else None
            ub = b[1] + 1e-3 if b[1] is not None else None
            bounds.append((lb, ub))

        # See Hyndman p.24
        start_params = [np.clip(0.1, *bounds[0])]
        idx = 1
        if self.trend:
            start_params += [np.clip(0.01, *bounds[idx])]
            idx += 1
        if self.seasonal:
            start_params += [np.clip(0.01, *bounds[idx])]
            idx += 1
        if self.damped_trend:
            start_params += [np.clip(0.98, *bounds[idx])]
            idx += 1

        # Initialization
        if self.initialization_method == "estimated":
            start_params += [self.initial_level]
            if self.has_trend:
                start_params += [self.initial_trend]
            if self.has_seasonal:
                start_params += self.initial_seasonal.tolist()

        return np.array(start_params)

    def _default_param_bounds(self):
        """
        Default lower and upper bounds for model parameters
        """
        # traditional bounds:
        # alpha, beta*, gamma* in (0, 1), phi in [0.8, 0.98]
        n = 1 + int(self.has_trend) + int(self.has_seasonal)
        bounds = [(1e-8, 1 - 1e-8)] * n
        if self.damped_trend:
            bounds += [(0.8, 0.98)]
        if self.initialization_method == "estimated":
            n = (
                1
                + int(self.has_trend)
                + int(self.has_seasonal) * self.seasonal_periods
            )
            bounds += [(None, None)] * n
        return bounds

    def _internal_bounds(self):
        """
        Returns bounds for internal parameters
        """
        bounds = []
        for name in self._internal_param_names:
            if name in self.param_names:
                bounds.append(self.bounds[self._params_index[name]])
            elif name == "damping_trend":
                # if damping_trend is not in param_names, it is set to 1
                bounds.append((1, 1))
            else:
                # all other internal-only parameters are 0, except initials
                if name == "initial_level":
                    b = self.initial_level
                    bounds.append((b, b))
                elif name == "initial_trend":
                    b = self.initial_trend
                    bounds.append((b, b))
                elif name == "initial_seasonal":
                    for i in range(self.seasonal_periods):
                        b = self.initial_seasonal[i]
                        bounds.append((b, b))
                else:
                    bounds.append((0, 0))

        # set fixed parameters bounds
        if self._has_fixed_params:
            for i, name in enumerate(self._fixed_params):
                idx = self._fixed_params_index[i]
                val = self._fixed_params[name]
                bounds[idx] = (val, val)
        return bounds

    def fit(
        self,
        start_params=None,
        maxiter=1000,
        full_output=True,
        disp=True,
        callback=None,
        return_params=False,
        **kwargs,
    ):
        r"""
        Fit an ETS model by maximizing log-likelihood.

        Log-likelihood is a function of the model parameters :math:`\alpha,
        \beta^*, \gamma^*, \phi` (depending on the chosen model), and, if
        `initialization_method` was set to `'estimated'` in the constructor,
        also the initial states :math:`l_{-1}, b_{-1}, s_{-1}, \ldots, s_{-m}`.

        The fit is performed using the L-BFGS algorithm.

        Parameters
        ----------
        start_params : array_like, optional
            Initial values for parameters that will be optimized. If this is
            ``None``, default values will be used.
            The length of this depends on the chosen model. This should contain
            the parameters in the following order, skipping parameters that do
            not exist in the chosen model.

            * `smoothing_level` (:math:`\alpha`)
            * `smoothing_trend` (:math:`\beta^*`)
            * `smoothing_season` (:math:`\gamma^*`)
            * `damping_trend` (:math:`\phi`)

            If ``initialization_method`` was set to ``'estimated'`` (the
            default), additionally, the parameters

            * `initial_level` (:math:`l_{-1}`)
            * `initial_trend` (:math:`l_{-1}`)
            * `initial_seasonal.0` (:math:`s_{-1}`)
            * ...
            * `initial_seasonal.<m-1>` (:math:`s_{-m}`)

            also have to be specified.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        results : ETSResults
        """
        if start_params is None:
            start_params = self.start_params

        # set fixed params
        self._set_fixed_params(start_params)

        if self._has_fixed_params and len(self._free_params_index) == 0:
            final_params = start_params
            mlefit = Bunch(
                params=start_params, mle_retvals=None, mle_settings=None
            )
        else:
            start_params = self._internal_params(start_params)
            bounds = self._internal_bounds()
            yhat = np.zeros(self.nobs)
            xhat = np.zeros((self.nobs, self._k_states_internal))

            # add 'approx_grad' to solver kwargs
            kwargs["approx_grad"] = True
            kwargs["bounds"] = bounds

            with self.use_internal_loglike():
                mlefit = super().fit(
                    start_params,
                    fargs=(yhat, xhat),
                    method="lbfgs",
                    maxiter=maxiter,
                    full_output=full_output,
                    disp=disp,
                    callback=callback,
                    skip_hessian=True,
                    **kwargs,
                )
            # convert params back
            final_params = self._model_params(mlefit.params)

        if return_params:
            return final_params
        else:

            result = ETSResults(self, final_params)
            result.mlefit = mlefit
            result.mle_retvals = mlefit.mle_retvals
            result.mle_settings = mlefit.mle_settings
            return result

    def _loglike_internal(self, params, yhat, xhat):
        """
        Log-likelihood function to be called from fit to avoid reallocation of
        memory.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta^*, gamma^*, phi, l[-1],
            b[-1], s[-1], ..., s[-m]). This must be in the format of internal
            parameters.
        yhat : np.ndarray
            Array of size (n,) where fitted values will be written to.
        xhat : np.ndarray
            Array of size (n, _k_states_internal) where fitted states will be
            written to.
        """
        self._smoothing_func(params, self.endog, yhat, xhat)
        res = self._residuals(yhat)
        logL = -self.nobs / 2 * (np.log(2 * np.pi * np.mean(res ** 2)) + 1)
        if self.error == "mul":
            logL -= np.sum(np.log(yhat))
            if np.isnan(logL):
                logL = np.inf
        return logL

    @contextlib.contextmanager
    def use_internal_loglike(self):
        external_loglike = self.loglike
        self.loglike = self._loglike_internal
        try:
            yield
        finally:
            self.loglike = external_loglike

    def loglike(self, params):
        r"""
        Log-likelihood of model.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta^*, gamma^*, phi, l[-1],
            b[-1], s[-1], ..., s[-m])

        Notes
        -----
        The log-likelihood of a exponential smoothing model is [1]_:

        .. math::

           l(\theta, x_0|y) = - \frac{n}{2}(\log(2\pi s^2) + 1)
                              - \sum\limits_{t=1}^n \log(k_t)

        with

        .. math::

           s^2 = \frac{1}{n}\sum\limits_{t=1}^n \frac{\hat{y}_t - y_t}{k_t}

        where :math:`k_t = 1` for the additive error model and :math:`k_t =
        y_t` for the multiplicative error model.

        References
        ----------
        .. [1] J. K. Ord, A. B. Koehler R. D. and Snyder (1997). Estimation and
           Prediction for a Class of Dynamic Nonlinear Statistical Models.
           *Journal of the American Statistical Association*, 92(440),
           1621-1629
        """
        if len(params) != self._k_params_internal:
            params = self._internal_params(params)
        yhat = np.zeros(self.nobs)
        xhat = np.zeros((self.nobs, self._k_states_internal))
        return self._loglike_internal(np.asarray(params), yhat, xhat)

    def _residuals(self, yhat):
        """Calculates residuals of a prediction"""
        if self.error == "mul":
            return (self.endog - yhat) / yhat
        else:
            return self.endog - yhat

    def _smooth(self, params):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters

        Returns
        -------
        yhat : pd.Series or np.ndarray
            Predicted values from exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.Series``, else a ``np.ndarray``.
        xhat : pd.DataFrame or np.ndarray
            Internal states of exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.DataFrame``, else a ``np.ndarray``.
        """
        internal_params = self._internal_params(params)
        yhat = np.zeros(self.nobs)
        xhat = np.zeros((self.nobs, self._k_states_internal))
        self._smoothing_func(internal_params, self.endog, yhat, xhat)

        # remove states that are only internal
        states = self._get_states(xhat)

        if self.use_pandas:
            _, _, _, index = self._get_prediction_index(0, self.nobs - 1)
            yhat = pd.Series(yhat, index=index)
            statenames = ["level"]
            if self.has_trend:
                statenames += ["trend"]
            if self.has_seasonal:
                statenames += ["seasonal"]
            states = pd.DataFrame(states, index=index, columns=statenames)
        return yhat, states

    @property
    def _seasonal_index(self):
        return 1 + int(self.has_trend)

    def _get_states(self, xhat):
        states = np.empty((self.nobs, self._k_states))
        state_names = ["level"]
        states[:, 0] = xhat[:, 0]
        idx = 1
        if self.has_trend:
            state_names.append("slope")
            states[:, 1] = xhat[:, 1]
        if self.has_seasonal:
            state_names.append("season")
            states[:, self._seasonal_index] = xhat[:, 2]
            idx += 1
        return states

    def _get_internal_states(self, states, params):
        """
        Converts a state matrix/dataframe to the (nobs, 2+m) matrix used
        internally
        """
        internal_params = self._internal_params(params)
        if isinstance(states, (pd.Series, pd.DataFrame)):
            states = states.values
        internal_states = np.zeros((self.nobs, 2 + self.seasonal_periods))
        internal_states[:, 0] = states[:, 0]
        if self.has_trend:
            internal_states[:, 1] = states[:, 1]
        if self.has_seasonal:
            for j in range(self.seasonal_periods):
                internal_states[j:, 2 + j] = states[
                    0:self.nobs - j, self._seasonal_index
                ]
                internal_states[0:j, 2 + j] = internal_params[6:6 + j][::-1]
        return internal_states

    @property
    def _res_classes(self):
        return {"fit": (ETSResults, ETSResultsWrapper)}

    def _wrap_results(
        self,
        params,
        result,
        return_raw,
        cov_type=None,
        cov_kwds=None,
        results_class=None,
        wrapper_class=None,
    ):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs["cov_type"] = cov_type
            if cov_kwds is not None:
                result_kwargs["cov_kwds"] = cov_kwds

            if results_class is None:
                results_class = self._res_classes["fit"][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes["fit"][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result


class ETSResults(base.StateSpaceMLEResults):
    def __init__(self, model, params):
        super().__init__(model, params)
        yhat, xhat = self.model._smooth(params)
        self._llf = self.model.loglike(params)
        self._residuals = self.model._residuals(yhat)
        self._fittedvalues = yhat

        # get model definition
        self.error = self.model.error
        self.trend = self.model.trend
        self.seasonal = self.model.seasonal
        self.damped_trend = self.model.damped_trend
        self.has_trend = self.model.has_trend
        self.has_seasonal = self.model.has_seasonal
        self.seasonal_periods = self.model.seasonal_periods
        self.initialization_method = self.model.initialization_method

        # get fitted states and parameters
        internal_params = self.model._internal_params(params)
        self.states = xhat
        if self.model.use_pandas:
            states = self.states.iloc
        else:
            states = self.states

        self.level = states[:, 0]
        self.initial_level = internal_params[4]
        self.alpha = self.params[0]

        if self.has_trend:
            self.slope = states[:, 1]
            self.initial_trend = internal_params[5]
            self.beta = self.params[1] * self.params[0]
        if self.has_seasonal:
            self.season = states[:, self.model._seasonal_index]
            self.initial_seasonal = internal_params[6:]
            self.gamma = self.params[self.model._seasonal_index] * (
                1 - self.alpha
            )
        if self.damped_trend:
            self.phi = internal_params[3]

    @cache_readonly
    def nobs_effective(self):
        return self.nobs

    @cache_readonly
    def df_model(self):
        return self.model.k_params

    @cache_readonly
    def fittedvalues(self):
        return self._fittedvalues

    @cache_readonly
    def resid(self):
        return self._residuals

    @cache_readonly
    def llf(self):
        """
        log-likelihood function evaluated at the fitted params
        """
        return self._llf

    def _get_prediction_params(self, start_idx):
        """
        Returns internal parameter representation of smoothing parameters and
        "initial" states for prediction/simulation, that is the states just
        before the first prediction/simulation step.
        """
        internal_params = self.model._internal_params(self.params)
        if start_idx == 0:
            return internal_params
        else:
            internal_states = self.model._get_internal_states(
                self.states, self.params
            )
            start_state = np.empty(6 + self.seasonal_periods)
            start_state[0:4] = internal_params[0:4]
            start_state[4:] = internal_states[start_idx - 1, :]
            return start_state

    def simulate(
        self,
        nsimulations,
        anchor=None,
        repetitions=1,
        random_errors=None,
        random_state=None,
    ):
        r"""
        Random simulations using the state space formulation.

        Parameters
        ----------
        nsimulations : int
            The number of simulation steps.
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        random_errors : optional
            Specifies how the random errors should be obtained. Can be one of
            the following:

            * ``None``: Random normally distributed values with variance
              estimated from the fit errors drawn from numpy's standard
              RNG (can be seeded with the `random_state` argument). This is the
              default option.
            * A distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm``: Fits the distribution function to the fit
              errors and draws from the fitted distribution.
              Note the difference between ``scipy.stats.norm`` and
              ``scipy.stats.norm()``, the latter one is a frozen distribution
              function.
            * A frozen distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm(scale=2)``: Draws from the frozen distribution
              function.
            * A ``np.ndarray`` with shape (`nsimulations`, `repetitions`): Uses
              the given values as random errors.
            * ``"bootstrap"``: Samples the random errors from the fit errors.

        random_state : int or np.random.RandomState, optional
            A seed for the random number generator or a
            ``np.random.RandomState`` object. Only used if `random_errors` is
            ``None``. Default is ``None``.

        Returns
        -------
        sim : pd.Series, pd.DataFrame or np.ndarray
            An ``np.ndarray``, ``pd.Series``, or ``pd.DataFrame`` of simulated
            values.
            If the original data was a ``pd.Series`` or ``pd.DataFrame``, `sim`
            will be a ``pd.Series`` if `repetitions` is 1, and a
            ``pd.DataFrame`` of shape (`nsimulations`, `repetitions`) else.
            Otherwise, if `repetitions` is 1, a ``np.ndarray`` of shape
            (`nsimulations`,) is returned, and if `repetitions` is not 1 a
            ``np.ndarray`` of shape (`nsimulations`, `repetitions`) is
            returned.
        """

        r"""
        Implementation notes
        --------------------
        The simulation is based on the state space model of the Holt-Winter's
        methods. The state space model assumes that the true value at time
        :math:`t` is randomly distributed around the prediction value.
        If using the additive error model, this means:

        .. math::

            y_t &= \hat{y}_{t|t-1} + e_t\\
            e_t &\sim \mathcal{N}(0, \sigma^2)

        Using the multiplicative error model:

        .. math::

            y_t &= \hat{y}_{t|t-1} \cdot (1 + e_t)\\
            e_t &\sim \mathcal{N}(0, \sigma^2)

        Inserting these equations into the smoothing equation formulation leads
        to the state space equations. The notation used here follows
        [1]_.

        Additionally,

        .. math::

           B_t = b_{t-1} \circ_d \phi\\
           L_t = l_{t-1} \circ_b B_t\\
           S_t = s_{t-m}\\
           Y_t = L_t \circ_s S_t,

        where :math:`\circ_d` is the operation linking trend and damping
        parameter (multiplication if the trend is additive, power if the trend
        is multiplicative), :math:`\circ_b` is the operation linking level and
        trend (addition if the trend is additive, multiplication if the trend
        is multiplicative), and :math:'\circ_s` is the operation linking
        seasonality to the rest.

        The state space equations can then be formulated as

        .. math::

           y_t = Y_t + \eta \cdot e_t\\
           l_t = L_t + \alpha \cdot (M_e \cdot L_t + \kappa_l) \cdot e_t\\
           b_t = B_t + \alpha\beta^* \cdot (M_e \cdot B_t+\kappa_b) \cdot e_t\\
           s_t = S_t + (1-\alpha)\gamma^* \cdot (M_e \cdot S_t + \kappa_s)
                       \cdot e_t\\

        with

        .. math::

           \eta &= \begin{cases}
                       Y_t\quad\text{if error is multiplicative}\\
                       1\quad\text{else}
                   \end{cases}\\
           M_e &= \begin{cases}
                       1\quad\text{if error is multiplicative}\\
                       0\quad\text{else}
                   \end{cases}\\

        and, when using the additve error model,

        .. math::

           \kappa_l &= \begin{cases}
                       \frac{1}{S_t}\quad
                       \text{if seasonality is multiplicative}\\
                       1\quad\text{else}
                   \end{cases}\\
           \kappa_b &= \begin{cases}
                       \frac{\kappa_l}{l_{t-1}}\quad
                       \text{if trend is multiplicative}\\
                       \kappa_l\quad\text{else}
                   \end{cases}\\
           \kappa_s &= \begin{cases}
                       \frac{1}{L_t}\quad
                       \text{if seasonality is multiplicative}\\
                       1\quad\text{else}
                   \end{cases}

        When using the multiplicative error model

        .. math::

           \kappa_l &= \begin{cases}
                       0\quad
                       \text{if seasonality is multiplicative}\\
                       S_t\quad\text{else}
                   \end{cases}\\
           \kappa_b &= \begin{cases}
                       \frac{\kappa_l}{l_{t-1}}\quad
                       \text{if trend is multiplicative}\\
                       \kappa_l + l_{t-1}\quad\text{else}
                   \end{cases}\\
           \kappa_s &= \begin{cases}
                       0\quad\text{if seasonality is multiplicative}\\
                       L_t\quad\text{else}
                   \end{cases}

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) *Forecasting:
           principles and practice*, 2nd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp2. Accessed on February 28th 2020.
        """

        # Get the starting location
        start_idx = self._get_prediction_start_index(anchor)

        # set initial values and obtain parameters
        start_params = self._get_prediction_params(start_idx)
        x = np.zeros((nsimulations, self.model._k_states_internal))
        (
            alpha,
            beta_star,
            gamma_star,
            phi,
            m,
            _,
        ) = smooth._initialize_ets_smooth(start_params, x)
        beta = alpha * beta_star
        gamma = (1 - alpha) * gamma_star
        # make x a 3 dimensional matrix: first dimension is nsimulations
        # (number of steps), next is number of states, innermost is repetitions
        nstates = x.shape[1]
        x = np.tile(np.reshape(x, (nsimulations, nstates, 1)), repetitions)
        y = np.empty((nsimulations, repetitions))

        # get random error eps
        sigma = np.sqrt(np.sum(self.resid ** 2) / self.df_resid)
        if isinstance(random_errors, np.ndarray):
            if random_errors.shape != (nsimulations, repetitions):
                raise ValueError(
                    "If random is an ndarray, it must have shape "
                    "(nsimulations, repetitions)!"
                )
            eps = random_errors
        elif random_errors == "bootstrap":
            eps = np.random.choice(
                self.resid, size=(nsimulations, repetitions), replace=True
            )
        elif random_errors is None:
            if random_state is None:
                eps = np.random.randn(nsimulations, repetitions) * sigma
            elif isinstance(random_state, int):
                rng = np.random.RandomState(random_state)
                eps = rng.randn(nsimulations, repetitions) * sigma
            elif isinstance(random_state, np.random.RandomState):
                eps = random_state.randn(nsimulations, repetitions) * sigma
            else:
                raise ValueError(
                    "Argument random_state must be None, an integer, "
                    "or an instance of np.random.RandomState"
                )
        elif isinstance(random_errors, (rv_continuous, rv_discrete)):
            params = random_errors.fit(self.resid)
            eps = random_errors.rvs(*params, size=(nsimulations, repetitions))
        elif isinstance(random_errors, _distn_infrastructure.rv_frozen):
            eps = random_errors.rvs(size=(nsimulations, repetitions))
        else:
            raise ValueError("Argument random_errors has unexpected value!")

        # get model settings
        mul_seasonal = self.seasonal == "mul"
        mul_trend = self.trend == "mul"
        mul_error = self.error == "mul"

        # define trend, damping and seasonality operations
        if mul_trend:
            op_b = np.multiply
            op_d = np.power
        else:
            op_b = np.add
            op_d = np.multiply
        if mul_seasonal:
            op_s = np.multiply
        else:
            op_s = np.add

        # x translation:
        # - x[t, 0, :] is level[t]
        # - x[t, 1, :] is trend[t]
        # - x[t, 2, :] is seasonal[t]
        # - x[t, 3, :] is seasonal[t-1]
        # - x[t, 2+j, :] is seasonal[t-j]
        # - similarly: x[t-1, 2+m-1, :] is seasonal[t-m]
        for t in range(nsimulations):
            B = op_d(x[t - 1, 1, :], phi)
            L = op_b(x[t - 1, 0, :], B)
            S = x[t - 1, 2 + m - 1, :]
            Y = op_s(L, S)
            if self.error == "add":
                eta = 1
                kappa_l = 1 / S if mul_seasonal else 1
                kappa_b = kappa_l / x[t - 1, 0, :] if mul_trend else kappa_l
                kappa_s = 1 / L if mul_seasonal else 1
            else:
                eta = Y
                kappa_l = 0 if mul_seasonal else S
                kappa_b = (
                    kappa_l / x[t - 1, 0, :]
                    if mul_trend
                    else kappa_l + x[t - 1, 0, :]
                )
                kappa_s = 0 if mul_seasonal else L

            y[t, :] = Y + eta * eps[t, :]
            x[t, 0, :] = L + alpha * (mul_error * L + kappa_l) * eps[t, :]
            x[t, 1, :] = B + beta * (mul_error * B + kappa_b) * eps[t, :]
            x[t, 2, :] = S + gamma * (mul_error * S + kappa_s) * eps[t, :]
            # update seasonals by shifting previous seasonal right
            x[t, 3:, :] = x[t - 1, 2:-1, :]

        # Wrap data / squeeze where appropriate
        if repetitions > 1:
            names = ["simulation.%d" % num for num in repetitions]
        else:
            names = "simulation"
        return self.model._wrap_data(
            y, start_idx, start_idx + nsimulations - 1, names=names
        )

    def forecast(self, steps=1, anchor=None):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        anchor : int, str, or datetime, optional
            First period for forecasting. The forecast will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'end'.

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        if anchor is None:
            anchor = "end"
        # forecast is the same as simulation without errors
        return self.simulate(
            steps, anchor=anchor, random_errors=np.zeros((steps, 1))
        )

    def predict(self, start=None, end=None, dynamic=False, index=None):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.

        Returns
        -------
        forecast : array_like or pd.Series.
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict,) array. If original data was a pd.Series
            or DataFrame, a pd.Series is returned.
        """
        if start is None:
            start = 0

        # Handle start, end, dynamic
        start, end, out_of_sample, _ = self.model._get_prediction_index(
            start, end, index
        )
        y = np.empty(start, end + out_of_sample)

        # Handle `dynamic`
        if isinstance(dynamic, (bytes, str)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)

        start_smooth = start
        end_smooth = min(dynamic, end + 1)
        nsmooth = end_smooth - start_smooth
        start_forecast = end_smooth
        end_forecast = end + out_of_sample + 1
        nforecast = end_forecast - start_forecast

        # In sample nondynamic prediction: smoothing
        y[0:nsmooth] = self.fittedvalues[start_smooth:end_smooth]

        # Out of sample/dynamic prediction: forecast
        y[nsmooth:] = self.forecast(steps=nforecast, anchor=start_forecast)

        # Wrap data / squeeze where appropriate
        return self.model._wrap_data(y, start, end_forecast - 1)

    def summary(self):
        ...  # TODO


class ETSResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        "fittedvalues": "rows",
        "level": "rows",
        "resid": "rows",
        "season": "rows",
        "slope": "rows",
    }
    _wrap_attrs = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {"predict": "dates", "forecast": "dates"}
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(ETSResultsWrapper, ETSResults)
