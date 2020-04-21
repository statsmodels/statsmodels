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

    \hat{y}_{t|t-1} &= (l_{t-1} \circ_b (b_{t-1} \circ_d \phi)) \circ_s s_{t-m}\\
    l_{t} &= \alpha (y_{t} \ominus_s s_{t-m})
             + (1 - \alpha) (l_{t-1} \circ_b (b_{t-1} \circ_d \phi))\\
    b_{t} &= \beta^* (l_{t} \ominus_b l_{t-1}) + (1 - \beta^*) b_{t-1}\\
    s_{t} &= \gamma (y_t \ominus_s (l_{t-1} \circ_b (b_{t-1} \circ_d \phi)))
             + (1 - \gamma) s_{t-m}

The notation here follows [1]_; :math:`l_t` denotes the level at time
:math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m` is the
number of seasonal periods, and :math:`\phi` a trend damping factor.
The parameters :math:`\alpha, \beta^*, \gamma` are the smoothing parameters,
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
   s_t &= S_t + \gamma \cdot (M_e \cdot S_t + \kappa_s) \cdot e_t\\

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

When fitting an ETS model, the parameters :math:`\alpha, \beta^*`, \gamma,
\phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \ldots, s_{-m}` are
selected as maximizers of log likelihood.

References
----------
.. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
   principles and practice*, 3rd edition, OTexts: Melbourne,
   Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
"""

import itertools
import contextlib

import numpy as np

from statsmodels.tsa.base.tsa_model import TimeSeriesModel, TimeSeriesModelResults
from statsmodels.tools.validation import (array_like, bool_like, float_like,
                                          string_like, int_like)
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
from statsmodels.tsa.exponential_smoothing import initialization as es_init

"""
Implementation details:

* The `smoothing_trend` parameter corresponds to \beta^*, not to \beta
* The smoothing equations are implemented only for models having all components
  (trend, dampening, seasonality). When using other models, the respective
  parameters (smoothing and initial parameters) are set to values that lead to
  the reduced model (often zero).
  The internal model is needed for smoothing (called from fit and loglike),
  forecasts, and simulations.
* Somewhat related to above: There are 3 sets of parameters: free params, model
  params, and internal params.
  - free params are what is passed by a user into the fit method
  - model params are all parameters necessary for a model, and are for example
    passed as argument to the likelihood function when
    ``internal_params=False`` is set.
  - internal params are what is used internally in the smoothing equations
"""


class ETS(TimeSeriesModel):

    def __init__(self, endog, error=None, trend=None, damped_trend=False,
                 seasonal=None, seasonal_periods=None,
                 initialization_method='estimated', initial_level=None,
                 initial_trend=None, initial_seasonal=None, bounds=None,
                 dates=None, freq=None, missing='none'):

        super().__init__(endog, None, dates, freq, missing=missing)
        self._y = array_like(endog, 'endog', contiguous=True, order='C')
        self.nobs = len(self._y)

        # MODEL DEFINITION
        # ================
        options = ("add", "mul", "additive", "multiplicative")
        # take first three letters of option -> either "add" or "mul"
        self.error = string_like(error, 'error', options=options)[:3]
        self.trend = string_like(
            trend, 'trend', options=options, optional=True
        )
        if self.trend is not None:
            self.trend = self.trend[:3]
        self.damped_trend = bool_like(damped_trend, 'damped_trend')
        self.seasonal = string_like(
            seasonal, 'seasonal', options=options, optional=True
        )
        if self.seasonal is not None:
            self.seasonal = self.seasonal[:3]

        self.has_trend = self.trend is not None
        self.has_seasonal = self.seasonal is not None

        if self.has_seasonal:
            self.seasonal_periods = int_like(seasonal_periods,
                                             'seasonal_periods', optional=True)
            if seasonal_periods is None:
                self.seasonal_periods = freq_to_period(self._index_freq)
            if self.seasonal_periods <= 1:
                raise ValueError('seasonal_periods must be larger than 1.')
        else:
            # in case the model has no seasonal component, we internally handle
            # this as if it had an additive seasonal component with
            # seasonal_periods=1, but restrict the smoothing parameter to 0 and
            # set the initial season to 0.
            self.seasonal_periods = 1

        # reject invalid models
        if np.any(self._y <= 0) and (
            self.error == "mul"
            or self.trend == "mul"
            or self.seasonal == "mul"
        ):
            raise ValueError(
                "endog must be strictly positive when using"
                "multiplicative error, trend or seasonal components."
            )
        if self.damped and not self.has_trend:
            raise ValueError('Can only dampen the trend component')


        # INITIALIZATION METHOD
        # =====================
        self.initialization_method = string_like(
            initialization_method, 'initialization_method',
            options=('estimated', 'known')
        )
        if self.initialization_method == 'known':
            if initial_level is None:
                raise ValueError(
                    '`initial_level` argument must be provided'
                    ' when initialization method is set to "known".'
                )
            if self.has_trend and initial_trend is None:
                raise ValueError(
                    '`initial_trend` argument must be provided'
                    ' for models with a trend component when'
                    ' initialization method is set to "known".'
                )
            if self.has_seaonal and initial_seasonal is None:
                raise ValueError(
                    '`initial_seasonal` argument must be provided'
                    ' for models with a seasonal component when'
                    ' initialization method is set to "known".'
                )

        # BOUNDS
        # ======
        if self.bounds is None:
            ... # TODO


        # PARAMETER HANDLING
        # ==================
        self._internal_params_index = {
            zip(self._internal_param_names,
                np.arange(self._k_params_internal))
        }
        self._params_index = {
            zip(self._param_names,
                np.arange(self.k_params))
        }
        self._has_fixed_params = False
        self._fixed_params = {}

    @property
    def param_names(self):
        param_names = ['smoothing_level']
        if self.has_trend:
            param_names += ['smoothing_trend']
        if self.has_seasonal:
            param_names += ['smoothing_seasonal']
        if self.damped_trend:
            param_names += ['damping_trend']

        # Initialization
        if self.initialization_method == 'estimated':
            param_names += ['initial_level']
            if self.has_trend:
                param_names += ['initial_trend']
            if self.has_seasonal:
                param_names += [
                    f'initial_seasonal.{i}'
                    for i in range(self.seasonal_periods)
                ]
        return param_names

    def _validate_params(self, param_names):
        for param_name in param_names:
            if param_name not in self.param_names:
                raise ValueError('Invalid parameter name passed: "%s".'
                                 % param_name)

    @property
    def _internal_param_names(self):
        param_names = [
            'smoothing_level',
            'smoothing_trend',
            'smoothing_seasonal',
            'damping_trend',
            'initial_level',
            'initial_trend',
        ]
        param_names += [
            f'initial_seasonal.{i}' for i in range(self.seasonal_periods)
        ]
        return param_names

    @property
    def k_params(self):
        k_params = (
            1 + int(self.has_trend) + int(self.has_seasonal)
            + int(self.damped_trend))
        if self.initialization_method == 'estimated':
            k_params += (
                1 + int(self.has_trend) +
                int(self.has_seasonal) * (self._seasonal_periods))
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
            internal_idx = self._internal_params_index(name)
            internal[internal_idx] = params[i]
        if not self.damped_trend:
            internal[3] = 1  # phi is 4th parameter
        return internal

    def _model_params(self, internal):
        """
        Converts internal parameters to model parameters
        """
        params = np.empty(self.k_params)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index(name)
            params[i] = internal[internal_idx]
        return params

    @contextlib.contextmanager
    def fix_params(self, params):
        """
        Fix parameters to specific values (context manager)

        Parameters
        ----------
        params : dict
            Dictionary describing the fixed parameter values, of the form
            `param_name: fixed_value`. See the `param_names` property for valid
            parameter names.

        Examples
        --------
        >>> mod = sm.tsa.ETS(endog, error='add', trend='add', seasonal='mul')
        >>> with mod.fix_params({'smoothing_level': 0.5}):
                res = mod.fit()
        """
        # Cache the current fixed parameters
        cache_fixed_params = self._fixed_params.copy()
        cache_has_fixed_params = self._has_fixed_params
        cache_fixed_params_index = self._fixed_params_index
        cache_free_params_index = self._free_params_index

        # Validate parameter names and values
        self._validate_params(set(params.keys()))

        # Set the new fixed parameters, keeping the order as given by
        # param_names
        self._fixed_params.update(params)
        self._fixed_params = {[
            (name, self._fixed_params[name]) for name in self.param_names
            if name in self._fixed_params
        ]}

        # Update associated values
        self._has_fixed_params = True
        self._fixed_params_index = [self._params_index[key]
                                    for key in self._fixed_params.keys()]
        self._free_params_index = list(
            set(np.arange(self._k_params))
            .difference(self._fixed_params_index)
        )

        try:
            yield
        finally:
            # Reset the fixed parameters
            self._has_fixed_params = cache_has_fixed_params
            self._fixed_params = cache_fixed_params
            self._fixed_params_index = cache_fixed_params_index
            self._free_params_index = cache_free_params_index

    def _set_fixed_params(self, params):
        for i, name in enumerate(self._fixed_params):
            idx = self._fixed_params_index[i]
            params[idx] = self._fixed_params[name]

    @property
    def start_params(self):
        # Make sure starting parameters aren't beyond or right on the bounds
        bounds = [(x[0] + 1e-3, x[1] - 1e-3) for x in self.bounds]

        # See Hyndman p.24
        start_params = [np.clip(0.1, *bounds[0])]
        if self.trend:
            start_params += [np.clip(0.01, *bounds[1])]
        if self.seasonal:
            start_params += [np.clip(0.01, *bounds[2])]
        if self.damped_trend:
            start_params += [np.clip(0.98, *bounds[3])]

        # Initialization
        if self.initialization_method == 'estimated':
            initial_level, initial_trend, initial_seasonal = (
                es_init._initialization_simple(
                    self.endog[:, 0],
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods))
            start_params += [initial_level]
            if self.has_trend:
                start_params += [initial_trend]
            if self.has_seasonal:
                start_params += initial_seasonal.tolist()[:-1]

        return np.array(start_params)

    def default_param_bounds(self):
        """
        Default lower and upper bounds for model parameters
        """
        # traditional bounds: alpha, beta*, gamma in (0, 1), phi in [0.8, 0.98]
        n = 1 + int(self.has_trend) + int(self.has_seaonal)
        bounds = [(1e-4, 1-1e-4)] * n
        if self.has_trend:
            bounds += [(0.8, 0.98)]
        if self.initialization_method == 'estimated':
            n = (
                1 + int(self.has_trend)
                + int(self.has_seasonal) * self.seasonal_periods
            )
            bounds += [(None, None)]
        return bounds

    def _internal_bounds(self):
        """
        Returns bounds for internal parameters
        """
        bounds = []
        for name in self._internal_param_names:
            if name in self.param_names:
                bounds.append(self.bounds[self._params_index[name]])
            elif name == 'damping_trend':
                # if damping_trend is not in param_names, it is set to 1
                bounds.append((1, 1))
            else:
                # all other internal-only parameters are 0
                bounds.append((0, 0))
        # set fixed parameters bounds
        for i, name in enumerate(self._fixed_params):
            idx = self._fixed_params_index[i]
            val = self._fixed_params[name]
            bounds[idx] = (val, val)


    def fit(self, start_params=None, maxiter=None, full_output=True,
            disp=5, callback=None, return_params=False, **kwargs):
        r"""
        Fit an ETS model by maximizing log-likelihood.

        Log-likelihood is a function of the model parameters :math:`\alpha,
        \beta, \gamma, \phi` (depending on the chosen model), and, if
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
            * `smoothing_season` (:math:`\gamma`)
            * `damping_slope` (:math:`\phi`)

            If ``initialization_method`` was set to ``'estimated'`` (the
            default), additionally, the parameters

            * `initial_level` (:math:`l_{-1}`)
            * `initial_trend` (:math:`l_{-1}`)
            * `initial_season.0` (:math:`s_{-1}`)
            * ...
            * `initial_season.<m-1>` (:math:`s_{-m}`)

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
            start_params = self.start_params()

        # set fixed params
        self._set_fixed_params(start_params)

        if len(self._free_params_index) == 0:
            final_params = start_params
            # mlefit = Bunch(params=start_params, mle_retvals=None,
            #                mle_settings=None)
        else:
            # transform parameters to internal parameters
            start_params = self._internal_params(start_params)
            bounds = self._internal_bounds()

            mlefit = super().fit(
                start_params, fargs={"_internal_params": True}, method=method,
                maxiter=maxiter, full_output=full_output, disp=disp,
                callback=callback, bounds=bounds, **kwargs
            )
            # convert params back
            final_params = self._model_params(mlefit.params)

        if return_params:
            return final_params
        else:

            result = ETSResults()

            return result

    def loglike(self, params, _internal_params=False):
        r"""
        Log-likelihood of model.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l0, b0, s0, ..., s[-m])

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
           *Journal of the American Statistical Association*, 92(440), 1621-1629
        """
        if not _internal_params:
            params = self._internal_params(params)
        yhat, _ = self._smooth(params)
        res = self._residuals(yhat)
        logL =  - self.nobs/2 * (np.log(2*np.pi*np.mean(res**2)) + 1)
        if self.error == 'mul':
            return logL - np.sum(np.log(yhat))
        else:
            return logL

    def _residuals(self, yhat):
        """Calculates residuals of a prediction"""
        if self.error == 'mul':
            return (yhat - self._y) / self._y
        else:
            return yhat - self._y

    def _smooth(self, params):
        """Holt-Winters smoothing based on all data"""

        # figure out which smoother to use
        if self.trend == "add" or self.trend is None:
            if self.seasonal == "add" or self.seasonal is None:
                yhat, xhat = smoothers._hw_smooth_add_add(params, self._y)
            else:
                yhat, xhat = smoothers._hw_smooth_add_mul(params, self._y)
        else:
            if self.seasonal == "add" or self.seasonal is None:
                yhat, xhat = smoothers._hw_smooth_mul_add(params, self._y)
            else:
                yhat, xhat = smoothers._hw_smooth_mul_mul(params, self._y)
        return yhat, xhat




# TODO: Results class
# - returned by ETS.fit
# - has params, fitted values, fitted states, residuals
# - loglik, aic, bic, aicc




class ETSResults(TimeSeriesModelResults):

    def __init__(self, model, params, fittedvalues, ):
        super().__init__(model, params, normalized_cov_params=None)
        self.nobs_effective = model.nobs
        self.df_model = model.k_params

    @cache_readonly
    def aic(self):
        return aic()


    def predict(self, exog=None, **kwargs):
        ... # TODO


    def summary(self):
        ... # TODO












# def _get_model_spec_string(self, model)
#     # get model specification
#     if not isinstance(model, str):
#         raise ValueError("model must be a string.")

#     # error model
#     error = model[0]
#     error_map = {"A": ["add"], "M": ["mul"], "Z": ["add", "mul"]}
#     if error not in error_map:
#         raise ValueError("Invalid model string.")
#     errors = error_map[error]

#     # trend model
#     if model[2] == "d":
#         trend = model[1:3]
#     else:
#         trend = model[1]
#     trend_map = {
#         "N": [(None, False)],
#         "A": [("add", False)],
#         "Ad": [("add", True)],
#         "M": [("mul", False)],
#         "Md": [("mul", True)],
#         "Z": list(itertools.product(["add", "mul", None], [True, False])),
#     }
#     if trend not in trend_map:
#         raise ValueError("Invalid model string.")
#     trends = trend_map[trend]

#     # seasonal model
#     season = model[-1]
#     if season != "N" and seasonal_periods is None:
#         raise ValueError(
#             "You must supply seasonal_periods when using a seasonal component."
#         )
#     season_map = {
#         "N": [None],
#         "A": ["add"],
#         "M": ["mul"],
#         "Z": ["add", "mul", None],
#     }
#     if season not in season_map:
#         raise ValueError("Invalid model string.")
#     seasons = season_map[season]
