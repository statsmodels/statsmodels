"""
Univariate structural time series models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from statsmodels.compat.collections import OrderedDict

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import ValueWarning, OutputWarning, SpecificationWarning
from .tools import (
    companion_matrix, constrain_stationary_univariate,
    unconstrain_stationary_univariate
)
import statsmodels.base.wrapper as wrap

_mask_map = {
    1: 'irregular',
    2: 'fixed intercept',
    3: 'deterministic constant',
    6: 'random walk',
    7: 'local level',
    8: 'fixed slope',
    11: 'deterministic trend',
    14: 'random walk with drift',
    15: 'local linear deterministic trend',
    31: 'local linear trend',
    27: 'smooth trend',
    26: 'random trend'
}


class UnobservedComponents(MLEModel):
    r"""
    Univariate unobserved components time series model

    These are also known as structural time series models, and decompose a
    (univariate) time series into trend, seasonal, cyclical, and irregular
    components.

    Parameters
    ----------

    level : bool or string, optional
        Whether or not to include a level component. Default is False. Can also
        be a string specification of the level / trend component; see Notes
        for available model specification strings.
    trend : bool, optional
        Whether or not to include a trend component. Default is False. If True,
        `level` must also be True.
    seasonal : int or None, optional
        The period of the seasonal component, if any. Default is None.
    cycle : bool, optional
        Whether or not to include a cycle component. Default is False.
    ar : int or None, optional
        The order of the autoregressive component. Default is None.
    exog : array_like or None, optional
        Exogenous variables.
    irregular : bool, optional
        Whether or not to include an irregular component. Default is False.
    stochastic_level : bool, optional
        Whether or not any level component is stochastic. Default is False.
    stochastic_trend : bool, optional
        Whether or not any trend component is stochastic. Default is False.
    stochastic_seasonal : bool, optional
        Whether or not any seasonal component is stochastic. Default is False.
    stochastic_cycle : bool, optional
        Whether or not any cycle component is stochastic. Default is False.
    damped_cycle : bool, optional
        Whether or not the cycle component is damped. Default is False.
    cycle_period_bounds : tuple, optional
        A tuple with lower and upper allowed bounds for the period of the
        cycle. If not provided, the following default bounds are used:
        (1) if no date / time information is provided, the frequency is
        constrained to be between zero and :math:`\pi`, so the period is
        constrained to be in [0.5, infinity].
        (2) If the date / time information is provided, the default bounds
        allow the cyclical component to be between 1.5 and 12 years; depending
        on the frequency of the endogenous variable, this will imply different
        specific bounds.

    Notes
    -----

    These models take the general form (see [1]_ Chapter 3.2 for all details)

    .. math::

        y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\mu_t` refers to the trend component, :math:`\gamma_t` refers to the
    seasonal component, :math:`c_t` refers to the cycle, and
    :math:`\varepsilon_t` is the irregular. The modeling details of these
    components are given below.

    **Trend**

    The trend component is a dynamic extension of a regression model that
    includes an intercept and linear time-trend. It can be written:

    .. math::

        \mu_t = \mu_{t-1} + \beta_{t-1} + \eta_{t-1} \\
        \beta_t = \beta_{t-1} + \zeta_{t-1}

    where the level is a generalization of the intercept term that can
    dynamically vary across time, and the trend is a generalization of the
    time-trend such that the slope can dynamically vary across time.

    Here :math:`\eta_t \sim N(0, \sigma_\eta^2)` and
    :math:`\zeta_t \sim N(0, \sigma_\zeta^2)`.

    For both elements (level and trend), we can consider models in which:

    - The element is included vs excluded (if the trend is included, there must
      also be a level included).
    - The element is deterministic vs stochastic (i.e. whether or not the
      variance on the error term is confined to be zero or not)
    
    The only additional parameters to be estimated via MLE are the variances of
    any included stochastic components.

    The level/trend components can be specified using the boolean keyword
    arguments `level`, `stochastic_level`, `trend`, etc., or all at once as a
    string argument to `level`. The following table shows the available
    model specifications:

    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Model name                       | Full string syntax                   | Abbreviated syntax | Model                                            |
    +==================================+======================================+====================+==================================================+
    | No trend                         | `'irregular'`                        | `'ntrend'`         | .. math:: y_t &= \varepsilon_t                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Fixed intercept                  | `'fixed intercept'`                  |                    | .. math:: y_t &= \mu                             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Deterministic constant           | `'deterministic constant'`           | `'dconstant'`      | .. math:: y_t &= \mu + \varepsilon_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local level                      | `'local level'`                      | `'llevel'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \eta_t                  |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random walk                      | `'random walk'`                      | `'rwalk'`          | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \eta_t                  |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Fixed slope                      | `'fixed slope'`                      |                    | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Deterministic trend              | `'deterministic trend'`              | `'dtrend'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local linear deterministic trend | `'local linear deterministic trend'` | `'lldtrend'`       | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta + \eta_t          |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random walk with drift           | `'random walk with drift'`           | `'rwdrift'`        | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta + \eta_t          |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local linear trend               | `'local linear trend'`               | `'lltrend'`        | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\ |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Smooth trend                     | `'smooth trend'`                     | `'strend'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} \\          |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random trend                     | `'random trend'`                     | `'rtrend'`         | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} \\          |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+

    Following the fitting of the model, the unobserved level and trend
    component time series are available in the results class in the
    `level` and `trend` attributes, respectively.

    **Seasonal**

    The seasonal component is modeled as:

    .. math::

        \gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \\
        \omega_t \sim N(0, \sigma_\omega^2)

    The periodicity (number of seasons) is s, and the defining character is
    that (without the error term), the seasonal components sum to zero across
    one complete cycle. The inclusion of an error term allows the seasonal
    effects to vary over time (if this is not desired, :math:`\sigma_\omega^2`
    can be set to zero using the `stochastic_seasonal=False` keyword argument).

    This component results in one parameter to be selected via maximum
    likelihood: :math:`\sigma_\omega^2`, and one parameter to be chosen, the
    number of seasons `s`.

    Following the fitting of the model, the unobserved seasonal component
    time series is available in the results class in the `seasonal`
    attribute.

    **Cycle**

    The cyclical component is intended to capture cyclical effects at time
    frames much longer than captured by the seasonal component. For example,
    in economics the cyclical term is often intended to capture the business
    cycle, and is then expected to have a period between "1.5 and 12 years"
    (see Durbin and Koopman).

    .. math::

        c_{t+1} & = \rho_c (\tilde c_t \cos \lambda_c t
                + \tilde c_t^* \sin \lambda_c) +
                \tilde \omega_t \\
        c_{t+1}^* & = \rho_c (- \tilde c_t \sin \lambda_c  t +
                \tilde c_t^* \cos \lambda_c) +
                \tilde \omega_t^* \\

    where :math:`\omega_t, \tilde \omega_t iid N(0, \sigma_{\tilde \omega}^2)`

    The parameter :math:`\lambda_c` (the frequency of the cycle) is an
    additional parameter to be estimated by MLE.

    If the cyclical effect is stochastic (`stochastic_cycle=True`), then there
    is another parameter to estimate (the variance of the error term - note
    that both of the error terms here share the same variance, but are assumed
    to have independent draws).

    If the cycle is damped (`damped_cycle=True`), then there is a third
    parameter to estimate, :math:`\rho_c`.

    In order to achieve cycles with the appropriate frequencies, bounds are
    imposed on the parameter :math:`\lambda_c` in estimation. These can be
    controlled via the keyword argument `cycle_period_bounds`, which, if
    specified, must be a tuple of bounds on the **period** `(lower, upper)`.
    The bounds on the frequency are then calculated from those bounds.

    The default bounds, if none are provided, are selected in the following
    way:

    1. If no date / time information is provided, the frequency is
       constrained to be between zero and :math:`\pi`, so the period is
       constrained to be in :math:`[0.5, \infty]`.
    2. If the date / time information is provided, the default bounds
       allow the cyclical component to be between 1.5 and 12 years; depending
       on the frequency of the endogenous variable, this will imply different
       specific bounds.

    Following the fitting of the model, the unobserved cyclical component
    time series is available in the results class in the `cycle`
    attribute.

    **Irregular**

    The irregular components are independent and identically distributed (iid):

    .. math::

        \varepsilon_t \sim N(0, \sigma_\varepsilon^2)

    **Autoregressive Irregular**

    An autoregressive component (often used as a replacement for the white
    noise irregular term) can be specified as:

    .. math::

        \varepsilon_t = \rho(L) \varepsilon_{t-1} + \epsilon_t \\
        \epsilon_t \sim N(0, \sigma_\epsilon^2)

    In this case, the AR order is specified via the `autoregressive` keyword,
    and the autoregressive coefficients are estimated.

    Following the fitting of the model, the unobserved autoregressive component
    time series is available in the results class in the `autoregressive`
    attribute.

    **Regression effects**

    Exogenous regressors can be pass to the `exog` argument. The regression
    coefficients will be estimated by maximum likelihood unless
    `mle_regression=False`, in which case the regression coefficients will be
    included in the state vector where they are essentially estimated via
    recursive OLS.

    If the regression_coefficients are included in the state vector, the
    recursive estimates are available in the results class in the
    `regression_coefficients` attribute.

    References
    ----------

    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(self, endog, level=False, trend=False, seasonal=None,
                 cycle=False, autoregressive=None, exog=None, irregular=False,
                 stochastic_level=False, stochastic_trend=False,
                 stochastic_seasonal=True, stochastic_cycle=False,
                 damped_cycle=False, cycle_period_bounds=None,
                 mle_regression=True,
                 **kwargs):

        # Model options
        self.level = level
        self.trend = trend
        self.seasonal_periods = seasonal if seasonal is not None else 0
        self.seasonal = self.seasonal_periods > 0
        self.cycle = cycle
        self.ar_order = autoregressive if autoregressive is not None else 0
        self.autoregressive = self.ar_order > 0
        self.irregular = irregular

        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_cycle = stochastic_cycle

        self.damped_cycle = damped_cycle
        self.mle_regression = mle_regression

        # Check for string trend/level specification
        self.trend_specification = None
        if isinstance(self.level, str):
            self.trend_specification = level
            self.level = False

            # Check if any of the trend/level components have been set, and
            # reset everything to False
            trend_attributes = ['irregular', 'level', 'trend',
                                'stochastic_level', 'stochastic_trend']
            for attribute in trend_attributes:
                if not getattr(self, attribute) is False:
                    warn("Value of `%s` may be overridden when the trend"
                         " component is specified using a model string."
                         % attribute, SpecificationWarning)
                    setattr(self, attribute, False)

            # Now set the correct specification
            spec = self.trend_specification
            if spec == 'irregular' or spec == 'ntrend':
                self.irregular = True
                self.trend_specification = 'irregular'
            elif spec == 'fixed intercept':
                self.level = True
            elif spec == 'deterministic constant' or spec == 'dconstant':
                self.irregular = True
                self.level = True
                self.trend_specification = 'deterministic constant'
            elif spec == 'local level' or spec == 'llevel':
                self.irregular = True
                self.level = True
                self.stochastic_level = True
                self.trend_specification = 'local level'
            elif spec == 'random walk' or spec == 'rwalk':
                self.level = True
                self.stochastic_level = True
                self.trend_specification = 'random walk'
            elif spec == 'fixed slope':
                self.level = True
                self.trend = True
            elif spec == 'deterministic trend' or spec == 'dtrend':
                self.irregular = True
                self.level = True
                self.trend = True
                self.trend_specification = 'deterministic trend'
            elif (spec == 'local linear deterministic trend' or
                    spec == 'lldtrend'):
                self.irregular = True
                self.level = True
                self.stochastic_level = True
                self.trend = True
                self.trend_specification = 'local linear deterministic trend'
            elif spec == 'random walk with drift' or spec == 'rwdrift':
                self.level = True
                self.stochastic_level = True
                self.trend = True
                self.trend_specification = 'random walk with drift'
            elif spec == 'local linear trend' or spec == 'lltrend':
                self.irregular = True
                self.level = True
                self.stochastic_level = True
                self.trend = True
                self.stochastic_trend = True
                self.trend_specification = 'local linear trend'
            elif spec == 'smooth trend' or spec == 'strend':
                self.irregular = True
                self.level = True
                self.trend = True
                self.stochastic_trend = True
                self.trend_specification = 'smooth trend'
            elif spec == 'random trend' or spec == 'rtrend':
                self.level = True
                self.trend = True
                self.stochastic_trend = True
                self.trend_specification = 'random trend'
            else:
                raise ValueError("Invalid level/trend specification: '%s'"
                                 % spec)

        # Check for a model that makes sense
        if trend and not level:
            warn("Trend component specified without level component;"
                 " deterministic level component added.", SpecificationWarning)
            self.level = True
            self.stochastic_level = False

        if not (self.irregular or
                (self.level and self.stochastic_level) or
                (self.trend and self.stochastic_trend) or
                (self.seasonal and self.stochastic_seasonal) or
                (self.cycle and self.stochastic_cycle) or
                self.autoregressive):
            warn("Specified model does not contain a stochastic element;"
                 " irregular component added.", SpecificationWarning)
            self.irregular = True

        if self.seasonal and self.seasonal_periods < 2:
            raise ValueError('Seasonal component must have a seasonal period'
                             ' of at least 2.')

        # Create a bitmask holding the level/trend specification
        self.trend_mask = (
            self.irregular * 0x01 |
            self.level * 0x02 |
            self.level * self.stochastic_level * 0x04 |
            self.trend * 0x08 |
            self.trend * self.stochastic_trend * 0x10
        )

        # Create the trend specification, if it wasn't given
        if self.trend_specification is None:
            # trend specification may be none, e.g. if the model is only
            # a stochastic cycle, etc.
            self.trend_specification = _mask_map.get(self.trend_mask, None)

        # Exogenous component
        self.k_exog = 0
        if exog is not None:
            exog_is_using_pandas = _is_using_pandas(exog, None)
            if not exog_is_using_pandas:
                exog = np.asarray(exog)

            # Make sure we have 2-dimensional array
            if exog.ndim < 2:
                if not exog_is_using_pandas:
                    exog = np.atleast_2d(exog).T
                else:
                    exog = pd.DataFrame(exog)

            self.k_exog = exog.shape[1]
        self.regression = self.k_exog > 0

        # Model parameters
        k_states = (
            self.level + self.trend +
            (self.seasonal_periods - 1) * self.seasonal +
            self.cycle * 2 +
            self.ar_order +
            (not self.mle_regression) * self.k_exog
        )
        k_posdef = (
            self.stochastic_level * self.level +
            self.stochastic_trend * self.trend +
            self.stochastic_seasonal * self.seasonal +
            self.stochastic_cycle * (self.cycle * 2) +
            self.autoregressive
        )

        # We can still estimate the model with just the irregular component,
        # just need to have one state that does nothing.
        loglikelihood_burn = kwargs.get('loglikelihood_burn',
                                        k_states - self.ar_order)
        if k_states == 0:
            if not self.irregular:
                raise ValueError('Model has no components specified.')
            k_states = 1
        if k_posdef == 0:
            k_posdef = 1

        # Setup the representation
        super(UnobservedComponents, self).__init__(
            endog, k_states, k_posdef=k_posdef, exog=exog, **kwargs
        )
        self.setup()

        # Set as time-varying model if we have exog
        if self.k_exog > 0:
            self.ssm._time_invariant = False

        # Initialize the model
        self.ssm.loglikelihood_burn = loglikelihood_burn

        # Need to reset the MLE names (since when they were first set, `setup`
        # had not been run (and could not have been at that point))
        self.data.param_names = self.param_names

        # Get bounds for the frequency of the cycle, if we know the frequency
        # of the data.
        if cycle_period_bounds is None:
            freq = self.data.freq[0] if self.data.freq is not None else ''
            if freq == 'A':
                cycle_period_bounds = (1.5, 12)
            elif freq == 'Q':
                cycle_period_bounds = (1.5*4, 12*4)
            elif freq == 'M':
                cycle_period_bounds = (1.5*12, 12*12)
            else:
                # If we have no information on data frequency, require the
                # cycle frequency to be between 0 and pi
                cycle_period_bounds = (2, np.inf)

        self.cycle_frequency_bound = (
            2*np.pi / cycle_period_bounds[1], 2*np.pi / cycle_period_bounds[0]
        )

        # update _init_keys attached by super
        self._init_keys += ['level', 'trend', 'seasonal', 'cycle',
                            'autoregressive', 'exog', 'irregular',
                            'stochastic_level', 'stochastic_trend',
                            'stochastic_seasonal', 'stochastic_cycle',
                            'damped_cycle', 'cycle_period_bounds',
                            'mle_regression'] + list(kwargs.keys())
        # TODO: I think the kwargs or not attached, need to recover from ???

    def _get_init_kwds(self):
        # Get keywords based on model attributes
        kwds = super(UnobservedComponents, self)._get_init_kwds()

        # Modifications
        kwds['seasonal'] = self.seasonal_periods
        kwds['autoregressive'] = self.ar_order

        for key, value in kwds.items():
            if value is None and hasattr(self.ssm, key):
                kwds[key] = getattr(self.ssm, key)

        return kwds

    def setup(self):
        """
        Setup the structural time series representation
        """
        # Initialize the ordered sets of parameters
        self.parameters = OrderedDict()
        self.parameters_obs_intercept = OrderedDict()
        self.parameters_obs_cov = OrderedDict()
        self.parameters_transition = OrderedDict()
        self.parameters_state_cov = OrderedDict()

        # Initialize the fixed components of the state space matrices,
        i = 0  # state offset
        j = 0  # state covariance offset

        if self.irregular:
            self.parameters_obs_cov['irregular_var'] = 1
        if self.level:
            self.ssm['design', 0, i] = 1.
            self.ssm['transition', i, i] = 1.
            if self.trend:
                self.ssm['transition', i, i+1] = 1.
            if self.stochastic_level:
                self.ssm['selection', i, j] = 1.
                self.parameters_state_cov['level_var'] = 1
                j += 1
            i += 1
        if self.trend:
            self.ssm['transition', i, i] = 1.
            if self.stochastic_trend:
                self.ssm['selection', i, j] = 1.
                self.parameters_state_cov['trend_var'] = 1
                j += 1
            i += 1
        if self.seasonal:
            n = self.seasonal_periods - 1
            self.ssm['design', 0, i] = 1.
            self.ssm['transition', i:i + n, i:i + n] = (
                companion_matrix(np.r_[1, [1] * n]).transpose()
            )
            if self.stochastic_seasonal:
                self.ssm['selection', i, j] = 1.
                self.parameters_state_cov['seasonal_var'] = 1
                j += 1
            i += n
        if self.cycle:
            self.ssm['design', 0, i] = 1.
            self.parameters_transition['cycle_freq'] = 1
            if self.damped_cycle:
                self.parameters_transition['cycle_damp'] = 1
            if self.stochastic_cycle:
                self.ssm['selection', i:i+2, j:j+2] = np.eye(2)
                self.parameters_state_cov['cycle_var'] = 1
                j += 2
            self._idx_cycle_transition = np.s_['transition', i:i+2, i:i+2]
            i += 2
        if self.autoregressive:
            self.ssm['design', 0, i] = 1.
            self.parameters_transition['ar_coeff'] = self.ar_order
            self.parameters_state_cov['ar_var'] = 1
            self.ssm['selection', i, j] = 1
            self.ssm['transition', i:i+self.ar_order, i:i+self.ar_order] = (
                companion_matrix(self.ar_order).T
            )
            self._idx_ar_transition = (
                np.s_['transition', i, i:i+self.ar_order]
            )
            j += 1
            i += self.ar_order
        if self.regression:
            if self.mle_regression:
                self.parameters_obs_intercept['reg_coeff'] = self.k_exog
            else:
                design = np.repeat(self.ssm['design', :, :, 0], self.nobs, axis=0)
                self.ssm['design'] = design.transpose()[np.newaxis, :, :]
                self.ssm['design', 0, i:i+self.k_exog, :] = self.exog.transpose()
                self.ssm['transition', i:i+self.k_exog, i:i+self.k_exog] = (
                    np.eye(self.k_exog)
                )

                i += self.k_exog

        # Update to get the actual parameter set
        self.parameters.update(self.parameters_obs_cov)
        self.parameters.update(self.parameters_state_cov)
        self.parameters.update(self.parameters_transition)  # ordered last
        self.parameters.update(self.parameters_obs_intercept)

        self.k_obs_intercept = sum(self.parameters_obs_intercept.values())
        self.k_obs_cov = sum(self.parameters_obs_cov.values())
        self.k_transition = sum(self.parameters_transition.values())
        self.k_state_cov = sum(self.parameters_state_cov.values())
        self.k_params = sum(self.parameters.values())

        # Other indices
        idx = np.diag_indices(self.ssm.k_posdef)
        self._idx_state_cov = ('state_cov', idx[0], idx[1])

    def initialize_state(self):
        # Initialize the AR component as stationary, the rest as approximately
        # diffuse
        initial_state = np.zeros(self.k_states)
        initial_state_cov = (
            np.eye(self.k_states, dtype=self.ssm.transition.dtype) *
            self.ssm.initial_variance
        )

        if self.autoregressive:

            start = (
                self.level + self.trend +
                (self.seasonal_periods - 1) * self.seasonal +
                self.cycle * 2
            )
            end = start + self.ar_order
            selection_stationary = self.ssm.selection[start:end, :, 0]
            selected_state_cov_stationary = np.dot(
                np.dot(selection_stationary, self.ssm.state_cov[:, :, 0]),
                selection_stationary.T
            )
            try:
                initial_state_cov_stationary = solve_discrete_lyapunov(
                    self.ssm.transition[start:end, start:end, 0],
                    selected_state_cov_stationary
                )
            except:
                initial_state_cov_stationary = solve_discrete_lyapunov(
                    self.ssm.transition[start:end, start:end, 0],
                    selected_state_cov_stationary,
                    method='direct'
                )

            initial_state_cov[start:end, start:end] = (
                initial_state_cov_stationary
            )

        self.ssm.initialize_known(initial_state, initial_state_cov)

    def filter(self, params, **kwargs):
        kwargs.setdefault('results_class', UnobservedComponentsResults)
        kwargs.setdefault('results_wrapper_class',
                          UnobservedComponentsResultsWrapper)
        return super(UnobservedComponents, self).filter(params, **kwargs)

    def smooth(self, params, **kwargs):
        kwargs.setdefault('results_class', UnobservedComponentsResults)
        kwargs.setdefault('results_wrapper_class',
                          UnobservedComponentsResultsWrapper)
        return super(UnobservedComponents, self).smooth(params, **kwargs)

    @property
    def start_params(self):
        if not hasattr(self, 'parameters'):
            return []

        # Eliminate missing data to estimate starting parameters
        endog = self.endog
        exog = self.exog
        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]

        # Level / trend variances
        # (Use the HP filter to get initial estimates of variances)
        _start_params = {}
        if self.level:
            resid, trend1 = hpfilter(endog)

            if self.stochastic_trend:
                cycle2, trend2 = hpfilter(trend1)
                _start_params['trend_var'] = np.std(trend2)**2
                if self.stochastic_level:
                    _start_params['level_var'] = np.std(cycle2)**2
            elif self.stochastic_level:
                _start_params['level_var'] = np.std(trend1)**2
        else:
            resid = self.ssm.endog[0]

        # Regression
        if self.regression and self.mle_regression:
            _start_params['reg_coeff'] = (
                np.linalg.pinv(exog).dot(resid).tolist()
            )
            resid = np.squeeze(
                resid - np.dot(exog, _start_params['reg_coeff'])
            )

        # Autoregressive
        if self.autoregressive:
            Y = resid[self.ar_order:]
            X = lagmat(resid, self.ar_order, trim='both')
            _start_params['ar_coeff'] = np.linalg.pinv(X).dot(Y).tolist()
            resid = np.squeeze(Y - np.dot(X, _start_params['ar_coeff']))
            _start_params['ar_var'] = np.var(resid)

        # The variance of the residual term can be used for all variances,
        # just to get something in the right order of magnitude.
        var_resid = np.var(resid)

        # Seasonal
        if self.stochastic_seasonal:
            _start_params['seasonal_var'] = var_resid

        # Cyclical
        if self.cycle:
            _start_params['cycle_var'] = var_resid
            # Clip this to make sure it is postive and strictly stationary
            # (i.e. don't want negative or 1)
            _start_params['cycle_damp'] = np.clip(
                np.linalg.pinv(resid[:-1, None]).dot(resid[1:])[0], 0, 0.99
            )

            # Set initial period estimate to 3 year, if we know the frequency
            # of the data observations
            freq = self.data.freq[0] if self.data.freq is not None else ''
            if freq == 'A':
                _start_params['cycle_freq'] = 2 * np.pi / 3
            elif freq == 'Q':
                _start_params['cycle_freq'] = 2 * np.pi / 12
            elif freq == 'M':
                _start_params['cycle_freq'] = 2 * np.pi / 36
            else:
                if not np.any(np.isinf(self.cycle_frequency_bound)):
                    _start_params['cycle_freq'] = (
                        np.mean(self.cycle_frequency_bound))
                elif np.isinf(self.cycle_frequency_bound[1]):
                    _start_params['cycle_freq'] = self.cycle_frequency_bound[0]
                else:
                    _start_params['cycle_freq'] = self.cycle_frequency_bound[1]

        # Irregular
        if self.irregular:
            _start_params['irregular_var'] = var_resid

        # Create the starting parameter list
        start_params = []
        for key in self.parameters.keys():
            if np.isscalar(_start_params[key]):
                start_params.append(_start_params[key])
            else:
                start_params += _start_params[key]
        return start_params

    @property
    def param_names(self):
        if not hasattr(self, 'parameters'):
            return []
        param_names = []
        for key in self.parameters.keys():
            if key == 'irregular_var':
                param_names.append('sigma2.irregular')
            elif key == 'level_var':
                param_names.append('sigma2.level')
            elif key == 'trend_var':
                param_names.append('sigma2.trend')
            elif key == 'seasonal_var':
                param_names.append('sigma2.seasonal')
            elif key == 'cycle_var':
                param_names.append('sigma2.cycle')
            elif key == 'cycle_freq':
                param_names.append('frequency.cycle')
            elif key == 'cycle_damp':
                param_names.append('damping.cycle')
            elif key == 'ar_coeff':
                for i in range(self.ar_order):
                    param_names.append('ar.L%d' % (i+1))
            elif key == 'ar_var':
                param_names.append('sigma2.ar')
            elif key == 'reg_coeff':
                param_names += [
                    'beta.%s' % self.exog_names[i]
                    for i in range(self.k_exog)
                ]
            else:
                param_names.append(key)
        return param_names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)

        # Positive parameters: obs_cov, state_cov
        offset = self.k_obs_cov + self.k_state_cov
        constrained[:offset] = unconstrained[:offset]**2

        # Cycle parameters
        if self.cycle:
            # Cycle frequency must be between between our bounds
            low, high = self.cycle_frequency_bound
            constrained[offset] = (
                1 / (1 + np.exp(-unconstrained[offset]))
            ) * (high - low) + low
            offset += 1

            # Cycle damping (if present) must be between 0 and 1
            if self.damped_cycle:
                constrained[offset] = (
                    1 / (1 + np.exp(-unconstrained[offset]))
                )
                offset += 1

        # Autoregressive coefficients must be stationary
        if self.autoregressive:
            constrained[offset:offset + self.ar_order] = (
                constrain_stationary_univariate(
                    unconstrained[offset:offset + self.ar_order]
                )
            )
            offset += self.ar_order

        # Nothing to do with betas
        constrained[offset:offset + self.k_exog] = (
            unconstrained[offset:offset + self.k_exog]
        )

        return constrained

    def untransform_params(self, constrained):
        """
        Reverse the transformation
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, dtype=constrained.dtype)

        # Positive parameters: obs_cov, state_cov
        offset = self.k_obs_cov + self.k_state_cov
        unconstrained[:offset] = constrained[:offset]**0.5

        # Cycle parameters
        if self.cycle:
            # Cycle frequency must be between between our bounds
            low, high = self.cycle_frequency_bound
            x = (constrained[offset] - low) / (high - low)
            unconstrained[offset] = np.log(
                x / (1 - x)
            )
            offset += 1

            # Cycle damping (if present) must be between 0 and 1
            if self.damped_cycle:
                unconstrained[offset] = np.log(
                    constrained[offset] / (1 - constrained[offset])
                )
                offset += 1

        # Autoregressive coefficients must be stationary
        if self.autoregressive:
            unconstrained[offset:offset + self.ar_order] = (
                unconstrain_stationary_univariate(
                    constrained[offset:offset + self.ar_order]
                )
            )
            offset += self.ar_order

        # Nothing to do with betas
        unconstrained[offset:offset + self.k_exog] = (
            constrained[offset:offset + self.k_exog]
        )

        return unconstrained

    def update(self, params, **kwargs):
        params = super(UnobservedComponents, self).update(params, **kwargs)

        offset = 0

        # Observation covariance
        if self.irregular:
            self.ssm['obs_cov', 0, 0] = params[offset]
            offset += 1

        # State covariance
        if self.k_state_cov > 0:
            variances = params[offset:offset+self.k_state_cov]
            if self.stochastic_cycle and self.cycle:
                if self.autoregressive:
                    variances = np.r_[variances[:-1], variances[-2:]]
                else:
                    variances = np.r_[variances, variances[-1]]
            self.ssm[self._idx_state_cov] = variances
            offset += self.k_state_cov

        # Cycle transition
        if self.cycle:
            cos_freq = np.cos(params[offset])
            sin_freq = np.sin(params[offset])
            cycle_transition = np.array(
                [[cos_freq, sin_freq],
                 [-sin_freq, cos_freq]]
            )
            if self.damped_cycle:
                offset += 1
                cycle_transition *= params[offset]
            self.ssm[self._idx_cycle_transition] = cycle_transition
            offset += 1

        # AR transition
        if self.autoregressive:
            self.ssm[self._idx_ar_transition] = (
                params[offset:offset+self.ar_order]
            )
            offset += self.ar_order

        # Beta observation intercept
        if self.regression:
            if self.mle_regression:
                self.ssm['obs_intercept'] = np.dot(
                    self.exog,
                    params[offset:offset+self.k_exog]
                )[None, :]
            offset += self.k_exog

        # Initialize the state
        self.initialize_state()


class UnobservedComponentsResults(MLEResults):
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

    def __init__(self, model, params, filter_results, cov_type='opg',
                 **kwargs):
        super(UnobservedComponentsResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        # Save the model specification
        self.specification = Bunch(**{
            # Model options
            'level': self.model.level,
            'trend': self.model.trend,
            'seasonal_periods': self.model.seasonal_periods,
            'seasonal': self.model.seasonal,
            'cycle': self.model.cycle,
            'ar_order': self.model.ar_order,
            'autoregressive': self.model.autoregressive,
            'irregular': self.model.irregular,
            'stochastic_level': self.model.stochastic_level,
            'stochastic_trend': self.model.stochastic_trend,
            'stochastic_seasonal': self.model.stochastic_seasonal,
            'stochastic_cycle': self.model.stochastic_cycle,

            'damped_cycle': self.model.damped_cycle,
            'regression': self.model.regression,
            'mle_regression': self.model.mle_regression,
            'k_exog': self.model.k_exog,

            # Check for string trend/level specification
            'trend_specification': self.model.trend_specification
        })

    @property
    def level(self):
        """
        Estimates of unobserved level component

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
        # If present, level is always the first component of the state vector
        out = None
        spec = self.specification
        if spec.level:
            offset = 0
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

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
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def seasonal(self):
        """
        Estimates of unobserved seasonal component

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
        # If present, seasonal always follows level/trend (if they are present)
        # Note that we return only the first seasonal state, but there are
        # in fact seasonal_periods-1 seasonal states, however latter states
        # are just lagged versions of the first seasonal state.
        out = None
        spec = self.specification
        if spec.seasonal:
            offset = int(spec.trend + spec.level)
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def cycle(self):
        """
        Estimates of unobserved cycle component

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
        # If present, cycle always follows level/trend and seasonal
        # Note that we return only the first cyclical state, but there are
        # in fact 2 cyclical states. The second cyclical state is not simply
        # a lag of the first cyclical state, but the first cyclical state is
        # the one that enters the measurement equation.
        out = None
        spec = self.specification
        if spec.cycle:
            offset = int(spec.trend + spec.level +
                         spec.seasonal * (spec.seasonal_periods - 1))
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def autoregressive(self):
        """
        Estimates of unobserved autoregressive component

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
        # If present, autoregressive always follows level/trend, seasonal, and
        # cyclical. If it is an AR(p) model, then there are p associated
        # states, but the second - pth states are just lags of the first state.
        out = None
        spec = self.specification
        if spec.autoregressive:
            offset = int(spec.trend + spec.level +
                         spec.seasonal * (spec.seasonal_periods - 1) +
                         2 * spec.cycle)
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def regression_coefficients(self):
        """
        Estimates of unobserved regression coefficients

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
        # If present, state-vector regression coefficients always are last
        # (i.e. they follow level/trend, seasonal, cyclical, and
        # autoregressive states). There is one state associated with each
        # regressor, and all are returned here.
        out = None
        spec = self.specification
        if spec.regression:
            if spec.mle_regression:
                import warnings
                warnings.warn('Regression coefficients estimated via maximum'
                              ' likelihood. Estimated coefficients are'
                              ' available in the parameters list, not as part'
                              ' of the state vector.', OutputWarning)
            else:
                offset = int(spec.trend + spec.level +
                             spec.seasonal * (spec.seasonal_periods - 1) +
                             spec.cycle * (1 + spec.stochastic_cycle) +
                             spec.ar_order)
                start = offset
                end = offset + spec.k_exog
                out = Bunch(
                    filtered=self.filtered_state[start:end],
                    filtered_cov=self.filtered_state_cov[start:end, start:end],
                    smoothed=None, smoothed_cov=None,
                    offset=offset
                )
                if self.smoothed_state is not None:
                    out.smoothed = self.smoothed_state[start:end]
                if self.smoothed_state_cov is not None:
                    out.smoothed_cov = (
                        self.smoothed_state_cov[start:end, start:end])
        return out

    def plot_components(self, which=None, alpha=0.05,
                        observed=True, level=True, trend=True,
                        seasonal=True, cycle=True, autoregressive=True,
                        legend_loc='upper right', fig=None, figsize=None):
        """
        Plot the estimated components of the model.

        Parameters
        ----------
        which : {'filtered', 'smoothed'}, or None, optional
            Type of state estimate to plot. Default is 'smoothed' if smoothed
            results are available otherwise 'filtered'.
        alpha : float, optional
            The confidence intervals for the components are (1 - alpha) %
        level : boolean, optional
            Whether or not to plot the level component, if applicable.
            Default is True.
        trend : boolean, optional
            Whether or not to plot the trend component, if applicable.
            Default is True.
        seasonal : boolean, optional
            Whether or not to plot the seasonal component, if applicable.
            Default is True.
        cycle : boolean, optional
            Whether or not to plot the cyclical component, if applicable.
            Default is True.
        autoregressive : boolean, optional
            Whether or not to plot the autoregressive state, if applicable.
            Default is True.
        fig : Matplotlib Figure instance, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        If all options are included in the model and selected, this produces
        a 6x1 plot grid with the following plots (ordered top-to-bottom):

        0. Observed series against predicted series
        1. Level
        2. Trend
        3. Seasonal
        4. Cycle
        5. Autoregressive

        Specific subplots will be removed if the component is not present in
        the estimated model or if the corresponding keywork argument is set to
        False.

        All plots contain (1 - `alpha`) %  confidence intervals.
        """
        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        # Determine which results we have
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'

        # Determine which plots we have
        spec = self.specification
        components = OrderedDict([
            ('level', level and spec.level),
            ('trend', trend and spec.trend),
            ('seasonal', seasonal and spec.seasonal),
            ('cycle', cycle and spec.cycle),
            ('autoregressive', autoregressive and spec.autoregressive),
        ])

        llb = self.filter_results.loglikelihood_burn

        # Number of plots
        k_plots = observed + np.sum(list(components.values()))

        # Get dates, if applicable
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(len(self.data.endog))

        # Get the critical value for confidence intervals
        critical_value = norm.ppf(1 - alpha / 2.)

        plot_idx = 1

        # Observed, predicted, confidence intervals
        if observed:
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            # Plot the observed dataset
            ax.plot(dates[llb:], self.model.endog[llb:], color='k',
                    label='Observed')

            # Get the predicted values and confidence intervals
            predict = self.filter_results.forecasts[0]
            std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0, 0])
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors

            # Plot
            ax.plot(dates[llb:], predict[llb:],
                    label='One-step-ahead predictions')
            ci_poly = ax.fill_between(
                dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2
            )
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)

            # Proxy artist for fill_between legend entry
            # See e.g. http://matplotlib.org/1.3.1/users/legend_guide.html
            p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(p)
            labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)

            ax.set_title('Predicted vs observed')

        # Plot each component
        for component, is_plotted in components.items():
            if not is_plotted:
                continue

            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            component_bunch = getattr(self, component)

            # Check for a valid estimation type
            if which not in component_bunch:
                raise ValueError('Invalid type of state estimate.')

            which_cov = '%s_cov' % which

            # Get the predicted values
            value = component_bunch[which]

            # Plot
            state_label = '%s (%s)' % (component.title(), which)
            ax.plot(dates[llb:], value[llb:], label=state_label)

            # Get confidence intervals
            if which_cov in component_bunch:
                std_errors = np.sqrt(component_bunch['%s_cov' % which])
                ci_lower = value - critical_value * std_errors
                ci_upper = value + critical_value * std_errors
                ci_poly = ax.fill_between(
                    dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2
                )
                ci_label = ('$%.3g \\%%$ confidence interval'
                            % ((1 - alpha)*100))

            # Legend
            ax.legend(loc=legend_loc)

            ax.set_title('%s component' % component.title())

        # Add a note if first observations excluded
        if llb > 0:
            text = ('Note: The first %d observations are not shown, due to'
                    ' approximate diffuse initialization.')
            fig.text(0.1, 0.01, text % llb, fontsize='large')

        return fig

    def get_prediction(self, start=None, end=None, dynamic=False, exog=None,
                       **kwargs):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        exog : array_like, optional
            If the model includes exogenous regressors, you must provide
            exactly enough out-of-sample values for the exogenous variables if
            end is beyond the last observation in the sample.
        dynamic : boolean, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        full_results : boolean, optional
            If True, returns a FilterResults instance; if False returns a
            tuple with forecasts, the forecast errors, and the forecast error
            covariance matrices. Default is False.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts.
        """
        if start is None:
            start = 0

        # Handle end (e.g. date)
        _start = self.model._get_predict_start(start)
        _end, _out_of_sample = self.model._get_predict_end(end)

        # Handle exogenous parameters
        if _out_of_sample and self.model.k_exog > 0:
            # Create a new faux model for the extended dataset
            nobs = self.model.data.orig_endog.shape[0] + _out_of_sample
            endog = np.zeros((nobs, self.model.k_endog))

            if self.model.k_exog > 0:
                if exog is None:
                    raise ValueError('Out-of-sample forecasting in a model'
                                     ' with a regression component requires'
                                     ' additional exogenous values via the'
                                     ' `exog` argument.')
                exog = np.array(exog)
                required_exog_shape = (_out_of_sample, self.model.k_exog)
                if not exog.shape == required_exog_shape:
                    raise ValueError('Provided exogenous values are not of the'
                                     ' appropriate shape. Required %s, got %s.'
                                     % (str(required_exog_shape),
                                        str(exog.shape)))
                exog = np.c_[self.model.data.orig_exog.T, exog.T].T

            model_kwargs = self._init_kwds.copy()
            model_kwargs['exog'] = exog
            model = UnobservedComponents(endog, **model_kwargs)
            model.update(self.params)

            # Set the kwargs with the update time-varying state space
            # representation matrices
            for name in self.filter_results.shapes.keys():
                if name == 'obs':
                    continue
                mat = getattr(model.ssm, name)
                if mat.shape[-1] > 1:
                    if len(mat.shape) == 2:
                        kwargs[name] = mat[:, -_out_of_sample:]
                    else:
                        kwargs[name] = mat[:, :, -_out_of_sample:]
        elif self.model.k_exog == 0 and exog is not None:
            # TODO: UserWarning
            warn('Exogenous array provided to predict, but additional data not'
                 ' required. `exog` argument ignored.', ValueWarning)

        return super(UnobservedComponentsResults, self).get_prediction(
            start=start, end=end, dynamic=dynamic, exog=exog, **kwargs
        )

    def summary(self, alpha=.05, start=None):
        # Create the model name

        model_name = [self.specification.trend_specification]

        if self.specification.seasonal:
            seasonal_name = 'seasonal(%d)' % self.specification.seasonal_periods
            if self.specification.stochastic_seasonal:
                seasonal_name = 'stochastic ' + seasonal_name
            model_name.append(seasonal_name)

        if self.specification.cycle:
            cycle_name = 'cycle'
            if self.specification.stochastic_cycle:
                cycle_name = 'stochastic ' + cycle_name
            if self.specification.damped_cycle:
                cycle_name = 'damped ' + cycle_name
            model_name.append(cycle_name)

        if self.specification.autoregressive:
            autoregressive_name = 'AR(%d)' % self.specification.ar_order
            model_name.append(autoregressive_name)

        return super(UnobservedComponentsResults, self).summary(
            alpha=alpha, start=start, title='Unobserved Components Results',
            model_name=model_name
        )
    summary.__doc__ = MLEResults.summary.__doc__


class UnobservedComponentsResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(UnobservedComponentsResultsWrapper,
                      UnobservedComponentsResults)
