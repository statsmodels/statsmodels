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
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tools.tools import Bunch
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

    level : bool, optional
        Whether or not to include a level component. Default is False.
    trend : bool, optional
        Whether or not to include a trend component. Default is False. If True,
        `level` must also be True.
    seasonal_period : int or None, optional
        The period of the seasonal component. Default is None.
    cycle : bool, optional
        Whether or not to include a cycle component. Default is False.
    ar : int or None, optional
        The order of the autoregressive component. Default is None.
    exog : array_like or None, optional
        Exoenous variables.
    irregular : bool, optional
        Whether or not to include an irregular component. Default is True
    stochastic_level : bool, optional
        Whether or not any level component is stochastic. Default is True.
    stochastic_trend : bool, optional
        Whether or not any trend component is stochastic. Default is True.
    stochastic_seasonal : bool, optional
        Whether or not any seasonal component is stochastic. Default is True.
    stochastic_cycle : bool, optional
        Whether or not any cycle component is stochastic. Default is True.
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

    Thse models take the general form (see [1]_ Chapter 3.2 for all details)

    .. math::

        y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\mu_t` refers to the trend component, :math:`\gamma_t` refers to the
    seasonal component, :math:`c_t` refers to the cycle, and
    :math:`\varepsilon_t` is the irregular. The modeling details of these
    components are given below.

    **Trend**

    The trend is modeled either as a *local linear trend* model or as an
    *integrated random walk* model.

    The local linear trend is specified as:

    .. math::

        \mu_t = \mu_{t-1} + \nu_{t-1} + \xi_{t-1} \\
        \nu_t = \nu_{t-1} + \zeta_{t-1}

    with :math:`\xi_t \sim N(0, \sigma_\xi^2)` and
    :math:`\zeta_t \sim N(0, \sigma_\zeta^2)`.

    The integrated random walk model of order `r` is specified as:

    .. math::

        \Delta^r \mu_t = \xi_{t-1} \\

    This component results in two parameters to be selected via maximum
    likelihood: :math:`\sigma_\xi^2` and :math:`\sigma_\zeta^2`.

    In the case of the integrated random walk model, the parameter
    :math:`\sigma_\xi^2` is constrained to be zero, but the parameter `r` (the
    order of integration) must be chosen (it is not estimated by MLE).

    **Seasonal**

    The seasonal component is modeled as:

    .. math::

        \gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \\
        \omega_t \sim N(0, \sigma_\omega^2)

    where s is the number of seasons and :math:`\omega_t` is an error term that
    allows the seasonal constants to change over time (if this is not desired,
    :math:`\sigma_\omega^2` can be set to zero).

    This component results in one parameter to be selected via maximum
    likelihood: :math:`\sigma_\omega^2`, and one parameter to be chosen, the
    number of seasons `s`.

    **Cycle**

    The cyclical component is modeled as

    .. math::

        c_{t+1} = \rho_c (\tilde c_t \cos \lambda_c t
                + \tilde c_t^* \sin \lambda_c) +
                \tilde \omega_t \\
        c_{t+1}^* = \rho_c (- \tilde c_t \sin \lambda_c  t +
                \tilde c_t^* \cos \lambda_c) +
                \tilde \omega_t^* \\

    where :math:`\omega_t, \tilde \omega_t iid N(0, \sigma_{\tilde \omega}^2)`

    This component results in three parameters to be selected via maximum
    likelihood: :math:`\sigma_{\tilde \omega}^2`, :math:`\rho_c`, and
    :math:`\lambda_c`.

    **Irregular**

    The irregular components are independent and identically distributed (iid):

    .. math::

        \varepsilon_t \sim N(0, \sigma_\varepsilon^2)

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
        self.seasonal_period = seasonal if seasonal is not None else 0
        self.seasonal = self.seasonal_period > 0
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
                         % attribute)
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
                 " deterministic level component added.")
            self.level = True
            self.stochastic_level = False

        if not (self.irregular or
                (self.level and self.stochastic_level) or
                (self.trend and self.stochastic_trend) or
                (self.seasonal and self.stochastic_seasonal) or
                (self.cycle and self.stochastic_cycle) or
                self.autoregressive):
            warn("Specified model does not contain a stochastic element;"
                 " irregular component added.")
            self.irregular = True

        if self.seasonal and self.seasonal_period < 2:
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
            if exog.ndim == 1:
                if not exog_is_using_pandas:
                    exog = exog[:, None]
                else:
                    exog = pd.DataFrame(exog)

            self.k_exog = exog.shape[1]
        self.regression = self.k_exog > 0

        # Model parameters
        k_states = (
            self.level + self.trend +
            (self.seasonal_period - 1) * self.seasonal +
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

    def setup(self):
        """
        Setup the structural time series representation
        """
        # TODO fix this
        # (if we don't set it here, each instance shares a single dictionary)
        self._start_params = {
            'irregular_var': 0.1,
            'level_var': 0.1,
            'trend_var': 0.1,
            'seasonal_var': 0.1,
            'cycle_freq': 0.1,
            'cycle_var': 0.1,
            'cycle_damp': 0.1,
            'ar_coeff': 0,
            'ar_var': 0.1,
            'reg_coeff': 0,
        }
        self._param_names = {
            'irregular_var': 'sigma2.irregular',
            'level_var': 'sigma2.level',
            'trend_var': 'sigma2.trend',
            'seasonal_var': 'sigma2.seasonal',
            'cycle_var': 'sigma2.cycle',
            'cycle_freq': 'frequency.cycle',
            'cycle_damp': 'damping.cycle',
            'ar_coeff': 'ar.L%d',
            'ar_var': 'sigma2.ar',
            'reg_coeff': 'beta.%d',
        }

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
            n = self.seasonal_period - 1
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
            self._start_params['ar_coeff'] = (
                [self._start_params['ar_coeff']] * self.ar_order
            )
            self._param_names['ar_coeff'] = [
                self._param_names['ar_coeff'] % k
                for k in range(1, self.ar_order+1)
            ]
            j += 1
            i += self.ar_order
        if self.regression:
            if self.mle_regression:
                self.parameters_obs_intercept['reg_coeff'] = self.k_exog
                self._start_params['reg_coeff'] = (
                    [self._start_params['reg_coeff']] * self.k_exog
                )
                self._param_names['reg_coeff'] = [
                    self._param_names['reg_coeff'] % k
                    for k in range(1, self.k_exog+1)
                ]
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
                (self.seasonal_period - 1) * self.seasonal +
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

    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        result = super(UnobservedComponents, self).filter(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = UnobservedComponentsResultsWrapper(
                UnobservedComponentsResults(self, params, result,
                                            **result_kwargs)
            )

        return result

    @property
    def start_params(self):
        if not hasattr(self, 'parameters'):
            return []

        # Eliminate missing data to estimate starting parameters
        endog = self.endog
        exog = self.exog
        if np.any(np.isnan(endog)):
            endog = endog[~np.isnan(endog)]
            if exog is not None:
                exog = exog[~np.isnan(endog)]

        # Level / trend variances
        # (Use the HP filter to get initial estimates of variances)
        _start_params = self._start_params.copy()
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

        # Seasonal
        if self.stochastic_seasonal:
            # TODO seasonal variance starting values?
            pass

        # Cyclical
        if self.cycle:
            _start_params['cycle_var'] = np.std(resid)**2
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
        # Irregular
        else:
            _start_params['irregular_var'] = np.std(resid)**2

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
            if np.isscalar(self._param_names[key]):
                param_names.append(self._param_names[key])
            else:
                param_names += self._param_names[key]
        return param_names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation
        """
        unconstrained = np.array(unconstrained)
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
        constrained = np.array(constrained)
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

        # Save the model specification
        self.specification = Bunch(**{
            # Model options
            'level': self.model.level,
            'trend': self.model.trend,
            'seasonal_period': self.model.seasonal_period,
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
        Filtered value of unobserved level component
        """
        # If present, level is always the first component of the state vector
        out = None
        spec = self.specification
        if spec.level:
            offset = 0
            res = self.filter_results
            out = Bunch(filtered=res.filtered_state[offset],
                        filtered_cov=res.filtered_state_cov[offset, offset],
                        offset=offset)
        return out

    @property
    def trend(self):
        """
        Filtered value of unobserved trend component
        """
        # If present, trend is always the second component of the state vector
        # (because level is always present if trend is present)
        out = None
        spec = self.specification
        if spec.trend:
            offset = int(spec.level)
            res = self.filter_results
            out = Bunch(filtered=res.filtered_state[offset],
                        filtered_cov=res.filtered_state_cov[offset, offset],
                        offset=offset)
        return out

    @property
    def seasonal(self):
        """
        Filtered value of unobserved seasonal component
        """
        # If present, seasonal always follows level/trend (if they are present)
        # Note that we return only the first seasonal state, but there are
        # in fact seasonal_period-1 seasonal states, however latter states
        # are just lagged versions of the first seasonal state.
        out = None
        spec = self.specification
        if spec.seasonal:
            offset = int(spec.trend + spec.level)
            res = self.filter_results
            out = Bunch(filtered=res.filtered_state[offset],
                        filtered_cov=res.filtered_state_cov[offset, offset],
                        offset=offset)
        return out

    @property
    def cycle(self):
        """
        Filtered value of unobserved cycle component
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
                         spec.seasonal * (spec.seasonal_period - 1))
            res = self.filter_results
            out = Bunch(filtered=res.filtered_state[offset],
                        filtered_cov=res.filtered_state_cov[offset, offset],
                        offset=offset)
        return out

    @property
    def autoregressive(self):
        """
        Filtered value of unobserved autoregressive component
        """
        # If present, autoregressive always follows level/trend, seasonal, and
        # cyclical. If it is an AR(p) model, then there are p associated
        # states, but the second - pth states are just lags of the first state.
        out = None
        spec = self.specification
        if spec.autoregressive:
            offset = int(spec.trend + spec.level +
                         spec.seasonal * (spec.seasonal_period - 1) +
                         2 * spec.cycle)
            res = self.filter_results
            out = Bunch(filtered=res.filtered_state[offset],
                        filtered_cov=res.filtered_state_cov[offset, offset],
                        offset=offset)
        return out

    @property
    def regression_coefficients(self):
        """
        Filtered value of unobserved regression coefficients
        """
        # If present, state-vector regression coefficients always are last
        # (i.e. they follow level/trend, seasonal, cyclical, and
        # autoregressive states). There is one state associated with each
        # regressor, and all are returned here.
        out = None
        spec = self.specification
        if spec.regression:
            if spec.mle_regression:
                warnings.warn('Regression coefficients estimated via maximum'
                              ' likelihood. Estimated coefficients are'
                              ' available in the parameters list, not as part'
                              ' of the state vector.')
            else:
                offset = int(spec.trend + spec.level +
                             spec.seasonal * (spec.seasonal_period - 1) +
                             spec.cycle * (1 + spec.stochastic_cycle) +
                             spec.ar_order)
                res = self.filter_results
                start = offset
                end = offset + k_exog
                out = Bunch(
                    filtered=res.filtered_state[start:end],
                    filtered_cov=res.filtered_state_cov[start:end, start:end],
                    offset=offset
                )
        return out

    def plot_components(self, which='filtered', alpha=0.05,
                        observed=True, level=True, trend=True,
                        seasonal=True, cycle=True, autoregressive=True,
                        fig=None, figsize=None):
        """
        Plot the estimated components of the model.

        Parameters
        ----------
        which : {'filtered'}, optional
            Type of state estimate to plot. Default is 'filtered'.
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

        # Determine which plots we have
        spec = self.specification
        components = OrderedDict([
            ('level', level and spec.level),
            ('trend', trend and spec.trend),
            ('seasonal', seasonal and spec.seasonal),
            ('cycle', cycle and spec.cycle),
            ('autoregressive', autoregressive and spec.autoregressive),
        ])

        # Number of plots
        k_plots = observed + np.sum(list(components.values()))

        # Get dates, if applicable
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(len(resid))

        # Get the critical value for confidence intervals
        critical_value = norm.ppf(1 - alpha / 2.)

        plot_idx = 1

        # Observed, predicted, confidence intervals
        if observed:
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            # Plot the observed dataset
            ax.plot(dates, self.model.endog, color='k', label='Observed')

            # Get the predicted values and confidence intervals
            predict = self.filter_results.forecasts[0]
            std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0,0])
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors

            # Plot
            ax.plot(dates, predict, label='One-step-ahead predictions')
            ci_poly = ax.fill_between(dates, ci_lower, ci_upper, alpha=0.2)
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha)*100)

            # Proxy artist for fill_between legend entry
            # See e.g. http://matplotlib.org/1.3.1/users/legend_guide.html
            p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(p)
            labels.append(ci_label)
            ax.legend(handles, labels)

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

            # Get the predicted values and confidence intervals
            value = component_bunch[which]
            std_errors = np.sqrt(component_bunch['%s_cov' % which])
            ci_lower = value - critical_value * std_errors
            ci_upper = value + critical_value * std_errors

            # Plot
            state_label = '%s (%s)' % (component.title(), which)
            ax.plot(dates, value, label=state_label)
            ci_poly = ax.fill_between(dates, ci_lower, ci_upper, alpha=0.2)
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha)*100)

            # Legend
            ax.legend()

            ax.set_title('%s component' % component.title())

        return fig

    def summary(self, alpha=.05, start=None):
        # Create the model name

        model_name = [self.specification.trend_specification]

        if self.specification.seasonal:
            seasonal_name = 'seasonal(%d)' % self.specification.seasonal_period
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
