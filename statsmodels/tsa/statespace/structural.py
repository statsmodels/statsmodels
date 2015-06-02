"""
Univariate structural time series models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from collections import OrderedDict

import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from .mlemodel import MLEModel
from scipy.linalg import solve_discrete_lyapunov
from .tools import (
    companion_matrix, constrain_stationary_univariate,
    unconstrain_stationary_univariate
)


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
                 cycle=False, ar=None, exog=None, irregular=True,
                 stochastic_level=True, stochastic_trend=True,
                 stochastic_seasonal=True, stochastic_cycle=True,
                 damped_cycle=False, cycle_period_bounds=None,
                 mle_regression=True,
                 **kwargs):

        # Model options
        self.level = level
        self.trend = trend
        self.seasonal_period = seasonal if seasonal is not None else 0
        self.seasonal = seasonal > 0
        self.cycle = cycle
        self.ar_order = ar if ar is not None else 0
        self.ar = self.ar_order > 0
        self.irregular = irregular

        self.stochastic_level = bool(stochastic_level and level)
        self.stochastic_trend = bool(stochastic_trend and trend)
        self.stochastic_seasonal = bool(stochastic_seasonal and seasonal)
        self.stochastic_cycle = bool(stochastic_cycle and cycle)

        self.damped_cycle = bool(damped_cycle and cycle)
        self.mle_regression = mle_regression

        # Check for a model that makes sense
        if trend and not level:
            warn("Trend component specified without level component;"
                 " level component added.")
            self.level = True

        if (self.irregular + self.stochastic_level + self.stochastic_trend +
                self.stochastic_seasonal + self.stochastic_cycle) == 0:
            warn("Specified model does not contain a stochastic element;"
                 " irregular component added.")
            self.irregular = True

        # Exogenous component
        self.k_exog = 0
        if exog is not None:
            exog = np.array(exog)
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
            self.ar
        )

        # We can still estimate the model with just the irregular component,
        # just need to have one state that does nothing.
        loglikelihood_burn = k_states - self.ar_order
        if k_states == 0:
            if not irregular:
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
        self.loglikelihood_burn = loglikelihood_burn

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
            self['design', 0, i] = 1.
            self['transition', i, i] = 1.
            if self.trend:
                self['transition', i, i+1] = 1.
            if self.stochastic_level:
                self['selection', i, j] = 1.
                self.parameters_state_cov['level_var'] = 1
                j += 1
            i += 1
        if self.trend:
            self['transition', i, i] = 1.
            if self.stochastic_trend:
                self['selection', i, j] = 1.
                self.parameters_state_cov['trend_var'] = 1
                j += 1
            i += 1
        if self.seasonal:
            n = self.seasonal_period - 1
            self['design', 0, i] = 1.
            self['transition', i:i + n, i:i + n] = (
                companion_matrix(np.r_[1, [1] * n]).transpose()
            )
            if self.stochastic_seasonal:
                self['selection', i, j] = 1.
                self.parameters_state_cov['seasonal_var'] = 1
                j += 1
            i += n
        if self.cycle:
            self['design', 0, i] = 1.
            self.parameters_transition['cycle_freq'] = 1
            if self.damped_cycle:
                self.parameters_transition['cycle_damp'] = 1
            if self.stochastic_cycle:
                self['selection', i:i+2, j:j+2] = np.eye(2)
                self.parameters_state_cov['cycle_var'] = 1
                j += 1
            self._idx_cycle_transition = np.s_['transition', i:i+2, i:i+2]
            i += 2
        if self.ar:
            self['design', 0, i] = 1.
            self.parameters_transition['ar_coeff'] = self.ar_order
            self.parameters_state_cov['ar_var'] = 1
            self['selection', i, j] = 1
            self['transition', i:i+self.ar_order, i:i+self.ar_order] = (
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
                design = np.repeat(self['design', :, :, 0], self.nobs, axis=0)
                self['design'] = design.transpose()[np.newaxis, :, :]
                self['design', 0, i:i+self.k_exog, :] = self.exog.transpose()
                self['transition', i:i+self.k_exog, i:i+self.k_exog] = (
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
        idx = np.diag_indices(self.k_posdef)
        self._idx_state_cov = ('state_cov', idx[0], idx[1])

    def initialize_state(self):
        # Initialize the AR component as stationary, the rest as approximately
        # diffuse
        initial_state = np.zeros(self.k_states)
        initial_state_cov = (
            np.eye(self.k_states, dtype=self.transition.dtype) *
            self.initial_variance
        )

        if self.ar:

            start = (
                self.level + self.trend +
                (self.seasonal_period - 1) * self.seasonal +
                self.cycle * 2
            )
            end = start + self.ar_order
            selection_stationary = self.selection[start:end, :, 0]
            selected_state_cov_stationary = np.dot(
                np.dot(selection_stationary, self.state_cov[:, :, 0]),
                selection_stationary.T
            )
            try:
                initial_state_cov_stationary = solve_discrete_lyapunov(
                    self.transition[start:end, start:end, 0],
                    selected_state_cov_stationary
                )
            except:
                initial_state_cov_stationary = solve_discrete_lyapunov(
                    self.transition[start:end, start:end, 0],
                    selected_state_cov_stationary,
                    method='direct'
                )

            initial_state_cov[start:end, start:end] = (
                initial_state_cov_stationary
            )

        self.initialize_known(initial_state, initial_state_cov)

    @property
    def start_params(self):
        if not hasattr(self, 'parameters'):
            return []

        # Level / trend variances
        # (Use the HP filter to get initial estimates of variances)
        _start_params = self._start_params.copy()
        if self.level:
            resid, trend1 = hpfilter(self.endog)

            if self.stochastic_trend:
                cycle2, trend2 = hpfilter(trend1)
                _start_params['trend_var'] = np.std(trend2)**2
                if self.stochastic_level:
                    _start_params['level_var'] = np.std(cycle2)**2
            elif self.stochastic_level:
                _start_params['level_var'] = np.std(trend1)**2
        else:
            resid = self.endog

        # Seasonal
        if self.stochastic_seasonal:
            # TODO seasonal variance starting values?
            pass

        # Cyclical
        if self.cycle:
            _start_params['cycle_var'] = np.std(resid)**2
            _start_params['cycle_damp'] = (
                np.linalg.pinv(resid[:-1, None]).dot(resid[1:])[0]
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
        if self.ar:
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
        if self.ar:
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
            self['obs_cov', 0, 0] = params[offset]
            offset += 1

        # State covariance
        if self.k_state_cov > 0:
            variances = params[offset:offset+self.k_state_cov]
            if self.stochastic_cycle and self.cycle:
                if self.ar:
                    variances = np.r_[variances[:-1], variances[-2:]]
                else:
                    variances = np.r_[variances, variances[-1]]
            self[self._idx_state_cov] = variances
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
            self[self._idx_cycle_transition] = cycle_transition
            offset += 1

        # AR transition
        if self.ar:
            self[self._idx_ar_transition] = params[offset:offset+self.ar_order]
            offset += self.ar_order

        # Beta observation intercept
        if self.regression:
            if self.mle_regression:
                self['obs_intercept'] = np.dot(
                    self.exog,
                    params[offset:offset+self.k_exog]
                )[None, :]
            offset += self.k_exog

        # Initialize the state
        self.initialize_state()
