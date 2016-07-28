"""
Markov switching time varying parameters model

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from .switching_mlemodel import SwitchingMLEModel, SwitchingMLEResults
from statsmodels.tsa.statespace.tvp import TVPModel, TVPResults


class SwitchingTVPModel(SwitchingMLEModel):
    r"""
    Markov switching time varying parameters model

    Parameters
    ----------
    k_regimes : int
        The number of switching regimes.
    endog : array_like
        The observed time series process :math:`y`.
    exog : array_like
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog. Required for this model.
    switching_obs_cov : bool, optional
        Whether observation variance/covariance is switching. Default is `True`.
    switching_tvp_cov : bool or array_like of bool, optional
        If a boolean, sets whether or not all time varying parameter
        variances/covariances are switching across regimes. If an iterable,
        should be of length equal to `k_exog`, where each element is a boolean
        describing whether the corresponding coefficient variance/covariance is
        switching. Default is `False`.
    kwargs
        Keyword arguments, passed to superclass ('MLEModel') initializer.

    Attributes
    ----------
    k_exog : int
        The number of exog variables.

    Notes
    -----
    The specification of the model is as follows:

    .. math::

        y_t = \beta_{1t} x_{1t} + \beta_{2t} x_{2t} + ... +
        \beta_{kt} x_{kt} + e_t \\
        \beta_{it} = \beta_{i,t-1} + v_{it} \\
        e_t \sim N(0, \sigma_{S_t}^2) \\
        v_{it} \sim N(0, \sigma_{i,S_t}^2)

    where :math:`\beta_{it}` are time varying parameters and :math:`x_{it}` are
    exogenous variables. Observation and transition error terms have a switching
    variance - :math:`\sigma_{S_t}^2` and :math:`\sigma_{i,S_t}^2` respectively.
    :math:`S_t` is a regime at the moment :math:`t`.

    See Also
    --------
    statsmodels.tsa.statespace.tvp.TVPModel
    SwitchingMLEModel

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, k_regimes, endog, exog, switching_obs_cov=True,
            switching_tvp_cov=False, **kwargs):

        # Delete exog in optional arguments
        if 'exog' in kwargs:
            del kwargs['exog']

        # Transform to numpy array
        exog = np.asarray(exog)

        # Reshape exog, if it is one-dimensional array
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

        # The number of time varying parameters is defined by `exog` second
        # dimension
        self.k_exog = exog.shape[1]
        k_exog = self.k_exog

        # Superclass initialization
        super(SwitchingTVPModel, self).__init__(k_regimes, endog, k_exog,
                exog=exog, **kwargs)

        # Every observation is a signle value
        if self.k_endog != 1:
            raise ValueError('Endogenous vector must be univariate.')

        if isinstance(switching_tvp_cov, bool):
            switching_tvp_cov = [switching_tvp_cov] * k_exog
        else:
            switching_tvp_cov = list(switching_tvp_cov)

        if not any(switching_tvp_cov) and not switching_obs_cov:
            raise ValueError('None of the parameters is switching. Consider' \
                    ' using non-switching model.')

        self.switching_obs_cov = switching_obs_cov
        self.switching_tvp_cov = switching_tvp_cov

        # Parameter vector slices handling
        self.parameters['obs_var'] = [switching_obs_cov]
        self.parameters['tvp_var'] = switching_tvp_cov

        # Parameter names
        self._param_names += ['obs_var_{0}'.format(i) for i in \
                range(k_regimes)]

        for i, is_switching in zip(range(k_exog), switching_tvp_cov):
            if is_switching:
                self._param_names += ['tvp_var{0}_{1}'.format(i, j) for j in \
                        range(k_regimes)]
            else:
                self._param_names += ['tvp_var{0}'.format(i)]

        dtype = self.ssm.dtype

        # Setting up constant representation matrices

        self['design'] = np.array(exog.T.reshape(1, k_exog, -1))

        self['transition'] = np.identity(k_exog).reshape(k_exog, k_exog, 1)

        self['selection'] = np.identity(k_exog).reshape(k_exog, k_exog, 1)

    def get_nonswitching_model(self):

        regime_filters = self.ssm._regime_kalman_filters

        # Instantiate non-switching model
        model = TVPModel(self.endog, exog=self.exog, dtype=self.ssm.dtype)

        # Switching model initialization
        initial_states = [regime_filter._initial_state for regime_filter in \
                regime_filters]
        initial_state_covs = [regime_filter._initial_state_cov for \
                regime_filter in regime_filters]

        # By default, TVP model is not initialized
        # This is a simple heuristic - use the average initalization among
        # regimes
        model.initialize_known(np.asarray(initial_states).mean(axis=0),
                np.asarray(initial_state_covs).mean(axis=0))

        return model

    def update_params(self, params, nonswitching_params, noise=0.1, seed=1):
        """
        Update constrained parameters of the model, using parameters of
        non-switching model.
 
        Parameters
        ----------
        noise : float
            Relative normal noise scale, added to switching parameters to break
            the symmetry of regimes.
        seed : int
            Random seed, used to generate the noise.

        Notes
        -----
        `noise` and `seed` parameters are defined in keyword arguments so that
        they can be changed in child class.

        See Also
        --------
        SwitchingMLEModel.update_params
        """

        dtype = self.ssm.dtype
        k_regimes = self.k_regimes
        k_exog = self.k_exog

        switching_obs_cov = self.switching_obs_cov
        switching_tvp_cov = self.switching_tvp_cov

        # Set observation variance to one non-switching value for every regime
        params[self.parameters['obs_var']] = \
                nonswitching_params[TVPModel._obs_var_idx]

        # Set non-switching time varying parameters variance to those from
        # non-switching model
        for i in range(k_regimes):
            params[self.parameters[i, 'tvp_var']] = \
                    nonswitching_params[TVPModel._tv_params_cov_idx]

        # Set the seed
        np.random.seed(seed=seed)

        # Add noise to switching parameters to break the symmetry

        # Add noise to observation variance
        if switching_obs_cov:
            noise_scale = np.absolute(nonswitching_params[
                    TVPModel._obs_var_idx]) * noise
            params[self.parameters['obs_var']] += np.random.normal(
                    scale=noise_scale, size=k_regimes)

            # Keep variances non-negative
            params[self.parameters['obs_var']] = np.maximum(0,
                    params[self.parameters['obs_var']])

        # Add noise to tvp variance
        if any(switching_tvp_cov):

            mask = np.array(switching_tvp_cov)

            # Get normal white noise scale for every parameter
            noise_scales = np.absolute(nonswitching_params[
                    TVPModel._tv_params_cov_idx]) * noise

            switching_tvp_count = np.sum(mask)
            for i in range(k_regimes):
                params[self.parameters[i, 'tvp_var']][mask] += \
                        np.random.normal(scale=noise_scales[mask])

                # Keep variances non-negative
                params[self.parameters[i, 'tvp_var']][mask] = np.maximum(0,
                        params[self.parameters[i, 'tvp_var']])

        return params

    def transform_model_params(self, unconstrained):

        constrained = np.array(unconstrained)

        # Keep the variances non-negative
        constrained[self.parameters['obs_var']] **= 2
        constrained[self.parameters['tvp_var']] **= 2

        return constrained

    def untransform_model_params(self, constrained):

        unconstrained = np.array(constrained)

        # Keep the variances non-negative
        unconstrained[self.parameters['obs_var']] **= 0.5
        unconstrained[self.parameters['tvp_var']] **= 0.5

        return unconstrained

    def get_normal_regimes_permutation(self, params):

        k_regimes = self.k_regimes

        # Use increasing regime params tuple order as normal permutation
        return sorted(range(k_regimes),
                key=lambda x: tuple(params[self.parameters[x]]))

    def update(self, params, **kwargs):

        # Transform `params` vector if it is untransformed
        params = super(SwitchingTVPModel, self).update(params, **kwargs)

        k_regimes = self.k_regimes
        k_exog = self.k_exog

        # Decode transition matrix and store it in the representation
        self['regime_transition'] = self._get_param_regime_transition(params)

        # Update regime-switching observation covariance
        self['obs_cov'] = np.array(params[self.parameters['obs_var']]).reshape(
                k_regimes, 1, 1, 1)

        # Update time varying parameters covariance

        state_cov = np.zeros((k_regimes, k_exog, k_exog, 1))

        for i in range(k_regimes):
            state_cov[i, :, :, 0] = np.diag(params[self.parameters[i,
                    'tvp_var']])

        self['state_cov'] = state_cov

    def smooth(self, params, results_class=None, **kwargs):

        # `TVPResults` is compatible
        if results_class is None:
            results_class = SwitchingTVPResults

        return super(SwitchingTVPModel, self).smooth(params,
                results_class=results_class, **kwargs)


class SwitchingTVPResults(SwitchingMLEResults, TVPResults):

    pass
