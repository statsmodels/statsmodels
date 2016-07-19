"""
Markov switching time varying parameters model

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from .switching_mlemodel import SwitchingMLEModel
from statsmodels.tsa.statespace.tvp import TVPModel


class SwitchingTVPModel(SwitchingMLEModel):
    r"""
    Markov switching time varying parameters model

    Parameters
    ----------
    k_regimes : int
        The number of switching regimes.
    endog : array_like
        The observed time series process :math:`y`.
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog. Required for this model.
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
        e_t \sim N(0, \sigma_{(S_t)}^2) \\
        v_{it} \sim N(0, \sigma_i^2)

    where :math:`\beta_{it}` are time varying parameters and :math:`x_{it}` are
    exogenous variables. Observation error term has a switching variance -
    :math:`\sigma_{(S_t)}^2`. :math:`S_t` is a regime at the moment :math:`t`.

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

    def __init__(self, k_regimes, endog, exog=None, **kwargs):

        # Check if exogenous data is provided
        if exog is None:
            raise ValueError('Exogenous data is required for this model.')

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

        # Parameter vector slices handling
        self.parameters['obs_var'] = [True]
        self.parameters['tvp_var'] = [False] * k_exog

        # Parameter names
        self._param_names += ['obs_var_{0}'.format(i) for i in \
                range(k_regimes)]
        self._param_names += ['tvp_var{0}'.format(i) for i in \
                range(k_exog)]

        dtype = self.ssm.dtype

        # Setting up constant representation matrices

        self['design'] = np.array(exog.T.reshape(1, k_exog, -1), dtype=dtype)

        self['transition'] = np.identity(k_exog, dtype=dtype).reshape(k_exog,
                k_exog, 1)

        self['selection'] = np.identity(k_exog, dtype=dtype).reshape(k_exog,
                k_exog, 1)

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

    def update_params(self, params, nonswitching_params, noise=0.5, seed=1):

        k_regimes = self.k_regimes

        # Set observation variance to one non-switching value for every regime
        params[self.parameters['obs_var']] = \
                nonswitching_params[TVPModel._obs_var_idx]

        # Set non-switching time varying parameters variance to those from
        # non-switching model
        params[self.parameters['tvp_var']] = \
                nonswitching_params[TVPModel._tv_params_cov_idx]

        # Set the seed
        np.random.seed(seed=seed)

        # Get the noise scale
        noise_scale = np.linalg.norm(params[self.parameters['obs_var']],
                np.inf) * noise

        # Add noise to switching parameters to break the symmetry
        params[self.parameters['obs_var']] += np.random.normal(
                scale=noise_scale, size=k_regimes)

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

        # Use increasing variance order as normal permutation
        return sorted(range(k_regimes),
                key=lambda x: params[self.parameters[x, 'obs_var']])

    def update(self, params, **kwargs):

        # Transform `params` vector if it is untransformed
        params = super(SwitchingTVPModel, self).update(params, **kwargs)

        dtype = self.ssm.dtype
        k_regimes = self.k_regimes
        k_exog = self.k_exog

        # Decode transition matrix and store it in the representation
        self['regime_transition'] = self._get_param_regime_transition(params)

        # Update regime-switching observation covariance
        self['obs_cov'] = np.array(params[self.parameters['obs_var']]).reshape(
                k_regimes, 1, 1, 1)

        # Update time varying parameters covariance
        state_cov = np.identity(k_exog, dtype=dtype).reshape(k_exog, k_exog, 1)
        state_cov[:, :, 0] *= params[self.parameters['tvp_var']]
        self['state_cov'] = state_cov
