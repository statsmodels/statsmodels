import numpy as np
from .switching_mlemodel import SwitchingMLEModel
from statsmodels.tsa.statespace.tvp import TVPModel


class SwitchingTVPModel(SwitchingMLEModel):

    def __init__(self, k_regimes, endog, exog=None, **kwargs):

        if exog is None:
            raise ValueError('Exogenous data is required for this model.')

        exog = np.asarray(exog)

        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

        self.k_exog = exog.shape[1]
        k_exog = self.k_exog

        super(SwitchingTVPModel, self).__init__(k_regimes, endog, k_exog,
                exog=exog, **kwargs)

        if self.k_endog != 1:
            raise ValueError('Endogenous vector must be univariate.')

        self.parameters['obs_var'] = [True]
        self.parameters['tvp_var'] = [False] * k_exog

        self._param_names += ['obs_var_{0}'.format(i) for i in \
                range(k_regimes)]
        self._param_names += ['tvp_var{0}'.format(i) for i in \
                range(k_exog)]

        dtype = self.ssm.dtype

        self['design'] = np.array(exog.T.reshape(1, k_exog, -1), dtype=dtype)

        self['transition'] = np.identity(k_exog, dtype=dtype).reshape(k_exog,
                k_exog, 1)

        self['selection'] = np.identity(k_exog, dtype=dtype).reshape(k_exog,
                k_exog, 1)

    def get_nonswitching_model(self):

        regime_filters = self.ssm._regime_kalman_filters

        model = TVPModel(self.endog, exog=self.exog, dtype=self.ssm.dtype)

        initial_states = [regime_filter._initial_state for regime_filter in \
                regime_filters]
        initial_state_covs = [regime_filter._initial_state_cov for \
                regime_filter in regime_filters]

        model.initialize_known(np.asarray(initial_states).mean(axis=0),
                np.asarray(initial_state_covs).mean(axis=0))

        return model

    def update_params(self, params, nonswitching_params, noise=0.5, seed=1):

        k_regimes = self.k_regimes

        params[self.parameters['obs_var']] = \
                nonswitching_params[TVPModel._obs_var_idx]
        params[self.parameters['tvp_var']] = \
                nonswitching_params[TVPModel._tv_params_cov_idx]

        np.random.seed(seed=seed)

        noise_scale = np.linalg.norm(params[self.parameters['obs_var']],
                np.inf) * noise

        params[self.parameters['obs_var']] += np.random.normal(
                scale=noise_scale, size=k_regimes)

        return params

    def transform_model_params(self, unconstrained):

        constrained = np.array(unconstrained)

        constrained[self.parameters['obs_var']] **= 2
        constrained[self.parameters['tvp_var']] **= 2

        return constrained

    def untransform_model_params(self, constrained):

        unconstrained = np.array(constrained)

        unconstrained[self.parameters['obs_var']] **= 0.5
        unconstrained[self.parameters['tvp_var']] **= 0.5

        return unconstrained

    def get_normal_regimes_permutation(self, params):

        k_regimes = self.k_regimes

        return sorted(range(k_regimes),
                key=lambda x: params[self.parameters[x, 'obs_var']])

    def update(self, params, **kwargs):

        params = super(SwitchingTVPModel, self).update(params, **kwargs)

        dtype = self.ssm.dtype
        k_regimes = self.k_regimes
        k_exog = self.k_exog

        self['regime_transition'] = self._get_param_regime_transition(params)

        self['obs_cov'] = np.array(params[self.parameters['obs_var']]).reshape(
                k_regimes, 1, 1, 1)

        state_cov = np.identity(k_exog, dtype=dtype).reshape(k_exog, k_exog, 1)
        state_cov[:, :, 0] *= params[self.parameters['tvp_var']]

        self['state_cov'] = state_cov
