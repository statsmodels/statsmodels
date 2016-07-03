import numpy as np
from .kim_smoother import KimSmoother
from .switching_mlemodel import SwitchingMLEModel
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


class _SwitchingDynamicFactorSmoother(KimSmoother):
    '''
    This is required for compatibility, when we set DynamicFactor `ssm` property
    to KimSmoother instance.
    '''

    def __getitem__(self, key):

        item = super(_SwitchingDynamicFactorSmoother, self).__getitem__(key)

        if key is tuple and key[0] == 'state_cov':
            # `state_cov` is the same for every regime, so any index is OK
            return item[0]
        else:
            return item


class _DynamicFactorWithFactorIntercept(DynamicFactor):
    '''
    Extended Dynamic factor model with factor intercept term
    '''

    # Params vector for this model is params vector for parent class
    # concatenated with single value of factor intercept.
    _dynamic_factor_params_idx = np.s_[:-1]
    _factor_intercept_idx = np.s_[-1]

    def __init__(self, *args, **kwargs):

        super(_DynamicFactorWithFactorIntercept, self).__init__(*args, **kwargs)

        # For the sake of clarity
        self.dynamic_factor_k_params = self.k_params

        # One more value for factor intercept
        self.k_params_with_factor_intercept = self.k_params + 1

    @property
    def start_params(self):

        dynamic_factor_params = super(_DynamicFactorWithFactorIntercept,
                self).start_params

        start_params = np.zeros((self.k_params_with_factor_intercept,),
                dtype=self.ssm.dtype)

        start_params[self._dynamic_factor_params_idx] = dynamic_factor_params

        return start_params

    def transform_params(self, unconstrained):

        constrained = np.array(unconstrained)

        dynamic_factor_unconstrained = \
                unconstrained[self._dynamic_factor_params_idx]
        dynamic_factor_constrained = super(_DynamicFactorWithFactorIntercept,
                self).transform_params(dynamic_factor_unconstrained)

        constrained[self._dynamic_factor_params_idx] = \
                dynamic_factor_constrained

        return constrained

    def untransform_params(self, constrained):

        unconstrained = np.array(constrained)

        dynamic_factor_constrained = constrained[self._dynamic_factor_params_idx]
        dynamic_factor_unconstrained = super(_DynamicFactorWithFactorIntercept,
                self).untransform_params(dynamic_factor_constrained)

        unconstrained[self._dynamic_factor_params_idx] = \
                dynamic_factor_unconstrained

        return unconstrained

    def update(self, params, **kwargs):

        k_states = self.k_states
        dtype = self.ssm.dtype

        dynamic_factor_params = params[self._dynamic_factor_params_idx]
        factor_intercept = params[self._factor_intercept_idx]

        super(_DynamicFactorWithFactorIntercept,
                self).update(dynamic_factor_params, **kwargs)

        state_intercept = np.zeros((k_states, 1), dtype=dtype)

        state_intercept[0, 0] = factor_intercept

        self['state_intercept'] = state_intercept


class SwitchingDynamicFactor(SwitchingMLEModel):
    '''
    Dynamic factor model with switching intercept term in factor changing law
    '''

    def __init__(self, k_regimes, endog, k_factors, factor_order, exog=None,
            error_order=0, error_var=False, error_cov_type='diagonal',
            enforce_stationarity=True, **kwargs):

        # Most of the logic is delegated to non-switching dynamic factor model
        self._dynamic_factor_model = DynamicFactor(endog,
                k_factors, factor_order, exog=exog, error_order=error_order,
                error_var=error_var, error_cov_type=error_cov_type,
                enforce_stationarity=enforce_stationarity, **kwargs)

        super(SwitchingDynamicFactor, self).__init__(k_regimes, endog,
                self._dynamic_factor_model.k_states, exog=exog, **kwargs)

        # A dirty hack.
        # This is required to delegate "update" method to non-switching model
        # No way to do it without rewriting DynamicFactor code
        self._dynamic_factor_model.ssm = self.ssm

        # Initializing fixed components of state space matrices, one time
        # again for new `ssm`
        self._dynamic_factor_model._initialize_loadings()
        self._dynamic_factor_model._initialize_exog()
        self._dynamic_factor_model._initialize_error_cov()
        self._dynamic_factor_model._initialize_factor_transition()
        self._dynamic_factor_model._initialize_error_transition()

        # This is required to initialize nonswitching_model
        self._init_kwargs = kwargs

        self.parameters['dynamic_factor'] = [False] * \
                self._dynamic_factor_model.k_params
        self.parameters['factor_intercept'] = [True]

    def initialize_statespace(self, **kwargs):

        endog = self.endog.T

        self.ssm = _SwitchingDynamicFactorSmoother(endog.shape[0],
                self.k_states, self.k_regimes, **kwargs)

        self.ssm.bind(endog)

        self.k_endog = self.ssm.k_endog

    def get_nonswitching_model(self):

        endog = self.endog
        k_factors = self._dynamic_factor_model.k_factors
        factor_order = self._dynamic_factor_model.factor_order
        exog = self.exog
        error_order = self._dynamic_factor_model.error_order
        error_var = self._dynamic_factor_model.error_var
        error_cov_type = self.error_cov_type
        enforce_stationarity = self._dynamic_factor_model.enforce_stationarity
        kwargs = self._init_kwargs

        return _DynamicFactorWithFactorIntercept(endog, k_factors, factor_order,
                exog=exog, error_order=error_order, error_var=error_var,
                error_cov_type=error_cov_type,
                enforce_stationarity=enforce_stationarity, **kwargs)

    def update_params(self, params, nonswitching_params, noise=0.5, seed=1):
        '''
        `nonswitching_params` is DynamicFactorWithFactorIntercept parameters
        vector. It is consists of DynamicFactor parameters concatenated with
        single factor intercept value.
        `noise` is a white noise scale, relative to factor intercept
        absolute value. It is used to break the symmetry of equal factor
        intercepts. It's defined in arguments to make redefinition in
        ancestors possible.
        '''

        dynamic_factor_params = nonswitching_params[ \
                _DynamicFactorWithFactorIntercept._dynamic_factor_params_idx]
        factor_intercept = nonswitching_params[ \
                _DynamicFactorWithFactorIntercept._factor_intercept_idx]

        params[self.parameters['dynamic_factor']] = dynamic_factor_params

        # Setting all intercepts to one value.
        params[self.parameters['factor_intercept']] = factor_intercept

        np.random.seed(seed=seed)

        # Adding noise to break the symmetry
        noise_scale = np.linalg.norm(
                params[self.parameters['factor_intercept']], np.inf) * noise
        params[self.parameters['factor_intercept']] += \
                np.random.normal(scale=noise_scale, size=self.k_regimes)

        return params

    def transform_model_params(self, unconstrained):

        constrained = np.array(unconstrained)

        dynamic_factor_unconstrained = \
                unconstrained[self.parameters['dynamic_factor']]
        dynamic_factor_constrained = \
                self._dynamic_factor_model.transform_params(
                dynamic_factor_unconstrained)

        constrained[self.parameters['dynamic_factor']] = \
                dynamic_factor_constrained

        return constrained

    def untransform_model_params(self, constrained):

        unconstrained = np.array(constrained)

        dynamic_factor_constrained = \
                constrained[self.parameters['dynamic_factor']]
        dynamic_factor_unconstrained = \
                self._dynamic_factor_model.untransform_params(
                dynamic_factor_constrained)

        unconstrained[self.parameters['dynamic_factor']] = \
                dynamic_factor_unconstrained

        return unconstrained

    def get_normal_regimes_permutation(self, params):

        k_regimes = self.k_regimes

        factor_intercepts = list(params[self.parameters['factor_intercept']])

        permutation = sorted(range(k_regimes),
                key=lambda x:factor_intercepts[x])

        return permutation

    def update(self, params, **kwargs):
        '''
        `params` vector is concatenated transition matrix params,
        DynamicFactor params and intercepts
        for every regime.
        '''

        k_regimes = self.k_regimes
        k_states = self.k_states
        dtype = self.ssm.dtype

        params = super(SwitchingDynamicFactor, self).update(params,
                **kwargs)

        self['regime_transition'] = self._get_param_regime_transition(params)

        dynamic_factor_params = params[self.parameters['dynamic_factor']]

        # `ssm` in `_dynamic_factor_model` is a KimFilter instance.
        # So this call makes sence.
        self._dynamic_factor_model.update(dynamic_factor_params, **kwargs)

        factor_intercepts = params[self.parameters['factor_intercept']]

        state_intercept = np.zeros((k_regimes, k_states, 1), dtype=dtype)
        state_intercept[:, 0, 0] = factor_intercepts

        self['state_intercept'] = state_intercept
