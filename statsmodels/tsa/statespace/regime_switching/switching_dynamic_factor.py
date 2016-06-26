import numpy as np
from statsmodels.tsa.statespace.regime_switching.rs_mlemodel import \
        RegimeSwitchingMLEModel
from statsmodels.tsa.statespace.regime_switching.tools import \
        MarkovSwitchingParams
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

class _DynamicFactorWithFactorIntercept(DynamicFactor):
    '''
    Extended Dynamic factor model with factor intercept term
    '''

    def __init__(self, *args, **kwargs):
        '''
        Params vector for this model is params vector for parent class
        concatenated with single value of factor intercept.
        '''

        super(_DynamicFactorWithFactorIntercept, self).__init__(*args,
                **kwargs)

        self._dynamic_factor_params_idx = np.s_[:-1]
        self._factor_intercept_idx = np.s_[-1]

    @property
    def start_params(self):

        dynamic_factor_params = super(_DynamicFactorWithFactorIntercept,
                self).start_params

        start_params = np.zeros((self.k_params + 1,), dtype=self.ssm.dtype)
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

        state_intercept = np.zeros((k_states,), dtype=dtype)

        state_intercept[0] = factor_intercept

        self['state_intercept'] = state_intercept


class RegimeSwitchingDynamicFactor(RegimeSwitchingMLEModel):
    '''
    Dynamic factor model with switching intercept term in factor changing law
    '''

    def __init__(self, k_regimes, endog, k_factors, factor_order, exog=None,
            error_order=0, error_var=False, error_cov_type='diagonal',
            enforce_stationarity=True, **kwargs):

        # Most of the logic is delegated to non-switching dynamic factor model
        self._dynamic_factor_model = DynamicFactor(endog, k_factors,
                factor_order, exog=exog, error_order=error_order,
                error_var=error_var, error_cov_type=error_cov_type,
                enforce_stationarity=enforce_stationarity, **kwargs)

        super(RegimeSwitchingDynamicFactor, self).__init__(k_regimes, endog,
                self._dynamic_factor_model.k_states, exog=exog, **kwargs)

        # A dirty hack.
        # This is required to delegate "update" method to non-switching model
        # No way to do it without rewriting DynamicFactor code
        self._dynamic_factor_model.ssm = self.ssm

        # This is required to initialize nonswitching_model
        self._init_kwargs = kwargs


        # `params` vector is concatenated transition matrix params,
        # DynamicFactor params and intercepts
        # for every regime.
        transition_params_length = k_regimes * (k_regimes - 1)

        self._dynamic_factor_params_idx = \
                np.c_[transition_params_length:-k_regimes]
        self._factor_intercepts_idx = np.c_[-k_regimes:]

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

    def update_params(self, params, nonswitching_params):
        '''
        `nonswitching_params` is _DynamicFactorWithFactorIntercept parameters
        vector. It is consists of DynamicFactor parameters concatenated with
        single factor intercept value.
        '''

        dynamic_factor_params = nonswitching_params[:-1]
        factor_intercept = nonswitching_params[-1]

        params[self._dynamic_factor_params_idx] = dynamic_factor_params

        # Setting all intercepts to one value.
        params[self._factor_intercepts_idx] = factor_intercept

        return params

    def get_nonswitching_params(self, params):

        dynamic_factor_params = params[self._dynamic_factor_params_idx]
        factor_intercept = params[self._factor_intercepts_idx].mean()

        return np.hstack((dynamic_factor_params, factor_intercept))

    def transform_model_params(self, unconstrained):

        constrained = np.array(unconstrained)

        dynamic_factor_unconstrained = \
                unconstrained[self._dynamic_factor_params_idx]
        dynamic_factor_constrained = \
                self._dynamic_factor_model.transform_params(
                dynamic_factor_unconstrained)

        constrained[self._dynamic_factor_params_idx] = \
                dynamic_factor_constrained

        return constrained

    def untransform_model_params(self, constrained):

        unconstrained = np.array(constrained)

        dynamic_factor_constrained = \
                constrained[self._dynamic_factor_params_idx]
        dynamic_factor_unconstrained = \
                self._dynamic_factor_model.untransform_params(
                dynamic_factor_constrained)

        unconstrained[self._dynamic_factor_params_idx] = \
                dynamic_factor_unconstrained

        return unconstrained

    def get_normal_regimes_permutation(self, params):

        k_regimes = self.k_regimes

        factor_intercepts = list(params[self._factor_intercepts_idx])

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

        params = super(RegimeSwitchingDynamicFactor, self).update(params,
                **kwargs)

        dynamic_factor_params = params[self._dynamic_factor_params_idx]

        # ssm in _dynamic_factor_model.update is a KimFilter instance.
        # So this call makes sence.
        self._dynamic_factor_model.update(dynamic_factor_params, **kwargs)

        factor_intercepts = params[self._factor_intercept_idx]

        state_intercept = np.zeros((k_regime, k_states), dtype=dtype)
        state_intercept[:, 0] = factor_intercepts

        self['state_intercept'] = state_intercept
