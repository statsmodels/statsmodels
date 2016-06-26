import numpy as np
from statsmodels.tsa.statespace.api import MLEModel
from .tools import MarkovSwitchingParams
from .kim_smoother import KimSmoother


class SwitchingMLEModel(MLEModel):

    def __init__(self, k_regimes, endog, k_states, param_k_regimes=None,
            **kwargs):
        '''
        param_k_regimes is specified in the case, when
        regime_transition matrix in parameters is different from that in state
        space representation, like it happens with MS-AR.
        '''

        self.k_regimes = k_regimes

        if param_k_regimes is None:
            self.param_k_regimes = self.k_regimes
        else:
            self.param_k_regimes = param_k_regimes

        self._init_kwargs = kwargs

        self.parameters = MarkovSwitchingParams(self.param_k_regimes)

        self.parameters['regime_transition'] = [False] * \
                self.param_k_regimes * (self.param_k_regimes - 1)

        super(SwitchingMLEModel, self).__init__(endog, k_states, **kwargs)

    def initialize_statespace(self, **kwargs):

        endog = self.endog.T

        self.ssm = KimSmoother(endog.shape[0], self.k_states, self.k_regimes,
                **kwargs)

        self.ssm.bind(endog)

        self.k_endog = self.ssm.k_endog

    def get_nonswitching_model(self):
        '''
        Returns non-switching model instance
        To override, if fitting non-switching model is performed.
        '''

        raise NotImplementedError

    def update_params(self, params, nonswitching_params):
        '''
        This method is used when non-switching fitting is done, and we need to
        update switching model starting params using obtained non-switching
        params. To override, if fitting non-switching model is performed.
        '''

        return np.array(params, ndmin=1)

    def get_nonswitching_params(self, params):
        '''
        Used when switching model params are provided in fit method, and we
        need to transform them into params for nonswitching model fitting.
        To override, if fitting non-switching model is performed.
        '''

        return np.array(params, ndmin=1)

    def transform_regime_transition(self, unconstrained):
        '''
        This is default implementation of logistic transformation of transition
        matrix.
        unconstrained transition is a raveled transition matrix without last
        row.
        '''

        param_k_regimes = self.param_k_regimes

        constrained = np.array(unconstrained)

        unconstrained_transition = \
                unconstrained[self.parameters['regime_transition']].reshape(
                (param_k_regimes - 1, param_k_regimes))

        constrained_transition = np.exp(unconstrained_transition)

        constrained_transition /= \
                (1 + constrained_transition.sum(axis=0)).reshape((1, -1))

        constrained[self.parameters['regime_transition']] = \
                constrained_transition.ravel()

        return constrained

    def untransform_regime_transition(self, constrained):
        '''
        This is default implementation of logistic transformation of transition
        matrix.
        constrained transition is a raveled transformed transition matrix
        without last row.
        '''

        param_k_regimes = self.param_k_regimes

        unconstrained = np.array(constrained)

        #TODO: pass as an argument?
        eps = 1e-8
        constrained_transition = \
                constrained[self.parameters['regime_transition']].reshape(
                (param_k_regimes - 1, param_k_regimes))
        unconstrained_transition = np.array(constrained_transition)
        unconstrained_transition[unconstrained_transition == 0] = eps
        unconstrained_transition /= \
                (1 - unconstrained_transition.sum(axis=0)).reshape(1, -1)
        unconstrained_transition = np.log(unconstrained_transition)

        unconstrained[self.parameters['regime_transition']] = \
                unconstrained_transition.ravel()

        return unconstrained

    def transform_model_params(self, unconstrained):
        '''
        Model params are all parameters except regime transition.
        This method is to be overridden by user.
        '''

        return np.array(unconstrained)

    def untransform_model_params(self, constrained):
        '''
        Model params are all parameters except regime transition.
        This method is to be overridden by user.
        '''

        return np.array(constrained)

    def transform_params(self, unconstrained):

        return self.transform_model_params(
                self.transform_regime_transition(unconstrained))

    def untransform_params(self, constrained):
        return self.untransform_model_params(
                self.untransform_regime_transition(constrained))

    def _permute_regimes(self, params, permutation):
        '''
        `permutation` is provided by `get_normal_regimes_permutation` method.
        This method is used in `normalize_regimes` method.
        '''

        param_k_regimes = self.param_k_regimes
        dtype = self.ssm.dtype

        ar_regime_transition = self._get_param_regime_transition(params)
        new_ar_regime_transition = np.zeros((param_k_regimes, param_k_regimes),
                dtype=dtype)

        for i in range(param_k_regimes):
            for j in range(param_k_regimes):
                new_ar_regime_transition[i, j] = \
                        ar_regime_transition[permutation[i],
                        permutation[j]]

        new_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        self._set_param_regime_transition(new_params, new_ar_regime_transition)

        for i in range(param_k_regimes):
            new_params[self.parameters[i]] = \
                    params[self.parameters[permutation[i]]]

        return new_params

    def get_normal_regimes_permutation(self, params):
        '''
        Parameters vector depends on permutation of regimes in it, which means
        that several different vectors can represent the only model
        configuration. To compare configurations (e.g. for testing), we need to
        normalize parameters.
        This method should be overridden, if required, to return a permutation
        of indices [0, ..., k_regimes], later used to permute parameters in
        normalize_params.
        `params` is supposed to be constrained here.
        '''

        param_k_regimes = self.param_k_regimes

        # Identity permutation
        return list(range(param_k_regimes))

    def normalize_params(self, params, transformed=True):

        if not transformed:
            params = self.transform_params(params)

        permutation = self.get_normal_regimes_permutation(params)
        params = self._permute_regimes(params, permutation)

        if not transformed:
            params = self.untransform_params(params)

        return params

    def set_smoother_output(self, **kwargs):

        raise NotImplementedError

    def initialize_approximate_diffuse(self, **kwargs):

        raise NotImplementedError

    def initialize_known_regime_probs(self, *args):

        self.ssm.initialize_known_regime_probs(*args)

    def initialize_uniform_regime_probs(self):

        self.ssm.initialize_uniform_regime_probs()

    def initialize_stationary_regime_probs(self):

        self.ssm.initialize_stationary_regime_probs()

    def _get_param_regime_transition(self, constrained_params):

        dtype = self.ssm.dtype
        param_k_regimes = self.param_k_regimes

        regime_transition = np.zeros((param_k_regimes, param_k_regimes),
                dtype=dtype)
        regime_transition[:-1, :] = constrained_params[
                self.parameters['regime_transition']].reshape((-1,
                param_k_regimes))

        regime_transition[-1, :] = 1 - regime_transition[:-1, :].sum(axis=0)

        return regime_transition

    def _set_param_regime_transition(self, constrained_params, regime_transition):

        constrained_params[self.parameters['regime_transition']] = \
                regime_transition[:-1, :].ravel()

    @property
    def start_params(self):
        return self.transform_params(np.ones((self.parameters.k_params,),
            dtype=self.ssm.dtype))

    def fit(self, start_params=None, transformed=True,
            set_equal_transition_probs=False, fit_nonswitching_first=False,
            **kwargs):

        if start_params is None:
            start_params = self.start_params
            transformed = True

        if not transformed:
            start_params = self.transform_params(start_params)

        if fit_nonswitching_first:
            nonswitching_model = self.get_nonswitching_model()
            nonswitching_kwargs = dict(kwargs)
            nonswitching_kwargs['return_params'] = True
            start_nonswitching_params = \
                    self.get_nonswitching_params(start_params)
            nonswitching_params = nonswitching_model.fit(
                    start_params=start_nonswitching_params,
                    **nonswitching_kwargs)

            start_params = self.update_params(start_params, nonswitching_params)

        if set_equal_transition_probs:
            start_params[self.parameters['regime_transition']] = \
                    np.ones((self.param_k_regimes, self.param_k_regimes),
                    dtype=self.ssm.dtype)[:-1, :].ravel() / self.param_k_regimes

        kwargs['start_params'] = start_params
        # smoothing is not defined yet
        kwargs['return_params'] = True
        return super(SwitchingMLEModel, self).fit(**kwargs)

    def filter(self, params, transformed=True, complex_step=False, **kwargs):

        self.update(params, transformed=transformed, complex_step=complex_step)

        return self.ssm.filter(complex_step=complex_step, **kwargs)

    def smooth(self, params, transformed=True, complex_step=False, **kwargs):

        self.update(params, transformed=True, complex_step=complex_step)

        kwargs['run_filter'] = True

        return self.ssm.smooth(complex_step=complex_step, **kwargs)

    def simulation_smoother(self, *args, **kwargs):

        raise NotImplementedError

    def _forecast_error_partial_derivatives(self, *args, **kwargs):

        raise NotImplementedError

    def observed_information_matrix(self, *args, **kwargs):

        raise NotImplementedError

    def opg_information_matrix(self, *args, **kwargs):

        raise NotImplementedError

    def _score_complex_step(self, *args, **kwargs):

        raise NotImplementedError

    def _score_finite_difference(self, *args, **kwargs):

        raise NotImplementedError

    def _score_harvey(self, *args, **kwargs):

        raise NotImplementedError

    def score_obs_harvey(self, *args, **kwargs):

        raise NotImplementedError

    def score(self, *args, **kwargs):

        raise NotImplementedError

    def score_obs(self, *args, **kwargs):

        raise NotImplementedError

    def hessian(self, *args, **kwargs):

        raise NotImplementedError

    def _hessian_oim(self, *args, **kwargs):

        raise NotImplementedError

    def _hessian_opg(self, *args, **kwargs):

        raise NotImplementedError

    def _hessian_finite_difference(self, *args, **kwargs):

        raise NotImplementedError

    def _hessian_complex_step(self, *args, **kwargs):

        raise NotImplementedError

    def transform_jacobian(self, *args, **kwargs):

        raise NotImplementedError

    def simulate(self, *args, **kwargs):

        raise NotImplementedError

    def impulse_responses(self, *args, **kwargs):

        raise NotImplementedError
