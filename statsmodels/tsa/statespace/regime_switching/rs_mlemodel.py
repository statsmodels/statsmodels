import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.regime_switching.tools import \
        MarkovSwitchingParams
from statsmodels.tsa.statespace.regime_switching.kim_filter import KimFilter


class RegimeSwitchingMLEModel(MLEModel):

    def __init__(self, k_regimes, endog, k_states,
            param_k_regimes=None, exog=None, dates=None, freq=None,
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

        super(RegimeSwitchingMLEModel, self).__init__(endog, k_states,
                exog=exog, dates=dates, freq=freq, **kwargs)

    def initialize_statespace(self, **kwargs):

        endog = self.endog.T

        self.ssm = KimFilter(endog.shape[0], self.k_states, self.k_regimes,
                **kwargs)

        self.ssm.bind(endog)

        self.k_endog = self.ssm.k_endog

    @property
    def nonswitching_model_type(self):
        '''
        To override, if fitting non-switching model is performed.
        '''

        return MLEModel

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

    def normalize_params(self, params, transformed=True):
        '''
        Parameters vector depends on permutation of regimes in it, which means
        that several different vectors can represent the only model
        configuration. To compare configurations (e.g. for testing), we need to
        normalize parameters.
        To override, if needed.
        '''

        raise NotImplementedError

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
            nonswitching_model = self.nonswitching_model_type(
                    self.endog, self.k_states, exog=self.exog,
                    dates=self.data.dates, freq=self.data.freq,
                    **self._init_kwargs)
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
        return super(RegimeSwitchingMLEModel, self).fit(**kwargs)

    def filter(self, params, transformed=True, complex_step=False, **kwargs):

        self.update(params, transformed=transformed, complex_step=complex_step)

        #if complex_step:
        #    kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        self.ssm.filter(complex_step=complex_step, **kwargs)

    def get_smoothed_regime_probs(self, params, transformed=True,
            complex_step=False, **kwargs):

        self.update(params, transformed=True, complex_step=complex_step)

        kwargs['filter_first'] = True

        return self.ssm.get_smoothed_regime_probs(**kwargs)

    def smooth(self, *args, **kwargs):

        raise NotImplementedError

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
