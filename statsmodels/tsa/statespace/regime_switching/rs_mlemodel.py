import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.regime_switching.kim_filter import KimFilter


class RegimeSwitchingMLEModel(MLEModel):

    def __init__(self, k_regimes, endog, k_states, exog=None, dates=None,
            freq=None, **kwargs):

        self.k_regimes = k_regimes
        self._init_kwargs = kwargs

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

    def get_model_params(self, nonswitching_model_params):
        '''
        Note that switching model has two groups of parameters: parameters of
        regime transition matrix and model parameters.
        This method is used when non-switching fitting is done, and we need to
        get switching model starting params.
        To override, if fitting non-switching model is performed.
        '''
        return np.array(nonswitching_model_params, ndmin=1)

    def get_nonswitching_model_params(self, model_params):
        '''
        Note that switching model has two groups of parameters: parameters of
        regime transition matrix and model parameters.
        Used when switching model params are provided in fit method, and we
        need to transform them into params for nonswitching model fitting.
        To override, if fitting non-switching model is performed.
        '''

        return np.array(model_params, ndmin=1)

    def transform_transition(self, unconstrained_transition, k_regimes=None):
        '''
        Note that switching model has two groups of parameters: parameters of
        regime transition matrix and model parameters.
        This is default implementation of logistic transformation of transition
        matrix.
        unconstrained_transition is a raveled transition matrix without last
        row.
        k_regimes is specified in the case, when regime_transition matrix in
        parameters is different from that in state space representation, like
        it happens with MS-AR.
        '''

        if k_regimes is None:
            k_regimes = self.k_regimes

        unconstrained_transition = unconstrained_transition.reshape(
                (k_regimes - 1, k_regimes))

        constrained_transition = np.exp(unconstrained_transition)
        constrained_transition /= \
                (1 + constrained_transition.sum(axis=0)).reshape((1, -1))

        return constrained_transition.ravel()

    def untransform_transition(self, constrained_transition, k_regimes=None):
        '''
        Note that switching model has two groups of parameters: parameters of
        regime transition matrix and model parameters.
        This is default implementation of logistic transformation of transition
        matrix.
        constrained_transition is a raveled transformed transition matrix
        without last row.
        k_regimes is specified in the case, when regime_transition matrix in
        parameters is different from that in state space representation, like
        it happens with MS-AR.
        '''

        if k_regimes is None:
            k_regimes = self.k_regimes

        #TODO: pass as an argument?
        eps = 1e-8
        constrained_transition = \
                constrained_transition.reshape((k_regimes - 1, k_regimes))
        unconstrained_transition = np.array(constrained_transition)
        unconstrained_transition[unconstrained_transition == 0] = eps
        unconstrained_transition /= \
                (1 - unconstrained_transition.sum(axis=0)).reshape(1, -1)
        unconstrained_transition = np.log(unconstrained_transition)

        return unconstrained_transition.ravel()

    def transform_model_params(self, unconstrained_model_params):
        '''
        Note that switching model has two groups of parameters: parameters of
        regime transition matrix and model parameters.
        This method is to be overridden by user.
        '''

        return np.array(unconstrained_model_params, ndmin=1)

    def untransform_model_params(self, constrained_model_params):
        '''
        Note that switching model has two groups of parameters: parameters of
        regime transition matrix and model parameters.
        This method is to be overridden by user.
        '''

        return np.array(constrained_model_params, ndmin=1)

    def transform_params(self, unconstrained):

        border = self.k_regimes * (self.k_regimes - 1)

        constrained_transition = \
                self.transform_transition(unconstrained[:border])
        constrained_model_params = \
                self.transform_model_params(unconstrained[border:])

        return np.hstack((constrained_transition, constrained_model_params))

    def untransform_params(self, constrained):

        border = self.k_regimes * (self.k_regimes - 1)

        unconstrained_transition = \
                self.untransform_transition(constrained[:border])
        unconstrained_model_params = \
                self.untransform_model_params(constrained[border:])
        return np.hstack((unconstrained_transition,
                unconstrained_model_params))

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

    def _get_params_vector(self, start_transition, start_model_params):
        return np.hstack((start_transition[:-1, :].ravel(),
                start_model_params))

    def get_explicit_params(self, constrained_params, k_regimes=None):
        '''
        k_regimes is specified in the case, when regime_transition matrix in
        parameters is different from that in state space representation, like
        it happens with MS-AR.
        '''

        if k_regimes is None:
            k_regimes = self.k_regimes

        border = k_regimes * (k_regimes - 1)

        transition = constrained_params[:border].reshape((-1, k_regimes))
        transition = np.vstack((transition, np.ones((1, k_regimes),
                dtype=self.ssm.dtype)))

        model_params = constrained_params[border:]

        return (transition, model_params)

    def fit(self, start_transition=None, start_model_params=None,
            fit_nonswitching_first=True, **kwargs):

        if fit_nonswitching_first:
            nonswitching_model = self.nonswitching_model_type(
                    self.endog, self.k_states, exog=self.exog,
                    dates=self.data.dates, freq=self.data.freq,
                    **self._init_kwargs)
            nonswitching_kwargs = dict(kwargs)
            nonswitching_kwargs['return_params'] = True
            if start_model_params is not None:
                nonswitching_model_params = self.get_nonswitching_model_params(
                        start_model_params)
            else:
                nonswitching_model_params = None

            nonswitching_model_params = nonswitching_model.fit(
                    start_params=nonswitching_model_params,
                    **nonswitching_kwargs)

            start_model_params = self.get_model_params(
                    nonswitching_model_params)

        if start_transition is None and start_model_params is not None:
            # Other heuristics?
            start_transition = np.identity(self.k_regimes,
                    dtype=self.ssm.dtype)

        if start_transition is not None and start_model_params is None:
            start_transition = None

        if start_transition is None and start_model_params is None:
            start_params = None
        else:
            start_params = self._get_params_vector(start_transition,
                    start_model_params)

        kwargs['start_params'] = start_params
        # smoothing is not defined yet
        kwargs['return_params'] = True
        return super(RegimeSwitchingMLEModel, self).fit(**kwargs)

    def smooth(self, *args, **kwargs):

        raise NotImplementedError

    def simulation_smoother(self, *args, **kwargs):

        raise NotImplementedError

    def simulate(self, *args, **kwargs):

        raise NotImplementedError

    def impulse_responses(self, *args, **kwargs):

        raise NotImplementedError
