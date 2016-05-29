import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.regime_switching.kim_filter import KimFilter


class RegimeSwitchingMLEModel(MLEModel):

    def __init__(self, k_regimes, endog, k_states, exog=None, dates=None,
            freq=None, **kwargs):

        self.k_regimes = k_regimes
        self._init_kwargs = kwargs

        super(RegimeSwitchingMLEModel, self).__init__(endog, k_states,
                exog=None, dates=None, freq=None, **kwargs)

    def initialize_statespace(self, **kwargs):

        endog = self.endog.T

        self.ssm = KimFilter(endog.shape[0], self.k_states, self.k_regimes,
                **kwargs)

        self.ssm.bind(endog)

        self.k_endog = self.ssm.k_endog

    @property
    def nonswitching_model_type(self):
        # to override, if fitting nonswitching is performed

        return MLEModel

    def get_model_params(self, nonswitching_model_params):
        # to override, if fitting nonswitching is performed

        return nonswitching_model_params

    def get_nonswitching_model_params(self, model_params):
        # to override, if fitting nonswitching is performed

        return model_params

    def transform_switch_probs(self, unconstrained_switch_probs):
        # to override, if needed

        k_regimes = self.k_regimes

        unconstrained_switch_probs = unconstrained_switch_probs.reshape(
                (k_regimes - 1, k_regimes))

        constrained_switch_probs = np.exp(unconstrained_switch_probs)
        constrained_switch_probs /= \
                (1 + constrained_switch_probs.sum(axis=0)).reshape((1, -1))

        return constrained_switch_probs.ravel()

    def untransform_switch_probs(self, constrained_switch_probs):
        # to override, if needed

        k_regimes = self.k_regimes
        #TODO: pass as an argument?
        eps = 1e-8
        constrained_switch_probs = \
                constrained_switch_probs.reshape((k_regimes - 1, k_regimes))
        unconstrained_switch_probs = np.array(constrained_switch_probs)
        unconstrained_switch_probs[unconstrained_switch_probs == 0] = eps
        unconstrained_switch_probs /= \
                (1 - unconstrained_switch_probs.sum(axis=0)).reshape(1, -1)
        unconstrained_switch_probs = np.log(unconstrained_switch_probs)

        return unconstrained_switch_probs.ravel()

    def transform_model_params(self, unconstrained_model_params):
        # to override

        return np.array(unconstrained_model_params, ndmin=1)

    def untransform_model_params(self, constrained_model_params):
        # to override

        return np.array(constrained_model_params, ndmin=1)

    def transform_params(self, unconstrained):

        border = self.k_regimes * (self.k_regimes - 1)

        constrained_switch_probs = \
                self.transform_switch_probs(unconstrained[:border])
        constrained_model_params = \
                self.transform_model_params(unconstrained[border:])

        return np.hstack((constrained_switch_probs, constrained_model_params))

    def untransform_params(self, constrained):

        border = self.k_regimes * (self.k_regimes - 1)

        unconstrained_switch_probs = \
                self.untransform_switch_probs(constrained[:border])
        unconstrained_model_params = \
                self.untransform_model_params(constrained[border:])
        return np.hstack((unconstrained_switch_probs,
                unconstrained_model_params))

    def set_smoother_output(self, **kwargs):

        raise NotImplementedError

    def initialize_approximate_diffuse(self, **kwargs):

        raise NotImplementedError

    def _get_params_vector(self, start_switch_probs, start_model_params):

        return np.hstack((start_switch_probs[:-1, :].ravel(),
                start_model_params))

    def _get_explicit_params(self, constrained_params):

        k_regimes = self.k_regimes
        border = k_regimes * (k_regimes - 1)

        switch_probs = constrained_params[:border].reshape((-1, k_regimes))
        switch_probs = np.vstack((switch_probs, np.ones((1, k_regimes),
                dtype=self.ssm.dtype)))

        model_params = constrained_params[border:]

        return (switch_probs, model_params)

    def fit(self, start_switch_probs=None, start_model_params=None,
            fit_nonswitching_first=True, **kwargs):

        if start_switch_probs is None:
            start_switch_probs = np.identity(self.k_regimes,
                    dtype=self.ssm.dtype)

        if fit_nonswitching_first:
            nonswitching_model = self.nonswitching_model_type(
                    self.endog, self.k_states, exog=self.exog,
                    dates=self.data.dates, freq=self.data.freq,
                    **self._init_kwargs)
            nonswitching_kwargs = dict(kwargs)
            nonswitching_kwargs['return_params'] = True
            nonswitching_model_params = self.get_nonswitching_model_params(
                    start_model_params)
            nonswitching_model_params = nonswitching_model.fit(
                    start_params=nonswitching_model_params,
                    **nonswitching_kwargs)

            start_model_params = self.get_model_params(
                    nonswitching_model_params)

        start_params = self._get_params_vector(start_switch_probs,
                start_model_params)

        kwargs['start_params'] = start_params
        return super(RegimeSwitchingMLEModel, self).fit(**kwargs)

    def smooth(self, **kwargs):

        raise NotImplementedError

    def simulation_smoother(self, **kwargs):

        raise NotImplementedError

    def simulate(self, *args, **kwargs):

        raise NotImplementedError

    def impulse_responses(self, *args, **kwargs):

        raise NotImplementedError
