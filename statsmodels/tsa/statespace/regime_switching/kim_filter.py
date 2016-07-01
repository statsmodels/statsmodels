import numpy as np
from .switching_representation import SwitchingRepresentation, \
        FrozenSwitchingRepresentation
from scipy.misc import logsumexp


class _KimFilter(object):

    def __init__(self, model, **kwargs):

        self.model = model

        if not hasattr(model, '_nobs'):
            raise RuntimeError(
                    'No endog data binded. Consider using bind() first.')

        model._initialize_filters(**kwargs)

    def _hamilton_prediction_step(self, t,
            predicted_prev_and_curr_regime_logprobs):

        model = self.model

        if t == 0:
            regime_logprobs = model._initial_regime_logprobs
        else:
            regime_logprobs = self.filtered_regime_logprobs[t - 1, :]

        np.add(model._log_regime_transition.transpose(),
                regime_logprobs.reshape(-1, 1),
                out=predicted_prev_and_curr_regime_logprobs)

        # This is used in smoothing
        self.predicted_regime_logprobs[t, :] = logsumexp(
                predicted_prev_and_curr_regime_logprobs, axis=0)

    def _kalman_filter_step(self, t, prev_regime, curr_regime, state_buffer,
            state_cov_buffer, state_batteries, state_cov_batteries,
            prev_and_curr_regime_cond_obs_logprobs):

        model = self.model

        curr_kfilter = model._kfilters[curr_regime]
        prev_kfilter = model._kfilters[prev_regime]
        curr_regime_filter = model._regime_kalman_filters[curr_regime]
        prev_regime_filter = model._regime_kalman_filters[prev_regime]
        curr_kfilter.seek(t)

        if t == 0:
            state_buffer = curr_regime_filter._initial_state
            state_cov_buffer = curr_regime_filter._initial_state_cov
            curr_regime_filter.initialize_known(
                    prev_regime_filter._initial_state,
                    prev_regime_filter._initial_state_cov)
            curr_regime_filter._initialize_state(
                    **model._state_init_kwargs[curr_regime])
        else:
            np.copyto(state_buffer, curr_kfilter.filtered_state[:, t - 1])
            np.copyto(state_cov_buffer,
                    curr_kfilter.filtered_state_cov[:, :, t - 1])
            np.copyto(np.asarray(curr_kfilter.filtered_state[:, t - 1]),
                    prev_kfilter.filtered_state[:, t - 1])
            np.copyto(np.asarray(curr_kfilter.filtered_state_cov[:, :, t - 1]),
                    prev_kfilter.filtered_state_cov[:, :, t - 1])

        next(curr_kfilter)

        if t == 0:
            curr_regime_filter.initialize_known(state_buffer, state_cov_buffer)
        else:
            np.copyto(np.asarray(curr_kfilter.filtered_state[:, t - 1]),
                    state_buffer)
            np.copyto(np.asarray(curr_kfilter.filtered_state_cov[:, :, t - 1]),
                    state_cov_buffer)

        np.copyto(state_batteries[prev_regime, curr_regime, :],
                curr_kfilter.filtered_state[:, t])
        np.copyto(state_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.filtered_state_cov[:, :, t])

        # This is more related to hamilton step, but loglikelihood is
        # internally calculated inside `next` call.
        prev_and_curr_regime_cond_obs_logprobs[prev_regime, curr_regime] = \
                curr_kfilter.loglikelihood[t]

    def _hamilton_filtering_step(self, t,
            predicted_prev_and_curr_regime_logprobs,
            prev_and_curr_regime_cond_obs_logprobs,
            predicted_prev_and_curr_regime_and_obs_logprobs,
            filtered_prev_and_curr_regime_logprobs):

        model = self.model

        k_regimes = model._k_regimes

        np.add(prev_and_curr_regime_cond_obs_logprobs,
                predicted_prev_and_curr_regime_logprobs,
                out=predicted_prev_and_curr_regime_and_obs_logprobs)

        obs_loglikelihood = \
                logsumexp(predicted_prev_and_curr_regime_and_obs_logprobs)

        self.obs_loglikelihoods[t] = obs_loglikelihood

        # Condition to avoid -np.inf - (-np.inf) operation
        if obs_loglikelihood != -np.inf:
            np.subtract(predicted_prev_and_curr_regime_and_obs_logprobs,
                    obs_loglikelihood,
                    out=filtered_prev_and_curr_regime_logprobs)
        else:
            filtered_prev_and_curr_regime_logprobs[:, :] = -np.inf

        self.filtered_regime_logprobs[t, :] = logsumexp(
                filtered_prev_and_curr_regime_logprobs, axis=0)

    def _regime_uncond_filtering(self, t,
            filtered_prev_and_curr_regime_probs, state_batteries,
            weighted_state_batteries, state_cov_batteries,
            weighted_state_cov_batteries):

        model = self.model

        k_regimes = model._k_regimes

        np.multiply(filtered_prev_and_curr_regime_probs.reshape(k_regimes,
                k_regimes, 1), state_batteries, out=weighted_state_batteries)

        np.sum(weighted_state_batteries, axis=(0, 1),
                out=self.filtered_states[t, :])

        np.multiply(filtered_prev_and_curr_regime_probs.reshape(k_regimes,
                k_regimes, 1, 1), state_cov_batteries,
                out=weighted_state_cov_batteries)

        np.sum(weighted_state_cov_batteries, axis=(0, 1),
                out=self.filtered_state_covs[t, :, :])

    def _approximation_step(self, t, curr_regime,
            filtered_prev_and_curr_regime_logprobs,
            filtered_prev_cond_on_curr_regime_logprobs,
            state_batteries, weighted_states, state_biases,
            state_bias_sqrs, state_cov_batteries,
            state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs, approx_state_cov):

        model = self.model

        k_regimes = model._k_regimes
        k_states = model._k_states

        curr_filter = model._kfilters[curr_regime]

        approx_state = np.asarray(curr_filter.filtered_state[:, t])

        if self.filtered_regime_logprobs[t, curr_regime] == -np.inf:

            # Any value would be alright, since it is multiplied by zero weight
            # in the next iteration
            approx_state[:] = 0

            approx_state_cov = \
                    np.asarray(curr_filter.filtered_state_cov[:, :, t])

            approx_state_cov[:, :] = 0

            return

        np.subtract(filtered_prev_and_curr_regime_logprobs[:, curr_regime],
                self.filtered_regime_logprobs[t, curr_regime],
                out=filtered_prev_cond_on_curr_regime_logprobs)

        filtered_prev_cond_on_curr_regime_probs = \
                filtered_prev_cond_on_curr_regime_logprobs

        np.exp(filtered_prev_cond_on_curr_regime_logprobs,
                out=filtered_prev_cond_on_curr_regime_probs)

        np.multiply(filtered_prev_cond_on_curr_regime_probs.reshape(-1, 1),
                state_batteries[:, curr_regime, :], out=weighted_states)

        np.sum(weighted_states, axis=0, out=approx_state)

        np.subtract(approx_state, state_batteries[:, curr_regime, :],
                out=state_biases)

        for i in range(k_regimes):
            np.outer(state_biases[i], state_biases[i],
                    out=state_bias_sqrs[i])

        np.add(state_cov_batteries[:, curr_regime, :, :], state_bias_sqrs,
                out=state_covs_and_state_bias_sqrs)

        np.multiply(filtered_prev_cond_on_curr_regime_probs.reshape(-1, 1,
                1), state_covs_and_state_bias_sqrs,
                out=weighted_state_covs_and_state_bias_sqrs)

        # It turns out that I can't just pass
        # np.asarray(curr_filter.filtered_state_cov[:, :, t]) to np.sum out,
        # because it leads to unexpected results.
        # I spent some time to figure this out.
        np.sum(weighted_state_covs_and_state_bias_sqrs, axis=0,
                out=approx_state_cov)

        np.copyto(np.asarray(curr_filter.filtered_state_cov[:, :, t]),
                approx_state_cov)

    def __call__(self):

        model = self.model

        k_endog = model._k_endog
        k_states = model._k_states
        k_regimes = model._k_regimes
        dtype = model._dtype
        nobs = model._nobs

        self.obs_loglikelihoods = np.zeros((nobs,), dtype=dtype)

        self.filtered_regime_logprobs = np.zeros((nobs, k_regimes),
                dtype=dtype)
        self.predicted_regime_logprobs = np.zeros((nobs, k_regimes),
                dtype=dtype)

        self.filtered_states = np.zeros((nobs, k_states), dtype=dtype)
        self.filtered_state_covs = np.zeros((nobs, k_states, k_states),
                dtype=dtype)

        if not hasattr(model, '_initial_regime_probs'):
            try:
                model.initialize_stationary_regime_probs()
            except RuntimeError:
                model.initialize_uniform_regime_probs()

        # Allocation of buffers

        state_buffer = np.zeros((k_states,), dtype=dtype)
        state_cov_buffer = np.zeros((k_states, k_states), dtype=dtype)

        state_batteries = np.zeros((k_regimes, k_regimes, k_states),
                dtype=dtype)
        state_cov_batteries = np.zeros((k_regimes, k_regimes, k_states,
                k_states), dtype=dtype)

        predicted_prev_and_curr_regime_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        prev_and_curr_regime_cond_obs_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        predicted_prev_and_curr_regime_and_obs_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        filtered_prev_and_curr_regime_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)

        filtered_prev_cond_on_curr_regime_logprobs = np.zeros((k_regimes,),
                dtype=dtype)
        weighted_states = np.zeros((k_regimes, k_states), dtype=dtype)
        state_biases = np.zeros((k_regimes, k_states), dtype=dtype)
        state_bias_sqrs = np.zeros((k_regimes, k_states, k_states),
                dtype=dtype)
        state_covs_and_state_bias_sqrs = np.zeros((k_regimes, k_states,
                k_states), dtype=dtype)
        weighted_state_covs_and_state_bias_sqrs = np.zeros((k_regimes,
                k_states, k_states), dtype=dtype)
        approx_state_cov = np.zeros((k_states, k_states), dtype=dtype)

        weighted_state_batteries = np.zeros((k_regimes, k_regimes, k_states),
                dtype=dtype)
        weighted_state_cov_batteries = np.zeros((k_regimes, k_regimes,
                k_states, k_states), dtype=dtype)

        for t in range(nobs):

            # Hamilton prediction
            self._hamilton_prediction_step(t,
                    predicted_prev_and_curr_regime_logprobs)

            # Kalman filter
            for prev_regime in range(k_regimes):
                for curr_regime in range(k_regimes):
                    # This condition optimizes calculation time in case of
                    # sparse regime transition  matrix (as it happens in MS AR)
                    if predicted_prev_and_curr_regime_logprobs[prev_regime, \
                            curr_regime] != -np.inf:
                        self._kalman_filter_step(t, prev_regime, curr_regime,
                                state_buffer, state_cov_buffer, state_batteries,
                                state_cov_batteries,
                                prev_and_curr_regime_cond_obs_logprobs)

            # Hamilton filter
            self._hamilton_filtering_step(t,
                    predicted_prev_and_curr_regime_logprobs,
                    prev_and_curr_regime_cond_obs_logprobs,
                    predicted_prev_and_curr_regime_and_obs_logprobs,
                    filtered_prev_and_curr_regime_logprobs)

            # Approximation
            for curr_regime in range(k_regimes):
                self._approximation_step(t, curr_regime,
                        filtered_prev_and_curr_regime_logprobs,
                        filtered_prev_cond_on_curr_regime_logprobs,
                        state_batteries, weighted_states, state_biases,
                        state_bias_sqrs, state_cov_batteries,
                        state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs,
                        approx_state_cov)

            filtered_prev_and_curr_regime_probs = \
                    filtered_prev_and_curr_regime_logprobs
            np.exp(filtered_prev_and_curr_regime_logprobs,
                    out=filtered_prev_and_curr_regime_probs)

            # Collecting filtering results
            self._regime_uncond_filtering(t,
                    filtered_prev_and_curr_regime_probs, state_batteries,
                    weighted_state_batteries, state_cov_batteries,
                    weighted_state_cov_batteries)


class KimFilter(SwitchingRepresentation):
    '''
    Kim Filter
    '''

    def __init__(self, k_endog, k_states, k_regimes, loglikelihood_burn=0,
            results_class=None, **kwargs):

        self._loglikelihood_burn = loglikelihood_burn

        if results_class is not None:
            self._results_class = results_class
        else:
            self._results_class = KimFilterResults

        super(KimFilter, self).__init__(k_endog, k_states, k_regimes, **kwargs)

    def filter(self, results=None, **kwargs):

        kfilter = _KimFilter(self, **kwargs)
        kfilter()

        if results is None:
            results = self._results_class

        if isinstance(results, type):
            if not issubclass(results, KimFilterResults):
                raise ValueError('Invalid results type.')
            results = results(self)
            results.update_representation(self)

        results.update_filter(kfilter)

        return results

    def loglikeobs(self, loglikelihood_burn=0, **kwargs):

        kfilter = _KimFilter(self, **kwargs)
        kfilter()

        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        loglikelihoods = np.array(kfilter.obs_loglikelihoods)
        loglikelihoods[:loglikelihood_burn] = 0

        return loglikelihoods

    def loglike(self, loglikelihood_burn=0, **kwargs):

        kfilter = _KimFilter(self, **kwargs)
        kfilter()

        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        return kfilter.obs_loglikelihoods[loglikelihood_burn:].sum()


class KimFilterResults(FrozenSwitchingRepresentation):

    _filter_attributes = ['loglikelihood_burn', 'initial_regime_logprobs',
            'obs_loglikelihoods', 'filtered_states', 'filtered_state_covs',
            'filtered_regime_logprobs', 'predicted_regime_logprobs']

    _attributes = FrozenSwitchingRepresentation._attributes + \
            _filter_attributes

    def update_representation(self, model):

        super(KimFilterResults, self).update_representation(model)

        self.loglikelihood_burn = model._loglikelihood_burn
        self.initial_regime_logprobs = model._initial_regime_logprobs

    def update_filter(self, kfilter):

        self.obs_loglikelihoods = kfilter.obs_loglikelihoods

        self.filtered_states = kfilter.filtered_states
        self.filtered_state_covs = kfilter.filtered_state_covs

        self.filtered_regime_logprobs = kfilter.filtered_regime_logprobs
        self.predicted_regime_logprobs = kfilter.predicted_regime_logprobs

    @property
    def filtered_regime_probs(self):

        return np.exp(self.filtered_regime_logprobs)

    @property
    def predicted_regime_probs(self):

        return np.exp(self.predicted_regime_logprobs)

    def loglikeobs(self, loglikelihood_burn=0):

        loglikelihood_burn = max(loglikelihood_burn, self.loglikelihood_burn)

        loglikelihoods = np.array(self.obs_loglikelihoods)
        loglikelihoods[:loglikelihood_burn] = 0

        return loglikelihoods

    def loglike(self, loglikelihood_burn=0):

        loglikelihood_burn = max(loglikelihood_burn, self.loglikelihood_burn)

        return self.obs_loglikelihoods[loglikelihood_burn:].sum()
