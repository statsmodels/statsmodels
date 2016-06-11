import numpy as np
from scipy.misc import logsumexp
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

try:
    from scipy.stats import multivariate_normal
    multivariate_normal_logpdf = multivariate_normal.logpdf
except ImportError:
    def multivariate_normal_logpdf(x, mean=None, cov=None):
        n = x.shape[0]
        x = x.reshape(-1, 1)
        mean = x.reshape(-1, 1)
        if np.linalg.matrix_rank(cov) < n:
            raise RuntimeError('Cov matrix is singular.')
        #TODO: to test this
        logpdf = -0.5 * n * np.log(np.pi) - \
                0.5 * np.log(np.linalg.det(cov)) - \
                0.5 * (x - mean).T.dot(np.linalg.inv(cov)).dot(x - mean)
        return logpdf

class KimFilter(object):
    '''
    Kim Filter
    '''

    def __init__(self, k_endog, k_states, k_regimes, dtype=np.float64,
            loglikelihood_burn=0, design=None, obs_intercept=None,
            obs_cov=None, transition=None, state_intercept=None,
            selection=None, state_cov=None, regime_transition=None,
            **kwargs):

        if k_regimes < 1:
            raise ValueError('Only multiple regimes are available.'
                    'Consider using regular KalmanFilter.')

        self._k_endog = k_endog
        self._k_states = k_states
        self._k_regimes = k_regimes
        self._dtype = dtype
        self._loglikelihood_burn = loglikelihood_burn

        self.per_regime_dims = {
                'design': 3,
                'obs_intercept': 2,
                'obs_cov': 3,
                'transition': 3,
                'state_intercept': 2,
                'selection': 3,
                'state_cov': 3
        }

        design = self._prepare_data_for_regimes(design,
                self.per_regime_dims['design'])
        obs_intercept = self._prepare_data_for_regimes(obs_intercept,
                self.per_regime_dims['obs_intercept'])
        obs_cov = self._prepare_data_for_regimes(obs_cov,
                self.per_regime_dims['obs_cov'])
        transition = self._prepare_data_for_regimes(transition,
                self.per_regime_dims['transition'])
        state_intercept = self._prepare_data_for_regimes(state_intercept,
                self.per_regime_dims['state_intercept'])
        selection = self._prepare_data_for_regimes(selection,
                self.per_regime_dims['selection'])
        state_cov = self._prepare_data_for_regimes(state_cov,
                self.per_regime_dims['state_cov'])

        # Kalman filters for each regime

        kwargs['alternate_timing'] = True
        self._regime_kalman_filters = [KalmanFilter(k_endog, k_states,
                dtype=dtype, loglikelihood_burn=loglikelihood_burn,
                design=design[i], obs_intercept=obs_intercept[i],
                obs_cov=obs_cov[i], transition=transition[i],
                state_intercept=state_intercept[i], selection=selection[i],
                state_cov=state_cov[i], **kwargs) for i in range(k_regimes)]

        self.set_regime_transition(regime_transition)

    def set_regime_transition(self, regime_transition):

        dtype = self._dtype
        k_regimes = self._k_regimes

        if regime_transition is not None:
            regime_transition = np.asarray(regime_transition, dtype=dtype)
            if regime_transition.shape != (k_regimes, k_regimes):
                raise ValueError('Regime transition matrix should have shape'
                        ' (k_regimes, k_regimes)')
            if not self._is_left_stochastic(regime_transition):
                raise ValueError(
                        'Provided regime transition matrix is not stochastic')
            self._regime_transition = regime_transition
        else:
            self._regime_transition = np.identity(k_regimes, dtype=dtype)

    def _prepare_data_for_regimes(self, data, per_regime_dims):
        '''
        This duplicates representation matrix for every regime, if it's
        provided in the only example.
        '''

        k_regimes = self._k_regimes

        if data is None:
            return [None for _ in range(k_regimes)]

        data = np.asarray(data, dtype=self._dtype)

        if len(data.shape) == per_regime_dims:
            return [data for _ in range(k_regimes)]

        if data.shape[0] != k_regimes:
            raise ValueError('First dimension is not k_regimes')

        return data

    def __getitem__(self, key):

        if key == 'regime_transition':
            return self._regime_transition

        if key not in self.per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % key)

        return np.asarray([regime_filter[key] for regime_filter in \
                self._regime_kalman_filters])

    def __setitem__(self, key, value):

        if key == 'regime_transition':
            self.set_regime_transition(value)
            return

        if key not in self.per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % key)

        value = self._prepare_data_for_regimes(value,
                self.per_regime_dims[key])

        for regime_filter, regime_value in zip(self._regime_kalman_filters,
                value):
            regime_filter[key] = regime_value

    def _is_left_stochastic(self, matrix):

        if np.any(matrix < 0):
            return False
        if not np.all(matrix.sum(axis=0) == 1):
            return False
        return True

    @property
    def k_regimes(self):
        return self._k_regimes

    @property
    def k_endog(self):
        return self._k_endog

    @property
    def dtype(self):
        return self._dtype

    @property
    def regime_filters(self):
        return self._regime_kalman_filters

    @property
    def initialization(self):
        return self.regime_filters[0].initialization

    @property
    def initial_variance(self):
        return self.regime_filters[0].initial_variance

    @initial_variance.setter
    def initial_variance(self, value):
        for regime_filter in self.regime_filters:
            regime_filter.initial_variance = value

    @property
    def loglikelihood_burn(self):
        return self._loglikelihood_burn

    @loglikelihood_burn.setter
    def loglikelihood_burn(self, value):
        self._loglikelihood_burn = value
        for regime_filter in self.regime_filters:
            regime_filter.loglikelihood_burn = value

    @property
    def tolerance(self):
        return self.regime_filters[0].tolerance

    @tolerance.setter
    def tolerance(self, value):
        for regime_filter in self.regime_filters:
            regime_filter.tolerance = value

    @property
    def _complex_endog(self):
        return self.regime_filters[0]._complex_endog

    def _initialize_filter(self, **kwargs):

        kwargs['filter_timing'] = 1

        for regime_filter in self._regime_kalman_filters:
            regime_filter._initialize_filter(**kwargs)

    def set_filter_method(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_filter_method(**kwargs)

    def set_inversion_method(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_inversion_method(**kwargs)

    def set_stability_method(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_stability_method(**kwargs)

    def set_conserve_memory(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_conserve_memory(**kwargs)

    def bind(self, endog):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.bind(endog)

        self._nobs = self._regime_kalman_filters[0].nobs

    def initialize_known(self, initial_state, initial_state_cov):

        k_regimes = self._k_regimes
        regime_filters = self._regime_kalman_filters

        initial_state = self._prepare_data_for_regimes(initial_state, 1)
        initial_state_cov = self._prepare_data_for_regimes(initial_state_cov, 2)

        for i in range(k_regimes):
            regime_filters[i].initialize_known(initial_state[i],
                    initial_state_cov[i])

    def initialize_stationary(self):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.initialize_stationary()

    def initialize_known_regime_probs(self, initial_regime_probs):
        '''
        Initialization of marginal regime distribution at t=0.
        '''

        self._initial_regime_probs = initial_regime_probs

    def initialize_uniform_regime_probs(self):
        '''
        Initialization of marginal regime distribution at t=0.
        '''

        k_regimes = self._k_regimes
        self._initial_regime_probs = np.ones((k_regimes,), dtype=self._dtype) \
                / k_regimes

    def initialize_stationary_regime_probs(self):
        '''
        Initialization of marginal regime distribution at t=0.
        '''

        k_regimes = self._k_regimes
        dtype = self._dtype

        constraint_matrix = np.vstack((self._regime_transition - \
                np.identity(k_regimes, dtype=dtype),
                np.ones((1, k_regimes), dtype=dtype)))

        candidate = np.linalg.pinv(constraint_matrix)[:, -1]

        if np.any(candidate < 0):
            raise RuntimeError('Regime switching chain doesn\'t have ' \
                'a stationary distribution')

        self._initial_regime_probs = candidate

    def _initialize_filters(self, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):

        kfilters = []

        for regime_filter in self._regime_kalman_filters:
            prefix = regime_filter._initialize_filter(
                    filter_method=filter_method,
                    inversion_method=inversion_method,
                    stability_method=stability_method,
                    conserve_memory=conserve_memory, tolerance=tolerance)[0]
            kfilters.append(regime_filter._kalman_filters[prefix])
            regime_filter._initialize_state(prefix=prefix,
                    complex_step=complex_step)

        self._kfilters = kfilters

    def _kalman_filter_step(self, t, prev_regime, curr_regime, state_buffer,
            state_cov_buffer, state_batteries, state_cov_batteries,
            forecast_error_batteries, forecast_error_cov_batteries):

        curr_kfilter = self._kfilters[curr_regime]
        prev_kfilter = self._kfilters[prev_regime]
        curr_regime_filter = \
                self._regime_kalman_filters[curr_regime]
        prev_regime_filter = self._regime_kalman_filters[prev_regime]
        curr_kfilter.seek(t)

        if t == 0:
            state_buffer = curr_regime_filter._initial_state
            state_cov_buffer = curr_regime_filter._initial_state_cov
            curr_regime_filter._initial_state = prev_regime_filter._initial_state
            curr_regime_filter._initial_state_cov = \
                    prev_regime_filter._initial_state_cov
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
            curr_regime_filter._initial_state = state_buffer
            curr_regime_filter._initial_state_cov = state_cov_buffer
        else:
            np.copyto(np.asarray(curr_kfilter.filtered_state[:, t - 1]),
                    state_buffer)
            np.copyto(np.asarray(curr_kfilter.filtered_state_cov[:, :, t - 1]),
                    state_cov_buffer)

        np.copyto(state_batteries[prev_regime, curr_regime, :],
                curr_kfilter.filtered_state[:, t])
        np.copyto(state_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.filtered_state_cov[:, :, t])
        np.copyto(forecast_error_batteries[prev_regime, curr_regime, :],
                curr_kfilter.forecast_error[:, t])
        np.copyto(forecast_error_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.forecast_error_cov[:, :, t])

    def _hamilton_filter_step(self, t, predicted_prev_and_curr_regime_probs,
            forecast_error_batteries,
            forecast_error_cov_batteries,
            forecast_error_mean,
            prev_and_curr_regime_cond_obs_logprobs,
            regimes_cond_obs_logprobs_minus_uncond_logprob,
            filtered_prev_and_curr_regime_probs):

        k_regimes = self._k_regimes

        if t == 0:
            regime_probs = self._initial_regime_probs
        else:
            regime_probs = self._filtered_regime_probs[t - 1, :]

        np.multiply(self._regime_transition.transpose(),
                regime_probs.reshape(-1, 1),
                out=predicted_prev_and_curr_regime_probs)

        # This is used in smoothing
        np.sum(predicted_prev_and_curr_regime_probs, axis=0,
                out=self._predicted_regime_probs[t, :])

        for prev_regime in range(k_regimes):
            for curr_regime in range(k_regimes):
                forecast_error = forecast_error_batteries[prev_regime,
                        curr_regime, :]
                forecast_error_cov = forecast_error_cov_batteries[prev_regime,
                        curr_regime, :, :]

                # Should I manage memory allocation here?
                prev_and_curr_regime_cond_obs_logprobs[prev_regime,
                        curr_regime] = multivariate_normal_logpdf(forecast_error,
                        mean=forecast_error_mean, cov=forecast_error_cov)

        obs_loglikelihood = logsumexp(prev_and_curr_regime_cond_obs_logprobs,
                b=predicted_prev_and_curr_regime_probs)

        self._obs_loglikelihoods[t] = obs_loglikelihood

        np.subtract(prev_and_curr_regime_cond_obs_logprobs, obs_loglikelihood,
                out=regimes_cond_obs_logprobs_minus_uncond_logprob)

        regimes_cond_obs_probs_divided_by_uncond_prob = \
                regimes_cond_obs_logprobs_minus_uncond_logprob

        np.exp(regimes_cond_obs_logprobs_minus_uncond_logprob,
                out=regimes_cond_obs_probs_divided_by_uncond_prob)

        np.multiply(regimes_cond_obs_probs_divided_by_uncond_prob,
                predicted_prev_and_curr_regime_probs,
                out=filtered_prev_and_curr_regime_probs)

        np.sum(filtered_prev_and_curr_regime_probs, axis=0,
                out=self._filtered_regime_probs[t, :])

    def _regime_uncond_filtering(self, t,
            filtered_prev_and_curr_regime_probs, state_batteries,
            weighted_state_batteries, state_cov_batteries,
            weighted_state_cov_batteries):

        k_regimes = self._k_regimes

        np.multiply(filtered_prev_and_curr_regime_probs.reshape(k_regimes,
                k_regimes, 1), state_batteries, out=weighted_state_batteries)

        np.sum(weighted_state_batteries, axis=(0, 1),
                out=self._filtered_states[t, :])

        np.multiply(filtered_prev_and_curr_regime_probs.reshape(k_regimes,
                k_regimes, 1, 1), state_cov_batteries,
                out=weighted_state_cov_batteries)

        np.sum(weighted_state_cov_batteries, axis=(0, 1),
                out=self._filtered_state_covs[t, :, :])

    def _approximation_step(self, t, curr_regime,
            filtered_prev_and_curr_regime_probs, state_batteries,
            weighted_states, weighted_states_sum, state_biases,
            transposed_state_biases, state_bias_sqrs,
            state_cov_batteries, state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs_sum):

        k_states = self._k_states

        curr_filter = self._kfilters[curr_regime]

        approx_state = np.asarray(curr_filter.filtered_state[:, t])

        # Should be compared by eps?
        if self._filtered_regime_probs[t, curr_regime] == 0:

            # Any value would be alright, since it is multiplied by zero weight
            # in the next iteration
            approx_state[:] = 0

            approx_state_cov = \
                    np.asarray(curr_filter.filtered_state_cov[:, :, t])

            approx_state_cov[:, :] = 0

            return

        np.multiply(filtered_prev_and_curr_regime_probs[:,
                curr_regime].reshape(-1, 1),
                state_batteries[:, curr_regime, :], out=weighted_states)

        np.sum(weighted_states, axis=0, out=weighted_states_sum)

        np.divide(weighted_states_sum,
                self._filtered_regime_probs[t, curr_regime],
                out=approx_state)

        np.subtract(approx_state.reshape(1, -1, 1),
                state_batteries[:, curr_regime, :].reshape(-1, k_states, 1),
                out=state_biases)

        np.subtract(approx_state.reshape(1, 1, -1),
                state_batteries[:, curr_regime, :].reshape(-1, 1, k_states),
                out=transposed_state_biases)

        np.multiply(state_biases, transposed_state_biases, out=state_bias_sqrs)

        np.add(state_cov_batteries[:, curr_regime, :, :], state_bias_sqrs,
                out=state_covs_and_state_bias_sqrs)

        np.multiply(filtered_prev_and_curr_regime_probs[:,
                curr_regime].reshape(-1, 1, 1), state_covs_and_state_bias_sqrs,
                out=weighted_state_covs_and_state_bias_sqrs)

        np.sum(weighted_state_covs_and_state_bias_sqrs, axis=0,
                out=weighted_state_covs_and_state_bias_sqrs_sum)

        np.divide(weighted_state_covs_and_state_bias_sqrs_sum,
                self._filtered_regime_probs[t, curr_regime],
                out=np.asarray(curr_filter.filtered_state_cov[:, :, t]))

    def filter(self, **kwargs):

        k_endog = self._k_endog
        k_states = self._k_states
        k_regimes = self._k_regimes
        dtype = self._dtype

        if not hasattr(self, '_nobs'):
            raise RuntimeError(
                    'No endog data binded. Consider using bind() first')

        nobs = self._nobs

        self._initialize_filters(**kwargs)

        self._obs_loglikelihoods = np.zeros((nobs,), dtype=dtype)

        self._filtered_regime_probs = np.zeros((nobs, k_regimes), dtype=dtype)
        self._predicted_regime_probs = np.zeros((nobs, k_regimes), dtype=dtype)

        self._filtered_states = np.zeros((nobs, k_states), dtype=dtype)
        self._filtered_state_covs = np.zeros((nobs, k_states, k_states),
                dtype=dtype)

        if not hasattr(self, '_initial_regime_probs'):
            try:
                self.initialize_stationary_regime_probs()
            except RuntimeError:
                raise
                self.initialize_uniform_regime_probs()

        # Allocation of buffers

        state_buffer = np.zeros((k_states,), dtype=dtype)
        state_cov_buffer = np.zeros((k_states, k_states), dtype=dtype)

        state_batteries = np.zeros((k_regimes, k_regimes, k_states),
                dtype=dtype)
        state_cov_batteries = np.zeros((k_regimes, k_regimes, k_states,
                k_states), dtype=dtype)

        forecast_error_batteries = np.zeros((k_regimes, k_regimes, k_endog),
                dtype=dtype)
        forecast_error_cov_batteries = np.zeros((k_regimes, k_regimes,
                k_endog, k_endog), dtype=dtype)
        forecast_error_mean = np.zeros((k_endog,), dtype=dtype)

        predicted_prev_and_curr_regime_probs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)
        prev_and_curr_regime_cond_obs_logprobs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)
        regimes_cond_obs_logprobs_minus_uncond_logprob = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        filtered_prev_and_curr_regime_probs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)

        weighted_state_batteries = np.zeros((k_regimes, k_regimes, k_states),
                dtype=dtype)
        weighted_state_cov_batteries = np.zeros((k_regimes, k_regimes,
                k_states, k_states), dtype=dtype)

        weighted_states = np.zeros((k_regimes, k_states), dtype=dtype)
        weighted_states_sum = np.zeros((k_states,), dtype=dtype)
        state_biases = np.zeros((k_regimes, k_states, 1), dtype=dtype)
        transposed_state_biases = np.zeros((k_regimes, 1, k_states),
                dtype=dtype)
        state_bias_sqrs = np.zeros((k_regimes, k_states, k_states),
                dtype=dtype)
        state_covs_and_state_bias_sqrs = np.zeros((k_regimes, k_states,
                k_states), dtype=dtype)
        weighted_state_covs_and_state_bias_sqrs = np.zeros((k_regimes,
                k_states, k_states), dtype=dtype)
        weighted_state_covs_and_state_bias_sqrs_sum = np.zeros((k_states,
                k_states), dtype=dtype)

        for t in range(nobs):

            # Kalman filter
            for prev_regime in range(k_regimes):
                for curr_regime in range(k_regimes):
                    self._kalman_filter_step(t, prev_regime, curr_regime,
                            state_buffer, state_cov_buffer, state_batteries,
                            state_cov_batteries, forecast_error_batteries,
                            forecast_error_cov_batteries)

            # Hamilton filter
            self._hamilton_filter_step(t, predicted_prev_and_curr_regime_probs,
                    forecast_error_batteries, forecast_error_cov_batteries,
                    forecast_error_mean,
                    prev_and_curr_regime_cond_obs_logprobs,
                    regimes_cond_obs_logprobs_minus_uncond_logprob,
                    filtered_prev_and_curr_regime_probs)

            # Collecting filtering results
            self._regime_uncond_filtering(t,
                    filtered_prev_and_curr_regime_probs, state_batteries,
                    weighted_state_batteries, state_cov_batteries,
                    weighted_state_cov_batteries)

            # Approximation
            for curr_regime in range(k_regimes):
                self._approximation_step(t, curr_regime,
                        filtered_prev_and_curr_regime_probs, state_batteries,
                        weighted_states, weighted_states_sum, state_biases,
                        transposed_state_biases, state_bias_sqrs,
                        state_cov_batteries, state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs_sum)

    def get_smoothed_regime_probs(self, filter_first=True, **kwargs):
        '''
        this is tested in test_ms_ar.py

        p. 107 Kim-Nelson
        '''

        if filter_first:
            self.filter(**kwargs)

        dtype = self._dtype
        nobs = self._nobs
        k_regimes = self._k_regimes

        smoothed_regime_probs = np.zeros((nobs, k_regimes), dtype=dtype)
        smoothed_curr_and_next_regime_probs = np.zeros((nobs - 1, k_regimes,
                k_regimes), dtype=dtype)

        predicted_curr_and_next_regime_probs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)

        filtered_curr_regime_cond_on_next = np.zeros((k_regimes, k_regimes),
                dtype=dtype)

        predicted_next_regime_prob_is_zero = np.zeros((k_regimes,), dtype=bool)

        smoothed_regime_probs[-1, :] = self._filtered_regime_probs[-1, :]

        for t in range(nobs - 2, -1, -1):

            np.multiply(self._regime_transition.transpose(),
                    self._filtered_regime_probs[t, :].reshape(-1, 1),
                    out=predicted_curr_and_next_regime_probs)

            np.equal(self._predicted_regime_probs[t + 1, :], 0,
                    out=predicted_next_regime_prob_is_zero)

            # Division by zero warnings, if predicted regime prob is zero?
            np.divide(predicted_curr_and_next_regime_probs,
                    self._predicted_regime_probs[t + 1, :].reshape(1, -1),
                    out=filtered_curr_regime_cond_on_next)

            filtered_curr_regime_cond_on_next[:,
                    predicted_next_regime_prob_is_zero] = 0

            np.multiply(smoothed_regime_probs[t + 1, :].reshape(1, -1),
                    filtered_curr_regime_cond_on_next,
                    out=smoothed_curr_and_next_regime_probs[t, :, :])

            np.sum(smoothed_curr_and_next_regime_probs[t, :, :], axis=1,
                    out=smoothed_regime_probs[t, :])

        return (smoothed_regime_probs, smoothed_curr_and_next_regime_probs)

    def loglike(self, loglikelihood_burn=0, filter_first=True, **kwargs):

        if filter_first:
            self.filter(**kwargs)

        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        return self._obs_loglikelihoods[loglikelihood_burn:].sum()

    def loglikeobs(self, loglikelihood_burn=0, filter_first=True, **kwargs):

        if filter_first:
            self.filter(**kwargs)

        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        loglikelihoods = np.array(self._obs_loglikelihoods)
        loglikelihoods[:loglikelihood_burn] = 0

        return loglikelihoods

    @property
    def initial_regime_probs():
        '''
        Marginal regime distribution at t=0.
        '''

        return self._initial_regime_probs

    @property
    def filtered_states(self):

        return self._filtered_states

    @property
    def filtered_state_covs(self):

        return self._filtered_state_covs

    @property
    def filtered_regime_probs(self):

        return self._filtered_regime_probs

    @property
    def predicted_regime_probs(self):

        return self._predicted_regime_probs
