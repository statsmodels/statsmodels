import numpy as np
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

try:
    from scipy.stats.multivariate_normal import pdf as multivariate_normal_pdf
except ImportError:
    def multivariate_normal_pdf(x, mean=mean, cov=cov):
        n = x.shape[0]
        if np.linalg.matrix_rank(cov) < n:
            raise RuntimeError('Cov matrix is singular.')
        return np.power(np.pi, -0.5*n)/np.sqrt(np.linalg.det(cov))*\
                np.exp(-0.5*(x - mean).T.dot(np.linalg.inv(cov)).dot(x - mean))

class KimFilter(object):
    '''
    Kim Filter
    '''

    def __init__(self, k_endog, k_states, k_regimes, dtype=np.float64,
            loglikelihood_burn=0, designs=None, obs_intercepts=None,
            obs_covs=None, transitions=None, state_intercepts=None,
            selections=None, state_covs=None, regime_switch_probs=None,
            **kwargs):

        if k_regimes < 1:
            raise ValueError('Only multiple regimes are available.'
                    'Consider using regular KalmanFilter.')

        self._k_endog = k_endog
        self._k_states = k_states
        self._k_regimes = k_regimes
        self._dtype = dtype
        self._loglikelihood_burn = loglikelihood_burn

        designs = self._prepare_data_for_regimes(designs, 3)
        obs_intercepts = self._prepare_data_for_regimes(obs_intercepts, 2)
        obs_covs = self._prepare_data_for_regimes(obs_covs, 3)
        transitions = self._prepare_data_for_regimes(transitions, 3)
        state_intercepts = self._prepare_data_for_regimes(state_intercepts, 2)
        selections = self._prepare_data_for_regimes(selections, 3)
        state_covs = self._prepare_data_for_regimes(state_covs, 3)

        # Kalman filters for each regime

        kwargs['alternate_timing'] = True
        self._regime_kalman_filters = [KalmanFilter(k_endog, k_states,
                dtype=dtype, loglikelihood_burn=loglikelihood_burn,
                design=designs[i], obs_intercept=obs_intercepts[i],
                obs_cov=obs_covs[i], transition=transitions[i],
                state_intercept=state_intercepts[i], selection=selections[i],
                state_cov=state_covs[i], **kwargs) for i in range(k_regimes)]

        if regime_switch_probs is not None:
            regime_switch_probs = np.asarray(regime_switch_probs, dtype=dtype)
            if regime_switch_probs.shape != (k_regimes, k_regimes):
                raise ValueError('Regime switching matrix should have shape'
                        ' (k_regimes, k_regimes)')
            if not self._is_right_stochastic(regime_switch_probs):
                raise ValueError(
                        'Provided regime switching matrix is not stochastic')
            self._regime_switch_probs = regime_switch_probs
        else:
            self._regime_switch_probs = np.identity(k_regimes, dtype=dtype)

    def _prepare_data_for_regimes(self, data, per_regime_dims):

        k_regimes = self._k_regimes

        if data is None:
            return [None for _ in range(k_regimes)]

        data = np.asarray(data, dtype=self._dtype)

        if len(data.shape) == per_regime_dims:
            return [data for _ in range(k_regimes)]

        if data.shape[0] != k_regimes:
            raise ValueError('First dimension is not k_regimes')

        return data

    def _is_right_stochastic(self, matrix):

        if np.any(matrix < 0):
            return False
        if not np.all(matrix.sum(axis=1) == 1):
            return False
        return True

    @property
    def regime_filters(self):
        return self._regime_kalman_filters

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

    def initialize_known(self, initial_states, initial_state_covs):

        k_regimes = self._k_regimes
        regime_filters = self._regime_kalman_filters

        initial_states = self._prepare_data_for_regimes(initial_states, 1)
        initial_state_covs = self._prepare_data_for_regimes(initial_state_covs,
                2)

        for i in range(k_regimes):
            regime_filters[i].initialize_known(initial_states[i],
                    initial_state_covs[i])

    def initialize_stationary(self):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.initialize_stationary()

    def initialize_known_regime_probs(self, initial_regime_probs):

        self._initial_regime_probs = initial_regime_probs

    def initialize_uniform_regime_probs(self):

        k_regimes = self._k_regimes
        self._initial_regime_probs = np.ones((k_regimes,), dtype=self._dtype) \
                / k_regimes

    def initialize_stationary_regime_probs(self):

        eigenvalues, eigenvectors = np.linalg.eig(self._regime_switch_probs.T)
        one_eigenvalue_indices = np.where(eigenvalues == 1)[0]

        non_uniq_stat_distr_message = 'Regime switching chain doesn\'t have ' \
                'a unique stationary distribution'

        if one_eigenvalue_indices.shape[0] != 1:
            raise RuntimeError(non_uniq_stat_distr_message)

        candidate_eigenvector = eigenvectors[:, one_eigenvalue_indices[0]]

        if not np.all(candidate_eigenvector >= 0):
            raise RuntimeError(non_uniq_stat_distr_message)

        self._initial_regime_probs = candidate_eigenvector / \
                np.linalg.norm(candidate_eigenvector)

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

    def _hamilton_filter_step(self, t, filtered_curr_regime_probs,
            predicted_prev_and_curr_regime_probs,
            forecast_error_batteries,
            forecast_error_cov_batteries,
            forecast_error_mean,
            prev_and_curr_regime_cond_obs_probs,
            predicted_obs_prev_and_curr_regime_probs,
            filtered_prev_and_curr_regime_probs):

        k_regimes = self._k_regimes

        if t == 0:
            regime_probs = self._initial_regime_probs
        else:
            regime_probs = filtered_curr_regime_probs

        np.multiply(self._regime_switch_probs,
                regime_probs.reshape(-1, 1),
                out=predicted_prev_and_curr_regime_probs)

        for prev_regime in range(k_regimes):
            for curr_regime in range(k_regimes):
                forecast_error = forecast_error_batteries[prev_regime,
                        curr_regime, :]
                forecast_error_cov = forecast_error_cov_batteries[prev_regime,
                        curr_regime, :, :]
                # Should I manage memory allocation here?
                prev_and_curr_regime_cond_obs_probs[prev_regime,
                        curr_regime] = multivariate_normal_pdf(forecast_error,
                        mean=forecast_error_mean, cov=forecast_error_cov)

        np.multiply(predicted_prev_and_curr_regime_probs,
                prev_and_curr_regime_cond_obs_probs,
                out=predicted_obs_prev_and_curr_regime_probs)

        obs_likelihood = predicted_obs_prev_and_curr_regime_probs.sum()

        self._obs_likelihoods[t] = obs_likelihood

        np.divide(predicted_obs_prev_and_curr_regime_probs, obs_likelihood,
                out=filtered_prev_and_curr_regime_probs)

        np.sum(filtered_prev_and_curr_regime_probs, axis=0,
                out=filtered_curr_regime_probs)

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
            weighted_states, weighted_states_sum, filtered_curr_regime_probs,
            state_biases, transposed_state_biases, state_bias_sqrs,
            state_cov_batteries, state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs_sum):

        k_states = self._k_states

        curr_filter = self._kfilters[curr_regime]

        np.multiply(filtered_prev_and_curr_regime_probs[:,
                curr_regime].reshape(-1, 1),
                state_batteries[:, curr_regime, :], out=weighted_states)

        np.sum(weighted_states, axis=0, out=weighted_states_sum)

        approx_state = np.asarray(curr_filter.filtered_state[:, t])

        np.divide(weighted_states_sum,
                filtered_curr_regime_probs[curr_regime],
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
                filtered_curr_regime_probs[curr_regime],
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

        self._obs_likelihoods = np.zeros((nobs,), dtype=dtype)

        self._filtered_states = np.zeros((nobs, k_states), dtype=dtype)
        self._filtered_state_covs = np.zeros((nobs, k_states, k_states),
                dtype=dtype)

        if not hasattr(self, '_initial_regime_probs'):
            try:
                self.initialize_stationary_regime_probs()
            except RuntimeError:
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

        filtered_curr_regime_probs = np.zeros((k_regimes,), dtype=dtype)
        predicted_prev_and_curr_regime_probs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)
        prev_and_curr_regime_cond_obs_probs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)
        predicted_obs_prev_and_curr_regime_probs = np.zeros((k_regimes,
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
            self._hamilton_filter_step(t, filtered_curr_regime_probs,
                    predicted_prev_and_curr_regime_probs,
                    forecast_error_batteries, forecast_error_cov_batteries,
                    forecast_error_mean, prev_and_curr_regime_cond_obs_probs,
                    predicted_obs_prev_and_curr_regime_probs,
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
                        weighted_states, weighted_states_sum,
                        filtered_curr_regime_probs, state_biases,
                        transposed_state_biases, state_bias_sqrs,
                        state_cov_batteries, state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs_sum)

    def loglike(self, loglikelihood_burn=0):

        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        return np.log(self._obs_likelihoods[loglikelihood_burn:]).sum()

    def loglikeobs(self, loglikelihood_burn=0):

        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        loglikelihoods = np.log(self._obs_likelihoods)
        loglikelihoods[:loglikelihood_burn] = 0

        return loglikelihoods

    @property
    def initial_regime_probs():
        return self._initial_regime_probs

    @property
    def filtered_states(self):

        return self._filtered_states

    @property
    def filtered_state_covs(self):

        return self._filtered_state_covs
