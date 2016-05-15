import numpy as np
from scipy.stats import multivariate_normal
from ..statespace.representation import Representation
from ..statespace.kalman_filter import KalmanFilter

class KimFilter(object):
    '''
    Kim Filter
    '''
    
    def __init__(self, k_endog, k_states, k_regimes, endog=None,
            regime_switch_probs=None, k_posdef=None, tolerance=1e-19,
            kalman_filter_classes=None, results_class=None, **kwargs):

        if k_regimes == 1:
            raise ValueError('Only multiple regimes are available.'
                    'Consider using regular KalmanFilter.')
        
        self._k_regimes = k_regimes

        if endog is not None:
            self.bind(endog)

        if regime_switch_probs is not None:
            if regime_switch_probs.shape != (k_regimes, k_regimes):
                raise ValueError('Regime switching matrix should have shape'
                        ' (k_regimes, k_regimes)')
            if not self._is_stochastic(regime_switch_probs):
                raise ValueError(
                        'Provided regime switching matrix is not stochastic')
            self._regime_switch_probs = regime_switch_probs
        else:
            self._regime_switch_probs = None

        # Kalman filters for each regime
        # Diffuse initialization ???
        self._regime_kalman_filters = [KalmanFilter(k_endog, k_states,
            k_posdef=k_posdef, tolerance=tolerance,
            kalman_filter_classes=kalman_filter_classes,
            **kwargs) for _ in range(k_regimes)]

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_filter_timing(alternate_timing=True)

        self.initial_regime_probs = None

    def _is_stochastic(self, matrix):
        
        if (matrix < 0).max() != False:
            return False
        if (matrix.sum(axis=1) == 1).min() == False:
            return False
        return True

    def _initialize_filter(self, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None):
        
        for regime_filter in self._regime_kalman_filters:
            regime_filter._initialize_filter(filter_method=filter_method,
                    inversion_method=inversion_method,
                    stability_method=stability_method,
                    conserve_memory=conserve_memory, tolerance=tolerance,
                    filter_timing=1)

    def set_filter_method(self, filter_method=None, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_filter_method(filter_method=filter_method,
                    **kwargs)

    def set_inversion_method(self, inversion_method=None, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_inversion_method(
                    inversion_method=inversion_method, **kwargs)

    def set_stability_method(self, stability_method=None, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_stability_method(
                    stability_method=stability_method, **kwargs)

    def set_conserve_memory(self, conserve_memory=None, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_conserve_memory(conserve_memory=conserve_memory,
                    **kwargs)

    def bind(self, endog):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.bind(endog)

        self._endog = endog
        self._nobs = endog.shape[1]

    def initialize_known(self, initial_states, initial_state_covs):
        
        if (initial_states.shape[0] != self._k_regimes and \
                initial_states.shape[0] != 1) or \
                (initial_state_covs.shape[0] != self._k_regimes and \
                initial_state_covs.shape[0] != 1):
            raise ValueError(
                    'One or k_regimes initial parameters should be provided')

        regime_filters = self._regime_kalman_filters
        
        for i in range(self._k_regimes):
            regime_filters[i].initialize_known(initial_states[i],
                    initial_state_covs[i])

    def initialize_stationary(self):
        
        for regime_filter in self._regime_kalman_filters:
            regime_filter.initialize_stationary()
    
    def initialize_known_regime_probs(self, initial_regime_probs):
        
        self.initial_regime_probs = initial_regime_probs

    def initialize_stationary_regime_probs(self):
        #TODO:implement stationary distribution

        self.initial_regime_probs = np.ones((self._k_regimes,))/self._k_regimes

    def _initialize_filters(self, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):
        
        kfilters = []
        
        for regime_filter in self._regime_kalman_filters:
            prefix = regime_filter._initialize_filter(
                    filter_method=filter_method,
                    inversion_method=inversion_method, stability_method,
                    conserve_memory=conserve_memory, tolerance=tolerance,
                    filter_timing=1)[0]
            kfilters.append(regime_filter._kalman_filters[prefix])
            regime_filter._initialize_state(prefix=prefix,
                    complex_step=complex_step)
        
        return kfilters

    def _kalman_filter_step(self, t, kfilters, prev_regime, curr_regime,
            state_buffer, state_cov_buffer, state_batteries, 
            state_cov_batteries, forecast_error_batteries,
            forecast_error_cov_batteries):
        
        curr_kfilter = kfilters[curr_regime]
        prev_kfilter = kfilters[prev_regime]
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
            np.copyto(curr_kfilter.filtered_state[:, t - 1],
                    prev_kfilter.filtered_state[:, t - 1])
            np.copyto(curr_kfilter.filtered_state_cov[:, :, t - 1],
                    prev_kfilter.filtered_state_cov[:, :, t - 1])
        
        next(curr_kfilter)

        if t == 0:
            curr_regime_filter._initial_state = state_buffer
            curr_regime_filter._initial_state_cov = state_cov_buffer
        else:
            np.copyto(curr_kfilter.filtered_state[:, t - 1], state_buffer)
            np.copyto(curr_kfilter.filtered_state_cov[:, :, t - 1],
                    state_cov_buffer)
        
        np.copyto(state_batteries[prev_regime, curr_regime, :],
                curr_kfilter.filtered_state[:, t])
        np.copyto(state_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.filtered_state_cov[:, :, t])
        np.copyto(forecast_error_batteries[prev_regime, curr_regime, :],
                curr_kfilter.forecast_error[:, t])
        np.copyto(forecast_error_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.forecast_error_cov[:, :, t])

    def _hamilton_filter_step(self, t, kfilters, filtered_curr_regime_probs,
            predicted_prev_and_curr_regime_probs,
            forecast_error_batteries,
            forecast_error_cov_batteries, 
            prev_and_curr_regime_cond_obs_probs,
            predicted_obs_prev_and_curr_regime_probs,
            filtered_prev_and_curr_regime_probs):

        k_regimes = self._k_regimes

        if t == 0:
            regime_probs = self.initial_regime_probs
        else:
            regime_probs = filtered_curr_regime_probs
            
        np.multiply(self.regime_switch_probs,
                regime_probs.reshape(-1, 1),
                out=predicted_prev_and_curr_regime_probs)
        
        for prev_regime in range(k_regimes):
            for curr_regime in range(k_regimes):
                forecast_error = forecast_error_batteries[prev_regime,
                        curr_regime, :]
                forecast_error_cov = forecast_error_cov_batteries[prev_regime,
                        curr_regime, :, :]
                prev_and_curr_regime_cond_obs_probs[prev_regime,
                        curr_regime] = multivariate_normal.pdf(
                        mean=forecast_error,
                        cov=forecast_error_cov)
        
        np.multiply(predicted_prev_and_curr_regime_probs,
                prev_and_curr_regime_cond_obs_probs,
                out=predicted_obs_prev_and_curr_regime_probs)

        obs_likelihood = predicted_obs_prev_and_curr_regime_probs.sum()
        
        self.obs_likelihoods[t] = obs_likelihood
        
        np.divide(predicted_obs_prev_and_curr_regime_probs, obs_likelihood,
                out=filtered_prev_and_curr_regime_probs)
        
        np.sum(filtered_prev_and_curr_regime_probs, axis=0,
                out=filtered_curr_regime_probs)

    def _approximation_step(self, t, kfilters, curr_regime,
            filtered_prev_and_curr_regime_probs, state_batteries,
            weighted_states, weighted_states_sum, filtered_curr_regime_probs,
            state_biases, transposed_state_biases, state_bias_sqrs,
            state_cov_batteries, state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs_sum):
        
        k_states = self._k_states

        curr_filter = kfilters[curr_regime]
        
        np.multiply(filtered_prev_and_curr_regime_probs[:,
                curr_regime].reshape(-1, 1),
                state_batteries[:, curr_regime, :], out=weighted_states)

        np.sum(weighted_states, axis=0, out=weighted_states_sum)

        approx_state = curr_filter.filtered_state[:, t]

        np.divide(weighted_states_sum,
                filtered_current_regime_probs[current_regime],
                out=approximate_state)
        
        np.subtract(approx_state.reshape(1, -1, 1),
                state_batteries[:, curr_regime, :].reshape(-1, k_states, 1),
                out=state_biases)

        np.subtract(approx_state.reshape(1, 1, -1),
                state_batteries[:, curr_regime, :].reshape(-1, 1, k_states),
                out=transposed_state_biases)

        np.multiply(state_biases, transposed_state_biases, out=state_bias_sqrs)

        np.sum(state_cov_batteries[:, curr_regime, :, :], state_bias_sqrs,
                out=state_covs_and_bias_sqrs)

        np.multiply(filtered_prev_and_curr_regime_probs[:,
                curr_regime].reshape(-1, 1, 1), state_covs_and_state_bias_sqrs,
                out=weighted_state_covs_and_state_bias_sqrs)

        np.sum(weighted_state_covs_and_state_bias_sqrs, axis=0,
                out=weighted_state_covs_and_state_bias_sqrs_sum)

        np.divide(weighted_state_covs_and_state_bias_sqrs_sum,
                filtered_curr_regime_probs[curr_regime],
                out=curr_filter.filtered_state_cov[:, :, t])

    def filter(self, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):

        nobs = self.nobs
        k_regimes = self._k_regimes
        k_states = self._k_states

        kfilters = self._initialize_filters(filter_method=filter_method,
                inversion_method=inversion_method,
                stability_method=stability_method,
                conserve_memory=conserve_memory, tolerance=tolerance,
                complex_step=complex_step) 
        
        self.obs_likelihoods = np.zeros((nobs,))
        
        if self.initial_regime_probs is None:
            self.initialize_stationary_regime_probs()

        # Allocation of buffers

        state_buffer = np.zeros((k_states,))
        state_cov_buffer = np.zeros((k_states, k_states))
        
        state_batteries = np.zeros((k_regimes, k_regimes, k_states))
        state_cov_batteries = np.zeros((k_regimes, k_regimes, k_states,
                k_states))
        
        forecast_error_batteries = np.zeros((k_regimes, k_regimes, k_states))
        forecast_error_cov_batteries = np.zeros((k_regimes, k_regimes,
                k_states, k_states))

        filtered_curr_regime_probs = np.zeros((k_regimes,))
        predicted_prev_and_curr_regime_probs = np.zeros((k_regimes, k_regimes))
        prev_and_curr_regime_cond_obs_probs = np.zeros((k_regimes, k_regimes))
        predicted_obs_prev_and_curr_regime_probs = np.zeros((k_regimes,
                k_regimes))
        filtered_prev_and_curr_regime_probs = np.zeros((k_regimes, k_regimes))        
        
        weighted_states = np.zeros((k_regimes, k_states))
        weighted_states_sum = np.zeros((k_states,))
        state_biases = np.zeros((k_regimes, k_states, 1))
        transposed_state_biases = np.zeros((k_regimes, 1, k_states))
        state_bias_sqrs = np.zeros((k_regimes, k_states, k_states))
        state_covs_and_state_bias_sqrs = np.zeros((k_regimes, k_states,
                k_states))
        weighted_state_covs_and_state_bias_sqrs = np.zeros((k_regimes,
                k_states, k_states))
        weighted_state_covs_and_state_bias_sqrs_sum = np.zeros((k_states,
                k_states))

        for t in range(self.nobs):
            
            # Kalman filter
            for prev_regime in range(k_regimes):
                for curr_regime in range(k_regimes):
                    self._kalman_filter_step(t, kfilters, prev_regime,
                            current_regime, state_buffer, state_cov_buffer,
                            state_batteries, state_cov_batteries,
                            forecast_error_batteries,
                            forecast_error_cov_batteries)

            # Hamilton filter
            self._hamilton_filter_step(t, kfilters,
                    filtered_curr_regime_probs,
                    predicted_prev_and_curr_regime_probs,
                    forecast_error_batteries, forecast_error_cov_batteries,
                    prev_and_curr_regime_cond_obs_probs,
                    predicted_obs_prev_and_curr_regime_probs,
                    filtered_prev_and_curr_regime_probs)

            # Approximation
            for curr_regime in range(k_regimes):
                self._approximation_step(t, kfilters, curr_regime,
                        filtered_prev_and_curr_regime_probs, state_batteries,
                        weighted_states, weighted_states_sum,
                        filtered_curr_regime_probs, state_biases,
                        transposed_state_biases, state_bias_sqrs,
                        state_cov_batteries, state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs_sum)
    
    def loglike(self, **kwargs):

        return np.log(self.obs_likelihoods).sum()
    
    def loglikeobs(self, **kwargs):

        return np.log(self.obs_likelihoods)
