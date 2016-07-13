"""
Markov Switching State Space Representation and Kim Filter

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from .switching_representation import SwitchingRepresentation, \
        FrozenSwitchingRepresentation
from scipy.misc import logsumexp

def _marginalize_vector(event_probs, event_conditional_vectors,
        weighted_vectors, marginal_vector, vector_biases, vector_bias_sqrs,
        event_conditional_covs, covs_plus_bias_sqrs,
        weighted_covs_plus_bias_sqrs, marginal_cov):
    r"""
    Generic method, marginalizing random vector's expectation and covariance
    matrix

    Parameters
    ----------
    event_probs : array_like
        Probabilities of the set of the collectively exhaustive events
        :math:`{ A_1, ..., A_n }`. In Kim filter these events are determined by
        Markov switching model regime value.
    event_conditional_vectors : array_like
        (Hereafter let's denote random vector as :math:`\alpha`)
        Vector expectations, conditional on events: :math:`E[ \alpha | A_i ]`.
    weighted_vectors : array_like
        Buffer to store :math:`Pr[ A_i ] * E[ \alpha | A_i ]`.
    marginal_vector : array_like
        Buffer to store the result: :math:`E[ \alpha ]`.
    vector_biases : array_like
        Buffer to store :math:`E[ \alpha ] - E[ \alpha | A_i ]`.
    vector_bias_sqrs : array_like
        Buffer to store :math:`( E[ \alpha ] - E[ \alpha | A_i ]) *
        ( E[ \alpha ] - E[ \alpha | A_i ] )^T`.
    event_conditional_covs : array_like
        Vector covariance matrices, conditional on events:
        :math:`Var[ \alpha | A_i ]`.
    covs_plus_bias_sqrs : array_like
        Buffer to store :math:` Var[ \alpha ] ( E[ \alpha ] -
        E[ \alpha | A_i ]) * ( E[ \alpha ] - E[ \alpha | A_i ] )^T`.
    weighted_covs_plus_bias_sqrs : array_like
        Buffer to store :math:`Pr[ A_i ] * ( Var[ \alpha ] ( E[ \alpha ] -
        E[ \alpha | A_i ]) * ( E[ \alpha ] - E[ \alpha | A_i ] )^T )`.
    marginal_cov : array_like
        Buffer to store the result: :math:`Var[ \alpha ]`.
    """

    # Pr[ A_i ] * E[ \alpha | A_i ]
    np.multiply(event_probs.reshape(-1, 1), event_conditional_vectors,
            out=weighted_vectors)

    # E[ \alpha ] = \sum_{i} Pr[ A_i ] * E[ \alpha | A_i ]
    np.sum(weighted_vectors, axis=0, out=marginal_vector)

    # E[ \alpha ] - E[ \alpha | A_i ]
    np.subtract(marginal_vector, event_conditional_vectors,
            out=vector_biases)

    # ( E[ \alpha ] - E[ \alpha | A_i ] ) *
    # * ( E[ \alpha ] - E[ \alpha | A_i ] )^T
    for i in range(event_probs.shape[0]):
        np.outer(vector_biases[i], vector_biases[i], out=vector_bias_sqrs[i])

    # Var[ \alpha | A_i ] + ( E[ \alpha ] - E[ \alpha | A_i ] ) *
    # * ( E[ \alpha ] - E[ \alpha | A_i ] )^T
    np.add(event_conditional_covs, vector_bias_sqrs, out=covs_plus_bias_sqrs)

    # Pr[ A_i ] * ( Var[ \alpha | A_i ] + ( E[ \alpha ] - E[ \alpha | A_i ] ) *
    # * ( E[ \alpha ] - E[ \alpha | A_i ] )^T )
    np.multiply(event_probs.reshape(-1, 1, 1), covs_plus_bias_sqrs,
            out=weighted_covs_plus_bias_sqrs)

    # Var[ \alpha ] = \sum_{i}
    # Pr[ A_i ] * ( Var[ \alpha | A_i ] + ( E[ \alpha ] - E[ \alpha | A_i ] ) *
    # * ( E[ \alpha ] - E[ \alpha | A_i ] )^T )
    np.sum(weighted_covs_plus_bias_sqrs, axis=0,
            out=marginal_cov)


class _KimFilter(object):

    def __init__(self, model, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):

        # This class does all hard filtering work

        self.model = model

        # Check if endogenous data is binded to switching representation
        if not hasattr(model, '_nobs'):
            raise RuntimeError(
                    'No endog data binded. Consider using bind() first.')

        # Initialize filters and save some useful low-level references
        model._initialize_filters(filter_method=filter_method,
            inversion_method=inversion_method,
            stability_method=stability_method, conserve_memory=conserve_memory,
            tolerance=tolerance, complex_step=complex_step)

    def _hamilton_prediction_step(self, t,
            predicted_prev_and_curr_regime_logprobs):

        # The goal of this step is to calculate Pr[ S_t, S_{t-1} | \psi_{t-1} ]

        model = self.model

        # Filtered regime log-probabilities at previous moment
        if t == 0:
            regime_logprobs = model._initial_regime_logprobs
        else:
            regime_logprobs = self.filtered_regime_logprobs[:, t - 1]

        # Pr[ S_t, S_{t-1} | \psi_{t-1} ] = Pr[ S_t | S_{t-1} ] *
        # * Pr[ S_{t-1} | \psi_{t-1} ]
        np.add(model._log_regime_transition.transpose(),
                regime_logprobs.reshape(-1, 1),
                out=predicted_prev_and_curr_regime_logprobs)

        # Pr[ S_t | \psi_{t-1} ] =
        # = \sum_{S_{t-1}} Pr[ S_t, S_{t-1} | \psi_{t-1} ]
        self.predicted_regime_logprobs[:, t] = logsumexp(
                predicted_prev_and_curr_regime_logprobs, axis=0)

    def _kalman_filter_step(self, t, prev_regime, curr_regime, state_buffer,
            state_cov_buffer, state_batteries, state_cov_batteries,
            prev_and_curr_regime_cond_obs_logprobs, forecast_error_batteries,
            forecast_error_cov_batteries):

        # `SwitchingRepresentation` aggregates `k_regimes` `KalmanFilter`
        # instances. During this step, for every combination (i, j)
        # \beta_{t-1|t-1}^{i} (and P_{t-1|t-1}^{i} respectively) is passed
        # through j-th Kalman filter to produce \beta_{t|t}^{(i,j)},
        # P_{t|t}^{(i,j)} and f( y_t | S_{t-1} = i, S_t = j, \psi_{t-1} ).

        # All heavy-weight computations of this step are delegated to
        # `KalmanFilter` class

        model = self.model

        # Saving useful references of high and low level Kalman filters

        # Low-level Cython filters
        curr_kfilter = model._kfilters[curr_regime]
        prev_kfilter = model._kfilters[prev_regime]
        # Corresponding high-level Python filters
        curr_regime_filter = model._regime_kalman_filters[curr_regime]
        prev_regime_filter = model._regime_kalman_filters[prev_regime]

        # Every `curr_kfilter` does the filtering iteration `k_regimes` times
        # from one time position, so its time pointer has to be corrected every
        # time.
        curr_kfilter.seek(t)

        # `Feeding` \beta_{t-1|t-1}^{i} and P_{t-1|t-1}^{i} to j-th Kalman
        # Filter input.
        if t == 0:
            # Saving previous initialization in temporary buffer
            state_buffer = curr_regime_filter._initial_state
            state_cov_buffer = curr_regime_filter._initial_state_cov
            # High level initialization
            curr_regime_filter.initialize_known(
                    prev_regime_filter._initial_state,
                    prev_regime_filter._initial_state_cov)
            # Low level initialization - this method "pushes" new
            # initialization to Cython Kalman filter
            curr_regime_filter._initialize_state(
                    **model._state_init_kwargs[curr_regime])
        else:
            # Saving previous moment filter data to temporary buffer
            np.copyto(state_buffer, curr_kfilter.filtered_state[:, t - 1])
            np.copyto(state_cov_buffer,
                    curr_kfilter.filtered_state_cov[:, :, t - 1])
            # Subtitution j-th filter filtered data at previous moment by
            # \beta_{t-1|t-1}^{i} and P_{t-1|t-1}^{i}
            np.copyto(np.asarray(curr_kfilter.filtered_state[:, t - 1]),
                    prev_kfilter.filtered_state[:, t - 1])
            np.copyto(np.asarray(curr_kfilter.filtered_state_cov[:, :, t - 1]),
                    prev_kfilter.filtered_state_cov[:, :, t - 1])

        # Do the Kalman filter iteration
        next(curr_kfilter)

        # Putting filtered data from previous moment from temporary buffer back
        # to filter's matrices
        if t == 0:
            curr_regime_filter.initialize_known(state_buffer, state_cov_buffer)
        else:
            np.copyto(np.asarray(curr_kfilter.filtered_state[:, t - 1]),
                    state_buffer)
            np.copyto(np.asarray(curr_kfilter.filtered_state_cov[:, :, t - 1]),
                    state_cov_buffer)

        # Saving \beta_{t|t}^{(i,j)} and P_{t|t}^{(i,j)} in batteries
        np.copyto(state_batteries[prev_regime, curr_regime, :],
                curr_kfilter.filtered_state[:, t])
        np.copyto(state_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.filtered_state_cov[:, :, t])

        # Saving f( y_t | S_{t-1} = i, S_t = j, \psi_{t-1} )
        prev_and_curr_regime_cond_obs_logprobs[prev_regime, curr_regime] = \
                curr_kfilter.loglikelihood[t]

        np.copyto(forecast_error_batteries[prev_regime, curr_regime, :],
                curr_kfilter.forecast_error[:, t])
        np.copyto(forecast_error_cov_batteries[prev_regime, curr_regime, :, :],
                curr_kfilter.forecast_error_cov[:, :, t])

    def _collapse_forecasts(self, t, predicted_prev_and_curr_regime_logprobs,
            predicted_prev_and_curr_regime_probs, forecast_error_batteries,
            weighted_error_batteries, error_biases, error_bias_sqrs,
            forecast_error_cov_batteries, error_covs_plus_bias_sqrs,
            weighted_error_covs_plus_bias_sqrs):

        # This method calculates \eta_{t|t-1} and f_{t|t-1}, which are primarily
        # used in `FilterResults` and `SwitchingMLEResults` for hypothesis
        # testing.

        model = self.model

        k_endog = model._k_endog

        # Switching from logprobs to probs
        np.exp(predicted_prev_and_curr_regime_logprobs,
                out=predicted_prev_and_curr_regime_probs)

        # Calculate \eta_{t|t-1}, f_{t|t-1}, collapsing \eta_{t|t-1}^{(i,j)} and
        # f_{t|t-1}^{(i,j)} with probability weights
        _marginalize_vector(predicted_prev_and_curr_regime_probs,
                forecast_error_batteries.reshape(-1, k_endog),
                weighted_error_batteries, self.forecast_error[:, t],
                error_biases, error_bias_sqrs,
                forecast_error_cov_batteries.reshape(-1, k_endog, k_endog),
                error_covs_plus_bias_sqrs, weighted_error_covs_plus_bias_sqrs,
                self.forecast_error_cov[:, :, t])

    def _hamilton_filtering_step(self, t,
            predicted_prev_and_curr_regime_logprobs,
            prev_and_curr_regime_cond_obs_logprobs,
            predicted_prev_and_curr_regime_and_obs_logprobs,
            filtered_prev_and_curr_regime_logprobs):

        # This step is related to different probability inferences

        model = self.model

        k_regimes = model._k_regimes

        # f( y_t, S_t, S_{t-1} | \psi_{t-1} ) =
        # = f( y_t | S_t, S_{t-1}, \psi_{t-1} ) *
        # * Pr[ S_t, S_{t-1} | \psi_{t-1} ]
        np.add(prev_and_curr_regime_cond_obs_logprobs,
                predicted_prev_and_curr_regime_logprobs,
                out=predicted_prev_and_curr_regime_and_obs_logprobs)

        # f( y_t | \psi_{t-1} ) = \sum_{S_t, S_{t-1}}
        # f( y_t, S_t, S_{t-1} | \psi_{t-1} )
        obs_loglikelihood = \
                logsumexp(predicted_prev_and_curr_regime_and_obs_logprobs)

        # Saving f( y_t | \psi_{t-1} )
        self.obs_loglikelihoods[t] = obs_loglikelihood

        # Pr[ S_t, S_{t-1} | \psi_{t} ] = f( y_t, S_t, S_{t-1} | \psi_{t-1} ) /
        # / f( y_t | \psi_{t-1} )
        # Condition to avoid -np.inf - (-np.inf) operation
        if obs_loglikelihood != -np.inf:
            np.subtract(predicted_prev_and_curr_regime_and_obs_logprobs,
                    obs_loglikelihood,
                    out=filtered_prev_and_curr_regime_logprobs)
        else:
            filtered_prev_and_curr_regime_logprobs[:, :] = -np.inf

        # Pr[ S_t | \psi_t ] = \sum_{S_{t-1}} Pr[ S_t, S_{t-1} | \psi_{t} ]
        self.filtered_regime_logprobs[:, t] = logsumexp(
                filtered_prev_and_curr_regime_logprobs, axis=0)

    def _approximation_step(self, t, curr_regime,
            filtered_prev_and_curr_regime_logprobs, approx_states,
            filtered_prev_cond_on_curr_regime_logprobs,
            state_batteries, weighted_states, state_biases,
            state_bias_sqrs, state_cov_batteries,
            state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs, approx_state_covs):

        # During this step approximate \beta_{t|t}^{i} and P_{t|t}^{i}
        # are calculated and stored inside Kalman filters' `filtered_state` and
        # `filtered_state_cov` matrices (using these matrices allows not to
        # allocate another heavy arrays but seems to conform name semantics)

        model = self.model

        # Reference to Cython KalmanFilter
        curr_filter = model._kfilters[curr_regime]

        # Zero joint probability indicates that this pair of current and
        # previous regimes is impossible and no need to do further calculations
        if self.filtered_regime_logprobs[curr_regime, t] == -np.inf:
            # Filling \beta_{t-1|t-1}^{i} and P_{t-1|t-1}^{i} with zeros
            # Any value would be alright, since these data is multiplied by zero
            # weight in the next iteration
            approx_states[curr_regime, :] = 0
            approx_state_covs[curr_regime, :, :] = 0

            # Copying data form temporary buffer to Kalman filter matrices
            # There is no need to do it here, but just for consistency
            np.copyto(np.asarray(curr_filter.filtered_state[:, t]),
                    approx_states[curr_regime, :])
            np.copyto(np.asarray(curr_filter.filtered_state_cov[:, :, t]),
                    approx_state_covs[curr_regime, :, :])
            return

        # Pr[ S_{t-1} | S_t, \psi_t ] = Pr[ S_t, S_{t-1} | \psi_t ] /
        # / Pr[ S_t | \psi_t ]
        np.subtract(filtered_prev_and_curr_regime_logprobs[:, curr_regime],
                self.filtered_regime_logprobs[curr_regime, t],
                out=filtered_prev_cond_on_curr_regime_logprobs)

        # Switching from logprobs to probs
        filtered_prev_cond_on_curr_regime_probs = \
                filtered_prev_cond_on_curr_regime_logprobs
        np.exp(filtered_prev_cond_on_curr_regime_logprobs,
                out=filtered_prev_cond_on_curr_regime_probs)

        # Calculate \beta_{t|t}^j, P_{t|t}^j, collapsing \beta_{t|t}^{(i,j)} and
        # P_{t|t}^{(i,j)} with probability weights
        _marginalize_vector(filtered_prev_cond_on_curr_regime_logprobs,
                state_batteries[:, curr_regime, :], weighted_states,
                approx_states[curr_regime, :], state_biases, state_bias_sqrs,
                state_cov_batteries[:, curr_regime, :, :],
                state_covs_and_state_bias_sqrs,
                weighted_state_covs_and_state_bias_sqrs,
                approx_state_covs[curr_regime, :, :])

        # Copying data form temporary buffer to Kalman filter matrices
        np.copyto(np.asarray(curr_filter.filtered_state[:, t]),
                approx_states[curr_regime, :])
        np.copyto(np.asarray(curr_filter.filtered_state_cov[:, :, t]),
                approx_state_covs[curr_regime, :, :])

    def _collapse_states(self, t, filtered_regime_probs,
            approx_states, weighted_states, state_biases, state_bias_sqrs,
            approx_state_covs, state_covs_and_state_bias_sqrs,
            weighted_state_covs_and_state_bias_sqrs):

        # This method calculates \beta_{t|t} and P_{t|t} using \beta_{t|t}^j
        # and P_{t|t}^j.

        # Switching from logprobs to probs
        np.exp(self.filtered_regime_logprobs[:, t], out=filtered_regime_probs)

        # Collapsing \beta_{t|t}^i and P_{t|t}^i to get the result
        _marginalize_vector(filtered_regime_probs, approx_states,
                weighted_states, self.filtered_state[:, t], state_biases,
                state_bias_sqrs, approx_state_covs,
                state_covs_and_state_bias_sqrs,
                weighted_state_covs_and_state_bias_sqrs,
                self.filtered_state_cov[:, :, t])

    def __call__(self):

        # This method is based on section 5 of
        # Kim, Chang-Jin, and Charles R. Nelson. 1999.
        # "State-Space Models with Regime Switching:
        # Classical and Gibbs-Sampling Approaches with Applications".
        # MIT Press Books. The MIT Press.

        # Also, notation in comments follows this section

        model = self.model

        k_endog = model._k_endog
        k_states = model._k_states
        k_regimes = model._k_regimes
        dtype = model._dtype
        nobs = model._nobs

        # Array, storing \ln( f( y_t | \psi_{t-1} ) )
        self.obs_loglikelihoods = np.zeros((nobs,), dtype=dtype)

        # Array, storing \ln( Pr[ S_t | \psi_{t-1} ] )
        self.filtered_regime_logprobs = np.zeros((k_regimes, nobs),
                dtype=dtype)
        # Array, storing \ln( Pr[ S_t | \psi_t ] )
        self.predicted_regime_logprobs = np.zeros((k_regimes, nobs),
                dtype=dtype)

        # Array, storing \beta_{t|t}
        self.filtered_state = np.zeros((k_states, nobs), dtype=dtype)
        # Array, storing P_{t|t}
        self.filtered_state_cov = np.zeros((k_states, k_states, nobs),
                dtype=dtype)

        # Array, storing \eta_{t|t-1}
        self.forecast_error = np.zeros((k_endog, nobs), dtype=dtype)
        # Array, storing f_{t|t-1}
        self.forecast_error_cov = np.zeros((k_endog, k_endog, nobs),
                dtype=dtype)

        # If user didn't specify initialization, try to find stationary regime
        # distribution. If it is not found, use simple uniform distribution.
        if not hasattr(model, '_initial_regime_probs'):
            try:
                model.initialize_stationary_regime_probs()
            except RuntimeError:
                model.initialize_uniform_regime_probs()

        # Allocation of buffers, which are reused during numerous iterations
        # of Kim filtering routine.

        # These buffers are used during Kalman filter step as a temporary
        # location of replaced from filters states.
        state_buffer = np.zeros((k_states,), dtype=dtype)
        state_cov_buffer = np.zeros((k_states, k_states), dtype=dtype)

        # Batteries of \beta_{t|t}^{(i,j)}
        state_batteries = np.zeros((k_regimes, k_regimes, k_states),
                dtype=dtype)
        # Batteries of P_{t|t}^{(i,j)}
        state_cov_batteries = np.zeros((k_regimes, k_regimes, k_states,
                k_states), dtype=dtype)

        # Batteries of \eta_{t|t-1}^{(i,j)}
        forecast_error_batteries = np.zeros((k_regimes, k_regimes, k_endog),
                dtype=dtype)
        # Batteries of f_{t|t-1}^{(i,j)}
        forecast_error_cov_batteries = np.zeros((k_regimes, k_regimes, k_endog,
                k_endog), dtype=dtype)

        # Buffer for Pr[ S_{t-1} = i, S_t = j | \psi_{t-1} ] *
        # * \eta_{t|t-1}^{(i,j)}
        weighted_error_batteries = np.zeros((k_regimes * k_regimes, k_endog),
                dtype=dtype)
        # Buffer for \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)}
        error_biases = np.zeros((k_regimes * k_regimes, k_endog), dtype=dtype)
        # Buffer for ( \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)} ) *
        # * ( \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)} )^T
        error_bias_sqrs = np.zeros((k_regimes * k_regimes, k_endog, k_endog),
                dtype=dtype)
        # Buffer for f_{t|t-1}^{(i,j)} +
        # + ( \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)} ) *
        # * ( \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)} )^T
        error_covs_plus_bias_sqrs = np.zeros((k_regimes * k_regimes, k_endog,
                k_endog), dtype=dtype)
        # Buffer for Pr[ S_{t-1} = i, S_t = j | \psi_{t-1} ] *
        # * ( f_{t|t-1}^{(i,j)} + ( \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)} ) *
        # * ( \eta_{t|t-1} - \eta_{t|t-1}^{(i,j)} )^T )
        weighted_error_covs_plus_bias_sqrs = np.zeros((k_regimes * k_regimes,
                k_endog, k_endog), dtype=dtype)

        # Buffer for \ln( Pr[ S_t, S_{t-1} | \psi_{t-1} ] )
        predicted_prev_and_curr_regime_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        # Buffer for Pr[ S_t, S_{t-1} | \psi_{t-1} ]
        predicted_prev_and_curr_regime_probs = np.zeros((k_regimes, k_regimes),
                dtype=dtype)
        # Buffer for \ln( f( y_t | S_t, S_{t-1}, \psi_{t-1} ) )
        prev_and_curr_regime_cond_obs_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        # Buffer for \ln( f( y_t, S_t, S_{t-1} | \psi_{t-1} ) )
        predicted_prev_and_curr_regime_and_obs_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)
        # Buffer for \ln( Pr[ S_t, S_{t-1} | \psi_t ] )
        filtered_prev_and_curr_regime_logprobs = np.zeros((k_regimes,
                k_regimes), dtype=dtype)

        # Buffer for \ln( Pr[ S_{t-1} | S_t, \psi_t ] )
        filtered_prev_cond_on_curr_regime_logprobs = np.zeros((k_regimes,),
                dtype=dtype)

        # Buffer for \beta_{t|t}^j
        approx_states = np.zeros((k_regimes, k_states), dtype=dtype)
        # Buffer for P_{t|t}^j
        approx_state_covs = np.zeros((k_regimes, k_states, k_states),
                dtype=dtype)

        # Buffer for Pr[ S_{t-1} = i | S_t = j, \psi_t ] * \beta_{t|t}^{(i,j)}
        # (in `_approximation_step`) and for
        # Pr[ S_t = j | \psi_t ] * \beta_{t|t}^j (in `_collapse_states`)
        weighted_states = np.zeros((k_regimes, k_states), dtype=dtype)

        # Buffer for \beta_{t|t}^j - \beta_{t|t}^{(i,j)}
        # (in `_approximation_step`) and for \beta_{t|t} - \beta_{t|t}^j
        # (in `_collapse_states`)
        state_biases = np.zeros((k_regimes, k_states), dtype=dtype)

        # Buffer for ( \beta_{t|t}^j - \beta_{t|t}^{(i,j)} ) *
        # * ( \beta_{t|t}^j - \beta_{t|t}^{(i,j)} )^T (in `_approximation_step`)
        # and for ( \beta_{t|t} - \beta_{t|t}^j ) *
        # * ( \beta_{t|t} - \beta_{t|t}^j )^T (in `_collapse_states`)
        state_bias_sqrs = np.zeros((k_regimes, k_states, k_states),
                dtype=dtype)

        # Buffer for P_{t|t}^{(i,j)} + ( \beta_{t|t}^j - \beta_{t|t}^{(i,j)} ) *
        # * ( \beta_{t|t}^j - \beta_{t|t}^{(i,j)} )^T (in `_approximation_step`)
        # and for P_{t|t}^j + ( \beta_{t|t} - \beta_{t|t}^j ) *
        # * ( \beta_{t|t} - \beta_{t|t}^j )^T (in `_collapse_states`)
        state_covs_and_state_bias_sqrs = np.zeros((k_regimes, k_states,
                k_states), dtype=dtype)

        # Buffer for Pr[ S_{t-1} = i | S_t = j, \psi_t ] * ( P_{t|t}^{(i,j)} +
        # + ( \beta_{t|t}^j - \beta_{t|t}^{(i,j)} ) *
        # * ( \beta_{t|t}^j - \beta_{t|t}^{(i,j)} )^T )
        # (in `_approximation_step`) and for
        # Pr[ S_t = j | \psi_t ] * ( P_{t|t}^j +
        # + ( \beta_{t|t} - \beta_{t|t}^j ) *
        # * ( \beta_{t|t} - \beta_{t|t}^j )^T ) (in `_collapse_states`)
        weighted_state_covs_and_state_bias_sqrs = np.zeros((k_regimes,
                k_states, k_states), dtype=dtype)

        # Buffer for Pr[ S_t = j | \psi_t ]
        filtered_regime_probs = np.zeros((k_regimes,), dtype=dtype)

        # Iterating over observation period
        for t in range(nobs):
            # Kim filter iteration consists of three consecutive steps: Kalman
            # filter step, Hamilton filter step and Approximation step.
            # Here Hamilton filter step is splitted into two parts: prediction
            # and filtering.

            # To optimize computation time, several buffers are reused over
            # iterations, so no array reallocations required. That's why
            # methods, corresponding to Kim filter phases, have a lot of
            # buffers in their argument lists.

            # Hamilton prediction
            self._hamilton_prediction_step(t,
                    predicted_prev_and_curr_regime_logprobs)

            # Kalman filter
            for prev_regime in range(k_regimes):
                for curr_regime in range(k_regimes):
                    # This condition optimizes calculation time in case of
                    # sparse regime transition  matrix (e.g. for MS AR)
                    if predicted_prev_and_curr_regime_logprobs[prev_regime, \
                            curr_regime] != -np.inf:
                        self._kalman_filter_step(t, prev_regime, curr_regime,
                                state_buffer, state_cov_buffer, state_batteries,
                                state_cov_batteries,
                                prev_and_curr_regime_cond_obs_logprobs,
                                forecast_error_batteries,
                                forecast_error_cov_batteries)

            # Collecting forecast errors
            self._collapse_forecasts(t, predicted_prev_and_curr_regime_logprobs,
                    predicted_prev_and_curr_regime_probs,
                    forecast_error_batteries, weighted_error_batteries,
                    error_biases, error_bias_sqrs, forecast_error_cov_batteries,
                    error_covs_plus_bias_sqrs,
                    weighted_error_covs_plus_bias_sqrs)

            # Hamilton filter
            self._hamilton_filtering_step(t,
                    predicted_prev_and_curr_regime_logprobs,
                    prev_and_curr_regime_cond_obs_logprobs,
                    predicted_prev_and_curr_regime_and_obs_logprobs,
                    filtered_prev_and_curr_regime_logprobs)

            # Approximation
            for curr_regime in range(k_regimes):
                self._approximation_step(t, curr_regime,
                        filtered_prev_and_curr_regime_logprobs, approx_states,
                        filtered_prev_cond_on_curr_regime_logprobs,
                        state_batteries, weighted_states, state_biases,
                        state_bias_sqrs, state_cov_batteries,
                        state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs,
                        approx_state_covs)

            # Collecting filtering results
            self._collapse_states(t, filtered_regime_probs,
                    approx_states, weighted_states, state_biases,
                    state_bias_sqrs, approx_state_covs,
                    state_covs_and_state_bias_sqrs,
                    weighted_state_covs_and_state_bias_sqrs)

class KimFilter(SwitchingRepresentation):
    """
    Markov switching state space representation of a time series process, with
    Kim filter

    Parameters
    ----------
    k_endog : int
        The number of variables in the process.
    k_states : int
        The dimension of the unobserved state process.
    k_regimes : int
        The number of switching regimes.
    loglikelihood_burn : int, optional
        The number of initial periods during which the loglikelihood is not
        recorded. Default is 0.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `KimFilterResults`. If specified, class must extend from
        `KimFilterResults`.
    **kwargs
        Additional keyword arguments, passed to `SwitchingRepresentation`
        initializer.

    Notes
    -----
    This class extends `SwitchingRepresentation` and performs filtering and
    likelihood estimation.

    See Also
    --------
    _KimFilter
    KimFilterResults
    statsmodels.tsa.statespace.regime_switching.switching_representation. \
    SwitchingRepresentation
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    """

    def __init__(self, k_endog, k_states, k_regimes, loglikelihood_burn=0,
            results_class=None, **kwargs):

        # Filter options
        self._loglikelihood_burn = loglikelihood_burn

        # Set results class
        if results_class is not None:
            self._results_class = results_class
        else:
            self._results_class = KimFilterResults

        # Initialize representation
        super(KimFilter, self).__init__(k_endog, k_states, k_regimes, **kwargs)

    def filter(self, results=None, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):
        """
        Apply the Kim filter to the Markov switching statespace model

        Parameters
        ----------

        results : class or object, optional
            If a class, then that class is instantiated and returned with the
            result of filtering. It must be a subclass of `KimFilterResults`.
            If an object, then that object is updated with the filtering data.
            Its class should extend `KimFilterResults`.
            If `None`, then a `KimFilterResults` object is returned.

        Notes
        -----
        `filter_method`, `inversion_method`, `stability_method`,
        `conserve_memory` and `tolerance` keyword arguments are passed to
        `k_regimes` `KalmanFilter` instances, used in Kim filtering routine.
        `filter_timing` doesn't allow modification, because Kim filter
        requires Kim and Nelson timing convention.
        See keyword arguments documentation in
        `statsmodels.tsa.statespace.kalman_filter.KalmanFilter.filter`.

        See Also
        --------
        statsmodels.tsa.statespace.kalman_filter.KalmanFilter.filter
        """

        # Actual calculations are done by `_KimFilter` class. See this class
        # for details.
        kfilter = _KimFilter(self, filter_method=filter_method,
            inversion_method=inversion_method,
            stability_method=stability_method, conserve_memory=conserve_memory,
            tolerance=tolerance, complex_step=complex_step)
        kfilter()

        # If results are not provided, use the class from initializer
        # (or default `KimFilterResults` class)
        if results is None:
            results = self._results_class

        # If results is a class, create an instance of this class
        if isinstance(results, type):
            if not issubclass(results, KimFilterResults):
                raise ValueError('Invalid results type.')
            results = results(self)

        # save representation data in results
        results.update_representation(self)
        # Save filtering data in results
        results.update_filter(kfilter)

        return results

    def _initialize_filters(self, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):

        # This method is used before filtering, see `_KimFilter.__init__` method

        kfilters = []
        state_init_kwargs = []

        # Using Kim and Nelson timing convention is required for Kim filter
        filter_timing = 1

        for regime_filter in self._regime_kalman_filters:
            # Initializating
            prefix = regime_filter._initialize_filter(
                    filter_method=filter_method,
                    inversion_method=inversion_method,
                    stability_method=stability_method,
                    conserve_memory=conserve_memory, tolerance=tolerance,
                    filter_timing=filter_timing)[0]
            kfilters.append(regime_filter._kalman_filters[prefix])

            state_init_kwargs.append({'prefix': prefix,
                    'complex_step': complex_step})
            #regime_filter._initialize_state(prefix=prefix,
            #        complex_step=complex_step)

        # Store Cython filter references in the class
        self._kfilters = kfilters

        # These arguments are stored for `KalmanFilter.initialize_state`
        # method call later. See `_KimFilter._kalman_filter_step` method
        self._state_init_kwargs = state_init_kwargs

    @property
    def loglikelihood_burn(self):
        """
        (int) The number of initial periods during which the loglikelihood is
        not recorded. Default is 0.
        """
        return self._loglikelihood_burn

    @loglikelihood_burn.setter
    def loglikelihood_burn(self, value):
        """
        (int) The number of initial periods during which the loglikelihood is
        not recorded. Default is 0.
        """
        self._loglikelihood_burn = value

    @property
    def initialization(self):
        return self._regime_kalman_filters[0].initialization

    def loglikeobs(self, loglikelihood_burn=0, **kwargs):
        """
        Calculate the loglikelihood for each observation associated with the
        statespace model.

        Parameters
        ----------
        loglikelihood_burn : int
            The number of initial periods during which the loglikelihood is
            not recorded.

        **kwargs
            Additional keyword arguments to pass to the Kim filter. See
            `KimFilter.filter` for more details.

        Notes
        -----
        If `loglikelihood_burn` is positive, then the entries in the returned
        loglikelihood vector are set to be zero for those initial time periods.

        Returns
        -------
        loglike : array of float
            Array of loglikelihood values for each observation.
        """

        # Perform filtering
        kfilter = _KimFilter(self, **kwargs)
        kfilter()

        # If loglikelihood is provided in both `loglikeobs` arguments and
        # constructor (or setter), choose bigger one
        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        # Copying observations loglikelihoods from `_KimFilter`
        loglikelihoods = np.array(kfilter.obs_loglikelihoods)
        loglikelihoods[:loglikelihood_burn] = 0

        return loglikelihoods

    def loglike(self, loglikelihood_burn=0, **kwargs):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        loglikelihood_burn : int
            The number of initial periods during which the loglikelihood is
            not recorded.

        **kwargs
            Additional keyword arguments to pass to the Kim filter. See
            `KimFilter.filter` for more details.

        Returns
        -------
        loglike : float
            The joint loglikelihood.
        """

        # Perform filtering
        kfilter = _KimFilter(self, **kwargs)
        kfilter()

        # If loglikelihood is provided in both `loglike` arguments and
        # constructor (or setter), choose bigger one
        loglikelihood_burn = max(loglikelihood_burn, self._loglikelihood_burn)

        # Summarize observation loglikelihoods and return results
        return kfilter.obs_loglikelihoods[loglikelihood_burn:].sum()


class KimFilterResults(FrozenSwitchingRepresentation):
    """
    Results from applying the Kim filter to a Markov switching state space
    model.

    Parameters
    ----------
    model : SwitchingRepresentation
        A Markov switching state space representation

    Attributes
    ----------
    loglikelihood_burn : int
        The number of initial periods during which
        the loglikelihood is not recorded.
    initial_regime_logprobs : array
        Marginal regime distribution at t=0, array of shape `(k_regimes,)`.
    obs_loglikelihoods : array
        Loglikelihood for each observation associated with the
        statespace model (ignoring loglikelihood_burn).
    filtered_state : array
        State mean at the moment t, conditional on all observations measured
        till the moment t. A `(nobs, k_states)` shaped array.
    filtered_state_cov : array
        State covariance at the moment t, conditional on all observations
        measured till the moment t. A `(nobs, k_states, k_states)` shaped
        array.
    filtered_regime_logprobs : array
        Log-probability of a given regime being active at the moment t,
        conditional on all observations measured till the moment t. A
        `(nobs, k_regimes)` shaped array.
    predicted_regime_logprobs : array
        Log-probability of a given regime being active at the moment t,
        conditional on all observations measured until the moment t. A
        `(nobs, k_regimes)` shaped array.
    """

    _filter_attributes = ['loglikelihood_burn', 'initial_regime_logprobs',
            'obs_loglikelihoods', 'filtered_state', 'filtered_state_cov',
            'filtered_regime_logprobs', 'predicted_regime_logprobs']

    _attributes = FrozenSwitchingRepresentation._attributes + \
            _filter_attributes

    def update_representation(self, model):
        """
        Update the results to match a given model

        Parameters
        ----------
        model : SwitchingRepresentation
            The model object from which to take the updated values.

        Notes
        -----
        This method is rarely required except for internal usage.
        """

        super(KimFilterResults, self).update_representation(model)

        # This is defined in `KimFilter` class
        self.loglikelihood_burn = model._loglikelihood_burn

        # When user didn't define initial regime distribution himself, these
        # values are initialized only during filtering.
        self.initial_regime_logprobs = model._initial_regime_logprobs

    def update_filter(self, kfilter):
        """
        Update the filter results

        Parameters
        ----------
        kfilter : _KimFilter
            Object, handling filtering, which to take the updated values from.

        Notes
        -----
        This method is rarely required except for internal usage.
        """

        self.obs_loglikelihoods = kfilter.obs_loglikelihoods

        self.filtered_state = kfilter.filtered_state
        self.filtered_state_cov = kfilter.filtered_state_cov

        self.forecasts_error = np.array(kfilter.forecast_error, copy=True)
        self.forecasts = self.forecasts_error + \
                kfilter.model.endog
        self.forecasts_error_cov = np.array(kfilter.forecast_error_cov,
                copy=True)

        self.filtered_regime_logprobs = kfilter.filtered_regime_logprobs
        self.predicted_regime_logprobs = kfilter.predicted_regime_logprobs

        self._standardized_forecasts_error = None

    @property
    def standardized_forecasts_error(self):
        """
        (array) Standardized forecast errors
        """
        if self._standardized_forecasts_error is None:
            # Logic partially copied from `kalman_filter.FilterResults` class
            from scipy import linalg
            self._standardized_forecasts_error = \
                    np.zeros_like(self.forecasts_error)
            for t in range(self.forecasts_error_cov.shape[2]):
                upper, _ = linalg.cho_factor(self.forecasts_error_cov[:, :, t])
                self._standardized_forecasts_error[:, t] = \
                        linalg.solve_triangular(upper,
                        self.forecasts_error[:, t])

        return self._standardized_forecasts_error

    @property
    def filtered_regime_probs(self):
        """
        (array) Probability of a given regime being active at the moment t,
        conditional on all observations measured till the moment t. A
        `(k_regimes, nobs)` shaped array.
        """
        return np.exp(self.filtered_regime_logprobs)

    @property
    def predicted_regime_probs(self):
        """
        (array) Probability of a given regime being active at the moment t,
        conditional on all observations measured until the moment t. A
        `(k_regimes, nobs)` shaped array.
        """
        return np.exp(self.predicted_regime_logprobs)

    def loglikeobs(self, loglikelihood_burn=0):
        """
        Calculate the loglikelihood for each observation associated with the
        statespace model.

        Parameters
        ----------
        loglikelihood_burn : int
            The number of initial periods during which the loglikelihood is
            not recorded.

        Notes
        -----
        If `loglikelihood_burn` is positive, then the entries in the returned
        loglikelihood vector are set to be zero for those initial time periods.

        Returns
        -------
        loglike : array of float
            Array of loglikelihood values for each observation.
        """

        # If loglikelihood is provided in both `loglikeobs` arguments and
        # constructor, choose bigger one
        loglikelihood_burn = max(loglikelihood_burn, self.loglikelihood_burn)

        loglikelihoods = np.array(self.obs_loglikelihoods)
        loglikelihoods[:loglikelihood_burn] = 0

        return loglikelihoods

    def loglike(self, loglikelihood_burn=0):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        loglikelihood_burn : int
            The number of initial periods during which the loglikelihood is
            not recorded.

        Returns
        -------
        loglike : float
            The joint loglikelihood.
        """

        # If loglikelihood is provided in both `loglike` arguments and
        # constructor, choose bigger one
        loglikelihood_burn = max(loglikelihood_burn, self.loglikelihood_burn)

        # Summarize observation loglikelihoods and return results
        return self.obs_loglikelihoods[loglikelihood_burn:].sum()
