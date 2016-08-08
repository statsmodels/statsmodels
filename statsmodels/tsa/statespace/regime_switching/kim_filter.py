"""
Markov Switching State Space Representation and Kim Filter

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from .switching_representation import SwitchingRepresentation, \
        FrozenSwitchingRepresentation
from scipy.misc import logsumexp
from statsmodels.tsa.statespace.kalman_filter import PredictionResults

from warnings import warn

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
            complex_step=False, nobs=None):

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

        # Check if filter initialization is provided
        if any((regime_filter.initialization is None for regime_filter in \
                model._regime_kalman_filters)):
            raise RuntimeError('Statespace model not initialized.')

        if nobs is None:
            self.nobs = model.nobs
        else:
            self.nobs = nobs

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
        nobs = self.nobs

        # Array, storing \ln( f( y_t | \psi_{t-1} ) )
        self.obs_loglikelihoods = np.zeros((nobs,), dtype=dtype)

        # Array, storing \ln( Pr[ S_t | \psi_t ] )
        self.filtered_regime_logprobs = np.zeros((k_regimes, nobs),
                dtype=dtype)
        # Array, storing \ln( Pr[ S_t | \psi_{t-1} ] )
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
            complex_step=False, nobs=None):
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
        nobs : int
            If specified, filtering is performed on data prefix of length
            `nobs`. This parameter is usually used internally for prediction.

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
            tolerance=tolerance, complex_step=complex_step, nobs=nobs)
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
        results.update_filter(kfilter, self)

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

    @property
    def tolerance(self):
        """
        (float) The tolerance at which the Kalman filter determines convergence
        to steady-state. Handling is delegated to `KalmanFilter`.
        """
        return self.regime_filters[0].tolerance

    @tolerance.setter
    def tolerance(self, value):
        """
        (float) The tolerance at which the Kalman filter determines convergence
        to steady-state. Handling is delegated to `KalmanFilter`.
        """
        for regime_filter in self.regime_filters:
            regime_filter.tolerance = value

    def set_filter_method(self, **kwargs):
        """
        This method propagates the action of the same name among regime filters.
        See `KalmanFilter.set_filter_method` documentation for details.
        """

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_filter_method(**kwargs)

    @property
    def filter_method(self):
        """
        (int) Filter method, used in underlying Kalman filters
        """
        return self.regime_filters[0].filter_method

    def set_inversion_method(self, **kwargs):
        """
        This method propagates the action of the same name among regime filters.
        See `KalmanFilter.set_inversion_method` documentation for details.
        """

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_inversion_method(**kwargs)

    @property
    def inversion_method(self): 
        """
        (int) Inversion method, used in underlying Kalman filters
        """
        return self.regime_filters[0].inversion_method

    def set_stability_method(self, **kwargs):
        """
        This method propagates the action of the same name among regime filters.
        See `KalmanFilter.set_stability_method` documentation for details.
        """

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_stability_method(**kwargs)

    @property
    def stability_method(self):
        """
        (int) Stability method, used in underlying Kalman filters
        """
        return self.regime_filters[0].stability_method

    def set_conserve_memory(self, **kwargs):
        """
        This method propagates the action of the same name among regime filters.
        See `KalmanFilter.set_conserve_method` documentation for details.
        """

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_conserve_memory(**kwargs)

    @property
    def conserve_memory(self):
        """
        (int) Conserve memory property, used in underlying Kalman filters
        """
        return self.regime_filters[0].conserve_memory

    @property
    def filter_timing(self):
        """
        (int) Filter timing, used in underlying Kalman filters
        """
        return self.regime_filters[0].filter_timing

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
    filter_method : int
        Bitmask representing the Kalman filtering method (for underlying Kalman
        filters).
    inversion_method : int
        Bitmask representing the method used to invert the forecast error
        covariance matrix (for underlying Kalman filters).
    stability_method : int
        Bitmask representing the methods used to promote numerical stability in
        the Kalman filter recursions (for underlying Kalman filters).
    conserve_memory : int
        Bitmask representing the selected memory conservation method (for
        underlying Kalman filters).
    filter_timing : int
        Whether or not to use the alternate timing convention (for underlying
        Kalman filters).
    tolerance : float
        The tolerance at which the Kalman filter determines convergence to
        steady-state (for underlying Kalman filters).
    """

    _filter_attributes = ['loglikelihood_burn', 'initial_regime_logprobs',
            'obs_loglikelihoods', 'filtered_state', 'filtered_state_cov',
            'filtered_regime_logprobs', 'predicted_regime_logprobs',
            'filter_method', 'inversion_method', 'stability_method',
            'conserve_method', 'filter_timing', 'tolerance']

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

        # When user didn't define initial regime distribution himself, these
        # values are initialized only during filtering.
        self.initial_regime_logprobs = model._initial_regime_logprobs

    def update_filter(self, kfilter, model):
        """
        Update the filter results

        Parameters
        ----------
        kfilter : _KimFilter
            Object, handling filtering, which to take the updated values from.
        model : KimFilter
            `KimFilter` object with metadata.

        Notes
        -----
        This method is rarely required except for internal usage.
        """

        # Save Kalman filter parameters
        self.filter_method = model.filter_method
        self.inversion_method = model.inversion_method
        self.stability_method = model.stability_method
        self.conserve_memory = model.conserve_memory
        self.filter_timing = model.filter_timing
        self.tolerance = model.tolerance
        self.loglikelihood_burn = model.loglikelihood_burn

        # Save filtering result matrices

        self.obs_loglikelihoods = kfilter.obs_loglikelihoods

        self.filtered_state = kfilter.filtered_state
        self.filtered_state_cov = kfilter.filtered_state_cov

        self.forecasts_error = np.array(kfilter.forecast_error, copy=True)
        self.forecasts = self.forecasts_error + \
                kfilter.model.endog[:, :self.forecasts_error.shape[1]]
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

    def predict(self, start=None, end=None, dynamic=None, **kwargs):
        """
        In-sample and out-of-sample prediction for state space models generally

        Parameters
        ----------
        start : int, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast will be at start.
        end : int, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast will be at end.
        dynamic : int, optional
            This option is added for compatibility, using it will cause
            `NotImplementedError`.
        **kwargs
            If the prediction range is outside of the sample range, any
            of the state space representation matrices that are time-varying
            must have updated values provided for the out-of-sample range.
            For example, of `obs_intercept` is a time-varying component and
            the prediction range extends 10 periods beyond the end of the
            sample, a (`k_regimes` x `k_endog` x 10) matrix must be provided
            with the new intercept values.

        Returns
        -------
        results : PredictionResults
            A PredictionResults object.

        Notes
        -----
        Only one-step-ahead prediction and forecasting are available.

        All prediction is performed by applying the deterministic part of the
        measurement equation using the predicted state variables and ignoring
        the Hamilton filtering step.

        See Also
        --------
        statsmodels.tsa.statespace.kalman_filter.FilterResults.predict
        """

        nobs = self.nobs

        # Raise error, if `dynamic` argument is provided
        if dynamic is not None and dynamic != 0:
            raise NotImplementedError

        if end is None:
            end = nobs

        # Length of prediction periods
        nstatic = nobs
        ndynamic = 0
        nforecast = max(0, end - nobs)

        # If no forecasting needed, the existing result object can be used
        if nforecast == 0:
            return PredictionResults(self, start, end, nstatic, ndynamic,
                    nforecast)

        # Collect representation matrices
        representation = {}
        for name in SwitchingRepresentation._per_regime_dims.keys():
            representation[name] = np.asarray([getattr(x, name) for x in \
                    self.regime_representations])

        warning = ('Model has time-invariant {0} matrix, so the {0}'
                ' argument to `predict` has been ignored.')
        exception = ('Forecasting for models with time-varying {0} matrix'
                ' requires an updated time-varying matrix for the'
                ' period to be forecasted.')

        # Extending representation matrices, if they are not time-invariant
        for name in SwitchingRepresentation._per_regime_dims.keys():
            # If matrix is time invariant, no extension is required
            if representation[name].shape[-1] == 1:
                if name in kwargs:
                    warn(warning.format(name))
            # If it's not, and no extension provided, raise an error
            elif name not in kwargs:
                raise ValueError(exception.format(name))
            # Extend matrix with out-of-sample range
            else:
                mat = self.model._broadcast_per_regime(kwargs[name],
                        SwitchingRepresentation._per_regime_dims[name])
                if mat.shape[:-1] != representation[name].shape[:-1]:
                    raise ValueError(exception.format(name))

                representation = np.c_[representation, mat]

        # Add regime transition matrix to arguments
        representation['regime_transition'] = np.exp(self.log_regime_transition)

        # Extend `endog` array with NaNs
        endog = np.empty((self.k_endog, nforecast))
        endog.fill(np.nan)
        endog = np.asfortranarray(np.c_[self.endog, endog])

        # Other options
        model_kwargs = {
                'filter_method': self.filter_method,
                'inversion_method': self.inversion_method,
                'stability_method': self.stability_method,
                'conserve_memory': self.conserve_memory,
                'filter_timing': self.filter_timing,
                'tolerance': self.tolerance,
                'loglikelihoood_burn': self.loglikelihood_burn
        }

        model_kwargs.update(representation)

        # The size of state covariance matrix is the same for all regimes
        k_posdef = self.regime_representations[0].k_posdef

        # Instantiate the model for forecasting
        model = KimFilter(self.k_endog, self.k_states, self.k_regimes,
                k_posdef=k_posdef, **model_kwargs)

        # Bind the endogenous data
        model.bind(endog)

        # Provide state space initialization
        if self.model.initialization is not None:
            model.initialize_known(
                    [x.initial_state for x in self.regime_representations],
                    [x.initial_state_cov for x in self.regime_representations])

        # Do actual prediction
        results = self._predict(nstatic, nforecast, model)

        return PredictionResults(results, start, end, nstatic, ndynamic,
                nforecast)

    def _predict(self, nstatic, nforecast, model):
        # This method performs actual prediction

        # Notation in comments follows Kim and Nelson book, although the
        # prediction algorithm is not described there.

        dtype = self.dtype
        k_regimes = self.k_regimes
        k_endog = self.k_endog
        k_states = self.k_states

        # One-step-ahead prediction is usual filtering of non-NaN prefix of the
        # data
        filter_results = model.filter(nobs=nstatic)

        # Allocating some useful arrays for prediction

        log_regime_transition = self.log_regime_transition

        # Low-level Kalman filters 
        kfilters = model._kfilters

        # High-level Kalman filters
        regime_filters = model.regime_filters

        # Predicted probabilities
        predicted_regime_logprobs = np.zeros((k_regimes, nstatic + nforecast),
                dtype=dtype)

        # First `nstatic` columns of the matrix contain
        # \ln( Pr[ S_t | \psi_{t-1} ] ), t <= T, - one-step-ahead prediction of
        # regime. The remaining part is fulfilled with
        # \ln( Pr[ S_t | \psi_T ] ), where t > T.
        predicted_regime_logprobs[:, :nstatic] = \
                filter_results.predicted_regime_logprobs

        # Batteries of \beta_{t|T}^{(i,j)}, t > T
        predicted_state_batteries = np.zeros((k_regimes, k_regimes, k_states),
                dtype=dtype)

        # Batteries of P_{t|T}^{(i,j)}, t > T
        predicted_state_cov_batteries = np.zeros((k_regimes, k_regimes,
                k_states, k_states), dtype=dtype)

        # Forecasts array
        # First `nstatic` columns of the matrix contain E[ y_t | \psi_{t-1} ],
        # t <= T, - one-step-ahead prediction of observation. The remaining part
        # is fulfilled with E[ y_t | \psi_T ], t > T.
        forecasts = np.zeros((k_endog, nstatic + nforecast), dtype=dtype)
        forecasts[:, :nstatic] = filter_results.forecasts

        # Forecast covariance
        # First `nstatic` columns of the matrix contain f_{t|t-1}, t <= T, -
        # one-step-ahead prediction error covariance. The remaining part is
        # fulfilled with f_{t|T}, t > T - forecast error covariance.
        forecasts_error_cov = np.zeros((k_endog, k_endog, nstatic + nforecast),
                dtype=dtype)
        forecasts_error_cov[:, :, :nstatic] = filter_results.forecasts_error_cov

        # Batteries of E[ y_t | S_t, S_{t-1}, \psi_T ], t > T
        forecast_batteries = np.zeros((k_regimes, k_regimes, k_endog),
                dtype=dtype)

        # Batteries of f_{t|T}^{(i,j)}, t > T
        forecast_cov_batteries = np.zeros((k_regimes, k_regimes, k_endog,
                k_endog), dtype=dtype)

        # These are helping buffers, passed to `_marginalize_vector` function.
        # See `_marginalize_vector` documentation for details.
        weighted_forecasts = np.zeros((k_regimes * k_regimes, k_endog),
                dtype=dtype)
        forecast_biases = np.zeros((k_regimes * k_regimes, k_endog),
                dtype=dtype)
        forecast_bias_sqrs = np.zeros((k_regimes * k_regimes, k_endog, k_endog),
                dtype=dtype)
        forecast_covs_and_bias_sqrs = np.zeros((k_regimes * k_regimes, k_endog,
                k_endog), dtype=dtype)
        weighted_forecast_covs_and_bias_sqrs = np.zeros((k_regimes * k_regimes,
                k_endog, k_endog), dtype=dtype)

        # Buffers for \beta_{t|T}^{j}, t > T
        approx_predicted_states = np.zeros((k_regimes, k_states), dtype=dtype)

        # Buffers for P_{t|T}^{j}, t > T
        approx_predicted_state_covs = np.zeros((k_regimes, k_states, k_states),
                dtype=dtype)

        # Initialize approximate predicted state with \beta_{T|T}^{j} and
        # P_{T|T}^{j}
        for i in range(k_regimes):
            # Scenario, when `nstatic` = `nobs` = 0 is strange, but is handled
            # here
            if nstatic != 0:
                # \beta_{T|T}^{j} and P_{T|T}^{j} are stored in `filtered_state`
                # and `filtered_state_cov` arrays
                np.copyto(approx_predicted_states[i],
                        np.asarray(kfilters[i].filtered_state[:, nstatic - 1]))
                np.copyto(approx_predicted_state_covs[i],
                        np.asarray(kfilters[i].filtered_state_cov[:, :,
                        nstatic - 1]))
            else:
                # In this case T = 0, \beta_{0|0}^{j} and P_{0|0}^{j} is an
                # initialization of the model
                np.copyto(approx_predicted_states[i],
                        regime_filters[i]._initial_state)
                np.copyto(approx_predicted_state_covs[i],
                        regime_filters[i]._initial_state_cov)

        # These are helping buffers, passed to `_marginalize_vector` function.
        # See `_marginalize_vector` documentation for details.
        weighted_states = np.zeros((k_regimes, k_states), dtype=dtype)
        state_biases = np.zeros((k_regimes, k_states), dtype=dtype)
        state_bias_sqrs = np.zeros((k_regimes, k_states, k_states),
                dtype=dtype)
        state_covs_and_state_bias_sqrs = np.zeros((k_regimes, k_states,
                k_states), dtype=dtype)
        weighted_state_covs_and_state_bias_sqrs = np.zeros((k_regimes,
                k_states, k_states), dtype=dtype)

        # Forecasting
        for t in range(nstatic, nstatic + nforecast):

            # Pr[ S_t, S_{t-1} | \psi_T ] = Pr[ S_t | S_{t-1} ] *
            # * Pr[ S_{t-1} | \psi_T ]
            predicted_prev_and_curr_regime_logprobs = \
                    log_regime_transition.transpose() + \
                    predicted_regime_logprobs[:, t - 1].reshape(-1, 1)

            # Pr[ S_t | \psi_T ] = \sum_{S_{t-1}} Pr[ S_t, S_{t-1} | \psi_T ]
            predicted_regime_logprobs[:, t] = logsumexp(
                    predicted_prev_and_curr_regime_logprobs, axis=0)

            # Performing Kalman prediction in underlying Kalman filters
            # Remember, that NaNs in endog data force Kalman filter to do
            # prediction in the `__next__` calls
            for prev_regime in range(k_regimes):
                for curr_regime in range(k_regimes):
                    # This condition optimizes calculation time in case of
                    # sparse regime transition matrix (e.g. for MS AR)
                    if predicted_prev_and_curr_regime_logprobs[prev_regime, \
                            curr_regime] != -np.inf:

                        # References to current Kalman filter
                        curr_regime_filter = regime_filters[curr_regime]
                        curr_kfilter = kfilters[curr_regime]

                        # Pass \beta_{t|T}^{i} and P_{t|T}^{i} to j-th filter
                        # Again, the case of t == 0 is strange and rare
                        if t != 0:
                            np.copyto(np.asarray(curr_kfilter.filtered_state[:,
                                    t - 1]), approx_predicted_states[
                                    prev_regime])
                            np.copyto(np.asarray(
                                    curr_kfilter.filtered_state_cov[:,
                                    :, t - 1]), approx_predicted_state_covs[
                                    prev_regime])
                        else:
                            # If t == 0, use approximate state as initialization
                            curr_regime_filter.initialize_known(
                                    approx_predicted_states[prev_regime],
                                    approx_predicted_state_covs[prev_regime])
                            # Low level initialization - this method "pushes"
                            # new initialization to Cython Kalman filter
                            curr_regime_filter._initialize_state(
                                    **model._state_init_kwargs[curr_regime])

                        # We use every regime filter `k_regimes` times in the
                        # given moment t, so it needs to be moved one step back
                        curr_kfilter.seek(t)

                        # Iteration of prediction
                        next(curr_kfilter)

                        # Kalman filter produces \beta_{t|T}^{(i,j)} and
                        # P_{t|T}^{(i,j)}

                        np.copyto(predicted_state_batteries[prev_regime,
                                curr_regime], np.asarray(
                                curr_kfilter.predicted_state[:, t]))

                        np.copyto(predicted_state_cov_batteries[prev_regime,
                                curr_regime],
                                curr_kfilter.predicted_state_cov[:, :, t])

                        # Use convenient reference
                        predicted_state = predicted_state_batteries[prev_regime,
                                curr_regime].reshape(-1, 1)
                        predicted_state_cov = predicted_state_cov_batteries[
                                prev_regime, curr_regime]

                        # Get some state-space matrices of current regime and
                        # moment

                        if curr_regime_filter.design.shape[-1] == 1:
                            design = curr_regime_filter.design[:, :, 0]
                        else:
                            design = curr_regime_filter.design[:, :, t]

                        if curr_regime_filter.obs_intercept.shape[-1] == 1:
                            obs_intercept = curr_regime_filter.obs_intercept[:,
                                    0]
                        else:
                            obs_intercept = curr_regime_filter.obs_intercept[:,
                                    t]

                        if curr_regime_filter.obs_cov.shape[-1] == 1:
                            obs_cov = curr_regime_filter.obs_cov[:, :, 0]
                        else:
                            obs_cov = curr_regime_filter.obs_cov[:, :, t]

                        # Manually calculate forecast
                        # E[ y_t | S_t, S_{t-1}, \psi_T ] and its covariance
                        # f_{t|T}^{(i,j)}, since Kalman prediction fulfills
                        # corresponding arrays with zero

                        forecast_batteries[prev_regime, curr_regime] = \
                                design.dot(predicted_state) + obs_intercept

                        forecast_cov_batteries[prev_regime, curr_regime] = \
                                design.dot(predicted_state_cov).dot(
                                design.T) + obs_cov

            # Switch from logprobs to probs
            predicted_prev_and_curr_regime_probs = np.exp(
                    predicted_prev_and_curr_regime_logprobs)

            # Collapse E[ y_t | S_t, S_{t-1}, \psi_T ] and f_{t|T}^{(i,j)}
            # to obtain E[ y_t | \psi_T ] and f_{t|T}
            _marginalize_vector(predicted_prev_and_curr_regime_probs.ravel(),
                    forecast_batteries.reshape(-1, k_endog), weighted_forecasts,
                    forecasts[:, t], forecast_biases, forecast_bias_sqrs,
                    forecast_cov_batteries.reshape(-1, k_endog, k_endog),
                    forecast_covs_and_bias_sqrs,
                    weighted_forecast_covs_and_bias_sqrs, forecasts_error_cov[:,
                    :, t])

            # Pr[ S_{t-1} | S_t, \psi_T ] = Pr[ S_t, S_{t-1} | \psi_T ] /
            # / Pr[ S_t | \psi_T ]
            predicted_prev_cond_on_curr_regime_logprobs = \
                    predicted_prev_and_curr_regime_logprobs - \
                    predicted_regime_logprobs[:, t].reshape(1, -1)

            # Switch from logprobs to probs
            predicted_prev_cond_on_curr_regime_probs = np.exp(
                    predicted_prev_cond_on_curr_regime_logprobs)

            # Collapse \beta_{t|T}^{(i,j)} and P_{t|T}^{(i,j)}
            # to obtain \beta_{t|T}^{j} and P_{t|T}^{j}
            for curr_regime in range(k_regimes):
                _marginalize_vector(predicted_prev_cond_on_curr_regime_probs[:,
                        curr_regime], predicted_state_batteries[:, curr_regime,
                        :], weighted_states, approx_predicted_states[
                        curr_regime], state_biases, state_bias_sqrs,
                        predicted_state_cov_batteries[:, curr_regime, :, :],
                        state_covs_and_state_bias_sqrs,
                        weighted_state_covs_and_state_bias_sqrs,
                        approx_predicted_state_covs[curr_regime])

        # Use existing `KimFilterResults` object to substitute corresponding
        # fields with forecasted data
        filter_results.forecasts = forecasts
        filter_results.forecasts_error_cov = forecasts_error_cov
        filter_results.predicted_regime_logprobs = predicted_regime_logprobs

        return filter_results
